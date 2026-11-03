"""
Multi-Stock Turnaround Screener
Replicates the Confirmed/Early scoring logic from the TradingView Pine script,
scaled to run against an arbitrarily large stock list (no 40-symbol cap).
"""

import os
import time
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import numpy as np
import pandas as pd
import yfinance as yf

# ============================================================
# CONFIG
# ============================================================
STOCKS_FILE = "stocks.txt"
LOOKBACK_PERIOD = "2y"      # need 200+ trading days of history for SMA200
BATCH_SIZE = 50             # tickers per yfinance batch request
MIN_CONFIRMED_SCORE = 4     # out of 6
MIN_EARLY_SCORE = 4         # out of 6
OBV_LOOKBACK = 10
OBV_DIV_LOOKBACK = 15
SWING_LOOKBACK = 10
NEAR_PCT = 5.0               # "nearing SMA50" threshold, in %

# --- Liquidity filter: thin stocks are shown but never flagged ---
LIQ_LOOKBACK = 20                # days used for average volume / traded value
MIN_AVG_VOLUME = 100_000         # minimum average daily shares traded
MIN_AVG_TRADED_VALUE = 1e7       # minimum average daily traded value, in INR (1e7 = Rs 1 crore)

# --- "Stabilizing after oversold" flag ---
STAB_OVERSOLD_RSI = 30           # RSI must have dipped below this recently...
STAB_OVERSOLD_LOOKBACK = 10      # ...within this many trading days
STAB_LOW_WINDOW = 20             # window used to locate the recent low
STAB_MIN_DAYS_SINCE_LOW = 5      # no fresh low for at least this many days
STAB_MAX_BOUNCE_PCT = 15.0       # skip if already bounced more than this from the low (don't chase)
STAB_SMA_DAYS = 10               # price must be back above this short average
STAB_MIN_EARLY_SCORE = 2         # stabilizing table only lists stocks with at least this Early score (0 = list all)


# ============================================================
# DATA FETCH
# ============================================================
def load_tickers(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def fetch_data(tickers, batch_size=BATCH_SIZE):
    """Download price history in batches, with basic retry on failure."""
    data = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"Fetching batch {i // batch_size + 1} ({len(batch)} tickers)...")

        df = None
        for attempt in range(2):
            try:
                df = yf.download(
                    batch, period=LOOKBACK_PERIOD, interval="1d",
                    group_by="ticker", auto_adjust=True, threads=True, progress=False,
                )
                break
            except Exception as e:
                print(f"  Batch failed (attempt {attempt + 1}): {e}")
                time.sleep(5)

        if df is None:
            print(f"  Skipping batch after retries: {batch}")
            continue

        for t in batch:
            try:
                tdf = df if len(batch) == 1 else df[t]
                # Drop rows with no Close (Yahoo sometimes returns a partial/unfinalized
                # latest-day row with NaN Close, which would make every metric "nan").
                tdf = tdf.dropna(subset=["Close"])
                if len(tdf) < 210:
                    print(f"  Skipping {t}: insufficient history ({len(tdf)} rows)")
                    continue
                data[t] = tdf
            except Exception as e:
                print(f"  Skipping {t}: {e}")

        time.sleep(1)  # be polite between batches
    return data


# ============================================================
# INDICATORS
# ============================================================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_obv(close, volume):
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


# ============================================================
# SCORING — mirrors the Pine script's Confirmed (lagging) and
# Early (leading) condition sets
# ============================================================
def score_stock(df):
    close, volume = df["Close"], df["Volume"]

    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    sma20 = close.rolling(20).mean()
    sma_short = close.rolling(STAB_SMA_DAYS).mean()
    rsi = compute_rsi(close, 14)
    macd_line, signal_line, hist = compute_macd(close)
    obv = compute_obv(close, volume)
    lowest10 = close.rolling(SWING_LOOKBACK).min()

    i = -1  # "today" = most recent row

    c_now, c50, c200, c20 = close.iloc[i], sma50.iloc[i], sma200.iloc[i], sma20.iloc[i]
    rsi_now, rsi_1ago, rsi_3ago, rsi_5ago = rsi.iloc[i], rsi.iloc[i - 1], rsi.iloc[i - 3], rsi.iloc[i - 5]
    macd_now, sig_now = macd_line.iloc[i], signal_line.iloc[i]
    macd_1ago, sig_1ago = macd_line.iloc[i - 1], signal_line.iloc[i - 1]
    hist_now, hist_3ago = hist.iloc[i], hist.iloc[i - 3]
    obv_now = obv.iloc[i]
    obv_lb1, obv_lb2 = obv.iloc[i - OBV_LOOKBACK], obv.iloc[i - OBV_DIV_LOOKBACK]
    close_lb2 = close.iloc[i - OBV_DIV_LOOKBACK]
    sma50_1ago, sma200_1ago = sma50.iloc[i - 1], sma200.iloc[i - 1]
    sma50_5ago, sma200_5ago = sma50.iloc[i - 5], sma200.iloc[i - 5]
    close_1ago, sma20_1ago = close.iloc[i - 1], sma20.iloc[i - 1]

    # ---- CONFIRMED (lagging) ----
    c1 = c_now > c50
    c2 = c_now > c200
    c3 = (sma50_1ago <= sma200_1ago and c50 > c200) or (c50 > c200 and sma50_5ago <= sma200_5ago)
    c4 = (rsi_1ago <= 50 and rsi_now > 50) or (rsi_now > 50 and rsi_3ago <= 50)
    c5 = macd_1ago <= sig_1ago and macd_now > sig_now
    c6 = obv_now > obv_lb1
    confirmed_score = sum([c1, c2, c3, c4, c5, c6])

    # ---- EARLY (leading) ----
    e1 = hist_now > hist_3ago and hist_now < 0
    e2 = rsi_now > rsi_5ago and rsi_5ago < 35
    e3 = lowest10.iloc[i] > lowest10.iloc[i - SWING_LOOKBACK]
    e4 = obv_now > obv_lb2 and c_now < close_lb2
    pct_to_sma50 = (c50 - c_now) / c_now * 100
    e5 = 0 < pct_to_sma50 < NEAR_PCT
    e6 = close_1ago <= sma20_1ago and c_now > c20
    early_score = sum([e1, e2, e3, e4, e5, e6])

    # ---- STABILIZING AFTER OVERSOLD ----
    # Was oversold recently, has stopped making new lows, and is turning up -- without
    # having already run away. Helps separate a base from a falling knife.
    window = close.tail(STAB_LOW_WINDOW)
    low_val = float(window.min())
    days_since_low = len(window) - 1 - int(np.argmin(window.values))
    was_oversold = bool(rsi.tail(STAB_OVERSOLD_LOOKBACK).min() < STAB_OVERSOLD_RSI)
    no_new_low = days_since_low >= STAB_MIN_DAYS_SINCE_LOW
    above_short_sma = bool(c_now > sma_short.iloc[i])
    bounce_pct = (c_now - low_val) / low_val * 100
    not_chasing = bounce_pct <= STAB_MAX_BOUNCE_PCT
    stabilizing = bool(was_oversold and no_new_low and above_short_sma and not_chasing)

    # ---- LIQUIDITY ----
    avg_volume = float(volume.tail(LIQ_LOOKBACK).mean())
    avg_value = float((close * volume).tail(LIQ_LOOKBACK).mean())
    liquid = bool(avg_volume >= MIN_AVG_VOLUME and avg_value >= MIN_AVG_TRADED_VALUE)

    return {
        "price": round(float(c_now), 2),
        "rsi": round(float(rsi_now), 2),
        "pct_to_sma50": round(float(pct_to_sma50), 2),
        "confirmed_score": int(confirmed_score),
        "early_score": int(early_score),
        "stabilizing": stabilizing,
        "bounce_pct": round(float(bounce_pct), 2),
        "avg_volume": avg_volume,
        "liquid": liquid,
    }


# ============================================================
# EMAIL
# ============================================================
def build_warnings(data, n_requested):
    """Return a list of human-readable data-quality warnings for the email."""
    from zoneinfo import ZoneInfo
    warns = []
    last_dates = {t: df.index[-1].normalize() for t, df in data.items()}
    newest = max(last_dates.values())
    today = pd.Timestamp(datetime.now(ZoneInfo("Asia/Kolkata")).date())

    lag = int(np.busday_count(newest.date(), today.date()))
    if lag >= 1:
        warns.append(
            f"Latest data is from {newest.strftime('%d %b %Y')}, {lag} trading day(s) behind today. "
            "Today may be a market holiday, or Yahoo has not finalized today's close yet."
        )

    behind = sorted(t for t, d in last_dates.items() if d < newest)
    if behind:
        shown = ", ".join(behind[:10]) + (" ..." if len(behind) > 10 else "")
        warns.append(f"{len(behind)} ticker(s) have older data than the rest: {shown}")

    if n_requested and len(data) < 0.9 * n_requested:
        warns.append(
            f"Only {len(data)} of {n_requested} tickers loaded (under 90%). "
            "Results may be incomplete; some downloads probably failed."
        )
    return warns


def warning_html(warns):
    if not warns:
        return ""
    items = "".join(f"<li>{w}</li>" for w in warns)
    return (
        '<div style="background:#fff3cd;border:1px solid #e0a800;padding:8px 12px;margin:8px 0;">'
        f"<b>Data check</b><ul style='margin:4px 0;'>{items}</ul></div>"
    )


def build_email_html(results_df, data_as_of="", warns=None):
    sort_cols = ["early_score", "confirmed_score"]
    liquid_df = results_df[results_df.liquid]

    # Low-liquidity stocks are listed in the full table but never flagged.
    flagged = liquid_df[
        (liquid_df.confirmed_score >= MIN_CONFIRMED_SCORE) |
        (liquid_df.early_score >= MIN_EARLY_SCORE)
    ].sort_values(sort_cols, ascending=False)
    stab_all = liquid_df[liquid_df.stabilizing]
    # Best candidates first: higher Early score, then smallest bounce from the low.
    stabilizing = stab_all[stab_all.early_score >= STAB_MIN_EARLY_SCORE].sort_values(
        ["early_score", "bounce_pct"], ascending=[False, True]
    )
    n_stab_hidden = len(stab_all) - len(stabilizing)
    all_sorted = results_df.sort_values(sort_cols, ascending=False)
    n_low_liq = int((~results_df.liquid).sum())

    def rows(df):
        out = ""
        for _, r in df.iterrows():
            name = r["ticker"] + ("" if r["liquid"] else " (low liq)")
            out += (
                f"<tr><td>{name}</td><td>{r['price']}</td><td>{r['rsi']}</td>"
                f"<td>{r['pct_to_sma50']}%</td><td>{r['early_score']}</td>"
                f"<td>{r['confirmed_score']}</td>"
                f"<td>{'Yes' if r['stabilizing'] else '-'}</td>"
                f"<td>{r['bounce_pct']}%</td><td>{r['avg_volume']:,.0f}</td></tr>"
            )
        return out

    header = (
        "<tr><th>Ticker</th><th>Price</th><th>RSI</th><th>% to SMA50</th>"
        "<th>Early/6</th><th>Confirmed/6</th><th>Stabilizing</th>"
        "<th>Bounce from low</th><th>Avg Vol (20d)</th></tr>"
    )

    return f"""
    <html><body style="font-family: Arial, sans-serif;">
    <h2>Turnaround Screener — {datetime.now().strftime('%d %b %Y')}</h2>
    {warning_html(warns)}
    <p>Data as of last close: {data_as_of}</p>
    <p>{len(results_df)} stocks scanned, {len(flagged)} flagged (Confirmed ≥ {MIN_CONFIRMED_SCORE} or Early ≥ {MIN_EARLY_SCORE}),
    {len(stabilizing)} stabilizing after oversold with Early ≥ {STAB_MIN_EARLY_SCORE}
    ({n_stab_hidden} more stabilizing with a lower Early score are marked "Yes" in the full list).</p>
    <p style="color:#666;font-size:12px;">
    Low liquidity = under {MIN_AVG_VOLUME:,} shares or under Rs {MIN_AVG_TRADED_VALUE/1e7:g} crore traded per day (20-day average).
    {n_low_liq} such stock(s) are marked "(low liq)" in the full list and excluded from the flagged and stabilizing tables.<br>
    Stabilizing = RSI fell below {STAB_OVERSOLD_RSI} in the last {STAB_OVERSOLD_LOOKBACK} days, no new low for {STAB_MIN_DAYS_SINCE_LOW}+ days,
    price back above its {STAB_SMA_DAYS}-day average, and bounce from the low no more than {STAB_MAX_BOUNCE_PCT:g}%.
    The stabilizing table is sorted by Early score (high to low), then smallest bounce.</p>

    <h3>Flagged</h3>
    <table border="1" cellpadding="6" cellspacing="0">{header}{rows(flagged)}</table>

    <h3>Stabilizing after oversold</h3>
    <table border="1" cellpadding="6" cellspacing="0">{header}{rows(stabilizing)}</table>

    <h3>Full list</h3>
    <table border="1" cellpadding="4" cellspacing="0">{header}{rows(all_sorted)}</table>
    </body></html>
    """


def send_email(subject, html_body):
    sender = os.environ["GMAIL_USER"]
    password = os.environ["GMAIL_APP_PASSWORD"]
    recipient = os.environ.get("ALERT_RECIPIENT", sender)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, recipient, msg.as_string())
    print("Email sent.")


# ============================================================
# MAIN
# ============================================================
def main():
    tickers = load_tickers(STOCKS_FILE)
    print(f"Loaded {len(tickers)} tickers from {STOCKS_FILE}")

    data = fetch_data(tickers)
    print(f"Successfully fetched data for {len(data)}/{len(tickers)} tickers")

    results = []
    for t, df in data.items():
        try:
            r = score_stock(df)
            r["ticker"] = t
            results.append(r)
        except Exception as e:
            print(f"Scoring failed for {t}: {e}")

    if not results:
        print("No results to report — exiting without sending email.")
        return

    results_df = pd.DataFrame(results)
    data_as_of = max(df.index[-1] for df in data.values()).strftime("%d %b %Y")
    warns = build_warnings(data, len(tickers))
    for w in warns:
        print("WARNING:", w)
    html = build_email_html(results_df, data_as_of, warns)
    subject = ("[CHECK DATA] " if warns else "") + f"Turnaround Screener — {datetime.now().strftime('%d %b %Y')} ({len(results_df)} scanned)"
    send_email(subject, html)


if __name__ == "__main__":
    main()
