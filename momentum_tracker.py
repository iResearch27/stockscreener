"""
Momentum Tracker
Gives every stock in the watchlist ONE momentum state, so you can see at a glance
which stocks have the right amount of momentum right now:

    Building   - turning up from a base, rank improving, some volume support (entry zone)
    Strong     - confirmed uptrend, leading the pack and beating Nifty
    Extended   - strong but stretched far above its averages (don't chase)
    Fading     - below the SMA50 and relative strength dropping
    Dormant    - weak trend, low relative strength, no sign of improvement
    Neutral    - fits none of the above
    Low liquidity - too thinly traded to judge (listed separately, never given a state)

Inputs: relative strength rank (vs all tracked stocks and vs Nifty), trend points,
participation (volume / OBV), and stretch (distance from SMA50 measured in ATRs).
See METHODOLOGY.md for the full logic.
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
STOCKS_FILE = "uptrend_watchlist.txt"   # shares the Uptrend tracker's (larger) watchlist
BENCHMARK_TICKER = "^NSEI"
LOOKBACK_PERIOD = "2y"
BATCH_SIZE = 50
MIN_ROWS = 280                  # 252-day return + 21-day lookback needs ~274 rows

# --- Relative strength ---
RS_WEIGHTS = {21: 0.2, 63: 0.4, 126: 0.2, 252: 0.2}   # weight per return window (trading days)
RS_CHANGE_LOOKBACK = 21         # compare today's rank with the rank this many days ago
NIFTY_LOOKBACK = 63             # window for "vs Nifty"

# --- State thresholds ---
STRONG_MIN_RANK = 60            # Strong: RS rank at least this
DORMANT_MAX_RANK = 40           # Dormant: RS rank below this (and weak trend)
BUILDING_MIN_RS_CHANGE = 10     # Building: rank up at least this many points in 21 days
FADING_MAX_RS_CHANGE = -10      # Fading: rank down at least this many points in 21 days
STRONG_MIN_TREND = 3            # trend points (0-4) needed for Strong / Extended
BUILDING_MAX_TREND = 2          # the RS-improvement route to Building needs a still-weak trend
BUILDING_MIN_PARTICIPATION = 1  # participation points (0-3) needed for Building
DORMANT_MAX_TREND = 1

# --- Stretch ---
ATR_DAYS = 14
EXTENDED_ATR_MULT = 6.0         # Extended if (close - SMA50) / ATR >= this
EXTENDED_RSI = 70               # ...or RSI >= this
EXTENDED_RSI_MIN_PCT = 10.0     # ...and at least this % above the SMA50

# --- Participation ---
OBV_DAYS = 20
UPDOWN_DAYS = 20
UPDOWN_MIN_RATIO = 1.1
VOL_SHORT, VOL_LONG = 10, 50

# --- Stabilizing (same rules as the Turnaround screener) ---
STAB_OVERSOLD_RSI = 30
STAB_OVERSOLD_LOOKBACK = 10
STAB_LOW_WINDOW = 20
STAB_MIN_DAYS_SINCE_LOW = 5
STAB_MAX_BOUNCE_PCT = 15.0
STAB_SMA_DAYS = 10

# --- Liquidity (same as the Turnaround screener) ---
LIQ_LOOKBACK = 20
MIN_AVG_VOLUME = 100_000
MIN_AVG_TRADED_VALUE = 1e7      # INR per day (1e7 = Rs 1 crore)

# --- Score weights (used only to sort within a state) ---
W_RANK, W_TREND, W_PART = 0.5, 0.25, 0.25

STATE_ORDER = ["Building", "Strong", "Extended", "Fading", "Neutral", "Dormant"]
ACTIONS = {
    "Building": "Buy in tranches",
    "Strong": "Add on a pullback",
    "Extended": "Don't chase",
    "Fading": "Avoid / exit",
    "Neutral": "Watch",
    "Dormant": "Watch",
}


# ============================================================
# DATA FETCH
# ============================================================
def load_tickers(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def flatten_columns(df):
    """Newer yfinance versions return MultiIndex columns even for a single ticker."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_data(tickers, batch_size=BATCH_SIZE):
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
                tdf = flatten_columns(df) if len(batch) == 1 else df[t]
                # Drop rows with no Close (partial/unfinalized latest-day rows would give "nan").
                tdf = tdf.dropna(subset=["Close"])
                if len(tdf) < MIN_ROWS:
                    print(f"  Skipping {t}: insufficient history ({len(tdf)} rows)")
                    continue
                data[t] = tdf
            except Exception as e:
                print(f"  Skipping {t}: {e}")

        time.sleep(1)
    return data


def fetch_benchmark():
    df = yf.download(BENCHMARK_TICKER, period=LOOKBACK_PERIOD, interval="1d",
                     auto_adjust=True, progress=False)
    return flatten_columns(df).dropna(subset=["Close"])


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


def compute_atr(high, low, close, period=ATR_DAYS):
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def compute_obv(close, volume):
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def pct_return(close, n, end=-1):
    """% return over n trading days, ending at index `end` (-1 = today)."""
    return (close.iloc[end] / close.iloc[end - n] - 1) * 100


def rs_raw(close, end=-1):
    return sum(w * pct_return(close, n, end) for n, w in RS_WEIGHTS.items())


# ============================================================
# PER-STOCK METRICS (everything that does not need the rest of the universe)
# ============================================================
def stock_metrics(df, nifty_return):
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    sma_short = close.rolling(STAB_SMA_DAYS).mean()
    rsi = compute_rsi(close, 14)
    atr = compute_atr(high, low, close)
    obv = compute_obv(close, volume)

    c_now = float(close.iloc[-1])
    sma50_now, sma200_now = float(sma50.iloc[-1]), float(sma200.iloc[-1])
    rsi_now = float(rsi.iloc[-1])
    atr_now = float(atr.iloc[-1])

    # ---- Trend points (0-4) ----
    trend_pts = int(sum([
        c_now > sma50_now,
        sma50_now > sma200_now,
        c_now > sma200_now,
        sma50_now > float(sma50.iloc[-11]),
    ]))

    # ---- Stretch ----
    pct_above_sma50 = (c_now - sma50_now) / sma50_now * 100
    ext_atr = (c_now - sma50_now) / atr_now if atr_now > 0 else 0.0
    stretched = bool(
        ext_atr >= EXTENDED_ATR_MULT or
        (rsi_now >= EXTENDED_RSI and pct_above_sma50 >= EXTENDED_RSI_MIN_PCT)
    )

    # ---- Participation points (0-3) ----
    p_obv = bool(obv.iloc[-1] > obv.tail(OBV_DAYS).mean() and obv.iloc[-1] > obv.iloc[-1 - OBV_DAYS])

    chg = close.diff().tail(UPDOWN_DAYS)
    vol_recent = volume.tail(UPDOWN_DAYS)
    up_vol = float(vol_recent[chg > 0].sum())
    down_vol = float(vol_recent[chg < 0].sum())
    p_updown = bool(up_vol > 0 and (down_vol == 0 or up_vol / down_vol > UPDOWN_MIN_RATIO))

    p_volume = bool(volume.tail(VOL_SHORT).mean() >= volume.tail(VOL_LONG).mean())
    participation = int(p_obv) + int(p_updown) + int(p_volume)

    # ---- Stabilizing after oversold (same rules as the Turnaround screener) ----
    window = close.tail(STAB_LOW_WINDOW)
    low_val = float(window.min())
    days_since_low = len(window) - 1 - int(np.argmin(window.values))
    was_oversold = bool(rsi.tail(STAB_OVERSOLD_LOOKBACK).min() < STAB_OVERSOLD_RSI)
    bounce_pct = (c_now - low_val) / low_val * 100
    stabilizing = bool(
        was_oversold and days_since_low >= STAB_MIN_DAYS_SINCE_LOW and
        c_now > float(sma_short.iloc[-1]) and bounce_pct <= STAB_MAX_BOUNCE_PCT
    )

    # ---- Liquidity ----
    avg_volume = float(volume.tail(LIQ_LOOKBACK).mean())
    avg_value = float((close * volume).tail(LIQ_LOOKBACK).mean())
    liquid = bool(avg_volume >= MIN_AVG_VOLUME and avg_value >= MIN_AVG_TRADED_VALUE)

    return {
        "price": round(c_now, 2),
        "rsi": round(rsi_now, 1),
        "rs_raw": rs_raw(close),
        "rs_raw_prev": rs_raw(close, end=-1 - RS_CHANGE_LOOKBACK),
        "vs_nifty": round(pct_return(close, NIFTY_LOOKBACK) - nifty_return, 2),
        "trend_pts": trend_pts,
        "participation": participation,
        "pct_above_sma50": round(pct_above_sma50, 2),
        "ext_atr": round(ext_atr, 1),
        "stretched": stretched,
        "stabilizing": stabilizing,
        "avg_volume": avg_volume,
        "liquid": liquid,
    }


# ============================================================
# UNIVERSE-LEVEL: RS rank, state, score
# ============================================================
def assign_state(r):
    if not r["liquid"]:
        return "Low liquidity"
    trend, rank, chg = r["trend_pts"], r["rs_rank"], r["rs_change"]

    if trend >= STRONG_MIN_TREND and r["stretched"]:
        return "Extended"
    if trend >= STRONG_MIN_TREND and rank >= STRONG_MIN_RANK and r["vs_nifty"] > 0:
        return "Strong"
    if ((r["stabilizing"] or (chg >= BUILDING_MIN_RS_CHANGE and trend <= BUILDING_MAX_TREND))
            and r["participation"] >= BUILDING_MIN_PARTICIPATION):
        return "Building"
    if r["pct_above_sma50"] < 0 and chg <= FADING_MAX_RS_CHANGE:
        return "Fading"
    if rank < DORMANT_MAX_RANK and trend <= DORMANT_MAX_TREND:
        return "Dormant"
    return "Neutral"


def build_universe(results_df):
    df = results_df.copy()
    df["rs_rank"] = df["rs_raw"].rank(pct=True) * 100
    df["rs_rank_prev"] = df["rs_raw_prev"].rank(pct=True) * 100
    df["rs_change"] = df["rs_rank"] - df["rs_rank_prev"]
    df["score"] = (
        W_RANK * df["rs_rank"] +
        W_TREND * (df["trend_pts"] / 4 * 100) +
        W_PART * (df["participation"] / 3 * 100)
    )
    df["state"] = df.apply(assign_state, axis=1)
    df["action"] = df["state"].map(ACTIONS).fillna("Too thinly traded to judge")
    return df


# ============================================================
# EMAIL
# ============================================================
def build_warnings(data, n_requested):
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


def participation_label(p):
    return {3: "Confirming", 0: "Not confirming"}.get(int(p), "Mixed") + f" ({int(p)}/3)"


def build_email_html(df, nifty_return, data_as_of="", warns=None):
    counts = df["state"].value_counts()

    def rows(sub):
        out = ""
        for _, r in sub.sort_values("score", ascending=False).iterrows():
            out += (
                f"<tr><td>{r['ticker']}</td><td>{r['score']:.0f}</td>"
                f"<td>{r['rs_rank']:.0f}</td><td>{r['rs_change']:+.0f}</td>"
                f"<td>{r['vs_nifty']:+.1f}%</td><td>{int(r['trend_pts'])}/4</td>"
                f"<td>{participation_label(r['participation'])}</td>"
                f"<td>{r['ext_atr']:.1f}x</td><td>{r['rsi']}</td>"
                f"<td>{r['action']}</td></tr>"
            )
        return out

    header = (
        "<tr><th>Ticker</th><th>Score</th><th>RS rank</th><th>RS chg (21d)</th>"
        "<th>vs Nifty (3M)</th><th>Trend</th><th>Participation</th>"
        "<th>Stretch (ATR)</th><th>RSI</th><th>Action</th></tr>"
    )

    blurbs = {
        "Building": "Turning up from a base with rank improving and some volume support. The entry zone: consider buying in tranches.",
        "Strong": "Confirmed uptrend, high relative strength, beating Nifty. Add on a pullback toward the SMA50 rather than chasing.",
        "Extended": "Strong but stretched far above its averages. Don't chase; tighten stops if you hold.",
        "Fading": "Below the SMA50 with relative strength falling. Avoid new money; review if you hold.",
    }

    sections = ""
    for state in ["Building", "Strong", "Extended", "Fading"]:
        sub = df[df.state == state]
        sections += (
            f"<h3>{state} ({len(sub)})</h3><p style='color:#666;font-size:12px;'>{blurbs[state]}</p>"
            f"<table border='1' cellpadding='5' cellspacing='0'>{header}{rows(sub)}</table>"
        )

    rest = df[df.state.isin(["Neutral", "Dormant"])].sort_values("score", ascending=False)
    rest_rows = "".join(
        f"<tr><td>{r['ticker']}</td><td>{r['state']}</td><td>{r['score']:.0f}</td>"
        f"<td>{r['rs_rank']:.0f}</td><td>{r['vs_nifty']:+.1f}%</td><td>{int(r['trend_pts'])}/4</td></tr>"
        for _, r in rest.iterrows()
    )
    low_liq = ", ".join(sorted(df[df.state == "Low liquidity"]["ticker"])) or "none"

    summary = ", ".join(f"{s}: {int(counts.get(s, 0))}" for s in STATE_ORDER + ["Low liquidity"])

    return f"""
    <html><body style="font-family: Arial, sans-serif;">
    <h2>Momentum Tracker — {datetime.now().strftime('%d %b %Y')}</h2>
    {warning_html(warns)}
    <p>Data as of last close: {data_as_of}</p>
    <p>Nifty 50 3-month return: {round(nifty_return, 2)}%</p>
    <p>{len(df)} stocks scanned. {summary}.</p>
    <p style="color:#666;font-size:12px;">
    RS rank = percentile of a blended 1/3/6/12-month return among all stocks scanned (100 = strongest).
    RS chg = rank change over 21 trading days. Trend = 4 checks on SMA50/SMA200. Participation = 3 volume/OBV checks.
    Stretch = distance above the SMA50 in ATRs (Extended at {EXTENDED_ATR_MULT:g}x or more). Score sorts stocks within a state only.
    Price and volume signals only; no earnings or fundamentals. Not investment advice.</p>

    {sections}

    <h3>Neutral and Dormant ({len(rest)})</h3>
    <table border="1" cellpadding="4" cellspacing="0">
    <tr><th>Ticker</th><th>State</th><th>Score</th><th>RS rank</th><th>vs Nifty (3M)</th><th>Trend</th></tr>
    {rest_rows}</table>

    <h3>Low liquidity (no state given)</h3>
    <p style="font-size:12px;">{low_liq}</p>
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
    if not tickers:
        print(f"{STOCKS_FILE} is empty — add tickers before running. Exiting.")
        return

    nifty_df = fetch_benchmark()
    if len(nifty_df) < NIFTY_LOOKBACK + 5:
        print("Not enough Nifty history. Exiting.")
        return
    nifty_return = float(pct_return(nifty_df["Close"], NIFTY_LOOKBACK))
    print(f"Nifty 50 3-month return: {round(nifty_return, 2)}%")

    data = fetch_data(tickers)
    print(f"Successfully fetched data for {len(data)}/{len(tickers)} tickers")

    results = []
    for t, df in data.items():
        try:
            r = stock_metrics(df, nifty_return)
            r["ticker"] = t
            results.append(r)
        except Exception as e:
            print(f"Scoring failed for {t}: {e}")

    if not results:
        print("No results to report — exiting without sending email.")
        return

    universe = build_universe(pd.DataFrame(results))
    print(universe["state"].value_counts().to_string())

    data_as_of = max(df.index[-1] for df in data.values()).strftime("%d %b %Y")
    warns = build_warnings(data, len(tickers))
    for w in warns:
        print("WARNING:", w)

    html = build_email_html(universe, nifty_return, data_as_of, warns)
    subject = ("[CHECK DATA] " if warns else "") + \
        f"Momentum Tracker — {datetime.now().strftime('%d %b %Y')} ({len(universe)} scanned)"
    send_email(subject, html)


if __name__ == "__main__":
    main()
