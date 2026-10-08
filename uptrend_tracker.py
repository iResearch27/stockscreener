"""
Confirmed Uptrend Tracker
Identifies stocks in a SUSTAINED, established uptrend — distinct from the
turnaround screener, which catches the transition INTO strength. This one
looks for trend strength and durability: proximity to 52-week highs,
persistence above key moving averages, structural higher highs/lows, and
relative strength vs. the Nifty 50 benchmark.
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
STOCKS_FILE = "uptrend_watchlist.txt"
BENCHMARK_TICKER = "^NSEI"   # Nifty 50 index
LOOKBACK_PERIOD = "2y"       # need 252+ trading days for 52-week high + SMA200
BATCH_SIZE = 50
MIN_UPTREND_SCORE = 4        # out of 6

NEAR_52WK_PCT = 10.0          # "near 52-week high" threshold, in %
SUSTAINED_DAYS = 20           # must have stayed above SMA50 this many trading days
SWING_LOOKBACK = 20           # window for higher-high / higher-low structure check
RS_LOOKBACK = 63              # ~3 months of trading days, for relative strength vs Nifty
RSI_SUSTAIN_DAYS = 10         # average RSI over this many days must stay above 50


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
                tdf = flatten_columns(df) if len(batch) == 1 else df[t]
                tdf = tdf.dropna(how="all")
                if len(tdf) < 260:
                    print(f"  Skipping {t}: insufficient history ({len(tdf)} rows)")
                    continue
                data[t] = tdf
            except Exception as e:
                print(f"  Skipping {t}: {e}")

        time.sleep(1)
    return data


def flatten_columns(df):
    """Newer yfinance versions return MultiIndex columns even for a single ticker.
    Flatten to plain column names (Open, High, Low, Close, Volume)."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_benchmark():
    df = yf.download(BENCHMARK_TICKER, period=LOOKBACK_PERIOD, interval="1d",
                      auto_adjust=True, progress=False)
    return flatten_columns(df.dropna(how="all"))


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


# ============================================================
# SCORING — 6 trend-strength conditions
# ============================================================
def score_stock(df, nifty_return_3m):
    close = df["Close"]

    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    rsi = compute_rsi(close, 14)
    high_52wk = close.rolling(252, min_periods=200).max()
    rolling_high = close.rolling(SWING_LOOKBACK).max()
    rolling_low = close.rolling(SWING_LOOKBACK).min()

    i = -1
    c_now = close.iloc[i]
    sma50_now, sma200_now = sma50.iloc[i], sma200.iloc[i]
    high_52wk_now = high_52wk.iloc[i]

    # ---- 1. Near 52-week high ----
    pct_from_high = (high_52wk_now - c_now) / high_52wk_now * 100
    c1_near52wkHigh = pct_from_high <= NEAR_52WK_PCT

    # ---- 2. Aligned bullish structure (today) ----
    c2_alignedStructure = c_now > sma50_now and sma50_now > sma200_now

    # ---- 3. Sustained above SMA50 for SUSTAINED_DAYS straight ----
    recent_close = close.tail(SUSTAINED_DAYS)
    recent_sma50 = sma50.tail(SUSTAINED_DAYS)
    c3_sustainedAboveSMA50 = bool((recent_close > recent_sma50).all())

    # ---- 4. Structural higher highs & higher lows ----
    c4_higherHighLow = (
        rolling_high.iloc[i] > rolling_high.iloc[i - SWING_LOOKBACK] and
        rolling_low.iloc[i] > rolling_low.iloc[i - SWING_LOOKBACK]
    )

    # ---- 5. Outperforming Nifty over ~3 months ----
    stock_return_3m = (c_now - close.iloc[i - RS_LOOKBACK]) / close.iloc[i - RS_LOOKBACK] * 100
    c5_outperformingNifty = stock_return_3m > nifty_return_3m

    # ---- 6. Sustained momentum: avg RSI over last N days > 50 ----
    avg_rsi = rsi.tail(RSI_SUSTAIN_DAYS).mean()
    c6_sustainedMomentum = avg_rsi > 50

    uptrend_score = sum([
        c1_near52wkHigh, c2_alignedStructure, c3_sustainedAboveSMA50,
        c4_higherHighLow, c5_outperformingNifty, c6_sustainedMomentum,
    ])

    return {
        "price": round(float(c_now), 2),
        "pct_from_52wk_high": round(float(pct_from_high), 2),
        "stock_return_3m": round(float(stock_return_3m), 2),
        "vs_nifty_3m": round(float(stock_return_3m - nifty_return_3m), 2),
        "rsi_avg10d": round(float(avg_rsi), 2),
        "uptrend_score": int(uptrend_score),
    }


# ============================================================
# EMAIL
# ============================================================
def build_email_html(results_df, nifty_return_3m):
    flagged = results_df[results_df.uptrend_score >= MIN_UPTREND_SCORE].sort_values(
        ["uptrend_score", "vs_nifty_3m"], ascending=False
    )
    all_sorted = results_df.sort_values(["uptrend_score", "vs_nifty_3m"], ascending=False)

    def rows(df):
        out = ""
        for _, r in df.iterrows():
            out += (
                f"<tr><td>{r['ticker']}</td><td>{r['price']}</td>"
                f"<td>{r['pct_from_52wk_high']}%</td><td>{r['stock_return_3m']}%</td>"
                f"<td>{r['vs_nifty_3m']}%</td><td>{r['rsi_avg10d']}</td>"
                f"<td>{r['uptrend_score']}</td></tr>"
            )
        return out

    header = (
        "<tr><th>Ticker</th><th>Price</th><th>% From 52wk High</th>"
        "<th>3M Return</th><th>vs Nifty (3M)</th><th>Avg RSI (10d)</th>"
        "<th>Uptrend/6</th></tr>"
    )

    return f"""
    <html><body style="font-family: Arial, sans-serif;">
    <h2>Confirmed Uptrend Tracker — {datetime.now().strftime('%d %b %Y')}</h2>
    <p>Nifty 50 3-month return: {round(nifty_return_3m, 2)}%</p>
    <p>{len(results_df)} stocks scanned, {len(flagged)} flagged (Uptrend score ≥ {MIN_UPTREND_SCORE}/6).</p>

    <h3>Flagged</h3>
    <table border="1" cellpadding="6" cellspacing="0">{header}{rows(flagged)}</table>

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

    if not tickers:
        print(f"{STOCKS_FILE} is empty — add tickers before running. Exiting.")
        return

    nifty_df = fetch_benchmark()
    if len(nifty_df) < RS_LOOKBACK + 5:
        print("Not enough Nifty history to compute relative strength. Exiting.")
        return
    nifty_close = nifty_df["Close"]
    nifty_return_3m = float(
        (nifty_close.iloc[-1] - nifty_close.iloc[-1 - RS_LOOKBACK])
        / nifty_close.iloc[-1 - RS_LOOKBACK] * 100
    )
    print(f"Nifty 50 3-month return: {round(nifty_return_3m, 2)}%")

    data = fetch_data(tickers)
    print(f"Successfully fetched data for {len(data)}/{len(tickers)} tickers")

    results = []
    for t, df in data.items():
        try:
            r = score_stock(df, nifty_return_3m)
            r["ticker"] = t
            results.append(r)
        except Exception as e:
            print(f"Scoring failed for {t}: {e}")

    if not results:
        print("No results to report — exiting without sending email.")
        return

    results_df = pd.DataFrame(results)
    html = build_email_html(results_df, nifty_return_3m)
    subject = f"Confirmed Uptrend Tracker — {datetime.now().strftime('%d %b %Y')} ({len(results_df)} scanned)"
    send_email(subject, html)


if __name__ == "__main__":
    main()
