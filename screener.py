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
                tdf = tdf.dropna(how="all")
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

    return {
        "price": round(float(c_now), 2),
        "rsi": round(float(rsi_now), 2),
        "pct_to_sma50": round(float(pct_to_sma50), 2),
        "confirmed_score": int(confirmed_score),
        "early_score": int(early_score),
    }


# ============================================================
# EMAIL
# ============================================================
def build_email_html(results_df):
    flagged = results_df[
        (results_df.confirmed_score >= MIN_CONFIRMED_SCORE) |
        (results_df.early_score >= MIN_EARLY_SCORE)
    ].sort_values(["early_score", "confirmed_score"], ascending=False)

    all_sorted = results_df.sort_values(["early_score", "confirmed_score"], ascending=False)

    def rows(df):
        out = ""
        for _, r in df.iterrows():
            out += (
                f"<tr><td>{r['ticker']}</td><td>{r['price']}</td><td>{r['rsi']}</td>"
                f"<td>{r['pct_to_sma50']}%</td><td>{r['early_score']}</td>"
                f"<td>{r['confirmed_score']}</td></tr>"
            )
        return out

    header = "<tr><th>Ticker</th><th>Price</th><th>RSI</th><th>% to SMA50</th><th>Early/6</th><th>Confirmed/6</th></tr>"

    return f"""
    <html><body style="font-family: Arial, sans-serif;">
    <h2>Turnaround Screener — {datetime.now().strftime('%d %b %Y')}</h2>
    <p>{len(results_df)} stocks scanned, {len(flagged)} flagged (Confirmed ≥ {MIN_CONFIRMED_SCORE} or Early ≥ {MIN_EARLY_SCORE}).</p>

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
    html = build_email_html(results_df)
    subject = f"Turnaround Screener — {datetime.now().strftime('%d %b %Y')} ({len(results_df)} scanned)"
    send_email(subject, html)


if __name__ == "__main__":
    main()
