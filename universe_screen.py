"""
Universe screen, stage 1: price, liquidity and momentum for every stock in universe.txt.

Cheap on purpose: one batched price download per stock, no fundamentals. It does three things:
  1. Drops stocks you could not trade comfortably (liquidity floor), very low-priced stocks and recent listings.
  2. Ranks the rest on price momentum (3-month and 6-month returns, trend, distance from the 52-week high).
  3. Saves results/universe_prices.csv for the later stages (Screener pre-filter, then the full trackers).

Run weekly (see .github/workflows/universe_screen.yml). Needs universe.txt: one NSE symbol per line with .NS (TCS.NS).

Liquidity: median daily traded value (close x volume) over the last 20 sessions. Rule of thumb: a position should be about
5% of that or less, so the floor is roughly your usual position size x 20 (Rs 1,00,000 position -> Rs 20 lakh).
"""
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import momentum_tracker as m

UNIVERSE_FILE = os.environ.get("UNIVERSE_FILE", "universe.txt")
OUT_FILE = "results/universe_prices.csv"
MIN_LIQUIDITY = 25e5            # Rs 25 lakh a day (median traded value, 20 sessions). Change here.
SHOW_AT = [10e5, 25e5, 50e5, 1e7, 5e7]   # survivor counts are printed for these floors, to help you choose
MIN_PRICE = 20.0                # Rs
MIN_HISTORY = 210               # sessions needed for the SMA200 and a 6-month return
MIN_DOWNLOAD_ROWS = 40          # keep anything with this much data so short-history stocks can be counted and listed
MIN_COVERAGE = 0.85             # below this share downloaded, keep last week's file and warn
TOP_N = 30                      # rows in the email
LIQ_WINDOW = 20


def money(x):
    if pd.isna(x):
        return "-"
    return f"Rs {x / 1e7:.1f} cr" if x >= 1e7 else f"Rs {x / 1e5:.0f} lakh"


def metrics(t, df):
    c = df["Close"].astype(float)
    v = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(np.nan, index=df.index)
    n = len(c)
    out = {"ticker": t, "rows": n, "price": c.iloc[-1],
           "turnover": float((c * v).tail(LIQ_WINDOW).median()) if v.notna().any() else np.nan}
    for k, d in (("ret_1m", 21), ("ret_3m", 63), ("ret_6m", 126)):
        out[k] = (c.iloc[-1] / c.iloc[-1 - d] - 1) * 100 if n > d else np.nan
    hi = c.tail(252).max()
    out["off_high"] = (c.iloc[-1] / hi - 1) * 100
    out["above_sma200"] = bool(c.iloc[-1] > c.tail(200).mean()) if n >= 200 else None
    out["above_sma50"] = bool(c.iloc[-1] > c.tail(50).mean()) if n >= 50 else None
    return out


def build(data):
    df = pd.DataFrame([metrics(t, d) for t, d in data.items()])
    df["history_ok"] = df["rows"] >= MIN_HISTORY
    df["liquid"] = df["turnover"] >= MIN_LIQUIDITY
    df["price_ok"] = df["price"] >= MIN_PRICE
    df["eligible"] = df["history_ok"] & df["liquid"] & df["price_ok"]
    # Momentum score: percentile ranks among eligible stocks (weights as in the Momentum tracker's 21/63/126-day windows)
    el = df["eligible"]
    score = pd.Series(np.nan, index=df.index)
    parts = {"ret_1m": 0.25, "ret_3m": 0.45, "ret_6m": 0.30}
    s = sum(df.loc[el, k].rank(pct=True) * w for k, w in parts.items())
    score.loc[el] = s * 100
    df["momentum"] = score
    return df


def email_html(df, requested, missing, warns):
    el = df[df["eligible"]]
    h = f'<html><body style="font-family:Arial,sans-serif;font-size:13px;"><h2>Universe Screen, {datetime.now().strftime("%d %b %Y")}</h2>'
    h += (f"<p><b>{len(el)}</b> of {requested} stocks pass: median daily traded value of at least {money(MIN_LIQUIDITY)}, "
          f"price of at least Rs {MIN_PRICE:.0f} and {MIN_HISTORY}+ sessions of history. Price data came back for {len(df)}.</p>")
    for w in warns:
        h += f'<p style="color:#a00;">{w}</p>'
    h += "<p><b>How many stocks survive at different liquidity floors</b> (also with price and history filters)</p><table>"
    base = df["history_ok"] & df["price_ok"]
    for fl in SHOW_AT:
        mark = " (current)" if fl == MIN_LIQUIDITY else ""
        h += f'<tr><td style="padding:2px 10px;">{money(fl)} a day{mark}</td><td style="padding:2px 10px;"><b>{int((base & (df["turnover"] >= fl)).sum())}</b></td></tr>'
    h += "</table>"
    head = ["Ticker", "Momentum", "Price", "1M", "3M", "6M", "From 52w high", "Trend", "Daily value"]
    th = "".join(f'<th style="text-align:left;padding:3px 8px;background:#eee;font-size:12px;">{x}</th>' for x in head)
    top = el.sort_values("momentum", ascending=False).head(TOP_N)
    rows = ""
    for _, r in top.iterrows():
        trend = "Above SMA200" if r["above_sma200"] is True else "Below SMA200"
        cells = [r["ticker"].replace(".NS", ""), f'{r["momentum"]:.0f}', f'{r["price"]:.1f}', f'{r["ret_1m"]:+.0f}%', f'{r["ret_3m"]:+.0f}%',
                 f'{r["ret_6m"]:+.0f}%', f'{r["off_high"]:+.0f}%', trend, money(r["turnover"])]
        rows += "<tr>" + "".join(f'<td style="padding:3px 8px;font-size:12px;border-bottom:1px solid #eee;">{c}</td>' for c in cells) + "</tr>"
    h += f"<h3>Top {len(top)} by price momentum</h3><table style='border-collapse:collapse;'><tr>{th}</tr>{rows}</table>"
    if missing:
        h += f'<p style="font-size:12px;">No price data for {len(missing)} tickers (delisted, renamed or a Yahoo problem): ' + ", ".join(x.replace(".NS", "") for x in missing[:60]) + (" ..." if len(missing) > 60 else "") + "</p>"
    h += """<h3>How to read this</h3><p style="font-size:12px;">Momentum is a percentile rank (0 to 100) among the stocks that pass the filters, from 1, 3 and 6 month returns.
This is a price-only screen: it says nothing about business quality or valuation. The next stages add that.
The full list, with every stock and its filter results, is saved in results/universe_prices.csv.
Daily value is the median of close x volume over the last 20 sessions.</p></body></html>"""
    return h


def main():
    tickers = m.load_tickers(UNIVERSE_FILE)
    print(f"Loaded {len(tickers)} tickers from {UNIVERSE_FILE}")
    if not tickers:
        print(f"{UNIVERSE_FILE} is empty or missing. Exiting.")
        return
    data = m.fetch_data(tickers, min_rows=MIN_DOWNLOAD_ROWS)
    cover = len(data) / len(tickers)
    print(f"Price data for {len(data)}/{len(tickers)} tickers ({cover:.0%})")
    missing = [t for t in tickers if t not in data]
    warns = []
    if cover < MIN_COVERAGE:
        msg = (f"Price data came back for only {len(data)} of {len(tickers)} tickers ({cover:.0%}). Yahoo probably throttled the run, "
               "so last week's results file was kept. Run the workflow again later.")
        print("WARNING:", msg)
        try:
            m.send_email(f"Universe Screen, {datetime.now().strftime('%d %b %Y')} [CHECK DATA]", f"<html><body><p>{msg}</p></body></html>")
        except Exception as ex:
            print("Could not send the warning email:", ex)
        return
    df = build(data)
    print(df[["history_ok", "liquid", "price_ok", "eligible"]].sum().to_string())
    for fl in SHOW_AT:
        print(f"  floor {money(fl)}: {int((df['history_ok'] & df['price_ok'] & (df['turnover'] >= fl)).sum())} stocks")
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    df.assign(run_date=datetime.now().strftime("%Y-%m-%d")).to_csv(OUT_FILE, index=False)
    m.send_email(f"Universe Screen, {datetime.now().strftime('%d %b %Y')}", email_html(df, len(tickers), missing, warns))


if __name__ == "__main__":
    main()
