"""
Momentum tracker backtest
Replays the Momentum tracker's states over ~5 years of history for the stocks in the watchlist, then answers:

  1. What did each state go on to do?  (average / median forward return, hit rate, excess vs Nifty)
  2. How early does each state show up?  For every major turning point (a low followed by a 15%+ rally),
     how many days after the low did Emerging / Building / Strong first appear, and how much of the rally
     was already gone by then?
  3. Do faster windows help?  Everything is run twice: the current settings and a faster-window variant.

It calls the SAME functions as the live tracker (momentum_tracker.py), so the rules cannot drift apart.
Run it manually from GitHub: Actions -> Momentum Backtest -> Run workflow. Results are emailed.

Limits (also printed in the email): the watchlist is today's list, so it has survivorship bias; observations
overlap; no costs or slippage; states are sampled every few days; the thresholds were not tuned on this data.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd

import momentum_tracker as m

# ============================================================
# CONFIG
# ============================================================
BT_PERIOD = "5y"                                  # history to download
STEP = int(os.environ.get("BT_STEP", "5"))        # evaluate the states every N trading days
MAX_TICKERS = int(os.environ.get("BT_MAX_TICKERS", "0"))   # 0 = all (set a small number for a quick test)
WINDOW_ROWS = 500                                 # rows of history given to the tracker on each date (live uses ~2 years)
HORIZONS = (10, 21, 63)                           # forward-return horizons, trading days
TP_WINDOW = 40                                    # a turning point is the lowest close within +/- this many days...
TP_FWD = 63                                       # ...followed by a rally within this many days...
TP_MIN_RALLY = 15.0                               # ...of at least this many percent
TARGET_STATES = ["Emerging", "Building", "Strong"]
REPORT_STATES = ["Emerging", "Building", "Strong", "Extended", "Fading", "Neutral", "Dormant"]

VARIANTS = {
    "Current windows": {},
    "Faster windows": {
        "RS_CHANGE_LOOKBACK": 10,
        "RS_WEIGHTS": {21: 0.35, 63: 0.35, 126: 0.15, 252: 0.15},
        "OBV_DAYS": 10,
        "UPDOWN_DAYS": 10,
        "TREND_SLOPE_DAYS": 5,
        "RET_1M_DAYS": 10,
    },
}


# ============================================================
# REPLAY
# ============================================================
def eval_dates(nifty_close):
    start = m.MIN_ROWS + 20
    return list(nifty_close.index[start::STEP])


def replay_states(data, nifty_close, dates):
    """Return a DataFrame (ticker, date, state) with the state of every stock on every evaluation date."""
    out = []
    for k, d in enumerate(dates):
        nifty_slice = nifty_close.loc[:d]
        nifty_ret = float(m.pct_return(nifty_slice, m.NIFTY_LOOKBACK))
        rows = []
        for t, df in data.items():
            pos = df.index.searchsorted(d, side="right")
            if pos < m.MIN_ROWS or (d - df.index[pos - 1]).days > 5:
                continue
            sl = df.iloc[max(0, pos - WINDOW_ROWS):pos]
            try:
                r = m.stock_metrics(sl, nifty_ret)
            except Exception:
                continue
            r["ticker"] = t
            rows.append(r)
        if len(rows) < 10:
            continue
        uni = m.build_universe(pd.DataFrame(rows))
        out.append(pd.DataFrame({"ticker": uni["ticker"], "date": d, "state": uni["state"]}))
        if (k + 1) % 20 == 0:
            print(f"    {k + 1}/{len(dates)} dates replayed")
    return pd.concat(out, ignore_index=True)


def add_forward_returns(states, data, nifty_close):
    """Attach forward returns (stock, and stock minus Nifty) at each horizon."""
    nifty_idx = nifty_close.index
    nifty_vals = nifty_close.values
    pieces = []
    for t, g in states.groupby("ticker"):
        df = data[t]
        closes = df["Close"].values
        pos = df.index.searchsorted(g["date"].values, side="right") - 1
        npos = nifty_idx.searchsorted(g["date"].values, side="right") - 1
        g = g.copy()
        for h in HORIZONS:
            ok = pos + h < len(closes)
            nok = npos + h < len(nifty_vals)
            fwd = np.where(ok, closes[np.minimum(pos + h, len(closes) - 1)] / closes[pos] - 1, np.nan) * 100
            nfwd = np.where(nok, nifty_vals[np.minimum(npos + h, len(nifty_vals) - 1)] / nifty_vals[npos] - 1, np.nan) * 100
            g[f"f{h}"] = fwd
            g[f"x{h}"] = fwd - nfwd
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def forward_table(obs):
    """Forward-return summary by state, plus an all-stocks baseline row."""
    def summarize(g):
        return pd.Series({
            "Obs": len(g),
            "Avg 21d": g["f21"].mean(),
            "Median 21d": g["f21"].median(),
            "Hit rate 21d": (g["f21"] > 0).mean() * 100 if g["f21"].notna().any() else np.nan,
            "Excess 21d": g["x21"].mean(),
            "Avg 63d": g["f63"].mean(),
            "Excess 63d": g["x63"].mean(),
        })
    rows = {}
    for st in REPORT_STATES:
        g = obs[(obs.state == st) & obs["f21"].notna()]
        if len(g):
            rows[st] = summarize(g)
    base = obs[obs["f21"].notna() & (obs.state != "Low liquidity")]
    rows["All stocks (baseline)"] = summarize(base)
    return pd.DataFrame(rows).T


# ============================================================
# TURNING POINTS: how early does each state show up?
# ============================================================
def find_turning_points(closes):
    tps = []
    n = len(closes)
    for t0 in range(TP_WINDOW, n - TP_FWD - 1):
        c0 = closes[t0]
        if c0 > closes[t0 - TP_WINDOW:t0 + TP_WINDOW + 1].min():
            continue
        peak = closes[t0:t0 + TP_FWD + 1].max()
        if peak >= c0 * (1 + TP_MIN_RALLY / 100):
            tps.append(t0)
    return tps


def capture_table(states, data):
    by_ticker = {t: g for t, g in states.groupby("ticker")}
    first_eval = states["date"].min()
    stats = {st: {"lag": [], "gone": [], "n": 0, "caught": 0} for st in TARGET_STATES + ["First of the three"]}

    for t, df in data.items():
        if t not in by_ticker:
            continue
        closes = df["Close"].values
        idx = df.index
        g = by_ticker[t].copy()
        g["pos"] = idx.searchsorted(g["date"].values, side="right") - 1
        for t0 in find_turning_points(closes):
            if idx[t0] < first_eval:
                continue
            window = g[(g["pos"] >= t0) & (g["pos"] <= t0 + TP_FWD)]
            peak = closes[t0:t0 + TP_FWD + 1].max()
            denom = peak - closes[t0]
            firsts = []
            for st in TARGET_STATES:
                stats[st]["n"] += 1
                hit = window[window["state"] == st]
                if len(hit):
                    p = int(hit["pos"].iloc[0])
                    stats[st]["caught"] += 1
                    stats[st]["lag"].append(p - t0)
                    stats[st]["gone"].append((closes[p] - closes[t0]) / denom * 100)
                    firsts.append(p)
            stats["First of the three"]["n"] += 1
            if firsts:
                p = min(firsts)
                stats["First of the three"]["caught"] += 1
                stats["First of the three"]["lag"].append(p - t0)
                stats["First of the three"]["gone"].append((closes[p] - closes[t0]) / denom * 100)

    rows = {}
    for st, s in stats.items():
        rows[st] = {
            "Turning points": s["n"],
            "Caught": (s["caught"] / s["n"] * 100) if s["n"] else np.nan,
            "Median lag (days after low)": float(np.median(s["lag"])) if s["lag"] else np.nan,
            "Median % of rally already gone": float(np.median(s["gone"])) if s["gone"] else np.nan,
        }
    return pd.DataFrame(rows).T


# ============================================================
# REPORT
# ============================================================
def df_to_html(df, fmt):
    head = "<tr><th></th>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr>"
    body = ""
    for idx, r in df.iterrows():
        cells = ""
        for c in df.columns:
            v = r[c]
            f = fmt.get(c, "{:.1f}")
            cells += "<td>" + ("-" if pd.isna(v) else f.format(v)) + "</td>"
        body += f"<tr><td><b>{idx}</b></td>{cells}</tr>"
    return f"<table border='1' cellpadding='5' cellspacing='0'>{head}{body}</table>"


FWD_FMT = {"Obs": "{:,.0f}", "Avg 21d": "{:+.1f}%", "Median 21d": "{:+.1f}%", "Hit rate 21d": "{:.0f}%",
           "Excess 21d": "{:+.1f}%", "Avg 63d": "{:+.1f}%", "Excess 63d": "{:+.1f}%"}
CAP_FMT = {"Turning points": "{:,.0f}", "Caught": "{:.0f}%", "Median lag (days after low)": "{:.0f}",
           "Median % of rally already gone": "{:.0f}%"}


def build_report_html(results, n_stocks, n_dates, span):
    sections = ""
    for name, (fwd, cap) in results.items():
        sections += (
            f"<h2>{name}</h2>"
            "<h3>What each state went on to do</h3>"
            "<p style='color:#666;font-size:12px;'>Forward returns from the evaluation date. Excess = stock minus Nifty over the same period. "
            "Compare each row with the baseline: a state is only useful if it beats it.</p>"
            f"{df_to_html(fwd, FWD_FMT)}"
            "<h3>How early did each state show up at a real turning point?</h3>"
            f"<p style='color:#666;font-size:12px;'>Turning point = lowest close within +/-{TP_WINDOW} days that was followed by a rally of "
            f"{TP_MIN_RALLY:g}% or more within {TP_FWD} days. Lag = trading days from that low to the first time the state appeared. "
            "% of rally already gone = how much of the low-to-peak move had happened by then (lower is better). "
            "Lag has a resolution of about the sampling step.</p>"
            f"{df_to_html(cap, CAP_FMT)}"
        )
    return f"""
    <html><body style="font-family: Arial, sans-serif;">
    <h1>Momentum Tracker Backtest — {datetime.now().strftime('%d %b %Y')}</h1>
    <p>{n_stocks} stocks, {n_dates} evaluation dates (every {STEP} trading days), {span}.</p>
    <p style="color:#666;font-size:12px;"><b>Read with care.</b> The watchlist is today's list, so it favors stocks that survived and did well
    (survivorship bias). Observations on neighboring dates overlap, so the sample is smaller than it looks. No costs or slippage.
    The state thresholds were not tuned on this data, but if you tune them using this report, results will look better than they will be live.
    Not investment advice.</p>
    {sections}
    </body></html>
    """


# ============================================================
# MAIN
# ============================================================
def run_variant(overrides, data, nifty_close, dates):
    saved = {k: getattr(m, k) for k in overrides}
    try:
        for k, v in overrides.items():
            setattr(m, k, v)
        states = replay_states(data, nifty_close, dates)
    finally:
        for k, v in saved.items():
            setattr(m, k, v)
    obs = add_forward_returns(states, data, nifty_close)
    return forward_table(obs), capture_table(states, data)


def main():
    tickers = m.load_tickers(m.STOCKS_FILE)
    if MAX_TICKERS:
        tickers = tickers[:MAX_TICKERS]
    print(f"Loaded {len(tickers)} tickers from {m.STOCKS_FILE}")

    m.LOOKBACK_PERIOD = BT_PERIOD
    nifty_df = m.fetch_benchmark()
    data = m.fetch_data(tickers)
    # Make every index timezone-naive so dates compare cleanly.
    for df in list(data.values()) + [nifty_df]:
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
    nifty_close = nifty_df["Close"]
    print(f"Fetched {len(data)}/{len(tickers)} tickers, {len(nifty_close)} Nifty rows")
    if len(data) < 20:
        print("Too few stocks loaded to backtest. Exiting.")
        return

    dates = eval_dates(nifty_close)
    span = f"{dates[0].strftime('%b %Y')} to {dates[-1].strftime('%b %Y')}"

    results = {}
    for name, overrides in VARIANTS.items():
        print(f"Replaying: {name} ({len(dates)} dates)...")
        results[name] = run_variant(overrides, data, nifty_close, dates)
        fwd, cap = results[name]
        print(f"\n=== {name}: forward returns ===\n{fwd.round(1).to_string()}")
        print(f"\n=== {name}: turning-point capture ===\n{cap.round(1).to_string()}\n")

    html = build_report_html(results, len(data), len(dates), span)
    m.send_email(f"Momentum Backtest — {datetime.now().strftime('%d %b %Y')}", html)


if __name__ == "__main__":
    main()
