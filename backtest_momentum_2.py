"""
Momentum backtest #2: which rules actually have an edge?

The first backtest showed that the strength states (Strong, Extended) beat the all-stocks baseline, while the
recovery states (Building, Emerging) did not. This second backtest digs into that:

  1. EDGE, not raw returns: for every rule, the average over dates of (rule's average forward return minus
     all-stocks average forward return on the same date). That strips out market-wide moves.
  2. RISK: the worst fall from entry during the next 63 days, and how often it exceeded -15%.
  3. HONESTY CHECKS: results are split into a first half (the "tuning" half) and a second half (the "test" half).
     A rule only counts as holding up if its edge is positive in BOTH halves. A rough t-statistic is shown
     (|t| above about 2 suggests the edge is unlikely to be luck). Results are also split into up and down markets.
  4. CANDIDATES: the current states and flags (Stabilizing, Pullback to SMA50), several tighter versions of
     Building, and some simple alternatives (for example just "top 20% by relative strength").

The states are replayed ONCE with the live tracker's own functions; every candidate rule is then evaluated on
the stored metrics, so rules can be compared cheaply and cannot drift from the live code.

Run it manually: Actions -> Momentum Backtest 2 -> Run workflow. Results are emailed.

Limits: today's watchlist (survivorship bias), overlapping observations (the t-statistic allows for this roughly),
no costs or slippage, one 3.5-year window with one market regime, and testing many candidates raises the chance
that one looks good by luck. Do not tune the live tracker on this report alone.
"""

import math
import os
from datetime import datetime

import numpy as np
import pandas as pd

import momentum_tracker as m
import backtest_momentum as b1

# ============================================================
# CONFIG
# ============================================================
BT_PERIOD = "5y"
STEP = int(os.environ.get("BT_STEP", "5"))
MAX_TICKERS = int(os.environ.get("BT_MAX_TICKERS", "0"))
WINDOW_ROWS = 500
FWD_SHORT, FWD_LONG = 21, 63
DEEP_DROP = -15.0               # "deep drop" threshold for the risk column, in %
MIN_OBS_HALF = 100              # a half with fewer observations than this is called "thin"
MIN_DATES_T = 8                 # minimum dates needed to compute a t-statistic

KEEP_COLS = [
    "ticker", "date", "state", "liquid", "rs_rank", "rs_change", "rs_change_fast", "trend_pts",
    "participation", "participation_fast", "ret_1m", "ret_3m", "ret_6m", "ret_fast", "pct_above_sma50",
    "vs_nifty", "stabilizing", "above_sma20", "trend_quality", "rsi",
    "sma50_gt_200", "c_gt_200", "sma50_rising", "off_peak_pct",
]


# ============================================================
# REPLAY (once) with a few extra fields needed by the candidate rules
# ============================================================
def extra_metrics(sl):
    close = sl["Close"]
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    c = float(close.iloc[-1])
    peak20 = float(close.tail(20).max())
    return {
        "sma50_gt_200": bool(sma50.iloc[-1] > sma200.iloc[-1]),
        "c_gt_200": bool(c > sma200.iloc[-1]),
        "sma50_rising": bool(sma50.iloc[-1] > sma50.iloc[-1 - m.TREND_SLOPE_DAYS]),
        "off_peak_pct": (peak20 - c) / peak20 * 100,
    }


def replay(data, nifty_close, dates):
    out = []
    for k, d in enumerate(dates):
        nifty_ret = float(m.pct_return(nifty_close.loc[:d], m.NIFTY_LOOKBACK))
        rows = []
        for t, df in data.items():
            pos = df.index.searchsorted(d, side="right")
            if pos < m.MIN_ROWS or (d - df.index[pos - 1]).days > 5:
                continue
            sl = df.iloc[max(0, pos - WINDOW_ROWS):pos]
            try:
                r = m.stock_metrics(sl, nifty_ret)
                r.update(extra_metrics(sl))
            except Exception:
                continue
            r["ticker"] = t
            rows.append(r)
        if len(rows) < 10:
            continue
        uni = m.build_universe(pd.DataFrame(rows))
        uni["date"] = d
        out.append(uni[KEEP_COLS])
        if (k + 1) % 20 == 0:
            print(f"    {k + 1}/{len(dates)} dates replayed")
    return pd.concat(out, ignore_index=True)


def add_forward(obs, data, nifty_close):
    """Forward returns, excess vs Nifty, and the worst fall from entry over the next FWD_LONG days."""
    nidx, nvals = nifty_close.index, nifty_close.values
    pieces = []
    for t, g in obs.groupby("ticker"):
        df = data[t]
        closes = df["Close"].values
        pos = df.index.searchsorted(g["date"].values, side="right") - 1
        g = g.copy()
        for h, col in ((FWD_SHORT, "f21"), (FWD_LONG, "f63")):
            ok = pos + h < len(closes)
            g[col] = np.where(ok, closes[np.minimum(pos + h, len(closes) - 1)] / closes[pos] - 1, np.nan) * 100
        mdd = np.full(len(g), np.nan)
        for i, p in enumerate(pos):
            if p + FWD_LONG < len(closes):
                mdd[i] = (closes[p + 1:p + FWD_LONG + 1].min() / closes[p] - 1) * 100
        g["mdd63"] = mdd
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


# ============================================================
# CANDIDATE RULES
# ============================================================
PULLBACK_BELOW_PCT, PULLBACK_ABOVE_PCT, PULLBACK_MIN_DROP_PCT = 2.0, 3.0, 3.0   # same as uptrend_tracker.py


def pullback(d):
    """Uptrend tracker's Pullback-to-SMA50 flag, rebuilt from stored metrics."""
    return (d.sma50_gt_200 & d.sma50_rising & d.c_gt_200 & (d.vs_nifty > 0)
            & (d.pct_above_sma50 >= -PULLBACK_BELOW_PCT) & (d.pct_above_sma50 <= PULLBACK_ABOVE_PCT)
            & (d.off_peak_pct >= PULLBACK_MIN_DROP_PCT))


GROUPS = [
    ("Current states and flags", [
        ("Strong", lambda d: d.state == "Strong"),
        ("Extended", lambda d: d.state == "Extended"),
        ("Building", lambda d: d.state == "Building"),
        ("Emerging", lambda d: d.state == "Emerging"),
        ("Fading", lambda d: d.state == "Fading"),
        ("Stabilizing flag", lambda d: d.stabilizing),
        ("Pullback to SMA50 flag", pullback),
    ]),
    ("Tighter versions of Building", [
        ("Building + RS rank >= 50", lambda d: (d.state == "Building") & (d.rs_rank >= 50)),
        ("Building + all 3 participation checks", lambda d: (d.state == "Building") & (d.participation == 3)),
        ("Building + own 3M return > 0", lambda d: (d.state == "Building") & (d.ret_3m > 0)),
        ("Building + beating Nifty (3M)", lambda d: (d.state == "Building") & (d.vs_nifty > 0)),
        ("Building + above SMA50", lambda d: (d.state == "Building") & (d.pct_above_sma50 > 0)),
        ("Building + trend quality > 0", lambda d: (d.state == "Building") & (d.trend_quality > 0)),
    ]),
    ("Simple alternatives", [
        ("Reclaim: rank up 10+, above SMA50 and SMA50 rising, 2+ participation",
         lambda d: (d.rs_change >= 10) & (d.pct_above_sma50 > 0) & d.sma50_rising & (d.participation >= 2)),
        ("Strong, within 5% of SMA50", lambda d: (d.state == "Strong") & (d.pct_above_sma50 <= 5)),
        ("Pullback flag + RS rank >= 60", lambda d: pullback(d) & (d.rs_rank >= 60)),
        ("Top 20% by RS rank", lambda d: d.rs_rank >= 80),
        ("Top 20% by RS rank, beating Nifty, 6M return > 0",
         lambda d: (d.rs_rank >= 80) & (d.vs_nifty > 0) & (d.ret_6m > 0)),
    ]),
]
CORE = [name for name, _ in GROUPS[0][1]]


# ============================================================
# STATISTICS
# ============================================================
def edge_and_t(sub, base, col, horizon):
    """Average over dates of (rule mean - all-stocks mean that date), and a rough t-statistic."""
    s = sub[sub[col].notna()].groupby("date")[col].mean()
    bmean = base[base[col].notna()].groupby("date")[col].mean()
    e = (s - bmean.reindex(s.index)).dropna()
    if len(e) < 3:
        return np.nan, np.nan
    mean_edge = float(e.mean())
    if len(e) < MIN_DATES_T or e.std(ddof=1) == 0:
        return mean_edge, np.nan
    block = max(1, math.ceil(horizon / STEP))          # neighboring dates overlap, so shrink the effective sample
    t = mean_edge / (e.std(ddof=1) / math.sqrt(len(e) / block))
    return mean_edge, float(t)


def full_stats(sub, base):
    e21, t21 = edge_and_t(sub, base, "f21", FWD_SHORT)
    e63, t63 = edge_and_t(sub, base, "f63", FWD_LONG)
    f21 = sub["f21"].dropna()
    mdd = sub["mdd63"].dropna()
    return {
        "Obs": len(sub),
        "Avg 21d": f21.mean() if len(f21) else np.nan,
        "Hit rate 21d": (f21 > 0).mean() * 100 if len(f21) else np.nan,
        "Edge 21d (pts)": e21, "t 21d": t21,
        "Edge 63d (pts)": e63, "t 63d": t63,
        "Avg worst fall (63d)": mdd.mean() if len(mdd) else np.nan,
        "Share with fall > 15%": (mdd < DEEP_DROP).mean() * 100 if len(mdd) else np.nan,
    }


def split_stats(sub, base):
    e21, _ = edge_and_t(sub, base, "f21", FWD_SHORT)
    e63, _ = edge_and_t(sub, base, "f63", FWD_LONG)
    return len(sub), e21, e63


# ============================================================
# REPORT
# ============================================================
MAIN_FMT = {"Obs": "{:,.0f}", "Avg 21d": "{:+.1f}%", "Hit rate 21d": "{:.0f}%", "Edge 21d (pts)": "{:+.1f}",
            "t 21d": "{:.1f}", "Edge 63d (pts)": "{:+.1f}", "t 63d": "{:.1f}",
            "Avg worst fall (63d)": "{:.1f}%", "Share with fall > 15%": "{:.0f}%"}
HALF_FMT = {"Obs": "{:,.0f}", "Edge 21d": "{:+.1f}", "Edge 63d": "{:+.1f}"}


def verdict(h1, h2):
    (n1, a1, b1_), (n2, a2, b2) = h1, h2
    if n1 < MIN_OBS_HALF or n2 < MIN_OBS_HALF:
        return "Thin sample"
    vals = [a1, b1_, a2, b2]
    if any(pd.isna(v) for v in vals):
        return "Unclear"
    return "Yes" if all(v > 0 for v in vals) else "No"


def build_report(obs, dates, n_stocks):
    liquid = obs[obs.liquid]
    split = pd.Timestamp(pd.Series(dates).quantile(0.5, interpolation="nearest"))
    first, second = liquid[liquid.date <= split], liquid[liquid.date > split]

    main_rows, half_rows, regime_rows, groups_idx = {}, {}, {}, []
    nifty63 = (liquid["ret_3m"] - liquid["vs_nifty"]).groupby(liquid["date"]).median()
    up_dates = set(nifty63[nifty63 > 0].index)
    up, down = liquid[liquid.date.isin(up_dates)], liquid[~liquid.date.isin(up_dates)]

    for group, rules in GROUPS:
        names = []
        for name, fn in rules:
            mask = fn(liquid).fillna(False)
            sub = liquid[mask]
            if not len(sub):
                continue
            row = full_stats(sub, liquid)
            h1 = split_stats(first[fn(first).fillna(False)], first)
            h2 = split_stats(second[fn(second).fillna(False)], second)
            row["Holds in both halves?"] = verdict(h1, h2)
            main_rows[name] = row
            half_rows[name] = {"1st half Obs": h1[0], "1st half Edge 21d": h1[1], "1st half Edge 63d": h1[2],
                               "2nd half Obs": h2[0], "2nd half Edge 21d": h2[1], "2nd half Edge 63d": h2[2]}
            if name in CORE:
                u = split_stats(up[fn(up).fillna(False)], up)
                dn = split_stats(down[fn(down).fillna(False)], down)
                regime_rows[name] = {"Up-mkt Obs": u[0], "Up-mkt Edge 21d": u[1], "Up-mkt Edge 63d": u[2],
                                     "Down-mkt Obs": dn[0], "Down-mkt Edge 21d": dn[1], "Down-mkt Edge 63d": dn[2]}
            names.append(name)
        groups_idx.append((group, names))

    main_df = pd.DataFrame(main_rows).T
    half_df = pd.DataFrame(half_rows).T
    regime_df = pd.DataFrame(regime_rows).T
    return main_df, half_df, regime_df, groups_idx, split


def render_html(main_df, half_df, regime_df, groups_idx, split, n_stocks, n_dates, span):
    sections = ""
    for group, names in groups_idx:
        if not names:
            continue
        sections += f"<h3>{group}</h3>" + b1.df_to_html(main_df.loc[names], {**MAIN_FMT, "Holds in both halves?": "{}"})

    half_fmt = {**HALF_FMT, "1st half Obs": "{:,.0f}", "2nd half Obs": "{:,.0f}",
                "1st half Edge 21d": "{:+.1f}", "1st half Edge 63d": "{:+.1f}",
                "2nd half Edge 21d": "{:+.1f}", "2nd half Edge 63d": "{:+.1f}"}
    reg_fmt = {"Up-mkt Obs": "{:,.0f}", "Down-mkt Obs": "{:,.0f}", "Up-mkt Edge 21d": "{:+.1f}",
               "Up-mkt Edge 63d": "{:+.1f}", "Down-mkt Edge 21d": "{:+.1f}", "Down-mkt Edge 63d": "{:+.1f}"}

    return f"""
    <html><body style="font-family: Arial, sans-serif;">
    <h1>Momentum Backtest #2: which rules have an edge? — {datetime.now().strftime('%d %b %Y')}</h1>
    <p>{n_stocks} stocks, {n_dates} evaluation dates (every {STEP} trading days), {span}. Liquid stocks only.
    Split point between the two halves: {split.strftime('%d %b %Y')}.</p>

    <h3>How to read this</h3>
    <ul style="font-size:13px;">
    <li><b>Edge (pts)</b> = on average, how many percentage points the rule's stocks beat ALL stocks on the same date over the next
    21 / 63 trading days. Positive is good. It removes market-wide moves.</li>
    <li><b>t</b> = a rough reliability score for the edge. Above about +2 (or below -2) suggests it is unlikely to be luck. It allows for overlapping observations
    only roughly, so treat it as a guide.</li>
    <li><b>Avg worst fall</b> = the average of the largest fall from the entry price within the next 63 days. <b>Share with fall &gt; 15%</b> = how often that fall exceeded 15%.</li>
    <li><b>Holds in both halves?</b> = Yes only if the edge is positive at both horizons in BOTH the first and second half. A rule that works only in one half is probably luck or a regime effect.</li>
    </ul>
    <p style="color:#a00;font-size:12px;"><b>Caution:</b> many rules are tested here, so some will look good by chance. Prefer rules that are Yes in both halves,
    have a clearly positive t, and make sense on their own. Survivorship bias (today's watchlist), overlapping observations, no costs, and one market regime
    still apply. Not investment advice.</p>

    <h2>1. Edge, risk and consistency</h2>
    {sections}

    <h2>2. First half vs second half</h2>
    <p style="color:#666;font-size:12px;">Edge in points. The first half is where you would be tempted to tune; the second half is the fairer test.</p>
    {b1.df_to_html(half_df, half_fmt)}

    <h2>3. Up markets vs down markets</h2>
    <p style="color:#666;font-size:12px;">Up market = dates when Nifty's own 3-month return was positive; down market = it was zero or negative.
    Rules that only work in up markets will disappoint in a falling market.</p>
    {b1.df_to_html(regime_df, reg_fmt)}
    </body></html>
    """


# ============================================================
# MAIN
# ============================================================
def main():
    tickers = m.load_tickers(m.STOCKS_FILE)
    if MAX_TICKERS:
        tickers = tickers[:MAX_TICKERS]
    print(f"Loaded {len(tickers)} tickers from {m.STOCKS_FILE}")

    m.LOOKBACK_PERIOD = BT_PERIOD
    nifty_df = m.fetch_benchmark()
    data = m.fetch_data(tickers)
    for df in list(data.values()) + [nifty_df]:
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
    nifty_close = nifty_df["Close"]
    print(f"Fetched {len(data)}/{len(tickers)} tickers, {len(nifty_close)} Nifty rows")
    if len(data) < 20:
        print("Too few stocks loaded to backtest. Exiting.")
        return

    dates = b1.eval_dates(nifty_close)
    span = f"{dates[0].strftime('%b %Y')} to {dates[-1].strftime('%b %Y')}"
    print(f"Replaying {len(dates)} dates...")
    obs = replay(data, nifty_close, dates)
    obs = add_forward(obs, data, nifty_close)
    print(f"{len(obs):,} observations")

    main_df, half_df, regime_df, groups_idx, split = build_report(obs, dates, len(data))
    pd.set_option("display.width", 250)
    print("\n=== Edge, risk and consistency ===\n", main_df.round(1).to_string())
    print("\n=== Halves ===\n", half_df.round(1).to_string())
    print("\n=== Regimes ===\n", regime_df.round(1).to_string())

    html = render_html(main_df, half_df, regime_df, groups_idx, split, len(data), len(dates), span)
    m.send_email(f"Momentum Backtest #2 — {datetime.now().strftime('%d %b %Y')}", html)


if __name__ == "__main__":
    main()
