"""
Momentum backtest #3: can relative strength find winners EARLY?

Backtest #2 found that the tracker's recovery states (Building, Emerging) had no edge and carried more downside risk,
while one simple idea, "top 20% by relative strength rank", was the only rule with a t-score near 2 (though its edge faded
in the second half). This backtest explores that family:

  1. QUINTILES: does forward performance rise steadily with RS rank? (rank < 20, 20-40, 40-60, 60-80, 80+)
     A steady staircase is far more convincing than one lucky bucket.
  2. EARLY LEADERS: stocks that have just moved into the top group (a rank jump), or are rising through the ranks.
  3. 52-WEEK-HIGH IDEAS: near a 52-week high, or a fresh 52-week breakout, with and without a volume surge.
  4. REFINEMENTS OF LEADERS: leaders that are not yet Extended, leaders sitting near their SMA50, leaders with a smooth trend.

Same honesty checks as backtest #2: edge versus all stocks on the same date, a rough t-score, worst fall within 63 days
(with an all-stocks baseline row), first-half vs second-half, and up vs down markets.

It reuses the replay and statistics from backtest_momentum_2.py (so that file must be in the repo too).
Run it manually: Actions -> Momentum Backtest 3 -> Run workflow. Results are emailed.

Limits: as in #1 and #2 (today's watchlist, overlapping observations, no costs, one market regime). About 20 rules are
tested, so one or two will look good by luck; treat |t| below about 3 as suggestive only.
"""

import os

import pandas as pd

import momentum_tracker as m
import backtest_momentum as b1
import backtest_momentum_2 as b2

# ============================================================
# CANDIDATE RULES
# ============================================================
GROUPS = [
    ("Relative strength quintiles (does performance rise step by step with rank?)", [
        ("RS rank below 20", lambda d: d.rs_rank < 20),
        ("RS rank 20 to 40", lambda d: (d.rs_rank >= 20) & (d.rs_rank < 40)),
        ("RS rank 40 to 60", lambda d: (d.rs_rank >= 40) & (d.rs_rank < 60)),
        ("RS rank 60 to 80", lambda d: (d.rs_rank >= 60) & (d.rs_rank < 80)),
        ("RS rank 80 and above (top 20%)", lambda d: d.rs_rank >= 80),
    ]),
    ("Early leaders: moving up the ranks", [
        ("Just entered top 20% (was below 60 twenty days ago)", lambda d: (d.rs_rank >= 80) & (d.rs_rank_prev < 60)),
        ("Just entered top 20% (was below 70 ten days ago)", lambda d: (d.rs_rank >= 80) & (d.rs_rank_prev_fast < 70)),
        ("Rising through the ranks (rank 70 to 90, up 10+ in 21 days)",
         lambda d: (d.rs_rank >= 70) & (d.rs_rank < 90) & (d.rs_change >= 10)),
        ("Long-standing leader (top 20% now and 21 days ago)", lambda d: (d.rs_rank >= 80) & (d.rs_rank_prev >= 80)),
    ]),
    ("52-week-high ideas", [
        ("Within 3% of 52-week high", lambda d: d.pct_from_52w_high <= 3),
        ("Within 5% of 52-week high and RS rank 70+", lambda d: (d.pct_from_52w_high <= 5) & (d.rs_rank >= 70)),
        ("Fresh 52-week breakout (last 5 days)", lambda d: d.breakout_5d),
        ("Fresh breakout + volume surge (5-day volume 1.5x the 50-day)", lambda d: d.breakout_5d & (d.vol_ratio >= 1.5)),
        ("Fresh breakout + RS rank 70+", lambda d: d.breakout_5d & (d.rs_rank >= 70)),
    ]),
    ("Refinements of leaders (top 20% by RS rank)", [
        ("Leader, not yet Extended", lambda d: (d.rs_rank >= 80) & (d.state != "Extended")),
        ("Leader and Extended", lambda d: (d.rs_rank >= 80) & (d.state == "Extended")),
        ("Leader near SMA50 (-3% to +5%)", lambda d: (d.rs_rank >= 80) & (d.pct_above_sma50 >= -3) & (d.pct_above_sma50 <= 5)),
        ("Leader with smooth trend (trend quality > 30)", lambda d: (d.rs_rank >= 80) & (d.trend_quality > 30)),
        ("Leader with a volume surge", lambda d: (d.rs_rank >= 80) & (d.vol_ratio >= 1.5)),
        ("Leader, beating Nifty, 6M return > 0", lambda d: (d.rs_rank >= 80) & (d.vs_nifty > 0) & (d.ret_6m > 0)),
    ]),
]
ALL_NAMES = [name for _, rules in GROUPS for name, _ in rules]

INTRO = """
<p style="font-size:13px;"><b>What this tests:</b> whether relative strength can find winners early. Compare the quintile rows first: if the edge
rises step by step from the weakest to the strongest bucket, relative strength is a real signal. A single lucky bucket is not.
Then look at the early-leader and 52-week-high rows, and at whether the edge survives in the second half and in down markets.</p>
<p style="color:#a00;font-size:12px;"><b>Multiple testing:</b> about 20 rules are tested here. By chance alone, one or two will show a t-score near 2.
Treat |t| below about 3 as suggestive, not proven.</p>
"""


def main():
    tickers = m.load_tickers(m.STOCKS_FILE)
    if b2.MAX_TICKERS:
        tickers = tickers[:b2.MAX_TICKERS]
    print(f"Loaded {len(tickers)} tickers from {m.STOCKS_FILE}")

    m.LOOKBACK_PERIOD = b2.BT_PERIOD
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
    obs = b2.replay(data, nifty_close, dates)
    obs = b2.add_forward(obs, data, nifty_close)
    print(f"{len(obs):,} observations")

    main_df, half_df, regime_df, groups_idx, split = b2.build_report(
        obs, dates, len(data), groups=GROUPS, core=ALL_NAMES)
    pd.set_option("display.width", 250)
    print("\n=== Edge, risk and consistency ===\n", main_df.round(1).to_string())
    print("\n=== Halves ===\n", half_df.round(1).to_string())
    print("\n=== Regimes ===\n", regime_df.round(1).to_string())

    html = b2.render_html(main_df, half_df, regime_df, groups_idx, split, len(data), len(dates), span,
                          title="Momentum Backtest #3: can relative strength find winners early?", intro_html=INTRO)
    m.send_email(f"Momentum Backtest #3 — {pd.Timestamp.now().strftime('%d %b %Y')}", html)


if __name__ == "__main__":
    main()
