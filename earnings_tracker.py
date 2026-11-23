"""
Earnings tracker: what are the latest results, and has the market priced them in yet?

For every stock in watchlist.txt it reads the latest quarterly results from Yahoo Finance and shows:
  1. RESULTS DUE in the next 14 days (so you are never surprised), with the recent run-up into results.
  2. STRONG RESULTS, MUTED REACTION: good numbers where the price has not reacted much and the trend is intact.
     (Research on post-earnings drift suggests prices can keep adjusting for weeks after good results. This is a
     candidate list for research, not a buy signal.)
  3. STRONG / GOOD RESULTS: everything that reported well, with the price reaction so far.
  4. WEAK RESULTS: profit or sales fell, to avoid or review if you own them.
  5. PENDING: stocks that have not yet reported the latest quarter.

Result quality is judged from ACTUAL growth against the same quarter last year (reliable), not against analyst estimates
(Yahoo has these for few Indian stocks, so beat/miss versus consensus is not available for most).

Limits: yearly comparison needs 5 quarters of data and Yahoo sometimes gives only 4; results dates come from Yahoo's
earnings calendar, which is missing for some stocks (those show reaction as n/a). Banks and lenders have no operating
margin, so the margin test is skipped for them.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

import momentum_tracker as m

# ============================================================
# CONFIGURATION
# ============================================================
STOCKS_FILE = m.STOCKS_FILE
MAX_WORKERS = 6
MAX_TICKERS = int(os.environ.get("ET_MAX_TICKERS", "0"))   # testing only; 0 = all

DUE_DAYS = 14                    # list results due within this many days
YOY_TOLERANCE_DAYS = 25          # year-ago quarter must be within this many days of 365
CYCLE_GRACE_DAYS = 20            # a quarter ending within this of the latest quarter-end counts as "reported"

# Result quality (growth versus the same quarter last year)
STRONG_MIN_REV_YOY = 10.0
STRONG_MIN_NI_YOY = 15.0
STRONG_MIN_MARGIN_CHG_BPS = -100.0
GOOD_MIN_REV_YOY = 5.0
GOOD_MIN_NI_YOY = 5.0

# Price reaction
REACTION_DAYS_AFTER = 1          # reaction = close one trading day after the result date vs the close before it
MUTED_MAX_DRIFT_PCT = 5.0        # "muted" = rose less than this since the result
SOLD_ON_NEWS_PCT = -3.0          # fell more than this after strong results
RUNUP_DAYS = 20
RUNUP_PCT = 10.0                 # "already ran" = up this much in the 20 trading days before results
MIN_ROWS = 210                   # price rows needed for SMA200


# ============================================================
# QUARTERLY DATA
# ============================================================
def qseries(df, names):
    """First matching row of a quarterly statement as a numeric Series indexed by quarter-end, oldest first."""
    if df is None or getattr(df, "empty", True):
        return pd.Series(dtype=float)
    labels = {str(i).lower().replace(" ", ""): i for i in df.index}
    for n in names:
        key = n.lower().replace(" ", "")
        if key in labels:
            s = pd.to_numeric(df.loc[labels[key]], errors="coerce").dropna().sort_index()
            if len(s):
                idx = pd.to_datetime(s.index)
                if getattr(idx, "tz", None) is not None:
                    idx = idx.tz_localize(None)
                s.index = idx
                return s[~s.index.duplicated()]
    return pd.Series(dtype=float)


def year_ago(s, d):
    """Value of series s about a year before date d, or NaN."""
    if s.empty:
        return np.nan
    target = d - pd.Timedelta(days=365)
    gaps = np.abs((s.index - target).values.astype("timedelta64[D]").astype(float))
    i = int(np.argmin(gaps))
    return float(s.iloc[i]) if gaps[i] <= YOY_TOLERANCE_DAYS else np.nan


def pct_change(cur, prev):
    if np.isnan(cur) or np.isnan(prev) or prev <= 0:
        return np.nan
    return (cur / prev - 1) * 100


def fetch_results(t):
    """Latest-quarter numbers and results dates for one ticker."""
    for attempt in range(2):
        try:
            tk = yf.Ticker(t)
            q = None
            for attr in ("quarterly_income_stmt", "quarterly_financials"):
                q = getattr(tk, attr, None)
                if q is not None and not q.empty:
                    break
            rev = qseries(q, ["Total Revenue", "Operating Revenue"])
            ni = qseries(q, ["Net Income", "Net Income Common Stockholders"])
            opi = qseries(q, ["Operating Income", "EBIT"])
            if ni.empty:
                if attempt == 0:
                    time.sleep(1.0)
                    continue
                return {"ticker": t, "has_data": False}

            d0 = ni.index[-1]
            r = {"ticker": t, "has_data": True, "q_end": d0}
            r["rev"] = float(rev.get(d0, np.nan)) if len(rev) else np.nan
            r["ni"] = float(ni.iloc[-1])
            ni_prev = year_ago(ni, d0)
            r["rev_yoy"] = pct_change(r["rev"], year_ago(rev, d0))
            r["turned_profit"] = bool(r["ni"] > 0 and not np.isnan(ni_prev) and ni_prev <= 0)
            r["ni_yoy"] = pct_change(r["ni"], ni_prev)
            r["has_yoy"] = not (np.isnan(r["rev_yoy"]) and np.isnan(r["ni_yoy"]) and not r["turned_profit"])

            opm_now = float(opi.get(d0, np.nan)) / r["rev"] * 100 if len(opi) and r["rev"] and not np.isnan(r["rev"]) else np.nan
            rev_prev, opi_prev = year_ago(rev, d0), year_ago(opi, d0)
            opm_prev = opi_prev / rev_prev * 100 if not np.isnan(rev_prev) and rev_prev > 0 and not np.isnan(opi_prev) else np.nan
            r["margin_chg_bps"] = (opm_now - opm_prev) * 100 if not np.isnan(opm_now) and not np.isnan(opm_prev) else np.nan

            r["result_date"], r["next_date"] = pd.NaT, pd.NaT
            try:
                ed = tk.earnings_dates
                if ed is not None and len(ed):
                    idx = pd.to_datetime(ed.index)
                    idx = idx.tz_localize(None) if getattr(idx, "tz", None) is not None else idx
                    now = pd.Timestamp.now()
                    past = [d for d in idx if d <= now and d > d0]
                    fut = [d for d in idx if d > now]
                    if past:
                        r["result_date"] = max(past).normalize()
                    if fut:
                        r["next_date"] = min(fut).normalize()
            except Exception:
                pass
            return r
        except Exception as e:
            if attempt == 1:
                print(f"Results fetch failed for {t}: {e}")
            time.sleep(1.0)
    return {"ticker": t, "has_data": False}


def fetch_all_results(tickers):
    out = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for i, r in enumerate(ex.map(fetch_results, tickers), 1):
            out.append(r)
            if i % 50 == 0:
                print(f"  ...{i}/{len(tickers)}")
    return out


# ============================================================
# CLASSIFICATION
# ============================================================
def classify(r):
    """Strong / Good / Mixed / Weak / n/a from growth versus the same quarter last year."""
    if not r.get("has_yoy"):
        return "n/a"
    rev, ni, mg = r["rev_yoy"], r["ni_yoy"], r["margin_chg_bps"]
    if r["turned_profit"] and (np.isnan(rev) or rev >= GOOD_MIN_REV_YOY):
        return "Strong"
    margin_ok = np.isnan(mg) or mg >= STRONG_MIN_MARGIN_CHG_BPS
    if (not np.isnan(rev) and rev >= STRONG_MIN_REV_YOY) and (not np.isnan(ni) and ni >= STRONG_MIN_NI_YOY) and margin_ok:
        return "Strong"
    if (not np.isnan(rev) and rev >= GOOD_MIN_REV_YOY) and (not np.isnan(ni) and ni >= GOOD_MIN_NI_YOY):
        return "Good"
    if (not np.isnan(ni) and ni < 0) or (not np.isnan(rev) and rev < 0) or r["ni"] <= 0:
        return "Weak"
    return "Mixed"


def latest_cycle_end(rows):
    """The most common latest quarter-end among stocks that have reported recently, as the 'current cycle'."""
    today = pd.Timestamp.now().normalize()
    ends = [r["q_end"] for r in rows if r.get("has_data")]
    recent = [d for d in ends if (today - d).days <= 130]
    return max(recent) if recent else None


def price_metrics(df, result_date):
    """Price context from daily data: trend, run-up before results, reaction and drift after results."""
    c = df["Close"].dropna()
    out = {"price": float(c.iloc[-1])}
    if len(c) >= MIN_ROWS:
        out["above_sma200"] = bool(c.iloc[-1] > c.rolling(200).mean().iloc[-1])
        out["above_sma50"] = bool(c.iloc[-1] > c.rolling(50).mean().iloc[-1])
    else:
        out["above_sma200"] = out["above_sma50"] = None
    out["ret_1m"] = float((c.iloc[-1] / c.iloc[-22] - 1) * 100) if len(c) > 22 else np.nan

    out["reaction"] = out["drift"] = out["runup"] = np.nan
    out["reaction_pending"] = False
    if pd.notna(result_date):
        idx = c.index.tz_localize(None) if getattr(c.index, "tz", None) is not None else c.index
        c = pd.Series(c.values, index=idx)
        before = c[c.index < result_date]
        after = c[c.index > result_date]
        if len(before):
            base = float(before.iloc[-1])
            out["drift"] = (float(c.iloc[-1]) / base - 1) * 100
            if len(after) >= REACTION_DAYS_AFTER:
                out["reaction"] = (float(after.iloc[REACTION_DAYS_AFTER - 1]) / base - 1) * 100
            else:
                out["reaction_pending"] = True
            if len(before) > RUNUP_DAYS:
                out["runup"] = (base / float(before.iloc[-RUNUP_DAYS - 1]) - 1) * 100
    return out


def build_universe(rows, data):
    cycle_end = latest_cycle_end(rows)
    out = []
    for r in rows:
        if not r.get("has_data"):
            continue
        r = dict(r)
        r["quality"] = classify(r)
        r["reported"] = cycle_end is not None and (cycle_end - r["q_end"]).days <= CYCLE_GRACE_DAYS
        df = data.get(r["ticker"])
        r.update(price_metrics(df, r["result_date"]) if df is not None else
                 {"price": np.nan, "above_sma200": None, "above_sma50": None, "ret_1m": np.nan,
                  "reaction": np.nan, "drift": np.nan, "runup": np.nan, "reaction_pending": False})
        r["flag"] = ""
        if r["reported"] and r["quality"] in ("Strong", "Good"):
            if r["above_sma200"] is not False and not np.isnan(r["drift"]) and r["drift"] < MUTED_MAX_DRIFT_PCT \
                    and r["drift"] > SOLD_ON_NEWS_PCT and not r["reaction_pending"]:
                r["flag"] = "Muted"
            elif not np.isnan(r["reaction"]) and r["reaction"] <= SOLD_ON_NEWS_PCT:
                r["flag"] = "Sold on good news"
            elif not np.isnan(r["runup"]) and r["runup"] >= RUNUP_PCT:
                r["flag"] = "Ran before results"
        out.append(r)
    return pd.DataFrame(out), cycle_end


# ============================================================
# EMAIL
# ============================================================
def f(x, nd=1, suffix="", plus=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:+.{nd}f}{suffix}" if plus else f"{x:.{nd}f}{suffix}"


def date_s(d):
    return "-" if pd.isna(d) else d.strftime("%d %b")


def tbl(headers, rows_):
    th = "".join(f'<th style="text-align:left;padding:3px 6px;border-bottom:1px solid #999;">{h}</th>' for h in headers)
    body = "".join("<tr>" + "".join(f'<td style="padding:3px 6px;border-bottom:1px solid #eee;">{c}</td>' for c in row) + "</tr>"
                   for row in rows_)
    return f'<table style="border-collapse:collapse;font-size:12px;"><tr>{th}</tr>{body}</table>'


def trend_s(r):
    if r["above_sma200"] is None:
        return "-"
    return "Above SMA200" if r["above_sma200"] else "Below SMA200"


def result_rows(sub):
    rows_ = []
    for _, r in sub.iterrows():
        rows_.append([r["ticker"].replace(".NS", ""), r["quality"], date_s(r["q_end"]),
                      f(r["rev_yoy"], 0, "%", True), "turned profit" if r["turned_profit"] else f(r["ni_yoy"], 0, "%", True),
                      f(r["margin_chg_bps"], 0, " bps", True), date_s(r["result_date"]),
                      "pending" if r["reaction_pending"] else f(r["reaction"], 1, "%", True),
                      f(r["drift"], 1, "%", True), f(r["runup"], 0, "%", True), trend_s(r), r["flag"] or "-"])
    return rows_


RESULT_HEAD = ["Ticker", "Result", "Quarter", "Sales YoY", "Profit YoY", "Margin chg", "Declared",
               "Reaction (1d)", "Since result", "Run-up before", "Trend", "Note"]


def build_email_html(df, cycle_end, n_requested, n_with_data, warns):
    rep = df[df["reported"]]
    pend = df[~df["reported"]]
    due = df[df["next_date"].notna()].copy()
    now = pd.Timestamp.now().normalize()
    due["days"] = (due["next_date"] - now).dt.days
    due = due[(due["days"] >= 0) & (due["days"] <= DUE_DAYS)].sort_values("days")

    muted = rep[rep["flag"] == "Muted"].sort_values("ni_yoy", ascending=False)
    good = rep[rep["quality"].isin(["Strong", "Good"])].sort_values(["quality", "ni_yoy"], ascending=[False, False])
    weak = rep[rep["quality"] == "Weak"].sort_values("ni_yoy")

    warn_html = m.warning_html(warns)
    h = f"""<html><body style="font-family:Arial,sans-serif;">
<h2>Earnings Tracker, {datetime.now().strftime('%d %b %Y')}</h2>
<p style="font-size:13px;">Latest quarter in this cycle ends {cycle_end.strftime('%d %b %Y') if cycle_end is not None else '-'}.
<b>{len(rep)}</b> stocks have reported it, <b>{len(pend)}</b> have not. Result quality is judged from growth against the same
quarter last year, not against analyst estimates.</p>{warn_html}
"""
    h += f"<h3>Results due in the next {DUE_DAYS} days ({len(due)})</h3>"
    if len(due):
        h += tbl(["Ticker", "Due", "In (days)", "1M return", "Trend", "Last quarter"],
                 [[r["ticker"].replace(".NS", ""), date_s(r["next_date"]), int(r["days"]), f(r["ret_1m"], 1, "%", True),
                   trend_s(r), r["quality"]] for _, r in due.iterrows()])
        h += '<p style="font-size:11px;color:#555;">A stock that has run up a lot before results needs a better-than-expected result to rise further.</p>'
    else:
        h += "<p>None found. Dates come from Yahoo's calendar, which is missing for some stocks.</p>"

    h += f"<h3>Strong or good results with a muted price reaction ({len(muted)})</h3>"
    h += ('<p style="font-size:11px;color:#555;">Good numbers, price up less than 5% since results, not down more than 3%, '
          "and not below the SMA200. A research list: check why the market is not reacting.</p>")
    h += tbl(RESULT_HEAD, result_rows(muted)) if len(muted) else "<p>None right now.</p>"

    h += f"<h3>All strong and good results ({len(good)})</h3>"
    h += tbl(RESULT_HEAD, result_rows(good)) if len(good) else "<p>None.</p>"

    h += f"<h3>Weak results ({len(weak)})</h3>"
    h += tbl(RESULT_HEAD, result_rows(weak)) if len(weak) else "<p>None.</p>"

    if len(pend):
        h += (f'<h3>Not yet reported ({len(pend)})</h3><p style="font-size:12px;">'
              + ", ".join(sorted(t.replace(".NS", "") for t in pend["ticker"])) + "</p>")

    h += f"""
<h3>How to read this</h3>
<table style="border-collapse:collapse;font-size:12px;">
<tr><td style="padding:3px 8px;"><b>Strong</b></td><td style="padding:3px 8px;">Sales up {STRONG_MIN_REV_YOY:.0f}%+ and profit up {STRONG_MIN_NI_YOY:.0f}%+ versus the same quarter last year, with operating margin not down more than {abs(STRONG_MIN_MARGIN_CHG_BPS):.0f} bps (or a company that turned profitable). Banks skip the margin test.</td></tr>
<tr><td style="padding:3px 8px;"><b>Good</b></td><td style="padding:3px 8px;">Sales up {GOOD_MIN_REV_YOY:.0f}%+ and profit up {GOOD_MIN_NI_YOY:.0f}%+.</td></tr>
<tr><td style="padding:3px 8px;"><b>Weak</b></td><td style="padding:3px 8px;">Profit or sales fell versus last year, or the company made a loss. Mixed results are not listed.</td></tr>
<tr><td style="padding:3px 8px;"><b>Reaction (1d)</b></td><td style="padding:3px 8px;">Price change from the close before the result date to the close one trading day after it. Results announced after market hours show up the next day, so one day catches both cases.</td></tr>
<tr><td style="padding:3px 8px;"><b>Note</b></td><td style="padding:3px 8px;"><b>Muted</b>: good results, little price move, trend intact. <b>Sold on good news</b>: fell 3%+ after good results. <b>Ran before results</b>: up 10%+ in the 20 days before the result.</td></tr>
</table>
<p style="font-size:12px;color:#a00;"><b>Limits:</b> a strong quarter does not mean a strong stock. Check the Quality tracker, and read the company's own results (one-off gains can inflate profit).
Yahoo's data can be late or wrong, and its results calendar is missing for some Indian stocks (reaction shows as "-"). Yearly comparisons need 5 quarters of history; stocks without them are left out.</p>
</body></html>"""
    return h


# ============================================================
# MAIN
# ============================================================
def main():
    tickers = m.load_tickers(STOCKS_FILE)
    if MAX_TICKERS:
        tickers = tickers[:MAX_TICKERS]
    print(f"Loaded {len(tickers)} tickers from {STOCKS_FILE}")
    if not tickers:
        print("Watchlist is empty. Exiting.")
        return

    data = m.fetch_data(tickers)
    print(f"Price data for {len(data)}/{len(tickers)} tickers")
    rows = fetch_all_results(tickers)
    n_with = sum(1 for r in rows if r.get("has_data"))
    print(f"Quarterly results for {n_with}/{len(tickers)} tickers")
    if n_with == 0:
        print("No results data, no email.")
        return

    df, cycle_end = build_universe(rows, data)
    df = df[df["has_yoy"]].copy() if len(df) else df
    if df.empty:
        print("No stock has enough quarters for a yearly comparison. Exiting.")
        return
    print(df["quality"].value_counts().to_string())

    warns = m.build_warnings(data, len(tickers)) if data else []
    if n_with < 0.7 * len(tickers):
        warns.append(f"Quarterly results loaded for only {n_with} of {len(tickers)} tickers. Yahoo coverage is thin for some Indian companies.")
    for w in warns:
        print("WARNING:", w)

    html = build_email_html(df, cycle_end, len(tickers), n_with, warns)
    flag = " [CHECK DATA]" if warns else ""
    m.send_email(f"Earnings Tracker, {datetime.now().strftime('%d %b %Y')}{flag}", html)


if __name__ == "__main__":
    main()
