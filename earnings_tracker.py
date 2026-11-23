"""
Earnings tracker: what are the latest results, and has the market priced them in yet?

For every stock in watchlist.txt it reads the latest quarterly results from Yahoo Finance and shows:
  1. RESULTS DUE in the next 14 days (so you are never surprised), with the recent run-up into results.
  2. STRONG RESULTS, MUTED REACTION: good numbers where the price has not reacted much and the trend is intact.
     (Research on post-earnings drift suggests prices can keep adjusting for weeks after good results. This is a
     candidate list for research, not a buy signal.)
  3. STRONG / GOOD RESULTS: everything that reported well, with the price reaction so far.
  4. WEAK RESULTS: operating profit fell (or a loss), to avoid or review if you own them.
  5. MIXED RESULTS: e.g. profit up but sales down.
  6. PENDING and NOT ANALYSED: stocks that have not reported, and stocks that could not be analysed (and why).

Profit is judged on operating numbers (EBITDA and EBIT; pre-tax profit for banks and lenders), because net profit is
distorted by the tax rate and one-off items. Net profit and book value growth are shown alongside. Banks and lenders are
also judged on book value growth, a better yardstick for them than sales.

Result quality is judged from ACTUAL growth against the same quarter last year (reliable), not against analyst estimates
(Yahoo has these for few Indian stocks, so beat/miss versus consensus is not available for most).

Limits: yearly comparison needs 5 quarters of data and Yahoo sometimes gives only 4 (those stocks get a sequential
quarter-on-quarter view instead, which is seasonal and not graded). Result dates come from Yahoo's earnings calendar; when
missing, the date is estimated from the volume spike after quarter-end and shown with "~". Banks and lenders have no
operating margin, so the margin test is skipped for them.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

import momentum_tracker as m
try:
    import screener_import as si
except ImportError:          # file not uploaded to the repo: run without the Screener.in fallback
    si = None

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
STRONG_MIN_OP_YOY = 15.0         # operating profit (EBIT, or pre-tax profit for lenders)
STRONG_MIN_BV_YOY = 10.0         # lenders: book value growth
STRONG_MIN_NI_YOY = 0.0          # net profit must at least not have fallen
STRONG_MIN_MARGIN_CHG_BPS = -100.0
GOOD_MIN_REV_YOY = 5.0
GOOD_MIN_OP_YOY = 5.0
GOOD_MIN_NI_YOY = 0.0
ONE_OFF_GAP_PTS = 40.0           # net profit growth differs from operating profit growth by this much: flag it
CRASH_NI_PCT = -70.0             # net profit down this much while operating profit is not: likely one-off

# Price reaction
REACTION_DAYS_AFTER = 1          # reaction = close one trading day after the result date vs the close before it
MUTED_MAX_DRIFT_PCT = 5.0        # "muted" = rose less than this since the result
SOLD_ON_NEWS_PCT = -3.0          # fell more than this after strong results
RUNUP_DAYS = 20
RUNUP_PCT = 10.0                 # "already ran" = up this much in the 20 trading days before results
MIN_ROWS = 210                   # price rows needed for SMA200 (trend shows "-" below this)
SCREENER_ALIGN_TOL = 0.25        # Yahoo and Screener quarterly sales must agree within this to be the same quarter
SCREENER_DIFF_PTS = 30.0         # Yahoo net profit growth differs from Screener's by this much: flag it
MIN_PRICE_ROWS = 40              # price rows needed at all (reaction and drift only)

# Estimating a missing result date from the volume spike
EST_WINDOW_START = 21            # days after quarter-end to start looking (few mid/small caps report earlier)
EST_WINDOW_END = 46              # SEBI deadline is 45 days after a quarter-end (60 days after March: see estimate_result_date)
BV_CHECK_PCT = 40.0              # book value change beyond this is probably a split/bonus/demerger, not real
LAGGING_DRIFT_PCT = -3.0         # good results, trend intact, but price is down this much since the result
EPS_TOLERANCE_DAYS = 30          # year-ago EPS announcement must be within this of 365 days earlier
EPS_STRONG_PCT = 15.0
EPS_DOWN_PCT = 0.0
EST_MIN_VOL_RATIO = 2.0          # spike day volume vs median of the prior 20 days


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


def eps_growth(ed, q_end):
    """Year-on-year growth in reported EPS from Yahoo's earnings history (fallback when the income statement is short).

    The EPS announced after q_end is compared with the EPS announced about a year earlier."""
    col = next((c for c in ed.columns if str(c).lower().replace(" ", "") == "reportedeps"), None)
    if col is None:
        return np.nan
    e = pd.to_numeric(ed[col], errors="coerce").dropna()
    if e.empty:
        return np.nan
    idx = pd.to_datetime(e.index)
    idx = idx.tz_localize(None) if getattr(idx, "tz", None) is not None else idx
    e = pd.Series(e.values, index=idx).sort_index()
    now = pd.Timestamp.now()
    cur = e[(e.index > q_end) & (e.index <= now)]
    if cur.empty:
        return np.nan
    d_cur, v_cur = cur.index[-1], float(cur.iloc[-1])
    gaps = np.abs((e.index - (d_cur - pd.Timedelta(days=365))).days.values)
    i = int(np.argmin(gaps))
    if gaps[i] > EPS_TOLERANCE_DAYS:
        return np.nan
    return pct_change(v_cur, float(e.iloc[i]))


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
            ebit = qseries(q, ["EBIT", "Operating Income"])
            ebitda = qseries(q, ["EBITDA", "Normalized EBITDA"])
            if ebitda.empty and len(opi):
                da = qseries(q, ["Reconciled Depreciation", "Depreciation And Amortization In Income Statement",
                                 "Depreciation Amortization Depletion"])
                if len(da):
                    ebitda = (opi + da.reindex(opi.index)).dropna()
            pbt = qseries(q, ["Pretax Income"])
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
            r["ebitda_yoy"] = pct_change(float(ebitda.get(d0, np.nan)), year_ago(ebitda, d0)) if len(ebitda) else np.nan
            r["ebit_yoy"] = pct_change(float(ebit.get(d0, np.nan)), year_ago(ebit, d0)) if len(ebit) else np.nan
            r["pbt_yoy"] = pct_change(float(pbt.get(d0, np.nan)), year_ago(pbt, d0)) if len(pbt) else np.nan
            # Operating profit growth used for grading: EBIT, else pre-tax profit (banks and lenders)
            r["is_lender"] = bool(len(opi) == 0)
            r["op_yoy"] = r["ebit_yoy"] if not np.isnan(r["ebit_yoy"]) else r["pbt_yoy"]
            # Sequential view, used when a year-ago quarter is not available
            def qoq(series):
                if len(series) < 2 or series.index[-1] != d0 or series.iloc[-2] <= 0:
                    return np.nan
                return (float(series.iloc[-1]) / float(series.iloc[-2]) - 1) * 100
            r["rev_qoq"], r["ni_qoq"] = qoq(rev), qoq(ni)
            r["op_qoq"] = qoq(ebit) if len(ebit) else qoq(pbt)
            r["bv_yoy"] = np.nan
            try:
                bs = getattr(tk, "quarterly_balance_sheet", None)
                if bs is None or bs.empty:
                    bs = getattr(tk, "balance_sheet", None)
                eq = qseries(bs, ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"])
                sh = qseries(bs, ["Ordinary Shares Number", "Share Issued"])
                if len(eq):
                    bd = eq.index[-1]
                    if len(sh) and bd in sh.index and not np.isnan(year_ago(sh, bd)) and year_ago(sh, bd) > 0 and sh[bd] > 0:
                        bv_now = eq.iloc[-1] / sh[bd]
                        bv_prev = year_ago(eq, bd) / year_ago(sh, bd)
                    else:
                        bv_now, bv_prev = float(eq.iloc[-1]), year_ago(eq, bd)
                    r["bv_yoy"] = pct_change(float(bv_now), float(bv_prev))
            except Exception:
                pass

            opm_now = float(opi.get(d0, np.nan)) / r["rev"] * 100 if len(opi) and r["rev"] and not np.isnan(r["rev"]) else np.nan
            rev_prev, opi_prev = year_ago(rev, d0), year_ago(opi, d0)
            opm_prev = opi_prev / rev_prev * 100 if not np.isnan(rev_prev) and rev_prev > 0 and not np.isnan(opi_prev) else np.nan
            r["margin_chg_bps"] = (opm_now - opm_prev) * 100 if not np.isnan(opm_now) and not np.isnan(opm_prev) else np.nan

            r["result_date"], r["next_date"] = pd.NaT, pd.NaT
            r["eps_yoy"] = np.nan
            try:
                ed = tk.earnings_dates
                if ed is not None and len(ed):
                    r["eps_yoy"] = eps_growth(ed, d0)
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
    """Strong / Good / Mixed / Weak / n/a from growth versus the same quarter last year.

    Judged on operating profit (EBIT; pre-tax profit for lenders) so that tax and one-off items do not decide the grade.
    Net profit must not have fallen for Strong or Good. Lenders must also show book value growth for Strong.
    """
    if not r.get("has_yoy"):
        return "n/a"
    rev, ni, op, bv, mg = r["rev_yoy"], r["ni_yoy"], r.get("op_yoy", np.nan), r.get("bv_yoy", np.nan), r["margin_chg_bps"]
    core = op if not np.isnan(op) else ni            # fall back to net profit when no operating figure exists
    if np.isnan(core) and not r["turned_profit"]:
        return "No profit data"                      # sales only: cannot tell if profit rose
    ni_ok = np.isnan(ni) or ni >= STRONG_MIN_NI_YOY
    if r["turned_profit"] and (np.isnan(rev) or rev >= GOOD_MIN_REV_YOY):
        return "Strong"
    margin_ok = np.isnan(mg) or mg >= STRONG_MIN_MARGIN_CHG_BPS
    bv_ok = (not r.get("is_lender")) or np.isnan(bv) or bv >= STRONG_MIN_BV_YOY
    if (not np.isnan(rev) and rev >= STRONG_MIN_REV_YOY) and (not np.isnan(core) and core >= STRONG_MIN_OP_YOY) \
            and ni_ok and margin_ok and bv_ok:
        return "Strong"
    if (not np.isnan(rev) and rev >= GOOD_MIN_REV_YOY) and (not np.isnan(core) and core >= GOOD_MIN_OP_YOY) and ni_ok:
        return "Good"
    if (not np.isnan(core) and core < 0) or r["ni"] <= 0:
        return "Weak"
    return "Mixed"      # includes profit up with sales down, and results that are only modestly better


def profit_note(r):
    """Flag when net profit moved very differently from operating profit (tax rate or one-off items)."""
    ni, op = r["ni_yoy"], r.get("op_yoy", np.nan)
    if np.isnan(ni) or np.isnan(op):
        return ""
    if ni <= CRASH_NI_PCT and op > CRASH_NI_PCT / 2:
        return "Net profit crash not in operations (one-off?)"
    if ni - op >= ONE_OFF_GAP_PTS:
        return "Net profit well above operating (tax/other income?)"
    if op - ni >= ONE_OFF_GAP_PTS:
        return "Net profit well below operating (tax/one-off?)"
    return ""


def estimate_result_date(df, q_end):
    """Best guess of the result date from the biggest volume spike between quarter-end + 15 and + 80 days."""
    try:
        if "Volume" not in df.columns:
            return pd.NaT
        v = df["Volume"].astype(float)
        idx = v.index.tz_localize(None) if getattr(v.index, "tz", None) is not None else v.index
        v = pd.Series(v.values, index=idx)
        base = v.rolling(20).median().shift(1)
        ratio = (v / base).replace([np.inf, -np.inf], np.nan)
        end_days = EST_WINDOW_END + 15 if q_end.month == 3 else EST_WINDOW_END      # March results are due in 60 days
        win = ratio[(ratio.index >= q_end + pd.Timedelta(days=EST_WINDOW_START)) &
                    (ratio.index <= q_end + pd.Timedelta(days=end_days))].dropna()
        if win.empty or win.max() < EST_MIN_VOL_RATIO:
            return pd.NaT
        spike = win.idxmax()
        prior = v.index[v.index < spike]
        return prior[-1].normalize() if len(prior) else pd.NaT     # result date = trading day before the spike
    except Exception:
        return pd.NaT


def apply_screener(rows, sc):
    """Use Screener.in's year-on-year quarterly growth where Yahoo has none, and cross-check Yahoo where it does.

    Returns (rows, n_filled, n_flagged). Screener's quarter is accepted when its quarterly sales agree with Yahoo's latest
    quarter (within 25%); lenders skip this check because their 'sales' are defined differently."""
    n_fill = n_flag = 0
    if sc is None or sc.empty:
        return rows, 0, 0
    for r in rows:
        r["src_note"] = ""
        if not r.get("has_data") or r["ticker"] not in sc.index:
            continue
        x = sc.loc[r["ticker"]]
        ys, ss = r.get("rev", np.nan), x.get("sales_cr", np.nan)
        if not r.get("is_lender") and not np.isnan(ys) and not np.isnan(ss) and ss > 0:
            if abs(ys / 1e7 / ss - 1) > SCREENER_ALIGN_TOL:
                r["src_note"] = "Screener quarter does not match Yahoo's: ignored"
                continue
        yahoo_core = r.get("op_yoy", np.nan)
        yahoo_core = r["ni_yoy"] if np.isnan(yahoo_core) else yahoo_core
        if not r.get("has_yoy") or np.isnan(yahoo_core):
            ni_s, rev_s = x.get("ni_yoy", np.nan), x.get("rev_yoy", np.nan)
            if np.isnan(ni_s) and np.isnan(rev_s):
                continue
            if np.isnan(r["rev_yoy"]):
                r["rev_yoy"] = rev_s
            if np.isnan(r["ni_yoy"]):
                r["ni_yoy"] = ni_s
            if np.isnan(r.get("op_yoy", np.nan)) and not np.isnan(x.get("op_yoy", np.nan)):
                r["op_yoy"] = x["op_yoy"]
            r["has_yoy"] = True
            r["src_note"] = "Growth from Screener.in (Yahoo has no year-ago quarter); net profit basis"
            n_fill += 1
        else:
            ni_s = x.get("ni_yoy", np.nan)
            if not np.isnan(ni_s) and not np.isnan(r["ni_yoy"]) and abs(ni_s - r["ni_yoy"]) >= SCREENER_DIFF_PTS:
                r["src_note"] = f"Screener net profit {ni_s:+.0f}% vs Yahoo {r['ni_yoy']:+.0f}%: check"
                n_flag += 1
    return rows, n_fill, n_flag


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
        r["note_profit"] = profit_note(r) if r["quality"] != "n/a" else ""
        df = data.get(r["ticker"])
        r["date_est"] = False
        if df is not None and pd.isna(r["result_date"]) and r["reported"]:
            est = estimate_result_date(df, r["q_end"])
            if pd.notna(est):
                r["result_date"], r["date_est"] = est, True
        r.update(price_metrics(df, r["result_date"]) if df is not None else
                 {"price": np.nan, "above_sma200": None, "above_sma50": None, "ret_1m": np.nan,
                  "reaction": np.nan, "drift": np.nan, "runup": np.nan, "reaction_pending": False})
        r["flag"] = ""
        if r["reported"] and r["quality"] in ("Strong", "Good"):
            if not np.isnan(r["reaction"]) and r["reaction"] <= SOLD_ON_NEWS_PCT:
                r["flag"] = "Sold on good news"
            elif r["above_sma200"] is True and not np.isnan(r["drift"]) and r["drift"] <= LAGGING_DRIFT_PCT:
                r["flag"] = "Lagging"
            elif r["above_sma200"] is not False and not np.isnan(r["drift"]) and r["drift"] < MUTED_MAX_DRIFT_PCT \
                    and r["drift"] > SOLD_ON_NEWS_PCT and not r["reaction_pending"]:
                r["flag"] = "Muted"
            elif not np.isnan(r["runup"]) and r["runup"] >= RUNUP_PCT:
                r["flag"] = "Ran before results"
        r["data_note"] = "; ".join(x for x in [
            "No price data" if np.isnan(r["price"]) else "",
            "Book value change over 40%: check (corporate action, equity raise or data issue)" if not np.isnan(r["bv_yoy"]) and abs(r["bv_yoy"]) > BV_CHECK_PCT else "",
        ] if x)
        out.append(r)
    return pd.DataFrame(out), cycle_end


# ============================================================
# EMAIL
# ============================================================
def f(x, nd=1, suffix="", plus=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:+.{nd}f}{suffix}" if plus else f"{x:.{nd}f}{suffix}"


def date_s(d, est=False):
    return "-" if pd.isna(d) else ("~" if est else "") + d.strftime("%d %b")


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
        note = "; ".join(x for x in [r["flag"], r["note_profit"], r["data_note"], r.get("src_note", "")] if x) or "-"
        rows_.append([r["ticker"].replace(".NS", ""), r["quality"], date_s(r["q_end"]),
                      f(r["rev_yoy"], 0, "%", True), f(r["ebitda_yoy"], 0, "%", True), f(r["ebit_yoy"], 0, "%", True),
                      f(r["pbt_yoy"], 0, "%", True),
                      "turned profit" if r["turned_profit"] else f(r["ni_yoy"], 0, "%", True),
                      f(r["bv_yoy"], 0, "%", True),
                      f(r["margin_chg_bps"], 0, " bps", True), date_s(r["result_date"], r["date_est"]),
                      "pending" if r["reaction_pending"] else f(r["reaction"], 1, "%", True),
                      f(r["drift"], 1, "%", True), f(r["runup"], 0, "%", True), trend_s(r), note])
    return rows_


RESULT_HEAD = ["Ticker", "Result", "Quarter", "Sales YoY", "EBITDA YoY", "EBIT YoY", "Pre-tax YoY", "Net profit YoY",
               "Book value YoY", "EBIT margin chg", "Declared", "Reaction (1d)", "Since result", "Run-up before", "Trend", "Note"]


def build_email_html(df, cycle_end, n_requested, n_with_data, warns, not_analysed=None, limited=None):
    not_analysed = not_analysed or {}
    limited = limited if limited is not None else pd.DataFrame()
    rep = df[df["reported"]]
    pend = df[~df["reported"]]
    due = df[df["next_date"].notna()].copy()
    now = pd.Timestamp.now().normalize()
    due["days"] = (due["next_date"] - now).dt.days
    due = due[(due["days"] >= 0) & (due["days"] <= DUE_DAYS)].sort_values("days")

    muted = rep[rep["flag"] == "Muted"].sort_values("ni_yoy", ascending=False)
    lagging = rep[rep["flag"] == "Lagging"].sort_values("drift")
    good = rep[rep["quality"].isin(["Strong", "Good"])].sort_values(["quality", "ni_yoy"], ascending=[False, False])
    weak = rep[rep["quality"] == "Weak"].sort_values("op_yoy")
    mixed = rep[rep["quality"] == "Mixed"].sort_values("op_yoy", ascending=False)

    warn_html = m.warning_html(warns)
    h = f"""<html><body style="font-family:Arial,sans-serif;">
<h2>Earnings Tracker, {datetime.now().strftime('%d %b %Y')}</h2>
<p style="font-size:13px;">Latest quarter in this cycle ends {cycle_end.strftime('%d %b %Y') if cycle_end is not None else '-'}.
<b>{len(rep)}</b> stocks have reported it, <b>{len(pend)}</b> have not, and <b>{sum(len(v) for v in not_analysed.values()) + len(limited)}</b>
could not be graded (listed at the bottom, with the reason). Result quality is judged from growth in operating profit against the
same quarter last year, not against analyst estimates.</p>{warn_html}
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

    h += f"<h3>Strong or good results, price lagging, trend intact ({len(lagging)})</h3>"
    h += ('<p style="font-size:11px;color:#555;">Good numbers, price down more than 3% since results, but still above the SMA200. '
          "Either the market sees something the numbers hide, or it is an opportunity. Read the company's commentary.</p>")
    h += tbl(RESULT_HEAD, result_rows(lagging)) if len(lagging) else "<p>None right now.</p>"

    h += f"<h3>All strong and good results ({len(good)})</h3>"
    h += tbl(RESULT_HEAD, result_rows(good)) if len(good) else "<p>None.</p>"

    h += f"<h3>Weak results ({len(weak)})</h3>"
    h += tbl(RESULT_HEAD, result_rows(weak)) if len(weak) else "<p>None.</p>"

    h += f"<h3>Mixed results ({len(mixed)})</h3>"
    h += ('<p style="font-size:11px;color:#555;">Neither strong nor weak: for example profit up but sales down, or only modest growth. '
          "Worth a look at the company's own commentary.</p>")
    h += tbl(RESULT_HEAD, result_rows(mixed)) if len(mixed) else "<p>None.</p>"

    if len(pend):
        h += (f'<h3>Not yet reported ({len(pend)})</h3><p style="font-size:12px;">'
              + ", ".join(sorted(t.replace(".NS", "") for t in pend["ticker"])) + "</p>")

    if len(limited):
        h += f"<h3>Limited history: EPS and sequential view only ({len(limited)})</h3>"
        h += ('<p style="font-size:11px;color:#555;">Yahoo has no year-ago quarter in its income statement for these. <b>EPS YoY</b> comes from its earnings history '
              "(blank where unavailable) and is the better guide, but Yahoo's EPS can mix standalone and consolidated figures or include one-offs, so check extreme values. Sorted by EPS YoY. Quarter-on-quarter figures are seasonal and are not graded: treat them as pointers only.</p>")
        h += tbl(["Ticker", "Quarter", "Sales YoY", "EPS YoY", "Sales QoQ", "Operating profit QoQ", "Net profit QoQ", "Declared",
                  "Reaction (1d)", "Since result", "Trend", "Note"],
                 [[r["ticker"].replace(".NS", ""), date_s(r["q_end"]), f(r["rev_yoy"], 0, "%", True), f(r["eps_yoy"], 0, "%", True), f(r["rev_qoq"], 0, "%", True),
                   f(r["op_qoq"], 0, "%", True), f(r["ni_qoq"], 0, "%", True), date_s(r["result_date"], r["date_est"]),
                   "pending" if r["reaction_pending"] else f(r["reaction"], 1, "%", True), f(r["drift"], 1, "%", True),
                   trend_s(r), "; ".join(x for x in [
                       "EPS up, trend intact" if not np.isnan(r["eps_yoy"]) and r["eps_yoy"] >= EPS_STRONG_PCT and r["above_sma200"] is True else "",
                       r["data_note"]] if x) or "-"]
                  for _, r in limited.sort_values("eps_yoy", ascending=False, na_position="last").iterrows()])
    for reason, names in not_analysed.items():
        if names:
            h += (f'<h3>Not analysed: {reason} ({len(names)})</h3><p style="font-size:12px;">'
                  + ", ".join(sorted(t.replace(".NS", "") for t in names)) + "</p>")

    h += f"""
<h3>How to read this</h3>
<table style="border-collapse:collapse;font-size:12px;">
<tr><td style="padding:3px 8px;"><b>Operating profit</b></td><td style="padding:3px 8px;">EBIT growth (pre-tax profit for banks and lenders). EBITDA, pre-tax and net profit are shown beside it. Net profit is affected by the tax rate and one-offs, so it is not used to grade, except that it must not have fallen for Strong or Good.</td></tr>
<tr><td style="padding:3px 8px;"><b>Strong</b></td><td style="padding:3px 8px;">Sales up {STRONG_MIN_REV_YOY:.0f}%+ and operating profit up {STRONG_MIN_OP_YOY:.0f}%+ versus the same quarter last year, net profit not down, and EBIT margin not down more than {abs(STRONG_MIN_MARGIN_CHG_BPS):.0f} bps (or a company that turned profitable). Banks and lenders skip the margin test and also need book value up {STRONG_MIN_BV_YOY:.0f}%+.</td></tr>
<tr><td style="padding:3px 8px;"><b>Good</b></td><td style="padding:3px 8px;">Sales up {GOOD_MIN_REV_YOY:.0f}%+ and operating profit up {GOOD_MIN_OP_YOY:.0f}%+, net profit not down.</td></tr>
<tr><td style="padding:3px 8px;"><b>Weak</b></td><td style="padding:3px 8px;">Operating profit fell versus last year, or the company made a loss.</td></tr>
<tr><td style="padding:3px 8px;"><b>Mixed</b></td><td style="padding:3px 8px;">Everything else, for example profit up while sales fell.</td></tr>
<tr><td style="padding:3px 8px;"><b>Source notes</b></td><td style="padding:3px 8px;"><b>Growth from Screener.in</b>: Yahoo had no year-ago quarter, so Screener's year-on-year sales and net profit growth were used (net profit basis, no operating profit). <b>Screener net profit X% vs Yahoo Y%: check</b>: the two sources disagree by 30+ points, so one has an error or a one-off; check the company's results.</td></tr>
<tr><td style="padding:3px 8px;"><b>Book value YoY</b></td><td style="padding:3px 8px;">Growth in book value per share (equity if share count is missing) over the year, from Yahoo's latest balance sheet, which can be older than the quarter. The key growth measure for banks and lenders.</td></tr>
<tr><td style="padding:3px 8px;"><b>Declared</b></td><td style="padding:3px 8px;">Result date from Yahoo. A date with "~" is estimated from the volume spike and may be a day off.</td></tr>
<tr><td style="padding:3px 8px;"><b>Reaction (1d)</b></td><td style="padding:3px 8px;">Price change from the close before the result date to the close one trading day after it. Results announced after market hours show up the next day, so one day catches both cases.</td></tr>
<tr><td style="padding:3px 8px;"><b>Note</b></td><td style="padding:3px 8px;"><b>Muted</b>: good results, little price move, trend intact. <b>Sold on good news</b>: fell 3%+ in the day after good results. <b>Sold on good news</b> (checked first) beats the others. <b>Lagging</b>: good results, down 3%+ since, still above the SMA200. <b>Ran before results</b>: up 10%+ in the 20 days before the result. Profit notes flag net profit moving very differently from operating profit (tax rate or one-off items): check the company's own results.</td></tr>
</table>
<p style="font-size:12px;color:#a00;"><b>Limits:</b> a strong quarter does not mean a strong stock. Check the Quality tracker, and read the company's own results (one-off gains can inflate profit).
Yahoo's data can be late or wrong, and its results calendar is missing for some Indian stocks (dates then estimated, marked "~"). EBITDA is missing for some companies. Yearly comparisons need 5 quarters of history; stocks without them appear in the sequential table or the not-analysed lists.</p>
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

    # Earnings only needs about 40 days of prices for the reaction; the SMA200 is skipped when history is short.
    data = m.fetch_data(tickers, min_rows=MIN_PRICE_ROWS)
    print(f"Price data for {len(data)}/{len(tickers)} tickers")
    rows = fetch_all_results(tickers)
    n_with = sum(1 for r in rows if r.get("has_data"))
    print(f"Quarterly results for {n_with}/{len(tickers)} tickers")
    if n_with == 0:
        print("No results data, no email.")
        return

    if si is None:
        sc, sc_info = pd.DataFrame(), "screener_import.py is not in the repo"
    else:
        sc, sc_info = si.load_screener()
    print("Screener.in export:", sc_info)
    rows, n_fill, n_flag = apply_screener(rows, sc)
    if len(sc):
        print(f"Screener.in: {n_fill} stocks graded from Screener data, {n_flag} Yahoo figures flagged as different")

    df, cycle_end = build_universe(rows, data)
    no_data = [r["ticker"] for r in rows if not r.get("has_data")]
    full = df
    graded_q = ~full["quality"].isin(["n/a", "No profit data"]) if len(full) else None
    df = full[graded_q].copy() if len(full) else full
    sales_only = full[full["quality"] == "No profit data"] if len(full) else full
    # Everything reported but not gradable on a yearly basis gets the EPS / sequential view
    limited = full[~graded_q & full["reported"] & ((full["quality"] == "n/a") | (full["quality"] == "No profit data"))].copy() if len(full) else full
    limited = limited[limited["rev_qoq"].notna() | limited["ni_qoq"].notna() | limited["eps_yoy"].notna()] if len(limited) else limited
    lim_names = set(limited["ticker"]) if len(limited) else set()
    not_analysed = {
        "no quarterly data from Yahoo": no_data,
        "too little history for any comparison": [t for t in (full[~graded_q]["ticker"] if len(full) else []) if t not in lim_names],
    }
    if df.empty:
        print("No stock has enough quarters for a yearly comparison. Exiting.")
        return
    print(df["quality"].value_counts().to_string())
    print(f"Coverage: graded {len(df)}, sequential only {len(lim_names)}, no data {len(no_data)}, of {len(tickers)}")

    warns = m.build_warnings(data, len(tickers)) if data else []
    if n_with < 0.7 * len(tickers):
        warns.append(f"Quarterly results loaded for only {n_with} of {len(tickers)} tickers. Yahoo coverage is thin for some Indian companies.")
    for w in warns:
        print("WARNING:", w)

    html = build_email_html(df, cycle_end, len(tickers), n_with, warns, not_analysed, limited)
    src_line = (f'<p style="font-size:12px;">Screener.in export used: {sc_info}. {n_fill} stocks graded from it, {n_flag} Yahoo figures flagged as different.</p>'
                if len(sc) else f'<p style="font-size:12px;color:#a00;">Screener.in export not used: {sc_info}.</p>')
    html = html.replace("</body></html>", src_line + "</body></html>")
    flag = " [CHECK DATA]" if warns else ""
    m.send_email(f"Earnings Tracker, {datetime.now().strftime('%d %b %Y')}{flag}", html)


if __name__ == "__main__":
    main()
