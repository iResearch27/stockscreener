"""
Quality tracker: is this a fundamentally strong business?

Scores every stock in watchlist.txt from 0 to 100 on four pillars using annual financial statements from Yahoo Finance:
  Profitability   (ROCE, ROE)            - does the business earn good returns on the money invested in it?
  Growth          (sales, profit)        - is it growing, and consistently?
  Balance sheet   (debt, interest cover) - can it survive a bad year?
  Cash quality    (cash vs profit, FCF)  - are the profits real?
Banks and other financial companies are scored on their own pillars (ROE, ROA, growth), because debt, ROCE and cash flow
mean something different for a lender.

A grade (A/B/C/D) and red flags are shown. This is a QUALITY filter, not a buy signal: it tells you which stocks are worth
spending time on. Pair it with the Momentum and Earnings trackers for timing.

Fundamentals change only when companies report, so this runs weekly.

Limits (also in the email): Yahoo's coverage of Indian companies is patchy, especially small caps; promoter pledging and
holding trends are not available from this source; annual data usually has only 3 to 4 years. Stocks without enough data
are listed separately and not scored.
"""

import os
import time
import smtplib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import numpy as np
import pandas as pd
import yfinance as yf

# ============================================================
# CONFIGURATION
# ============================================================
STOCKS_FILE = "watchlist.txt"   # one shared list for all trackers
MAX_WORKERS = 6
MAX_TICKERS = int(os.environ.get("QT_MAX_TICKERS", "0"))   # testing only; 0 = all
MIN_COVERAGE = 0.6        # share of the score that must be computable, otherwise "insufficient data"
GRADE_APLUS, GRADE_A, GRADE_B, GRADE_C = 85, 70, 55, 40   # A+ also needs zero red flags
NI_CAGR_CAP = 30.0        # profit growth above this earns no extra points (a very low starting year inflates CAGR)
LOW_BASE_CAGR = 60.0      # profit CAGR above this is noted as "low base"
# Companies whose consolidated accounts include a lending/financing arm. Their debt and operating cash flow figures are
# distorted by the loan book, so debt/equity, interest cover and cash-flow measures are left out of their score.
# Add or remove tickers here as you learn more.
FINANCE_ARM_TICKERS = {"M&M.NS", "ASHOKLEY.NS", "TITAN.NS", "LT.NS"}
FINANCE_ARM_SKIP = {"de", "int_cover", "ocf_ni", "fcf_pos"}
LENDER_APLUS = 90         # lenders skip the debt and cash tests and bad-loan data is missing, so A+ needs a higher score
BANK_ROA_FULL = 2.0       # ROA that earns full marks: banks
NBFC_ROA_FULL = 3.0       # ...and non-bank lenders, which normally earn a higher ROA
# Tickers whose recent ratios are distorted by a merger (equity and sales jumped). Shown with a note. Edit as needed.
MERGER_DISTORTED = {"HDFCBANK.NS"}

# Red-flag thresholds
FLAG_ROCE_MIN = 10.0
FLAG_ROE_MIN = 8.0
FLAG_DE_MAX = 1.5
FLAG_INT_COVER_MIN = 2.5
FLAG_OCF_NI_MIN = 0.6

# Score weights: (metric, max points, value at 0 points, value at full points)
NONFIN_RULES = [
    ("roce", 15, 10.0, 25.0),
    ("roe", 15, 10.0, 22.0),
    ("rev_cagr", 10, 0.0, 20.0),
    ("ni_cagr", 10, 0.0, 20.0),
    ("consistency", 10, 0.0, 1.0),
    ("de", 10, 1.5, 0.3),            # lower is better: 0 points at 1.5, full at 0.3
    ("int_cover", 10, 2.0, 10.0),
    ("ocf_ni", 10, 0.5, 1.1),
    ("fcf_pos", 5, 0.0, 1.0),
    ("margin_trend", 5, -3.0, 1.0),
]
FIN_RULES = [
    ("roe", 30, 8.0, 20.0),
    ("roa", 20, 0.5, 2.0),           # full-marks ROA is set per stock in score_stock (bank vs NBFC)
    ("ni_cagr", 15, 0.0, 20.0),
    ("rev_cagr", 10, 0.0, 15.0),
    ("consistency", 10, 0.0, 1.0),
    ("eq_assets", 15, 6.0, 12.0),    # equity / total assets: a rough capital-strength measure for lenders
]


# ============================================================
# DATA HELPERS
# ============================================================
def load_tickers(path):
    """One ticker per line; blank lines and # comments ignored; duplicates removed (first one kept)."""
    seen, out = set(), []
    with open(path) as f:
        for line in f:
            t = line.strip().upper()
            if t and not t.startswith("#") and t not in seen:
                seen.add(t)
                out.append(t)
    return out


def _norm(s):
    return str(s).lower().replace(" ", "").replace("_", "")


def series(df, names):
    """First matching row of a statement as a numeric Series indexed by fiscal year, oldest first."""
    if df is None or getattr(df, "empty", True):
        return pd.Series(dtype=float)
    labels = {_norm(i): i for i in df.index}
    for n in names:
        if _norm(n) in labels:
            s = pd.to_numeric(df.loc[labels[_norm(n)]], errors="coerce").dropna()
            if s.empty:
                continue
            s = s.sort_index()
            try:
                s.index = [d.year for d in s.index]
            except AttributeError:
                continue
            return s[~s.index.duplicated()]
    return pd.Series(dtype=float)


def ratio(a, b):
    if a.empty or b.empty:
        return pd.Series(dtype=float)
    j = pd.concat([a, b], axis=1, join="inner").dropna()
    j = j[j.iloc[:, 1] != 0]
    return j.iloc[:, 0] / j.iloc[:, 1]


def cagr(s):
    if len(s) < 3:
        return np.nan
    years = s.index[-1] - s.index[0]
    if years <= 0 or s.iloc[0] <= 0 or s.iloc[-1] <= 0:
        return np.nan
    return ((s.iloc[-1] / s.iloc[0]) ** (1.0 / years) - 1) * 100


def last_mean(s, n=3):
    return float(s.tail(n).mean()) if len(s) else np.nan


def get_info(tk):
    try:
        return tk.info or {}
    except Exception:
        return {}


def get_stmt(tk, attr):
    try:
        return getattr(tk, attr)
    except Exception:
        return None


def compute_metrics(tk):
    """All raw metrics for one ticker. Missing values are NaN."""
    info = get_info(tk)
    fin = get_stmt(tk, "financials")
    bs = get_stmt(tk, "balance_sheet")
    cf = get_stmt(tk, "cashflow")

    ni = series(fin, ["Net Income", "Net Income Common Stockholders"])
    rev = series(fin, ["Total Revenue", "Operating Revenue"])
    ebit = series(fin, ["EBIT", "Operating Income"])
    opinc = series(fin, ["Operating Income", "EBIT"])
    intexp = series(fin, ["Interest Expense"]).abs()
    ta = series(bs, ["Total Assets"])
    cl = series(bs, ["Current Liabilities", "Total Current Liabilities"])
    eq = series(bs, ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"])
    debt = series(bs, ["Total Debt"])
    ocf = series(cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"])
    capex = series(cf, ["Capital Expenditure"]).abs()

    sector = info.get("sector") or ""
    industry = info.get("industry") or ""
    is_fin = sector == "Financial Services" and "insurance" not in industry.lower()

    m = {"sector": sector, "industry": industry, "is_fin": is_fin,
         "mcap_cr": (info.get("marketCap") or np.nan) / 1e7 if info.get("marketCap") else np.nan,
         "years": int(len(ni))}

    roe = ratio(ni, eq) * 100
    m["roe"] = last_mean(roe)
    if np.isnan(m["roe"]) and info.get("returnOnEquity") is not None:
        m["roe"] = float(info["returnOnEquity"]) * 100
    m["roa"] = last_mean(ratio(ni, ta) * 100)
    ea = ratio(eq, ta) * 100
    m["eq_assets"] = float(ea.iloc[-1]) if len(ea) else np.nan

    if len(cl) and len(ta):
        ce = pd.concat([ta, cl], axis=1, join="inner").dropna()
        capital_employed = ce.iloc[:, 0] - ce.iloc[:, 1]
    else:
        capital_employed = pd.Series(dtype=float)
    m["roce"] = last_mean(ratio(ebit, capital_employed) * 100)

    m["rev_cagr"] = cagr(rev)
    m["ni_cagr"] = cagr(ni)
    if len(ni) >= 3:
        d = ni.diff().dropna()
        m["consistency"] = float((d > 0).sum()) / len(d)
        m["ni_last_growth"] = float(ni.iloc[-1] / ni.iloc[-2] - 1) * 100 if ni.iloc[-2] > 0 else np.nan
    else:
        m["consistency"] = np.nan
        m["ni_last_growth"] = np.nan

    de = ratio(debt, eq)
    m["de"] = float(de.iloc[-1]) if len(de) else np.nan
    if np.isnan(m["de"]) and info.get("debtToEquity") is not None:
        m["de"] = float(info["debtToEquity"]) / 100.0

    ic = ratio(ebit, intexp)
    if len(ic):
        m["int_cover"] = min(float(ic.iloc[-1]), 20.0)
    elif len(ebit) and len(debt) and float(debt.iloc[-1]) <= 0:
        m["int_cover"] = 20.0      # no debt, nothing to cover
    else:
        m["int_cover"] = np.nan

    oc = ratio(ocf, ni)
    oc = oc[ni.reindex(oc.index) > 0] if len(oc) else oc     # ignore years with a loss
    m["ocf_ni"] = last_mean(oc)
    if len(ocf) and len(capex):
        j = pd.concat([ocf, capex], axis=1, join="inner").dropna()
        m["fcf_pos"] = 1.0 if len(j) and (j.iloc[-1, 0] - j.iloc[-1, 1]) > 0 else 0.0
    else:
        m["fcf_pos"] = np.nan

    om = ratio(opinc, rev) * 100
    m["margin_trend"] = float(om.iloc[-1] - om.iloc[0]) if len(om) >= 3 else np.nan
    return m


# ============================================================
# SCORING
# ============================================================
def lin(x, lo, hi, pts):
    """Points from 0 (at lo) to pts (at hi), linear and clipped. Works when lo > hi (lower is better)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    f = (x - lo) / (hi - lo)
    return pts * min(max(f, 0.0), 1.0)


def score_stock(m):
    rules = FIN_RULES if m["is_fin"] else NONFIN_RULES
    if is_finance_arm(m):
        rules = [r for r in rules if r[0] not in FINANCE_ARM_SKIP]
    total_max = sum(r[1] for r in rules)
    earned, avail = 0.0, 0.0
    for name, pts, lo, hi in rules:
        v = m.get(name)
        if name == "ni_cagr" and v is not None and not np.isnan(v):
            v = min(v, NI_CAGR_CAP)
        if name == "roa":
            hi = BANK_ROA_FULL if "bank" in str(m.get("industry", "")).lower() else NBFC_ROA_FULL
        p = lin(v, lo, hi, pts)
        if p is not None:
            earned += p
            avail += pts
    coverage = avail / total_max
    score = (earned / avail * 100) if avail else np.nan
    return score, coverage


def is_finance_arm(m):
    return m.get("ticker") in FINANCE_ARM_TICKERS


def is_insurer(m):
    return "insurance" in str(m.get("industry", "")).lower()


def grade(score, n_flags=0, m=None):
    if np.isnan(score):
        return "-"
    m = m or {}
    aplus_bar = LENDER_APLUS if m.get("is_fin") else GRADE_APLUS
    if score >= aplus_bar and n_flags == 0 and not is_insurer(m):
        return "A+"
    return "A" if score >= GRADE_A else "B" if score >= GRADE_B else "C" if score >= GRADE_C else "D"


def red_flags(m):
    f = []

    def below(k, lim):
        return not np.isnan(m.get(k, np.nan)) and m[k] < lim

    def above(k, lim):
        return not np.isnan(m.get(k, np.nan)) and m[k] > lim

    if below("roe", FLAG_ROE_MIN):
        f.append("Low ROE")
    if not m["is_fin"]:
        if below("roce", FLAG_ROCE_MIN):
            f.append("Low ROCE")
        if not is_finance_arm(m):
            if above("de", FLAG_DE_MAX):
                f.append("High debt")
            if below("int_cover", FLAG_INT_COVER_MIN):
                f.append("Thin interest cover")
            if below("ocf_ni", FLAG_OCF_NI_MIN):
                f.append("Cash lags profit")
            if m.get("fcf_pos") == 0.0:
                f.append("Negative FCF")
    if below("rev_cagr", 0.0):
        f.append("Sales shrinking")
    if below("ni_last_growth", 0.0):
        f.append("Profit fell last year")
    return f


def notes_for(m):
    """Neutral context shown beside the score (not red flags)."""
    n = []
    if not np.isnan(m.get("ni_cagr", np.nan)) and m["ni_cagr"] > LOW_BASE_CAGR:
        n.append("Low-base growth (score capped)")
    if is_finance_arm(m):
        n.append("Debt and cash flow left out: accounts include a finance arm")
    if is_insurer(m):
        n.append("Insurer: few metrics available, not eligible for A+")
    if m.get("ticker") in MERGER_DISTORTED:
        n.append("Merger-distorted ROE and sales growth: read with caution")
    return n


# ============================================================
# FETCH
# ============================================================
def fetch_one(t):
    for attempt in range(2):
        try:
            m = compute_metrics(yf.Ticker(t))
            if m["years"] == 0 and attempt == 0:
                time.sleep(1.0)
                continue
            m["ticker"] = t
            return m
        except Exception as e:
            if attempt == 1:
                print(f"Fundamentals failed for {t}: {e}")
            time.sleep(1.0)
    return None


def fetch_all(tickers):
    out = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for i, r in enumerate(ex.map(fetch_one, tickers), 1):
            if r is not None:
                out.append(r)
            if i % 50 == 0:
                print(f"  ...{i}/{len(tickers)}")
    return out


def build_table(rows):
    df = pd.DataFrame(rows)
    sc = df.apply(score_stock, axis=1, result_type="expand")
    df["score"], df["coverage"] = sc[0], sc[1]
    df["flags"] = [", ".join(red_flags(r)) for r in rows]
    df["notes"] = ["; ".join(notes_for(r)) for r in rows]
    df["grade"] = [grade(sc_, 1 if fl else 0, r) for sc_, fl, r in zip(df["score"], df["flags"], rows)]
    df["ok"] = (df["coverage"] >= MIN_COVERAGE) & df["score"].notna()
    return df


# ============================================================
# EMAIL
# ============================================================
def fmt(x, nd=1, suffix=""):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}{suffix}"


def table_html(sub):
    head = ("Ticker", "Sector", "Score", "Grade", "ROCE / ROA", "ROE", "Sales CAGR", "Profit CAGR",
            "Debt/Equity (lenders: equity/assets)", "Cash/Profit", "Red flags", "Notes")
    th = "".join(f'<th style="text-align:left;padding:3px 6px;border-bottom:1px solid #999;">{h}</th>' for h in head)
    body = ""
    for _, r in sub.iterrows():
        rc = fmt(r["roa"], 1, "% ROA") if r["is_fin"] else fmt(r["roce"], 1, "%")
        cells = [r["ticker"].replace(".NS", ""), (r["sector"] or "-")[:22], fmt(r["score"], 0), r["grade"], rc,
                 fmt(r["roe"], 1, "%"), fmt(r["rev_cagr"], 1, "%"), fmt(r["ni_cagr"], 1, "%"),
                 fmt(r["eq_assets"], 1, "% eq/assets") if r["is_fin"] else fmt(r["de"], 2),
                 "-" if r["is_fin"] else fmt(r["ocf_ni"], 2), r["flags"] or "none", r["notes"] or ""]
        color = "#a00" if r["flags"] else "#000"
        td = "".join(f'<td style="padding:3px 6px;border-bottom:1px solid #eee;">{c}</td>' for c in cells[:-2])
        td += f'<td style="padding:3px 6px;border-bottom:1px solid #eee;color:{color};">{cells[-2]}</td>'
        td += f'<td style="padding:3px 6px;border-bottom:1px solid #eee;color:#666;">{cells[-1]}</td>'
        body += f"<tr>{td}</tr>"
    return f'<table style="border-collapse:collapse;font-size:12px;"><tr>{th}</tr>{body}</table>'


def build_email_html(df, n_requested):
    ok = df[df["ok"]].sort_values("score", ascending=False)
    bad = df[~df["ok"]]
    ap, a, b = ok[ok["grade"] == "A+"], ok[ok["grade"] == "A"], ok[ok["grade"] == "B"]
    cs, ds = ok[ok["grade"] == "C"], ok[ok["grade"] == "D"]
    c, d = len(cs), len(ds)

    warns = []
    if len(df) < 0.9 * n_requested:
        warns.append(f"Only {len(df)} of {n_requested} tickers returned data (under 90%). Some downloads probably failed.")
    if len(bad) > 0.3 * max(len(df), 1):
        warns.append(f"{len(bad)} of {len(df)} stocks had too little financial data to score. Yahoo coverage is thin for some Indian companies.")
    warn_html = ""
    if warns:
        items = "".join(f"<li>{w}</li>" for w in warns)
        warn_html = ('<div style="background:#fff3cd;border:1px solid #e0a800;padding:8px 12px;margin:8px 0;">'
                     f"<b>Data check</b><ul style='margin:4px 0;'>{items}</ul></div>")

    html = f"""<html><body style="font-family:Arial,sans-serif;">
<h2>Quality Tracker, {datetime.now().strftime('%d %b %Y')}</h2>
<p style="font-size:13px;">{len(ok)} stocks scored: <b>{len(ap)} grade A+</b>, <b>{len(a)} grade A</b>, <b>{len(b)} grade B</b>, {c} grade C, {d} grade D.
{len(bad)} could not be scored (not enough data). This is a quality filter, not a buy signal: use it to choose which stocks
deserve your time, then use the Momentum and Earnings trackers for timing.</p>{warn_html}
<p style="font-size:12px;margin-bottom:2px;"><b>Time period behind each column</b> (all from Yahoo's annual statements, which usually give the last 4 financial years;
the latest year is the most recent full year reported, not the trailing 12 months):</p>
<table style="border-collapse:collapse;font-size:12px;">
<tr><td style="padding:2px 8px;"><b>ROCE, ROE, ROA</b></td><td style="padding:2px 8px;">Average of the last 3 years (year-end balance sheet figures)</td></tr>
<tr><td style="padding:2px 8px;"><b>Sales CAGR, Profit CAGR</b></td><td style="padding:2px 8px;">Compound growth from the oldest to the latest year available (normally 3 years of growth from 4 years of data, sometimes more)</td></tr>
<tr><td style="padding:2px 8px;"><b>Years of profit growth</b> (in the score)</td><td style="padding:2px 8px;">Share of year-on-year steps, across all years available, in which profit rose</td></tr>
<tr><td style="padding:2px 8px;"><b>Debt/Equity, Equity/assets, Interest cover</b></td><td style="padding:2px 8px;">Latest year only (a snapshot, not an average)</td></tr>
<tr><td style="padding:2px 8px;"><b>Cash/Profit</b></td><td style="padding:2px 8px;">Average of the last 3 profit-making years</td></tr>
<tr><td style="padding:2px 8px;"><b>Free cash flow positive</b> (in the score)</td><td style="padding:2px 8px;">Latest year only</td></tr>
<tr><td style="padding:2px 8px;"><b>Operating margin trend</b> (in the score)</td><td style="padding:2px 8px;">Change from the oldest to the latest year available (needs 3+ years)</td></tr>
<tr><td style="padding:2px 8px;"><b>Red flag "Profit fell last year"</b></td><td style="padding:2px 8px;">Latest year versus the year before</td></tr>
</table>
<h3>Grade A+: the strongest, with no red flags ({len(ap)})</h3>{table_html(ap) if len(ap) else '<p>None.</p>'}
<h3>Grade A: strong businesses ({len(a)})</h3>{table_html(a) if len(a) else '<p>None.</p>'}
<h3>Grade B: good, with some gaps ({len(b)})</h3>{table_html(b) if len(b) else '<p>None.</p>'}
<h3>Grade C: average or weak on several measures ({len(cs)})</h3>{table_html(cs) if len(cs) else '<p>None.</p>'}
<h3>Grade D: weak ({len(ds)})</h3>{table_html(ds) if len(ds) else '<p>None.</p>'}
"""
    if len(bad):
        names = ", ".join(sorted(t.replace(".NS", "") for t in bad["ticker"]))
        html += f'<h3>Not scored: insufficient data ({len(bad)})</h3><p style="font-size:12px;">{names}</p>'

    html += """
<h3>How the score works</h3>
<table style="border-collapse:collapse;font-size:12px;">
<tr><td style="padding:3px 8px;"><b>Operating companies (100 points)</b></td><td style="padding:3px 8px;">
Profitability 30 (ROCE 15, ROE 15) + Growth 30 (sales CAGR 10, profit CAGR 10 with credit capped at 30%, years of profit growth 10) +
Balance sheet 20 (debt/equity 10, interest cover 10) + Cash quality 20 (operating cash / profit 10, free cash flow positive 5,
operating margin trend 5)</td></tr>
<tr><td style="padding:3px 8px;"><b>Banks and lenders (100 points)</b></td><td style="padding:3px 8px;">
ROE 30 + ROA 20 + profit CAGR 15 + sales CAGR 10 + years of profit growth 10 + capital strength (equity / total assets) 15.
Loan quality (bad loans) is not available from this data source, so a lender's score does not reflect it.</td></tr>
<tr><td style="padding:3px 8px;"><b>Full marks at</b></td><td style="padding:3px 8px;">
ROCE 25%+, ROE 22%+ (20%+ for lenders; ROA 2%+ for banks, 3%+ for other lenders), sales and profit growth 20%+ a year, debt/equity 0.3 or less, interest cover 10x+,
cash flow 1.1x profit or better, equity/assets 12%+ for lenders. Zero points at the low end of each range (for example ROCE 10%).</td></tr>
<tr><td style="padding:3px 8px;"><b>Grades</b></td><td style="padding:3px 8px;">A+: 85+ with no red flags (90+ for lenders; insurers are not eligible), A: 70 to 85, B: 55 to 70, C: 40 to 55, D: below 40.
Averages use the last 3 years (see the period table at the top). If some metrics are missing, the score is rescaled over those available, and stocks with under
60% of the score available are not graded.</td></tr>
<tr><td style="padding:3px 8px;"><b>Red flags</b></td><td style="padding:3px 8px;">
ROE below 8%; ROCE below 10%; debt/equity above 1.5; interest cover below 2.5; cash flow below 60% of profit;
negative free cash flow; sales shrinking over the period; profit fell last year.</td></tr>
<tr><td style="padding:3px 8px;"><b>Notes</b></td><td style="padding:3px 8px;">
<b>Low-base growth</b>: profit CAGR above 60%, usually from a very low starting year, so the score gives no credit above 30%.
<b>Finance arm</b>: for companies whose accounts include a lending arm (set in FINANCE_ARM_TICKERS), debt and cash-flow measures are left out of the score
because the loan book distorts them.</td></tr>
</table>
<p style="font-size:12px;color:#a00;"><b>Limits:</b> data comes from Yahoo Finance and is sometimes wrong or missing for Indian companies, so check anything
you plan to act on against the company's own results. Promoter holding, pledging and one-off items are not captured.
Yearly figures are only 3 to 4 years deep, which is a short history. A high score means a strong past, not a guaranteed future.</p>
</body></html>"""
    return html


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
    if MAX_TICKERS:
        tickers = tickers[:MAX_TICKERS]
    print(f"Loaded {len(tickers)} tickers from {STOCKS_FILE}")
    if not tickers:
        print("Watchlist is empty. Exiting.")
        return

    rows = fetch_all(tickers)
    print(f"Fundamentals returned for {len(rows)}/{len(tickers)} tickers")
    if not rows:
        print("No data, no email.")
        return

    df = build_table(rows)
    print(df[df["ok"]]["grade"].value_counts().to_string())
    df.to_csv("quality_scores.csv", index=False)
    # Saved in the repo for the weekly combined shortlist (combined_shortlist.py)
    os.makedirs("results", exist_ok=True)
    df.assign(run_date=datetime.now().strftime("%Y-%m-%d")).to_csv("results/quality.csv", index=False)

    html = build_email_html(df, len(tickers))
    flag = " [CHECK DATA]" if len(df) < 0.9 * len(tickers) else ""
    send_email(f"Quality Tracker, {datetime.now().strftime('%d %b %Y')}{flag}", html)


if __name__ == "__main__":
    main()
