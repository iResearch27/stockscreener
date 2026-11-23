"""
Combined shortlist: joins the latest Quality, Earnings and Valuation results and emails one tiered list.

Reads results/quality.csv, results/earnings.csv, results/valuation.csv (each tracker saves its own file after every run;
the workflows commit them to the repo). Writes results/shortlist.csv and emails the tiers, with what changed since the
previous run. Run weekly after the other trackers (see .github/workflows/combined_shortlist.yml).

Tiers (a stock needs Quality A+/A and an earnings pass to be listed at all):
  1 Core                 valuation cheap or fair against peers, price above the SMA200
  2 Waiting for trend    valuation cheap or fair, price below the SMA200 (or trend unknown)
  3 Good but expensive   valuation rich or stretched against peers
  4 Valuation n/a        cyclical, insurer or not valued: judge on EV/EBITDA and P/B yourself
Earnings pass: result Strong or Good; for stocks Yahoo cannot grade yearly, EPS growth of 15%+ (marked *).
"""
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

RES = "results"
STALE_DAYS = 10            # warn when an input file is older than this
EPS_PASS = 15.0
TIERS = ["1 Core", "2 Waiting for trend", "3 Good but expensive", "4 Valuation n/a"]
TIER_TEXT = {
    "1 Core": "Quality A+/A, strong or good results, cheap or fair against peers, price above the SMA200. The best fit for the idea: good business, good numbers, fair price, trend intact.",
    "2 Waiting for trend": "Same as Core, but the price is below the SMA200 (or the trend is unknown). Watch for a move back above it.",
    "3 Good but expensive": "Good business and good numbers, but rich or stretched against peers. The market already pays for the quality.",
    "4 Valuation n/a": "Good business and good numbers, but P/E cannot judge them (cyclicals, insurers, loss-makers). Check EV/EBITDA and P/B across the cycle.",
}


def short(t):
    return str(t).replace(".NS", "")


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    d["ticker"] = d["ticker"].map(short)
    return d.drop_duplicates("ticker").set_index("ticker")


def age_days(d):
    try:
        return (datetime.now() - datetime.strptime(str(d.iloc[0]["run_date"]), "%Y-%m-%d")).days
    except Exception:
        return None


def pct_s(x, nd=0):
    return "-" if pd.isna(x) else f"{x:+.{nd}f}%"


def build():
    q, e, v = load("quality.csv"), load("earnings.csv"), load("valuation.csv")
    missing = [n for n, d in (("quality.csv", q), ("earnings.csv", e), ("valuation.csv", v)) if d is None]
    if missing:
        return None, [f"results/{m} is missing: run that tracker's workflow once so it saves its results." for m in missing]
    warns = []
    for n, d in (("Quality", q), ("Earnings", e), ("Valuation", v)):
        a = age_days(d)
        if a is not None and a > STALE_DAYS:
            warns.append(f"{n} results are {a} days old. Run that workflow before trusting this list.")
    q = q[q["ok"].astype(str).str.lower() == "true"] if "ok" in q.columns else q
    j = q[["sector", "score", "grade", "flags"]].join(
        e.reindex(columns=["quality", "reported", "rev_yoy", "ebit_yoy", "ni_yoy", "turned_profit", "eps_yoy", "reaction", "drift",
                           "above_sma200", "flag", "src_note"]).rename(columns={"quality": "e_result"}), how="left"
    ).join(v[["tag", "premium", "peg", "ev_ebitda", "pe", "own_peers"]], how="left")

    def earn(r):
        res = r["e_result"]
        if isinstance(res, str) and res in ("Strong", "Good", "Weak", "Mixed"):
            basis = "Net profit only (Screener)" if isinstance(r["src_note"], str) and "net profit basis" in r["src_note"] else "EBIT-graded"
            return res, basis, res in ("Strong", "Good")
        if pd.notna(r["eps_yoy"]) and not (isinstance(res, str) and res in ("Strong", "Good", "Weak", "Mixed")):
            return f"EPS {r['eps_yoy']:+.0f}%*", "EPS only (Yahoo)", r["eps_yoy"] >= EPS_PASS
        return "", "", False

    ev = j.apply(earn, axis=1, result_type="expand")
    j["earn"], j["evidence"], j["epass"] = ev[0], ev[1], ev[2]
    j["qpass"] = j["grade"].isin(["A+", "A"])

    def trend(x):
        return "Above SMA200" if x is True or str(x) == "True" else ("Below SMA200" if str(x) == "False" else "-")

    j["trend"] = j["above_sma200"].map(trend)
    tag = j["tag"].fillna("")
    ok_v, exp_v = tag.isin(["Cheap vs peers", "Fair vs peers"]), tag.isin(["Rich vs peers", "Stretched vs peers"])

    def tier(i):
        r = j.iloc[i]
        if not (r["qpass"] and r["epass"]):
            return ""
        if ok_v.iloc[i]:
            return "1 Core" if r["trend"] == "Above SMA200" else "2 Waiting for trend"
        return "3 Good but expensive" if exp_v.iloc[i] else "4 Valuation n/a"

    j["tier"] = [tier(i) for i in range(len(j))]
    return j, warns


def tbl(rows, head):
    th = "".join(f'<th style="text-align:left;padding:3px 8px;background:#eee;font-size:12px;">{h}</th>' for h in head)
    body = "".join("<tr>" + "".join(f'<td style="padding:3px 8px;font-size:12px;border-bottom:1px solid #eee;">{c}</td>' for c in r) + "</tr>" for r in rows)
    return f'<table style="border-collapse:collapse;">{th and "<tr>" + th + "</tr>"}{body}</table>'


def email_html(j, warns, changes):
    h = f'<html><body style="font-family:Arial,sans-serif;font-size:13px;"><h2>Combined Shortlist, {datetime.now().strftime("%d %b %Y")}</h2>'
    n = {t: int((j["tier"] == t).sum()) for t in TIERS}
    h += (f'<p><b>{n[TIERS[0]]}</b> core, {n[TIERS[1]]} waiting for trend, {n[TIERS[2]]} good but expensive, {n[TIERS[3]]} valuation n/a, '
          f'out of {len(j)} stocks scored by the Quality tracker. A shortlist to research, not a buy list.</p>')
    for w in warns:
        h += f'<p style="color:#a00;">{w}</p>'
    if changes:
        h += "<p><b>Changes since the last run</b><br>" + "<br>".join(changes) + "</p>"
    head = ["Ticker", "Sector", "Quality", "Earnings", "Evidence", "Net profit YoY", "Since result", "Trend", "Valuation", "Premium", "PEG", "Quality flags", "Earnings note"]
    for t in TIERS:
        sub = j[j["tier"] == t].sort_values("score", ascending=False)
        h += f"<h3>{t} ({len(sub)})</h3><p style='font-size:12px;'>{TIER_TEXT[t]}</p>"
        if sub.empty:
            h += "<p>None right now.</p>"
            continue
        rows = []
        for i, r in sub.iterrows():
            ni = "turned profit" if str(r["turned_profit"]) == "True" else pct_s(r["ni_yoy"])
            prem = "-" if pd.isna(r["premium"]) else f"{r['premium']:+.0f}%"
            peg = "-" if pd.isna(r["peg"]) else f"{r['peg']:.1f}"
            note = "; ".join(x for x in [r["flag"] if isinstance(r["flag"], str) else "", r["src_note"] if isinstance(r["src_note"], str) and "ignored" in r["src_note"] else ""] if x) or "-"
            fl = r["flags"] if isinstance(r["flags"], str) and r["flags"] else "none"
            rows.append([i, str(r["sector"])[:20], f'{r["grade"]} ({r["score"]:.0f})', r["earn"], r["evidence"], ni, pct_s(r["drift"], 1),
                         r["trend"], str(r["tag"]).replace(" vs peers", ""), prem, peg,
                         f'<span style="color:#a00;">{fl}</span>' if fl != "none" else fl, note])
        h += tbl(rows, head)
    h += """<h3>How to read this</h3>
<p style="font-size:12px;">Quality, Earnings and Valuation are the latest saved results of the three trackers (see their own emails for the full tables and the definitions).
<b>Net profit only (Screener)</b> and <b>EPS only (Yahoo)</b> are weaker evidence than <b>EBIT-graded</b>: one-offs and tax can distort net profit and EPS.
A stock must be Quality A+ or A and have strong or good results to be listed at all; stocks that fail either are in results/shortlist.csv with no tier.
Valuation is a weak guide to the next 1 to 3 months, and a strong quarter does not mean a strong stock: read the company's own results before acting.</p></body></html>"""
    return h


def main():
    j, warns = build()
    if j is None:
        for w in warns:
            print("WARNING:", w)
        try:
            import momentum_tracker as m
            m.send_email(f"Combined Shortlist, {datetime.now().strftime('%d %b %Y')} [NO DATA]",
                         "<html><body><p>" + "<br>".join(warns) + "</p></body></html>")
        except Exception as ex:
            print("Could not send the problem email:", ex)
        sys.exit(0)
    prev_path = os.path.join(RES, "shortlist.csv")
    changes = []
    if os.path.exists(prev_path):
        prev = pd.read_csv(prev_path, keep_default_na=False).set_index("ticker")["tier"]
        for t in TIERS:
            now, was = set(j.index[j["tier"] == t]), set(prev.index[prev == t])
            new = sorted(now - was)
            if new:
                changes.append(f"<b>Joined {t}:</b> " + ", ".join(f"{x}" + (f" (was {prev.get(x) or 'not listed'})" if x in prev.index else "") for x in new))
        gone = sorted(x for x in prev.index[prev != ""] if x in j.index and j.loc[x, "tier"] == "")
        if gone:
            changes.append("<b>Dropped off the list:</b> " + ", ".join(gone))
    out = j.reset_index().rename(columns={"index": "ticker"})
    os.makedirs(RES, exist_ok=True)
    out.assign(run_date=datetime.now().strftime("%Y-%m-%d")).to_csv(prev_path, index=False)
    print(j["tier"].value_counts().to_string())
    for w in warns:
        print("WARNING:", w)
    html = email_html(j, warns, changes)
    import momentum_tracker as m
    m.send_email(f"Combined Shortlist, {datetime.now().strftime('%d %b %Y')}" + (" [CHECK DATA]" if warns else ""), html)


if __name__ == "__main__":
    main()
