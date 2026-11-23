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

New ideas: the same tiers for the universe candidates (stocks NOT on the watchlist), from results/universe_quality.csv,
universe_earnings.csv and universe_valuation.csv, written by the weekly universe workflow (universe_trackers.yml).
Valuation there uses Screener's industry P/E unless peers.csv has an entry, so read it as a rougher yardstick.
"""
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

RES = "results"
UNIVERSE_CAP = 20          # rows per tier in the New ideas section of the email (everything is in the CSV)
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


def build(prefix=""):
    q, e, v = load(f"{prefix}quality.csv"), load(f"{prefix}earnings.csv"), load(f"{prefix}valuation.csv")
    missing = [f"{prefix}{n}" for n, d in (("quality.csv", q), ("earnings.csv", e), ("valuation.csv", v)) if d is None]
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
    uc = load("universe_candidates.csv") if prefix else None
    if uc is not None:
        j = j.join(uc[["name", "ret_6m", "extended"]], how="left")
    return j, warns


def tbl(rows, head):
    """Compact HTML (no per-cell styles) so the email stays under Gmail's 102 KB clipping limit."""
    th = "".join(f"<th align='left' bgcolor='#eeeeee'>{h}</th>" for h in head)
    body = "".join(f"<tr{' bgcolor=#f7f7f7' if k % 2 else ''}>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for k, r in enumerate(rows))
    return f"<table cellpadding='3' cellspacing='0' style='border-collapse:collapse;font-size:12px;'><tr>{th}</tr>{body}</table>"


def tiers_html(j, universe=False, cap=None):
    head = ["Ticker", "Sector", "Quality", "Earnings", "Evidence", "Net profit YoY", "Since result", "Trend", "Valuation", "Premium", "PEG", "Quality flags", "Earnings note"]
    if universe:
        head = ["Ticker", "Company", "6M return"] + head[2:]
    h = ""
    for t in TIERS:
        sub = j[j["tier"] == t].sort_values("score", ascending=False)
        shown = sub.head(cap) if cap else sub
        h += f"<h3>{t} ({len(sub)})</h3><p style='font-size:12px;'>{TIER_TEXT[t]}</p>"
        if sub.empty:
            h += "<p>None right now.</p>"
            continue
        rows = []
        for i, r in shown.iterrows():
            ni = "turned profit" if str(r["turned_profit"]) == "True" else pct_s(r["ni_yoy"])
            prem = "-" if pd.isna(r["premium"]) else f"{r['premium']:+.0f}%"
            peg = "-" if pd.isna(r["peg"]) else f"{r['peg']:.1f}"
            note = "; ".join(x for x in [r["flag"] if isinstance(r["flag"], str) else "", r["src_note"] if isinstance(r["src_note"], str) and "ignored" in r["src_note"] else ""] if x) or "-"
            fl = r["flags"] if isinstance(r["flags"], str) and r["flags"] else "none"
            lead = [i, str(r["name"])[:22], (pct_s(r["ret_6m"]) + (" (extended)" if str(r["extended"]) == "True" else ""))] if universe else [i, str(r["sector"])[:20]]
            rows.append(lead + [f'{r["grade"]} ({r["score"]:.0f})', r["earn"], r["evidence"], ni, pct_s(r["drift"], 1),
                         r["trend"], str(r["tag"]).replace(" vs peers", ""), prem, peg,
                         f'<span style="color:#a00;">{fl}</span>' if fl != "none" else fl, note])
        h += tbl(rows, head)
        if cap and len(sub) > cap:
            h += f"<p style='font-size:12px;'>{len(sub) - cap} more in the CSV.</p>"
    return h


def email_html(j, warns, changes, ju=None, uwarns=(), uchanges=()):
    h = f'<html><body style="font-family:Arial,sans-serif;font-size:13px;"><h2>Combined Shortlist, {datetime.now().strftime("%d %b %Y")}</h2>'
    n = {t: int((j["tier"] == t).sum()) for t in TIERS}
    h += (f'<p><b>{n[TIERS[0]]}</b> core, {n[TIERS[1]]} waiting for trend, {n[TIERS[2]]} good but expensive, {n[TIERS[3]]} valuation n/a, '
          f'out of {len(j)} stocks scored by the Quality tracker. A shortlist to research, not a buy list.</p>')
    for w in warns:
        h += f'<p style="color:#a00;">{w}</p>'
    if changes:
        h += "<p><b>Changes since the last run</b><br>" + "<br>".join(changes) + "</p>"
    h += tiers_html(j)
    if ju is not None:
        nu = {t: int((ju["tier"] == t).sum()) for t in TIERS}
        h += ('<hr><h2>New ideas: stocks not on your watchlist</h2>'
              f'<p>From the weekly universe screen: <b>{nu[TIERS[0]]}</b> core, {nu[TIERS[1]]} waiting for trend, {nu[TIERS[2]]} good but expensive, '
              f'{nu[TIERS[3]]} valuation n/a, out of {len(ju)} candidates that went through the full Quality, Earnings and Valuation trackers. '
              'Valuation here uses Screener\'s industry P/E unless peers.csv has an entry, so it is a rougher yardstick. '
              '"Extended" means up more than 100% in six months. Check the business before anything else: these are names you have not looked at.</p>')
        for w in uwarns:
            h += f'<p style="color:#a00;">{w}</p>'
        if uchanges:
            h += "<p><b>New ideas: changes since the last run</b><br>" + "<br>".join(uchanges) + "</p>"
        h += tiers_html(ju, universe=True, cap=UNIVERSE_CAP)
    h += """<h3>How to read this</h3>
<p style="font-size:12px;">Quality, Earnings and Valuation are the latest saved results of the three trackers (see their own emails for the full tables and the definitions).
<b>Net profit only (Screener)</b> and <b>EPS only (Yahoo)</b> are weaker evidence than <b>EBIT-graded</b>: one-offs and tax can distort net profit and EPS.
A stock must be Quality A+ or A and have strong or good results to be listed at all; stocks that fail either are in results/shortlist.csv with no tier.
Valuation is a weak guide to the next 1 to 3 months, and a strong quarter does not mean a strong stock: read the company's own results before acting.</p></body></html>"""
    return h


def tier_changes(path, j):
    if not os.path.exists(path):
        return []
    prev = pd.read_csv(path, keep_default_na=False).set_index("ticker")["tier"]
    ch = []
    for t in TIERS:
        now, was = set(j.index[j["tier"] == t]), set(prev.index[prev == t])
        new = sorted(now - was)
        if new:
            ch.append(f"<b>Joined {t}:</b> " + ", ".join(f"{x}" + (f" (was {prev.get(x) or 'not listed'})" if x in prev.index else "") for x in new))
    gone = sorted(x for x in prev.index[prev != ""] if x in j.index and j.loc[x, "tier"] == "")
    if gone:
        ch.append("<b>Dropped off the list:</b> " + ", ".join(gone))
    return ch


def save(j, path):
    out = j.reset_index().rename(columns={"index": "ticker"})
    os.makedirs(RES, exist_ok=True)
    out.assign(run_date=datetime.now().strftime("%Y-%m-%d")).to_csv(path, index=False)


def main():
    import momentum_tracker as m
    j, warns = build()
    if j is None:
        for w in warns:
            print("WARNING:", w)
        try:
            m.send_email(f"Combined Shortlist, {datetime.now().strftime('%d %b %Y')} [NO DATA]",
                         "<html><body><p>" + "<br>".join(warns) + "</p></body></html>")
        except Exception as ex:
            print("Could not send the problem email:", ex)
        sys.exit(0)
    prev_path = os.path.join(RES, "shortlist.csv")
    changes = tier_changes(prev_path, j)
    save(j, prev_path)
    print(j["tier"].value_counts().to_string())
    # New ideas: universe candidates (only when the universe workflow has produced its three results files)
    ju, uwarns, uchanges = None, [], []
    if any(os.path.exists(os.path.join(RES, f"universe_{n}.csv")) for n in ("quality", "earnings", "valuation")):
        ju, uwarns = build("universe_")
        if ju is not None:
            upath = os.path.join(RES, "universe_shortlist.csv")
            uchanges = tier_changes(upath, ju)
            save(ju, upath)
            print("New ideas:", ju["tier"].value_counts().to_dict())
        else:
            warns = warns + ["New ideas not shown: " + " ".join(uwarns)]
            uwarns = []
    for w in warns + uwarns:
        print("WARNING:", w)
    html = email_html(j, warns, changes, ju, uwarns, uchanges)
    m.send_email(f"Combined Shortlist, {datetime.now().strftime('%d %b %Y')}" + (" [CHECK DATA]" if warns or uwarns else ""), html)


if __name__ == "__main__":
    main()
