"""
Universe screen, stage 2: fundamental pre-filter from the Screener export (no fundamentals fetched from Yahoo).

Takes the stocks that passed stage 1 (results/universe_prices.csv, 'eligible'), matches each to its row in
screener_export.xlsx, scores business quality from the export's own columns ("quality-lite"), and writes:
  results/universe_candidates.csv   every eligible, matched stock with its score, flags and price momentum
  candidates.txt                    the top stocks not already on your watchlist: the input for stage 3 (the full trackers)
and emails the top of the list with a match report.

Matching (the export has no company links): NSE's company name for each ticker (symbol_names.csv, two columns: SYMBOL.NS and
NAME OF COMPANY) is matched to Screener's abbreviated name, with Screener's CMP checked against the Yahoo price from stage 1
(see universe_match.py). No Yahoo requests are made in this stage. Banks and lenders are spotted from the export's own numbers
(high debt to equity, low return on assets), since the sector is not available here; the full trackers use Yahoo's sector later.

Quality-lite is a cut-down version of the Quality tracker, using only what the export has: 3-year return on capital,
5-year profit and sales growth, leverage and the latest quarter's growth. The full Quality tracker then re-scores the candidates
properly (cash flow, interest cover, margin trend). Run weekly after universe_screen.py.
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd

import momentum_tracker as m
import screener_import as si
import universe_match as um

PRICES_FILE = "results/universe_prices.csv"
NAMES_FILE = os.environ.get("SYMBOL_NAMES_FILE", "symbol_names.csv")
OUT_FILE = "results/universe_candidates.csv"
CAND_FILE = "candidates.txt"
WATCHLIST = m.STOCKS_FILE
N_TOP = 200                    # top stocks by quality-lite score that go to the full trackers
N_MAX = 350                    # cap on the total, compounders included
MIN_SCORE = 55                 # candidates also need at least this quality-lite score
MAX_MCAP_CR = None             # optional ceiling, not used
EXTENDED_6M = 100.0            # flag stocks up this much in six months
TOP_N = 40                     # rows in the email


# ---------------------------------------------------------------- names
def find_names_file():
    """symbol_names.csv, or any .csv/.xlsx in the repo root whose name has 'symbol' and 'name' in it (any capitalisation)."""
    import glob
    if os.path.exists(NAMES_FILE):
        return NAMES_FILE
    for p in sorted(glob.glob("*.csv") + glob.glob("*.xlsx")):
        low = os.path.basename(p).lower()
        if "symbol" in low and "name" in low:
            return p
    return None


def load_symbol_names():
    """NSE symbol -> official company name. First column symbol, second column name."""
    path = find_names_file()
    if not path:
        return None
    d = pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path)
    d = d.iloc[:, :2]
    d.columns = ["ticker", "name"]
    d["ticker"] = d["ticker"].astype(str).str.strip().str.upper()
    d["ticker"] = d["ticker"].where(d["ticker"].str.endswith(".NS"), d["ticker"] + ".NS")
    return dict(zip(d["ticker"], d["name"].astype(str)))


# ---------------------------------------------------------------- scoring
def clip01(x, lo, hi):
    return ((x - lo) / (hi - lo)).clip(0, 1)


def is_lender_row(r):
    """Banks and NBFCs from the export alone: borrowings several times equity and a thin return on assets."""
    return bool(pd.notna(r["de"]) and r["de"] >= 3 and pd.notna(r["roa3"]) and r["roa3"] <= 4)


def quality_lite(d):
    """d: DataFrame with Screener fields (roce3, roe3, roa3, de, profit_5y, sales_5y, ni_yoy, rev_yoy, np_cr, pe) and 'lender'."""
    ln = d["lender"]
    ret = pd.Series(np.where(ln, clip01(d["roe3"], 8, 18), clip01(d["roce3"], 10, 30)), index=d.index).fillna(0)
    prof = clip01(d["profit_5y"].clip(upper=60), 0, 25).fillna(0)
    sales = clip01(d["sales_5y"].clip(upper=60), 5, 20).fillna(0)
    lev = pd.Series(np.where(ln, clip01(d["roa3"], 0.5, 1.5), 1 - clip01(d["de"], 0, 2)), index=d.index).fillna(0.3)
    rec = (clip01(d["ni_yoy"].clip(upper=100), 0, 30).fillna(0) * 0.5 + clip01(d["rev_yoy"].clip(upper=100), 0, 20).fillna(0) * 0.5)
    d = d.copy()
    d["qscore"] = (ret * 30 + prof * 20 + sales * 10 + lev * 15 + rec * 25).round(0)
    flags = []
    for _, r in d.iterrows():
        f = []
        low_ret = (r["roe3"] < 8) if r["lender"] else (r["roce3"] < 10)
        if pd.isna(r["roe3"] if r["lender"] else r["roce3"]) or low_ret:
            f.append("Low returns")
        if pd.notna(r["profit_5y"]) and r["profit_5y"] < 0:
            f.append("Profit shrinking over 5y")
        if pd.isna(r["profit_5y"]):
            f.append("Short history")
        if pd.notna(r["np_cr"]) and r["np_cr"] < 0:
            f.append("Loss this quarter")
        if not r["lender"] and pd.notna(r["de"]) and r["de"] > 1.5:
            f.append("High debt")
        if pd.isna(r["pe"]):
            f.append("No P/E (loss-making)")
        flags.append(", ".join(f))
    d["qflags"] = flags
    d["hard_flag"] = d["qflags"].str.contains("Low returns|Loss this quarter|No P/E|Profit shrinking")
    return d


# ---------------------------------------------------------------- email
def email_html(res, report, n_elig, warns, new_cands):
    h = f'<html><body style="font-family:Arial,sans-serif;font-size:13px;"><h2>Universe Pre-filter, {datetime.now().strftime("%d %b %Y")}</h2>'
    h += (f"<p>{n_elig} stocks passed the price and liquidity screen. <b>{len(res)}</b> were matched to the Screener export and price-checked; "
          f"<b>{int(res['candidate'].sum())}</b> are candidates for the full Quality, Earnings and Valuation trackers "
          f"(top {N_TOP} by quality-lite score, plus steady high-return compounders; no hard red flags; not on your watchlist).</p>")
    for w in warns:
        h += f'<p style="color:#a00;">{w}</p>'
    if new_cands:
        h += f"<p><b>New candidates since last run ({len(new_cands)}):</b> " + ", ".join(x.replace(".NS", "") for x in new_cands[:60]) + "</p>"
    head = ["Ticker", "Company", "Type", "Quality-lite", "ROCE 3y", "5y profit", "5y sales", "D/E", "Qtr profit", "P/E", "6M return", "Trend", "Flags"]
    th = "".join(f'<th style="text-align:left;padding:3px 8px;background:#eee;font-size:12px;">{x}</th>' for x in head)
    rows = ""
    for t, r in res[res["candidate"]].sort_values("qscore", ascending=False).head(TOP_N).iterrows():
        def n(x, nd=0, suf=""):
            return "-" if pd.isna(x) else f"{x:.{nd}f}{suf}"
        ret = n(r["roe3"], 0, "% ROE") if r["lender"] else n(r["roce3"], 0, "%")
        trend = "Above SMA200" if r["above_sma200"] is True or str(r["above_sma200"]) == "True" else "Below SMA200"
        cells = [t.replace(".NS", ""), str(r["name"])[:24], "lender" if r["lender"] else "-", n(r["qscore"]), ret, n(r["profit_5y"], 0, "%"), n(r["sales_5y"], 0, "%"),
                 "-" if r["lender"] else n(r["de"], 1), n(r["ni_yoy"], 0, "%"), n(r["pe"], 1), n(r["ret_6m"], 0, "%") + (" (extended)" if r["extended"] else ""),
                 trend, r["qflags"] or "-"]
        rows += "<tr>" + "".join(f'<td style="padding:3px 8px;font-size:12px;border-bottom:1px solid #eee;">{c}</td>' for c in cells) + "</tr>"
    h += f"<h3>Top {min(TOP_N, int(res['candidate'].sum()))} candidates</h3><table style='border-collapse:collapse;'><tr>{th}</tr>{rows}</table>"
    h += (f'<p style="font-size:12px;"><b>Matching:</b> {report["exact"]} exact, {len(report["fuzzy"])} close (name and price both agree), '
          f'{len(report["unmatched"])} not matched. Not matched is mostly stocks outside the Screener export or with very different names; '
          f'{len(report.get("no_name", []))} tickers are missing from the symbol-name file.</p>')
    if report["fuzzy"]:
        h += '<p style="font-size:12px;">Close matches (check these): ' + "; ".join(f"{t.replace('.NS', '')}={n}" for t, n, _ in report["fuzzy"][:40]) + (" ..." if len(report["fuzzy"]) > 40 else "") + "</p>"
    if report["rejected_list"]:
        h += '<p style="font-size:12px;">name_map.csv entries rejected because the price differed by more than 15%: ' + ", ".join(f"{t.replace('.NS', '')} (vs {n})" for t, n in report["rejected_list"][:30]) + "</p>"
    h += """<h3>How to read this</h3><p style="font-size:12px;">Quality-lite scores 0 to 100 from the Screener export only: 3-year return on capital (return on equity for banks and lenders, 30 points),
5-year profit growth (20), 5-year sales growth (10), leverage (15, return on assets for lenders) and the latest quarter's profit and sales growth (25).
It is a cut-down version of the Quality tracker and deliberately generous: its job is to pick stocks worth the full analysis, not to rank them.
Steady compounders (return on capital 25%+, ROE 18%+ for lenders, low debt, 5-year profit growth of 5%+) are added even when growth keeps their score lower. Hard flags (excluded from candidates): return on capital under 10% (ROE under 8% for lenders), a loss this quarter, no P/E, shrinking 5-year profit.
"Extended" means up more than 100% in six months: still a candidate, but late. The full trackers re-score every candidate; the Combined Shortlist
then shows the ones that pass, in a separate "New ideas" section.</p></body></html>"""
    return h


def main():
    if not os.path.exists(PRICES_FILE):
        print(f"{PRICES_FILE} not found: run universe_screen.py first.")
        return
    px = pd.read_csv(PRICES_FILE).set_index("ticker")
    elig = px[px["eligible"].astype(str).str.lower() == "true"]
    print(f"{len(elig)} eligible stocks from the price screen")
    sc, info = si.load_screener(allow_names=True)
    if sc.empty:
        print("Screener export problem:", info)
        m.send_email(f"Universe Pre-filter, {datetime.now().strftime('%d %b %Y')} [NO DATA]", f"<html><body><p>{info}</p></body></html>")
        return
    print("Screener export:", info)
    nse_names = load_symbol_names()
    if nse_names is None:
        msg = (f"{NAMES_FILE} not found in the repo root (files there: {sorted(os.listdir('.'))[:40]}). It maps NSE symbols to company names "
               "(two columns, SYMBOL.NS and NAME OF COMPANY). Upload it to the repo root, next to universe.txt.")
        print(msg)
        m.send_email(f"Universe Pre-filter, {datetime.now().strftime('%d %b %Y')} [NO DATA]", f"<html><body><p>{msg}</p></body></html>")
        return
    info_ = {t: (nse_names[t], float(elig.loc[t, "price"])) for t in elig.index if t in nse_names}
    no_name = [t for t in elig.index if t not in nse_names]
    matched, rep = um.match(sc, info_, si.load_name_map())
    rep["no_name"] = no_name
    print(f"Matched {len(matched)} (exact {rep['exact']}, close {len(rep['fuzzy'])}); {len(rep['unmatched'])} unmatched; "
          f"{rep['price_rejected']} name_map overrides rejected on price; {len(no_name)} tickers not in {NAMES_FILE}")

    # P/E of every stock matched to the export (not only candidates), so peers.csv can name any listed ticker as a peer
    if "pe" in matched.columns:
        os.makedirs("results", exist_ok=True)
        matched[["pe"]].dropna().reset_index().rename(columns={matched.index.name or "index": "ticker"}).to_csv("results/universe_pe_lookup.csv", index=False)
    d = matched.join(elig[["price", "ret_1m", "ret_3m", "ret_6m", "off_high", "above_sma200", "turnover"]])
    d["lender"] = [is_lender_row(r) for _, r in d.iterrows()]
    for c in ("roce3", "roe3", "roa3", "de", "profit_5y", "sales_5y", "ni_yoy", "rev_yoy", "np_cr", "pe"):
        if c not in d.columns:
            d[c] = np.nan
    d = quality_lite(d)
    d["extended"] = d["ret_6m"] > EXTENDED_6M
    watch = set(m.load_tickers(WATCHLIST)) if os.path.exists(WATCHLIST) else set()
    d["on_watchlist"] = d.index.isin(watch)
    # Steady compounders (high returns on capital, modest growth) score lower on growth, so they get their own route in:
    # return on capital 25%+ (ROE 18%+ for lenders), low debt, 5-year profit growth of at least 5%, no hard flags.
    d["compounder"] = np.where(d["lender"], d["roe3"] >= 18, (d["roce3"] >= 25) & (d["de"].fillna(9) <= 1)) & (d["profit_5y"] >= 5)
    base = d[(~d["hard_flag"]) & (~d["on_watchlist"])].copy()
    base["sort_ret"] = np.where(base["lender"], base["roe3"], base["roce3"])
    base = base.sort_values(["qscore", "sort_ret"], ascending=False)
    top = base[base["qscore"] >= MIN_SCORE].head(N_TOP)
    comp = base[base["compounder"]]
    pool = pd.concat([top, comp[~comp.index.isin(top.index)]]).sort_values(["qscore", "sort_ret"], ascending=False)
    cands = list(pool.head(N_MAX).index)
    d["candidate"] = d.index.isin(cands)

    prev = set()
    if os.path.exists(CAND_FILE):
        prev = {l.strip() for l in open(CAND_FILE) if l.strip() and not l.startswith("#")}
    new = [t for t in cands if t not in prev]
    warns = []
    if len(d) < 0.5 * len(elig):
        warns.append(f"Only {len(d)} of {len(elig)} eligible stocks were matched to the Screener export. Names may be missing from the cache "
                     "(Yahoo throttled?) or the export may not cover the universe. Run again later or check name_map.csv.")
    if len(cands) < 50:
        warns.append(f"Only {len(cands)} candidates: the filters may be too strict, or the match rate is low.")
    for w in warns:
        print("WARNING:", w)
    os.makedirs("results", exist_ok=True)
    out = d.reset_index().rename(columns={"index": "ticker"})
    out.assign(run_date=datetime.now().strftime("%Y-%m-%d")).to_csv(OUT_FILE, index=False)
    if len(cands) >= 50:
        with open(CAND_FILE, "w") as f:
            f.write(f"# Written by universe_prefilter.py on {datetime.now():%Y-%m-%d}. Do not edit: it is overwritten each week.\n")
            f.write("\n".join(cands) + "\n")
    print(f"{len(cands)} candidates; {len(new)} new")
    m.send_email(f"Universe Pre-filter, {datetime.now().strftime('%d %b %Y')}" + (" [CHECK DATA]" if warns else ""),
                 email_html(d, rep, len(elig), warns, new))


if __name__ == "__main__":
    main()
