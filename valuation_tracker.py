"""
Valuation tracker: is the price sensible for what the company earns and how fast it grows?

Reads the manual Screener.in export (screener_export.xlsx, see screener_import.py) and, for every stock in watchlist.txt,
compares its P/E with a PEER P/E and checks it against growth:

  * Peer P/E = the median P/E of the peers you list in peers.csv, or else Screener's own industry P/E (the "Ind PE" column).
  * Premium vs peers = P/E / peer P/E - 1.
  * Growth check (PEG) = P/E divided by the 5-year profit growth rate (growth capped at 40%; n/a below 5% or above 60%).
  * Own history: if the export has a historical P/E column (3, 5 or 10 year), the premium to the stock's own history is shown.
  * Banks and lenders are judged on P/B per point of ROE, against other lenders, because P/E and growth say little about them.
  * Cyclicals (steel, cement, oil, metals, chemicals...) are not tagged on P/E: their P/E is lowest at peak profits.
    EV/EBITDA and P/B are shown instead.

A tag is not a buy or sell signal. Valuation says little about the next 1 to 3 months, and expensive stocks can keep rising.
Use it to size positions and to decide how much you are paying for quality.

Files: screener_export.xlsx (from Screener.in), peers.csv (optional, your own peer groups and stock types), watchlist.txt.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

import momentum_tracker as m
import screener_import as si

# ============================================================
# CONFIGURATION
# ============================================================
STOCKS_FILE = m.STOCKS_FILE
PEERS_FILE = os.environ.get("PEERS_FILE", "peers.csv")
MAX_WORKERS = 6
MAX_TICKERS = int(os.environ.get("VT_MAX_TICKERS", "0"))   # testing only; 0 = all

PE_MAX_VALID = 100.0          # a P/E above this (or negative) is ignored when taking peer medians
MIN_PEERS = 3                 # a peer median needs at least this many valid peers
GROWTH_CAP = 40.0             # profit growth above this earns no extra credit in PEG
GROWTH_MIN = 5.0              # below this, PEG is not meaningful
GROWTH_LOW_BASE = 60.0        # above this, the 5-year growth comes from a tiny starting profit: PEG is not shown (same as Quality tracker)
# Premium to peers (P/E / peer P/E - 1, in %)
CHEAP_BELOW = -20.0
FAIR_UP_TO = 20.0
RICH_UP_TO = 50.0             # above this: Stretched
PEG_GOOD = 1.0                # PEG at or below this: growth backs the price
PEG_HIGH = 2.5                # PEG above this: growth does not back the price
HIST_RICH_PCT = 25.0          # premium to own historical P/E called rich
HIST_CHEAP_PCT = -20.0

# Industries (Yahoo wording) where P/E is unreliable because profits swing with the cycle
CYCLICAL_WORDS = ("steel", "aluminum", "copper", "metal", "coal", "oil & gas", "oil and gas", "building materials",
                  "commodity chemicals", "paper", "sugar", "shipping", "marine", "airlines", "auto parts", "agricultural inputs",
                  "construction materials", "mining", "tyre", "tire")
# Groups whose businesses span several industries, so an industry P/E is a poor yardstick. Add to peers.csv to fix.
CONGLOMERATE_HINT = {"RELIANCE.NS", "LT.NS", "M&M.NS", "ITC.NS", "ADANIENT.NS", "TATACHEM.NS", "GRASIM.NS", "BAJAJFINSV.NS",
                     "JSWENERGY.NS", "SIEMENS.NS", "TATAMOTORS.NS", "TITAN.NS"}


# ============================================================
# STOCK TYPE (lender / insurer / cyclical) FROM YAHOO
# ============================================================
def fetch_type(t):
    try:
        info = yf.Ticker(t).info or {}
        return t, (info.get("sector") or ""), (info.get("industry") or ""), [info.get("longName"), info.get("shortName")]
    except Exception:
        return t, "", "", []


def fetch_types(tickers, names=None):
    """{ticker: (sector, industry)}. If a dict is passed as `names`, it is filled with {ticker: [Yahoo names]}."""
    out = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for t, sec, ind, nm in ex.map(fetch_type, tickers):
            out[t] = (sec, ind)
            if names is not None:
                names[t] = nm
    return out


def load_name_map(path="name_map.csv"):
    """Optional name_map.csv (columns: ticker, screener_name) to fix names that do not match automatically."""
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, comment="#", dtype=str).fillna("")
    df.columns = [c.strip().lower() for c in df.columns]
    if "ticker" not in df or "screener_name" not in df:
        return {}
    return {r["ticker"].strip().upper(): r["screener_name"].strip() for _, r in df.iterrows() if r["ticker"].strip() and r["screener_name"].strip()}


def classify_type(sector, industry, override=None):
    if override in ("lender", "insurer", "cyclical", "normal"):
        return override
    ind = industry.lower()
    if sector == "Financial Services":
        if "insurance" in ind:
            return "insurer"
        if "asset management" in ind or "capital markets" in ind or "exchange" in ind:
            return "normal"
        return "lender"
    if any(w in ind for w in CYCLICAL_WORDS):
        return "cyclical"
    return "normal"


# ============================================================
# PEERS
# ============================================================
def load_peers(path):
    """peers.csv columns: ticker, peers (semicolon separated tickers), type (optional: lender, insurer, cyclical, normal)."""
    if not os.path.exists(path):
        return {}, {}
    df = pd.read_csv(path, comment="#", dtype=str).fillna("")
    df.columns = [c.strip().lower() for c in df.columns]
    peers, types = {}, {}
    for _, r in df.iterrows():
        t = r.get("ticker", "").strip().upper()
        if not t:
            continue
        if not t.endswith(".NS"):
            t += ".NS"
        plist = [p.strip() for p in r.get("peers", "").split(";") if p.strip()]   # tickers, or company names as Screener spells them
        if plist:
            peers[t] = plist
        if r.get("type", "").strip():
            types[t] = r["type"].strip().lower()
    return peers, types


def valid_pe(x):
    return x is not None and not np.isnan(x) and 0 < x <= PE_MAX_VALID


FULL_BY_NAME = None   # the whole export keyed by company name, so peers outside the watchlist can be named in peers.csv


def peer_pe(p, sc):
    """P/E of a peer given as a ticker (ICICIBANK or ICICIBANK.NS) or a Screener company name; None if not found."""
    tk = p.upper() if p.upper().endswith(".NS") else p.upper() + ".NS"
    if tk in sc.index:
        return float(sc.loc[tk, "pe"])
    if FULL_BY_NAME is not None:
        k = "NAME:" + si.norm_name(p)
        if k in FULL_BY_NAME.index:
            return float(FULL_BY_NAME.loc[k, "pe"])
    return None


def peer_median_pe(peer_list, sc):
    """Returns (median P/E or nan, valid count, peers not found in the export)."""
    vals, missing = [], []
    for p in peer_list:
        v = peer_pe(p, sc)
        if v is None:
            missing.append(p)
        elif valid_pe(v):
            vals.append(v)
    med = float(np.median(vals)) if len(vals) >= MIN_PEERS else np.nan
    return med, len(vals), missing


# ============================================================
# VALUATION
# ============================================================
LOW_ROE_NOTE = "ROE under 8%: weak returns, so the valuation ratio can mislead"
IND_PE_COUNTS = {}   # how many export rows share each Screener industry P/E; a value held by one stock is that stock itself


def value_stock(t, x, stype, peer_list, sc, lender_ref):
    r = {"ticker": t, "type": stype, "pe": x.get("pe", np.nan), "ev_ebitda": x.get("ev_ebitda", np.nan),
         "pb": x.get("pb", np.nan), "roe3": x.get("roe3", np.nan), "growth": x.get("profit_5y", np.nan),
         "notes": [], "peer_pe": np.nan, "peer_src": "-", "premium": np.nan, "peg": np.nan, "hist_prem": np.nan,
         "pb_per_roe": np.nan, "tag": "Not valued", "own_peers": bool(peer_list)}
    # own history
    for yrs in ("10", "5", "3"):
        h = x.get(f"pe_hist{yrs}", np.nan)
        if valid_pe(r["pe"]) and valid_pe(h):
            r["hist_prem"] = (r["pe"] / h - 1) * 100
            r["hist_yrs"] = yrs
            break
    if stype == "insurer":
        r["notes"].append("Insurer: not valued here")
        return r
    if stype == "lender":
        pb, roe = r["pb"], r["roe3"]
        if roe is not None and not np.isnan(roe) and roe < 8:
            r["notes"].append(LOW_ROE_NOTE)
        if pb and roe and not np.isnan(pb) and not np.isnan(roe) and roe > 0:
            r["pb_per_roe"] = pb / roe
            ref = lender_ref
            if ref and not np.isnan(ref):
                r["premium"] = (r["pb_per_roe"] / ref - 1) * 100
                r["peer_src"] = "lenders (P/B per ROE point)"
                r["tag"] = tag_from_premium(r["premium"])
            else:
                r["notes"].append("Too few lenders in the export to compare")
        else:
            r["notes"].append("P/B or ROE missing")
        return r
    if stype == "cyclical":
        r["notes"].append("Cyclical: P/E unreliable, judge on EV/EBITDA and P/B across the cycle")
        r["tag"] = "Cyclical (not tagged)"
        return r
    # normal companies
    if not valid_pe(r["pe"]):
        r["notes"].append("P/E missing, negative or above 100")
        return r
    if peer_list:
        r["peer_pe"], n, missing = peer_median_pe(peer_list, sc)
        r["peer_src"] = f"your peers ({n} valid)"
        if missing:
            r["notes"].append("Peers not found in the export: " + ", ".join(missing))
        if np.isnan(r["peer_pe"]):
            r["notes"].append("Your peer list has too few valid peers in the export")
    if np.isnan(r["peer_pe"]):
        ind = x.get("ind_pe", np.nan)
        if valid_pe(ind):
            gap = abs(r["pe"] / float(ind) - 1)
            alone = IND_PE_COUNTS.get(round(float(ind), 2), 2) <= 1
            if gap < 0.01 or (alone and gap < 0.03):
                r["notes"].append("Screener's industry P/E is essentially this stock's own (its group is just or mostly this company): "
                                  "no real comparison, add a peers.csv entry")
                return r
        if valid_pe(ind):
            r["peer_pe"], r["peer_src"] = float(ind), "Screener industry"
    if not np.isnan(r["peer_pe"]):
        r["premium"] = (r["pe"] / r["peer_pe"] - 1) * 100
        r["tag"] = tag_from_premium(r["premium"])
    g = r["growth"]
    if g is not None and not np.isnan(g) and g > GROWTH_LOW_BASE:
        r["notes"].append(f"5-year profit growth of {g:.0f}% a year comes from a low base: PEG not shown")
    elif g is not None and not np.isnan(g) and g >= GROWTH_MIN:
        r["peg"] = r["pe"] / min(g, GROWTH_CAP)
    elif g is not None and not np.isnan(g):
        r["notes"].append("Profit growth under 5% a year: PEG not meaningful")
    # modifiers
    if r["tag"] in ("Rich", "Stretched") and not np.isnan(r["peg"]) and r["peg"] <= PEG_GOOD:
        r["notes"].append("Premium backed by growth (PEG at or below 1)")
    if r["tag"] in ("Rich", "Stretched") and not np.isnan(r["peg"]) and r["peg"] > PEG_HIGH:
        r["notes"].append("Premium not backed by growth (PEG above 2.5)")
    low_base = g is not None and not np.isnan(g) and g > GROWTH_LOW_BASE
    if r["tag"] == "Cheap vs peers" and np.isnan(r["peg"]) and not low_base:
        r["notes"].append("Cheap but growth is low or unknown: possible value trap")
    if t in CONGLOMERATE_HINT and not peer_list:
        r["notes"].append("Conglomerate: industry P/E is a poor yardstick, add a peers.csv entry")
    if not np.isnan(r["premium"]) and abs(r["premium"]) > 100:
        r["notes"].append("Premium over 100% even against your peers: earnings look depressed, or the peer list is off"
                          if peer_list else "Premium over 100%: check the peer group")
    roe = r["roe3"]
    if roe is not None and not np.isnan(roe) and roe < 8:
        r["notes"].append(LOW_ROE_NOTE)
    if not np.isnan(r["hist_prem"]):
        if r["hist_prem"] >= HIST_RICH_PCT:
            r["notes"].append(f"{r['hist_prem']:+.0f}% above own {r['hist_yrs']}-year P/E")
        elif r["hist_prem"] <= HIST_CHEAP_PCT:
            r["notes"].append(f"{r['hist_prem']:+.0f}% below own {r['hist_yrs']}-year P/E")
    return r


def tag_from_premium(p):
    if p <= CHEAP_BELOW:
        return "Cheap vs peers"
    if p <= FAIR_UP_TO:
        return "Fair vs peers"
    if p <= RICH_UP_TO:
        return "Rich vs peers"
    return "Stretched vs peers"


def build_table(tickers, sc, types, peers, type_over, lender_ref_override=None):
    rows = []
    lender_vals = []
    for t in tickers:
        if t in sc.index and classify_type(*types.get(t, ("", "")), type_over.get(t)) == "lender":
            pb, roe = sc.loc[t].get("pb", np.nan), sc.loc[t].get("roe3", np.nan)
            if pb and roe and not np.isnan(pb) and not np.isnan(roe) and roe > 0:
                lender_vals.append(pb / roe)
    lender_ref = float(np.median(lender_vals)) if len(lender_vals) >= MIN_PEERS else np.nan
    if lender_ref_override is not None and not np.isnan(lender_ref_override):
        lender_ref = lender_ref_override    # universe run: the median over all lenders in the export, not just the candidates
    for t in tickers:
        if t not in sc.index:
            continue
        stype = classify_type(*types.get(t, ("", "")), type_over.get(t))
        rows.append(value_stock(t, sc.loc[t], stype, peers.get(t), sc, lender_ref))
    return pd.DataFrame(rows), lender_ref


# ============================================================
# EMAIL
# ============================================================
def f(x, nd=1, suffix="", plus=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:+.{nd}f}{suffix}" if plus else f"{x:.{nd}f}{suffix}"


def tbl(headers, rows_):
    th = "".join(f'<th style="text-align:left;padding:3px 6px;border-bottom:1px solid #999;">{h}</th>' for h in headers)
    body = "".join("<tr>" + "".join(f'<td style="padding:3px 6px;border-bottom:1px solid #eee;">{c}</td>' for c in row) + "</tr>"
                   for row in rows_)
    return f'<table style="border-collapse:collapse;font-size:12px;"><tr>{th}</tr>{body}</table>'


HEAD = ["Ticker", "Type", "P/E", "Peer P/E", "Peer source", "Premium", "5y profit growth", "PEG", "EV/EBITDA", "P/B",
        "ROE 3y", "vs own history", "Notes"]


def rows_html(sub):
    out = []
    for _, r in sub.iterrows():
        prem = r["premium"]
        out.append([r["ticker"].replace(".NS", ""), r["type"], f(r["pe"], 1), f(r["peer_pe"], 1), r["peer_src"],
                    f(prem, 0, "%", True), f(r["growth"], 0, "%"), f(r["peg"], 1), f(r["ev_ebitda"], 1), f(r["pb"], 1),
                    f(r["roe3"], 1, "%"), f(r["hist_prem"], 0, "%", True), "; ".join(r["notes"]) or "-"])
    return out


def build_email_html(df, n_watch, sc_info, lender_ref, has_hist):
    order = ["Cheap vs peers", "Fair vs peers", "Rich vs peers", "Stretched vs peers", "Cyclical (not tagged)", "Not valued"]
    counts = {k: int((df["tag"] == k).sum()) for k in order}
    h = f"""<html><body style="font-family:Arial,sans-serif;">
<h2>Valuation Tracker, {datetime.now().strftime('%d %b %Y')}</h2>
<p style="font-size:13px;">{len(df)} of {n_watch} watchlist stocks found in the Screener export ({sc_info}).
<b>{counts['Cheap vs peers']}</b> cheap, <b>{counts['Fair vs peers']}</b> fair, <b>{counts['Rich vs peers']}</b> rich,
<b>{counts['Stretched vs peers']}</b> stretched against peers; {counts['Cyclical (not tagged)']} cyclicals and {counts['Not valued']} not valued.
This says how much you pay for what a company earns. It is not a buy or sell signal.</p>"""
    if not has_hist:
        h += ('<p style="font-size:12px;color:#a00;">The export has no historical P/E column, so the comparison with each stock\'s own history is not shown. '
              "Add one (3, 5 or 10 year) in Screener's column picker if it is offered.</p>")
    ascending = {"Cheap vs peers": True, "Fair vs peers": True, "Rich vs peers": False, "Stretched vs peers": False}
    for k in order:
        sub = df[df["tag"] == k]
        if sub.empty and k in ("Cyclical (not tagged)", "Not valued"):
            continue
        sub = sub.sort_values("premium" if k in ascending else "ticker", ascending=ascending.get(k, True))
        h += f"<h3>{k} ({len(sub)})</h3>"
        h += tbl(HEAD, rows_html(sub)) if len(sub) else "<p>None.</p>"
    chk = df[df.apply(lambda r: (not r["own_peers"] and any(("poor yardstick" in x) or ("check the peer group" in x) or ("no real comparison" in x) for x in r["notes"]))
                      or any("too few valid peers" in x for x in r["notes"]), axis=1)]
    h += f"<h3>Peer groups to check ({len(chk)})</h3>"
    h += ('<p style="font-size:12px;">No usable peer list yet, and the industry P/E is probably the wrong yardstick (conglomerates, extreme premiums, or the stock is its own industry). '
          "Add or fix an entry in peers.csv: "
          + (", ".join(sorted(t.replace(".NS", "") for t in chk["ticker"])) if len(chk) else "none") + ".</p>")
    ext = df[df.apply(lambda r: r["own_peers"] and not np.isnan(r["premium"]) and abs(r["premium"]) > 100, axis=1)]
    if len(ext):
        h += ('<p style="font-size:12px;">Still more than double their peers, with your peer list in place (look at earnings quality and EV/EBITDA before trusting the P/E): '
              + ", ".join(sorted(t.replace(".NS", "") for t in ext["ticker"])) + ".</p>")
    h += f"""
<h3>How to read this</h3>
<table style="border-collapse:collapse;font-size:12px;">
<tr><td style="padding:3px 8px;"><b>Peer P/E</b></td><td style="padding:3px 8px;">Median P/E of the peers you list in peers.csv (peers with a negative P/E or above {PE_MAX_VALID:.0f} are ignored, at least {MIN_PEERS} needed), otherwise Screener's industry P/E.</td></tr>
<tr><td style="padding:3px 8px;"><b>Premium</b></td><td style="padding:3px 8px;">P/E against the peer P/E. Cheap: {CHEAP_BELOW:.0f}% or lower. Fair: up to +{FAIR_UP_TO:.0f}%. Rich: up to +{RICH_UP_TO:.0f}%. Stretched: above that.</td></tr>
<tr><td style="padding:3px 8px;"><b>PEG</b></td><td style="padding:3px 8px;">P/E divided by 5-year profit growth (capped at {GROWTH_CAP:.0f}%; not shown below {GROWTH_MIN:.0f}% or above {GROWTH_LOW_BASE:.0f}%, where the growth comes from a tiny base). At or below {PEG_GOOD:.0f} growth backs the price; above {PEG_HIGH} it does not.</td></tr>
<tr><td style="padding:3px 8px;"><b>Lenders</b></td><td style="padding:3px 8px;">Banks and NBFCs are compared on P/B divided by ROE (a lender earning more on its equity can justify a higher P/B), against the median of lenders in the export{'' if np.isnan(lender_ref) else f' ({lender_ref:.2f})'}.</td></tr>
<tr><td style="padding:3px 8px;"><b>Cyclicals</b></td><td style="padding:3px 8px;">Steel, cement, oil, metals, chemicals and similar: P/E looks lowest at peak profits, so they are not tagged. Judge them on EV/EBITDA and P/B over the cycle.</td></tr>
</table>
<p style="font-size:12px;color:#a00;"><b>Limits:</b> valuation is a weak guide to the next 1 to 3 months. A premium can be earned (growth, quality) or a bubble, and a discount can be a bargain or a trap. Industry P/E from Screener is a rough yardstick: check the peer group for anything that matters.
Screener's growth and P/E figures are as of the export date. Types (lender, cyclical) come from Yahoo and can be overridden in peers.csv.</p>
</body></html>"""
    return h


# ============================================================
# MAIN
# ============================================================
def notify_problem(reason):
    """Say why there is no valuation table, instead of finishing silently."""
    print("PROBLEM:", reason)
    html = (f'<html><body style="font-family:Arial,sans-serif;"><h2>Valuation Tracker, {datetime.now().strftime("%d %b %Y")}</h2>'
            f'<p style="color:#a00;">No valuation table this time. {reason}</p>'
            '<p style="font-size:12px;">To fix: export your stocks from Screener.in as Excel (.xlsx) with the columns P/E, Ind PE, EV / EBITDA, CMP / BV, ROE 3Yr and '
            'Profit Var 5Yrs %, save it as screener_export.xlsx in the repo root (next to watchlist.txt), commit it, and run the workflow again.</p></body></html>')
    try:
        m.send_email(f"Valuation Tracker, {datetime.now().strftime('%d %b %Y')} [NO DATA]", html)
    except Exception as e:
        print("Could not send the problem email:", e)


def main():
    global FULL_BY_NAME
    tickers = m.load_tickers(STOCKS_FILE)
    if MAX_TICKERS:
        tickers = tickers[:MAX_TICKERS]
    print(f"Loaded {len(tickers)} tickers from {STOCKS_FILE}")
    sc, info = si.load_screener(require=("pe",), allow_names=True)
    print("Screener.in export:", info)
    if sc.empty:
        print("No usable Screener export:", info)
        notify_problem(f"The Screener.in export could not be used: {info}")
        return
    if "ind_pe" in sc.columns:
        IND_PE_COUNTS.update(sc["ind_pe"].dropna().round(2).value_counts().to_dict())
    peers, type_over = load_peers(PEERS_FILE)
    print(f"peers.csv: {len(peers)} peer groups, {len(type_over)} type overrides")
    types = None
    lender_override = None
    if m.UNIVERSE_MODE and os.path.exists("results/universe_candidates.csv"):
        # Universe run: stage 2 already matched stocks to the export by NSE name and price. Peers named in peers.csv still
        # resolve against the whole export; stocks without a peer list use Screener's industry P/E.
        FULL_BY_NAME = sc
        sc = pd.read_csv("results/universe_candidates.csv").set_index("ticker")
        lv = sc[sc["lender"].astype(str) == "True"]
        lv = (lv["pb"] / lv["roe3"])[(lv["roe3"] > 0) & lv["pb"].notna()]
        lender_override = float(lv.median()) if len(lv) >= MIN_PEERS else None
        info = f"{len(sc)} stocks matched to the Screener export in stage 2 (NSE name and price); universe candidates"
    elif sc.attrs.get("by_name"):
        FULL_BY_NAME = sc
        # The export has no links, so match company names to tickers using Yahoo's names.
        names = {}
        types = fetch_types(tickers, names)
        sc, rep_ = si.match_by_name(sc, names, load_name_map())
        print(f"Matched by name: {rep_['exact']} exact, {len(rep_['fuzzy'])} close, {len(rep_['unmatched'])} not found")
        for t, nm, sc_ in rep_["fuzzy"]:
            print(f"  close match: {t} -> '{nm}' ({sc_})")
        print("  not found:", ", ".join(rep_["unmatched"]))
        info += (f"; matched by company name: {rep_['exact']} exact, {len(rep_['fuzzy'])} close"
                 + (" (" + "; ".join(f"{t.replace('.NS','')}={nm}" for t, nm, _ in rep_["fuzzy"][:15]) + ")" if rep_["fuzzy"] else "")
                 + (f", {len(rep_['conflicts'])} left out because two stocks claimed the same row ("
                    + ", ".join(t.replace(".NS", "") for t, _ in rep_["conflicts"]) + ": add to name_map.csv)" if rep_["conflicts"] else "")
                 + f", {len(rep_['unmatched'])} not found"
                 + (" (" + ", ".join(t.replace('.NS', '') for t in rep_["unmatched"][:25]) + ")" if rep_["unmatched"] else ""))
        if sc.empty:
            notify_problem(f"The export has no company links, and no watchlist stock could be matched by name. {info}")
            return
    known = [t for t in tickers if t in sc.index]
    print(f"{len(known)} of {len(tickers)} watchlist stocks found in the export")
    if not known:
        notify_problem(f"The export ({info}) was read, but none of your {len(tickers)} watchlist stocks were found in it. "
                       "Matching uses the symbol inside each company's link, so the export must be the .xlsx file with the company links intact.")
        return
    types = {t: types[t] for t in known} if types else fetch_types(known)
    df, lender_ref = build_table(known, sc, types, peers, type_over, lender_override)
    if df.empty:
        print("Nothing to value.")
        notify_problem("The export was read, but no stock could be valued.")
        return
    has_hist = any(c.startswith("pe_hist") for c in sc.columns)
    print(df["tag"].value_counts().to_string())
    df_out = df.assign(notes=df["notes"].map(lambda n: "; ".join(n) if isinstance(n, (list, tuple)) else (n or "")))   # notes saved as text
    if not m.UNIVERSE_MODE:
        df_out.to_csv("valuation_scores.csv", index=False)
    # Saved in the repo for the weekly combined shortlist (combined_shortlist.py)
    os.makedirs("results", exist_ok=True)
    df_out.assign(run_date=datetime.now().strftime("%Y-%m-%d")).to_csv(f"results/{m.RESULTS_PREFIX}valuation.csv", index=False)
    html = build_email_html(df, len(tickers), info, lender_ref, has_hist)
    m.send_email(f"Valuation Tracker, {datetime.now().strftime('%d %b %Y')}", html)


if __name__ == "__main__":
    main()
