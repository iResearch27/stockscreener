"""
Price-assisted matching of NSE tickers to rows of a Screener export that has no company links (used by universe_prefilter.py).

Two pieces of evidence per candidate pair: how alike the names are (NSE's official name vs Screener's abbreviated name) and
how close Screener's CMP is to the Yahoo price. Only Screener rows within 10% of the ticker's price are considered, which keeps the
search small and makes look-alike names harmless: a wrong company almost never has both a similar name and the same price.

Accepted when:
  name score >= 0.88 and price gap <= 10%        (same name; the gap allows for different price dates)
  name score >= 0.70 and price gap <= 5%
  name score >= 0.55 and price gap <= 2%, and no other row within 5% scores 0.55 or more   (name abbreviated heavily)
One Screener row goes to only one ticker: the better (score, price gap) wins, the other stays unmatched.
overrides ({ticker: Screener name}) come first and skip the name test, but still need the price to agree within 15%.
"""
import difflib
import re

import numpy as np
import pandas as pd

import screener_import as si

# Words that many companies share: a match that rests only on these ("Industries", "Capital") is not a name match
GENERIC = {"industries", "industry", "inds", "ind", "capital", "cap", "technologies", "technology", "tech", "finance", "financial", "fin",
           "enterprises", "enterprise", "infra", "infrastructure", "india", "indian", "international", "intl", "global", "services",
           "solutions", "holdings", "investments", "investment", "inv", "engineers", "engineering", "engg", "projects", "power",
           "energy", "pharma", "pharmaceuticals", "chemicals", "chem", "foods", "food", "textiles", "tex", "realty", "properties",
           "developers", "ventures", "trading", "exports", "corporation", "corpn", "company", "co", "and", "of", "the", "bank", "sugar",
           "steel", "steels", "metals", "plastics", "polymers", "paper", "papers", "cement", "motors", "auto", "electric", "electricals"}

TOL = {"same": 0.10, "good": 0.05, "tight": 0.02, "override": 0.15}


def _compact(x):
    return re.sub(r"[^a-z0-9]", "", str(x or "").lower())


def _shares_distinctive_word(sc_name, nse_name):
    """True unless both names have distinctive words and none of Screener's words abbreviates one of the NSE name's words."""
    at = [t for t in si._tokens(sc_name) if t not in GENERIC]
    yt = [t for t in si._tokens(nse_name) if t not in GENERIC]
    if not at or not yt:
        return True
    comp = "".join(yt)
    return any(any(si._abbrev(t, w) or t == w for w in yt) for t in at if len(t) >= 2) or (len("".join(at)) >= 3 and comp.startswith("".join(at)))


def name_score(sc_name, nse_name):
    a, y = si.norm_name(sc_name), si.norm_name(nse_name)
    if not a or not y:
        return 0.0
    if a == y:
        return 1.0
    if not _shares_distinctive_word(sc_name, nse_name):
        return min(0.5, difflib.SequenceMatcher(None, a, y).ratio())
    best = difflib.SequenceMatcher(None, a, y).ratio()
    ab = si._abbrev_score(sc_name, nse_name)
    if ab is not None:
        best = max(best, 0.9 * ab + 0.1 if ab >= 0.8 else 0.6)
    if len(a) >= 3 and y.startswith(a):
        best = max(best, 0.88 if len(a) >= 6 else 0.7)             # 'A B B' -> 'abb' of 'ABB India'; 'Allied Blenders' of '... and Distillers'
    else:
        at, yt = si._tokens(sc_name), si._tokens(nse_name)
        if at and len(yt) > len(at):
            ok = all(si._abbrev(t, yt[i]) for i, t in enumerate(at))   # Screener's tokens abbreviate the first NSE tokens
            if ok and (len(at) >= 2 or len(at[0]) >= 5):
                best = max(best, 0.7)
    return best


def match(sc, ticker_info, overrides=None):
    """sc: Screener frame from load_screener(allow_names=True) with 'name' and 'cmp'. ticker_info: {ticker: (nse_name, price)}.
    Returns (DataFrame of matched rows indexed by ticker, report)."""
    overrides = {t: si.norm_name(n) for t, n in (overrides or {}).items()}
    names = sc["name"].tolist()
    cmp_ = sc["cmp"].to_numpy(dtype=float)
    keys = list(sc.index)
    claims = {}                                                     # ticker -> (row index, score, gap, kind)
    unmatched, rejected = [], []
    for t, (nse_name, price) in ticker_info.items():
        if not price or not np.isfinite(price) or price <= 0:
            unmatched.append(t)
            continue
        gap = np.abs(cmp_ / price - 1)
        gap = np.where(np.isfinite(gap), gap, 9.9)
        ov = overrides.get(t)
        if ov and ("NAME:" + ov) in sc.index:
            i = keys.index("NAME:" + ov)
            if gap[i] <= TOL["override"]:
                claims[t] = (i, 1.0, float(gap[i]), "exact")
            else:
                rejected.append((t, names[i]))
            continue
        cand = np.where(gap <= TOL["same"])[0]
        scored = [(name_score(names[i], nse_name), float(gap[i]), i) for i in cand]
        scored.sort(key=lambda x: (-x[0], x[1]))
        pick = None
        for s, g, i in scored[:3]:
            tight_others = [x for x in scored if x[2] != i and x[0] >= 0.55 and x[1] <= 0.05]
            if s >= 0.88 and g <= TOL["same"]:
                pick = (i, s, g, "exact" if s >= 0.999 else "close")
            elif s >= 0.70 and g <= TOL["good"]:
                pick = (i, s, g, "close")
            elif s >= 0.55 and g <= TOL["tight"] and not tight_others:
                pick = (i, s, g, "close")
            if pick:
                break
        if pick:
            claims[t] = pick
        else:
            unmatched.append(t)
    by_row = {}
    for t, (i, s, g, kind) in claims.items():
        by_row.setdefault(i, []).append((s, -g, t))
    conflicts = []
    for i, lst in by_row.items():
        if len(lst) > 1:
            lst.sort(reverse=True)
            for _, _, t in lst[1:]:
                claims.pop(t)
                unmatched.append(t)
                conflicts.append((t, names[i]))
    rows = {t: sc.iloc[i] for t, (i, s, g, k) in claims.items()}
    out = pd.DataFrame.from_dict(rows, orient="index") if rows else pd.DataFrame()
    out.attrs["by_name"] = True
    fuzzy = [(t, names[i], round(s, 2)) for t, (i, s, g, k) in claims.items() if k == "close"]
    exact_n = sum(1 for v in claims.values() if v[3] == "exact")
    return out, {"exact": exact_n, "fuzzy": fuzzy, "unmatched": unmatched, "conflicts": conflicts,
                 "price_rejected": len(rejected), "rejected_list": rejected}
