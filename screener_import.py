"""
Reads a manual Screener.in export (xlsx) so the Earnings tracker can use Screener's year-on-year quarterly growth
when Yahoo does not have a year-ago quarter, and cross-check Yahoo's numbers when it does.

How the export is made (once per results season, about two minutes):
  1. On screener.in, log in, open a screen that lists your stocks (or any broad screen) and use "Edit columns" to include:
       Qtr Profit Var %, Qtr Sales Var %, Sales Qtr Rs.Cr., NP Qtr Rs.Cr.
     (and, if the column picker offers them: quarterly operating profit growth, last result date).
  2. Click Export, choose Excel. Keep the .xlsx format: the company-name cells carry links that contain the NSE symbol,
     which is how stocks are matched. A CSV export has no links, so it cannot be matched reliably.
  3. Save it as screener_export.xlsx in the repo root (next to watchlist.txt) and commit it.

Companies are matched by the symbol in their Screener link (https://www.screener.in/company/ICICIBANK/consolidated/ ->
ICICIBANK.NS). Companies Screener lists only by a numeric BSE code are ignored.
"""

import os
import re

import numpy as np
import pandas as pd

SCREENER_FILE = os.environ.get("SCREENER_FILE", "screener_export.xlsx")

# Normalised header text -> field name. Headers are compared with spaces, dots and case removed.
COLUMN_MAP = {
    "qtrprofitvar%": "ni_yoy",
    "qtrsalesvar%": "rev_yoy",
    "salesqtrrs.cr.": "sales_cr",
    "npqtrrs.cr.": "np_cr",
    "roe3yr%": "roe3",
    "roce3yr%": "roce3",
    "roa3yr%": "roa3",
    "debt/eq": "de",
    "cmp/bv": "pb",
    "p/e": "pe",
    "ev/ebitda": "ev_ebitda",
    "indpe": "ind_pe",
    "marcaprs.cr.": "mcap_cr",
    "cmprs.": "cmp",
    "profitvar5yrs%": "profit_5y",
    "salesvar5yrs%": "sales_5y",
    "peg": "peg",
}


def _norm(h):
    return re.sub(r"[\s]+", "", str(h or "")).lower()


def _find_extra(headers):
    """Optional columns whose exact names vary: quarterly operating profit growth and last result date."""
    op, dt = None, None
    for h in headers:
        n = _norm(h)
        if op is None and ("oper" in n or n.startswith("op")) and ("qtr" in n or "quarter" in n) and ("var" in n or "gr" in n):
            op = h
        if dt is None and "result" in n and "date" in n and "next" not in n:
            dt = h
    return op, dt


def _find_hist_pe(headers):
    """Optional historical P/E columns (3, 5 or 10 year), if the export includes them."""
    found = {}
    for h in headers:
        n = _norm(h)
        if "hist" in n and "pe" in n:
            for yrs in ("10", "5", "3"):
                if yrs in n and yrs not in found:
                    found[yrs] = h
    return found


_SUFFIX = {"limited", "ltd", "the"}


def norm_name(x):
    """'Reliance Industries Ltd.' and 'RELIANCE INDUSTRIES LIMITED' -> 'relianceindustries'."""
    x = str(x or "").lower().replace("&", " and ")
    words = [w for w in re.sub(r"[^a-z0-9 ]+", " ", x).split() if w not in _SUFFIX]
    return "".join(words)


def _tokens(x):
    x = str(x or "").lower().replace("&", " and ")
    return [w for w in re.sub(r"[^a-z0-9]+", " ", x).split() if w not in _SUFFIX]


def _abbrev(a, full):
    """True if a is full, or a shortened form of it ('Inds' of 'Industries', 'Hind' of 'Hindustan')."""
    if a == full:
        return True
    if len(a) > len(full) or a[0] != full[0]:
        return False
    if len(a) < 3:
        return full.startswith(a)
    if a[:3] != full[:3]:                                          # shortened names keep the opening letters
        return False
    it = iter(full[1:])
    return all(ch in it for ch in a[1:])


def _abbrev_score(sc_name, yahoo_name):
    """Screener shortens names ('Hind. Unilever', 'Sun Pharma.Inds.', 'M & M'). Returns a rank (higher is better) or None."""
    a, y = _tokens(sc_name), _tokens(yahoo_name)
    if not a or not y:
        return None
    if len(a) == 1 and 2 <= len(a[0]) <= 5 and len(y) > 1 and a[0] == "".join(w[0] for w in y if w != "and"):
        return 0.5                                                  # initials, e.g. TCS
    j, skips = 0, 0
    for tok in a:
        while j < len(y) and not _abbrev(tok, y[j]):
            j += 1
            skips += 1
        if j >= len(y):
            return None
        j += 1
    skips += len(y) - j
    return None if skips > 1 else 1.0 - 0.2 * skips


def match_by_name(sc, ticker_names, overrides=None, cutoff=0.88):
    """For exports without links: attach NSE tickers by company name.
    ticker_names: {ticker: [names, e.g. Yahoo long and short name]}. overrides: {ticker: Screener name}.
    A Screener row is given to only one ticker: if two tickers claim the same row by a loose match, neither gets it.
    Returns (DataFrame indexed by ticker, report dict)."""
    import difflib
    by_key = {k: k[5:] for k in sc.index}                   # strip 'NAME:'
    overrides = {t: norm_name(n) for t, n in (overrides or {}).items()}
    claims = {}                                             # ticker -> (key, kind, score)
    unmatched = []
    for t, names in ticker_names.items():
        cands = [norm_name(n) for n in names if n]
        hit, kind, score = None, None, 1.0
        if t in overrides and "NAME:" + overrides[t] in sc.index:
            hit, kind = "NAME:" + overrides[t], "exact"
        else:
            for c in cands:
                if c and "NAME:" + c in sc.index:
                    hit, kind = "NAME:" + c, "exact"
                    break
        if hit is None and cands:
            best, bscore = None, 0.0
            for c in cands:
                for k, nm in by_key.items():
                    r = difflib.SequenceMatcher(None, c, nm).ratio()
                    if r > bscore:
                        best, bscore = k, r
            if best is not None and bscore >= cutoff:
                hit, kind, score = best, "close", round(bscore, 2)
        if hit is None and names:
            ranked = []
            for k in by_key:
                for yn in names:
                    r = _abbrev_score(sc.loc[k, "name"], yn)
                    if r is not None:
                        ranked.append((r, k))
                        break
            if ranked:
                top = max(r for r, _ in ranked)
                best = [k for r, k in ranked if r == top]
                if len(best) == 1:
                    hit, kind, score = best[0], "close", round(top, 2)
        if hit is None:
            unmatched.append(t)
        else:
            claims[t] = (hit, kind, score)
    # one Screener row, one ticker
    by_row = {}
    for t, (k, kind, sco) in claims.items():
        by_row.setdefault(k, []).append((t, kind))
    conflicts = []
    for k, lst in by_row.items():
        if len(lst) < 2:
            continue
        exact = [t for t, kind in lst if kind == "exact"]
        losers = [t for t, _ in lst if not (len(exact) == 1 and t == exact[0])]
        for t in losers:
            claims.pop(t)
            unmatched.append(t)
            conflicts.append((t, sc.loc[k, "name"]))
    rows = {t: sc.loc[k] for t, (k, _, _) in claims.items()}
    fuzzy = [(t, sc.loc[k, "name"], sco) for t, (k, kind, sco) in claims.items() if kind == "close"]
    exact_n = sum(1 for v in claims.values() if v[1] == "exact")
    out = pd.DataFrame.from_dict(rows, orient="index") if rows else pd.DataFrame()
    return out, {"exact": exact_n, "fuzzy": fuzzy, "unmatched": unmatched, "conflicts": conflicts}


def load_screener(path=None, require=("ni_yoy", "rev_yoy"), allow_names=False):
    """Returns (DataFrame indexed by NSE ticker such as 'ICICIBANK.NS', info string) or (empty DataFrame, reason)."""
    path = path or SCREENER_FILE
    if not os.path.exists(path):
        return pd.DataFrame(), f"{path} not found (Screener fallback not used)"
    try:
        import openpyxl
        wb = openpyxl.load_workbook(path)
        ws = wb.active
        headers = [c.value for c in ws[1]]
        col = {}
        for i, h in enumerate(headers):
            f = COLUMN_MAP.get(_norm(h))
            if f:
                col[f] = i
        for yrs, h in _find_hist_pe(headers).items():
            col[f"pe_hist{yrs}"] = headers.index(h)
        op_h, dt_h = _find_extra(headers)
        if op_h is not None:
            col["op_yoy"] = headers.index(op_h)
        if dt_h is not None:
            col["result_date"] = headers.index(dt_h)
        name_i = next((i for i, h in enumerate(headers) if _norm(h) == "company"), 1)
        out = {}
        out_names = {}
        n_rows = n_links = 0
        samples = []
        for row in ws.iter_rows(min_row=2):
            cell = row[name_i]
            link = cell.hyperlink.target if cell.hyperlink else None
            shown = cell.value
            if not link and isinstance(shown, str) and shown.startswith("="):
                # Some exports/spreadsheet apps keep the link as =HYPERLINK("url","name")
                fm = re.match(r'=\s*HYPERLINK\(\s*"([^"]+)"\s*(?:[,;]\s*"([^"]*)")?', shown, re.I)
                if fm:
                    link, shown = fm.group(1), (fm.group(2) or shown)
            n_rows += 1
            if link:
                n_links += 1
            elif len(samples) < 3:
                samples.append(repr(cell.value)[:40])
            m = re.search(r"/company/([^/]+)/", link or "")
            if allow_names and not link and shown:
                key, store = "NAME:" + norm_name(shown), out_names
            elif not m or m.group(1).isdigit():
                continue
            else:
                key, store = m.group(1).upper() + ".NS", out
            rec = {"name": shown}
            for f, i in col.items():
                v = row[i].value
                if f == "result_date":
                    rec[f] = pd.to_datetime(v, errors="coerce", dayfirst=True)
                else:
                    rec[f] = float(v) if isinstance(v, (int, float)) else np.nan
            store[key] = rec
        by_name = False
        if not out and out_names:
            out, by_name = out_names, True
        df = pd.DataFrame.from_dict(out, orient="index")
        df.attrs["by_name"] = by_name
        if df.empty:
            hdr = [str(h) for h in headers[:6]]
            return pd.DataFrame(), (f"{path}: {n_rows} data rows read, but only {n_links} company names carry a Screener link "
                                    f"(name column taken as '{headers[name_i]}'; first headers {hdr}; names without links e.g. {samples}). "
                                    "The file was probably re-saved in Excel/Numbers/Google Sheets, which can drop the links. "
                                    "Upload the file exactly as downloaded from Screener.in.")
        modified = wb.properties.modified
        info = f"{len(df)} companies from {os.path.basename(path)}" + (f", file dated {modified:%d %b %Y}" if modified else "")
        missing = [f for f in require if f not in col]
        if missing:
            return pd.DataFrame(), f"{path}: columns not found for {missing}. Check the column headers in the export."
        return df, info
    except Exception as e:
        return pd.DataFrame(), f"Could not read {path}: {e}"
