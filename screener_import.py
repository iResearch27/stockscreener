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


def load_screener(path=None, require=("ni_yoy", "rev_yoy")):
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
        for row in ws.iter_rows(min_row=2):
            cell = row[name_i]
            link = cell.hyperlink.target if cell.hyperlink else None
            m = re.search(r"/company/([^/]+)/", link or "")
            if not m or m.group(1).isdigit():
                continue
            rec = {"name": cell.value}
            for f, i in col.items():
                v = row[i].value
                if f == "result_date":
                    rec[f] = pd.to_datetime(v, errors="coerce", dayfirst=True)
                else:
                    rec[f] = float(v) if isinstance(v, (int, float)) else np.nan
            out[m.group(1).upper() + ".NS"] = rec
        df = pd.DataFrame.from_dict(out, orient="index")
        modified = wb.properties.modified
        info = f"{len(df)} companies from {os.path.basename(path)}" + (f", file dated {modified:%d %b %Y}" if modified else "")
        missing = [f for f in require if f not in col]
        if missing:
            return pd.DataFrame(), f"{path}: columns not found for {missing}. Check the column headers in the export."
        return df, info
    except Exception as e:
        return pd.DataFrame(), f"Could not read {path}: {e}"
