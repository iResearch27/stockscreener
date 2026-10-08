# Stock Trackers: Logic and Methodology

Reference document for the trackers built in this repo (`iResearch27/stockscreener`).
It records what each tracker measures, the exact rules and parameters, how it runs,
and the known limitations, so the logic can be rebuilt or tuned later.

**Contents**
1. Overview of the trackers
2. Turnaround screener: Early and Confirmed scoring
3. Confirmed Uptrend tracker
4. TradingView Pine Script dashboard
5. Automation and schedules
6. Repo layout and operating notes
7. Known limitations
8. Ideas parked for later

---

## 1. Overview

| Tracker | Purpose | Where it runs | Output |
|---|---|---|---|
| Turnaround screener (`screener.py`) | Catch stocks turning up from a downtrend, ideally before the obvious crossovers | GitHub Actions | Email table |
| Confirmed Uptrend tracker (`uptrend_tracker.py`) | Find stocks already in a sustained, strong uptrend | GitHub Actions | Email table |
| Pine Script dashboard (TradingView) | Same turnaround logic as a live on-chart table, up to 40 stocks | TradingView | Table + optional alert |
| Keep Repo Active (`keepalive.yml`) | Stops GitHub pausing scheduled workflows after 60 days of no commits | GitHub Actions | A monthly commit |

**Core motivation:** a stock that has already crossed its SMA50/SMA200 has usually
made part of its move, which leaves less room for return. The turnaround screener
therefore scores *leading* signals (momentum stabilising while price is still below
the averages) separately from *lagging* confirmation signals.

---

## 2. Turnaround screener

Every stock gets two scores from 0 to 6. All checks use daily candles and the latest
bar ("today"). Indicators: SMA 20/50/200; RSI(14) with Wilder smoothing; MACD(12, 26, 9);
On-Balance Volume (OBV).

### 2.1 Confirmed score (lagging: has the turn already happened?)

| # | Condition | Rule |
|---|---|---|
| C1 | Price above SMA50 | Close > SMA50 |
| C2 | Price above SMA200 | Close > SMA200 |
| C3 | Golden cross | SMA50 crossed above SMA200 today, or SMA50 > SMA200 now and SMA50 was at or below SMA200 five bars ago |
| C4 | RSI reclaimed 50 | RSI crossed above 50 today, or RSI > 50 now and RSI three bars ago was at or below 50 |
| C5 | MACD bullish cross | MACD line crossed above the signal line today |
| C6 | OBV rising | OBV now > OBV 10 bars ago |

### 2.2 Early score (leading: is a turn forming?)

| # | Condition | Rule |
|---|---|---|
| E1 | MACD histogram improving while negative | Histogram (MACD minus signal) > histogram 3 bars ago, and histogram < 0 |
| E2 | RSI recovering from oversold | RSI now > RSI 5 bars ago, and RSI 5 bars ago < 35 |
| E3 | Higher swing low | Lowest close of the last 10 bars > lowest close of the 10 bars ending 10 bars ago |
| E4 | OBV bullish divergence | OBV now > OBV 15 bars ago, while close now < close 15 bars ago |
| E5 | Nearing SMA50 from below | 0 < (SMA50 minus close) / close x 100 < 5 (price below SMA50 but within 5%) |
| E6 | Reclaimed the 20-day average | Close crossed above SMA20 today |

The email also shows **% to SMA50**: `(SMA50 - close) / close x 100`. A positive number means
price is still below SMA50 by that much; a negative number means it is already above.

### 2.3 How Early and Confirmed relate

They are not a strict inverse. Think of it as a handoff over time: Early signals tend to
switch on first while price is still below the averages, then fade as Confirmed signals
switch on.

- **Mutually exclusive pairs** (cannot both be true on the same day): C1 vs E5 (price is either
  above SMA50 or below it), and C5 vs E1 (at the MACD cross the histogram turns positive).
- **Can overlap:** C4 with E2, C6 with E4, and E6 with C1/C3.
- There is no fixed total. A stock can score 0/0 (no signal), high on one side, or moderate on both.

### 2.4 Flagging and output

- A stock is **flagged** if Confirmed >= 4 **or** Early >= 4 (`MIN_CONFIRMED_SCORE`, `MIN_EARLY_SCORE` at the top of `screener.py`).
- Email: a "Flagged" table (sorted by Early, then Confirmed), then the full list.
- Columns: Ticker, Price, RSI, % to SMA50, Early/6, Confirmed/6. Scores are plain integers
  (a "4/6" style value was auto-converted to a date when pasted into Excel).
- Subject: `Turnaround Screener - <date> (<N> scanned)`.

### 2.5 Data handling

- Source: `yfinance` (Yahoo Finance), 2 years of daily bars, split/dividend adjusted.
- Tickers use Yahoo format with the `.NS` suffix for NSE (e.g. `RELIANCE.NS`), listed in `stocks.txt`.
- Downloaded in batches of 50 with one retry after a 5-second wait and a 1-second pause between batches.
- A ticker with fewer than 210 rows of history is skipped and logged as "Skipping ...".

---

## 3. Confirmed Uptrend tracker

Looks for strength and durability rather than the transition. Each stock scores 0 to 6.
The benchmark is the Nifty 50 (`^NSEI`).

| # | Condition | Rule |
|---|---|---|
| U1 | Near 52-week high | Within 10% of the highest close of the last 252 trading days |
| U2 | Aligned bullish structure | Close > SMA50 and SMA50 > SMA200 |
| U3 | Sustained above SMA50 | Every one of the last 20 closes was above that day's SMA50 |
| U4 | Higher highs and higher lows | 20-day high now > 20-day high 20 bars ago, and 20-day low now > 20-day low 20 bars ago |
| U5 | Outperforming Nifty | Stock's 63-day (about 3 months) return > Nifty's 63-day return |
| U6 | Sustained momentum | Average RSI(14) over the last 10 days > 50 |

- **Flagged** if score >= 4 (`MIN_UPTREND_SCORE`). Sorted by score, then by "vs Nifty (3M)".
- Email columns: Ticker, Price, % From 52wk High, 3M Return, vs Nifty (3M), Avg RSI (10d), Uptrend/6.
  The Nifty 3-month return is shown at the top. "vs Nifty (3M)" is the most informative column to scan first.
- Watchlist is separate from the turnaround screener: `uptrend_watchlist.txt` (same ticker format).
- A ticker with fewer than 260 rows of history is skipped (it needs about a year for the 52-week high and SMA200).
- Tunable constants at the top of `uptrend_tracker.py`: `NEAR_52WK_PCT`, `SUSTAINED_DAYS`, `SWING_LOOKBACK`, `RS_LOOKBACK`, `RSI_SUSTAIN_DAYS`.
- Bug fixed during setup: newer `yfinance` versions return multi-level column names even for one ticker,
  which made the Nifty return calculation fail. The script now flattens the columns first.

---

## 4. TradingView Pine Script dashboard

Pine Script v6 indicator ("Multi-Stock Turnaround Screener") using the same 12 conditions and the
same parameters as section 2.

- **Stock list:** one text box in the indicator settings, one symbol per line (e.g. `NSE:AXISBANK`).
  Adding a stock means adding a line there; the table resizes itself.
- **Hard cap of 40 symbols**, because TradingView limits `request.security()` calls to 40 per script.
  Extra symbols are dropped and a red warning row appears in the table.
- **Table:** Symbol, Price, RSI, % to SMA50, Early/6, Confirmed/6. Hover over a score cell to see
  which individual conditions passed (Y/N). Colour coding: 5-6 solid green, 4 light green, 3 orange, below 3 red.
- **Alerts:** one alert, not one per stock. The script builds a message such as
  `EARLY: <symbols> | CONFIRMED: <symbols>` and sends it through `alert()`. To use it, create one alert on the
  indicator with the condition **Any alert() function call** and frequency **Once Per Bar Close**.
- **Inputs (Settings):** minimum Early/Confirmed score to flag (default 4), OBV lookbacks (10 and 15),
  swing-low lookback (10), "near SMA50" threshold (5%).
- **Limit:** per-stock named alerts are not possible with a dynamic list, which is why the single dynamic alert is used.
- **Pine vs Python:** the rules are the same, but TradingView uses its own price data and built-in indicator
  functions, while Python uses adjusted Yahoo data and its own implementations. Small differences in values are normal.

---

## 5. Automation and schedules

Times are IST; GitHub schedules use UTC (IST = UTC + 5:30, no daylight saving). GitHub's scheduler is
best-effort and can run a few minutes late.

| Workflow file | Runs | Cron (UTC) |
|---|---|---|
| `daily_screener.yml` | Turnaround screener, Mon-Fri. Designed as three runs: 9:30 AM, 1:30 PM, 4:00 PM IST | `0 4`, `0 8`, `30 10` (`* * 1-5`) |
| `uptrend_tracker.yml` | Uptrend tracker, Mon-Fri 4:00 PM IST | `30 10 * * 1-5` |
| `keepalive.yml` | Monthly commit, 1st of each month 3:00 AM UTC | `0 3 1 * *` |

Check the `cron:` lines inside `.github/workflows/daily_screener.yml` to see which schedule is live.
All workflows can also be run by hand from the Actions tab (**Run workflow**).

**Daily bars caveat:** both trackers use daily candles, which only finalise after NSE closes (3:30 PM IST),
and Yahoo can lag after the close. Runs at 9:30 AM and 1:30 PM therefore show the previous close,
so only the 4:00 PM run can carry new information.
Switching to intraday bars was considered and rejected: the SMA/RSI/MACD periods count bars, so on
15-minute bars the "SMA200" would cover about 8 trading days instead of about 10 months, which changes what the signals mean.
Yahoo also only provides about 60 days of intraday history.

**Email delivery:** Gmail SMTP using an App Password. Three repository secrets are required
(Settings -> Secrets and variables -> Actions):

| Secret | Value |
|---|---|
| `GMAIL_USER` | Gmail address that sends the email |
| `GMAIL_APP_PASSWORD` | 16-character Google App Password (needs 2-Step Verification on) |
| `ALERT_RECIPIENT` | Address that receives the email (can be the same as the sender) |

**Cost:** GitHub Actions is free for public repos (unlimited minutes) and includes 2,000 free minutes per month
for private repos. Each run takes a few minutes, so usage stays far below that.

---

## 6. Repo layout and operating notes

```
stockscreener/
  screener.py               turnaround screener
  stocks.txt                watchlist for the turnaround screener
  uptrend_tracker.py        confirmed uptrend tracker
  uptrend_watchlist.txt     watchlist for the uptrend tracker
  requirements.txt          yfinance, pandas, numpy
  README.txt                setup steps
  METHODOLOGY.md            this document
  .github/workflows/
    daily_screener.yml
    uptrend_tracker.yml
    keepalive.yml
```

**Adding stocks:** edit `stocks.txt` or `uptrend_watchlist.txt`, one Yahoo-format ticker per line.
Lines starting with `#` and blank lines are ignored. Save as plain text.

**Changing thresholds:** edit the constants at the top of the relevant `.py` file and commit.

**Uploading files through the GitHub web page:**
- The uploader places files in whichever folder you opened it from. Workflow files must end up in
  `.github/workflows/`, so open that folder first, then use **Add file -> Upload files**.
- A file can also be moved by opening it, clicking the pencil icon and editing the path in the filename box.

**Problems already hit and their fixes:**
- Workflow not listed in the Actions sidebar: the file had landed in `.github/.github/workflows/`
  (nested folder), then had a stray trailing period in its name. Workflows must live in exactly
  `.github/workflows/` and end in `.yml` or `.yaml`.
- Script crashed with "float() argument must be a string or a real number, not 'Series'":
  the multi-level column issue described in section 3.
- Pine Script rejected a `\` line continuation: Pine does not support it, so long expressions stay on one line.
- Pine tooltip showed garbled characters: check/cross symbols did not render, so plain Y/N is used.
- Run log notices about Node.js 20 and about `ubuntu-latest` moving to Ubuntu 26 (announced for October 19, 2026)
  are informational. If a workflow ever breaks after such a change, update the versions of
  `actions/checkout` and `actions/setup-python` to the latest major release, or pin `runs-on` to an
  explicit Ubuntu version.

**Diagnosing a failed run:** Actions tab -> the failed run -> click the red step -> read the last lines of the log.
Look for lines starting `Traceback`, and for `Skipping` lines that name dropped tickers.

---

## 7. Known limitations

- **Leading signals are noisy.** A high Early score does not guarantee a turnaround; some stocks will keep falling.
  Treat it as a prompt to look closer, and use Confirmed as the follow-through check.
- **Thresholds are untested.** The 4-out-of-6 cut-offs and all lookback windows are reasonable starting values,
  not backtested results.
- **Data source:** `yfinance` is free but unofficial. It can throttle large requests, return gaps for illiquid
  tickers, or lag after the close. Missing tickers show up as "Skipping" lines in the run log.
- **No market-holiday awareness.** On NSE holidays the workflows still run and email the last available close.
- **Technicals only.** Neither tracker looks at fundamentals, news, regulatory events or company-specific risks.
- **Equal weighting.** Every condition counts as one point; none is weighted as more important.
- This is a screening aid, not investment advice.

---

## 8. Ideas parked for later

- Extra conditions: relative strength vs sector index, volume surge vs 20-day average, 52-week-high breakout with volume.
- Market-regime filter using India VIX and FII/DII flows (needs a data source other than `yfinance`).
- Delivery percentage, bulk/block deal and shareholding-change data (NSE publishes these; they need separate parsing).
- Market-holiday skip, and an email only when something changed since the last run.
- A separate intraday tool with its own period settings, rather than reusing the daily logic.
- Backtesting the Early and Confirmed thresholds on historical data.
