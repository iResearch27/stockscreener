Multi-Stock Turnaround Screener (Python + GitHub Actions)
============================================================

Scans your stock list daily after market close and emails you a table of
Confirmed (lagging) and Early (leading) turnaround scores — the same logic
as the TradingView Pine script, without the 40-symbol cap.

1. Add your stocks
-------------------
Edit stocks.txt — one ticker per line, in Yahoo Finance format
(NSE stocks need a .NS suffix, e.g. RELIANCE.NS). Add as many as you like.

2. Get a Gmail App Password
----------------------------
Regular Gmail passwords won't work for SMTP. You need an App Password:
1. Turn on 2-Step Verification on the Gmail account you'll send from
   (Google Account -> Security -> 2-Step Verification).
2. Go to Google Account -> Security -> App Passwords.
3. Create one (name it anything, e.g. "screener"), and copy the 16-character password.

3. Add GitHub Secrets
----------------------
In your repo: Settings -> Secrets and variables -> Actions -> New repository secret.
Add these three:

  Secret name           Value
  ---------------------------------------------------------------
  GMAIL_USER            The Gmail address you're sending from
  GMAIL_APP_PASSWORD    The 16-character App Password from step 2
  ALERT_RECIPIENT       Where you want alerts sent (can be the same address)

4. Push this repo to GitHub
-----------------------------
Public or private both work — public repos get unlimited free Actions minutes;
private repos get 2,000 free minutes/month, which is far more than this needs.

5. Test it
----------
Go to the Actions tab -> "Daily Turnaround Screener" -> Run workflow
to trigger it manually and confirm the email arrives before waiting for
the schedule.

Notes
-----
- The workflow runs 4:00 PM IST, Monday-Friday. It doesn't currently skip
  market holidays — you'll just get an email with the last available close
  price on holidays. Let me know if you want holiday-awareness added.
- stocks.txt supports 200+ tickers; data is fetched in batches of 50 with
  basic retry logic to handle occasional Yahoo Finance hiccups.
- Thresholds (MIN_CONFIRMED_SCORE, MIN_EARLY_SCORE, lookback windows) are
  configurable at the top of screener.py.
