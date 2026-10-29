# ForecastEx (IBKR) setup checklist: edge E5 (2026-10-28)

**Status:** sealed FX1 = PASS-UNDERPOWERED (219 bets, ROI +6.9% at last trades). No real money is authorized.
The forward paper log (FX1-fwd) is already running with no account needed.

## What is already built (nothing needs an account)
- **Paper logger:** `scripts/forecastex_paper.py`.
  - launchd label `com.jonathanwang.quant-forecastex-paper`, ticks every 5 min.
  - Logs every rule pick at 22:00 station-local, with the live public IBKR quote (bid/ask, no size).
  - Settles from the forecastex.com daily files.
  - Report: `cd ~/.claude/quant-kalshi && .venv/bin/python scripts/forecastex_paper.py report`
- **Venue client:** `scripts/venues/forecastex_client.py` (IBKR Web API).
  - Reads quotes publicly today.
  - Trading is gated behind all of: a session, an account id, `--live`, and paper mode by default.
- **First look (10-28 validate):** most NO picks sit on high strikes that have a NO *bid* (0.90-0.97) but **no NO ask**.
  - Expect the at-ask variant to be thin.
  - The maker-proxy variant is likely where this edge lives. Let the log decide.

## Account steps for you
1. **Eligibility (CA):** check before funding. UNVERIFIED by me.
   - Start the application at ndcdyn.interactivebrokers.com/Universal/Application?eventContract=T, or the normal IBKR Pro application.
   - Confirm that ForecastTrader / ForecastEx is offered to a California resident. ForecastEx is a CFTC-regulated DCM, but IBKR controls per-state availability.
   - If CA is excluded, stop here. The paper log still runs.
2. **Account type:** IBKR Pro, individual, cash account is fine. Event contracts are fully paid: you pay the full premium, with no margin.
3. **Trading permissions:** Settings → Account Settings → Trading Permissions → **Event Contracts / Forecast Contracts (ForecastEx)**, US.
   - You will have to acknowledge a risk disclosure.
   - Approval can take a day or so.
4. **Paper account:** Settings → Paper Trading Account → create or reset it. Note the paper id (it starts with **DU**).
   - Also enable event-contract permission on the paper account.
   - UNVERIFIED that ForecastEx is tradable in paper; if it is not, we stay on the public-quote log.
5. **Market data:** ForecastEx quotes are free, but accept the market-data subscriber agreement (non-professional) so API snapshots return data.
6. **API session (Client Portal Gateway, the Web API this client uses):**
   - Download the "Client Portal Gateway" (Java) from the IBKR Web API docs, unzip it, and run `bin/run.sh root/conf.yaml`.
   - Open https://localhost:5000 and log in with the **paper** username plus 2FA.
   - The session lasts about 24 h and needs a daily re-login.
7. **Config:** add these lines to `~/.claude/quant-kalshi/.env` (never commit it):
   ```
   IBKR_ACCOUNT_ID=DU…        # paper id
   IBKR_MODE=paper            # default; live needs IBKR_MODE=live + FX_ALLOW_LIVE_ACCOUNT=1 + a non-DU id
   FX_MAX_ORDER_USD=5  FX_MAX_CONTRACTS=10  FX_MAX_DAILY_USD=25
   ```
8. **Check (read-only):** `scripts/venues/forecastex_client.py preflight` should show the session as AUTHENTICATED.
   - Then run `book UHLGA_<MMDDYY>_<X>`. With a session, the book shows sizes.

## Before any real order (decision, not setup)
- At least 100 settled FX1-fwd A-picks, with the read-out rule from the `forecastex_paper.py` docstring.
- The user's explicit yes.
- A pre-registered order rule: maker bid vs ask, and size.
- IBKR order prompts are not auto-confirmed unless `FX_AUTO_CONFIRM=1`.

## Connection attempt 2026-10-28 (lead session): SSO works, brokerage session refused
- Gateway: ~/.claude/quant-kalshi/ibkr/ (start_gateway.sh). Port **5050**: macOS AirPlay owns 5000. IP allowlist
  trimmed to 127.0.0.1. .env has IBKR_API_BASE=https://localhost:5050/v1/api.
- Login: ibkr/login.py, Keychain item `fund-ibkr`, paper switch ON. The SMS 2FA code is read from Messages (user allowed it).
- Result: `sso/validate` RESULT=true, but `iserver/auth/ssodh/init` -> 500 `invalid challenge, machine id f0d366fe`
  on every attempt, `/reauthenticate` "triggered" with no effect, and auth/status stays authenticated=false.
- REFUTED: the Java version. Same error on Temurin-less Homebrew openjdk 27 and on openjdk@17.
- Open (account-side, unverified): is the account funded (Web API needs approved + funded)? Does the paper account
  exist and have ForecastEx/API permission? Next: user checks the portal; else email api@ibkr.com with the error line.
- 20:58 (after the user funded): the paper login repeats the same ssodh `invalid challenge`. A live-switch diagnostic
  (login.py --live-session) did NOT test anything: the gateway kept the existing paper SSO (AUTH_TIME 20:58,
  LOGIN_TYPE 2), so it is inconclusive. Each attempt costs the user an SMS, so stopped. Next options: (a) the user asks
  IBKR / checks that the paper account + API access exist and the deposit has cleared; (b) switch to IB Gateway (TWS API,
  ib_async, port 4002), which does not use ssodh and is the usual retail route.
- RESOLVED 21:22: the paper login was the cause. After `/logout`, a LIVE-switch login (`login.py --live-session`)
  gave ssodh/init 200 and AUTHENTICATED. The session sees one live individual cash account (STKCASH) and NO paper account.
  forecastex_client `preflight` = AUTHENTICATED, mode paper, account unset, live False, so trading is blocked as designed.
  `book UHBKF_102926_60` -> bid 0.91 x 10 (sizes now visible). Daily re-login: `start_gateway.sh` then
  `.venv/bin/python ibkr/login.py --live-session` (SMS read from Messages). For paper ORDERS the user must create the
  paper account (Settings -> Paper Trading Account) and then log in without --live-session.
