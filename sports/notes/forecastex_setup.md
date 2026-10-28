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
