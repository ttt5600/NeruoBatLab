# Go live: Kalshi temperature consensus-reject (orig7 + X1), unattended

Written 2026-10-21. The strategy rules are unchanged (sealed: orig7 +1.45%/1,844; X1 +0.78%/2,464). This file is
about the executor `scripts/kalshi_exec.py`, which turns the hourly paper picks into orders.
Claude never handles the key and never sets EXEC_MODE. You do every step marked YOU.

## How it decides (one screen)
- Hourly at :03 launchd runs `kalshi_paper.py tick` (picks each city 22:00-23:59 city-local), then
  `kalshi_exec.py run --live`. Mode = **dry** unless `~/.claude/quant-kalshi/.env` has `EXEC_MODE=live`;
  then **demo** if `EXEC_ENV=demo`, else **live** (real money). The flag is passed on every run; only the .env line matters.
- Per pick: re-check the frozen rule on the pick's logged forecasts -> pick age <= 120 min -> market active and
  >= 60 min to close -> NO ask within +1c / -3c of the pick price -> size = min(caps, 50% of depth within 1c,
  KALSHI_MAX_CONTRACTS) -> ask the exchange whether we already hold or have an order on that ticker -> write the
  decision to `data/kalshi_exec.jsonl` -> ONE immediate-or-cancel limit order at the ask (sell YES at 1 - ask
  = buy NO at ask), with a deterministic client_order_id. A ticker is decided once per mode, ever; skips are not retried.
- Every run in demo/live first reconciles: unresolved orders are read back from the exchange, any of our orders
  found resting > 10 min is cancelled, and journal holdings are compared to exchange positions (mismatch -> alert).
- Alerts (macOS notification): every fill / partial fill, rejected or unknown order, KILL, loss stop,
  position mismatch, any crash (once per day per distinct error).

## Step 0: deploy the new code (YOU, once now, and again after any code change)
launchd cannot read ~/Desktop, so the agent runs a COPY in `~/.claude/quant-kalshi`. From a Terminal:

    cd ~/Desktop/vocalizations_lab/.claude/worktrees/quant-trader/sports
    bash scripts/install_kalshi_paper_agent.sh

That copies exactly: `sports/scripts/kalshi_exec.py` and `sports/scripts/kalshi_paper.py` ->
`~/.claude/quant-kalshi/scripts/`, and the `sports/sports/` package -> `~/.claude/quant-kalshi/sports/`; rewrites
`run.sh`/`dispatch.sh`/the plist; `chmod 600`s `.env` if it exists; reloads the agent (bootout + bootstrap).
It never touches `.env`, `data/`, or keys. Then check:

    cd ~/.claude/quant-kalshi
    .venv/bin/python scripts/kalshi_exec.py preflight     # want "deployed copy: matches", launchd loaded
    .venv/bin/python scripts/kalshi_exec.py status

One-time effect: the first run appends `settle` rows for the old dry orders (all stamped that day). Harmless:
loss stops only apply in demo/live, and the demo/live journals start empty.

## Step 1: demo key (YOU)
1. Make a demo account at https://demo.kalshi.co (mock funds; Kalshi help article "Creating and using a demo account").
2. Demo profile -> API Keys -> Create New API Key (Ed25519 default is fine; RSA also works). Save the private key to
   `~/.claude/quant-kalshi/keys/kalshi-demo.pem` (NOT under ~/Desktop: launchd cannot read it), then
   `chmod 700 ~/.claude/quant-kalshi/keys && chmod 600 ~/.claude/quant-kalshi/keys/kalshi-demo.pem`.
3. Edit `~/.claude/quant-kalshi/.env` (create it; `chmod 600` it). Leave EXEC_MODE off for now:

       EXEC_ENV=demo
       KALSHI_KEY_ID=<demo key id>
       KALSHI_PRIVATE_KEY_PATH=/Users/jonathanwang/.claude/quant-kalshi/keys/kalshi-demo.pem
       KALSHI_BANKROLL_USD=1000
       KALSHI_MAX_ORDER_USD=10
       KALSHI_MAX_DAILY_USD=100
       KALSHI_MAX_OPEN_USD=100
       KALSHI_MAX_CONTRACTS=1

## Step 2: demo preflight + smoke (YOU)

    cd ~/.claude/quant-kalshi
    .venv/bin/python scripts/kalshi_exec.py preflight   # must end "0 FAIL"; shows demo balance
    .venv/bin/python scripts/kalshi_exec.py smoke       # demo only

`smoke` picks an open demo market, rests 1 post-only buy-NO at 1c (cannot fill), re-sends the SAME
client_order_id (expects HTTP 409 = Kalshi dedupes), reads it back (expects outcome_side=no), cancels it.
Want 4 PASS lines. If the duplicate step is not 409, stop and tell Claude: idempotency then rests on the journal
write-ahead + the exchange pre-check only (both are tested, but the server-side dedupe is the backstop).

## Step 3: 2-3 nights on demo (YOU flip one line)
Add `EXEC_MODE=live` to `.env` (with `EXEC_ENV=demo` still there -> orders go to DEMO). Each morning:

    .venv/bin/python scripts/kalshi_exec.py status      # decisions, fills, spend vs caps, exchange positions
    tail -30 exec.log

Caveat: demo prices/books are not real (Kalshi says so). Picks are made from PROD books; if a ticker does not exist
on demo it logs `skip_no_market`, and thin demo books give `skip_price_moved` / `skip_depth`. Preflight on
2026-10-22 saw 5 open KXHIGHNY markets on demo, so some orders should go through.
Pass before going live: at least one demo fill with no convention KILL (NO cost within 2c of the ask), zero
`reconcile_mismatch`, no ticker with more than one order on the exchange, and a kill drill:
`.venv/bin/python scripts/kalshi_exec.py kill drill` -> next hour exec.log says "KILL file present" -> `rm KILL`.

## Step 4: live with small caps (YOU)
1. kalshi.com -> Account -> Profile -> API Keys -> create a PRODUCTION key (demo keys do not work on prod).
   Save to `~/.claude/quant-kalshi/keys/kalshi-prod.pem`, `chmod 600`. Fund the account.
2. In `.env`: `EXEC_ENV=prod`, set the prod KEY_ID and key path, set `KALSHI_BANKROLL_USD` to the money set aside,
   and first REMOVE `EXEC_MODE=live` (back to dry). Run `preflight` -> 0 FAIL and the real balance.
3. Add `EXEC_MODE=live`. Keep `KALSHI_MAX_CONTRACTS=1` for the first night; next morning check the fill
   (`status`, the notification, and the Kalshi app) -> NO price paid ~= the ask, position shows as NO.
4. Then raise `KALSHI_MAX_CONTRACTS` (e.g. 25) and size caps per notes/risk.md. For B = $1,000:
   MAX_ORDER 10-25 (the code also caps orig7 at 1% of B, X1 at 0.5% of B), MAX_DAILY 100, MAX_OPEN 100.
   ~$100/month needs ~$23/bet and ~$250/day, $500 open -> only with B ~$2,500-5,000 under risk.md.

## Stop / pause
- Instant stop: `touch ~/.claude/quant-kalshi/KILL` or `.venv/bin/python scripts/kalshi_exec.py kill "reason"`
  (also cancels any resting order of ours). Resume: `rm ~/.claude/quant-kalshi/KILL`.
- Back to shadow mode: delete the `EXEC_MODE=live` line.
- Automatic: 24 h net settled loss <= 5% of B -> no new orders until it rolls off; 7-day <= 10% of B -> the
  executor writes KILL itself (live only; demo just pauses) = risk.md's "pause and review".
- A fill whose NO cost is > 2c from the ask writes KILL (would mean a price-convention error).

## .env reference (defaults in brackets)
EXEC_MODE [dry] | EXEC_ENV [prod] prod/demo | KALSHI_KEY_ID | KALSHI_PRIVATE_KEY_PATH | KALSHI_BANKROLL_USD (required
for demo/live) | KALSHI_MAX_ORDER_USD [25] | KALSHI_MAX_DAILY_USD [100] | KALSHI_MAX_OPEN_USD [300] |
KALSHI_MAX_CONTRACTS [1] (per demo/live order; dry ignores it) | KALSHI_MAX_SLIPPAGE [0.01] | KALSHI_MAX_PRICE_DROP [0.03] |
KALSHI_MAX_PICK_AGE_MIN [120] | KALSHI_MIN_MINUTES_TO_CLOSE [60] | KALSHI_DEPTH_FRAC [0.5] |
KALSHI_STALE_ORDER_MIN [10] | KALSHI_NOTIFY [1].
"Today" for the daily cap is the UTC date (a night's picks, 22:00-23:59 local = 02-07 UTC, fall in one UTC day).

## Known limits / not yet verified against a real account
- Response field shapes are from docs.kalshi.com (OpenAPI 3.34.0, read 2026-10-21) and the mocked tests; the demo
  nights are the first contact with the real API. In particular: 409 on duplicate client_order_id (smoke tests it),
  `position_fp` negative = NO (a wrong sign shows up as a reconcile_mismatch alert, not an order).
- Fee is modelled as 0.07 x C x P x (1-P) rounded up to the cent per order; the real fee is recorded from fills.
- Settlement P&L uses the paper trader's market results; the exchange's own settlement records are not read.
- X1 books are thin (STATE.md validation 10-19: LV 11, SDF 35, OKC 82 contracts within 1c; paper report 10-21:
  median 318 over all picks), so many X1 orders will be depth-capped at 50% of depth.
