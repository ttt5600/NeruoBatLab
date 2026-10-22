# Execution tests 2026-10-22 (~16:45-17:15Z)

User: "we need to backtest and test with the $50 and other payment routes so you can begin to test placing trades."
No keys, no account numbers in this file. Nothing committed.

## 1. Kalshi status (deployed executor ~/.claude/quant-kalshi)
- `preflight`: 0 FAIL, 0 WARN. Mode LIVE (EXEC_ENV=prod, EXEC_MODE=live). Caps: $1.05/order orig7, $1.00 x1, $15/day,
  $20 open, 1 contract/order, loss stops 24h -$10 / 7d -$20. Deployed copy == worktree copy. launchd loaded, hourly ticks OK.
- `status`: exchange balance **$50.00, 0 positions, 0 live orders**. The live journal is empty: live mode was switched on
  ~08:30Z, AFTER last night's 22:00-23:59 local pick window, so last night produced only dry rows (45 decisions, 15 with
  contracts, 10 settled, 10 wins, dry P&L +$13.17 at 25-contract dry sizes). No fills exist yet, so nothing to reconcile;
  the "avg fill ~ 1 - NO" check can first be made after tonight.
- Paper (all 24 cities): 54 picks, 38 settled, 38 wins, +1.04u.

## 2. Order-path test on PROD at zero economic cost -- PASS
Script: `scripts/kalshi_order_path_test.py` (uses the deployed executor's own `Kalshi` client and `.env`).
Market KXHIGHNY-26OCT22-B63.5 (today's NYC bracket, YES bid 0.42 -- will never be a pick again, tonight picks 26OCT23).
One post-only GTC buy-NO at $0.01 (= sell YES @ 0.99), 1 contract:

| step | result |
|---|---|
| create | HTTP 201, fill_count 0 |
| re-send same client_order_id | HTTP 409 `order_already_exists` -> server-side dedupe CONFIRMED on prod |
| GET order | resting, outcome_side=no, no_price 0.0100 (sign convention confirmed: side=ask on YES leg = NO) |
| listed in /portfolio/orders?status=resting | 1 match |
| cancel | HTTP 200, reduced_by 1 |
| confirm cancelled | status canceled, 0 resting |
| positions / balance | 0 / $50.00 (unchanged; $0.01 reservation did not show in balance_dollars) |

Run 3 times (3 `order_path_test` rows in data/kalshi_exec.jsonl; keyed `market`, so fold/reconcile ignore them). Run 1's
"confirm cancelled" read FAILED: a read 0.3 s after the 200 cancel still said resting; 2 s later it was canceled
(last_update_time 1.4 s after create). Read-after-write lag, not a failed cancel; the script now polls.
Cost: $0.

Findings for the executor (not changed -- live code, no tests run against these yet):
- **Fill response schema is still unverified.** A no-fill 201 returns only `{fill_count, order_id, remaining_count, ts_ms}`.
  `place()` reads `average_fill_price` / `average_fee_paid` on a fill. If a filled 201 lacks them (or names them
  differently), avg_no computes as 1.00, the 2c convention check trips, and the executor writes KILL after the FIRST
  fill and books worst-case cost $1. That is fail-safe (stops, alerts, no extra orders) but would halt trading after one
  order. The GET order object carries `taker_fill_cost_dollars`, `taker_fees_dollars`, `fill_count_fp`,
  `yes_price_dollars`, `no_price_dollars` -- a fallback read-back is the fix if it happens. Check tomorrow morning.
- `place()` pre-check `client.orders(ticker=t)` has no status filter, so ANY earlier order on a ticker (even
  cancelled) makes it skip that ticker. Harmless today (one decision per ticker anyway); never run tests on 26OCT23+ tickers.
- With $15/day and $20 open and next-day settlement, night n+1 only has $5 of room while night n is unsettled. The sim
  below shows this caps ~64% of picks (2,248 of 3,507). Raising KALSHI_MAX_OPEN_USD to 30 would let every night use $15
  (user's call; still inside $50).

## 3. Real-money trade now -- none
No fresh pick exists: picks are made only at 22:00-23:59 city-local and must be <= 120 min old; the 09:03 PDT tick said
"[live] no fresh unexecuted picks". Not forced. First real orders: tonight's ticks from 19:03 PDT (Eastern cities 22:03
EDT) through ~00:03 PDT (Pacific), up to 15 contracts / $15.

## 4. $50 backtest (scripts/kalshi_bankroll_sim.py) -- replay of the already-scored sealed streams, not a re-score
Reproduced both lockbox totals exactly first (E1 1,844 bets +26.82u; E2 2,464 bets +19.22u). Live caps as above,
1 contract per pick, random order within a night when caps bind, 22:00 quote prices (real fills can only be worse).
Bootstrap = whole nights from 2026-03-01..10-18 (232 nights, both streams live; 15.1 picks/night, 0 empty nights),
4,000 paths. 7-night blocks gave the same answers within ~$2/yr.

Fee rounding matters more than anything else at $1 size: 0.07 p(1-p) at NO 0.97 is $0.002/contract; rounded up to the
cent it is $0.01 -- 4-5x, which eats the whole edge. Kalshi docs (docs.kalshi.com/getting_started/fee_rounding, read
today): DIRECT members are charged at $0.0001 balance precision (with rounding rebates); non-direct/FCM accounts at $0.01.
So the user's direct account should pay ~exact fees -- UNVERIFIED until the first fill shows `taker_fees_dollars`.

| scenario | month 1 median [5%, 95%] | month 12 median [5%, 95%] | P(yr-1 < 0) | yr-1 max DD median / 95% | nights to $100 |
|---|---|---|---|---|---|
| observed sealed edge, exact fee | +$2.57 [-1.9, +6.1] | +$29.3 [+15.1, +42.6] | 0% | $4.65 / $8.26 | median 607 (~20 mo); P(by 1y) 1%, by 2y 79% |
| sealed CI LOWER bound edge (E1 +0.59%, E2 +0.19%) | +$0.74 [-3.8, +4.3] | +$7.2 [-7.1, +20.5] | 21% | $7.20 / $14.5 | P(by 3y) 4% |
| observed edge, fee rounded to the cent | +$0.17 [-4.3, +3.7] | +$0.36 [-13.8, +13.8] | 48% | $9.09 / $18.8 | never (0% by 3y) |

- Historical path (actual night order, exact fee): +$20.22 over 232 nights, max DD $3.79.
- Loss stops: P(24h -$10) = 0% and P(7d -$20 KILL) = 0% in every scenario -- worst real night was -$2.33 (2026-07-12,
  21 picks); with <= $15 placed per night the 24h stop cannot fire without ~11 losses in one night.
- The observed-edge row is optimistic by construction (it treats the 232-night sample edge as truth); read the
  CI-lower-bound row as the honest planning case: ~+$7/yr, one-in-five chance of a losing year.
- Bottom line (unchanged from the LIVE block): the $50 phase validates execution; it does not grow money. $100 takes
  ~1.5-2 years even if the sealed edge is real.

## 5. Other routes
(a) **Polymarket US** -- read access WORKS (`polymarket_us.py test` PASS 1.4 s: markets, event, book, BBO, price history,
settlement, search). Automated orders are allowed (create-order has `manualOrderIndicator` AUTOMATIC; docs have a
"For automated systems" section). No keys present (sports/.env does not exist). To enable (user does all of it):
  1. Install the Polymarket US app, create the account, pass KYC (name, DOB, address, SSN; maybe ID/selfie). CA listed
     available by a third party; the app's KYC is the final word.
  2. Fund: debit card, ACH, Apple Pay, PayPal, Venmo, USDC (iPhone), wire ($5k min).
  3. https://polymarket.us/developer, signed in with the SAME login method as the app -> Create API key -> Key ID +
     Secret (Ed25519; secret shown once).
  4. Create `sports/.env` (gitignored), `chmod 600`, add `POLYMARKET_KEY_ID=...` and `POLYMARKET_SECRET_KEY=...`.
  Caveat: the weather rule on PMUS dev was -0.14%/283 (FAIL) and there is no cross-venue arb, so there is nothing to
  trade there yet; an executor (kalshi_exec clone: caps, KILL, dry first) is not built.
(b) **Robinhood agentic account** (read-only check via the robinhood-trading MCP, available in this session): the agentic
  account exists (nicknamed "MCP", limited margin, options level 3), **value $0, cash $0, buying power $0, no pending
  deposit**. Trade approvals are currently **OFF** (human_must_approve_trades = false, set 08:05Z today). No orders placed.
  The MCP has equity/option/crypto order tools and NO event-contract tools, so Robinhood cannot run the weather strategy;
  its event contracts route to Kalshi/ForecastEx anyway (no new book) and as an FCM-cleared (non-direct) account it would
  likely pay cent-rounded fees -- the 3rd row of the table above. Funding: Robinhood app -> agentic account -> Transfer ->
  from linked bank or an internal transfer from the main account (exact limits/timing not checked; general app flow). Recommend
  turning approvals ON in the app before funding.
(c) **Kalshi funding** (help.kalshi.com, pages 1-2 months old): deposits min $10 -- ACH via Plaid (no fee, settles <= 5
  business days, part credited at once), debit card / Apple Pay / Google Pay (instant), PayPal, Venmo, Cash App
  (deposit only, has a fee), crypto via Zero Hash (~30 min), wire ($1,000 min, memo code required). Withdrawals: debit card
  (no fee, usually hours; some 3-5 business-day compliance reviews), bank ACH (no fee, a few business days), crypto (no fee,
  ~30 min, needs a prior crypto deposit), Venmo (no hold on Venmo-funded money). Deposits have security holds before
  withdrawal; daily withdrawal limit is per-account. Third-party pages claim a ~2% debit-deposit fee and a $25 wire fee;
  not on Kalshi's own pages -> check the Transfers screen.

## Next commands (user)
- Tomorrow morning: `~/.claude/quant-kalshi/.venv/bin/python ~/.claude/quant-kalshi/scripts/kalshi_exec.py status`
  and `tail -40 ~/.claude/quant-kalshi/exec.log` -> want fills with NO price ~ the ask, fee ~$0.002/contract, no KILL.
- If a KILL appears saying "fill price convention check failed": do NOT rm it; tell Claude (likely the fill-schema issue).
- Optional: `KALSHI_MAX_OPEN_USD=30` in ~/.claude/quant-kalshi/.env so every night can use the $15.
- Re-run the zero-cost path test any time: `~/.claude/quant-kalshi/.venv/bin/python scripts/kalshi_order_path_test.py run <today's ticker>`.
