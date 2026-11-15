# e4_ops_diag -- E4 ops failures 11-13..11-15 (agent e4_ops_diag, 2026-11-15)

Brief: three problems from critic row `critic_e4_step2` (findings_inbox.jsonl). Ops diagnosis, not an edge test,
so no success/kill rule; the rule followed was "print the actual failure mode before fixing". Note: this file was
written after the diagnosis, not before (protocol slip; nothing was scored).

Read-only throughout: GETs to Kalshi (/portfolio/balance, orders, fills, settlements, positions,
target_balance_allocation, /markets/{t}), log reads. No orders, no cancels, no .env/launchd/kill changes,
nothing deployed to ~/.claude/quant-kalshi. Scratch scripts: `sports/results/e4_ops_diag/{acct_snapshot,mkt_shard,alloc}.py`.

## 1. No picks for the nights logged 11-13 and 11-14 UTC -- CONFIRMED: open-meteo timeouts crash the whole tick

- Mac was awake: `~/.claude/quant-kalshi/run.log` has a tick every hour on time 11-12 18:03 .. 11-15 01:03 PST
  (no gaps; so pmset/sleep is ruled out).
- Every decision-window tick (19:03-23:10 PST = 03-07Z) on 11-12 PST and 11-13 PST ended with
  `ReadTimeoutError(host='api.open-meteo.com', read timeout=60)` raised from `live_forecasts` (kalshi_paper.py:57,
  called at :117) and `exit 1` in run.log: 5/5 ticks each night. Non-decision ticks never call open-meteo, which is
  why every failed tick sits in hours 19-23 PST.
- Mechanism: `cmd_pick` has no per-city try/except. The first city whose forecast fetch times out raises out of
  the loop, so NO city is logged that hour (cities whose fetch would have worked are lost too). The DONE file is not
  written, so the next hour retries -- but the window is only 2 hours per city (22, 23 local) and all 5 hours failed.
- Why launchd looked healthy: launchd runs dispatch.sh -> run.sh; run.sh logs `exit $?` of the paper script to
  run.log, then its last command (kalshi_exec.py) exits 0, so launchd's last exit = 0 and launchd.err stays empty.
  The failures are only visible in run.log.
- 41 open-meteo errors in run.log since 10-15; failed ticks by date: 11-12 5, 11-13 5, 11-14 1 (the 11-14 20:03
  PST failure is why the CT orig7 picks for NOV15 came at late_min 63). forecastex_paper.log also shows open-meteo
  ReadTimeouts on 11-14, so the outage was server/network-side, not our code (SUSPECTED: open-meteo slowness or
  throttling of our IP in the US evening; a manual curl at 09:31Z returned 200 in 0.33 s).
- E4 orders: `kalshi_exec.jsonl` has NO order rows dated 11-13 or 11-14 (only 3 settle rows on 11-13).
  So E4 placed zero orders for event dates NOV13 and NOV14. CONFIRMED.

Fix (worktree `sports/scripts/kalshi_paper.py`): `_retry` (3 tries, 5/10 s backoff) around the market list,
forecast and orderbook fetches; open-meteo timeout (10 s connect, 45 s read); per-city try/except -> `FETCH_FAIL`
line, city NOT marked done (next run in the window retries), other cities proceed; a failed orderbook read keeps
the pick with depth 0 (executor re-reads the live book); `tick` exits 3 when any city failed so run.log shows it.
Worst case in a total outage: ~7 due cities x 3 x 55 s ~ 20 min per tick, inside the hour.
Tests: `sports/tests/test_kalshi_paper_resilience.py` (3 tests: total outage then recovery, one city failing, transient
retry). Live read-only `validate` run of the patched script: 3 picks, no errors.
Proposal still open: a watchdog that alerts when a city-night passes its window unevaluated (e.g. fund_brief reads
run.log for FETCH_FAIL / exit 3), and maybe a second forecast source if open-meteo stays flaky.

## 2. 11-15 05:03Z insufficient_balance -- CONFIRMED: executor reads account TOTAL cash; weather trades use shard-0 cash only

- GET /portfolio/balance at 09:32Z: `balance_dollars` 604.39, but `balance_breakdown` = shard0 $4.81, shard1 $0,
  shard2 $570.73, shard3 $28.84. KXHIGHAUS/KXHIGHDEN markets have `exchange_index` 0 (GET /markets).
  Kalshi keeps cash per exchange shard and an order only sees its shard's cash (documented in quant-kalshi
  scripts/bet.py:17; lead_exec.py already uses `shard0_cash`).
- kalshi_exec.py `Kalshi.balance()` returned `balance_dollars` ($604) -> budget not bound -> two 24-ct maker orders
  ($23.52, $23.28) POSTed -> HTTP 400 insufficient_balance.
- Cash at 05:03Z: no fills after 02:13Z and no settlements after 02:44Z (GET fills/settlements), so shard-0 cash at
  05:03Z was almost certainly the same $4.81 (CONFIRMED modulo an unlogged transfer).
- Where shard-0 cash went: shard 0 holds long-dated KXNVDACEO-29JAN-TCOO (650.95 ct, exposure $18.73) and
  KXDHOUSESEATS-27-249 (87 ct, exposure $83.78), TSA (3 positions + 5 resting sells, ~$4.80 reserved). The
  $502 KXBTC15M round trip on 11-13 and its settlement live in shard 2 (exchange_index 2), NBA trades in shard 3.
  Auto-rebalance is OFF (`target_balance_allocation` = []). So the political positions are not settling "later" into
  shard 0 any time soon (2027/2029 markets); new cash lands in shards 2/3.
- No other bot was reserving shard-0 cash in resting orders beyond the TSA sells.
- Retry behaviour: a `rejected` row enters fold() state, and `fresh` excludes tickers already in state -> the pick
  is dropped for good (one REJECTED notification is sent; no retry). Same for `skip_balance`.

Fix (worktree `sports/scripts/kalshi_exec.py`): `Kalshi.balance(shard=...)` returns that shard's cash from
`balance_breakdown` (falls back to the total when the API sends no breakdown); `cmd_run` sizes against shard 0
(`WEATHER_SHARD`); `decide` skips (`skip_shard`) any market whose exchange_index is not 0; a `skip_balance` sends one
notification per day. With $4.81 the 11-15 run would have posted 5 ct on AUS and skip_balance on DEN instead of two
rejects. Tests added to `sports/tests/test_kalshi_exec.py` (3: replay of 11-15, legacy response, other shard);
51/51 pass. Retrying skip_balance/rejected picks later was NOT added: fold() keeps only the first order row per
ticker, so a retry needs a fold change -- proposal only.

NEEDS THE USER: shard 0 must be funded for E4 to trade at 24 ct (peak night 7 orig7 orders ~ $165 at 24 ct). Options:
`bet.py allocate 0=..` (auto-rebalance, POST target_balance_allocation) or a manual shard transfer. That moves money
-> human gate. Until then E4 is effectively off regardless of the code fix.

## 3. Maker orders ignore the depth cap -- by design; recommendation: keep it off for makers

- `_decide_maker` docstring: "No depth cap (a resting bid takes no liquidity)". `depth_1c` in the maker path is
  the resting YES-bid size at the best price = the NO-ask liquidity a TAKER would lift. A resting NO bid at q
  never touches it, so capping by it is the wrong quantity (AUS 3.9 ct says nothing about whether 24 ct at 0.98 fills).
- What size does change for a maker: (a) cash reserved for the full count while it rests (bites now because of
  shard 0); (b) fill/queue: the sealed backtest fill model is "N=10 size-through" (notes/kalshi_temp_maker.md L55),
  so fills at 24 ct are outside what was validated -- partial fills are harmless per contract, but bigger orders
  fill more completely exactly when an informed seller dumps (adverse selection), which the backtest cannot see.
- Recommendation: do not port depth_frac to makers. Instead log, per settled maker order, volume printed at <= q
  between post and close and fill fraction by size, and review after ~30 orig7 24-ct orders. If a cap is wanted,
  cap at a share of recent traded volume, not resting depth.

## Learnings

### A launchd exit 0 can hide a crashing job when the wrapper's last command succeeds
run.sh logs the paper script's `exit 1` to run.log, then runs kalshi_exec.py which exits 0, so launchd showed 525
runs, last exit 0, empty err while 10 decision-hour ticks crashed on 11-12/11-13 PST. Check the job's own log, not
launchd state (source: ~/.claude/quant-kalshi/run.log).

### One un-isolated network call in a per-city loop loses every city
A single open-meteo ReadTimeout in kalshi_paper.cmd_pick aborted all cities each hour -> zero E4 picks/orders for
event dates NOV13 and NOV14. Wrap each unit of work in try/except + retry and leave failed units retryable.

### Kalshi cash is per exchange shard; balance_dollars is not spendable cash
11-15: balance_dollars $604 but shard 0 (weather, TSA, politics) $4.81; most cash sat in shard 2 (crypto) after a
$502 BTC trade settled there. Any sizing code must read balance_breakdown[exchange_index of the market]
(source: GET /portfolio/balance 2026-11-15T09:32Z). Auto-rebalance (target_balance_allocation) was off.

### Depth caps are a taker concept
The resting depth on the opposite side measures what a taker can lift; a resting maker bid's fill depends on
traded volume through its price. Cap makers by traded volume (and reserved cash), not book depth.
