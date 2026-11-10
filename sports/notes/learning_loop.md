# learning_loop -- lessons in code and on disk, not in chat (agent started 2026-11-10 UTC)

Brief: user NEXT JOBS 11-10 (STATE.md top block). Three deliverables:
1. `sports/notes/lessons.jsonl` -- one row per lesson (date, incident, mechanism, check that would have caught it, where
   the check lives now, status = in_code | proposal | process_only).
2. Data traps -> functions in `~/agent-harness/lib/traps.py` (+ tests in `~/agent-harness/tests/test_traps.py`).
   Execution traps -> audit `sports/scripts/order_guard.py` + executors; gaps written as PROPOSALS below (no executor edits).
3. `sports/scripts/post_trade_review.py` (read-only: exchange GETs + journals) -> `sports/notes/post_trade_review_<date>.md`.

## Pre-registration (this is a tooling job, not an edge test)
- Question: do the last 14 days of live trading contain any anomaly of the six named types beyond the NFLF incident
  already known (BUFMIN -325 NO, BUF -60 NO)?
- Success rule for the tool: it must re-find the known NFLF incident from exchange fills alone (both tickers flagged as
  sells past held / opposite side, and the >1-cid-per-action burst). If it does not, the tool is wrong, not the data.
- Kill rule: none (no money decision). Every other flag is reported with the rows that produced it; no flag is called a
  real problem until its rows are read.

## Data sources
- Kalshi (read-only GET): /portfolio/fills, /portfolio/orders, /portfolio/positions, /portfolio/settlements via
  `kalshi_exec.Kalshi` with the deployed root (~/.claude/quant-kalshi). Fill schema seen 11-10: `book_side` bid/ask is the
  unambiguous signed-YES direction (bid = +YES, ask = -YES); `action`/`side` are NOT intent (a weather maker NO is
  action=sell side=yes). Position at window start is reconstructed from fills since start-30 d.
- Journals in ~/.claude/quant-kalshi/data: kalshi_exec.jsonl (weather E4, intent NO), lead_exec.jsonl (lead + in-game
  NHL/NFLF; `client_order_id`, `exit_coid`, `limit_price`), bets.jsonl, nyt_needle_house_log.jsonl,
  forecastex_exec.jsonl + forecastex_orders.jsonl (IBKR; journal-only checks), kalshi_paper.jsonl (paper twin of E4).

## Log
- 11-10: read AGENT_PROTOCOL, traps.py (8 checks: epoch, lookahead, checked_merge, join_aligned, interpolated,
  overround, too_good), order_guard.py (guarded_create: breaker 5 cids/10 min + per-ticker exposure cap from exchange
  position; safe_reduce: min(want, held), one attempt; resolve_attempt). Wired into ingame_exec, lead_exec, kalshi_exec
  weather path. NOT wired: bet.py:340, manual_bets.py:85, nyt_needle_house.py:286 (raw client.create_order).

## Results (11-10 ~04:25Z)

### 1. Ledger: notes/lessons.jsonl (29 rows)
27 seeded from STATE, FUND_BOOK s6, EDGES_AND_AGENTS waves 3-4, memory feedback_*/quant files (status in_code 15,
partial 2, proposal 1, process_only 9) + 2 appended by the review run below, then triaged by hand. Fields: id, date,
kind (data|execution|ops|research), incident, mechanism, check, check_location, status, open (gap), source/evidence.

### 2. traps.py (data traps): ~/agent-harness/lib/traps.py, tests ~/agent-harness/tests/test_traps.py (15 pass)
Added (nothing existing changed): check_final_status (ESPN final, not last play), check_fresh (stale needle/feed),
check_longshot_stake (<= 0.4 x price x bank), classify_fill + check_fill_not_worse (price improvement is not a failure),
check_reduce_only (an exit may not cross zero), check_one_cid_per_action, check_open_orders_on_venue (phantom open risk).
Already present and mapped: epoch_seconds/check_epoch_seconds, checked_merge/check_join_aligned, check_no_lookahead.
Command: `cd ~/agent-harness && <quant venv>/bin/python -m pytest -q tests/test_traps.py` -> 15 passed.

### Execution traps vs order_guard / executors (no executor edited). GAPS = PROPOSALS
| lesson | enforced where | gap |
|---|---|---|
| exits reduce-only / sell YES not held opens NO | order_guard.safe_reduce (ingame_exec NFLF/NHL) | nyt_needle_house.py:286, bet.py:340, manual_bets.py:85 POST raw `client.create_order`: no exposure cap, no breaker |
| one cid per action | order_guard breaker (ingame, lead, weather) | needle exit cid = `needle-exit-HHMMSS-ticker` = new cid every pass; manual_bets uuid4 per call |
| price improvement != failure | lead_exec.fill_price_bad: > limit+.02 kills; < limit-.10 ALSO kills | a legit >10c improvement on a fast in-game book still trips LEAD_KILL (P5) |
| ForecastEx phantom open | forecastex_exec NOT_FOUND reconcile (b6adf10) | none found |
| Kelly on longshots / stale model | needle bot shrinks toward market | no generic pre-trade gate (P3) |
| bank from exchange | order_guard uses exchange positions for caps | no shared bank() helper |
Also: data/order_guard.jsonl does not exist yet (deployed order_guard.py mtime 11-09 20:11 PT; the last weather order,
04:03Z, was before it). Lead: confirm the file appears after the next guarded POST, else the guard is not on the live path.
Needle bot: `pgrep -fl nyt_needle_house` returned nothing at ~04:30Z (STATE says PID 45461 still running; it is not).

### 3. post_trade_review.py on the last 14 days (report notes/post_trade_review_2026-11-10.md)
Pre-registered success rule MET: C1 flags BUFMIN-MIN (worst -325, now -325) and BUFMIN-BUF (-60) as sell_past_held +
opposite_side from exchange fills alone; C2 exchange bursts of 14 and 21 cids.
NEW FINDING (not in STATE): the same runaway exit happened on **11-08** on 7 more NFLF tickers (DAL, IND, DEN, WAS, NYG,
PHI, NYJ; worst PHI -71.58 NO, NYJ -27, WAS -22; journal exit cids per key up to 20). It went unnoticed because the
journal booked those keys as clean exits: journal P&L -$2.80 (the "NFLF 1-7 -$2.80" in STATE) vs exchange P&L rebuilt
from fills + settlement **+$31.18** (the accidental PHI short alone +$35.68). Validation of the rebuild: 87/94 settled
journal tickers agree within 5c; the 7 that disagree are exactly these. NFLF's real record is luck-dominated, not -$2.80.
Other checks: C3 0 fills worse than limit (282 at, 163 improved, 143 with no journal limit); 0 over-fills; 16 tickers
with exchange orders no journal claims (needle election tickers, KXNFLNEXTTEAM-26JTAYLOR28-MIN 12 filled, 2 NHL 10-27,
KXMAYORLA), so 23 settled tickers' P&L lives on no live journal. C2 also flags needle bursts (HOUSE*, 3-6 cids/10 min;
rebalancing by design, but no cid reuse). C4 E4 live minus paper twin +$0.0056/contract, 95% CI [0.0042, 0.0069], n 82
(maker price improvement; 0 result disagreements). C5 0 orders without a terminal state (journals + exchange + FX).
C6 only BUFMIN-MIN/-BUF open against intent; replayed end positions match all 12 exchange positions.
Raw exchange snapshot: sports/results/post_trade_review/2026-11-10/{fills,orders,positions,settlements}.json.
Wrong turn fixed during the build: settlement `revenue - total costs` is wrong once anything was sold or netted
(HOUSEAZ2-26-R yes 28 / no 28 / revenue 0); P&L is now rebuilt from fills in YES terms.

## Proposals for the lead (NOT installed)
P1 nightly review via launchd. TCC: launchd python cannot read ~/Desktop, so deploy a copy and write outside Desktop.
```
cp sports/scripts/post_trade_review.py ~/.claude/quant-kalshi/scripts/   # imports kalshi_exec + ~/agent-harness/lib/traps.py
# ~/Library/LaunchAgents/com.jonathanwang.quant-post-trade-review.plist
<plist version="1.0"><dict>
  <key>Label</key><string>com.jonathanwang.quant-post-trade-review</string>
  <key>ProgramArguments</key><array><string>/Users/jonathanwang/.claude/quant-kalshi/post_trade_review.sh</string></array>
  <key>StartCalendarInterval</key><dict><key>Hour</key><integer>7</integer><key>Minute</key><integer>15</integer></dict>
  <key>StandardOutPath</key><string>/Users/jonathanwang/.claude/quant-kalshi/post_trade_review.log</string>
  <key>StandardErrorPath</key><string>/Users/jonathanwang/.claude/quant-kalshi/post_trade_review.err</string>
</dict></plist>
# post_trade_review.sh (same venv as the other deployed agents):
#   cd ~/.claude/quant-kalshi && <venv>/bin/python scripts/post_trade_review.py --days 2 \
#     --out notes/post_trade_review_$(date -u +%F).md --ledger notes/lessons_inbox.jsonl --snap-root results/post_trade_review
#   then notify if C1 / C3-worse / C4-disagree / C5 / C6 fired; the lead merges lessons_inbox.jsonl into sports/notes/lessons.jsonl
```
P2 route nyt_needle_house.py, bet.py, manual_bets.py POSTs through order_guard.guarded_create (exposure cap + breaker),
   with a stable cid per intended action (needle: per ticker + target step, not per HHMMSS).
P3 pre-trade gate in bet.py / guarded_create: traps.check_longshot_stake (+ check_fresh where an input ts exists).
P4 one shared order journal written by guarded_create for every POST (cid prefix per bot), so "unjournaled" goes to 0.
P5 lead_exec.fill_price_bad: drop the "< limit - .10" kill or replace it with a YES/NO mix-up test (paid ~ 1 - ref).
P6 STATE/EDGES: restate the NFLF live record from the exchange (+$31.18 on the 11-08 tickers incl. accidental shorts), and
   widen open item (a): find_order missed the IOC exit fills on all 8 NFLF exits of 11-08/09 with extra sells, not one.
