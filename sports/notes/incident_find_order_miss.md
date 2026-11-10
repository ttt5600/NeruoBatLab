# incident_find_order_miss -- why find_order missed the NFLF IOC exit fills (2026-11-10)

Question: in the 11-09 NFLF runaway (STATE.md top), why did find_order(client_order_id) report no fill for IOC exit
sells that executed? Prereg: success = a printed API response that shows, for the incident exit cids, the field/listing
that find_order's caller read as "not filled", reproduced by a regression test. Kill = cannot reproduce from API ->
report "suspected" only.

Sources: journal ~/.claude/quant-kalshi/data/lead_exec.jsonl (copy of BUFMIN rows in
sports/results/incident_find_order_miss/bufmin_rows.jsonl, 77 rows). Diagnostic: sports/scripts/diag_find_order.py
(GET only).

## Journal facts (bufmin_rows.jsonl)
- 3 NFLF rows on BUFMIN: -BUF key ...1609 (3 YES filled, 21 exit cids, then exit_gave_up), -MIN key ...2011 (8 YES,
  14 exit cids, then a lookup succeeded: exit_raw outcome_side "no", status "executed", taker_fill_cost $5.68),
  -MIN key ...2952 (17 YES, 21 exit cids, gave up).
- POST response body has `fill_count` (no _fp) and NO `status`.

## Result (2026-11-10): root cause CONFIRMED. The order list lags right after the POST, and the POST response was thrown away
Pre-fix exit code (lead session transcript c5152e86, ingame_exec.py L300-322 at the time of the incident) called
`client.create_order(body)` and discarded the response. It then immediately called `x = find_order(client, o, cid)` and
read `f = fill_count_fp or 0`. When f was 0, nothing was recorded, and the next tick sent `/exit/{k+1}` with a NEW cid.
Earlier cids were never re-checked, and there was no attempt cap.

Read-only API today (scripts/diag_find_order.py -> results/incident_find_order_miss/diag_out.txt, raw.json):
- GET /portfolio/orders?ticker=T (no status filter) lists every incident exit cid that reached the exchange. Each has
  status `executed`, fill_count_fp == count (3 / 8 / 17), and fits on a single page (22 and 30 orders, under the
  limit of 200). created_time == last_update_time, so each IOC was finalised instantly. outcome_side "no", side "yes",
  action "sell".
- Fills endpoint: 22 and 33 fills. Every fill maps to a listed order_id (0 orphan fills).
- RULED OUT: status filter (find_order passes none; all orders 'executed', 0 'canceled'), pagination, a field
  rename on the GET Order (fill_count_fp is present), and IOC partial-then-cancel (all fills were full).
- 7 exit cids (MIN row ...2952, attempts 15-21, 03:45:54-03:48:01) were NEVER created. They were rejected at the
  POST (response discarded; most likely insufficient cash), so they did no harm. 21+14+14 = 49 exits executed:
  BUF 3-21*3 = -60 and MIN 25-14*8-14*17 = -325, which match the positions now.
Journal evidence of the lag (results/incident_find_order_miss/lead_exec.jsonl, all live POSTs through ingame/lead place()):
- The GET /portfolio/orders lookup made right after a 201 did NOT list the new order on 27 of 28 POSTs. That includes
  NHL IOC entries whose POST response said fill_count 1.00/2.00. All 27 were found on a later lookup. In 8 cases
  that lookup came 1-2 s later, so the lag is about a second.
- The one old-exit lookup that succeeded (MIN ...2011 attempt 14) was the lucky read.
- So GET /portfolio/orders is eventually consistent for about 1 s: "not listed" does not mean "not filled".
- Create-response shape on prod: {"client_order_id","fill_count":"8.00","order_id","remaining_count","ts_ms"}. It
  has `fill_count` WITHOUT _fp and no status. kx.fills_from_order reads only fill_count_fp, so never pass it a POST body.

## Fix (worktree, NOT deployed)
- scripts/order_guard.py: safe_reduce now sets filled = max(position change, POST response fill), using the new
  `resp_fill()` (handles fill_count, fill_count_fp, and a nested order). The deployed guard used only the position
  change, which may also be stale right after the POST (unverified).
- tests/test_find_order_lag.py has 4 tests: the prod response shape; listing lag reproduces the miss; lagged
  listing + stale positions -> filled is taken from the response; incident replay -> one sell, booked on the first tick.
- Fixture fix in tests/test_order_guard.py: when the fake kills an IOC unfilled, its POST response now also says 0
  (it used to claim a fill).
- Full suite: 420 pass. 2 fail in tests/test_bet.py on a post_only assertion. bet.py does not use order_guard, so
  these failures are unrelated.

## Exposure of other paths
- kalshi_exec weather: reads the POST response fill_count first, so it is not exposed.
- lead_exec.place / ingame_exec.place entries: the immediate lookup misses the same way, but the row goes to
  "unknown" and is re-read later with the SAME deterministic cid. No runaway is possible. The only effect is
  cosmetic: a filled NHL IOC shows "unknown" for one tick.
- needle bot nyt_needle_house.py: reads the POST fill and sizes sells from /portfolio/positions on each 60 s pass,
  with a new cid per pass. It does not use find_order. It is exposed only if positions lags by more than one pass
  (not observed; unverified).
- Unverified: whether /portfolio/positions lags like the order list. No read-only test is possible without an order.
