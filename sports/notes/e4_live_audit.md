# e4_live_audit -- E4 (Kalshi temp consensus-reject NO, MAKER) live audit, 2026-11-09

## Questions (from lead)
1. Live realized ROI/$ risked on maker fills, bootstrap CI (by settle date), vs sealed [+1.57,+2.84]%; power.
2. Adverse selection: filled vs unfilled maker picks' outcomes.
3. Fill-price < .90 vs >= .90 and per-city, on DEV only (post hoc -> any rule = NEW trial, registered in edgelab).
4. Capacity: are caps binding; 2x/4x caps on the live fill book.
5. Logging/P&L reconciliation bugs (kalshi_exec.jsonl vs bets.jsonl vs exchange, read-only GETs).

## Pre-registered rules (written before computing any outcome)
- Q1: "consistent" = sealed point +2.22% inside the live 95% bootstrap CI (resample event dates). "weakened" if the live
  CI upper bound < +1.57% (sealed CI lower). "dead" needs live upper < 0. Otherwise "holds". With small n, say
  "not distinguishable" explicitly.
- Q2: dev expectation = losers ~always fill (unfilled loss rate 0% in dev). Flag if live unfilled loss rate > filled.
- Q3: dev only (E1 dev + X1 pre-2026-03 picks in data/kalshi_temp_maker_*.parquet); a price-floor rule passes only if
  it beats M3_N10 at the edgelab family bar; otherwise reported as noise.
- Q4: binding = share of maker orders whose count was limited by order/day/open caps rather than contract cap.

## Data
~/.claude/quant-kalshi/data/kalshi_exec.jsonl (journal: order/update/settle), kalshi_paper.jsonl (picks + result),
bets.jsonl; deployed script ~/.claude/quant-kalshi/scripts/kalshi_exec.py. Live .env (non-secret keys):
MAX_ORDER 6.20, MAX_DAILY 60, MAX_OPEN 80, MAX_CONTRACTS 6, EXEC_STYLE maker, KALSHI_COHORTS=orig7 (x1 PAUSED).

## Q3 pre-registration (written before any dev slice was computed, 2026-11-09)
New trial `e4f_floor90` = E4 M3_N10 on E1 dev picks restricted to NO ask at T >= 0.90 (rule
edgelab/rules/kalshi_wx_floor.py:maker_m3_floor, spec edgelab/specs/e4f_floor90_e1.yaml). Post hoc (motivated by
live 11-09 ACT_LIST slice), so it counts as a new trial in family kalshi_weather_temp.
Primary (the question that matters): on DEV (E1 dev + X1 dev, data/kalshi_temp_maker_picks.parquet + trades), the
M3_N10 units in the DROPPED slice (ask < 0.90). A floor helps only if dropped-slice units < 0 with the date-cluster
95% CI entirely below 0. If the CI crosses 0 -> "not distinguishable", no rule proposed. Per-city: descriptive only,
no rule (24 cities = forking paths).

## Log

## Results (2026-11-09; scripts + outputs in sports/results/e4_live_audit/)
Fold: build.py -> live_rows.parquet. The "111 settled, 109-2, +$1.42" in STATE is MAKER + TAKER combined: maker = 39
settled fills (38-1, +$1.84), taker (pre-10-29) = 72 (71-1, -$0.42). E4 itself has only 39 settled fills.

Q1 (q1_q2.py -> q1_q2.json): maker ROI/$ = +1.72%, date-cluster bootstrap 95% CI [-4.33%, +6.11%] (39 fills, 9 event
dates); equal-weight per fill +1.52% [-4.35%, +6.34%]. Sealed +2.22% sits inside -> consistent; CI crosses 0 ->
not distinguishable from 0 either. By cohort: orig7 18 fills 17-1 +$0.69 (+1.01%), x1 21 fills 21-0 +$1.15 (x1 now
paused). Power: per-fill ROI SD 0.18 (live) to 0.24 (5.65% loss-rate model) -> ~530-930 fills to separate 0 from +2.2%
at 80%/5%; current half-width +/-5.7pp. At ~3 orig7 fills/night that is 6-10 months. Pre-reg verdict: HOLDS.
Maker vs taker on the same 71 posted picks (taker at pick price + fee, same count): maker - taker = -$0.11 [-1.35, +0.95].

Q2 adverse selection: fill rate 39/71 settled = 55%; filled loss 1/39, unfilled loss 0/32 (all 32 unfilled won).
Direction matches dev (losers trade through and always fill), but only 1 loser -> no rate estimate possible.
Fill-model check vs public tape (q4_tape.py -> q4_tape.csv/json, read-only GETs): on the same 74 orders the dev fill
model (>=10 printed at <= q after post) says 67.6% fill, live 56.8%. Zero false negatives; 8 false positives. Excluding
the 6 externally cancelled orders: every order whose tape traded THROUGH q filled (34/34); orders with prints only AT q
filled 8/14 (queue position). Dev re-score with that queue haircut (q2b_queue_haircut.py -> json; exploratory, not a
promotion): 91% of dev fills traded through (all 70 dev losers did), so units 39.9 -> 37.8, ROI/fill 3.18%
[1.68, 4.67], maker - taker +7.38 [+2.16, +12.38] units. The fill-model optimism is real but small for E4's edge.

Q3 price floor (q3_dev_floor.py -> q3_dev_floor.json; trial e4f_floor90_e1 registered in edgelab, family now 46 trials,
bar 1.09e-3; NOT run through `evaluate`): on DEV the slice a 0.90 floor would DROP (NO ask < .90; 286 picks, 227 fills)
made +11.89 units [-4.36, +27.69], ROI/fill +5.24% [-1.73, +12.36], filled loss rate 19.8%. Pre-reg: floor helps only if
that CI is below 0 -> it is not; point estimate says the floor would REMOVE profit. No rule proposed. The live <.90 slice
is n=4 (maker part: 3 fills, 2-1, -$0.64). Kept slice (>=.90) +28.0 [16.8, 38.9]. Per-city on dev: descriptive only;
legacy HIGHNY/HIGHMIA/HIGHCHI series negative (CIs cross 0); no rule (24 cities = forking paths).

Q4 capacity (q4_scale.json): binding cap = per-order KALSHI_MAX_CONTRACTS=6 (every order since 11-06 is exactly 6;
MAX_ORDER $6.20 is the same size at q~.97). Peak night 7 orig7 orders ~ $41 committed < day $60 / open $80. Depth:
volume traded through q >= 2x our size on 32/39 settled fills, >= 4x on 29/39 (median printed at <= q: 515 contracts).
Realized live book: 1x +$1.84 on $107; 2x conservative +$3.12 on $190 (1.64%); 4x conservative +$4.96 on $326 (1.52%),
optimistic +$7.36. 2x per-order also needs open cap > $80 on 7-order nights. Absolute stakes stay small.

Q5 bugs/reconciliation:
- P&L reconciles exactly: 111/111 KXHIGH settlements from GET /portfolio/settlements match journal settle rows
  (contracts, result, pnl; maker $1.84 both, taker -$0.4183 both, exchange maker fees 0). exchange_settlements.json.
- BUG (execution): 6 maker orders CANCELLED on the exchange before close by something other than kalshi_exec (no
  `cancel` event journaled; GET order status 'canceled'): last_update 2026-11-04T03:05:26Z (TTTN-26NOV03-B74.5,
  DEN-26NOV03-B61.5, TLV-26NOV03-B72.5, MIA-26NOV04-B87.5) and 07:18:42Z (DEN-26NOV04-B60.5, DEN-26NOV04-T61). Times
  coincide with the NYT needle House bot (exits at 07:18:10, nyt_needle_house_log.jsonl) -> likely a cancel-all on the
  shared account (probable, not proven). All 6 won; DEN-B60.5 traded through q (111 contracts) and would have filled.
  Journal records them as plain no_fill. Same night 2 orig7 picks skip_balance (cash drained).
  Proposal: other bots cancel only their own client_order_ids; kalshi_exec reconcile tags 'cancelled_external' when
  exchange status == 'canceled' and now < expiration_time, and notifies.
- GAP: bets.jsonl (fund ledger) has 0 KXHIGH rows -> E1/E2/E4 P&L lives only in kalshi_exec.jsonl.
- edgelab status shows e4_maker_e1 "forward n 0/100" although 18 orig7 maker fills are settled -> forward evaluate not
  run since; lead: `python -m edgelab evaluate e4_maker_e1 --split forward`.
- Headline confusion: STATE "111 settled 109-2 +$1.42" and ACT_LIST's ".90 slice" mix taker E1/E2 with E4 maker.

VERDICT: HOLDS (live +1.72% [-4.33, +6.11] contains sealed +2.22%; 39 fills is ~5% of the n needed to tell).
