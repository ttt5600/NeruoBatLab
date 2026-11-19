# LX2-climate sealed shot (agent lx2_climate_sealed, started 2026-11-18)

Question: does the post-hoc LX2 climate slice (long-dated tail NO on Kalshi "Climate and Weather" markets; dev A T-90
+3.0% [+2.6, +3.4], 36 events / 70 bets, 0 losses -- notes/lx2_tail_no_carry.md) hold on the held-out window
(settled 2025-10-01..2026-09-12)? User approved the ONE-SHOT sealed test 2026-11-18 ("you have my approval on everything").
Related: notes/tropical_nhc.md (KXHURCAT, different contract, its own dev/sealed), notes/bar_calibration_v2.md
(a 0-loss 96-98c rule cannot pass the family bar on short history -> report exact binomial loss CIs).

## FROZEN RULE (written 2026-11-18, before any sealed price/outcome was fetched or read)
Identical to LX2 trial A (edgelab/rules/longdated_tail.py:a_t90, scripts/lx2_tail_no_carry.py) plus ONE filter:
- Universe: settled binary Kalshi markets, series frequency not hourly/15min/daily/weekly, series category ==
  "Climate and Weather" (exactly lx2's "climate" bucket; no hurricane-only sub-filter, no series hand-picking).
- Anchor S = expected_expiration_time; entry t = S - 90 d (T-90 only; T-180 NOT registered: B climate was 9 events,
  and one trial keeps the family bar at 3 trials). Market open_time <= t < close_time.
- Quote: last DAILY candle ending in (t - 7 d, t]; need yes_bid >= 0.01, yes_ask <= 0.10, yes_ask > yes_bid.
- Trade: buy 1 NO at NO ask = 1 - yes_bid, taker, Kalshi fee exact (quadratic 0.07 x mult x p(1-p)); 1-ct ceil
  rounding as sensitivity. Hold to settlement.
- Interest: evaluator ROI EXCLUDES interest (primary). Reported alongside: + cost x 3.5% x days/365 (Kalshi APY; real
  for the 2025 season since Kalshi paid on positions from ~Oct 2024). Interest = Kalshi's cash rate, not an edge.
- Edgelab trial: lx2c_a_t90_climate_tail_no, family kalshi_longdated_tail (3rd trial -> bar .05/3), dataset
  kalshi_longdated_tail, min_n 30, kill roi_ci_below_zero after 30. Sealed window = vault sealed [2025-10-01, 2026-09-12].
- Unit: event_ticker cluster (evaluator); also report by season / series (seasons, not markets, carry the tail).
- Reporting: n bets, n events, losses with exact Clopper-Pearson 95% CI vs break-even loss rate (1 - mean cost),
  ROI with/without APY, concentration by series/season.
SUCCESS (sealed): edgelab verdict pass. KILL: edgelab fail. The binomial loss CI is reported, not a second gate.
Gate: edgelab only unseals a trial in state dev_pass (edgelab/vault.py sealed_gate). If dev fails, the sealed shot is
BLOCKED; no bypass (no pricing/scoring of sealed rows outside edgelab).

## Log
- 2026-11-18: registered lx2c_a_t90_climate_tail_no-a7c423d2 (spec edgelab/specs/lx2c_a_t90_climate_tail_no.yaml, rule
  edgelab/rules/longdated_tail.py:a_t90_climate) -> family kalshi_longdated_tail 3 trials, bar p < .0167.
- `evaluate --split sealed` REFUSED: "state is 'registered'; sealed needs dev_pass" (edgelab/vault.py sealed_gate).
  No sealed price or outcome was fetched or read. NB the vault frame (data/lx2_tail_no_carry/priced.parquet) holds DEV
  rows only, so sealed rows would also have to be priced first -- not done.
- `evaluate --split dev` -> FAIL, state registered -> DEAD. n 70 bets / 36 event clusters, 0 losses, net ROI +3.01%
  [+2.61, +3.42], p 2.5e-5 (< bar), halves +3.08 / +2.94%. Failed checks: c concentration (week share 0.48, top week
  2023-W48 = season-end settlement; cluster share 0.06), e CLV (coverage 0: frame has no close price), plus too-good
  guard (Sharpe 6.3 > 5). Both checks fail for structural reasons: every city/season hurricane market settles ~Nov 30,
  so one ISO week holds half the profit; no close price exists in this frame. => the rule as defined cannot reach
  dev_pass, so the sealed shot is blocked by edgelab, not by the data.
- Exact binomial (results/lx2_climate_sealed/dev_report.py): 0/70 losses, Clopper-Pearson 95% loss rate [0, 5.13%] vs
  break-even 2.92% (mean cost 97.08c) -> not distinguishable from break-even. At the event unit 0/36 -> upper 9.74%.
  ROI no interest +3.01%; + 3.5% APY +3.86% (mean hold 89 d; pre-Oct-2024 interest counterfactual).
  By settlement year: 2022 15 bets/7 ev 24% of P&L, 2023 33/15 52%, 2024 21/13 21%, 2025 1/1 4% (the one 2025 row
  settled before 10-01 = dev). 22 series; top KXHURNO 12%, KXHURORL 12%, KXHURNYC 11%. = 3 hurricane seasons.

## Result
DEAD at dev under edgelab (concentration + no CLV + too-good guard); sealed 2025-season shot BLOCKED by the gate (not
bypassed). To actually test the hurricane slice the lead/user would need either (a) an edgelab path for season-settled
near-certain contracts (week-concentration and CLV checks cannot pass here) -- a rule change, user decision -- or
(b) a forward-only test on the 2026/2027 seasons.

## Learnings
### Season-settled contracts cannot pass edgelab dev
LX2-climate +3.01% [+2.61, +3.42], p 2.5e-5, still FAIL: all hurricane season markets settle the same week (week share
0.48 > 0.30) and the lx2 frame has no close price (CLV coverage 0). Source: lab/edgelab_registry.jsonl
lx2c_a_t90_climate_tail_no-a7c423d2 dev. Any season-long Kalshi rule needs a different gate design before a sealed shot.
### 0 losses in 70 at 97c is not evidence of an edge
Clopper-Pearson 95% loss rate [0, 5.13%] vs break-even 2.92%; at the 36-event unit the upper bound is 9.7%. One
landfall season would flip it. Source: results/lx2_climate_sealed/dev_report.py.
### The sealed gate refuses before data is touched
`evaluate --split sealed` on a registered (not dev_pass) trial raises VaultError before loading; the user's approval
does not change the ladder -- report blocked rather than score sealed rows by hand.
