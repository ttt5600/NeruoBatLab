# SIZING RULE -- per-strategy Kelly step-ups (written 2026-11-10, BEFORE applying any change)

**Authority:** user 2026-11-10: "if you have an edge and see in the paper test it's working well (split out what our
return is on just those trades ...) then you can bump up the bet size. You can do kelly sizing." fund.yaml lists
change_size / raise_caps as human gates; this message is the user exercising that gate, scoped by the rule below.
Proposed to user: record this as a standing amendment in fund.yaml (agents cannot edit it).

## Eligibility (all must hold, judged on the strategy's OWN trades only, never pooled)
1. Out-of-sample evidence: sealed lockbox PASS, or >= 100 settled live/forward trades with ROI 90% CI lower bound > 0.
   Dev/backtest numbers never count.
2. Live record (real fills, from exchange fills -- not journals) not significantly negative: ROI 95% CI upper > 0, and
   none of its pre-set kill rules has fired.
3. No open incident / reconciliation gap on that strategy's executor.

## Size
- Edge r = lower bound of the 90% CI of ROI from the eligible out-of-sample sample (sealed if that is the source).
- Per bet at price c: Kelly fraction f* = c*r/(1-c) of bankroll; we use 1/4 Kelly. (Rationale: near-certain contracts
  give huge f*, and one point of win-rate error flips the sign -- see memory feedback_longshot_kelly_stale_model.)
- Same-night / same-event bets are ONE correlated basket: basket exposure <= min(sum of 1/4-Kelly, 20% of bankroll).
- Depth: per order <= 50% of resting depth within 1c (executor KALSHI_DEPTH_FRAC), and only step up where the live tape
  shows volume through our price at >= 2x the new size on most fills.
- Steps: at most 2x per step, at most one step per 14 days, each step needs the live record SINCE the last step not
  significantly negative. Revert one step if the strategy loses >= 10% of bankroll since the step, or a kill fires.
- Not eligible -> stays at minimum units ($1 / 1 contract).

## Applied 2026-11-10
| Strategy | Eligible? | Evidence (own trades) | Action |
|---|---|---|---|
| E4 maker orig7 (kalshi_exec) | YES | sealed maker 4,308 picks ROI +2.22% CI [+1.57,+2.84]; live maker 45 filled, 1 loss, +$2.32 (`kalshi_exec.py status` 11-10); audit maker-only 39 fills ROI +1.72% [-4.33,+6.11] -> not significantly negative; tape through-price >= 2x on 32/39 fills | STEP 1: 6 -> 12 ct/order |
| E4 X1 cities | no | sealed taker +0.78% only; maker trial needs_audit; already paused (KALSHI_COHORTS=orig7) | none |
| B5 NHL late taker | no | live 3-3, -$1.92, ROI -39% [-80,+2.7]; paper 5-4 -25.7% | stays $1 |
| L39 / TSA / DUL | no | 0 live settled | stay at preset stakes |
| E5 ForecastEx | no | adverse selection (stop proposed) | none |
| NFLF | no | disabled | none |

E4 math (r = 0.0157): at c .96/.97/.98 (fill quantiles .10/.25-.50/.75-.90, results/e4_live_audit/q4_tape.csv),
1/4-Kelly per bet = 6.0% / 12.7% / 19.2% of bankroll. Bankroll $384.56 (exchange 11-10). Step 1 per order 12 ct x ~.97
= ~$11.70 = 3.0% of bankroll, well under 1/4 Kelly -> depth and correlation bind, not Kelly. Night basket cap 20% x
$400 = $80 (was 10% = $40; busiest night at 6 ct was ~$41 so 12 ct needs ~$82).

.env changes (backup .env.bak-2026-11-10-preKelly1):
KALSHI_MAX_CONTRACTS 6 -> 12; KALSHI_MAX_ORDER_USD 6.20 -> 12.40; KALSHI_ORIG7_FRAC 0.02 -> 0.031 (12.40/400);
KALSHI_NIGHT_FRAC (default .10) -> 0.20; KALSHI_MAX_DAILY_USD 60 -> 80; KALSHI_MAX_OPEN_USD 80 -> 100.
Unchanged: daily loss stop 5% of B ($20), weekly 10%, depth_frac .5, maker style, X1 paused.
Next earliest step: 2026-11-24, if the live record since 11-10 is not significantly negative.

## Applied 2026-11-12 -- E4 step 2 (USER WAIVED the 14-day spacing: "we could increase past 11/24 unless you really see
## some probability saying no")
Record since step 1 (`kalshi_exec.py status` 11-12 03:10Z): maker 47 filled, filled loss rate 2.2% of 45, P&L +$2.98
(+$0.66 24 h); total live +$2.56; balance $382.72. Not significantly negative.
Size: 12 -> 24 ct/order (= 4x the original 6; tape through-price at 4x on 29/39 fills -> depth ceiling; no step 3 above
24 without new depth evidence). Per order ~$23 = ~6% of bankroll vs 1/4 Kelly ~12% at c .97. Worst plausible night
(2 losses) ~-$47 = 12% of bankroll. Daily loss stop 5% ($20) unchanged: one loss at 24 ct trips it (24 h pause) -- an
intended brake.
.env (backup .env.bak-2026-11-12-preKelly2): MAX_CONTRACTS 24, MAX_ORDER_USD 24.80, ORIG7_FRAC .062, NIGHT_FRAC .30,
MAX_DAILY_USD 120, MAX_OPEN_USD 120. Revert to step 1 if E4 loses >= 10% of bankroll (~$38) since 11-12 or a kill fires.

Critic 11-15 (critic_e4_step2): KEEP 24, no step 3. Corrections: r = 1.57% is the POOLED E1+X1 sealed CI; orig-7 (E1)
alone sealed 1,844 picks ROI +2.87%, fill .857, no E1-only CI on record (to compute). Live orig-7 fill 30/44 (68%) vs
sealed .857 -> sealed fill model optimistic. Live 1 loss in 95 paper picks vs ~2 expected = lucky. Bad night 1-in-100
~-$23, 1-in-1000 ~-$43 at 24 ct; max ~3 losses/night. Ops issues -> agent e4_ops_diag: no picks 11-13/14, insufficient
balance rejects 11-15 05:03Z, maker depth cap not enforced.

## E4 math redone on E1 (orig-7) only -- agent e4_evidence 2026-11-15 (notes/e4_evidence.md, results/e4_evidence/)
Source: per-pick M3_N10 vector RE-DERIVED from the frozen sealed inputs of the 10-23 lockbox run (data sha e980f9c66792
asserted; recorded E1 1844 picks / fill .8574 / ROI .0287 reproduced exactly; no new scoring, nothing written to
lockbox.jsonl). Date-cluster bootstrap (406 dates, 4000). results/e4_evidence/t1_e1_ci.json.
- E1 sealed ROI +2.87%, 90% CI [+1.98, +3.71], 95% [+1.84, +3.87]. 55 losses, all filled (0 unfilled losses).
- Live orig-7 maker (kalshi_exec.jsonl -> results/e4_evidence/live_maker_orders.csv): 29 filled / 47 (61.7%; critic
  counted 30/44 = 68%); losses: filled 1/29, unfilled 0/18 -> adverse model (losses always fill).
  The shortfall is at q .98: live 15/30 filled vs sealed 420/505; q < .98 live 14/17 vs sealed 1161/1339.
- Fill-adjusted ROI, model A (every filled loser kept, winners thinned to the live fill rate; pre-registered as the
  sizing model because live unfilled loss rate <= filled): f .68 -> +1.91%, 90% [+0.80, +2.96]; f .617 -> +1.43%,
  90% [+0.23, +2.58]. Model C (shortfall only on q .98 winners, what the live split suggests): f .68 +3.11% [+1.97,+4.15].
1/4 Kelly per bet (c r/(1-c)/4) vs 24 ct x c / $382.72 = 6.0-6.2% of bankroll:
| r (90% lo) | c .96 | c .97 | c .98 |
|---|---|---|---|
| E1 sealed 1.98% | 11.9% | 16.0% | 24.3% |
| E1 fill-adj A, f .68: 0.80% | 4.8% | 6.5% | 9.9% |
| E1 fill-adj A, f .617: 0.23% | 1.4% | 1.9% | 2.9% |
Verdict: on the sealed E1 bound, 24 ct is about half of 1/4 Kelly at every price. On the pre-registered conservative fill
model with the critic's 68% fill, 24 ct is inside 1/4 Kelly at c .97-.98 and ~25% over at c <= .96; with the 61.7%
recount it is 2-3x over at all prices. The binding uncertainty is the fill model (47 live orders), not the sealed edge.
PROPOSAL (lead/user, not applied): keep 24 ct for c >= .97; cap c <= .96 orders at 1/4 Kelly under model A f .68
(~19 ct at c .96); no step 3. Re-check when live orig-7 maker orders reach ~100.
Correction: the 11-10 E4 math lists 1/4 Kelly 6.0% at c .96 for r .0157; c r/(1-c)/4 = 9.4% (12.7%/19.2% are right).
X1 (17 cities): sealed cohort re-derived the same way: ROI +1.67%, 90% CI [+1.01, +2.29] (232 dates;
results/e4_evidence/t2b_x1_sealed_ci.json), so item 1 is met on its own cohort. Forward since the pause (paper, 94 picks,
10 nights 11-03..11-12, 3 losses): conservative tape-through ROI +7.1%, 95% CI [-10.7, +26.4]; N10 rule +5.7%
[-7.5, +19.7] (results/e4_evidence/t2_x1_forward.json). Live X1 before the pause: 21 fills, 0 losses. Item 2 (95% upper
> 0) is met. X1 is eligible at minimum units if the user restarts it (the 11-03 pause is the user's call).
