# e4_evidence -- E1-only sealed CI, fill-adjusted ROI, X1 forward score, Kelly check (agent e4_evidence, 2026-11-15)

## Questions (from lead, after critic_e4_step2)
1. E1-only (orig-7) ROI CI from the RECORDED sealed maker run (lockbox kalshi_temp_maker_M3_N10, 10-23), date-clustered;
   plus a fill-model-adjusted version using the live orig-7 fill rate (~68%) and live filled/unfilled loss behaviour.
2. X1 forward record since pause (paper picks, tape-through fill model) with CIs; SIZING_RULE eligibility?
3. Update SIZING_RULE.md E4 math with the E1-only lower bound; is 24 ct still <= 1/4 Kelly at c .96-.98?

## Pre-registration (written 2026-11-15 BEFORE any number below was computed)
Re-scoring guard: the lockbox per-pick P&L was not saved, only its frozen inputs (data/kalshi_temp_maker_picks_sealed.parquet,
data/kalshi_temp_maker_trades_sealed.parquet; lockbox sha e980f9c66792). I recompute the per-pick M3_N10 vector with the
SAME frozen functions (scripts/kalshi_temp_maker.py variant/taker; no new rule, no new variant, nothing appended to
lab/lockbox.jsonl, cmd_sealed not called). This is a deterministic re-derivation of an already-recorded result, valid only
if (a) the data sha equals e980f9c66792 and (b) E1 picks/fill/ROI reproduce the recorded 1844 / .8574 / .0287 exactly.
If either fails I stop and report blocked. It is NOT a new test; no verdict changes from it.
T1 stats: E1 ROI per $ on filled bets, cluster bootstrap over event dates (4000, seed 0), report 90% and 95% CI.
  Fill-adjusted (live fill f_live from kalshi_exec.jsonl orig-7 maker orders): model A "adverse" = every sealed filled
  loser stays filled, filled winners thinned at random so the overall fill rate = f_live (losers concentrate in fills);
  model B "neutral" = uniform thinning (ROI unchanged in expectation, CI widens). Which model is used for sizing is
  decided by the live filled vs unfilled loss behaviour: if live unfilled picks lost at a rate <= filled -> model A
  (conservative). Bootstrap = dates resampled + fresh thinning draw per replicate.
T2: X1 forward = kalshi_paper.jsonl cohort x1, logged_at >= 2026-11-04T03:00Z (first skip_cohort_paused order in
  kalshi_exec.jsonl), settled. q = M3 (as wx_capacity). Fill models: conservative = winner fills iff tape volume strictly
  through q (< q) >= 10 ct, loser always fills (verified in wx_capacity: 6/6 losers traded through by >= 920 ct);
  optimistic = <= q >= 10 ct (lockbox N10 rule). ROI per $ with date-cluster bootstrap; also the whole forward x1 record
  10-22..now for context. Eligibility (SIZING_RULE): item 1 needs sealed PASS or >= 100 fwd trades with 90% CI lo > 0;
  item 2 needs the live/forward 95% CI upper > 0. Kill for "X1 eligible": forward 95% upper <= 0.
T3: r = E1-only sealed 90% CI lower (and model-A version); f*/4 = c r/(1-c)/4 at c .96/.97/.98; compare with
  24 ct x c / bankroll.

## Log
### Live fill (results/e4_evidence/live_fill.py -> live_maker_orders.csv)
orig-7 live maker orders (excl. 2 rejected 11-15): 29 filled / 47 = 61.7% (critic's 30/44 = 68% used other exclusions;
both carried). Losses: filled 1/29 (MIA B84.5 q .85, 11-02), unfilled 0/18. Fill by q: q >= .98 15/30, q < .98 14/17.
X1 live before pause: 21 filled 0 losses, 17 no_fill 0 losses. X1 pause = first skip_cohort_paused 2026-11-04T03:03Z.

### T1 E1 CI (results/e4_evidence/t1_e1_ci.py -> t1_e1_ci.json)
Guard passed: sha e980f9c66792; pooled 4308/.8022/.022181 and E1 1844/.8574/.0287 reproduced exactly.
E1 sealed ROI +2.87%, 90% [+1.98,+3.71], 95% [+1.84,+3.87] (406 dates). 55 losses all filled; sealed fill q>=.98 420/505,
q<.98 1161/1339; filled ROI q>=.98 +2.04%, q<.98 +3.17%.
Fill-adjusted (bootstrap = dates + thinning draw): model A f .68 point +1.91%, 90% [+0.80,+2.96]; f .617 +1.43%,
90% [+0.23,+2.58]. Model B neutral ~+2.9% [+1.7/+1.8, +3.9/+4.0]. Model C (thin q .98 winners only) f .68 +3.11% [+1.97,+4.15].
Pre-reg picks model A (live unfilled loss 0/18 <= filled 1/29). Caveat: live split says the shortfall sits on q .98
picks (C), which would leave ROI unchanged or higher; A is a pessimistic bound, n is 47 orders.

### T2 X1 forward (results/e4_evidence/t2_x1_forward.py -> x1_forward_rows.csv, t2_x1_forward.json)
Post-pause (logged >= 11-04T03Z): 94 picks, 10 nights (11-03..11-12 ET), 3 losses (all traded through q by >= 922 ct).
Conservative (winner fills iff >= 10 ct strictly through q): 38 fills, ROI +7.1%, 90% [-8.5,+23.6], 95% [-10.7,+26.4].
Optimistic N10 (<= q): 53 fills, +5.7%, 95% [-7.5,+19.7]. All forward 10-20..11-12: 247 picks, 5 losses, cons +2.7%
95% [-5.1,+11.4], opt +2.6% [-2.2,+8.6]. Not distinguishable from 0 or from sealed; not significantly negative.
(wx_capacity's -$0.82/night at S24 used min(S, vol<q) dollar fills; different model, same small-n verdict.)
T2b (added after pre-reg, disclosed): X1 sealed cohort re-derived (guard: 2464/.761/.0167 reproduced): +1.67%,
90% [+1.01,+2.29], 95% [+0.87,+2.42], 232 dates. So X1 meets SIZING_RULE item 1 (own sealed CI lo > 0) and item 2.

### T3 Kelly -> SIZING_RULE.md section "E4 math redone on E1 (orig-7) only"
1/4 Kelly at c .96/.97/.98: sealed E1 r 1.98% -> 11.9/16.0/24.3%; model A f .68 r 0.80% -> 4.8/6.5/9.9%; model A f .617
r 0.23% -> 1.4/1.9/2.9%. 24 ct = 6.0-6.2% of $382.72. Proposal: keep 24 at c >= .97, ~19 ct at c <= .96, no step 3.
Also found: the 11-10 table's 6.0% at c .96 (r .0157) should be 9.4%.

## Learnings
### E1 (orig-7) alone has a tighter sealed edge than the pooled number used for sizing
E1 sealed maker ROI +2.87%, 90% CI [+1.98,+3.71] vs pooled E1+X1 lower bound 1.57% (results/e4_evidence/t1_e1_ci.json).
Sizing on the pooled bound understated E1's edge; X1 alone is +1.67% [+1.01,+2.29].

### For maker sizing the fill model dominates, not the sealed edge
Thinning only winners to the live 62-68% fill (losses always fill) cuts E1's 90% lower bound from 1.98% to 0.23-0.80%,
moving 1/4 Kelly from 2-4x above 24 ct to below it (results/e4_evidence/t1_e1_ci.json). Measure fill by price bucket.

### Live maker fill shortfall is at the 98c level, where losses are rare
Live orig-7 q .98 filled 15/30 vs sealed 420/505; q < .98 14/17 vs sealed 1161/1339 (live_maker_orders.csv). If that holds,
the shortfall removes low-ROI winners and the adverse-thinning model is too pessimistic.

### Lockbox per-pick rows can be re-derived without re-scoring when inputs are frozen
Assert the data sha and the recorded headline numbers before using a recomputed vector; save per-pick rows next time.
