# bar_calibration_v2 -- experiment A revised per the critic (agent, 2026-11-15)

Revises notes/bar_calibration.md after inbox row `critic_bar_calibration` (ACCEPT-WITH-CHANGES). Read-only on edgelab/
and GOAL.md. No trial registered (instrument calibration). Code/results: results/bar_calibration_v2/ (sim2.py, seeded).

## Critic points addressed
1. H1 for near-certain contracts was L1 = 1 - c(1+r): degenerate at c >= 1/(1+r) (99c fills, 13/44 live orig7 fills).
   -> H1 stated as a LOSS-RATE RATIO: true loss prob = rho x break-even (1 - c_i). ROI = (1 - rho) x mean((1-c)/c).
   Real price mixes: live orig7 fills (results/e4_live_audit/live_rows.parquet, 44 filled, cost = avg_no + fee/filled,
   incl. 13 at .99 and 9 below .96) and sealed E1 (lab/edgelab_runs/e1_taker-40389097_sealed.parquet, 1,844 bets,
   cost = price + taker fee). True edges include the E1-sized edge (sealed loss ratio .684 = +1.45% ROI) and +1%/+1.5%.
   Report the SPRT's KILL (H0-cross) probability at each true edge.
2. Seasonality (monthly volume profile from the E4 sealed frame; forward start month drawn uniformly) + month-level loss
   shock + 3x night clustering stress. SPRT alpha tuned so stress placebo false-accept <= nominal.
3. Family control: 'forward SPRT only after a sealed pass' vs '.05 per family per year across forward starts', at our
   real trial rate (registry).
4. Sports: drop net-CLV as primary; ROI-based sequential test (win/loss SPRT vs break-even at cost), CLV diagnostic only.

## Pre-registration (written before any v2 simulation output)
Near-certain (forward SPRT, per mix):
- Tuning rule (placebo only, fixed before looking at power): for each (rho1 in {.5,.6,.7}, beta in {.1,.2}), the design
  alpha a is the LARGEST of {.05,.04,.03,.025,.02,.015,.01,.005} whose STRESS-regime placebo false-accept CP 95% upper
  <= .05. Then validated on fresh-seed placebos (base + stress).
- Config choice rule: among tuned configs, the one with the LOWEST kill probability at the E1-sized edge (rho .684,
  stress regime), subject to P(accept H1 within 24 months) >= .50 at that edge; ties -> shorter median time.
- success: chosen config has validated stress placebo false-accept CP upper <= .055 AND E1-edge kill <= .20 AND
  E1-edge P(H1 within 36 months) >= .60, on BOTH mixes.
- kill: no config meets placebo <= .05 with E1-edge kill <= .20 -> do not propose a forward SPRT for E1-sized edges;
  propose a fixed-n forward test with its n instead.
Sports (forward ROI SPRT): report P(H1)/P(H0)/months at edges -2..+5%; tuned the same way on placebo.
- 'infeasible' verdict (pre-set): if median in-season months to decide a +2% edge > 24 at 60 bets/month, state that
  50c sports edges of that size cannot be promoted on forward ROI within the 12-month mandate.

## Parameters (sources)
- Forward-evaluated trials since 2026-11-02 (lab/edgelab_registry.jsonl, event=evaluate split=forward): 16 events,
  11 distinct trials, 5 families; sealed passes 2 (e1_taker, e4_maker_e1); trials ever in forward_paper: 1.
- E1 sealed: loss .02983 vs break-even .04363 -> ratio .684, ROI +1.455%. E4 maker sealed: ratio .566, ROI +2.87%.
- Mean odds (1-c)/c: live orig7 mix .0439, E1 sealed mix .0505 -> +1% ROI = rho .772 / .802; +1.5% = .658 / .703.
- Month-level loss dispersion: sealed E1+E4 pooled Pearson 1.51 (v = .118, 16 months, same picks in both frames);
  dev .85 (v 0). Seasonal volume profile (E4 sealed by calendar month, mean 1): Jan-Apr 1.5-1.9, Nov-Dec .19.
  Forward volume 55 fills/month mean (v1). All in results/bar_calibration_v2/params.json.

## Amendment 1 (after a 10%-reps quick run, BEFORE the full run; disclosed)
Quick run (numbers not quoted as results): stress placebo false-accept was .03-.05 even at design alpha .005, and the
BASE regime gave ~.09 at alpha .05. Diagnostic (results/bar_calibration_v2/diag.py, live mix, rho1 .6, beta .2, 3,000
reps each): iid/flat .054 at a=.05; seasonal volume alone .054; night ICC .031 alone .044; month shock v .118 alone
.084; both .086. => the MONTH-level loss shock is what breaks the SPRT: a lucky low-loss month drives the LLR over A.
Night clustering and seasonality do not. Change: add a min-information rule -- H1 may be accepted only after M
forward months (M in {0, 6, 12}; kill allowed any time) -- and extend the alpha grid to .0025/.001. Tuning and choice
rules otherwise unchanged (M is one more config dimension).
Month v used: base .118 (sealed), stress max(3 x .118, .15) = .354.

## Results (full run: seed 20261116 (+1 validation, +2 sports), 175 s; results/bar_calibration_v2/)
Files: nc_grid.csv (all configs x edges x regimes; placebo 20,000 reps, edges 3,000), nc_tune.csv, nc_valid.csv
(fresh seed: placebo 20,000, E1 edge 4,000), sp_grid.csv (placebo 4,000, edges 1,000), family.json, params.json.
Rates are [95% Clopper-Pearson]. Volume 55 fills/month mean with the seasonal profile; horizon 36 months.

### 1. Month-level loss clustering, not night clustering, sets the forward alpha
- Base regime (night ICC .031 + month v .118) at design alpha .05: placebo false-accept .087 [hi .091] (E1 mix) /
  .077 [hi .081] (live mix) -- v1's .0535 had no month shock. Largest calibrated design alpha (rho1 .7, beta .1): base
  .025 (E1, .048 [hi .051]) / .03 (live, .049 [hi .052]); STRESS (ICC x3, month v .354): .005 (.045 / .034); .002 for
  rho1 .5-.6 on the E1 mix. The min-information rule (H1 only after 6/12 months) barely helps (nc_tune.csv).
### 2. Chosen config by the pre-registered rule: rho1 .70, beta .10, design alpha .005, M = 0 (nc_tune.csv)
No config met the choice constraint P(H1 <= 24 months) >= .50 at the E1-sized edge (best .41), so the pick is the
lowest E1-edge kill among stress-calibrated configs. Fresh-seed validation (nc_valid.csv):
| | E1 mix stress | E1 mix base | live mix stress | live mix base |
|---|---|---|---|---|
| placebo false-accept | .0446 [.0418,.0476] | .0110 [.0096,.0125] | .0337 [.0312,.0363] | .0088 [.0075,.0101] |
| E1-size edge (rho .684): H1 by 36 mo | .615 [.600,.630] | .610 [.594,.625] | .546 [.531,.562] | .518 [.502,.533] |
| E1-size edge: KILL (H0 cross) | .173 [.161,.185] | .096 [.087,.105] | .162 [.151,.174] | .096 [.087,.105] |
| E1-size edge: H1 within 24 mo | .394 | .340 | .309 | .244 |
Other edges, stress, E1 mix (nc_grid.csv): rho .5 (+2.5%) H1 .970 kill .019, median 16 mo; rho .566 (E4 sealed ratio,
+2.2%) H1 .887 kill .050; +1% (rho .80) H1 .29 KILL .43; +0.5% kill .66; placebo kill .835; -1.3% (rho 1.25) kill .989
(median 7 months). Seasonal edge (1.6x Dec-Mar) moves none of these by more than ~.03. Live mix: same pattern, ~2
months slower (99c fills carry almost no information: (1-c)/c = .010).
Alternative (base-tuned alpha .02): base placebo .039 / .033, E1 edge H1 .77 / .71, median 18-20 months, kill .09;
but stress false-accept .107 / .090 -> defensible only if month overdispersion is shown to be ~0.
### 3. Verdict vs pre-registration
- kill NOT triggered (a calibrated config with E1-edge kill <= .20 exists).
- success NOT met: live-mix E1-edge P(H1 by 36 mo) .546 < .60 (E1 mix .615 passes); choice constraint failed.
- => PARTIAL. A calibrated forward SPRT exists and does not routinely kill an E1-sized edge (kill .10-.17), but at 55
  fills/month it needs ~21-25 months (median) to accept it and leaves ~20-30% undecided at 3 years. A +1% edge is
  killed more often (.43) than accepted (.29). It guards against decay; it is not a fast promotion route.
- Fixed-n comparison (analytic, approximate, not simulated): E1 edge at alpha .05 / 80% power needs ~1,400 x deff ~1.3
  = ~1,800 fills = ~33 months -> the SPRT is faster but not fast.
### 4. Family control (family.json, registry read-only)
Forward-evaluated trials since 11-02: 11 distinct (16 events; 5 families), nearly all from the 11-02 retroactive import;
one new forward evaluation since (e4_maker_e1, 11-10). Sealed evals 7 (weather 4, data_ladders 2, nba_sides 1), passes
2. Expected false accepts/year if every trial were null, forward alpha .05, planning rate 11 forward starts/yr:
- no family charge (v1 proposal): .55
- .05 per family per year across forward starts (5 families): <= .25
- forward SPRT only after a sealed pass: .05 x sum_f .05 H(k_f) = .011 (H = harmonic number).
Side finding: sealed alpha .05/(shots incl. this) is NOT a .05 family bound -- it sums to .05 x H(k): weather (4 shots)
= .104. Spending .05/2^j (j-th sealed shot in the family) keeps the family total <= .05.
Recommendation: sealed-pass gate. Trials that failed dev/sealed (X1, L32) may log forward paper but cannot be promoted by
a forward SPRT unless the user grants an exception charged to that family's .05/year budget.
### 5. Sports: ROI-based forward SPRT (win/loss vs break-even at cost; date ICC .05 / stress .15; sp_grid.csv)
Placebo false-accept CP upper <= .028 at design alpha .05 in every cell (truncation makes it conservative).
At 60 bets/in-season month, H1 r1 +3%, beta .2: true +2% -> H1 .27-.30 within 120 in-season months, kill .30, median
to H1 ~75 months; +5% -> H1 .98-.99, median 45 months; -2% killed .99-1.0 (median ~24 months; ~11 with r1 +5%).
At 200 bets/month: +2% H1 .56-.59 (median 35 mo); +5% H1 .99 (median 14 mo, .38 within 12 mo).
=> pre-set 'infeasible' verdict TRIGGERED: a +2% 50c sports edge cannot be promoted on forward ROI within the 12-month
mandate at our volume; only >= +5% edges at >= 200 bets/month decide in ~1 year. CLV stays a diagnostic (reported,
never a gate). The ROI SPRT's practical value for sports is the kill side.

## Amendment text for the USER (GOAL.md rules 5/6/7; nothing applied; edgelab would need code changes)
> **Rule 6 amendment (proposed 2026-11-15):** Replace "`forward_paper` -> `live_small` needs >= 100 forward bets passing
> rule 5 on forward data alone" with: "`forward_paper` -> `live_small` needs the trial's pre-registered forward sequential
> test to accept H1 on forward data alone. Only trials that reached `sealed_pass` may enter `forward_paper`; a trial that
> failed dev or sealed may run a paper logger but its forward data cannot promote it, unless the user grants a named
> exception charged to that family's forward budget of alpha .05 per calendar year (split equally among that year's
> exceptions). Sealed alpha for the j-th sealed shot in a family = .05 / 2^j."
> **Forward test, near-certain contracts (cost >= .90):** Wald SPRT on filled losses; H0 loss prob = 1 - c_i (c = fill
> price + fee), H1 = 0.70 x (1 - c_i); design alpha .005, beta .10; evaluated after each night; LLR divided by
> 1 + (mean fills per night) x .031. Crossing the lower boundary = the rule-7 kill. If neither boundary is crossed in 36
> months the trial stays at forward_paper and the user decides.
> **Forward test, other contracts (sports ~50c):** Wald SPRT on win/loss vs break-even at cost; H1 = +3% ROI (or +5% if
> registered so); design alpha .05, beta .20; evaluated per date; LLR divided by 1 + (bets per date - 1) x .05.
> Net CLV is reported as a diagnostic only and is never part of a pass or kill.
> **Rule 5 (dev/sealed) is unchanged by this amendment** (its separate defects are in notes/bar_calibration.md s1).
What the user accepts: forward false-accept <= ~.045 per sealed-passed null trial under 3x clustering (~.01 base);
an E1-sized true edge is killed 10-17% of the time and accepted within 3 years only 52-62% (median ~2 years at 55
fills/month); +1% edges are killed more often than accepted; sports promotion via forward ROI is effectively closed at
our volume. The 11-05 one-step 2x size amendment is untouched.

## Residual risks
- Month dispersion comes from 16 months of ONE pick stream (E1/E4 sealed share losses; Pearson 1.51, chi2 15 df, not
  significant; dev .85). If it is ~0, alpha .005 costs ~5 months of median decision time; if it exceeds 3x (.354),
  false-accept exceeds .05. Re-estimate after 12 forward months.
- Volume: times scale ~1/fills per month. Live orig7 had 44 fills in 18 days (~73/mo), but Nov-Dec are the seasonal
  trough (profile .19).
- Price mix drift: more 99c fills = less information per fill (live mix ~2 months slower than the E1 mix).
- Sports sims assume date ICC .05 (no estimate), 7 bets/date, months = in-season months.
- Break-even uses the recorded fill cost; fee or price-recording errors bias H0 directly.
- Not critic-reviewed. Amendment 1 (min-months) came after a quick-run look and turned out inert.

## Learnings
### A month-level loss shock, not night clustering, breaks the near-certain SPRT
Month extra-Poisson variance v .118 (sealed E1/E4 monthly dispersion 1.51) makes a nominal .05 SPRT false-accept
.077-.087; night ICC .031 alone gives .044. Calibrated design alpha .025-.03 (base) or .005 (3x stress).
Source results/bar_calibration_v2/nc_grid.csv, diag.py.
### Define the near-certain H1 as a loss-rate ratio, not +x% ROI
ROI = (1 - rho) x mean((1-c)/c); at 99c (1-c)/c = .010 so an ROI-based H1 is degenerate. E1 sealed rho .684 = +1.45%.
H1 rho1 .70 keeps an E1-sized edge's kill probability at .10-.17. Source results/bar_calibration_v2/nc_valid.csv.
### A calibrated forward SPRT needs ~2 years to accept an E1-sized edge at 55 fills/month
rho1 .7, alpha .005, beta .1: E1-edge H1 by 36 months .52-.62, median 21-25 months, 20-30% undecided; a +1% edge is
killed .43 vs accepted .29. Forward data guard against decay; they do not promote fast. Source nc_grid.csv.
### Sports forward ROI cannot promote a +2% edge within a year
50c bets at 60/in-season month: +2% accepted .27-.30 within 10 in-season years (median ~75 months); +5% at 200/month
median 14 months. CLV can only be a diagnostic. Source results/bar_calibration_v2/sp_grid.csv.
### Sealed alpha .05/(shots incl. this) sums harmonically
Family total = .05 x H(k): weather with 4 sealed shots = .104, not .05. Spend .05/2^j to bound it at .05.
Source results/bar_calibration_v2/family.json.
### Gate forward SPRTs behind a sealed pass
Expected false accepts/year if all trials were null: no charge .55 (11 forward trials), per-family-year budget .25,
sealed-pass gate .011. Source results/bar_calibration_v2/family.json.
