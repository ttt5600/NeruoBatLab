# bar_calibration -- experiment A of bar_review.md: placebo + planted-edge calibration (agent, 2026-11-15)

Read-only on edgelab/ and GOAL.md (locked). edgelab.evaluator.score is IMPORTED and called unchanged on synthetic frames;
family bars use registry.family_bar's formula (alpha / family trials) with counts read read-only from the real registry.
No trial registered: this measures the instrument, not an edge (same status as bar_review exp B). Code, seeds, results:
results/bar_calibration/.

## Question
For each decision rule -- CURRENT (edgelab rule 5 at the family bar, every stage) and PROPOSED (bar_review s2) -- what is
the false-pass rate on fair-priced placebos and the power on planted edges (+1/+2/+4% ROI), at n 100/300/1000, and
what does that mean in months at our real volumes?

## Pre-registration (written before any simulation output was looked at)
Copied from bar_review.md exp A:
- success: the proposed rule's false-pass 95% Clopper-Pearson upper <= 1.5 x its nominal alpha in every regime, AND its
  power at +2% is >= 20 points above rule 5 with non-overlapping CIs (in at least the regime it is meant for).
- kill: the proposed rule's false-pass exceeds its nominal alpha with the CI excluding it in any regime -> do not
  propose it to the user.
Nominal alpha: CURRENT = the family bar (that is the per-trial alpha it claims); PROPOSED = the stage alpha
(dev = family bar, sealed = .05/(sealed shots incl. this), forward = .05 sequential).
Added before running: a stress regime for each rule's weak spot -- (a)/(c) loss clustering 3x the dev estimate that the
rule assumes; (b) a biased Kalshi close (+0.5c, +1c in our favour, outcomes not following it). These are reported as
"misspecified" rows; the kill applies to the correctly specified regimes, the stress rows say what experiment B / the
design-effect estimate must guarantee.

## Parameters from our data (each cites its source)
- Family counts (Registry().family_counts(), read-only, 11-15): kalshi_weather_temp 59 -> bar 8.47e-4; nba_totals 37 ->
  1.35e-3. Sealed evaluations ever: kalshi_weather_temp 4, nba_totals 0. min_n from specs: weather 200, l32b_k1 100.
- E4 (a): live orig-7 filled prices (results/e4_evidence/live_maker_orders.csv, 29 fills): .96 x3, .97 x7, .98 x15 among
  96-98c -> price mix used. Volume ~55 fills/month (e4_maker_e1 dev frame 871 fills / 15.9 months = 54.9/mo).
  Maker fee 0 (quadratic series).
- Night clustering: dev dispersion ratio of losses per night vs independent = 1.095 (e4_maker_e1 dev) / 1.124 (e1_taker
  dev) -> ICC ~.031 at weighted night size ~4-5; X1 dev -.018 (11 losses, noise). Max losses/night 2 in dev.
- X1 (c): dev frame lab/edgelab_runs/e4_maker_x1-e02827f2_dev.parquet: 368 fills, 45 nights, 2026-01-15..02-28, price
  q10/50/90 .89/.96/.98, ramp by ISO week; prices and weekly weights resampled from it. ~250 fills/month at tape fills.
- Sports (b): results/k50_strike/rows_*.csv CLV SD 2.69 pts (NBA n202) / 2.54 (CFB n776); 2.14 / 2.59 c per pt
  (clv_value.json) -> per-bet price-space CLV SD ~5.8c (stress 7.6c = cbb_opener 3.3 pts). Cost ~.53 (ask ~.5125 + taker
  fee). ~6-9 bets per date; ~50-75 bets/month in season per league-rule (k50 nba k3.0 25 in 10 days; cfb k3.5 71 in 41).

## Plan
results/bar_calibration/sim.py (seeded): generators for (a) (b) (c); rules: current = evaluator.score; proposed =
clustered exact binomial (Kish design effect, betainc) / SPRT (near-certain), net-CLV cluster t + ROI-consistency
(sports), leave-top-week/top-cluster-out; stage bars. Clopper-Pearson CIs on every rate.

### Wrong turns / fixes before the full run (quick runs at 10% reps; numbers not quoted as results)
- Night clustering re-estimated properly (centred per night on the scaled expectation, pooled): dispersion .90
  (e4_maker_e1 dev) / .94 (e1_taker dev) nights, .79 / .92 city-months -> NO detectable overdispersion in dev. The
  earlier 1.095 / 1.124 came from a common-mean variance (heterogeneous night sizes inflate it). Kept ICC .031 as the
  base (conservative: what the critic's "losses co-occur" implies) and x3 + a city-month ICC .01 as stress.
- First generator (gamma multiplier on L) made the ICC grow with L (ICC ~ L v/(1-L)): ~.5 for X1's 70-80c fills, so
  the X1 placebo false-passed ~6-8% at .05 -- a generator artifact, not the rule. Replaced with a shared Exp(1) shock
  scaled per fill for a constant ICC and divided by its clipping mean: verified realized ICC .027 (e4) / .030 (x1),
  stress .065 / .075, mean loss - L0 within 2 SE of 0 (600 sims each, inline check).
- Added a data-driven design effect to the proposed exact test: max(dev-ICC prior, Pearson dispersion of night loss
  counts around the observed level). Both versions are reported.
- Column-name collision (rate's "n" overwrote the sample size) fixed before any result was read.

## Results (full run: seed 20261115, 5,360 tasks, 792 s; results/bar_calibration/{fixed_n,sprt,n80}.csv, report.txt)
Reps: proposed 20,000 per placebo cell / 4,000 per planted cell; current rule (evaluator.score) 2,000 / 1,000 (stress
1,000). Rates below are base regime unless labelled; [95% Clopper-Pearson]. "pass|audit" = current verdict pass OR
needs_audit (generous to the current rule). Nominal: current = family bar (8.47e-4 wx, 1.35e-3 nba_totals);
proposed = dev family bar / sealed .05/5 = .01 (wx), .05/1 (nba_totals) / forward SPRT .05.

### 1. The CURRENT rule is mis-calibrated in both directions
- Too loose on placebos (E4-like 96-98c): pass|audit .0050 [.0024,.0092] at n300, .0040 [.0017,.0079] at n1000 vs
  nominal .00085 -> 5-6x, CI excludes nominal. Strict "pass" alone .0035 [.0014,.0072] at n1000 (CI excludes too).
  Driver: the cluster-bootstrap p (check a) is anti-conservative: p < bar on 7.6% / 1.2% / 0.6% of E4 placebos at
  n100 / 300 / 1000 (9-90x the bar), and p < .05 on 6.8-10.7% (E4) and 8.4-9.7% (X1). Other checks mask part of it.
- Too strict on real edges: E4 +2% "pass" .201 at n1000 and .122 at n2000 -- FALLS as n grows because the Sharpe > 5
  guard turns real near-certain edges into needs_audit (+4%: pass 0.000, pass|audit 1.000). Sports +2%: pass .002 at
  n1000, .025 at n2000 (the 30% profit-share cap passes 0-40% of real +2% samples, n100-2000). X1 7-week +2%: pass 0.000 at n100-1000.
### 2. PROPOSED rule: calibrated in every correctly specified regime except one small-n cell
- E4 placebo: dev .0005 [.0002,.0009] (nom .00085); sealed .0046-.0069 (nom .01); fwd fixed .021-.039 (nom .05).
- X1 placebo: dev <= .0001; sealed .0029-.0037; fwd fixed .025-.029.
- Sports placebo: sealed/fwd .047-.048 [CI upper <= .0513] (nom .05); dev n300-2000 .0011-.0014 (nom .00135);
  dev n100 .0019 [.0013,.0025] = CI upper 1.85x nominal (fails the 1.5x success clause; the t-statistic alone
  .0021 [.0015,.0028] excludes nominal). Cause: cluster-robust t with ~14 date clusters at a 1e-3 tail. Fix: >= 30
  date clusters (n200: .0015 [.0010,.0021]; n300 fine) or an exact sign-flip test.
- SPRT (forward): E4 placebo false-accept .0535 [.0492,.0581] (nom .05), sports .0431 [.0392,.0473].
### 3. Power at +2% and time to a decision at real volumes (E4 55 fills/mo, sports 60/mo in season, X1 250/mo)
| regime | current pass (pass|audit) | proposed sealed | proposed fwd |
|---|---|---|---|
| E4 n300 | .000 (.561) | .485 | .755 fixed; SPRT 89% decided H1 by 300 fills |
| E4 n1000 | .201 (.994) | .998 | 1.000 |
| sports n300 | .000 (.004) | .799 | .799 fixed; SPRT 71% by 300 |
| sports n1000 | .002 (.008) | .956 | .956 |
| X1 7wk n1000 | .000 (.119) | .313 | .638 (dev bar: .062) |
- E4 SPRT: +2% -> H1 96.0%, mean 193 fills = 3.5 months, 91.7% within 6 months; placebo -> H0 in mean 78 fills
  (1.4 mo). (bar_review's 84% / 299 fills used c .965; our live mix is 60% .98 fills, where +2% means L .02 -> .0004.)
- Sports SPRT: +2% -> H1 84.2%, mean 184 bets = 3.1 in-season months; +1% only 37%.
- Fixed-n 80% power at +2%: E4 proposed sealed n500 (~9 mo) vs current pass never (pass|audit n1000, ~18 mo);
  sports proposed sealed n500 (~8 mo) vs current not reached by n2000 (~33 in-season months).
- X1 dev: the dev family bar still binds (proposed dev .062 at n1000) -- the rule change does NOT rescue X1's dev
  verdict; its route is forward SPRT on new nights.
### 4. Stress (misspecified) regimes: what the user's approval depends on
- Biased Kalshi close (sports): +0.5c bias in our favour -> proposed sealed false-pass .165 (n100) .. .707 (n1000);
  SPRT .348; +1c -> .374 .. .903, SPRT .809. The ROI-consistency check barely helps. => net-CLV-primary is valid only
  if experiment B bounds the close bias well under ~0.25c.
- CLV SD 7.6c while the SPRT assumes 5.8c: SPRT false-accept .114 -> estimate sigma from data (or use the stress sd).
  Fixed-n t-test unaffected (.048).
- Loss clustering x3 + city-month (near-certain): E4 fixed fine (sealed .006-.011, fwd .027-.050); E4 SPRT .0632
  [.0585,.0681] (excess); X1 packed 7 weeks at n1000: sealed .023, fwd .081 (excess). The prior-only design effect is
  worse (E4 sealed n2000 .0118 [.0104,.0134]); the data-driven night dispersion helps but cannot see city-month
  clustering -> estimate the design effect at night AND city-month level before adopting.
- Current rule under the same stresses: still near 0 false-pass on sports (its concentration cap rejects everything),
  ~.003-.004 on E4 (as in base).

### Verdict vs the pre-registration
- Kill (proposed false-pass CI excluding nominal in a correctly specified regime): NOT triggered for the composite
  rule; triggered for the bare t-statistic at sports dev n100 (fix above).
- Success: power clause met for E4 (n1000 sealed .998 vs pass .201; n300 .485 vs .000) and sports (n300 .799 vs
  .000) with non-overlapping CIs; NOT met for X1 dev. Calibration clause fails in one cell (sports dev n100, 1.85x).
  => PARTIAL: holds for E4 and sports at n >= 200 / >= 30 date clusters, conditional on stress conditions in s4.
- Not yet critic-reviewed (lead: run /finding before quoting as confirmed).

### What the user would be approving (GOAL.md amendment; nothing applied)
1. Stage-specific alpha: dev keeps the family bar; sealed at .05 / (sealed shots in the family incl. this one);
   forward = Wald SPRT at alpha .05 / beta .2 (H1 = +2% ROI) with no family charge, replacing "100 forward bets passing
   rule 5 at the family bar".
2. Near-certain contracts (cost >= ~.90): primary test = exact binomial on filled losses vs break-even 1 - c, with
   design effect = max(dev ICC prior, observed night dispersion) -- and a city-month check before adoption.
3. Sports: primary = net CLV (Kalshi close - fill - fee, filled only, date-cluster t, >= 30 dates), ROI as the
   consistency check -- ONLY after experiment B shows the Kalshi close is calibrated (bias << .25c).
4. Concentration: replace the 30% profit-share cap with "statistic > 0 after dropping the top week and, separately,
   the top cluster".
5. Too-good: drop Sharpe > 5 as an audit trigger for near-certain contracts (it fires on every real edge here).
Risk the user accepts: forward false-accept 5% per strategy (vs ~0.1% claimed now, ~0.4-0.5% actual); a biased
close or under-estimated clustering raises it (s4).

## Learnings
### edgelab rule 5 false-passes near-certain placebos 5-6x its bar
Fair 96-98c placebos: pass-or-audit .0050 [.0024,.0092] at n300 and .0040 [.0017,.0079] at n1000 vs bar .00085.
The cluster bootstrap p < .05 on 6.8-10.7% of E4 placebos; p < bar on 7.6% at n100. Source results/bar_calibration/fixed_n.csv (seed 20261115).
### The Sharpe > 5 guard turns real near-certain edges into needs_audit
E4-like +2%: strict pass .201 at n1000 and .122 at n2000 while pass-or-audit is .994 / 1.000; at +4% pass is 0.
More edge = fewer passes. Source results/bar_calibration/fixed_n.csv.
### Current rule has ~0 power on 50c sports edges at any realistic n
+2% ROI at n2000: pass .025; the 30% profit-share cap passes 0-40% of real-edge samples (n100-2000). Proposed net-CLV rule:
.799 at n300 (sealed alpha .05). Source results/bar_calibration/fixed_n.csv.
### Net-CLV-primary is only as good as the close's calibration
A +0.5c bias of the Kalshi close in our favour makes the proposed sports rule false-pass 17-71% (n100-1000) and the
SPRT 35%. Experiment B must bound the bias well under 0.25c. Source results/bar_calibration/fixed_n.csv, sprt.csv.
### Loss SPRT decides E4 in ~3.5 months at live volume
H1 +2% accepted 96% (mean 193 fills / 55 per month); placebo false-accept .0535 [.0492,.0581]; 3x clustering pushes it
to .063. Source results/bar_calibration/sprt.csv.
### Cluster-robust t is anti-conservative at 1e-3 tails with ~14 clusters
Sports dev n100 (~14 dates): .0021 [.0015,.0028] vs nominal .00135; fine from ~30 dates. Require >= 30 clusters at dev
bars. Source results/bar_calibration/fixed_n.csv.
### Dev data show no loss overdispersion by night or city-month
Pooled Pearson dispersion .90 / .94 (nights) and .79 / .92 (city-months) in e4_maker_e1 / e1_taker dev; a common-mean
variance overstates it (1.10 / 1.12). Inline check on lab/edgelab_runs/*_dev.parquet.
### A multiplicative loss shock makes clustering grow with the loss rate
Gamma multiplier on L gives ICC ~ L v/(1-L): ~.5 at 75c fills, a fake 6-8% false-pass in X1 placebos. Simulate with a
per-fill-scaled shared shock and check realized ICC and mean before trusting a placebo study (bar_calibration sim.py).
