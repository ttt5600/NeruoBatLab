# bar_review -- is edgelab's bar rejecting real small edges, and is it testing the wrong quantity? (agent, 2026-11-15)

Read-only review. Nothing in edgelab/ or GOAL.md was changed (locked; changes are the user's call). Every number below
comes from a file read or a command run in this session; the simulations are inline python (seeds given), not saved.
Sources: sports/GOAL.md, edgelab/evaluator.py, edgelab/run.py line 107 (one bar for every split), edgelab/registry.py
family_counts, notes/FUND_BOOK.md s1/s3/s4, notes/SIZING_RULE.md, notes/l39_expansion.md, notes/wx_capacity.md,
notes/LEARNINGS_2026-11.md, notes/cbb_opener_model.md, results/e4_evidence/live_maker_orders.csv.

Family counts right now (Registry().family_counts()): pattern_mine 2280, nfl_props 263, nba_sides 198, nfl_lines 136,
equities_options 79, kalshi_weather_temp 59 (the brief said ~46; it is 59 now -> bar 8.5e-4), kalshi_weather_precip 11,
cfb_lines 21, cbb_totals 4. Sealed evaluations ever run: kalshi_weather_temp 4, kalshi_data_ladders 2, nba_sides 1.

## 1. Diagnosis

### Too LOOSE: the ROI cluster bootstrap is anti-conservative when losses are rare
evaluator.bootstrap resamples observed clusters. With 0 losses there is no loss to resample, so p = 1/(B+1) whatever n
is. P1m: edgelab p < 1e-4, while the exact test is .96^94 = .0216, and .117 when clustered by city-month
(l39_expansion.md). Simulation through edgelab's own bootstrap() on FAIR-priced placebos (zero edge, seed 3, 1,000 reps):
n94 c.96 bar .00455 -> check (a) false-pass 2.2% (5x the bar; exact binomial 0.0%); n200 c.965 bar 8.5e-4 -> 0.7% (8x).
The rule-8 self-test placebos are coin-flip brackets and NFL sides, so they never hit this regime. In P1m, concentration,
the CLV and too-good guards are what killed it, not (a).

### Wrong quantity for near-certain contracts: "market-null" and exact binomial are the SAME test
For a NO bought at all-in cost c, ROI > 0 iff loss rate L < 1 - c. The P(losses <= k | L = 1-c) with per-bet costs
(Poisson-binomial) IS the market-null. Swapping to an exact binomial or a flat Beta posterior buys no power: posterior
P(L < .04 | 0/94) = .979 with a Beta(1,1) prior, which is 1 minus the same .0216. The power problem comes from alpha and
n, not from the test family. What IS wrong: (i) edgelab tests near-certain strategies with the bootstrap above, not this
test; (ii) the test must cluster by the unit losses share (night for E4, city-month for L39). Losses co-occur: E4's
worst plausible night has 2-3 losses (SIZING_RULE.md critic 11-15).

### Too STRICT: the same family bar is charged three times (dev, sealed, forward)
run.py line 107 uses reg.family_bar() for every split. Dev is the screen. Sealed is one pre-registered hypothesis on
independent data, so its p is valid at alpha / (number of SEALED shots in the family). Those shots are few (4 in
kalshi_weather_temp). Re-paying alpha/59 at sealed and again at forward double-counts the search.
Concrete consequence: GOAL rule 6 needs >= 100 forward bets passing rule 5. With ZERO losses, p = c^100: .028 at c .965
and .048 at c .97, so a perfect 100-bet record still fails any family bar below .03. For E4 at bar 8.5e-4 the exact
test needs ~1,130 fills at true ROI +2% (c .965, 80% power), about 20 months at the live rate of ~55 orig-7 fills/month
(87 orders 10-30..11-15, ~30 filled). E1 shows the contradiction: it is the golden sealed PASS, yet the retroactive dev
re-score marks it DEAD (dev p .0021 > .0011, EDGES_AND_AGENTS 11-02).

### Mis-specified: concentration measured as a share of NET profit
share = max cluster or week profit / total profit. With a small true edge the denominator is small, so the share mostly
measures noise. Simulated true-edge FAIL rates (seed 7, 4,000 reps, bets spread evenly over weeks):
- 95c contracts, n368 over 7 weeks, true ROI +2.3% (the X1 dev case): fails 33%.
- 96.5c contracts, n100 over 13 weeks, +2%: fails 20%.
- 50c sports bets, true ROI +2%: n300 over 13 weeks fails 93%; n1000 over 26 weeks fails 77%.
The question you want answered is "does the result survive without its best week?" That is a leave-top-cluster-out test,
not a share cap. The X1 fail (W09 = 0.357) is mostly this artifact (wx_capacity.md). TSA's shutdown week is the one real
concentration case.

### Mis-specified: the Sharpe > 5 guard fires on near-certain contracts by construction
X1 dev Sharpe 8.04, E4 maker about 7 (evaluator docstring). Daily clusters of 95c contracts with few losses always look
like this. The guard should run on a loss-rate posterior, not on Sharpe.

### CLV is a pass/fail sign check, not a statistic, so it cannot rescue or kill anything early
Check (e) only asks mean CLV >= 0. Gross CLV > 0 is the wrong bar: K50 had CLV +0.88 pts but ROI -2.2%, and the CLV
model predicted -2.1% (LEARNINGS line 136-138). Sports models carry ~0.8-0.9 pts against a taker cost of 1.2-1.4 pts.
The right statistic is NET CLV: close-implied EV minus the all-in cost, measured on the instrument we actually trade and
on FILLED orders only (maker filled CLV -0.90 vs unfilled +2.21, maker_fairfill). Power: per-bet CLV SD ~3.3 pts
(cbb_opener_model: n2,213, CI half-width .14). Detecting +0.5 pts of net CLV at alpha .05 with 80% power needs ~270
bets. Detecting the same edge on ROI (~+1.7%, per-bet SD ~.95) needs ~19,300. About 70x less data, valid only if the
venue's close is calibrated (see experiment B).

### Family definitions: mostly fine; the cost is in charging dev multiplicity at the confirmatory stages
pattern_mine 2280 and nfl_props 263 really are searched hypotheses, so their dev bars are honest. The bias comes from
two places. A new mechanism in nba_sides pays for 198 imported trials. kalshi_weather_temp mixes taker, maker, cohort and
LIP ideas. Splitting families would be a judgment call and is open to gaming. Counting sealed shots (above) fixes most of
the harm without that.

## 2. Pre-registerable alternative (PROPOSAL for the user; not applied)
Stages:
- DEV = screen: family bar unchanged (or BH q < .10 within the family).
- SEALED = confirm: one shot, alpha .05 / (sealed shots ever in the family, this one included).
- FORWARD = monitor: sequential, alpha .05, and no family charge. The hypothesis was already confirmed, so forward data
  checks for drift and execution decay, not for a discovery.
- Concentration: drop the share cap. Pass if ROI (or net CLV) > 0 with the top cluster AND the top week removed.
- Too-good: on a loss-rate posterior, not Sharpe.

A) Near-certain NO (E4, X1, L39, TSA, P1m):
- Statistic: losses k among filled n, tested against the Poisson-binomial of per-fill break-even L_i = 1 - c_i.
- Clustering: losses per night (E4) or city-month (L39), applied as a design effect estimated from dev.
- Sequential version: Wald SPRT with H0 L = 1 - c and H1 L = 1 - c(1 + r1). E4 (c .965, r1 +2%, alpha .05, beta .2;
  4,000 simulations, iid fills): true +2% accepts H1 84% of the time, mean 299 fills (~5-6 months at 55/month); a true 0%
  edge false-accepts 4.9%, mean 175 fills. At alpha .01: 86% power, mean 502 fills.
- Fixed-n reference (exact, 80% power): c .965, r +2% -> n 410 at alpha .05, 845 at .00455, 1,130 at 8.5e-4. Zero-loss
  minimum n = ln(alpha) / ln(c): 85 / 152 / 199 at c .965.
- Hard limit: ROI cannot exceed 1/c - 1 (3.6% at c .965), so at c >= .975 any edge above ~2.5% is impossible.
- Realistic monthly n: E4 orig-7 ~55 fills, L39 ~30-45 in Nov, TSA ~4-8. TSA and P1m stay underpowered for years under
  any rule.

B) CLV-positive sports (K1, K-NFL, K-CFB, L32, CBB):
- Primary statistic: net CLV = (Kalshi close mid - fill price - fee) on filled orders, date-cluster bootstrap.
- H0: net CLV <= 0. Fixed n 270 at alpha .05 for +0.5 pts (557 at the cfb_lines bar 3.1e-3; 750 for +0.3 pts).
- ROI stays the secondary check: its CI must not exclude the CLV-implied ROI.
- Precondition: experiment B passes.

## 3. Dead / held strategies under the corrected rule (NEW data only; dev/sealed rows already seen never re-count)
- X1 maker (dev fail = 7-week concentration artifact + Sharpe guard): the strongest case. Its sealed cohort is already
  +1.67% [90% CI +1.01, +2.29] (e4_evidence). Re-test = loss-rate SPRT on forward nights after the registration date.
  Zero marginal cost: the paper logger already runs.
- E4 orig-7 (live): same SPRT from the registration date. It does not re-litigate the sealed pass; it replaces the
  unpassable 100-bet gate.
- L32 NBA totals (sealed p .054 at ~25% power) and K-CFB/L36: forward net-CLV test on K1 / K-CFB forward rows only.
  L36 gross CLV +1.53 against a cost of ~1.2-1.4 pts means net ~+0.2, so expect underpowered (~1,500 bets).
- P1m: the clustered market-null p is .117 even under the honest test. It is "not distinguishable", not a victim of
  the bar. Forward-log as a column only.
- TSA T2: its concentration was a real regime week, and 4-8 fills/month is underpowered. Keep the B4 kill rule only.
- CBB opener (CLV +0.83 < 1.2 cost), K50, L34, KS12: the corrected rule kills these too. CLV-primary does not
  resurrect them.

## 4. Experiments (ranked by information per unit cost)

### A. Harness calibration: placebo + planted-edge study, current rule 5 vs the proposed rule
question:   For each rule, what is the false-pass rate on fair-priced placebos, and the power on planted edges?
            Placebos: 96c contracts at n 94-400; 50c sports bets. Planted edges: ROI +1%, +2%, +3%.
baseline:   edgelab/evaluator.score as locked (rule 5) at the real family bars; selftest.py placebo batches.
change:     the scoring rule only (stage alpha + clustered Poisson-binomial / SPRT + leave-top-out + loss-rate guard).
            The data is identical: real dev frames with outcomes re-drawn from the fair price, or the price shifted for
            planted edges.
metric:     false-pass rate (must be <= nominal alpha) and power, per regime, with 95% Clopper-Pearson CIs.
success:    the proposed rule's false-pass CI upper <= 1.5x alpha in every regime, AND its power at +2% is at least
            20 points above rule 5 with non-overlapping CIs.
kill:       the proposed rule's false-pass exceeds alpha (CI excludes it) in any regime -> do not propose it to the user.
power:      2,000 reps per cell -> the false-pass CI half-width at a 5% rate is ~1 point. Expected gaps are larger:
            2.2% vs 0.455% already seen; concentration fail 93% vs ~0.
confounds:  synthetic outcomes ignore real loss clustering. Re-draw outcomes per night/city-month with the dev ICC.
cost:       ~1-2 h CPU on the Mac, $0, no market data, no sealed access, sandbox registry (as in selftest).
cheapest version: the inline sims in this note (10 min): bootstrap false pass 2.2% at n94 and concentration fail 93%.
            Already done. The remaining open item is the clustered re-draw.

### B. Is net CLV an unbiased proxy for ROI on Kalshi sports? (instrument validation, not an edge trial)
question:   Pooled over already-seen dev/forward rows that have a Kalshi close, what is the slope of realized
            per-bet P&L on close-implied EV? Rows: CBB 2,213; K50 NBA/CFB; K1/K-CFB/K-NFL forward; L34 110.
baseline:   slope 1, intercept 0 (an efficient close). Current practice: gross CLV >= 0 as check (e).
change:     none to any strategy. This measures the instrument.
metric:     OLS slope and intercept of per-bet P&L on (close_prob - cost), date-cluster bootstrap. Also the calibration
            slope of outcome on Kalshi close prob.
success:    slope 95% CI contains 1 and excludes 0, AND intercept 95% CI within [-1%, +1%] per $ risked.
kill:       slope CI excludes 1 or intercept CI below -1% -> net CLV is not a valid primary statistic on Kalshi; keep ROI.
power:      SE of the slope ~ SD(P&L)/(SD(EV) sqrt n). At n ~3,000 and SD(EV) ~5% the SE is ~.35, enough to exclude 0 and
            borderline for testing 1. Underpowered for a tight slope: pool all leagues, or add the larger Kalshi
            sports close/outcome history (G1/G4 kalshi_sports dataset) to get SD(EV) up.
confounds:  selection (only bets we would make) -> also run on all listed markets. The close is measured at a stale
            last trade -> use the mid at close, with a spread filter.
cost:       ~2-3 h agent time, $0, read-only, no new trial (it validates the instrument, not an edge).
cheapest version: K50 rows alone. CLV predicted -2.1% vs observed -2.2% (LEARNINGS) is one point. Compute the slope on
            K50 + CBB in 10 min. If it is negative or far from 1, stop.

### C. Pre-registered forward SPRT on E4 orig-7 live fills and X1 forward paper (new nights only)
question:   Does the loss-rate SPRT (H0 L = 1 - c_i, H1 L = 1 - c_i * 1.02, alpha .05, beta .2, night-clustered) cross
            H1 or H0? Reported as months to a decision.
baseline:   sealed E1 +2.87% [90% CI +1.98, +3.71]; X1 sealed +1.67% [+1.01, +2.29] (results/e4_evidence/); current gate =
            GOAL rule 6 (100 forward bets at the family bar, structurally unpassable at c >= .965).
change:     the forward decision statistic (SPRT on filled losses) instead of rule 5 at the family bar.
metric:     the SPRT log-likelihood path. Filled orders only. Unfilled orders are logged for the fill-model check.
success:    crosses the H1 boundary -> evidence for the next size step (still the user's gate).
kill:       crosses the H0 boundary -> stop E4/X1 sizing up and propose a revert. Losses > 3 in one night -> audit.
power:      84% at a true +2% with mean 299 fills (iid simulation). Night clustering inflates n, so use the design effect
            from sealed E1 (55 losses / 406 dates). Expect 5-8 months for E4 at ~55 fills/month. X1 is slower unless
            the user restarts it.
confounds:  live fill rate .62-.68 vs sealed .857 (e4_evidence). Adverse fills raise L, and the test sees that, which is
            correct. Seasonality (winter cities): stratify by month in the report.
cost:       $0 beyond the live E4 already trading; a reporting script plus an edgelab registration only if the user
            adopts the rule.
cheapest version: compute today's SPRT position from live_maker_orders.csv (29 fills, 1 loss) in 10 min. It will sit
            between the boundaries, which confirms the plumbing.

## Learnings
### Cluster-bootstrap ROI p is anti-conservative on near-certain contracts
With 0 losses the bootstrap returns p = 1/(B+1) for any n. On fair 96c placebos at n94 edgelab's bootstrap false-passes
2.2% at a .00455 bar (5x). The exact test gives .96^94 = .0216 for P1m vs edgelab's < 1e-4 (inline sim, seed 3; notes/bar_review.md).
### Market-null and exact binomial on the loss rate are the same test
For a NO bought at c, ROI > 0 iff loss rate < 1 - c. A flat Beta posterior gives 1 minus the same number (.979 vs .0216 for
P1m 0/94). Power comes from alpha and n, not from switching tests (notes/bar_review.md).
### A 100-bet forward gate at a family bar is unpassable for 96-97c strategies
Zero losses in 100 gives p .028 at c .965 and .048 at c .97. Exact n for a true +2% at bar 8.5e-4 is ~1,130 fills, ~20
months for E4. Sequential SPRT at alpha .05 needs a mean of 299 fills (notes/bar_review.md).
### Profit-share concentration caps fail real small edges
A true +2% ROI on 50c bets fails the 30% share cap 93% of the time at n300 over 13 weeks. X1-like 95c at n368 over 7 weeks
fails 33%. Use leave-top-week-out ROI instead (inline sim, seed 7; notes/bar_review.md).
### CLV must be net of cost and on the traded instrument
Gross CLV > 0 passed K50 (+0.88 pts), which lost -2.2%. CLV-implied -2.1% matched. Net CLV needs ~270 bets vs ~19,300 on ROI
for the same +1.7% edge, but only after validating that the Kalshi close is calibrated (notes/bar_review.md exp B).
### Dev multiplicity should be paid once
edgelab charges alpha / all-time family trials at dev, sealed AND forward (run.py line 107). A sealed shot on independent
data only needs alpha / (number of sealed shots in the family). E1 is a golden sealed pass yet dead under its own dev bar.
