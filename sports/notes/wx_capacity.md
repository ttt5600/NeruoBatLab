# wx_capacity -- more E4-like capacity (agent wx_capacity, started 2026-11-12)

## Questions (from lead)
1. X1 maker (`e4_maker_x1`): why dev-fail / needs_audit? Fixable without peeking at sealed?
2. Newly listed Kalshi weather series since the 10-28 / 11-06 census with enough history.
3. Per-city depth at the maker price for orig-7 + X1 -> realistic $/night at full deployment; ranked capacity table.

## Pre-registration (written 2026-11-12 BEFORE computing any depth/outcome number below)
Q1 audit: read-only on the edgelab dev rows already scored (lab/edgelab_runs/e4_maker_x1-e02827f2_dev.parquet).
  Classify each failed gate as (a) bug / leak, (b) structural (window length), (c) real weakness. No sealed rows are
  loaded; the only sealed number quoted is the one already in lab/lockbox.jsonl (10-23 pooled run, by_cohort X1).
Q2 census: re-run the existing frozen census (scripts/kalshi_wx_new.py) rules: a new series qualifies only with
  >= 86 settled event-days. Anything below -> "lead/watch", no trial.
Q3 capacity (descriptive, NOT a promotion, no edgelab trial): sample = forward paper picks in
  ~/.claude/quant-kalshi/data/kalshi_paper.jsonl, logged 2026-10-22..2026-11-10 (settled), both cohorts.
  Maker price q = M3 = min(floor_cent(1 - mid), NO ask - 1c) from the logged yes_bid/yes_ask. Public tape
  (read-only GET) in (logged_at, close]: V_lt = contracts printed at NO price < q (level cleared -> fill regardless of
  queue), V_le = at <= q (optimistic).
  Filled contracts at order size S: winner min(S, V_lt), loser S (losers always trade through; checked, not assumed).
  Expected P&L/night at S = mean over nights of sum_winners fill*(1-q) - sum_losers fill*q, using the forward outcomes
  (paper `result`). Because forward losers are few (n small), ALSO report a model version: replace realised losses by
  the sealed filled-loss rate (lockbox 10-23: pooled filled loss 3.07%) applied to S*q per pick -> E[$] =
  sum_picks [ (1-L) * min(S,V_lt)*(1-q) - L * S * q ] with L per cohort from lockbox. Report S in {6,12,24,48,96}.
  "Capacity" for a city = largest S at which modelled E[$]/night is still increasing by >= 50% of the linear step.
  Decision rule: X1 is worth a user ask only if its modelled E[$]/night at the depth-optimal S >= 50% of orig-7's.

## Log
### Q1 X1 maker audit (2026-11-12) -- STRUCTURAL fail, not a bug
Source: lab/edgelab_registry.jsonl evaluate row e4_maker_x1-e02827f2 split dev (11-02, family 44, bar 1.14e-3; family now
59, bar 8.5e-4) + rows lab/edgelab_runs/e4_maker_x1-e02827f2_dev.parquet.
- n 368 fills, ROI +2.29% [+0.70,+3.84], p 3.4e-3 (> bar), halves +1.45%/+3.14%, CLV +1.27c (99% coverage), 11 losses.
- Failed: (a) p >= bar; (c) week concentration 0.357 (2026-W09 = +3.02 of +8.44u; cluster-day share only 0.145);
  too-good guard Sharpe 8.04 (> 5).
- Why: dev window is 2026-01-14..02-28 = 46 event-days / 7 ISO weeks (X1 series listed ~Jan 2026, vault split). With
  7 weeks and one losing week (W06 -1.33u) any good week tops 30%; picks ramp 15 -> 113/week as cities list.
  Sharpe annualises 46 daily clusters at ~365/yr for a 95c-contract strategy (E1 maker dev 3.1 with sparser 2024 days).
  p: at ROI 2.3% and per-fill SD ~0.2, n=368 gives z ~2.2-2.7; the bar needs z ~3.1 -> roughly 2x the fills, which do
  not exist before the sealed window.
- Rows look clean: prices .67-.98 (median .96), fills after decided_at, 7 of 11 losses at prices .73-.87, no
  duplicate tickers. Per city (13 cities in dev): ATL -1.60u (4 losses), DC -0.94 (3) negative; rest positive. Descriptive only.
- Verdict: (b) structural. Not fixable on dev without new pre-sealed data (none exists). The X1 sealed maker result is
  ALREADY on record (lab/lockbox.jsonl 10-23 kalshi_temp_maker_M3_N10 by_cohort X1: 2,464 picks, fill .761, ROI +1.67%
  vs taker +0.78%, delta +12.03u p<.001). Re-scoring sealed under edgelab = user decision (rule: sealed once after dev pass).
- Honest paths: (i) USER: accept the 10-23 cohort sealed pass as X1 maker evidence (gate-design call); (ii) forward:
  register a new X1 maker forward trial on paper picks (kalshi_paper.jsonl keeps logging x1 picks, ~10/night) scored by
  public tape. Live X1 maker record before pause: 21 fills 21-0 +$1.15 (notes/e4_live_audit.md).

### Q2 census (2026-11-12) -- nothing new
- scripts/kalshi_wx_new.py census (frozen rules) -> results/wx_capacity/census_daily_2026-11-12.txt: 505 weather series,
  qualifying new daily series: NONE. All 21 international KXHIGHT* still 0 events.
- Series added since data/phys_census/series_all.json (11-06): KXSNOWCITIES (custom), KXRAINDCOACHELLA (daily rain, dead
  family), KXBIGGESTQUAKEMONTH, KXLOWTPERIOD (lows, dead family), KXRAINGTF. None E4-like with history.
- Open-market 24h volume (results/wx_capacity/open_wx_2026-11-12.json): daily highs dominate (NY 132k, CHI 119k, LAX 47k,
  MIA 43k, AUS 32k, ATL 26k ...). KXAVGT* weekly: nothing open. No daily snow/wind series exist.

### Q3 capacity (2026-11-12) -- results/wx_capacity/{capacity.py,capacity_table.py,picks_depth.csv,capacity_table*.csv}
Sample: 291 forward paper picks (orig7 77, x1 214) on 21 nights 10-22..11-10, settled; tape fetched read-only.
Losers: 6/291; all 6 traded through q by >= 24 ct (920-34,845 ct), so "losers always fill fully" holds at size.
73% of picks are NO ask .99 (q .98); only ~30% of those see ANY volume strictly through .98 (orig7 33%, x1 30%).
AMENDMENT 1 (after seeing the first table, disclosed): the pre-registered model applied the FILLED-loss rate 3.07% to
every pick, incl. 99c picks that rarely lose, so every row came out negative. That is a mis-specification, not a finding. Model 2 = per-pick
loss rate by NO-ask bucket from DEV picks (>.98 0.62%, .97-.98 1.25%, .96-.97 1.77%, .94-.96 1.21%, .90-.94 4.66%,
<.90 14.5%); winners fill min(S, vol strictly through q) (conservative: ignores at-q fills; live at-q filled 8/14).
Modelled $/night (model 2):
| cohort | S=6 | 12 | 24 | 48 | 96 | 192 | capacity (doubling adds >= 50%) | realised fwd S=24 |
|---|---|---|---|---|---|---|---|---|
| orig7 (3.7 picks/n) | 0.09 | 0.16 | 0.32 | 0.36 | 0.14 | -1.00 | 24 ct (= current) | +1.03/night (1 loss) |
| x1 (10.2 picks/n) | 0.51 | 0.98 | 1.78 | 3.26 | 5.70 | 6.44 | 96 ct | -0.82/night (5 losses) |
Capital tied up per night: orig7 S24 ~$37; x1 S48 ~$172, S96 ~$328 (bankroll ~$383).
Per city (S=24): X1 MIN 0.87, ATL 0.35, PHX 0.23, SFO 0.23, SEA 0.15 carry it; TTN/NOLA/BOS/SDF/SATX/SAN/LV/HOU/EWR <= 0
(mostly 99c picks with no through-volume). orig7: PHIL 0.41, LAX 0.02, rest ~0 or slightly negative. Per-city = descriptive
(21 nights, forking paths); no city rule proposed.
Pre-reg decision: x1 at its depth-optimal S (96: $5.70) >= 50% of orig7 at 24 ($0.32), so X1 is where the capacity is.
Caveat that changes the decision: realised x1 forward is NEGATIVE (5 losses/214, -$0.82/night at S24). n is far too small
to tell (not distinguishable), and the model assumes the dev bucket loss rates hold for X1 cities.
Dev cross-check on the 99c slice (lab/edgelab_runs/*_dev.parquet, q>=.98 fills, at-or-below-q model): e1 74 fills +1.51u,
x1 71 fills +1.45u, 0 losses, so NOT negative on dev; no ceiling rule proposed.
Noticed: lab/edgelab_runs/e4_maker_x1-e02827f2_sealed.parquet exists although status says dead-at-dev. NOT opened by me;
lead should check whether a sealed evaluate is already on the registry for e4_maker_x1.

## Proposals (lead/user)
P1 (user gate): restarting X1 maker is the only material capacity lever (~$2-6/night modelled at 24-96 ct vs orig7 ~$0.3-1).
   Evidence: lockbox 10-23 X1 cohort maker ROI +1.67%, delta vs taker +12.0u p<.001 (sealed, already opened); edgelab
   dev fail is structural (6.5-week dev). Options: (a) user accepts the cohort sealed pass, so X1 restarts at minimum units
   (6 ct) under SIZING_RULE as its own strategy with a pre-set kill; (b) forward paper trial on kalshi_paper x1 picks
   scored on the public tape (new edgelab trial + adapter), decide at >= 300 fills.
P2: orig7 is depth-capped at ~24 ct (current). No step 3 (matches SIZING_RULE note).
P3: no new series to add (census NONE); re-run the census when the international KXHIGHT* series list events.
No new dev trial registered: no qualifying series, and X1 has no unused pre-sealed data.

## Learnings
### X1 maker dev-fail is structural, not a bug
edgelab e4_maker_x1 dev n368 +2.29% [+0.70,+3.84] fails p (3.4e-3 vs bar) and 7-week concentration (0.357) because the
dev window is only 46 days; Sharpe 8 is the shape of 95c contracts. Source lab/edgelab_registry.jsonl. Why it matters: no dev
re-run can fix it; only a user gate on the existing sealed cohort or forward data can.
### Most E4 picks are 99c picks that a maker barely fills
73% of 291 forward picks had NO ask .99 (q .98); only ~30% saw volume strictly through .98. The money is in the ~3.7
picks/night below .985. Source results/wx_capacity/picks_depth.csv.
### Maker capacity lives in X1, not orig-7
Model-2 $/night: orig7 peaks ~$0.36 at 48 ct (capacity ~24 ct); x1 $5.70 at 96 ct. Realised fwd x1 -$0.82/night at
24 ct (5 losses, n tiny). Source results/wx_capacity/capacity_table_m2.csv.
### Flat filled-loss rates mis-price per-pick risk
Applying a filled-loss rate (3.07%) to every pick turned every city negative; per-pick loss is 0.6% at .99 vs 14.5% <.90.
Use per-price-bucket all-pick loss rates when modelling maker size.
### No new Kalshi weather series as of 11-12
505 weather series; 0 qualifying new daily series; international KXHIGHT* still 0 events. Source results/wx_capacity/census_daily_2026-11-12.txt.
