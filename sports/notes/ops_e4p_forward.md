# ops_e4p_forward -- forward paper logger: E4 probabilistic rule C (+ C-agree) maker NO bids (2026-11-16)

Ops agent for the quant-trader lead (agent_topic ops_e4p_forward). READ-ONLY: public Kalshi GETs through
scripts/kx_throttle.py, IEM MOS archive + Open-Meteo forecast GETs, read-only reads of
~/.claude/quant-kalshi/data/kalshi_paper.jsonl (live E4 picks). No orders, no keys, no git, no .env / kill files,
live executor untouched. User approved read-only paper loggers under launchd.

## Lead (notes/e4_probabilistic.md; critic critic_e4p: no leak/bug, WEAKENED -> maker-only forward paper)
Rule C: at 22:00 station-local D-1, NO on KXHIGH brackets the 5-model E4 rule does NOT pick when the Student-t EMOS P(hit)
(NBM NBS txn + xnd + mean/var of the 5 deterministic models) <= 0.02. Dev TEST: C NEW maker 331 fills, 5 losses (all
filled; unfilled 0/213), +11.58u [+6.0,+16.5] p < .00025 (study bar); edgelab family bar FAILED (E1-only p .0014 vs 7.35e-4)
-> forward paper only. Critic: taker must be scored at live order size with per-order cent fee rounding; 3/5 dev losses
had market YES bid >= 5c -> C-agree variant.

## PRE-REGISTRATION (written 2026-11-16 ~10:15Z, before ANY forward decision; first decision 22:00 ET = 03:00Z 11-17)
### Deviation found before any data (disclosed): the NBS 19Z cycle no longer exists
IEM MOS archive (model=NBS, KNYC; results/ops_e4p_forward/probe_cycles.py): runtimes 01/07/13/19Z through 2026-04, then
00/06/12/18Z from 2026-06 on (NBM cycle change between 2026-04-28 and 2026-06-01). Dev used the 19Z D-1 run (lead 29 h to
the D+1 00Z txn). Forward uses the **19Z D-1 run if present, else the 18Z D-1 run** (lead 30 h, same txn/xnd fields, same
D+1 00Z ftime). The frozen fit was trained on the 19Z (pre-change) NBM; any NBM version shift in txn bias or xnd scale is
NOT corrected (no post-2026-06 outcome is used; those months are inside the sealed window 2025-07-01..2026-10-06).
Monitored descriptively: PIT tails and bracket reliability of forward P(hit) vs dev (status command).

### Population and decision (frozen)
- Cities: all 24 E4 paper series (orig-7 wx.CITIES + 17 X1, kalshi_paper.PAPER_CITIES), daily-high events for day D.
- Decision tick: first launchd tick (every 300 s) with station-local time in [22:00, 22:50) on D-1. If NBS is not yet in the
  IEM archive the tick waits (no row) until 22:40; at >= 22:40 it decides with the no-NBS fit (dev's missing-run model).
  Later than 22:50 -> `missed` city-day row (no picks; never back-filled).
- Models: the same 5 Open-Meteo models and the same live call as kalshi_paper.live_forecasts (max over hours <= 18 local on
  D, >= 15 hourly values, >= 3 models required). m = mean, s2 = var (ddof 0).
- P(hit) = dev formula (scripts/e4_probabilistic.py p_hit, integer settlement boundaries), Student-t EMOS.
- Calibration (FROZEN procedure = dev walk-forward): for forward month M, refit (max likelihood, L-BFGS, dev start values)
  ONCE on all dev station-days (results/e4_probabilistic/phit.parquet, 2,949 station-days: E1 2024-03..2025-06 + X1
  2026-01-14..02-28) PLUS this logger's own settled forward station-days dated before M's first day; never refit within
  M; the fit is written once to data/kx_forward/e4p_fits.json and never rewritten. No sealed-window row is ever used.
  (November 2026 fit = dev-only.)
- Tradeable (E4 filter, live /markets quote): yes_bid > 0, yes_ask < 1, spread <= 10c, yes_ask > 1c.
- E4 exclusion (NEW only): a bracket is excluded if (a) the E4 5-model rule (margin 3F) selects it on OUR forecasts, or
  (b) its ticker appears in the live E4 log ~/.claude/quant-kalshi/data/kalshi_paper.jsonl, checked at decision AND again
  at settle (E4 decides in the same 22-23h window; a later E4 pick excludes it retroactively; flagged e4_live_late).
- **Variant C**: NEW & tradeable & P(hit) <= 0.02.   **Variant C-agree**: C & market yes_bid < 0.05 at decision.
- Hypothetical maker NO bid (= kalshi_exec._decide_maker, E4 M3): live public orderbook at T; NO ask = 1 - best YES bid,
  NO bid = best NO bid; q = min(cent_floor((ask + bid)/2), ask - 0.01). Skip (logged) if no book, q <= 0, ask - q > 5c,
  market not active, series fee_type != quadratic. Rests (T, close]. Maker fee 0. 1 unit cost per fill.
- Fill (PRIMARY): TRADE-THROUGH = any public print with created_time in (T, close] and no_price < q. N10 diagnostic.
- Outcome: the market's own `result`; NO wins -> pnl (1 - q)/q, else -1. Station outcome = event expiration_value.
- Descriptive (never a trial, never a gate): taker at 24 contracts = walk the YES-bid book at T (NO price 1 - p per level),
  fee = ceil_cent(0.07 x sum_levels c x P x (1 - P)) per ORDER; cost/contract = (sum c x P + fee) / filled; contracts
  available (< 24 -> partial). Also filled-vs-unfilled loss rate (adverse selection).

### Forward judgement (frozen; per notes/bar_calibration_v2.md near-certain rule; applied to C and C-agree separately)
Unit = one trade-through-filled maker bid; c_i = q_i; break-even loss p0_i = 1 - c_i.
1. Continuous Wald SPRT: H0 loss p0_i, H1 0.70 x p0_i; design alpha .005, beta .10 -> A = ln(.9/.005) = +5.193,
   B = ln(.1/.995) = -2.298. Per fill: loss ln(.70); win ln((1 - .7 p0)/(1 - p0)). After each night, cumulative LLR divided by
   1 + (mean fills per night) x .031. LLR <= B -> KILL that variant. LLR >= A -> H1, BUT the trial failed the edgelab family
   bar on dev, so per bar_calibration_v2 s4 it cannot self-promote: it becomes an ask to the user, nothing more.
2. Review at 60 fills (per variant): exact Poisson-binomial test of the loss count L vs H0 (L_i ~ Bernoulli(p0_i)),
   one-sided p = P(L_H0 <= L_obs). Two variants -> Bonferroni: "holds" needs p < .025 and ROI > 0 (lead may then ask the user
   for a sized live A/B). L_obs >= expected L under H0 (ROI <= 0) -> KILL. Otherwise continue to 120 fills, same test; at 120
   not "holds" -> KILL. edgelab spec kill (mechanical backstop): roi_ci_below_zero after 60.
Volume expectation (stated before data, from dev 1.15 C maker fills/night over 24 cities in winter/spring): ~60 C fills in
  ~2 months; C-agree fewer. Nov-Dec is the seasonal trough.

## Build
- scripts/e4p_paper.py tick | settle | status | dry  (log: ~/.claude/quant-sports/data/kx_forward/e4p.jsonl;
  fits ~/.claude/quant-sports/data/kx_forward/e4p_fits.json; dev training ~/.claude/quant-sports/data/kx_forward/e4p_train_dev.parquet)
- edgelab/adapters/e4p_fwd.py (params variant C | C_agree); vault.yaml dataset kalshi_temp_e4p_fwd; specs
  edgelab/specs/e4p_c_maker_fwd.yaml, e4p_cagree_maker_fwd.yaml
- scripts/qs_job.sh case `e4p`; scripts/install_e4p_agent.sh; scorecards cards E4P-C, E4P-CA.

## Log
### 2026-11-16 ~10:10-10:25Z build + register + deploy (before any forward decision)
- NBS cycle probe (results/ops_e4p_forward/probe_cycles.py, KNYC): 01/07/13/19Z through 2026-04; 00/06/12/18Z from 2026-06
  -> 18Z D-1 fallback pre-registered above (written before the dry pass).
- edgelab (before any forward data): registered **e4p_c_maker_fwd-8d53a5e5** and **e4p_cagree_maker_fwd-9e8b607c**
  (family kalshi_weather_temp -> 70 trials, bar 7.14e-4); specs edgelab/specs/e4p_{c,cagree}_maker_fwd.yaml (rule
  forward:unwired, kill roi_ci_below_zero after 60); vault.yaml dataset kalshi_temp_e4p_fwd (forward 2026-11-17+, adapter
  edgelab/adapters/e4p_fwd.py with params variant C / C_agree).
- Port checks: `scripts/e4p_paper.py fitcheck` refits the dev 2026-01 month (E1 dev, 2,554 station-days) -> nbs and nonbs
  params identical to results/e4_probabilistic/fits.json to 4 dp (max |diff| 0.0). p_hit with fits.json params vs dev
  phit.parquet (1,778 rows, 2025-03 + 2026-01): max |diff| .0016 (fits.json is rounded to 4 dp; forward uses full precision).
  Dev training base exported: results/ops_e4p_forward/e4p_train_dev.parquet (2,949 station-days 2024-03-01..2026-02-28,
  no sealed rows) -> deployed data/kx_forward/e4p_train_dev.parquet.
- Dry pass (scratch only: results/ops_e4p_forward/dry_1116.jsonl, scratch fits; ~10:30Z on event day 11-16 = IN-DAY, not the
  22:00 decision, code-path test only): 24/24 cities decided with NBS run 2026-11-15 18:00Z; 4 C picks (LAX T68 phit .0092,
  BOS T47 .0060, OKC T51 .0187, SFO T58 .0163), all C-agree, all status bid q .97-.98; taker@24 cost/ct .98-.99
  (e.g. ask .99 -> fee ceil(24 x .99 x .01 x .07 = .0166) = 2c/order = .00083/ct). E4 exclusion working: 10 brackets
  flagged e4_own and/or e4_live_at_T, none selected. Deployed copy (~/.claude/quant-sports venv) reproduces LAX/BOS rows exactly.
- Open-Meteo was unreachable for ~5 min at 10:15Z (curl exit 28/35), then fine. The tick retries a failed city on every
  launchd tick inside [22:00, 22:50); a long outage -> `missed` city-days (counted, never back-filled).
- Adapter check (results/ops_e4p_forward/adapter_check.py, synthetic): C 2 bets / C-agree 1 bet as designed (unfilled and
  e4_live_final rows excluded); schema.enforce PIT 0 violations. status/judge on the synthetic log runs.
- scorecards.py cards E4P-C / E4P-CA (rules quoted from this file; too-early).
- Deploy: scripts/install_e4p_agent.sh -> ~/.claude/quant-sports/scripts/{qs_job.sh,kx_throttle.py,e4p_paper.py}; launchd
  com.jonathanwang.quant-sports-e4p every 300 s (qs_job.sh case e4p; deployed qs_job.sh was byte-identical to the repo copy
  apart from the new case). Kickstart 10:20Z: idle tick rc 0, settle ran (0), heartbeat written, launchd err empty.
  **First real decision: 22:00 ET 11-16 = 03:00Z 11-17 (event day 11-17); not yet observed.** The November fit is computed
  on that first decision tick (dev-only data) and frozen in data/kx_forward/e4p_fits.json.
- Monitor (descriptive, one day, NOT a trend): 11-15 18Z xnd on 24 stations {1:4, 2:17, 3:1, 4:2} vs dev Nov-Feb 19Z
  {1:.35, 2:.44, 3:.15, 4:.05}; txn - m +2.15F (in-day m) vs dev +1.17F. Re-check after ~30 forward days with PIT tails.

## Learnings
### The NBM NBS text bulletin moved from 01/07/13/19Z to 00/06/12/18Z cycles in May 2026
IEM MOS archive (model=NBS, KNYC): 19Z runs through 2026-04-28, 18Z from 2026-06-01 (results/ops_e4p_forward/probe_cycles.py).
Any rule trained on the 19Z run must disclose a 1-h-earlier run and a possible NBM version change on forward data.
### A forward fit procedure must be ported and checked to the decimal before it goes live
scripts/e4p_paper.py fitcheck reproduces the dev EMOS params exactly (max |diff| 0.0 at 4 dp) and P(hit) to .0016.
A silent port difference would change which brackets pass a 2% threshold.
### Dev-only training keeps forward fits clear of the sealed window
Forward monthly refits use dev station-days + this logger's own settled forward days only; the kalshi_weather_temp
sealed window (2025-07-01..2026-10-06) is never loaded, so the NBM cycle/version change cannot be recalibrated yet.
