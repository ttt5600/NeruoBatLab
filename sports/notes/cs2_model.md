# cs2_model -- public-results rating model vs Kalshi KXCS2GAME (agent started 2026-11-18)

## Question
Does a rating model (Elo/Glicko-style) fit only on public CS2 match results beat Kalshi KXCS2GAME prices after the
Kalshi taker fee, at a fixed decision time before match start? Dev only; sealed opened only on the lead's/user's yes.

Context: esports_pinnacle.md (ESP1, forward-only vs Pinnacle), forum_strategy_scout.md (KXCS2GAME med vol 5,622 ct).
Prior: big-league Kalshi sports are efficient after cost (opener models ~0.85 pt CLV vs 1.2-1.4 needed).

## Plan
0. Prior art. 1. Free legal results source. 2. Kalshi KXCS2GAME settled markets + candles. 3. Prereg. 4. Dev + placebo.

## Log
### 2026-11-18 step 0 -- prior art (web search, thin)
- No peer-reviewed Elo/Glicko-vs-bookmaker log-loss comparison for CS2 found in one search. ENSI.Score publishes a CS2
  Elo (elo-v1) tracker: 1,717 matches, accuracy 64.42%, Brier 0.2188, log loss 0.6270 (coin flip 0.6931)
  (https://ensigame.com/ranking/accuracy). Vendor "Project Leggy" self-reports 66.3% on 920 matches, no log loss
  (https://leggy-site.pages.dev/about/). UCT dissertation (RF/XGB/NN on esports; https://open.uct.ac.za/...) -- no
  bookmaker benchmark seen. Takeaway: public Elo ~0.63 log loss; bookmaker esports lines are typically sharper (assume
  Kalshi close << 0.63 until measured). Prior = no edge.
### 2026-11-18 step 1 -- results source
- No clean free 2025-26 CS2 results CSV exists (CounterQuant HF datasets empty/staging; Kaggle HLTV set ends 2020;
  HLTV forbids scraping/behind Cloudflare). Chosen source: Kalshi's OWN settled outcomes (public API, legal):
  KXCS2GAME 17,663 markets / 8,831 events (2026-01-08..2026-11-18; 1,314 scalar) and KXCS2MAP 28,258 markets /
  14,129 events (2025-11-05..), results/cs2_model/markets_{KXCS2GAME,KXCS2MAP}.parquet (fetch_markets.py).
  Team identity = custom_strike.esports_competitor UUID (stable across name variants). Limitation: rating universe =
  matches Kalshi listed (no off-Kalshi results); this is the same information any trader has, i.e. a fair test.
- Start time: ticker HHMM (ET) when present (all events from ~2026-03-31); expected_expiration_time = start + 4.0 h
  exactly for timed tickers (median and min 4.0 h, n 15,453); older date-only tickers -> start = exp - 4 h (warm-up only).

## Pre-registration (written 2026-11-18 BEFORE any Kalshi price/candle has been fetched or joined to an outcome)
Trial CS2E (edgelab id cs2e_elo_vs_kalshi_t60, family esports_model, dev-scored by results/cs2_model/dev.py).
- Universe: KXCS2GAME events with a parsed ticker start time; both team markets present.
- Split by scheduled start (UTC): WARM-UP (ratings + hyperparameter choice only, no bets) < 2026-04-15;
  DEV [2026-04-15, 2026-10-19 13:45Z) ; SEALED >= 2026-10-19 13:45Z (= 70th pct of event starts; touch once, lead/user yes).
- Ratings: Elo, online, updated ONLY with results whose settlement_ts < the decision time (no lookahead). Two candidate
  models, ONE picked by warm-up+first-dev-free log loss (warm-up period events only, NOT returns):
  (a) series Elo on GAME results; (b) map Elo on MAP results (+ GAME results for teams with no maps), series prob from
  map prob p via best-of-3 formula p^2(3-2p). K in {16, 24, 32, 48}, init 1500, scale 400, new-team K x1.5 for first
  10 games. Pick = min log loss on warm-up events where both teams have >= 5 rated results. Frozen thereafter.
- Decision time T = scheduled start - 60 min. Price = YES ask of each team market from the last hourly
  candle with end_period_ts <= T (period 60; candle close yes_ask). Require ask in [0.05, 0.95], bid > 0.
- Rule: for each event, edge_i = p_model(team i) - (ask_i + fee(ask_i)); fee = ceil(100*0.07*10*p(1-p))/1000 per
  contract (10-lot taker, cent-rounded, Kalshi quadratic). Bet the side with the larger edge if edge >= 0.05; both
  teams need >= 10 rated results before T. 1 contract, hold to settlement; scalar settlements pay settlement_value.
- Stats: ROI = sum pnl / sum (ask + fee). 95% CI by bootstrap over ET match dates (cluster = date; 10k resamples);
  also event-level. Family bar from edgelab.
- SUCCESS (dev): n >= 200 AND ROI CI lower > 0 AND both chronological halves of dev ROI > 0 AND placebo (team ratings
  permuted across teams, 20 draws) mean ROI <= 0 AND p < family bar. -> register, STOP, ask lead before sealed.
- KILL (dev): ROI point <= 0, or CI upper < 0 -> dead. Between: "not distinguishable" -> dead unless the stacking
  diagnostic below shows positive Elo information with CI > 0 (then "lead" for a better model, no bet).
- Diagnostics (not trials): log loss / Brier of model vs Kalshi mid at T on dev events; logistic stacking
  y ~ logit(mid_T) + elo_logit_diff, Elo coefficient with date-clustered bootstrap CI; calibration of mid in deciles;
  thresholds 0.02/0.03/0.08 descriptive; by volume tercile.

## Log (after prereg)
### Model pick on warm-up (events start < 2026-04-15, both teams >= 5 rated results in both streams, n 664)
results/cs2_model/elo.py -> preds.parquet. Log loss: series K16 .6772, K24 .6743, **K32 .6732 (picked, frozen)**,
K48 .6748; map(BO3 transform) K16 .6819, K24 .6868, K32 .6942, K48 .7134. Coin flip = .6931. The warm-up Elo is weak
(Kalshi history starts Nov 2025 maps / Jan 2026 games) -- ratings mature over the dev window.
Streams: 8,173 binary series results; 14,823 map(+no-map series) results; 8,830 two-market events.
Placebo implementation: at each prediction, A and B ratings drawn at random from the current rating table (seeded).
### Edgelab
Registered cs2e_elo_vs_kalshi_t60-836708b5, NEW family esports_model (1 trial -> bar p < .05). Caveat for the lead: the
closest existing family is the sports opener-models-vs-Kalshi set; a stricter bar may be warranted. vault entry
kalshi_cs2_game added to edgelab/vault.yaml (dev [04-15, 10-19 13:45Z), sealed after, untouched).
Candle fetch limited (to save ~3 h of shared-throttle GETs) to dev events with both teams >= 10 rated results = exactly
the betting-eligible set (fetch_candles.py --dev-eligible; 6,470 markets).

## RESULT (dev, 2026-11-19) -- CS2E DEAD (pre-registered kill: ROI point <= 0 and CI upper < 0)
results/cs2_model/dev.py -> result_dev_preds.json, rows_dev_preds.parquet. Dev events with quotes at T: 3,370; eligible
(both >= 10 rated results) 3,218. Mechanics: all quote candle ends <= T (asserted); median quote age 0 h; YES-ask
sum at T median 1.02 (mean 1.063) -> ~2-6% two-side overround before fee.
| thr | n | ROI | 95% CI (date cluster) | halves | win / mean ask |
|---|---|---|---|---|---|
| .02 | 2258 | -13.3% | [-17.5, -9.1] | -13.3 / -13.4 | .330 / .365 |
| .03 | 2103 | -15.2% | [-19.8, -10.7] | -14.9 / -15.5 | .318 / .360 |
| **.05 (primary)** | **1815** | **-15.6%** | **[-20.6, -10.8]** (event CI [-20.9,-10.1]) | -15.2 / -16.0 | .305 / .346 |
| .08 | 1429 | -15.2% | [-21.5, -9.2] | -15.0 / -15.5 | .292 / .330 |
- Placebo (random-team ratings, 20 draws, thr .05): mean ROI -13.6%, range [-16.9, -10.8]%, mean n 2,241
  (result_dev_preds_placebo*.json). The real Elo is no better than random ratings.
- Log loss on 3,028 binary dev events: Elo .6663 vs Kalshi mid at T .5963; Brier .2369 vs .2054.
- Stacking y ~ logit(mid) + logit(Elo): coef mid 1.19, coef Elo -0.016, date-bootstrap CI [-0.201, 0.178] -> Elo has
  no information beyond the Kalshi price. Mechanism: Elo disagrees most on underdogs (mean ask .35); the market is right.
- Calibration of mid (deciles): .168->.145, .297->.253, .369->.334 ... .700->.790, .824->.862 (favourite-longshot shape;
  coef on logit(mid) 1.19 > 1 says the same).
### POST-HOC descriptive only (not a trial; band chosen after seeing calibration)
Buy the favourite at its T-60 ask (10-lot fee): mid [.50,.65) n 1709 ROI -6.5% [-9.9,-3.0]; [.65,.80) n 1034 +4.0%
[+0.8,+7.1]; [.80,.95) n 315 -0.8% [-4.9,+3.1]; [.65,.95) n 1349 +2.8% [+0.2,+5.3]. Forking-path selected and the
adjacent band is strongly negative -> at most a lead for a fresh prereg on data not yet seen (sealed window is the only
untouched history; lead/user decide whether to spend it on this instead).

## Learnings
### A public-results Elo has no information beyond the Kalshi CS2 price
Dev Apr-Oct 2026, 3,028 KXCS2GAME events: Elo log loss .6663 vs Kalshi mid .5963; stacking Elo coef -0.016 [-0.20,0.18]
(results/cs2_model/result_dev_preds.json). Esports on Kalshi is not "less modelled" in the way that matters.
### Betting Elo-vs-Kalshi disagreements in CS2 loses ~15% after fee, same as random ratings
CS2E primary n 1815 ROI -15.6% [-20.6,-10.8]; placebo random ratings mean -13.6%. Disagreement bets land on longshots.
### Kalshi's own settled markets are a free, legal results database for esports ratings
KXCS2GAME 8,831 events since 2026-01 and KXCS2MAP 14,129 since 2025-11, with stable team UUIDs in
custom_strike.esports_competitor; start = ticker HHMM ET = expected_expiration_time - 4 h (results/cs2_model/fetch_markets.py).
### Kalshi CS2 mid shows a favourite-longshot shape (post hoc)
mid .70 -> .79 won, .17 -> .145; buying favourites at mid .65-.80 was +4.0% [+0.8,+7.1] on dev but band was picked after
seeing calibration; neighbouring .50-.65 band -6.5%. Lead only.

## Pre-registration CS2F (written 2026-11-19 BEFORE any sealed candle is fetched; lead request, user standing
## authority for sealed tests = FUND_BOOK.md line 40, 10-29)
Trial CS2F (edgelab id cs2f_fav_65_80_t60, family **kalshi_sports_pregame** -- holds the Kalshi sports longshot/FLB
trials G1-G4/L1/L2; 20 trials before this one -> 21 after registration -> bar one-sided p < 0.05/21 = 0.00238).
Origin: post-hoc dev band (n 1034, +4.0% [+0.8,+7.1]) -- this is its one confirmation test.
- Universe: KXCS2GAME events with a parsed ticker start time, two team markets, settled in the 11-18 market pull,
  start in SEALED [2026-10-19 13:45Z, 2026-11-16 00:00Z) (end cut 2 days before the pull so postponed/scalar settles
  are not selectively missing). NO Elo/history filter (the rule does not use Elo).
- T = scheduled start - 60 min. Quote per team market = last hourly candle with end_period_ts <= T (asserted);
  mid = (yes_ask + yes_bid)/2. Favourite = team with strictly higher mid. Require favourite ask <= 0.95, bid > 0.
- Rule: if favourite mid in [0.65, 0.80), buy 1 ct of the favourite's YES at its ask; fee = ceil(100*0.07*10*p(1-p))/1000
  (10-lot taker, cent-rounded). One bet per event. Hold to settlement; scalar settles pay settlement_value.
- Primary metric: ROI = sum pnl / sum (ask + fee); 95% CI by bootstrap over ET dates (10k, seed 0);
  p = one-sided bootstrap share of resampled ROI <= 0.
- SUCCESS: CI lower > 0 AND p < 0.00238. KILL: anything else. Scored ONCE; nothing tuned after.
- Descriptive only (no decision weight): favourite mid bands [.50,.65) and [.80,.95); primary rule restricted to the
  Elo-eligible subset (both teams >= 10 rated series results, as in the dev rows).

### CS2F status at handback (2026-11-19)
Registered; sealed candle fetch (2,782 markets, fetch_candles.py --sealed, log candles_sealed.log) still running in the background. NOT scored. To finish: when the log says done, run `../.venv_quant/bin/python results/cs2_model/sealed_cs2f.py` ONCE from sports/ -> result_sealed_cs2f.json.
