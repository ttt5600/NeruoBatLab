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
