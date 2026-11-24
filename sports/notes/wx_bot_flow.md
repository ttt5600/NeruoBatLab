# wx_bot_flow -- I4: can we see open-source KXHIGH bot flow in the Kalshi tape? (agent wx_bot_flow, started 2026-11-24 03:00Z)

Question (winners_transfer.md I4): public retail KXHIGH bots (GFS 31-member ensemble, "trade when |edge| > 8%") are a known
flow. Replicate their signal as-of, then test (a) does taker flow follow the signal right after a new GEFS run reaches
Open-Meteo, (b) are brackets the bots want to buy overpriced at that moment, (c) is there an E4 rule (veto / new pick class).

## 1. The public bots (read from source, 2026-11-24)
| repo (stars, created) | forecast input | probability | trade rule | schedule | real orders? |
|---|---|---|---|---|---|
| github.com/suislanchez/polymarket-kalshi-weather-bot (844, 2026-02-10; Kalshi added 03-01) | Open-Meteo Ensemble API `models=gfs_seamless` (31 GEFS members), `daily=temperature_2m_max`, NO timezone param -> GMT day; city-centre lat/lon (NYC 40.7128,-74.0060 etc.) | fraction of members > threshold, clip [.05,.95] | edge = p - yes_ask (or NO side) >= **0.08**; entry <= .70; skip yes_ask < .02 or > .98; Kelly .15, max $100 | every 5 min (APScheduler) | **NO**: trades are rows in a SQLite DB (platform "polymarket"), no Kalshi order call. Ticker parser BUG: `B45.5` (a 2F between-bracket) read as "high > 45.5"; every `T` read as "below" |
| github.com/AwpDemon/kalshi-weather-bot (2, 2026-07-18; rebuild of suislanchez) | same Open-Meteo call (gfs_seamless, GMT day, city centre) + 90-day rolling bias correction vs Open-Meteo archive | members in [floor-.5, cap+.5), clip [.05,.95] | edge >= **0.10**, entry price in [.10,.90], confidence >= .55, only D+1 or later, limit order AT the ask (taker), $2/trade, $24 bankroll | every 2 h, 13:00-03:00Z | paper by default (BOT_MODE=paper); live path exists. Phase-1 paper: 24 trades |
| github.com/Alextryfg/kalshi-weather-bot (2, 2026-05-19) | Open-Meteo ensemble gfs_seamless hourly, timezone UTC, airport coords, bias correction, METAR intraday | ensemble -> normal | edge >= 5 pp, price 10-80c, >= 6 h to settle, quarter Kelly | GitHub Actions cron `*/30 * * * *` (3,809 runs) | default `simulation` and `KALSHI_API_BASE` = demo API |
| github.com/ethanterrero/Kalshi-Weather-Bot (2, Rust) | NWS point forecast + GEFS sigma (Open-Meteo 30 members), LST window | Normal CDF | min_edge .05, price [.20,.92] | GEFS poll 30 min | paper; `never_send` guard needs a code edit |
| github.com/bnbi/kalshi-weather-trading (1) | GFS/ECMWF/ICON blend + ML | Normal | edge filters | scheduler | **live, 89 trades: "next-day forecasting edge ~zero against same-information market makers; died to adverse selection"** |
| github.com/jdeepsingh28/kalshi-weather-bot (0) | METAR only (GEFS/HRRR pipeline never wired) | -- | **maker selling temperature tails** (E4's own trade) | event-driven | **live, 5,143 fills: "a genuine premium edge, not quite enough to clear its maker fees"** |
Reading: the famous "31-member GFS, edge > 8%" bot is paper-only; its descendants default to paper/demo with $2 sizes. The
public code that actually traded live is (i) a directional forecaster that found no edge and (ii) a tail-selling maker
(our competitor for the same longshot premium). Expected live notional from the GEFS-bot family is tiny vs the orig-7 pool.

## 2. As-of data (no lookahead)
- Open-Meteo Ensemble API keeps only ~3 days of past data and has no run parameter (docs, 2026-11-24): NOT usable historically.
  Its AWS bucket (s3://openmeteo data_spatial/ncep_gefs025) holds only precipitation_probability, recent runs only.
- Used instead: NOAA GEFS 0.25 deg on AWS (noaa-gefs-pds, pgrb2sp25, as-issued per run), TMP:2 m, 31 members (gec00,
  gep01-30), byte-range reads; scripts/wx_bot_flow_gefs.py -> results/wx_bot_flow/gefs_points.parquet.
- Run arrival on Open-Meteo (S3 LastModified, 2026-11-16..22): 12Z run next-day fields written 15:56-16:25Z, run complete
  17:32-17:43Z; 06Z run complete 11:32-11:34Z. Kalshi KXHIGH events for D open 15:00Z on D-1 (KXHIGHNY-26NOV10 open_time).
  So at market open the bots see the 06Z D-1 run; the 12Z D-1 run arrives 16:00-17:40Z D-1 (11:00-12:40 ET).
- Replication approximations (disclosed): nearest 0.25 deg grid point (Open-Meteo also adjusts T for elevation and may pick a
  neighbour cell); daily max over the 9 three-hourly samples 00..24Z (Open-Meteo interpolates 3-hourly to hourly).

## 3. PRE-REGISTRATION (written 2026-11-24 ~03:30Z, before any tape or outcome in the test window is pulled or looked at)
Window: orig-7 KXHIGH (NY, CHI, MIA, AUS, LAX, DEN, PHIL), target dates D 2026-10-08..2026-11-22 (settled; signal day D-1 >=
10-07, outside every sealed window: orig-7 sealed ends 2026-10-06; edgelab vault.yaml). 10-22+ is the forward split (used
here for research, disclosed). X1 cities excluded (sealed to 10-18). Pre-bot dev years are NOT used (bots appeared 2026).
Signal p_bot(b, run): 31 members, bot city-centre point, member max over 00..24Z of D, AwpDemon bracket counting
([floor-.5, cap+.5); greater: t >= floor+.5; less: t < cap+.5 -- using Kalshi strike_type), clip [.05,.95].
p_stn: same but settlement-station point and the LST day (samples inside [00 LST D, 24 LST D]).
Prices: hourly candles yes_bid/yes_ask close; ask16 = yes_ask at 16:00Z D-1, mid19 = (bid+ask)/2 at 19:00Z D-1.
edge12 = p_bot(12Z D-1) - ask16; edge06 = p_bot(06Z D-1) - ask16.
Sample: brackets with ask16 in [.03,.90] and >= 1 trade in the outcome window.
H1 (primary, flow): OLS NYF_after ~ 1 + edge12 + edge06 + ask16, NYF = (YES-taker - NO-taker contracts)/(all contracts) in
  [16:00Z, 19:00Z) D-1. beta1 = coefficient on edge12 (flow follows the NEW run beyond the old one). 95% CI by bootstrap over
  signal dates (2,000 resamples).
  P1 timing placebo: same regression with NYF_before in [15:00Z, 16:00Z) (the 12Z run is not on Open-Meteo yet).
  P2 wrong-date placebo: edge12 built from the same city's 12Z run for target D-7 (same brackets, same ask16).
  SUCCESS (model-update flow identified): beta1 > 0 with CI excl. 0 AND beta1 - beta1_P1 > 0 with CI excl. 0 AND P2 CI incl. 0.
  H1b (bot-specific): add edge12_stn = p_stn(12Z) - ask16 (and edge06_stn); bot flow identified only if edge12 (the bot's
  mis-specified GMT-day/city-centre signal) keeps beta > 0 with CI excl. 0 given the correct signal.
  KILL: beta1 CI incl. 0, or fails P1/P2 -> "bot flow not identifiable in the tape"; I4 dead as a flow model.
H2 (secondary, pricing, E4-relevant): resid = won - mid19 over brackets with mid19 in [.02,.90]; OLS resid ~ edge12_19
  (= p_bot(12Z) - ask19). gamma < 0 (CI excl. 0) -> bot-favoured brackets are overpriced (fade). Cheap subset (mid19 <= .15,
  E4 territory): mean(won - mid19) for bot-BUY (edge12_19 >= .08) minus non-bot-BUY, date-cluster CI.
  RULE gate: an E4 proposal only if gamma < 0 (CI excl. 0) AND the cheap-subset difference <= -.02 with CI excl. 0; then
  forward paper only (no live change), registered in edgelab family kalshi_weather_temp. Otherwise no rule.
Score once; no re-specification after seeing outcomes (any extra cut is labelled post hoc).

## Log
- 03:05Z started row; bots surveyed (section 1); as-of source decided (section 2); GEFS pull launched (34,968 fields).
- 03:40Z replication check (scripts/wx_bot_flow_omcheck.py -> results/wx_bot_flow/omcheck.json): our NOAA-GEFS extraction vs
  the live Open-Meteo Ensemble API (run 2026-11-23T18Z, target 11-25 GMT day), 31 members x 5 bot cities: member daily max
  differs by a near-constant per-city offset (NY +0.08, CHI +0.56, MIA -0.05, LAX -0.17, DEN -0.42 F; sd 0.06-0.12 F; max
  |diff| 0.66 F) = Open-Meteo's elevation/cell choice. Decision (before outcomes): subtract the measured 3-hourly offset per city
  (NY .18, CHI .60, MIA -.02, LAX .22, DEN -.40; AUS/PHIL 0) and use the 9-sample 00..24Z max (closer than 00..21Z).
  -> the bot signal is replicated to ~0.1 F member-wise; bracket probabilities can differ only by members within 0.1 F of an edge.
- 03:55Z the live tail-selling maker (jdeepsingh28 results/RESULTS.md): 2025-09-10..2026-03-18, 1,026 settlements in 18 cities,
  85.2% NO / 14.8% tail-hit; reconciled Feb 27-Mar 16 2026: 5,143 fills, premium capture +$394, tail settlements -$355, gross
  +$39, est. maker fees -$55..-67 -> net about -$16..-28. It sold tails much nearer the money than E4 (E4 loses ~2-3%).
  Lesson for E4: the same longshot premium is contested by at least one public maker; its tails were too close.
- scripts: wx_bot_flow_gefs.py (GEFS points), wx_bot_flow_tape.py (Kalshi markets/trades/candles), wx_bot_flow_score.py
  (pre-registered H1/P1/P2/H1b/H2; written before any tape was loaded).
- 03:25Z tape pull hit Kalshi 429s at 6 threads (a plain curl also got 429) -> restarted at 1 thread, 0.25 s spacing, and it
  pauses 56-14 min around the live E4 executor runs (03/04/05/06:00Z) so a research pull never competes with the live bot.
  Lesson: public-endpoint research pulls share this machine's IP with the live executor; throttle and avoid its run minutes.
- 03:30Z market metadata: 1,974 markets = 47 dates x 7 cities x 6 brackets (1,316 between / 329 greater / 329 less); open_time
  14:00Z D-1 before the 11-01 DST end and 15:00Z after (always 10:00 ET). The pre-registered UTC windows are unchanged (the
  12Z GEFS run arrives ~16:00Z either way); during DST the [15,16)Z "before" window is the 2nd trading hour, not the 1st.
- bnbi RESEARCH.md (live, 2026-05-06..06-11, CHI/NYC/MIA): 89 trades, -$22.86 on ~$120 (-19%); point MAE 1.4-1.5 F was good,
  but realised win rate fell below the claimed probability in every bucket; market-vs-model log-likelihood blend weight w ~ 0
  (market log loss .386 vs model .770). A directional forecast bot that trades on model-market gaps is the counterparty who
  pays makers like E4 -- but its live size here is ~$120 over a month.
- 04:00Z GEFS pull done: 34,968 fields, 0 failures (results/wx_bot_flow/gefs_points.parquet, 489,552 point rows; 31 members on
  all 658 city-date-run groups; 9 GMT samples per bot member, 8-9 LST samples per station member). Signal-only descriptives
  (no outcomes): GEFS member-max spread (sd) at D+1 is small, 0.8-2.2 F (LAX .78, MIA .97, CHI 1.17, NY 1.27, AUS 2.15) --
  the 31-member counts are under-dispersed, so the bots' p is often 0/31 or 31/31 (clipped .05/.95). Mean |12Z - 06Z| shift of
  the ensemble mean 0.41 F (sd .56). Bot signal (GMT day, city centre) minus correct signal (LST day, station) ensemble mean:
  NY +0.63 (sd 2.36), CHI -0.26 (1.82), MIA +0.20 (.73), AUS +.73 (1.61), LAX +2.02 (1.84), DEN +.81 (3.84), PHIL +.35 (2.26) F
  -> the bots' mis-specification is large enough to separate bot flow from informed flow (H1b) if bot flow exists.
- 04:17Z the tape pull died at ~1,400/1,974 markets on one Kalshi ReadTimeout (requests exception not caught; no checkpoint),
  losing 40 min of pulling. Fixed: RequestException retried with backoff, per-ticker checkpoint
  (results/wx_bot_flow/tape_checkpoint.jsonl), restarted 04:20Z at 2 threads. Lesson: any pull > 5 min checkpoints per unit.
- 04:30Z AMENDMENT before real scoring (found on a SCRAMBLED dry run: taker_side and result permuted, /tmp/wxdry): the H2
  regression resid(won - mid19) ~ edge12_19 (= p - ask19) has a mechanical positive slope through the shared price (scrambled
  outcomes gave beta .27 [.12,.39]). H2 now controls mid19: resid ~ edge12_19 + mid19 (scrambled: .04 [-.15,.21]). H1 already
  controls ask16 (scrambled H1 edge12 .07 [-.72,1.04]). No real outcome had been seen. Lesson: dry-run every pre-registered
  regression on permuted outcomes; a stat that shares a price term on both sides is mechanically non-zero.

## RESULT (scored once 2026-11-24 ~04:50Z; scripts/wx_bot_flow_score.py -> results/wx_bot_flow/score.json, frame.parquet)
Data: 1,974 markets / 329 events (orig-7, D 10-08..11-23), 101,704 trades in [14,20)Z D-1, 10,767 hourly candles (ask16
coverage 99.7%), 31 GEFS members on every city-date-run. CIs = 2,000 bootstrap resamples of signal dates (47).
| test | coefficient on the 12Z bot signal | n | pre-registered reading |
|---|---|---|---|
| H1 NYF 16-19Z ~ edge12 + edge06 + ask16 | **+0.459 [+0.156, +0.754]** (edge06 -0.554 [-0.854,-0.265]) | 1,382 | condition 1 met |
| P1 same, NYF 15-16Z (12Z run not yet on Open-Meteo) | +0.316 [-0.066, +0.761] | 1,297 | |
| H1 - P1 | **+0.144 [-0.390, +0.623]** | | **condition 2 FAILS** |
| P2 wrong-date (D-7 run) | +0.010 [-0.099, +0.128] | 1,159 | placebo null, ok |
| H1b add station/LST signal | bot +0.418 [+0.066, +0.791]; station +0.080 [-0.285, +0.453] | | not attributable (H1 failed) |
| by hour 14..19Z (NYF in that hour) | .20, .32, .33, .30, .51, .40 (each CI ~ +-0.4) | | flat: no jump at arrival |
| H2 won - mid19 ~ edge12_19 + mid19 | **-0.023 [-0.115, +0.075]** | 1,306 | not distinguishable |
| H2 cheap (mid19 <= .15): bot-BUY minus other | +0.008 [-0.028, +0.045] (bot-BUY hit .055 vs mid .061, n 236; other n 318) | | rule gate FAILS |
Signal counts at 16Z (1,405 brackets, ask .03-.90): bot BUY-YES (edge >= .08) 413, of them 256 cheap (ask <= .15); BUY-NO 520;
newly BUY-YES with the 12Z run 58. 48% of brackets sit at the p = .05 floor (0 of 31 members).
POST HOC (results/wx_bot_flow/posthoc.py -> posthoc.json; descriptive, not pre-registered):
- paired after-minus-before NYF on the same 1,279 brackets: edge12 +0.060 [-0.344, +0.455] -> no jump when the run lands.
- cities the public bots trade (NY/CHI/MIA/LAX/DEN) +0.454 [+0.136, +0.817] vs AUS/PHIL (in no public bot's city list
  except Alextryfg's PHIL) +0.479 [-0.173, +1.108]: same slope where the bots do not trade.
- prints <= 20 contracts (bot sizes): after +0.483 [+0.149, +0.842], before +0.280 [-0.086, +0.674] -- same as all prints.
- YES-taker share of contracts 15-19Z D-1: 65.0% overall; cheap brackets (ask16 <= .15) 67.4%; cheap bot-BUY 65.6% vs cheap
  other 68.4%; cheap hit rates 2.8% vs 2.5%.
Rows inspected (frame.parquet): the largest 12Z-06Z swings are real GEFS jumps (KXHIGHLAX-26NOV12-B79.5 p .774 -> .097 with
ask .05; B77.5 .226 -> .903), and the market had ignored the 06Z GEFS view (ask .05 vs p .77). Binned dp: NYF rises with dp
by about the same amount before (.12 -> .34) and after (.10 -> .36) the run's arrival.

## VERDICT: DEAD as a flow model (I4). Bot flow is NOT identifiable in the Kalshi tape.
- Net taker flow does lean toward where the new GEFS run moves (+0.46), but it leans the same way BEFORE the run reaches
  Open-Meteo, shows no jump at arrival (paired +0.06 [-0.34,+0.46]), and has the same slope in cities no public GEFS bot
  trades. That is the market's general information flow (other models, obs, persistence), not a discrete bot reaction.
- The source code says why: the 844-star "31-member GFS, edge > 8%" bot never sends Kalshi orders (paper DB rows); its
  descendants default to paper/demo with $2 tickets; the public bots that did trade live were a directional forecaster that
  lost 19% on $120 (market better than model, w ~ 0) and a tail-selling maker (E4's own trade) that netted about -$16..-28.
- Pricing: brackets the GEFS bots would buy are priced fairly given the price (H2 -0.023 [-0.115,+0.075]); cheap bot-BUY
  brackets are no more overpriced than other cheap brackets (+0.8c [-2.8c,+4.5c]). No E4 veto and no new pick class.
- Who pays E4 (descriptive): ~2/3 of taker contracts on cheap brackets on D-1 are YES-buys, spread across brackets whatever
  GEFS says -- diffuse retail YES demand, not a forecast-driven bot class we can predict.
- No edgelab trial registered: no rule passed the pre-registered gate (H2/H2-cheap failed); this study is one look at the
  family's post-sealed window 2026-10-08..11-22 (lead may add it to the kalshi_weather_temp family count).
- Not done (would be the only remaining angle): pre-bot-era replication of the H1 slope (2025 GEFS + tape) to test whether
  the flow-on-model-change slope grew after the bots appeared. Low prior given the above; not recommended.

## Learnings
### The famous open-source Kalshi weather bot does not trade live
github.com/suislanchez/polymarket-kalshi-weather-bot (844 stars) records weather "trades" as paper rows in SQLite with
platform "polymarket" and never calls the Kalshi order endpoint; its forks default to paper/demo with $2 tickets
(AwpDemon, Alextryfg, ethanterrero). Read the order path before treating a GitHub strategy as market flow.
### Net taker flow follows GEFS run changes before the run is public, so it is not bot flow
Coefficient of 16-19Z net YES flow on the 12Z-run bot edge +0.459 [+0.156,+0.754], but 15-16Z (before the run is on
Open-Meteo) +0.316 [-0.066,+0.761], difference +0.144 [-0.390,+0.623]; paired jump +0.06 [-0.34,+0.46]; same slope in
AUS/PHIL where public bots do not trade (results/wx_bot_flow/score.json, posthoc.json). An event study needs a pre-arrival
placebo: forecast changes are predictable from other sources, so flow "following the model" proves nothing about who trades.
### Brackets that GEFS-ensemble bots would buy are priced fairly
won - mid19 on the bot edge, controlling mid: -0.023 [-0.115,+0.075] (n 1,306); cheap bot-BUY minus other cheap +0.8c
[-2.8c,+4.5c] (results/wx_bot_flow/score.json). No E4 veto or new pick class from the bot signal.
### Open-Meteo keeps no as-issued ensemble archive; NOAA GEFS on AWS replicates it to 0.1 F
Ensemble API keeps ~3 days and has no run parameter; NOAA noaa-gefs-pds pgrb2sp25 TMP:2 m byte-range reads (0.76 MB per
member-field) reproduce Open-Meteo gfs_seamless member daily maxima within a constant per-city offset (NY .08 ... DEN -.42 F,
sd <= .12 F) (results/wx_bot_flow/omcheck.json). The 12Z run's next-day fields land on Open-Meteo ~16:00-16:25Z, run complete
~17:33Z; the 06Z run ~11:33Z.
### GEFS 31-member counts are too narrow to be bracket probabilities
Day-1 member-max sd 0.8-2.2 F; 48% of in-band brackets get 0 of 31 members (p clipped .05), and the bot rule flags BUY-YES on
413 of 1,405 brackets (256 of them priced <= 15c) -- yet those brackets hit at their price (results/wx_bot_flow/score.json).
### Kalshi weather D-1 takers are mostly YES buyers
65.0% of taker contracts 15-19Z on D-1 are YES-buys; 67.4% on brackets priced <= 15c, equally on bot-BUY and other cheap
brackets (results/wx_bot_flow/posthoc.json). E4's longshot premium is paid by diffuse YES demand, not by a predictable bot.
### Dry-run every regression on permuted outcomes before scoring
won - mid regressed on (p - ask) gave a mechanical slope +.27 [.12,.39] on scrambled outcomes because both sides share the
price; adding mid as a control removed it (+.04 [-.15,.21]) (notes/wx_bot_flow.md 04:30Z amendment).
### Research pulls share the live bot's IP rate budget; throttle and checkpoint
6 unthrottled threads drew Kalshi 429s (a plain curl also got 429) and one uncaught ReadTimeout lost a 40-minute pull; use
1-2 threads, pause around the live executor runs (03-06:00Z), retry RequestException, checkpoint per ticker.
