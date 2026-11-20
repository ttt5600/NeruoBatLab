# STRATEGY v2 -- what ~100 pre-registered tests say about where edge lives (agent strategy_synthesis, 2026-11-19)

User: "take our learnings on building models to understand a better strategy". Synthesis only; no new trial. Sources: FUND_BOOK
(FB l.N), LEARNINGS_2026-11 (LRN l.N), EDGES_AND_AGENTS (EA l.N), forward_review_2026-11-19 (FR l.N), named result notes.
Scratch: results/strategy_synthesis/ (taxonomy.csv, rubric_score.py, rubric_kalshi.csv, candidate_probe_{1,2,3}.json).

## 1. Empirical taxonomy (taxonomy.csv: 107 idea-rows, each cited; rows bundle the ~351 ledger trials, FB l.385)
Mechanism codes: NOWCAST = official free source the settlement derives from, forecast/observed before the book reprices;
SHARPLINE = a sharp price (options, futures, sportsbook main line) translated into a derived Kalshi market; MODEL = our own
predictive model from public stats vs a primary line; BIAS = FLB / behavioural / situational angle; CROSSVENUE; TIMING =
news/latency; MAKER = resting-order / liquidity as the mechanism; ALTDATA; OFFVENUE = stocks/crypto carry.

| mechanism | rows | pass | lead | dead | CI>0 sealed | CI>0 dev only | CLV>0 but ROI not |
|---|---|---|---|---|---|---|---|
| NOWCAST | 26 | 3 (E1,E2,E4) | 7 | 16 | 5 (E1,E2,E4,E5,TSA) | 5 (L39,E4P,E4-timing,R1,SX1) | 0 |
| BIAS / FLB | 21 | 0 | 3 | 18 | 0 | 6 (M3,WCN,PM10,E2p,LX2c,CS2F post-hoc) | 0 |
| MODEL | 20 | 0 | 3 | 17 | 0 | 1 (R_REB) | 6 (O1/L32/L36,K1,K-CFB/K50/CBB,NHL-om,L33,P-PAINT) |
| TIMING | 12 | 0 | 2 | 10 | 0 | 0 | 1 (L35) |
| CROSSVENUE | 10 | 1 (E3, untradeable from CA) | 0 | 9 | 1 | 0 | 0 |
| MAKER | 7 | 0 | 0 | 7 | 0 | 2 (Q003,MKdog) | 0 |
| SHARPLINE | 7 | 0 | 3 (OT1,KFG1,CB1) | 4 | 0 | 2 (OT1,KFG1) | 0 |
| ALTDATA / OFFVENUE | 2 / 2 | 0 | 0 | 2 / 2 | 0 | 0 / 1 (FC1) | 0 |

Death reasons (77 dead rows): priced 35 · fees/spread 11 · adverse selection 10 · leakage/artefact 8 · model error 6 · too few
events 5 · decay dev->sealed 5 · paid/unobservable data 3 · other 4. By venue: Kalshi 48 dead / 18 lead / 3 pass; books 20 dead.
**Sealed replication by mechanism** (the number that matters): NOWCAST 5 of 6 sealed tests passed (only R1 failed, FB l.312);
soft-vs-sharp CROSSVENUE 1/1 (E3); every other mechanism 0 of 12 (O1, L32, L36, L37, R_REB, WCN, M3, Q003, L28, E2p, C1, PM10;
FB l.164-362). Dev CIs above zero outside NOWCAST were all forking-path or decay.
Reading: one mechanism class has produced every real edge -- a free official source + a stale tail nobody re-prices. Being a
better forecaster of a WATCHED number never paid; being the only one who re-prices an UNWATCHED tail did. The new partial
exception is SHARPLINE into a derived ladder (OT1 dev +4.60% [+1.07,+7.62], LRN l.268; KFG1 +23% game-CI [-0.7,+47.3]).

## 2. Model-building lessons
**When our model beat the market's calibration** (only with an independent, better-measured input):
weather ratio model log loss .1054 vs price .1098 (TRADING_PRINCIPLES l.144); Deribit options Brier .04568 < Kalshi .04617 (OT1,
LRN l.271); KFG1 all-rung Brier gap +0.0063 [+0.0016,+0.0109] -- but on tight (spread <= .03) rungs +0.0048 [-0.0013,+0.0112]
n.d. (kalshi_fg_markets.md l.61-63); crypto barrier M1 .0605 vs Kalshi .0634 n.s. (LRN l.223).
**When the market beat us** (every model built from public sports/news stats): NBA in-game Kalshi .1284 vs model .1319 (FB l.236);
Kalshi .171 vs ESPN .180 (FB l.232); NBA live totals .1196 vs .1240 (FB l.249); Elo .207 vs market .199 (FB l.262); L2 close
.1941 vs .2048 (FB l.190); public models +.008..+.019 log loss (FB l.258); CS2 Elo log loss .6663 vs Kalshi .5963, stacking coef
-0.016 [-0.20,+0.18] (cs2_model.md); NYC hourly index .0145 vs .0259 (LRN l.216); RT nowcast .236 vs .136 (LRN l.177); Q014
.138 vs .081 (FR l.290). CBB in-game probit beat ESPN (.1181 vs .1193) yet lost to Kalshi (EA l.352): beating a public model
is not beating the price.
**Feature classes the market already prices** (do not build features around them): rest/B2B/travel/altitude/time zone (L7 0/11,
LRN l.345; B2B markdown grew with the penalty, LRN l.351), referees (FB l.264), luck/PDO (FB l.265/278), injuries, lineups,
goalies, inactives (priced 6-12 h / before the PDF: LRN l.310, l.322, EA l.315), weather forecasts in totals (LRN l.297), team
"clutch"/comeback traits (LRN l.357), form/hype/recency (FB l.226, l.269), paint style (~30% priced but sub-vig, LRN l.106).
**Opener models are real but sub-cost**: CLV +0.8-0.9 pts across NBA/CFB/CBB/NHL vs taker break-even ~1.2-1.4 (LRN l.155,
l.334); K1 CLV vs Pinnacle close +0.78 [+0.04,+1.17] but vs Kalshi's own close -0.4c and -3.3c [-5.8,-1.2] after ask+fee (FR
l.273). The constraint is execution cost, not prediction.
**Model errors that killed rules**: Q014 0-10% bin predicted .006, happened .15 (FR l.305); Q004 raw bias +3.85 pts -> 47/48
overs (LRN l.341); nba_sim BPM-lite player impact made Morant's absence help POR, implied SD 23.7 vs ~12.5 (EA l.333, FB l.257);
HURCAT used NHC's point forecast as a ceiling, rapid intensification beat it by 23-48 kt (LRN l.248); X2 1-day low forecasts
miss > 3F (FB l.311); Q015 vol too wide (EA l.305); CS2 Elo disagreed most on underdogs, where the market was right. Common
thread: the model's TAILS were wrong, and rules select exactly the tails where model and market disagree most (E4P "rule PA":
where disagreement is largest, the market is right, LRN l.233).
**Recurring traps** (each cost a fake result once): datetime64[us]//1e9 -> fake +62% (FB l.231); mid vs ask -- PM favourite +3.3%
at mid lived entirely in bids that never fill (EA l.357); untraded markets print ~0.5 (pm_full_census; pm_timing); post-hoc
bands (CS2 .65-.80 +4.0% next to .50-.65 -6.5%); placebos that match the rule (XV-US stale-reference placebo +11.1% vs rule
+1.73%; KFG1 placebo +5% = base-rate shape, not team info); lifetime-volume selection (OVR +26.5% artefact, FB l.242); row-
position joins (NFL weather 53% wrong) and unsigned spreads (+7.2%) (FB l.453); revised FRED vintages; < ~30 clusters -> CI too
narrow (KFG1 date CI [+8.8,+41.1] vs game CI [-0.7,+47.3]); name matching 10-13% wrong without an end-date check; proportional
de-vig fakes longshot CLV; cached "free Pinnacle" (LRN l.149); maker fills = losers (E5 6/6, RD2 9/9, fairfill gap -3.10 pts).

## 3. Scoring rubric for a new candidate market (0-10; weights follow the death counts in section 1)
| item | 0 | 1 | 2 | evidence for the item |
|---|---|---|---|---|
| R1 settlement source free + reconstructable at its resolution | paid / proprietary index / revised first print | public, exact | -- | CA1, ND1, PW, home values died here |
| R2 sharper independent input the book may not translate | only our model from public stats | consensus/futures everyone reads | multi-model forecast, or a sharp main line one step from a DERIVED ladder | sealed passes all had 2; MODEL rows all 0 |
| R3 neglect: far rungs 2-15c with 0 24h volume (series_watch S4) | < 10%, or new series seeded by a maker | 10-30% | >= 30% | E4 old neglected book vs state gas/ERCOT/econ events (LRN l.255) |
| R4 power: settled independent events / month | < 4, or season-settled | 4-20 | >= 20 | L39 null p = price^n (LRN l.284); LX2c concentration (LRN l.188) |
| R5 cost at decision time: median spread on tradeable rungs + fee type | > 4c, or PM US 0.095 fee | <= 4c | <= 2c and maker-free (`quadratic`) | fees/spread 11 + adverse 10 deaths |
| R6 who sees settlement data first | subscribers / insiders / live-watchers | everyone, same instant | -- | Carbon Arc, NFL inactives, RT, L26 lock-in |
Gates on top of the score: not in FB 3d or a sibling with the same mechanism; any maker variant must report filled-vs-unfilled
outcomes; any rule must beat a base-rate-shape placebo (KFG1). Retro-check: E4 10, TSA weekly 8, L39 8, KFG1 8, E5 8 (died on
R5 execution), CS2E 5, M3 5, CA1 ~3. The rubric is a prior filter; R2 is what separated passes from deaths, R5 what separated
paper edges from money.

## 4. Universe scored (series_watch live snapshot 2026-11-19T21:43Z: 16,751 series, 5,181 with open events; 4,625 left after
the tested/dead filter; auto-score rubric_kalshi.csv, then the top ~40 checked by hand with live read-only GETs for spread and
settled events in the last 90 days -> candidate_probe_*.json). PM US from pm_full_census pmus_bbo.parquet.
**Overlap warning:** agent derived_ladders started 2026-11-20T01:14Z (inbox) on Kalshi derived sports ladders -- rows marked [DL]
belong to it; brief it with this table rather than starting a second agent.

| # | candidate (series) | R1-R6 = score | mechanism | data source | ev/mo (90d settled) | spread | one-line test design |
|---|---|---|---|---|---|---|---|
| 1 | Soccer correct score, 7 leagues (KXEPLSCORE, KXLALIGASCORE, KXBUNDESLIGASCORE, KXLIGUE1SCORE, KXMLSSCORE, KXLIGAMXSCORE, KXBRASILEIROSCORE) [DL] | 1,2,2,2,1,1 = 9 | SHARPLINE derived | Kalshi's own 1X2 + total (made off Pinnacle, FB l.333) -> Dixon-Coles; Kalshi settled results | ~201 combined (23-51/league) | .02-.03 | T-60 Poisson from Kalshi 1X2+total mids, buy rung if edge-fee >= .04, game-clustered CI, placebo = league-average score grid |
| 2 | NFL half-time/full-time result (KXNFL1HFT) [DL] | 1,2,2,2,1,1 = 9 | SHARPLINE derived | Kalshi spread+total; nflverse pbp for 1H/2H splits | 49 (season) | .04 | KFG1 template: logit on implied margin/total fit 2023-25, T-60 ask, placebo shuffle |
| 3 | NFL win-margin buckets (KXNFLWINMARGIN) [DL] | 1,2,2,2,0,1 = 8 | SHARPLINE derived | spread+total -> margin distribution (nflverse) | 49 | .06-.07 | same; maker only if spread > edge, with filled-vs-unfilled check |
| 4 | NFL overtime / tie (KXNFLOT, KXNFLTIE) [DL] | 1,2,0,2,2,1 = 8 | SHARPLINE derived (base-rate) | P(OT) from |spread|, total (nflverse 2015-25) | 49 / 34 | .01 | single rung per game; 1 season dev, placebo = flat league OT rate |
| 5 | UFC method of victory (KXUFCMOV) | 1,1,1,2,2,1 = 8 | SHARPLINE derived | Kalshi KXUFCFIGHT (spread .01) + weight-class finish rates (UFCStats, free) | 52 | .01-.02 | MOV = P(win) x finish-mix model; T-60 ask; cluster by card |
| 6 | Small-country central-bank decisions (26 KXCBD* series) | 1,1,2,1,2,1 = 8 | SHARPLINE (consensus) | central-bank statements + Trading Economics forecast (settlement source on some) | ~16 aggregate (19 settled/90d over 10 probed) | .01-.05 | sibling Fed-vs-futures died on a busy book (LRN l.58); test only thin banks; NO on rungs > 2 steps from consensus |
| 7 | NFL both teams score (KXNFLBOTH) [DL] | 1,2,2,2,0,1 = 8 | SHARPLINE derived | team implied totals | 49 | .11 | maker-only by cost; descriptive calibration first |
| 8 | Soccer 1H/2H/team totals (KXEPL1HTOTAL, KXBUNDESLIGA1HTOTAL, *TEAMTOTAL) [DL] | 1,2,0,2,1-2,1 = 7-8 | SHARPLINE derived | Kalshi main total + 1X2 | 20-26/league | .01-.05 | series created 11-18/19 (no history) -> forward paper only |
| 9 | NBA win margin (KXNBAWINMARGIN) [DL] | 1,2,1,2,0,1 = 7 | SHARPLINE derived | spread+total, NBA pbp | 74 | .07-.09 | as #3 |
| 10 | NCAAMB first-half winner (KXNCAAMB1H) [DL] | 1,2,0,2,1,1 = 7 | SHARPLINE derived | full-game spread (ESPN DK lines since 2023-24, LRN l.338) | 126 | .04 | 1H spread ~ half-game; largest n of any derived series |
| 11 | Daily TSA screenings (KXTSAD) | 1,2,2,1,0,1 = 7 | NOWCAST | TSA daily checkpoint page (sibling of live TSA weekly lead B4) | 13.7 | .05 | same-weekday-last-week x YoY + holiday calendar; far rungs only |
| 12 | UFC round of victory (KXUFCVICROUND) | 1,1,2,2,0,1 = 7 | SHARPLINE derived | as #5 | 52 | .05-.09 | with #5 as one family |
| 13 | NFL highest-scoring team of week (KXNFLWEEKHIGHSCORE) | 1,2,2,0,0,1 = 6 | SHARPLINE derived | 32 implied team totals -> simulation | 3 | .06 | forward only (power) |
| 14 | OpenRouter / LM Arena AI series (KXTOKENUSE, KXCHINAAI) | 1,1,2,1,1,0 = 6 | NOWCAST (partial week observed) | openrouter.ai/rankings (200 OK); LM Arena 403 (info_gap_census) | 4 | .035-.05 | G-TOKENS stub (info_gap_census l.111): partial-week tokens vs far rungs |
| 15 | NOAA monthly global temp (KXHMONTHRANGE) | 1,2,0,0,2,1 = 6 | NOWCAST (earlier free source) | Copernicus ERA5 Climate Pulse publishes before NOAA | 1 | .02 | ForecastEx sibling FC1 died n2 (fx_climate.md); forward log 12 months |
| -- | PM US best: technology/AI (226 mkts) 5; macro commodity gas/diesel 5 | R5 = 0-1 everywhere (fee .095) | -- | pmus_bbo.parquet | AI two-sided 23%; commodity 2 closed events/series | .01 / .04 | not worth agent time until a Kalshi version passes |
Excluded as dead siblings: KXDIESELW/D (N6 gas / mortgage pattern), Billboard/album/Netflix/Steam (LRN l.291, FB l.244), KX*15M
commodities and crypto ranges (Q012/Q016/OT1-ext), player-prop ladders (Kalshi vs DK gap 0c, FB l.295), season futures (R4 = 0).

## 5. Operating strategy, next 30 days (2026-11-20 .. 12-19)
**Agent time.** (a) 50%: SHARPLINE-into-derived-ladder, the only non-weather class with a fresh CI-above-zero signal and >= 20
events/month -- one agent (derived_ladders, already running) works #1-#5/#7-#10 as ONE family with one placebo design; first
deliverable is descriptive calibration by rung (no returns) so the family bar is paid once. (b) 20%: NOWCAST siblings with power
(#11 TSA daily, #14 OpenRouter) + E4P/E4-0600 forward. (c) 20%: ops -- E4 is the only money edge; health monitor, fills
reconciliation, X1 kill rule (SIZING_RULE.md). (d) 10%: rubric retro-validation (score every FB 3d row; does R2+R5 rank the 6
sealed passes above the 12 sealed fails?).
**Sizing for leads in forward paper** (from SIZING_RULE.md, unchanged): $0 real until >= 100 settled forward trades with ROI
90% CI lower bound > 0 on the strategy's own exchange fills; then 1/4 Kelly on that lower bound, capped by 50% of depth within
1c and a 20% same-event basket, steps <= 2x per 14 days. Tier-B judgment bets stay $1-3 with the pre-set kill (ACT_LIST.md).
At these edges the forward SPRT needs ~14-25 months (LRN l.165), so prefer families with >= 50 events/month.
**Stop doing.** New opener/total models for Kalshi taker (4 sports of CLV without ROI; let K1/K-CFB/NHL-om loggers reach their
pre-set rules, no new work). Models from public sports stats vs headline markets (CS2E, L2, public models: 0 of 7 sealed).
Maker strategies on event-day markets without a filled-vs-unfilled check. PM US cross-venue or PM US anything at fee .095.
Paid-panel / proprietary-index settlements. Season-settled or < 4 events/month series as dev trials (forward-log only).
Post-hoc bands as leads without a fresh window (CS2F is the last one; score once, then stop).

## Learnings
### Only the nowcast mechanism has replicated in sealed tests
NOWCAST 5 of 6 sealed tests passed (E1, E2, E4, E5, TSA; R1 failed); every MODEL / BIAS / MAKER / TIMING idea that reached a
sealed test failed (0 of 12). Source: results/strategy_synthesis/taxonomy.csv, FB l.152-362. Spend agent time where the
mechanism has replicated, not where dev CIs look good.
### "Priced" is the most common death, then cost and adverse selection
77 dead rows: priced 35, fees/spread 11, adverse selection 10, artefacts 8, model error 6. A new idea must answer "who else
re-prices this?" and "what does the spread cost?" before any model is built. Source: taxonomy.csv.
### Our models beat the market's calibration only with an independent measured input
Weather multi-model (.1054 vs .1098), options (.04568 vs .04617) beat the price; every public-stats model lost (CS2 .6663 vs
.5963; NBA in-game .1319 vs .1284; L2 .2048 vs .1941). Source: section 2 citations.
### Derived ladders of sharp main lines are the best untested family on Kalshi
~200 soccer correct-score events/month at 2-3c spreads plus NFL/NBA/NCAAMB/UFC derived series with 49-126 events/month; KFG1
showed base-rate-shape mispricing (placebo +5%). Source: results/strategy_synthesis/candidate_probe_*.json, kalshi_fg_markets.md.
### PM US fee .095 caps every PM US idea at the cost item
No PM US candidate scored above 5/10. Source: pm_full_census.md, pmus_bbo.parquet.
