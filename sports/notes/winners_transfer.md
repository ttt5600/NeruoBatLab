# winners_transfer -- what competition winners and profitable PM traders teach the fund (agent, started 2026-11-21 04:35Z)

User: "review our learnings from past competition winners and keep trying to synthesize better data or something to get
improvements." Protocol: notes/AGENT_PROTOCOL.md (own notes, inbox rows as I go, NO git, NO orders, no launchd).
Scratch: sports/results/winners_transfer/.

## Sources read (2026-11-21)
- Compete (read-only): ~/projects/compete/LEARNING_LOOP.md (batch-3 RESULT: gap is idea GENERATION; batch-4 DIFFGEN: wins
  came from O2 data-generating process / O6 reformulation, retrieval pushed to field defaults; proposed O8 default negation),
  PLAYBOOKS.md (C1-C26, META 1-8, C4 amendment, C20, C22, C25, C26 + guard), comps/wunder-lob/POSTMORTEM.md (K=11: HIT 5,
  NM 2, MISS 4; misses = teacher distillation, custom inference engine, per-group encoders, spending LB slots as
  measurements; process miss = host's earlier comps never researched), replays/BATCH3_JUDGE.md, BATCH4_JUDGE.md,
  DIFFGEN_STEP.md (operators O1-O7).
- Fund: STRATEGY_v2.md, FUND_BOOK.md s3 (3a-3d), LEARNINGS_2026-11.md headings, STATE.md, RESEARCH_QUEUE.md,
  prior prior-art notes (social_edges_2026-10-22.md, pm_literature_scout.md, forum_strategy_scout.md,
  trading_ways_survey.md), bar_calibration{,_v2}.md, consensus_reject_census.md.
- In-flight agents I must not duplicate (RESEARCH_QUEUE.md "Running"): e4_data_v2 (E4 headroom + CLI DGP + forecast
  backfill), ai_nowcast, climate_nowcast, cbank_thin, count_nowcast, money_now.

## Plan
1. C1 prior art on WINNERS (researcher subagent, web; rows tagged verified/claimed/vendor) -> section 1.
2. Map C1-C26, O1-O8, META 1-8 to fund actions -> section 2.
3. Generate ideas with the operators, filter vs FUND_BOOK 3d + wx_lockin (L26) + e4_timing dead -> top 10 -> section 3.
4. Run the cheapest 1-2 probes (pre-registered below BEFORE computing).
5. Process changes (<= 8 lines) -> section 4.

## PRE-REGISTRATIONS (written 2026-11-21 ~04:45Z, before any number below is computed)

### P1 DEVSEAL -- the fund's own calibrated pseudo-test (compete C26 + C15 "external gains shrink": T2 0.59x, T4 0.65x)
- Question: how much of a dev-period ROI survives into the one-shot sealed test, by mechanism class, and does a
  dev->sealed shrinkage model predict the sealed verdict better than the raw dev result?
- Data: every fund trial with BOTH a dev ROI and a sealed ROI in FUND_BOOK / EDGES_AND_AGENTS / the trial's notes file,
  each number cited by file:line. Units: ROI per $ staked under the trial's own price model. Mechanism class from
  results/strategy_synthesis/taxonomy.csv (NOWCAST vs everything else).
- Statistic: per trial transfer ratio k = sealed ROI / dev ROI; class-level k = median over trials, bootstrap CI over
  trials (2,000 resamples). Also third-stage live/forward vs sealed where it exists (E2, E4).
- Validator check (C26 step 5, leave-one-trial-out): rule A "dev passed -> predict sealed pass" (every row passed dev);
  rule B "predicted sealed ROI = k_class(-i) x dev_i > 0"; rule C "pass iff NOWCAST". Score = LOO accuracy on the
  sealed verdict as recorded in FUND_BOOK.
- Known before I start (no blinding possible): STRATEGY_v2 already counts NOWCAST 5/6 sealed passes vs 0/12 others,
  so rule C will score ~17/18. The NEW information is only the magnitude k (the haircut to apply to dev ROI of new
  nowcast trials such as CN1/TOK1/CB1) and whether magnitude adds anything beyond the class label.
- Success: NOWCAST k CI lower bound > 0 and a k point estimate usable as a haircut (report it). Kill/negative: B no
  better than C -> report "magnitude adds nothing beyond class; use the class gate".
- This is an instrument measurement (like bar_calibration), not a trading trial: no edgelab registration.

### P2 TAILPOOL -- oracle headroom before building (compete C22 "bound with an oracle arm before building")
- Question: for each candidate Kalshi series, how many $ per month did LOSING longshot buyers pay after the decision
  time? That is the ceiling for a perfect stale-tail nowcaster acting as their counterparty (E4's mechanism). If the
  ceiling is small, no nowcast on that series can be worth an agent.
- Definitions (public trade tape, GET /markets/trades; settled markets closing in the last 90 days):
  * decision window = trades in the final 24 h before close for markets open <= 3 days; final 7 days otherwise.
  * per trade: the YES side paid yes_price, the NO side paid no_price; the losing side is the one opposite settlement.
  * TAIL pool = sum(count x p_loser) over trades where p_loser <= 0.15 (someone paid <= 15c for an outcome that lost).
  * TAIL risk = sum(count x (1 - p_winner_side_sold)) where the <= 15c side WON (what a blind longshot-seller pays out).
  * FULL pool = sum(count x p_loser) over all trades (upper bound for any oracle, incl. coin-flip markets).
  * Report $/month, events/month, $/event, and each series as a fraction of the anchor.
- Anchor: KXHIGH orig-7 cities (E4's book), same window definition.
- Pre-set decision rule: a series whose TAIL pool < $100/month is "headroom-dead" (even 25% capture < $25/mo);
  a series with TAIL pool >= 25% of the anchor's is "headroom-OK". Everything between = "marginal".
- Measurement only (no trading rule, no edgelab trial). Read-only public GETs; >= 0.1 s between calls.

## Log
- 04:35Z started row in inbox. 04:40Z launched researcher subagent for C1 (named winners, verified vs claimed).
- 05:05Z P1 DEVSEAL computed (results/winners_transfer/devseal_pairs.csv = 15 trials with cited dev + sealed ROI;
  devseal.py -> devseal_result.json). See RESULT P1 below.
- 05:15Z P2 TAILPOOL smoke test on KXTSAW OK (114 requests, 38.5 s); full census launched in background
  (tailpool.py -> tailpool_result.json, log tailpool.log; 35 series).

## RESULT P1 DEVSEAL (2026-11-21; results/winners_transfer/devseal_result.json)
Transfer ratio k = sealed ROI / dev ROI, same price model per trial (sources per row in devseal_pairs.csv):
| class | n | k median | 90% bootstrap CI | per trial |
|---|---|---|---|---|
| NOWCAST | 6 | 0.664 | [0.415, 0.726] | E1 .737, X1 .640, E4 .689, E5 .522, TSA .726, R1 .142 |
| NOWCAST passes only | 5 | 0.689 | [0.522, 0.737] | -- |
| BIAS/MODEL/MAKER | 8 | 0.162 | [-0.573, 0.476] | M3 .104, Q003 .220, WCN .567, E2p -.568, L32 .732, L36 .385, R_REB -.578, CS2F -2.17 |
| CROSSVENUE | 1 | 1.020 | -- | E3 |
Leave-one-out validator (pre-registered): A "dev passed -> sealed pass" 6/15; B "k_class(-i) x dev > 0" 6/15;
C "pass iff NOWCAST/CROSSVENUE" 14/15. Verdict vs prereg: SUCCESS branch met (NOWCAST k CI lower 0.415 > 0, usable
haircut) AND the negative branch met for B (magnitude adds nothing beyond the class label: non-NOWCAST trials kept
positive sealed point estimates -- WCN +12.2%, L32 +6.4% -- and failed on power, so a shrunk-magnitude rule cannot
separate them). Reading:
- A NOWCAST dev ROI keeps about two-thirds into sealed (plan with 0.66, worst case 0.42). This is the same shrinkage the
  competitions showed CV->sealed (T2 0.59x, T4 0.65x; PLAYBOOKS 5i): the fund's dev->sealed step behaves like a
  calibrated CV only for the NOWCAST class.
- For BIAS/MODEL/MAKER, the median trial keeps ~16% with a CI spanning negative: a dev pass there carries almost no
  information about sealed sign. The class gate (STRATEGY_v2) is the validator; dev magnitude is not.
- Third stage (live/forward vs sealed) where it exists: E4 live maker +1.72% [-4.33,+6.11] n39 vs sealed +2.22%
  (FUND_BOOK 3a) and X1 forward -0.6% n91 vs sealed +0.78% -- too few to estimate a third ratio.
- Caveats: n = 6 vs 8; E5 sealed is at last trade (no executable channel, FUND_BOOK 3a); R_REB/CS2F dev n not in the
  ledger; L36 uses the -110 reference on both sides. Use k as a planning haircut, not a test statistic.
- Planning rule derived (proposal): before asking for a NOWCAST sealed shot, require dev ROI x 0.42 >= the effect the
  sealed window can detect at its expected n (EDGE_MATH power table); for any other class, skip the sealed shot unless
  the mechanism is moved into NOWCAST/SHARPLINE form.

### P3 E4CANCEL -- headroom of a cancel lever on E4 resting orders (C22 + O8 "negate rest-to-close"), DESCRIPTIVE
(written 2026-11-21 ~05:40Z before computing; dev rows only, results/e4_timing/picks_T22.parquet = e4_timing's dev
E4 picks at 22:00 with tape fill time fill_tt and the 06:00 D quote yes_bid_T06D/yes_ask_T06D; no sealed rows read.)
- Why: E4 losers are ALL filled (sealed loss_rate_filled .0307 vs unfilled 0, lab/lockbox.jsonl 2026-10-23;
  kalshi_temp_maker.md l.41). E4 rests to close with no cancel. A cancel that avoids part of the losing fills is the
  only lever with that much headroom; it is untested on Kalshi (E5 tried obs-cancel on ForecastEx: too late).
- Questions: (i) what share of LOSING fills happen after 06:00 local D (i.e. a 06:00 recheck could still act)?
  (ii) among picks still unfilled at 06:00, does a quote flag at 06:00 (YES bid at 06:00 >= 2x the 22:00 YES mid, or
  YES ask >= 0.10) separate later-losing from later-winning picks?
- Promising (pre-set): >= 30% of losing fills occur after 06:00 AND the flag catches >= 50% of those losers while
  flagging <= 15% of the post-06:00 winning fills. Then: propose an edgelab-registered cancel trial (lead/e4_data_v2),
  never a live change from here. Otherwise: "cancel lever has no 06:00 headroom" -> drop.
- Descriptive only: counts + loss/win shares, no ROI claim, no edgelab trial.

## RESULT P3 E4CANCEL (2026-11-21; dev picks_T22.parquet, 1,713 picks 2024-03-05..2026-02-28, 1,163 tape fills)
- Losing picks: 70, ALL filled (dev repeats the sealed pattern). 21/70 = 30.0% of losing fills happen after 06:00 D
  (49 before), i.e. most informed selling hits E4 overnight, before any 06:00 recheck.
- Of picks still unfilled at 06:00 (913): the pre-set 06:00 quote flag catches 18/21 later-losing fills (86%) but also
  159/342 later-winning fills (46%; pre-set ceiling 15%). Verdict vs prereg: NOT PROMISING (flag too broad).
- POST HOC economics (not pre-registered, reported to close the question): cancelling flagged post-06:00 fills changes
  dev units 32.65 -> 27.31, delta -5.34, date-clustered 90% CI [-10.71, +0.55] (355 dates); a random cancel at the same
  48.8% rate gives -4.79 on average -> the 06:00 quote adds nothing beyond chance. The flagged winners are the
  cheap-NO (high-payout) picks, so cancelling them throws away as much as the saved losers return.
- Lesson: the market's 06:00 move is fair pricing of risk, not a leak we can step aside from; E4's losers are bought
  overnight by informed flow (00Z runs) that a resting order cannot see coming. Cancel lever: DROP (both halves
  negative: H1 -2.63, H2 -2.71).

## 1. C1 PRIOR ART FROM WINNERS IN OUR ARENA (researcher subagent, 2026-11-21; raw pulls in results/winners_transfer/
pm_lb_all_pnl.json, pm_lb_category_top50.json, pm_user_windows.json, pm_gopfan2_by_category.json)
Grades: VERIFIED = on-chain / exchange API / academic data actually loaded; CLAIMED = interview or self-report;
VENDOR = selling something. "Status" = vs our ledger: DO (we already do it), DEAD (tested, FUND_BOOK row), NEVER.
| who / venue / period | record as stated (grade) | DIFFERENTIATOR | status for us |
|---|---|---|---|
| Theo (Fredi9999 + Theo4), Polymarket, 2024 election | Data API all-time PnL Theo4 $22,053,934, Fredi9999 $16,619,507 (VERIFIED, loaded); Chainalysis "nearly $79 million" over 9 accounts (The Block 2024-11-07) | commissioned his OWN poll using the "who are your neighbours voting for" question; split buys over days/4 accounts to avoid moving price (WSJ 2024-11-01 via tovima.com) | NEVER (primary data collection). Cheap analogue = measuring the settlement variable ourselves = NOWCAST class |
| gopfan2, Polymarket | all-time $1,706,473 on $70.8M volume; POLITICS $1,290,103, WEATHER $317,221 (weather #1), SPORTS -$230,194; last month -$15,361 (VERIFIED, data-api) | method only in blogs ("YES < 15c, ~$1/position", VENDOR/secondary) | weather fame is mostly politics money; longshot-YES = our dead FLB rows (FUND_BOOK 3d G2/G3) |
| Polymarket category leaders (Hans323 weather $87,714; bama124 mentions $293,246; TheReturnOfDarthMaul economics $1,523,001; swisstony sports $18,482,391 on $1.85B vol, -$725,810 last month; 0x8dxd crypto $2,050,245 on $200M) | VERIFIED (API) | methods not public; PnL/volume ~0.5 (Theo4) vs ~0.01 (swisstony) separates one-shot directional from volume/MM styles | weather all-time #1 is only $317k: weather is a small-dollar arena even for the best |
| Domer, Polymarket since 2007 | no verified P&L (not found in API) | speed then slow second look ("more than three seconds ... glacially slow"); discounts known-biased sources; prices the risk-free rate into long-dated markets; fights resolution rules; bots: "at least two people have tried it ... not much success" (ChinaTalk 2025-07-29, loaded) | risk-free rate = LX2 (cash rate, DEAD); rules = rules_quirks (prices follow Kalshi rules, DEAD); speed = TIMING class 0/12 |
| Eric & Jerry, Kalshi, 2024-25 | "over 100 times what we started with", ~$30M volume, worst day "a bit over $20,000" (CLAIMED, Kalshi blog 2025-07-11, loaded) | "Most of the time we're providing liquidity as market makers"; "the main thing we do differently is using the API and automation"; modelled the WEATHER STATION itself (sensors metres apart read differently); left weather when "losers left" | E4 = our maker-on-weather (DO); two-sided MM DEAD (symmetric ladder maker results.md 10-18); station siting -> route to e4_data_v2 |
| WindBorne staff / CEO John Dean, Kalshi weather | $200 -> $327.79 (CLAIMED, Bloomberg via Insurance Journal 2026-04-15) | official station sensors SPIKE in direct sun (sensor DGP) | NEVER as a feature -> e4_data_v2 (CLI max vs forecast on sunny calm days) |
| Iabvek, Kalshi elections 2016-25 | "near $1 million" (Kalshi blog 2025-10-31) vs $2.5M via NYT snippet (CLAIMED; figures conflict) | start from "sharp conventional wisdom", add race-specific primary data (voter files, uncounted ballots); Discord team shares models | elections n/a until 2028; pattern = sharp prior + private data = Benter |
| LanceUpper, Kalshi, Feb 2025 | "$2,700 into $105,000" (CLAIMED, Kalshi blog) | verified the SOURCE of a viral fake quote; bought NO at pennies while market sat in the 90s | one event; news-verification = TIMING class (GDELT: price moves before news, DEAD) |
| Evan Semet / Logan Sudeith, Kalshi + PM | "six figures a month" / "$100,000 last month" (CLAIMED, NPR 2026-01-17, unverified) | statistical models on AWS; first win on TSA counts ("pretty modelable") | TSA weekly maker = our live lead (DO); daily TSA DEAD (ufc_tsa) |
| Kalshi Trading (in-house MM) | "not profitable", ~$310M sports/month (co-founder, CDC Gaming 2025-12-01) | negative result | supports our maker-on-sports DEAD rows |
| rlafuente, Kalshi MM bot | -$150 in ~20 min (CLAIMED, blog) | sign bug + quoting the widest illiquid markets + unexitable combos | negative result; combos RFQ stays parked |
| Intrade 2012, 6,300 accounts (Rothschild & Sethi) | top account D +$867,059 directional; arbitrageur A +$61,871 on $9,877 margin; 40 arb accounts = 16% of volume (VERIFIED, JPM WP Table 1) | big money = few directional bets; steady money = arbitrage | LX1 arbs absent on Kalshi (DEAD) |
| Base rates | Pew: 58% of 11,989 PM accounts within $100 of breakeven (via gcn.com 2026-08-14); Juice Reel: only accounts > $500K activity had positive median ROI (+2.6%), smallest -26.8% (FinanceMagnates 2026-03-25) | survivorship is heavy; recent windows often negative (gopfan2, swisstony) | read every "winner" as a prior |
| Voulgaris (NBA) | Ewing model ROI > 6% (2010-11), 5.14% (2011-12) (ESPN 2013-02-21; bankrolls hypothetical) | books "arrived at a total for the full game and cut that figure roughly down the middle" (mechanical derived line); edge died when books fixed it | derived ladders calibrated at T-60 (DEAD, LEARNINGS l.406) |
| Benter (HK racing) | "five years of actual implementation" (abstract) | logit combining a fundamental model WITH the public odds; a model counts only if it adds information next to the price | DO in E4P/OT1/CS2 stacking (CS2 coef n.d.) -> make it mandatory |
| Kaunitz et al. 2017 | profitable 5 months real money, then restricted (arXiv 1710.02824) | bet where one book is off the consensus | E3 passed sealed, untradeable from CA |
Recurring differentiator TYPES among verified/credible winners: (1) data the trader collected or bought (Theo, Iabvek,
station modelling); (2) knowing how the number is generated/settled (Voulgaris' half-total, sensor spikes, CLI rules,
mention word forms, Domer's rule fights); (3) checking sources (LanceUpper, Domer); (4) API market making across many
markets earning ~1% of volume (Eric & Jerry, Intrade arbitrageur, swisstony); (5) being early and leaving when the venue
learns (Voulgaris, Eric & Jerry "losers left"). Our only proven edge (E4) already combines (2)+(4) on a neglected book.
Never considered by us: (1) primary data collection beyond free scraping; sensor-level DGP of weather stations.
Checkable Kalshi facts (help.kalshi.com 2026-07-22, loaded by the researcher): daily high/low settle on the FINAL NWS
CLI released next morning, in local STANDARD time (during DST the day runs 01:00-00:59); settlement is DELAYED if the
high conflicts with METAR 6-h/24-h highs or the final CLI is lower than the preliminary. Hourly markets settle on The
Weather Company's value. Mentions: exact phrase incl. plural/possessive counts; tense changes and pre-recorded clips do
not (DeFi Rate 2026-02-24). LIP score ~ size x discount^distance on random snapshots; ex-Kalshi MM: "snapshot farming
unlikely to be profitable" (UFO Holdings 2025-08-25). No public source found for "traders use HRRR". No Kalshi weather /
economics / mentions trader has a verifiable P&L in public sources.

## 2. COMPETE RULES / OPERATORS -> FUND ACTIONS (C1-C26, O1-O8, META 1-8; PLAYBOOKS.md, DIFFGEN_STEP.md)
| rule | fund action (or n/a) | one-line reason |
|---|---|---|
| C1 prior art from host's past + differentiators | section 1 above; every agent brief also reads the SIBLING series' history and Kalshi's own research posts | Wunder's process miss was never reading the host's earlier comps |
| C2 public claims (wins AND nulls) are hypotheses | forum/vendor claims stay graded A/S (forum_strategy_scout); a dead row whose death cites only literature gets one cheap own test | symmetric scepticism (META-4) |
| C3 policy from the metric's unit | already: edgelab/fees.py charges ceil-to-cent per order, quadratic vs maker-fee series, break-even per contract | HIT; table stakes |
| C4 + amendment: label-process out; reverse-engineer how the data were generated | settlement-source DGP table (below) + PRICE DGP (who quotes: designated MM formula, open-source retail bots) for every series before modelling | decisive in 5/5 tabular comps; every fund pass is a settlement-DGP edge |
| C5 read rules/API for every lever | rules_primary/secondary + fee_type + LIP programs + RFQ/combos per series; combos-as-retail-quoter stays "pending ruling" (ask Kalshi) with a compliant fallback | rules cut both ways |
| C6 adversarial validation train vs test | before a sealed ask: compare dev vs recent market microstructure (24h volume, spread, maker presence, # trades); a dev-vs-now gap = regime risk | M3 died as volume grew ~10x; R1 summer->autumn |
| C7 every shift axis gets a named fix | each prereg lists season / volume growth / fee change / maker entry and its fix (season-stratified dev, recency weights) | R1 k = 0.14 (P1) was an unnamed season shift |
| C8 unlabelled test inputs: online refit, cross-sectional test-time stats | rolling refit of calibration on forward rows (E4P); same-night cross-city stats | low priority; C4 amendment ranks online refit above pseudo-labels |
| C9 aggregate folds like the test | date/event-clustered bootstraps | HIT |
| C10 channels / confusers | n/a | weak rule, no fund analogue |
| C11 count effective units | month-level loss clusters set E4's effective n (bar_calibration_v2) | HIT |
| C12 drive arms from categorised errors | loss autopsy table for every E4 loser (time of fill, side, model miss, station); P3 shows 70% of losing fills are overnight | next arm should target the largest loss category |
| C13 engineer the interface first | execution reliability (timeouts, order-list lag, cancel-all bug) | HIT (health monitor) |
| C14 constraint changes the student, not the idea | dead-by-constraint rows (E3 not tradeable from CA, E5 no executable channel) get a reroute line (same signal on a legal venue, or as a read-only feature) | META-7 |
| C15 spend LB slots as measurements | a small standing budget of real-money MEASUREMENT orders for execution parameters paper cannot see (fill rate at offsets, queue position, adverse selection on real fills) | Wunder MISS D11; FX fills were ~8% vs a 67-77% Kalshi proxy (FUND_BOOK E5) |
| C16 who made the TEST labels; audit 50 rows | settlement audit: Kalshi settled value vs our reconstructed source on >= 50 markets per series before any rule | home values (Zillow first print), CLI vs METAR |
| C17 judge artefacts by the consumer's loss | judge forecasts by P&L at executable prices, not Brier/MAE | HIT (E4P Brier better, money not) |
| C18 final build refit + hedge operating point | n/a for frozen live rules (SPRT needs a frozen rule); refit only at a pre-registered review | conflicts with forward certification |
| C19 scale is an axis | capacity curve per edge (size vs fill vs adverse selection); P2 tail pools give the depth ceiling | HIT (SIZING_RULE) + P2 |
| C20 hybrid solver -> learner | the market price is ALWAYS a feature next to any physical/sharp model (Benter); judge the model by its stacking coefficient over the price | CS2 coef -0.016 n.d., L2 .115 n.d. died exactly this way |
| C21 per-segment policy | per-city/season E4 margin with OOF check -> e4_data_v2 | "wins on k of n" needs per-segment choice |
| C22 headroom / oracle before building | P2 TAILPOOL per series (oracle $ for a perfect tail nowcaster) + P3 cancel oracle; one headroom row per new trial | skip ideas whose perfect-information ceiling is tiny |
| C23 effective independence | the 5 forecast models share inputs (NBM ingests GFS); correlated voters overstate consensus -> e4_data_v2 EMOS | consensus of correlated members ~ one voter |
| C24 output stack is an arm | per-candidate break-even with fee + spread + P(fill) | HIT (edgelab) |
| C25 generated data | backfilled point-in-time forecasts (e4_data_v2); pseudo-history of new series from the settlement archive; tape-based fill simulators | lets a new series be judged on day 1 |
| C26 calibrated pseudo-test | P1 DEVSEAL: dev->sealed k = 0.66 for NOWCAST, 0.16 (CI spans 0) otherwise; class gate is the validator (14/15) | guard met: the points carry ranking information only through class |
| O1 transfer | section 1; analogues: Theo -> own measurement, Benter -> price as feature, Eric&Jerry -> station modelling | -- |
| O2 data-generating process | settlement-DGP table below + price-DGP (MM formula, retail bots) | the operator behind every compete differentiator in batch 4 |
| O3 metric | quadratic fee is smallest at tails; makers free on weather; positions earn ~3.5% APY (LX2) | HIT |
| O4 unused information | Kalshi public tape (flow, fills), incentive-program list, PM-international prices as READ-only signals, settlement archives, rules text | tape already used for fills; flow-as-signal untested beyond P3 |
| O5 persistence x scale | breadth: one generic stale-tail engine over all free-source ladders, ranked by P2 pool | Domer/swisstony breadth; our power problem is too few bets |
| O6 reformulation | forecasting -> being the counterparty to a modelled flow (retail bots); per-bracket -> ladder-level; trading -> paid liquidity on rungs a nowcast knows | batch-4 wins came from O6 |
| O7 median negation | median retail = taker YES on longshots; median quant = GFS-ensemble bot at 8% edge; median fund trial = public-stats model vs headline market (0/7 sealed) | fade what the median does |
| O8 default negation | negate our own defaults one at a time: rest-to-close (P3: dead), 22:00 decision (e4_timing: dead), free data only (paid feed: user), one margin for all cities (e4_data_v2), dev->sealed for season-settled (forward only) | -- |
| META-1 table stakes vs differentiators | each prereg names its differentiator: what information does the price lack? | 0/12 non-NOWCAST sealed |
| META-2 named then argued away | RESEARCH_QUEUE items parked without a measured reason become "pending" with a cheap probe | -- |
| META-3 over-built pseudo-tests | for NOWCAST, dev->sealed is reliable (k 0.66) -> go to sealed/forward sooner; for others skip dev frameworks | P1 |
| META-4 asymmetric scepticism | "makers lose to adverse selection" generalised from sports/FX, yet E4 (maker) is our only edge: maker is fine where the side is chosen by an unwatched source | -- |
| META-5 gates admit shortcuts | traps.py (datetime, lookahead, revised vintages) | HIT |
| META-6 external slots unused | sealed windows are LB slots: batch sealed asks; forward loggers are free slots; measurement orders (C15) | 17 sealed scores in a month (lab/lockbox.jsonl) |
| META-7 kill on constraint | see C14 | -- |
| META-8 prior art not filtered by RULES | CA legality, ToS, PM US fee .095 filter every idea | HIT (forum_strategy_scout) |

### Settlement-source DGP map (O2 / C4 amendment) -- series we trade or could trade
| series | who measures / publishes | quirk that a model of the PROCESS could exploit | status |
|---|---|---|---|
| KXHIGH*/KXLOWT* daily | NWS CLI (ASOS), final report next morning, LST day | LST day (01:00-00:59 in DST); CLI vs METAR 6-h/24-h conflicts delay settlement; sensor sun-spikes; 5-min vs hourly max | E4 live; e4_data_v2 owns CLI DGP; X2 lows died on late-evening minima (x2_lows_window.md l.29) |
| KXRAIN*M monthly | NWS CLI monthly precip | month-to-date observed; thresholds become unreachable deterministically | L39 live |
| KXTSAW / KXTSAD | TSA checkpoint page, previous day ~morning | weekly = sum of daily prints; by Fri/Sat most days are known | TSAW lead live; TSAD dead |
| KXHMONTHRANGE / KXHMONTH | NOAA NCEI monthly | ERA5 publishes days earlier | climate_nowcast (P2: tiny pool) |
| KXTOKENUSE / KX*SHARE | openrouter.ai rankings page | partial week observable; page snapshot time | ai_nowcast (P2: biggest non-weather pool) |
| KXNETFLIX* | Netflix Tudum Top 10 (Tuesday) | views = hours / runtime; daily ranks public (FlixPatrol) | count_nowcast |
| KXCBD* | central-bank statement | consensus surveys | cbank_thin (P2: thin banks < $100-400/30d) |
| KXEOWEEK / KXTRUMPNOMNUM | Federal Register / Senate records | counts accrue during the window | count_nowcast (P2: ~$2.7-3.6k tail/30d) |
| crypto KXBTCD etc. | CF Benchmarks RTI, 60-s average | averaging narrows the final-minute distribution | OT1 forward |
| home values | Zillow page, first print | settles on first-published page value, history re-levels | forward check 12-15 |
| mentions | official transcript/video | exact phrase + plural/possessive; no tense changes / clips | M3 dead (static); live ASR lag untested |
