# ufc_decision_gap -- Kalshi UFC winner (KXUFCFIGHT) between the final horn and the announced decision

Agent ufc_decision_gap, started 2026-11-20 (lead brief; pool_census B: KXUFCFIGHT #21, $2.0M/30d reach, 51 fights/30d).
Scratch: results/ufc_decision_gap/. Rules: no git, no orders, read-only GETs, no launchd; sealed untouched.
Do-not-redo: U1 MOV (dead, notes/ufc_tsa.md); public-stats models vs headline markets 0/7 sealed (STRATEGY_v2 s.2).

## Question
Descriptive first: on fights that go to the judges, how is the Kalshi winner market priced in the window between the
final horn and the announcement (mid, spread, volume, how long the window lasts), and does a judges model built from
free round-by-round stats (sig. strikes, takedowns, control, knockdowns) predict the official decision better than the
window price? Expected failure (stated up front): the crowd watches the same broadcast; the book already prices it.

## Plan
1. Join KXUFCFIGHT finalized markets to UFCStats bouts; keep decisions (any "Decision - *").
2. Fix DEV/SEALED split by card date BEFORE any price is joined to an outcome.
3. Fetch the trade tape (and 1-min candles) around each decision fight's close; locate the announcement (price jump to
   ~0/1) and the final horn; verify on a sample; describe window book (DEV + SEALED descriptive WITHOUT outcomes is OK
   for timing/volume only; price-vs-outcome on DEV only).
4. Judges model from UFCStats round stats (fit on decisions before 2025-01-01; test 2025+), point-in-time = stats the
   UFC live stats show during the fight.
5. Compare model vs window price on DEV decisions (Brier/log loss, stacking coefficient, cluster by card). Prereg a rule
   only if the model adds information with a CI.

## Log
### 2026-11-21 join + split (results/ufc_decision_gap/join.py -> fights.parquet; no prices, no Kalshi outcomes)
KXUFCFIGHT finalized events in census_raw.json (ufc_tsa, pulled 11-19): 824 (2 markets each). Matched to a UFCStats bout:
694 (full 565 / last-name 84 / token 45); unmatched 130 = 53 cancelled (scalar fair-price settle), Contender Series /
non-UFCStats cards (2026-08..10 Tuesdays, ~56) and a few name variants (Su Mudaerji, Billy Ray Goff). Decisions among
matched: 305 (unanimous 258, split 37, majority 10), 2025-05 .. 2026-11.
Contract facts (market JSON): "This market will close and expire after a winner is declared" -> close_time = Kalshi's
close after the declaration (not the horn); settlement_timer 300 s; draw / NC -> both $0.50.
**SPLIT FIXED NOW (before any tape is fetched):** DEV = card date < 2026-07-01 (220 decision fights); SEALED = card
>= 2026-07-01 (85, 27.9%). SEALED tape/candles are not fetched at all (timing descriptives included) until a DEV pass +
user yes. Judges model: fit on UFCStats decisions dated < 2025-01-01; model test years 2025-26 (outcome-only, no prices).
### Judges model (outcome-only; results/ufc_decision_gap/rounds.py -> rounds.parquet, judges.py, judges_v2.py)
Data: Greco1899 ufc_fight_stats.csv (round-level UFCStats; downloaded 11-21 to results/ufc_decision_gap/dl/), joined to
fight results by EVENT+BOUT; fighter names resolve to f1/f2 on 100% of rows; 99.76% of decisions have all scheduled rounds.
Point-in-time: every feature (KD, sig. strikes landed/attempted, total strikes, TD, control time, sub attempts, reversals,
head strikes) is shown per round by the official UFC live stats (same FightMetric/UFC Stats feed) during the fight. Caveat:
UFCStats values are the post-event final version; live values can be corrected afterwards (not measurable here).
Model forms (antisymmetric, no intercept; validation 2022-24 after fitting 2012-21; n val 759):
| model | val log loss | val Brier | val acc |
|---|---|---|---|
| MS rounds won on sig. strikes | .5127 | .1680 | .7563 |
| M1 total sig. strike diff | .4969 | .1594 | .7734 |
| MT logistic on fight totals (9 feats) | .3726 | .1160 | .8393 |
| MR round model: P = sigma(k * sum_r (2 sigma(w.x_r) - 1)) | .3609 | .1127 | .8458 |
| MR + 10-8 dominance term (judges_v2.py) | .3609 (g fit to 0) | .1127 | .8458 |
-> MR frozen (refit 2012-2024, judges_MR_theta.npy). TEST 2025-01-01.. (465 decisions, outcome-only): log loss .319, Brier
.1025, accuracy 84.1% (MT .3346/.1082/82.2%; MS .5003/.1643/75.5%). Weights (z-units): sig landed .83, head .46, ctrl .44,
KD .40, TD .27, sub att .20, non-sig .17, misses -.05, rev .04; k 2.13. A 10-8 term adds nothing over saturating rounds.
### Free real-time source found: UFC.com live-stats JSON (2026-11-21)
ufc.com event pages embed `ufcLiveStatsApi.baseUrl = https://d29dxerjsp82wz.cloudfront.net/api/v3` and `event_fmid`
(UFC 334 = 1345). Public, no key: `event/live/{event_fmid}.json` (whole card) and `fight/live/{fight_fmid}.json`.
Contents: per-fight FightStats + RoundStats (sig/total strikes by target/position, TD, control by position, KD, sub att),
Result (method, judges' FightScores AND per-round RoundScores after the fight), and `FightNightTracking` = wall-clock
UTC timestamps of fight_open, walkouts, round_start/round_end, `fight_over`, `results`, `unofficial_winner_decision`,
fight_complete. Example UFC 334 Gane-Hokit: fight_over 05:49:24Z, results 05:52:10Z, unofficial_winner_decision
05:52:32Z (horn -> winner 188 s). Event JSON has mode "live" + LiveRoundNumber/LiveRoundElapsedTime during events.
Use here: exact historical horn and announcement times for step 1 (replaces the start+17/29-min guess). Kalshi's own
/milestones also has a Sportradar start_date, and /live_data/mma_match/milestone/{id} gives round/time-left (free).
IDs are roughly chronological (1250 = 2025-04-05, 1300 = 2026-03-14, 1344 = 2026-10-10); crawl 1230-1364 (fetch_ufclive.py,
1 req/s) -> results/ufc_decision_gap/ufclive/. CloudFront: X-Cache Miss on a past event (publication latency during a live
event is NOT measured; needs a forward poll).
### PRE-REGISTRATION D1 (written 2026-11-21 before any window price is compared with an outcome)
Question: in the window [final horn, announcement), does the frozen judges model MR (judges_MR_theta.npy, fit < 2025)
carry information about the official winner beyond the Kalshi window price, on DEV decision fights (card < 2026-07-01)?
- Window price p_K(A) at t_dec = first minute boundary after the UFC-live `fight_over` timestamp (quote age 0-60 s after
  the horn): average of mid(A) and 1 - mid(B) from 1-min candle closes (both books need bid > 0 and ask < 1; else the
  available side; else last trade at horn+30 s). Late-window price = last trades <= results - 5 s (descriptive).
- Outcome y(A) = Kalshi result of A's market (scalar/draw settles dropped). MR p_M(A) from UFCStats round stats.
- Metrics: Brier and log loss of p_K vs p_M vs outcome; paired difference with a card-clustered bootstrap (2,000,
  seed 20261121); stacking logistic y ~ a*logit(p_K) + b*logit(p_M) (no intercept), card-bootstrap CI on b.
- "MODEL ADDS INFO" = b 95% CI lower > 0 AND Brier(blend refit by leave-one-card-out) < Brier(p_K) with CI below 0.
  If not -> verdict "book already prices round stats" -> DEAD, no trading rule, no sealed request.
- If it adds info -> write a rule prereg (t_dec ask + fee ceil(0.07*C*p*(1-p)), sealed = card >= 2026-07-01, placebo =
  MS rounds-won-by-sig model and a shuffled p_M) BEFORE computing any return.
- Too-good guard: Brier(p_M) < Brier(p_K) by > 0.03 -> inspect rows (timestamps / orientation) before reporting.
### Window timing from UFC live JSON (parse_ufclive.py -> fights_timed.parquet; outcome-free timing, all 305 decisions)
UFC live crawl: 127 events (ids 1230-1364), 1,346 fights; every one of the 694 Kalshi-matched fights has a row (305/305
decisions). Horn = `fight_over`; `results` and `unofficial_winner_decision` (uw) are the next actions.
| seconds | n | p10 | p25 | median | p75 | p90 |
|---|---|---|---|---|---|---|
| horn -> results | 303 | 84 | 95.5 | 105 | 115.5 | 128 |
| horn -> unofficial winner | 299 | 98 | 110.5 | 122 | 133 | 148 |
| uw -> Kalshi market close | 299 | 208 | 288 | 422 | 570 | 953 |
So the tradeable gap is ~1.5-2 min (results ~17 s before uw = the scorecard reading), and Kalshi keeps the market open
~7 min after the winner is named (trading continues at .99/.01).
Horn-time check (fights_timed.parquet, decisions): UFC-live round-1 start -> fight_over exceeds the nominal 17:00 (3-rd)
by median 30 s (p10 22, p90 106, max 447; n 272) and 29:00 (5-rd) by median 58 s (p90 143; n 31); 29.2% of decision
fights have >= 1 round_pause. So "start + rounds x 5 + breaks" would misplace the horn by about a third of the 105-s
window on a typical fight and by more than the whole window on the p90 fight -> use the UFC-live `fight_over` stamp.
Forward logger launched (read-only, self-terminating): fwd_logger.py 1348 26NOV21 2026-11-21T14:45Z..23:30Z (PID 59793 after two restarts: 5-s idle polling, backlog on first poll never opens a hot window;
log results/ufc_decision_gap/fwd/1348.jsonl). It records when the UFC JSON first shows fight_over / results /
unofficial_winner / Result+scorecards vs their stamps, and Kalshi orderbook depth (depth 10) in the window.
Kill: pkill -f fwd_logger.py.
Horn verification (horn_check.py -> out_horn_check.txt, all 117 DEV fights with a KXUFCDISTANCE tape; an earlier
partial run on 25 fights gave +2.1 s): DIST YES reaches >= .97 for good at median -1.0 s vs the UFC-live `fight_over`
stamp (IQR -21 .. +6.3 s, p10 -49, p90 +27). DIST was .95 median 20 s before the horn.

## STEP 1 RESULT -- the book in the window (DEV, 220 decision fights; window.py -> out_window.txt, window_summary.py ->
## out_window_summary.txt, horn_check.py -> out_horn_check.txt; tapes = winner-market trades on [horn-180 s, uw+90 s])
- Timing (UFC-live stamps, DEV): horn -> results median 104 s (IQR 96.5-113.5, n 219); horn -> unofficial winner 122 s
  (IQR 111-132); winner -> Kalshi close median 483 s. Horn stamp vs KXUFCDISTANCE >= .97 signature: median -1.0 s
  (IQR -21 .. +6.3, n 117).
- Activity: 96.4% of fights trade in the window. 2026H1 (n 124): window volume median 60,618 contracts on the two
  tickers (p10 13,840, p90 166,134), median 554 trades; tighter-book spread at t_dec median .01 (p75 .02, n 83 two-sided).
  2025 (n 96): median 2,678 contracts, spread median .07 -- the book matured in 2026.
- Price level: favourite at horn+30 s median .965 (2026H1); 57.3% of 2026H1 fights already >= .95, 17.7% < .80.
- When the market learns the winner (contested fights, favourite < .90, n 59): the price jumps to >= .95 median 20.0 s after
  `results` and +0.9 s after `unofficial_winner_decision` (IQR -1.5 .. +5.7 s). The book moves on the spoken name (Buffer),
  i.e. at the same second as the UFC JSON stamp. Execution: ~100 s of window with 1c spreads and tens of thousands of
  contracts traded in 2026 -> long enough to execute; displayed DEPTH is not in history (forward logger measures it).

## D1 RESULT (compare.py -> out_d1_compare.txt, d1_dev.parquet; scored once, DEV only)
n 209 fights / 47 cards (4 draw/scalar, 5 no MR prediction, 8 no window price dropped). Kalshi result == UFCStats winner 100%.
pK source: 1-min candle quotes 139, trade fallback 70. Quote time - horn: median 29 s, max 60 s; 0 quotes at/after
`results`; results - t_dec min 21 s, median 75 s (no lookahead).
| price / model | Brier [card CI] | log loss | accuracy |
|---|---|---|---|
| pK window (horn + 0-60 s) | .0609 [.0421, .0804] | .2053 | 92.3% |
| pK last trade before horn | .0672 [.0477, .0872] | .2266 | 91.4% |
| pK late (results - 5 s) | .0631 [.0436, .0844] | .2148 | 91.9% |
| pM judges model MR | .1192 [.0930, .1476] | .3576 | 80.9% |
- Brier(pM) - Brier(pK) = +.0583 [+.0267, +.0898]; log-loss diff +.1523 [+.0734, +.2340] -> the book is much better.
- Stacking (no intercept): a(pK) 1.093 [.812, 1.666], b(pM) .387 [-.058, .863] -> b CI includes 0.
- Leave-one-card-out blend: Brier .0610; blend - pK +.0002 [-.0067, +.0073] -> no gain.
- Disagreements: pM below pK by > .3 (n 26: pK .880, pM .374) -> winner rate .885; pM above pK by > .3 (n 18: pK .171,
  pM .660) -> .167. Where they disagree most, the market is right (rule PA again).
- Contested subset pK in (.1,.9), n 59: Brier pK .1814 vs pM .2357; accuracy 76.3% vs 64.4%. Fresh-quote subset n 139:
  pK .0911 vs pM .1733. Orientation check: corr(pM, y) .724, corr(pK, pM) .796.
- Calibration of pK by bin (descriptive): [0,.1) n78 mean .031 -> .013; [.1,.3) n13 .199 -> .231; [.3,.7) n24 .463 -> .375;
  [.7,.9) n22 .812 -> .773; [.9,1] n72 .964 -> .986. Favourites >= .9 won 71 of 72 (.986) vs .964 priced -- a post-hoc
  band on 72 fights (SE ~.02); NOT a lead (STRATEGY_v2: no post-hoc bands without a fresh window; sealed has 85 fights,
  far too few to resolve a 1-2 pt edge net of a 1c spread).
**VERDICT D1: DEAD by the prereg (expected failure confirmed); critic ACCEPT-WITH-CHANGES (2026-11-21).** Wording per critic: the
model's added information is NOT DISTINGUISHABLE from zero (b positive .35-.43 in every variant, CI just crosses 0; the
out-of-sample blend cannot rule out a gain up to .0067 Brier, ~11% of pK's Brier). Scope: this tests ONE round-stats judges
model (MR), not every scorecard signal. Critic recomputed all numbers (match; +.0583 is the observed diff, resample mean
.0589) and passed lookahead, orientation (209/209), out-of-sample, stacking variants, exclusions and split. Minor: the
printed MR test score (judges.py --test) included 87 UFCStats decisions dated >= 2026-07-01 (outcome-only, no Kalshi data). The window book already prices the round stats -- and much more:
the crowd watching the broadcast (damage, momentum, commentary, judges' habits) is 92% accurate at the horn vs 81% for
the stats model, and the stats add nothing distinguishable in a stacked blend. No trading rule, no sealed request, no edgelab trial
(D1 was a calibration test, no rule was run). Sealed (card >= 2026-07-01) untouched.

## What is left (forward only)
The one untested angle is latency, not modelling: does the UFC JSON show the winner (Result.Method / FightScores /
Outcome) at its `results` stamp, ~17-20 s before the name is read and the book jumps? History cannot answer it (past
JSON is the final version). fwd_logger.py (launched for UFC Fight Night 2026-11-21, fmid 1348) logs first-seen times of
every tracking action and of the Result block, plus Kalshi orderbooks in each decision window. Kill rule for the angle:
if on >= 5 decision fights the JSON Result/winner first appears later than (Kalshi jump - 3 s), the latency angle is dead.
If it leads by >= 10 s on most fights, write a prereg (forward paper, 1-lot, ask + fee) -- do not trade it.

## Learnings
### UFC.com exposes a free live-stats JSON with wall-clock fight timestamps
`https://d29dxerjsp82wz.cloudfront.net/api/v3/event/live/{event_fmid}.json` (event_fmid sits in the ufc.com event page as
`eventLiveStats.event_fmid`) holds per-round stats, judges' round scores after the fight and `FightNightTracking`
stamps (round_start/end, fight_over, results, unofficial_winner_*). 127 events crawled (ids 1230-1364); every Kalshi-matched
fight had a row. Source: results/ufc_decision_gap/fetch_ufclive.py, parse_ufclive.py. Any UFC in-fight study should start here.
### The UFC decision window is about 105 seconds and Kalshi stays open through it
Horn -> results median 104 s (IQR 96.5-113.5), horn -> winner named 122 s (n 219 DEV); the market closes ~8 min after the
name. 2026 windows trade a median 60,618 contracts at 1c spreads. Source: out_window.txt, out_window_summary.txt.
### Nominal fight length misplaces the horn by a third of the window
Round-1 start + 17:00 under-states the horn by median 30 s (p90 106 s); 29% of decisions have a pause. Use the UFC-live
fight_over stamp (agrees with the Kalshi DISTANCE market to median -1 s, n 117). Source: fights_timed.parquet, out_horn_check.txt.
### The Kalshi window price beats a round-stats judges model by a wide margin
Brier .0609 (92.3% correct) vs .1192 (80.9%) on 209 DEV fights, diff +.0583 [+.0267,+.0898]; stacking coef on the model
.387 [-.058,.863]; blend gains +.0002 [-.0067,+.0073]. The broadcast crowd sees more than the stat sheet. Source:
out_d1_compare.txt.
### Kalshi reprices on the spoken winner, within a second of the UFC stamp
On contested decisions the price reaches .95 median +0.9 s after the UFC JSON `unofficial_winner_decision` stamp and +20 s
after `results`. A JSON-polling bot only helps if the JSON publishes the Result before the name is read (forward test).
Source: out_window_summary.txt; fwd_logger.py.
### An out-of-time judges model from UFCStats round stats is ~84% accurate
Round model P = sigma(k * sum_r(2 sigma(w.x_r) - 1)): 2025-26 test log loss .319, Brier .1025, acc 84.1% (n 465); a 10-8
dominance term adds nothing. Sig strikes, head strikes, control and knockdowns carry most weight. Source: judges.py.

## Forward result (scored 2026-11-23 by the lead; results/ufc_decision_gap/fwd_score.py -> out_fwd_score.txt) -- LATENCY ANGLE DEAD
UFC FN 11-21 (fmid 1348): 8 decision fights. Per fight: seconds from the spoken-winner stamp (unofficial_winner_decision)
to (a) Kalshi winner best bid >= .95 and (b) our first poll showing the JSON Result block (Method + FightScores):
13200 jump +5.0 / JSON +23.3; 13195 -1.8 / +70.6; 13198 +13.3 / +13.3; 13193 +61.6 / +59.9 (lead 1.7 s < 3 s);
13194, 13192, 13199: book >= .95 100-133 s BEFORE the name (lopsided fights); 13262: no .95 crossing logged, JSON +20.3 s.
Pre-set kill (JSON not earlier than Kalshi jump - 3 s on >= 5 decisions): met on 7 of 7 fights with a measurable jump -> DEAD.
Caveat: the endpoint returned non-JSON (empty body) 2,663 times, so first-seen is an upper bound on 7 of 8 fights; the one
error-free fight (13262) still shows the Result block 20.3 s AFTER the name was read. The JSON's own unofficial_winner action
appeared 5-10 s after its stamp on clean polls. The UFC JSON trails the broadcast; Kalshi follows the broadcast.
### The UFC live JSON publishes the decision about 20 s after the announcer reads it
On the clean-polling fight the Result block appeared +20.3 s after the unofficial_winner_decision stamp; Kalshi reprices
within ~1-5 s of the name. A JSON-polling bot is a loser here. Source: results/ufc_decision_gap/out_fwd_score.txt.
