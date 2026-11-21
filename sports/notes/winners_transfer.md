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
