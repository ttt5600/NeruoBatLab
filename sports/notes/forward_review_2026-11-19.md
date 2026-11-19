# forward_review_1119 -- every forward paper trial + live strategy vs ITS OWN pre-registered rule (2026-11-19 ~06Z)

Review subagent for the quant-trader lead. Read-only (Kalshi public/authenticated GETs, log reads). No orders, no git,
no .env / launchd / kill-file changes. Scratch + outputs: `sports/results/forward_review_1119/`.

## Relation to the 11-15 review
`notes/forward_review_2026-11-15.md` stopped after its plan ("16:08Z started"); it produced no table, so there is no
11-15 verdict to diff against. Baseline used instead: STATE.md (2026-11-19 handoff) + the scorecards snapshot
`notes/scorecards.md` generated 2026-11-19T04:43Z. "Changed" below = differs from what STATE/that snapshot say.

## Judging (no new rules)
- Verdict words: KILL (the written kill condition is met), PROMOTE-ELIGIBLE (the written success condition is met),
  CONTINUE (neither; includes too-early), BROKEN (logger not writing, or a data problem that invalidates counting).
- Rules quoted from each trial's own notes / registry line (scorecards.py cites file:line at run time).
- Scorecards ROI CIs bootstrap over BETS. Where I re-computed, I clustered by game date (opener trials) or event (Q014,
  the logger's own event-clustered bootstrap). Settled only; voids excluded (none found in these trials).
- Sports opener trials: the pre-registered CLV is vs the BOOK close (K1 / K-CFB / K50: Pinnacle/DK close vs Kalshi strike S)
  or vs Kalshi's own rung close (NHL-om-A CLV_c). The verdict uses the pre-registered metric. I ALSO report CLV vs
  Kalshi's own later price (same-side close mid - entry mid, from the loggers' `close_mid`) as a descriptive check,
  added after I had seen the scorecards -- not a verdict input. Script: results/forward_review_1119/opener_clv.py.

## Commands run (all outputs in results/forward_review_1119/)
- `scripts/scorecards.py --kalshi --out results/forward_review_1119/scorecards.md` (exchange cache 2026-11-19T05:40:52Z)
- `python -m edgelab status` -> edgelab_status.txt (registry md5 unchanged before/after: 9357016c...)
- deployed `q013_logger.py report` -> q013_report.txt; `q014_logger.py report` -> q014_report.txt
- q014_count_check.py (kill-rule precondition: RC count vs Kalshi expiration_value) -> q014_count_check.txt
- opener_clv.py -> opener_clv.txt; fresh.py (rows per day per logger file) for output-growth checks.

## Table

| Trial | n settled | Metric + 95% CI | Pre-reg criterion (source) | Verdict | Source |
|---|---|---|---|---|---|
| E4 orig7 live maker | 32 fills (1 loss) | losses 1 vs break-even 1.73; SPRT LLR +0.24 (A +5.19 / B -2.30) | edgelab preset kill: >=5 losses in first 50, or CLV<0 sig after 100 (registry line 982) | CONTINUE | scorecards.md SPRT panel; kalshi_exec.jsonl |
| X1 live (restart 11-18) | 0 (3 filled 11-19, 3 resting) | n/a | kill: 2 losses in first 30 fills or ROI<0 after 60 (SIZING_RULE.md:60) | CONTINUE (first fills tonight) | kalshi_exec.jsonl cohort x1 |
| X1 paper since pause | 53 tt-fills | 3 losses vs b/e 3.79; LLR +0.54 | descriptive (X1 failed dev) | n/a | scorecards.md SPRT panel |
| lead_exec DUL (B2) | 3 | 0-3, ROI -100%; CLV -0.0067 [-0.0132,-0.0001] | after 40 bets ROI<0 or CLV<0 (ACT_LIST.md:21) | CONTINUE (too early; 0-3) | lead_exec.jsonl |
| lead_exec TSA (B4) | 3 fills | 3-0, ROI +4.2% | 2 losses in first 30 fills (ACT_LIST.md:23) | CONTINUE | lead_exec.jsonl |
| lead_exec NHL late (B5) | 19 | 7 losses; ROI -15.3% [-46.3,+14.6] not distinguishable; CLV -0.028 [-0.100,+0.044] | 10 losses in first 30 (ACT_LIST.md:85) | CONTINUE (3 more losses in next 11 kills) | lead_exec.jsonl |
| K1 (NBA totals) | 20 (7 dates) | pre-reg CLV vs S +0.78 pts, date-clust [+0.04,+1.17]; ROI +18.7% [-9.6,+48.8] n.d.; Kalshi-own CLV -0.4c [-3.0,+1.7] n.d.; net of ask+fee -3.3c [-5.8,-1.2] | judge at 100: ROI>0 p<.10 AND CLV vs S>0; kill ROI CI up<0 after 60 or CLV<=0 after 100 (nba_l32b_exec.md:95) | CONTINUE | kx_forward/k1.jsonl; opener_clv.txt |
| K-NFL | 0 (5 pending) | n/a | same as K1 (kalshi_openers_xleague.md:47) | CONTINUE | kx_forward/knfl.jsonl |
| K50-NFL | 9 (1 date) | CLV vs close +0.50 pts (1 cluster, no CI); ROI +25.6% | same as K1 (k50_strike.md:105) | CONTINUE | kx_forward/k50nfl.jsonl |
| K-CFB | 11 (3 dates) | CLV vs DK close +0.73 pts [+0.5,+2.0] (3 clusters); ROI -40.4%; Kalshi-own CLV -1.0c [-3.4,+7.5] n.d. | same as K1 (kalshi_openers_xleague.md:47) | CONTINUE | kx_forward/kcfb.jsonl |
| NHL-om-A | 5 | CLV_c +1.8c; ROI -64% | judge at 100; kill ROI CI up<0 after 60 or CLV_c<=0 after 100 (nhl_opener_model.md:129) | CONTINUE | kx_forward/nhla.jsonl |
| nhl-goalie | logger only | 236 events rows, writing 11-18 | test not before 2026-12-06 (ops_nhl_goalie.md) | CONTINUE (logging) | data/nhl_goalie/events.jsonl |
| OT1 / OT1b | 2 / 4 | ROI +17.2% / +11.6% (n tiny) | kill ROI CI <0 after 150 (ops_ot1_forward.md:30, ot1b_monotone.md:30) | CONTINUE | kx_forward/ot1.jsonl |
| CB1 / CB2 / CB3 | 0 (0 picks in 3 days) | n/a | no verdict before 60 bets & 12 months (crypto_barrier*.md) | CONTINUE (quotes 71/day logged) | kx_forward/crypto_barrier*.jsonl |
| E4P-C / E4P-CA | 0 fills (5 bids, 1 settled unfilled, 67 city-days) | SPRT 0.00 | review at 60 fills (ops_e4p_forward.md:55-60) | CONTINUE (very slow: 5 bids/67 city-days) | kx_forward/e4p.jsonl |
| E4-0600 | 1 tt fill | 0 losses; LLR +0.01 | review at 60 fills; SPRT B (ops_e4_0600.md:37-43) | CONTINUE | kx_forward/e4_0600.jsonl |
| ESP1 esports | 0 (logger live 4.3 h) | 0 bets; matched <=1/tick | kill(c) <5 matched/day after 14 days; score n 300 or day 28 (esports_pinnacle.md:30-33) | CONTINUE (too early) | data/esports_pin/fetch.jsonl |
| QM1 / QM2 | 0 | 0 picks; 10 maker posts, 0 filled | kill: first 6 months ROI<=0 on both (quake_monthly.md:43) | CONTINUE | kx_forward/quake_monthly.jsonl |
| TG1 / SD1 / SD2 | 0 / 0 / 0 | 1 / 5 / 0 fires pending (SD2 starts Dec 2) | observation; SD1 kill units<=0 at Apr-2027 settle (snow_markets.md:49) | CONTINUE | results/snow_markets/snow_log.jsonl |
| state-gas books | no trial (GS1 not frozen) | 500 rows/day 11-15..11-18 | n/a (GS1 registers at FIT freeze) | n/a (logger OK) | kx_forward/state_gas_books.jsonl.gz |
| econ-events books | no trial (EV1/EV2 unregistered) | 1,182-1,272 book rows/day 11-17/18; 1 MISSED slot | register only if >=5 eligible quotes/release (econ_event_ladders.md:114) | n/a (logger OK) | kx_forward/econ_events/*.jsonl.gz |
| LX1 logical arb | 22 violation rows, 1 distinct leg set | 1 distinct vs bar 5 in 7 days (window 11-12..11-19) | kill: <5 distinct in 7 days (lx1_logical_arb.md:18) | **KILL** (final at 11-19 17:36Z unless 4 new distinct sets appear) | results/lx1_logical_arb |
| Q013-T24 / T6 (RT) | 1 / 2 events | T24 ROI +181% (1 bet), T6 -3.7% (2 bets) | pass/kill at 30 events (q013_rt_scores.md:60-65) | CONTINUE -- was BROKEN 11-03..11-19 (170 rows "no score"), fixed 11-19 03:38Z; next picks 11-22 | q013_report.txt |
| Q014-D (Truth Social daily) | 81 bets / 12 events | ROI -18.7% event-clust [-24.0,-13.2]; Brier model .138 vs mid .081; CLV -0.143 | kill after 40 events: ROI<=0 or Brier>=mid (q014_truth_social.md:59) | CONTINUE (on track to KILL; 12/40 events) | q014_report.txt |
| Q014-W | 39 bets / 1 event | ROI -1.5%; Brier .146 vs mid .091 | kill after 12 weekly events | CONTINUE | q014_report.txt |

Already KILLED-by-rule per scorecards (not new; in STATE): L34 (CLV -0.0004 at n 188), Q004-NHL (ROI -17.2% [-34.1,-0.4] at n 156).
None of the trials meets its success criterion -> no PROMOTE-ELIGIBLE. No trial is BROKEN today (every logger wrote rows
11-18/11-19 -- fresh.py); Q013 was broken until 11-19 03:38Z.

## Key observations
- K1: the pre-registered CLV (book close vs S) is positive with a date-clustered CI above 0, but Kalshi's OWN close mid
  barely moved toward us (-0.4c [-3.0,+1.7], not distinguishable from 0) and after the ask + fee we are -3.3c
  [-5.8,-1.2] at the close. The book line moving through S does not show up as a Kalshi price gain we capture.
  Same pattern in K-CFB (net -9.3c) and NHL-om-A (Kalshi-own +1.8c but net -1.2c [-2.0,+1.8]). Descriptive only; the
  verdict stays on the written rule. Mechanism: the 100-ct taker fee (1.75c at 50c) + half-spread (~1-1.5c) is the
  hurdle; the book's close moving 0.5-0.8 pts on an NBA total is worth only ~1-2c on a 50c rung.
- Q014 count semantics are fine: RC count at window end = Kalshi expiration_value on 13/13 settled events
  (q014_count_check.txt), so the kill-rule precondition holds; the model is simply miscalibrated (bin 0.0-0.1:
  mean_p .006, freq_yes .15).
- Edgelab cannot formally score almost any of these: `edgelab status` shows K1/K-NFL/K50/K-CFB/NHL-om-A/CB*/QM*/
  E4P-*_fwd/E4-0600/ESP1/TG1/SD*/nhl_ig_t1 as "registered" with "no dev/sealed window: user must pre-register a sealed
  forward window in vault.yaml" (or no vault entry). Only e4_maker_e1 has a forward evaluation (n 18 on 11-10, stale).
- Ops: deployed qs_job.sh had a syntax error at ~01:30Z (launchd-nhl-ingame.err, line 236, esports-pin case); `bash -n`
  passes now (file rewritten 11-18 19:58 PT), jobs writing again. Health "STALE state-gas-books" is a threshold artefact:
  the two daily slots (14:04Z am, ~07:28Z close) are ~17.4 h apart vs the 14 h max.

## Needs the lead / user
1. LX1: confirm the kill after its window closes (11-19 17:36Z); the scanner self-stops 11-20. Lead retires it in
   edgelab/EDGES.
2. Edgelab forward wiring (user: vault.yaml windows) for the forward trials, or accept that they are judged by the
   scorecards only. Lead: run a fresh `edgelab evaluate e4_maker_e1 --split forward` (last one 11-10, n 18; exchange now 32 fills).
3. Q013: decide on the proposed point-in-time backfill of the 170 skipped rows (q013_rt_scores.md) -- touches trial data.
4. Health monitor: state-gas-books max-age 14 h -> ~18 h (false STALE).
5. Watch: B5 NHL late (7/19 losses, limit 10/30); DUL 0-3; Q014-D heading for its 40-event kill.

## Learnings
### The 11-15 forward review never produced a table
notes/forward_review_2026-11-15.md ends at its plan; a review agent that waits hours for settles can die silently. Write
the table incrementally (one row per trial as scored), not at the end.
### Book-close CLV can be positive while Kalshi's own price does not follow
K1 n 20: CLV vs S (Pinnacle close) +0.78 pts [+0.04,+1.17] but same-side Kalshi close mid - entry mid -0.4c [-3.0,+1.7]
and net of ask+fee -3.3c [-5.8,-1.2] (results/forward_review_1119/opener_clv.txt). For a taker on Kalshi, judge with
Kalshi's own close too.
### Q014's count matches Kalshi's settlement; the losses are the model
13/13 settled events agree (q014_count_check.txt); D ROI -18.7% [-24.0,-13.2] at 12 events, Brier .138 vs mid .081.
### A freshness limit must exceed the longest gap between scheduled slots
state-gas books: slots ~17.4 h apart vs a 14 h freshness limit -> a permanent false STALE.
