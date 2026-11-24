# e4_coldmargin -- does narrowing E4's COLD margin 3F -> 1F add money on a fresh window? (agent e4_coldmargin, started 2026-11-24 ~03:05Z)

Research agent for the fund lead. AGENT_PROTOCOL: own notes, inbox rows as I go, NO git, NO orders, NO executor/.env/launchd/
kill-file changes, NO deploys to ~/.claude/quant-kalshi. Sealed stays sealed (E1 sealed 2025-07-01..2026-10-06, X1 sealed
2026-03-01..2026-10-18). Scratch: results/e4_coldmargin/. Script: scripts/e4_coldmargin.py.

## Question (from lead)
e4_data_v2 (notes/e4_data_v2.md, "asymmetric margins" grid) found IN-SAMPLE on W1 (taker, 16-cell grid = forking paths) that
keeping the warm margin at 3F and narrowing the cold margin (brackets BELOW the coldest model) to 1F adds 816 bets / 14 losses /
+21.3u (Mw3,Mc1 +51.80 [+31.3,+71.7] vs E4 +30.46). DGP support: P(CLI < coldest model - 1F) = 2.6% (vs 23.7% warm side).
Test it once, cleanly, on a window the grid never saw.

## Frozen live E4 rule (read from ~/.claude/quant-kalshi/scripts/kalshi_paper.py + kalshi_exec.py, 2026-11-23)
- Members: gfs_seamless, ecmwf_ifs025, icon_seamless, gem_seamless, ncep_nbm_conus; each = max of hourly 2m T over hours 0..18
  local of D, needs >= 15 hourly values; >= 3 members available.
- rejected(): greater: hi + 3 < floor + 1; less: lo - 3 > cap - 1; between: hi + 3 < floor or lo - 3 > cap.
- Tradeable: yes_bid > 0, yes_ask < 1, yes_ask - yes_bid <= .10, yes_ask > .01. Decision 22:00-23:59 city-local D-1.
- Execution (kalshi_exec._decide_maker): post-only GTC NO bid at q = min(cent_floor(NO mid), NO ask - 1c), expiration = close,
  skip if NO ask - q > 5c, maker fee 0 on fee_type quadratic.
- Backtest definition of the members (fcsafe, wx_timing.fc_at(h, 22)): Open-Meteo previous-runs temperature_2m_previous_day1,
  hours <= 18 local, >= 15 values. Live uses the live /v1/forecast endpoint (fresher runs). The W1 grid used fcsafe.

## Do-not-redo
FUND_BOOK 3d; e4_data_v2 T1 (MOS voters), T2 (MOS consensus), T3 (EMOS NEW) and X2 lows are dead; not re-tested.

## In-sample reference (W1, already used; recomputed 2026-11-23 from results/e4_data_v2/frame_all.parquet for the power check)
Extra band (Mc1 & not E4): 816 taker bets, 14 losses, +21.34u (per bet +.0261, sd .153), mean NO ask .9566, E1 671 / X1 145,
over 2,900 city-nights (0.28 extra/city-night). Maker TT on the 74.9% with tapes: fill rate 57.8%, 353 fills, 10 losses, +14.27u
(+.040/fill). No extra pick has ask - q > 5c.

## PRE-REGISTRATION (written 2026-11-24 by 03:10:50Z [file creation; first draft said '11-23 ~19:30Z' = a local-time misread, corrected per critic] BEFORE any forward-window market, quote, tape or forecast was pulled, and before
## any outcome of a cold-band bracket in the window was seen)
### Window F (fresh holdout, retrospective)
- orig-7 (wx.CITIES): event dates 2026-10-07 .. 2026-11-22 (E1 sealed ends 10-06; 10-07..10-21 is in no edgelab split,
  10-22+ is the E1 forward split).
- X1 17 cities (kalshi_paper.X1_CITIES): event dates 2026-10-19 .. 2026-11-22 (X1 sealed ends 10-18).
- No date inside any sealed window is requested. The window was never used by the W1 grid or by any margin choice.
- Honest status: RETROSPECTIVE holdout, not a prospective forward test (the data exist before this variant is frozen). Seen
  before this prereg: the live E4 aggregate P&L in this window (E4's own picks: 137 settled / 134 wins, `kalshi_exec.py status`),
  not any outcome of a cold-band (1-3F below coldest) bracket. Other sessions may have seen E4P-C (e4p) forward outcomes since
  11-17, which partly overlap the cold band; I have not.
- Rows: every market of those events with result in {yes, no} at pull time; brackets whose result contradicts the expiration
  value (result_mismatch) are dropped; unsettled markets are dropped and counted.
### Inputs (PRIMARY = the frozen backtest definition, which is what the W1 grid used)
- Members: fcsafe as above (previous-runs previous_day1, hours 0..18 local, >= 15 values), all 5 models, >= 3 available.
- Quote: last hourly candle ending at or before 22:00 local D-1 (wx._row: yes_bid/yes_ask close), tradeable filter as live.
- T = 22:00 local D-1 (epoch via zoneinfo, checked with traps.check_epoch_seconds); close_ts from the market close_time.
### Selection (single change)
- E4 = rejected at Mw = Mc = 3 (as live). Mc1 = warm margin 3, cold margin 1:
  greater: hi + 3 < floor + 1; less: lo - 1 > cap - 1; between: hi + 3 < floor or lo - 1 > cap.
- EXTRA = Mc1 & tradeable & NOT E4 (= brackets whose top edge is 1-3F below the coldest member).
### Primary statistic
S = sum over EXTRA of maker trade-through units: q = round(min(cent_floor(1 - (yes_bid + yes_ask)/2), (1 - yes_bid) - .01), 4);
skip if (1 - yes_bid) - q > .05 or q <= 0; FILLED iff any public print with no_price < q (strict) in (T, close_ts]; filled pnl =
1/q - 1 if NO wins else -1 (1 unit of cost per bet, maker fee 0); unfilled = 0. Missing tape (fetch failed): loser counted
filled, winner unfilled. Tape coverage must be >= 97% of EXTRA, else TAKER is the primary (decided now).
Inference: per-date sums over ALL event dates in window F (dates with no extra pick add 0); one-sided bootstrap over dates
(every city on a date resamples together), 4000 draws, seed 0; p = P(S* <= 0); 95% percentile CI. Halves split at the median
date of the window's date list.
### Secondary (reported, not part of the verdict unless stated)
- Taker: cost = (1 - yes_bid) + .07 p(1 - p), every EXTRA pick.
- Maker N10 (edgelab e4_maker spec): >= 10 contracts printed at no_price <= q in (T, close].
- E4's own maker TT units on window F (reference); the Mc1 union.
- By cohort (orig7 / X1), loss rate + Wilson CI, break-even loss rate, mean NO price, fills / picks.
- U_live (as-issued LIVE forecasts): city-nights whose live forecasts are archived = kalshi_paper.jsonl pick rows (only
  city-nights with >= 1 live E4 pick, 10-16..11-16) + e4p.jsonl city_day rows (all city-nights from event day 11-17). Same
  selection on the live forecasts, same quote/fill model. Reported with CI; it must have the same sign as the primary for a PASS
  to be called "robust to the forecast source" (not a pass/fail gate).
- Reconstruction check: fcsafe E4 picks vs the live E4 pick log on window F (overlap), fcsafe vs live member values.
### Placebos (fixed now)
- P1 price-matched random: per (date, NO-ask bucket [<.90, .90-.95, .95-.97, .97-.98, >=.98]) draw the same number of brackets
  as EXTRA has in that cell, without replacement, from tradeable brackets on that date in that bucket that are neither E4 nor
  EXTRA (if a cell's pool is short, take all of it; shortfall reported). 2000 draws, seed 0, maker TT units (taker if pool tape
  coverage < 97%, then compared with the actual's taker S).
- P2 wrong-day forecasts: the same cold-band selection with the fcsafe members of D-7 (same city; issued a week earlier, wrong
  weather, same season), excluding the real E4 picks; maker TT units.
- P3 warm mirror (DESCRIPTIVE only): tradeable brackets 1-3F ABOVE the warmest member (Mw1 & not E4). Correction to the brief:
  the W1 grid does NOT say this band loses (Mw2,Mc3 +40.25 vs E4 +30.46 -> the warm 2-3F band added +9.79u taker with 113
  losses), so it is not a must-lose placebo; reported only.
### Verdict rules (fixed now)
- PASS: S > 0 AND p < .05 AND both halves > 0 AND S > S(P2) AND S > 90th percentile of P1.
  -> forward PAPER arm + propose a sealed/forward judge; no money change from this test alone.
- LEAD: S > 0 and any PASS condition fails -> forward PAPER arm (logs only) to accumulate n; no money.
- KILL: S <= 0 -> no paper arm; the cold band is not added.
- Additionally reported: is the in-sample per-fill maker ROI (+.040) inside the forward 95% CI of per-fill ROI?
### Power (stated before data)
Expected window F size at the W1 rate (0.28 extra / city-night): orig7 47 d x 7 + X1 35 d x 17 = 924 city-nights -> ~260 extra
picks, ~150 maker fills. If the in-sample effect were true (+.040/fill): S ~ +6u, sd ~ 2.1u -> z ~ 2.9 (power ~.9 at one-sided
.05). At half the in-sample effect: z ~ 1.4 (power ~.45). So a LEAD (p >= .05) is weak evidence against a real but shrunk
effect; a KILL (S <= 0) has probability ~.08 under the half effect and ~.5 under no effect.
### W2 (2021-08..2024-02) -- can any honest variant use it?
E4 cannot fire there (< 3 point-in-time Open-Meteo members before 2024-03: only GFS previous-runs). The single change "cold
margin 3 -> 1 on the 5 frozen members" therefore cannot be isolated on W2. A member-substituted variant (GFS + MAV + MET MOS) is a
different trial: MOS max-T is near-unbiased for the CLI (+0.1F) while the raw members are +1.35F cold, so "1-3F below the coldest
member" means a different band. Not run here.
### edgelab
Trial registered BEFORE any window-F outcome is computed (family kalshi_weather_temp): see log.
</content>
</invoke>

## Log
### 2026-11-24 03:11:01Z edgelab registration (after the prereg, before any window-F pull; lab/edgelab_registry.jsonl line 1431; first draft said 11-23 19:35Z = local-time misread)
Registered e4c_cold1_maker_fwd-c7e0b93e (spec edgelab/specs/e4c_cold1_maker_fwd.yaml; family kalshi_weather_temp -> 77 trials,
bar 6.49e-4). Forward-only spec (rule forward:unwired, dataset kalshi_temp_e4c_fwd has no vault entry yet = user's call).
### 2026-11-24 03:10-03:30Z data pulls (no outcome looked at)
- Clock note (corrected): everything ran on 11-24 UTC; prereg file 03:10:50Z, registration 03:11:01Z, scorer code frozen by
  03:17Z, markets.parquet (first outcome-bearing file) written 03:33:39Z. Pulls ran inside the E4/e4p decision window (03-07Z). Kalshi GETs go through kx_throttle PLUS a 0.7 s pause per request (~1 req/s) so the shared slots stay free for e4p.
- fcsafe members (results/e4_coldmargin/fc.parquet): all 5 models present on 100% of station-days for every series, D-7..11-22.
  D-7 placebo needs member values for 7 days before each cohort's window start (orig7 09-30..10-06, X1 10-12..10-18 = sealed
  CALENDAR dates): forecast values only, no market, quote or outcome row of a sealed date is requested.
- Live archive (results/e4_coldmargin/live_fc.parquet): 479 city-nights with as-issued live members (kalshi_paper 409 = nights
  with >= 1 live E4 pick, e4p 70 = all nights from 11-17 in this window's range); 83 nights in both sources agree to <= 0.7F
  (gfs 0.4, icon 0.7, others 0.0: fetched minutes apart). 482 live E4 pick tickers.
- Paper arm drafted early (while pulls ran): scripts/e4c_cold_paper.py + tests/test_e4c_cold_paper.py (9 passed). Not deployed.
### 2026-11-24 03:35-03:55Z markets, quotes, frame (selections only; no outcome read)
- markets.parquet: 5,544 markets on 924 events (orig7 7 x 47 + X1 17 x 35), 0 events missing; 5,538 finalized, 6 closed
  without result (KXHIGHTSAN-26NOV17, dropped as unsettled); result_mismatch 3 (dropped).
- quotes.parquet: 22:00-candle quote for 5,544/5,544 (candle age median 0 s, max 3 h); tradeable 4,443.
- frame.parquet (scripts/e4_coldmargin.py frame): fcsafe E4 611 (orig7 202, X1 409); Mc1 828; EXTRA 217 (orig7 66, X1 151;
  166 'less' tails + 51 between; NO ask mean .970, median .99 vs W1 extra mean .957); P2 wrong-day 306; P3 warm band 810;
  P1 pool 1,472; U_live: 446 city-nights, live E4 461, live EXTRA 92.
- Reconstruction check (fcsafe vs live, 446 live-covered nights): member fcsafe - live mean/MAE: GFS -0.48/1.18F, ECMWF
  0.00/0.10, ICON 0.06/0.30, GEM 0.10/0.40, NBM 0.01/0.30; coldest member -0.19/0.54F. Live E4 tickers in window 456: 407
  are fcsafe-E4 picks, 1 is a fcsafe-EXTRA pick. On live nights fcsafe E4 443 vs live-recomputed E4 461 (411 shared); EXTRA
  77 vs 92 (70 shared). The backtest definition reproduces the live selection closely.
- Tapes: 2,886 tickers in priority order (EXTRA, E4, P2, P1 pool, live, warm band), ~1.6/s.
### 2026-11-24 04:15Z smoke test of the scorer + DISCLOSURE
Ran cmd_score on a copy of frame.parquet with `result` replaced by random draws (fake outcomes) to check the code paths
before the one real scoring (scratch /tmp, deleted). Mistake: the fake-loser printout still carried the REAL
`expiration_value` column, so I saw the real CLI value for 21 randomly chosen EXTRA brackets (all 21 settled NO = wins).
Nothing in the prereg, code, windows or rules was changed after this (all were frozen before). Lesson in Learnings.

## PRE-REGISTERED RESULT (2026-11-24 04:40Z; scripts/e4_coldmargin.py score -> results/e4_coldmargin/score.json,
## score_stdout.txt, extra_losers.csv, scored_rows.parquet). Tapes: 2,886 tickers, coverage 100% of every scored set.
Window F: 47 event dates (halves split 2026-10-30), 924 city-nights, 5,535 scored markets.
| set (maker trade-through unless noted) | picks | fills | losses | units [95% date-cluster CI] | one-sided p(S<=0) | halves |
|---|---|---|---|---|---|---|
| **EXTRA (primary)** | 217 | 77 | 2 | **+4.07 [-0.002, +8.22]** | .025 | +2.40 / +1.68 |
| EXTRA orig7 | 66 | 36 | 0 | +4.11 [+1.95, +7.12] | <.001 | +1.75 / +2.36 |
| EXTRA X1 | 151 | 41 | 2 | -0.03 [-3.24, +2.37] | .48 | +0.65 / -0.68 |
| EXTRA taker (secondary) | 217 bets | | 2 | +4.52 [+0.60, +8.35] | | |
| EXTRA maker N10 (secondary) | | 135 | | +5.38 | | |
| E4 reference (fcsafe, same model) | 611 | 288 | 16 | +4.38 [-8.07, +15.21] | .22 | +3.81 / +0.57 |
| P2 wrong-day (D-7 members) | 306 | 240 | 82 | -28.97 [-50.77, -7.31] | | |
| P3 warm band 1-3F above warmest (descr.) | 810 | 623 | 136 | -33.00 [-68.44, +0.34] | | |
| U_live EXTRA (as-issued live members) | 92 | 30 | 1 | +1.54 [-1.47, +4.32] | | |
P1 price-matched random (2,000 draws, maker TT, 92 cells, shortfall 1 pick): mean +1.53u, p90 +5.62, p95 +7.02; actual +4.07 at
the 78th percentile (taker version: mean +1.59, p90 +5.60, actual +4.52 at 83rd).
Per-fill ROI +.053 (CI contains the in-sample +.040).
Conditions: S>0 yes; p<.05 yes (.025); halves>0 yes; S>P2 yes; S>P1 p90 NO -> **pre-registered verdict LEAD** (forward paper arm,
no money). Note the two-sided 95% CI lower bound is -0.002: by the fund's wording rule the effect is "not distinguishable from 0"
at 95% two-sided, though the pre-registered one-sided test is p .025.
### Loser rows (both X1; extra_losers.csv)
- KXHIGHTHOU-26NOV18-T69 (less, YES iff < 69): coldest member 70.6 (NBM), CLI 68 -> 2.6F below the coldest; NO .98 maker filled.
- KXHIGHTSATX-26NOV07-T70: members 70.1-73.3, CLI 69; the market had YES at .63/.66 at 22:00 (the market strongly disagreed
  with all 5 members) -> NO bid q .35 filled, -1u.
### Where the units came from (descriptive, post hoc)
By NO ask: <.90 11 picks / 11 fills / 1 loss / +2.34u; .90-.95 +0.84; .95-.97 +0.65; .97-.98 +0.33; .98-.99 +0.37; >=.99 145
picks / 27 fills / 1 loss / -0.47u. Top 5 fills = 60% of S (largest: KXHIGHDEN-26NOV14-T50 at q .44, +1.27u). So in window F
the money came from a handful of picks where the market priced YES at 10-55% against all 5 members; the 145 picks at a 99c
NO ask (2/3 of the band) lost a little under maker (one 98c loser wipes ~50 wins). In-sample W1 (taped maker) the band's money was
spread over all buckets (>=.99 bucket +3.04u on 138 fills) -> the bucket profile is not stable; no bucket filter is proposed
(it would be a post-hoc cut).
- Quote check: the 22 EXTRA picks also in e4p's live log have candle yes_bid/yes_ask identical to the logged live quote.
- Realized CLI minus coldest member on EXTRA rows: 5th pct -0.42F, median +2.8F (min -2.6F).

## PROPOSAL (lead / user): forward PAPER arm E4C (no orders, no money) -- verdict LEAD
Built, tested, NOT installed:
- scripts/e4c_cold_paper.py: reads e4p's decision rows (kx_forward/e4p.jsonl: live members + live quote of every tradeable
  bracket at 22:00-22:50 station-local D-1), selects the EXTRA cold-band picks (frozen formula, warm 3F / cold 1F, minus E4 own
  and live E4 picks), logs `pick` rows with the hypothetical maker q; `settle` rows once the market settles (result, trade-through
  fill after decided_at, n10, CLV close_no, e4_live_final). Output ~/.claude/quant-sports/data/kx_forward/e4c_cold.jsonl.
  FORWARD_FROM = 2026-11-25 (first event day whose decisions ALL follow the registration at 11-24 03:11Z; the 11-24 ET decisions
  ran at ~03:00-03:50Z, partly before it, so 11-24 is excluded. e4p rows are written at decision time, so a later deploy still
  starts from 11-25).
  Dry run on the real e4p log (scratch results/e4_coldmargin/dryrun_e4c_cold.jsonl, run with the earlier 11-24 start): 9
  city-days, 2 picks for 11-24 -- the file format works.
- tests/test_e4c_cold_paper.py: 11 tests (frozen formula reproduced at 3/3 vs kalshi_paper.rejected, cold band edges for
  between/less, warm side unchanged, maker q = live executor, strict trade-through after decision, CLV close, select idempotent,
  settle skips open markets, tick end-to-end with a torn last line, edgelab adapter rows). `../.venv_quant/bin/python -m pytest
  -q tests/test_e4c_cold_paper.py` -> 11 passed.
- edgelab/adapters/e4c_fwd.py: forward adapter for trial e4c_cold1_maker_fwd-c7e0b93e (mirrors e4p_fwd).
Exact deploy lines (lead; quant-sports e4p job is where e4p.jsonl is written, kx_throttle is there, launchd every 300 s):
```
cd /Users/jonathanwang/Desktop/vocalizations_lab/.claude/worktrees/quant-trader/sports
../.venv_quant/bin/python -m pytest -q tests/test_e4c_cold_paper.py
source scripts/atomic_deploy.sh
acp scripts/e4c_cold_paper.py ~/.claude/quant-sports/scripts/
acp edgelab/adapters/e4c_fwd.py ~/.claude/quant-sports/edgelab/adapters/     # only if edgelab runs from the deployed copy
# scripts/qs_job.sh, e4p) case: replace line 216
#     if [ $((now - last)) -ge 3600 ]; then run scripts/e4p_paper.py settle && echo "$now" > state/e4p_settle_last; fi ;;
# with
#     if [ $((now - last)) -ge 3600 ]; then run scripts/e4p_paper.py settle && echo "$now" > state/e4p_settle_last
#       run scripts/e4c_cold_paper.py tick; fi ;;   # E4C cold-band paper (notes/e4_coldmargin.md): reads e4p.jsonl, no orders
acp scripts/qs_job.sh ~/.claude/quant-sports/scripts/qs_job.sh
```
(Alternative if the lead prefers the kalshi job ~/.claude/quant-kalshi/run.sh: after the kalshi_exec line add
`if [ "$1" = tick ]; then /Users/jonathanwang/.claude/quant-sports/.venv/bin/python /Users/jonathanwang/.claude/quant-sports/scripts/e4c_cold_paper.py tick >> e4c_cold.log 2>&1; fi`
-- check the quant-sports venv path first; the e4p job placement is simpler.)
vault.yaml entry (user's file; proposed text, not added):
```
  kalshi_temp_e4c_fwd:    # e4_coldmargin 11-23: forward-only paper; E4 cold-band (cold margin 1F) EXTRA maker NO bids
    loader: kx_forward_adapter_only   # selected from e4p decision rows (prereg notes/e4_coldmargin.md)
    family: kalshi_weather_temp
    hidden: [result]
    outcome: {kind: result_side, key: bet_id}
    splits:                                # date = event day D; decision 22:00-22:50 station-local D-1 (pre-event)
      dev: null
      sealed: null
      forward: [2026-11-24, null]
    forward_adapters:
      e4c_cold1_maker_fwd: {module: edgelab.adapters.e4c_fwd, params: {}}
```
Forward judge (propose to pre-set before the first forward bet is scored, e.g. `edgelab setkill`): spec kill roi_ci_below_zero
after 60 fills; study decision at 150 forward fills (window F rate 1.64 fills/day -> ~3 months, through the Dec-Feb season where
E4's dev losses concentrated): PASS iff units > 0, one-sided date-cluster p < .05 AND above the 90th pct of a price-matched
random placebo on the same nights (needs e4p bracket rows + tapes, as here); units <= 0 -> retire. No money before that and the
user's yes. A NO-ask >= .99 exclusion is NOT proposed (post hoc; W1 and window F disagree on that bucket).

### 2026-11-24 ~04:50Z lead heads-up (wx_bot_flow) addressed -- prereg unchanged
- **edgelab split of every date in window F** (edgelab/vault.yaml): orig-7 (dataset kalshi_temp_e1) event dates 2026-10-07..10-21
  belong to NO split; 2026-10-22..11-22 are the E1 FORWARD split [2026-10-22, null]. X1 (kalshi_temp_x1) 2026-10-19..10-21 belong
  to NO split; 2026-10-22..11-22 are the X1 FORWARD split. No date is in a dev or sealed split. In window-F terms: 15 orig-7 dates
  + 3 X1 dates are split-less, 32 dates per cohort are E4-forward dates (the live E4 forward data).
- **This is the second look at post-sealed kalshi_weather_temp data**: wx_bot_flow used orig-7 days from 10-22 for research first
  (not on the cold-margin hypothesis). My window was chosen and pre-registered without seeing its results.
- Tape reuse: my pulls had finished (04:38Z) before the message. wx_bot_flow's results/wx_bot_flow/trades.parquet covers D-1
  daytime prints (e.g. KXHIGHAUS-26NOV02-B82.5 15:00-18:50Z on D-1), not the (22:00 D-1, close] fill window, so it could not
  have replaced these pulls and cannot cross-check them (0 overlapping trade spans on the 937 shared tickers).
- Live-bot courtesy, disclosed: my Kalshi pulls ran 03:15-04:38Z (inside the 03-06Z executor window) at ~1 req/s (kx_throttle +
  0.7 s pause; 0.3 s for the last 1,585 tapes from 04:18Z). kx_throttle retries 429s silently, so I cannot say none occurred.
  No further Kalshi pulls are planned by this agent.

### 2026-11-24 ~05:00Z critic review (fresh critic subagent; own code /tmp/e4c_critic/*.py): ACCEPT-WITH-CHANGES, verdict unchanged
Reproduced independently: S +4.074 [-0.002, 8.22] p .02525, halves 2.395/1.679, 217 picks / 77 fills / 2 losses, taker +4.518,
P2 -28.969, P1 1.525/5.619/.7845; selection 0 disagreements; Kalshi rule text confirms less = value < cap, between inclusive;
T correct across the 11-01 DST change and for Phoenix; fresh Open-Meteo fetch reproduces fc.parquet; S survives requiring the
fill print after T + 2 h (+3.86); duplicate tape rows are distinct trade ids. Changes adopted:
1. Wording: the 95% CI lower bound moves with the seed (seeds 1-5: -0.35..+0.07). Say "+4.07u, one-sided p .025 (meets the
   pre-registered p < .05); the 95% interval touches 0 -> not distinguishable from 0 at 95% two-sided".
2. BUG FIXED: result_mismatch flagged 3 brackets whose events have NO expiration_value (NaN compares False), contradicting
   nothing; the prereg keeps them. Fixed (scripts/e4_coldmargin.py frame), frame + score re-run once. EXTRA / E4 / S unchanged.
   Corrected placebos (results/e4_coldmargin/score.json; pre-fix copy score_v1_prefix.json): P1 mean +1.16, p90 +5.28, p95 +6.68,
   actual at the 81.6th pct (taker 85.1th); P2 308 picks, 242 fills, 84 losses, -30.97u (2 restored P2 rows have no tape -> counted
   filled losers by the pre-registered missing-tape rule). Verdict still LEAD (S 4.07 < P1 p90 5.28).
3. Disclosures (critic's computation): P1 split by price: at NO >= .90 (206 picks) EXTRA +1.73u vs random mean +2.04u (34th pct);
   at >= .98 EXTRA -0.10u vs random +0.99u. ALL of the excess over random is the 11 picks at NO < .90: +2.34u vs random -0.51u,
   and that bucket's pool spans NO .10-.89 (not really price-matched; it also drives most of P1's spread, sd 2.93 vs .95 at >= .98).
   orig7 +4.11u (66 picks, 0 losses), of which +2.84u from 7 picks at NO < .90; X1 -0.03u (151 picks, both losses).
   Reading: no sign of selection skill beyond price in the 99%-of-the-time cheap-NO part of the band; whatever lead exists is in
   ~11 picks where the market disagreed with all members, a subgroup found after the fact.
4. Timestamps: my first draft labelled local (PST) times as Z. True order: notes 03:10:50Z, registration 03:11:01Z, scorer frozen
   03:17Z, markets.parquet (outcomes) 03:33:39Z, smoke-test leak 04:15Z, scoring 04:40Z. Labels corrected above; the spec's
   legacy text "new 2026-11-23" is a local date (left as registered so the spec hash is unchanged). Consequence fixed: the paper
   arm's FORWARD_FROM moved 11-24 -> 11-25 (11-24 ET decisions ran 03:00-03:50Z, partly before registration).
Not verified (inherited caveat): whether previous_day1 at hours 16-18 local (west/central zones) can come from an ECMWF/GEM run
not yet published at T; probably negligible (fcsafe vs live member MAE 0.1-0.4F except GFS 1.2F).

## VERDICT (2026-11-24)
LEAD, not a pass. Cold margin 3F -> 1F EXTRA picks on fresh window F: +4.07u maker trade-through (217 picks, 77 fills, 2 losses),
95% CI [-0.002, +8.22] (not distinguishable from 0 at 95%; one-sided p .025), both halves > 0, beats wrong-day members (-30.97u),
but NOT a price-matched random selection (random mean +1.16u, p90 +5.28u; actual 81.6th pct). Taker +4.52u [+0.60, +8.35].
As-issued live members (U_live, 92 picks) +1.54u [-1.47, +4.32], same sign. The band's money in F came from ~11 contrarian
picks (NO < .90, market disagreeing with all members); the 145 picks at NO .99 lost -0.47u. Proposal: forward PAPER arm only
(scripts/e4c_cold_paper.py, deploy lines above); no change to live E4; no money.
W2 (2021-08..2024-02): no honest variant of THIS change (E4 cannot fire, < 3 point-in-time members; MOS substitution changes
the band because MOS max-T is unbiased while raw members are +1.35F cold).

## Learnings
### Narrowing E4's cold margin to 1F made money on a fresh window but not beyond price
EXTRA picks (1-3F below the coldest member) on 2026-10-07..11-22: +4.07u maker [-0.002, +8.22], 2 losses in 77 fills; a random
selection of the same count in the same date x NO-price cells averages +1.16u (p90 +5.28) (results/e4_coldmargin/score.json).
The cheap-NO longshot premium pays on its own; the forecast filter added no measurable skill at NO >= .90.
### The cold band's forward money came from a few contrarian picks
11 of 217 picks had NO < .90 (market YES 10-55% against all 5 members) and gave +2.34u of +4.07u; 145 picks at NO .99 gave -0.47u
maker (one 98c loser erases ~50 wins) (results/e4_coldmargin/scored_rows.parquet). Unit-cost staking makes cheap-NO bets carry
the upside; any band result should be split by price before it is believed.
### The fcsafe backtest members track the live members closely in late 2026
Open-Meteo previous_day1 vs the live 22:00 response on 446 city-nights: MAE ECMWF 0.10F, ICON 0.30, GEM 0.40, NBM 0.30, GFS 1.18
(mean -0.48); coldest member MAE 0.54F; 407 of 456 live E4 picks reproduced (results/e4_coldmargin/frame.parquet, live_fc.parquet).
### A fake-outcome smoke test must also mask outcome-derived columns
Randomising `result` but printing `expiration_value` showed me the real CLI on 21 EXTRA rows before the one real scoring
(notes above). Mask every column derived from the outcome (expiration_value, settlement fields), not just the label.
### A NaN settlement value is not a result mismatch
`own != result` with own computed from a NaN expiration_value flags the yes bracket of every value-less event; 3 such events were
wrongly dropped (critic). Guard mismatch checks with value.notna().
### Label clock times with their zone from the source, not from memory
Local PST file stamps written as Z put the prereg 8 h early and nearly let the paper arm start on a day whose first decisions
preceded the registration (FORWARD_FROM fixed 11-24 -> 11-25). Take times from `date -u` or the registry row.
