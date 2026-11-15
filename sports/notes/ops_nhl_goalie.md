# ops_nhl_goalie -- wire NHL-om-A forward paper + NHL starting-goalie confirmation logger (ops agent, 2026-11-15)

Ops agent for the quant-trader lead. User approved 11-13 ("lets keep it running") READ-ONLY paper loggers under
launchd. No orders, no keys, no git, no .env / kill files, live executors untouched. Every Kalshi GET goes through
scripts/kx_throttle.py (shared cross-process slot, <= ~3.3 req/s over all paper jobs).

## Task 1 -- forward paper logger for nhl_om_a_kft-754cfc46 (rule: notes/nhl_opener_model.md "Forward rule")

### Success / kill for this ops task (written before deploying)
- Success: (a) runs from ~/.claude/quant-sports under launchd; (b) after one run data/kx_forward/nhla.jsonl has
  plausible rows (events that exist, first-tight hour >= 2 h before tip, prices in (0,1), m in 4-8 goals, c within the
  dev range ~[-0.2, +0.2]); (c) a REPLAY of the forward code on dev games (FROM moved back to 11-08) reproduces
  kalshi.py's trial-A rows (T_A, S50, L_K, m, g, bet/skip) on the large majority of games, differences explained;
  (d) `edgelab evaluate nhl_om_a_kft-754cfc46 --split forward` runs on a SANDBOX registry copy.
- Fail: needs Desktop paths, writes outside its own files, or cannot reproduce dev decisions.
- The rule is NOT re-decided: m = results/nhl_opener_model/model.py predict() with projected goalies (last-10 start
  shares); T_A, ladder, S50, c, g, k = 0.40, cost = ask + exact taker fee exactly as results/nhl_opener_model/kalshi.py.

### Stated forward deviations (not tuned)
1. b2b comes from the NHL API schedule (team has a regular-season game on game date - 1). Dev read it from MoneyPuck
   results, which at decision time (~32 h pre-tip) do not yet contain the D-1 game. Same quantity, schedule-known.
2. Candles are fetched for EVERY strike of the event (dev: strikes within +-2.0 of the DK open, unknown pre-game).
3. c pool = kalshi.py trial-A dev rows (seeded from kalshi_rows.pkl, 10-07..11-14) + forward decisions, ordered by
   decision time, rows with tip < T, last 100 non-null (m - L_K) -- the dev formula on a growing pool.
4. Forward set = NHL regular-season games with ET date >= 2026-11-15 (registration 2026-11-15T09:42Z; the dev test
   ends 11-14). Some 11-15/11-16 first-tight hours precede registration; outcomes were unknown then (flagged).
5. MoneyPuck is refreshed once a day (05:00 PT) incl. the current-season summaries (new goalies). A decision waits for
   MoneyPuck to contain the last completed game day before D (up to tip + 12 h, else it is logged with mp_stale).

## Task 2 -- NHL starting-goalie confirmation timestamp logger (data only)

### Why
notes/nhl_opener_model.md: Kalshi absorbs the confirmed starter mostly between T-12 h and T-6 h (slope 0.21 -> 0.37),
and by T-60 the model with the actual goalie has no edge (CLV +0.10c [-0.18,+0.35]). Any goalie edge must trade at
confirmation time. No historical confirmation timestamps -> log them going forward.

### Sources (all free, read-only, low rate)
- DailyFaceoff starting-goalies page (https://www.dailyfaceoff.com/starting-goalies/<ET date>): robots.txt allows
  everything except /api/ and /cms/ -> we read the HTML page only (its __NEXT_DATA__ JSON: per side goalie, newsStrength
  Confirmed / Likely / Unconfirmed, newsCreatedAt, source name + URL). Polled at most every 10 min, only while a game
  is within 14 h.
- ESPN scoreboard `probables` (probableStartingGoalie; no confirmed flag) every 10 min -> log every change.
- NHL API gamecenter/<id>/boxscore: pre-game has no goalie block (checked 11-15: playerByGameStats empty); the
  `starter` flag appears with the official lineup -> polled every tick in [tip - 75 min, tip + 20 min].
- Kalshi KXNHLTOTAL + KXNHLGAME + KXNHLSPREAD top-of-book for every market of games with tip in the next 14 h, every
  ~2 min (3 GETs per tick: /markets?series_ticker=..&status=open).

### Pre-registered future test (do NOT run before >= 3 weeks of data, i.e. not before 2026-12-06)
Question: after a starting goalie is confirmed, does the Kalshi total ladder move slowly enough that buying at the ask
+ fee right after OUR first sighting of the confirmation earns more than the friction?
- Event: game-side confirmation. t_news = DailyFaceoff newsCreatedAt of the first Confirmed row for that side;
  t_seen = our first sighting of Confirmed (DFO) or of the NHL API starter flag, whichever first.
- Goalie delta per game: dm = m(confirmed starters) - m(projected, last-10 shares), model.py at the same D; both
  sides combined at the time the second side is confirmed, single-side dm when only one is confirmed (each side's
  confirmation is its own event, dm computed with the other side still projected).
- Absorption: regress L_K(t + x) - L_K(t_news - 2 min) on dm for x in {0, 2, 10, 30, 60 min, close}; report slopes
  with date-clustered bootstrap CIs (4,000, seed 0). Speed = slope(t_seen + 2 min) / slope(close).
- Trading rule tested: events with |dm| >= 0.15 goals; at the first snapshot >= t_seen + 2 min buy the S50 rung
  (mid nearest 0.50, two-sided) on dm's side at the ask (YES) / 1 - bid (NO) + exact taker fee.
  Primary metric: net CLV = (rung close mid at tip - entry mid) signed - (cost - entry mid), cents per contract.
  Secondary: ROI graded on Kalshi result.
- Success (-> write a fresh prereg for a forward paper trial, register in edgelab nhl_lines): net CLV > 0 with
  date-clustered 95% CI lower bound > 0, n events >= 60.
- Kill: net CLV point estimate <= 0, or slope at t_seen + 2 min >= 70% of the close slope (already priced when we
  see it). Otherwise NEEDS DATA (extend 3 more weeks once, then kill).
- Seen before writing this: only the probe rows below (11-15 page, 11-10 archive page); no Kalshi move computed.

## Log
- 11-15 probes: DFO page JSON has per side newsStrengthName + newsCreatedAt + source; date-archive pages exist
  (2026-11-10: 14/14 sides Confirmed, newsCreatedAt 15:33Z..02:32Z, one confirmed the day before). NHL API landing
  pre-game: goalieComparison only (season leaders), no starter. ESPN scoreboard: competitors[].probables (no status).
  Kalshi codes: SJ, LA, TB, UTA, NJ? (event KXNHLTOTAL-26NOV15NSHSJ); NHL API uses SJS, LAK, TBL, NJD.
- Kalshi event codes confirmed from the dev keys (c_seed_A / kalshi_rows.pkl): SJ, LA, TB, NJ, UTA; mapping
  NHL->Kalshi {SJS:SJ, LAK:LA, TBL:TB, NJD:NJ}.
- Built scripts/nhl_om_forward.py (tick|status) -> data/kx_forward/nhla.jsonl. c seed = results/ops_nhl_goalie/c_seed_A.csv
  (317 kalshi.py trial-A rows, make_seed.py).
- **Replay fidelity check** (NHLA_FROM=2026-11-08 into results/ops_nhl_goalie/replay/, compare_replay.py): 49 dev games
  compared; m identical (max |diff| 0.0000); S50 identical 49/49; T_A identical 48/49 (26NOV10MINMTL forward T 3 h
  later, L_K 6.20 vs 6.15 -- all-strike candles, deviation 2); c max |diff| 0.008 (pool tail ordering); g max |diff| 0.044;
  **bet/skip agreement 49/49** (2 bets both). Settles of the 2 replayed bets identical to bets_A.csv (result no / no,
  close mid 0.475 / 0.485, clv_g 0.106 / 0.161).
- edgelab: vault.yaml new dataset kalshi_openers_nhl (forward [2026-11-15, null], adapter kx_fwd trial nhla);
  adapters/kx_fwd.py KEY nhla -> key. `python -m edgelab --registry results/ops_nhl_goalie/registry_sandbox.jsonl
  evaluate nhl_om_a_kft-754cfc46 --split forward` runs (5 decisions, 1 bet pending, 0 settled); adapter_check.py on the
  replay log: 2 settled bets parse, kalshi_result disagreements 0. The spec still says rule forward:unwired (registered
  spec, hash-pinned -- not edited).
- Built scripts/nhl_goalie_logger.py (tick | dfo-archive | status) -> data/nhl_goalie/. Goalie sources poll while a
  game is within 36 h (starters are sometimes named the day before: Vladar 11-15 confirmed 11-14 20:59Z, ~27.5 h out);
  Kalshi snapshots only within 14 h (earlier moves are recoverable from Kalshi 1-min candles).
- Deploy: scripts/install_nhl_agents.sh (copies nhl_om_forward.py, nhl_goalie_logger.py, kx_forward.py, kx_throttle.py,
  qs_job.sh, kx.py, model.py, fetch_mp.py, c_seed_A.csv, mp/ seed 134 MB) + qs_job.sh cases nhl-om-a / nhl-goalie.
  MoneyPuck refreshed by hand in the deployed copy once (fetch_mp.py --refresh, 33 teams / 176 goalies).

## Jobs (READ-ONLY, launchd, deployed root ~/.claude/quant-sports, logs logs/<job>.log)
| label (com.jonathanwang.quant-sports-*) | does | schedule | output |
|---|---|---|---|
| nhl-om-a | NHL-om-A decide at first-tight hour + settle; MoneyPuck refresh | launchd 1800 s, 24 h; MP refresh daily from 05:00 PT | data/kx_forward/nhla.jsonl |
| nhl-goalie | DFO / ESPN / NHL-API goalie observations + Kalshi NHL top of book | launchd 120 s; idle unless a game within 36 h; DFO + ESPN <= 1 per 10 min; NHL boxscore tip-75..tip+20 min; Kalshi 3 GETs/tick within 14 h | data/nhl_goalie/events.jsonl, kx/<UTC date>.jsonl.gz, games.jsonl |
Stop one: `launchctl bootout gui/$(id -u)/com.jonathanwang.quant-sports-<label>` and delete
~/Library/LaunchAgents/com.jonathanwang.quant-sports-<label>.plist. Stop all quant-sports jobs: `touch ~/.claude/quant-sports/STOP`.

## Verification (11-15 ~10:10Z)
- nhl-om-a (launchd kickstart, exit 0): 5 pick rows for the 11-15 slate: 1 bet (KXNHLTOTAL-26NOV15EDMPHI under 6.5, NO at
  0.54 + fee 0.017; m 5.57, L_K 6.29, c -0.015 (pool 100), g -0.71, T 11-14 17:00Z = 31.5 h pre-tip), 4 skip gap
  (|g| 0.05-0.31). b2b flags checked against the NHL schedule for 11-14 (EDM, PHI, FLA, DAL, CGY, ANA, SJS played; NSH,
  VAN, VGK did not) -- correct. 26NOV16VANUTA waits for MoneyPuck to include 11-14 (not yet published at 02:09 PT).
  Caveat: the 11-15 decisions (T on 11-14) precede registration (11-15 09:42Z); outcomes unknown then.
- nhl-goalie (launchd, exit 0): first tick +20 observations (dfo 10 rows incl. 1 Confirmed: PHI Vladar, newsCreatedAt
  2026-11-14T20:59:16Z; espn 10 probables); next tick +0 (dedupe works). Kalshi: first game 00:30Z is > 14 h away at
  10:08Z, so no snapshot yet -- a scratch run with a 40 h window wrote 90 rows (6 games x 15 markets, 1.6 KB gz).
- DailyFaceoff archive backfill (dfo-archive 2026-10-07..11-14, 39 pages at 1 per 15 s, 0 errors ->
  ~/.claude/quant-sports/data/nhl_goalie/dfo_archive.jsonl): 546 sides, 543 Confirmed. Hours from the LATEST news item
  to tip: median 6.7, quartiles 0.5 / 7.9, 95% 23.9; 46% < 6 h. The 0.5 h cluster = warmup confirmations overwriting an
  earlier morning item, so archive times are upper bounds on confirmation time, not the first confirmation
  (results/ops_nhl_goalie/archive_summary.py). Not used for any test.

## For the lead
- Run the first REAL `python -m edgelab evaluate nhl_om_a_kft-754cfc46 --split forward` yourself (pins the adapter hash).
  vault.yaml changed (new dataset kalshi_openers_nhl) -> vault_sha256 changes for later evaluations.
- Deployed qs_job.sh = worktree copy (byte-identical before my two cases were added).
- Option (needs a fresh prereg + your yes, NOT run): the DFO archive + Kalshi 1-min candles could give a biased early
  read of the goalie-confirmation test now, instead of waiting 3 weeks. Bias: archive times are latest-item times.

## Learnings

### DailyFaceoff's starting-goalie page carries timestamps and has a date archive
The page's __NEXT_DATA__ JSON gives per side newsStrengthName (Confirmed/Likely) and newsCreatedAt (+ source tweet URL);
/starting-goalies/<date> serves past dates (39/39 pages 10-07..11-14). robots.txt allows all but /api/ and /cms/.
Matters: "no historical confirmation timestamps" was wrong -- but the archive keeps only the LATEST item per side.

### Archived DFO confirmation times are upper bounds (warmup overwrites)
Latest-item time to tip: quartiles 0.5 / 6.7 / 7.9 h, 46% under 6 h (dfo_archive.jsonl, archive_summary.py): many sides
show the "confirmed in warmups" update, not the morning confirmation. Only a forward logger records the FIRST Confirmed.

### Forward logger reproduced the NHL-om-A dev decisions 49/49
Replay with NHLA_FROM=2026-11-08: m identical, S50 49/49, T_A 48/49, bet/skip 49/49, settles identical on the 2 bets
(results/ops_nhl_goalie/compare_replay.py). Fetching every strike instead of +-2 of the DK open moved one first-tight hour.

### b2b must come from the schedule in a forward NHL model
model.py read back-to-backs from MoneyPuck results; at the first-tight hour (~31 h out) the D-1 game is not in MoneyPuck
yet, so a forward run would mark every b2b as rested. The forward logger reads b2b from the NHL API schedule.
On the 11-15 slate 4 of 5 games had a b2b team -- the error would have hit most decisions.
- nhl-goalie Kalshi snapshot verified 10:31Z (launchd tick, exit 0): kx/2026-11-15.jsonl.gz 15 rows = EDMPHI total 9 + game 2 + spread 4 markets (the only game inside 14 h), e.g. KXNHLTOTAL-26NOV15EDMPHI-9 bid .19 ask .20; ticks every 2 min (02:17..02:31 PST headers).
