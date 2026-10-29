# quant-trader / sports — STATE (read this first; overwrite at the end of every work chunk)

Last updated: 2026-10-22 ~09:15Z (HANDOFF #3 -- read it first). Branch `worktree-quant-trader` (repo ttt5600/NeruoBatLab), worktree
`.claude/worktrees/quant-trader`. Run Python as `../.venv_quant/bin/python` from `sports/`.
Push: `git -C <worktree> -c credential.helper='!gh auth git-credential' push -q https://github.com/ttt5600/NeruoBatLab.git worktree-quant-trader`

## FUND GOAL (user, 2026-10-27): double a $1-2k UNLEVERED bankroll in 12 months, proven live; scale capital only as
## verified qualified daily volume grows (~$140-280/day of 2%-edge bets needed). Blueprint: notes/FUND_ARCHITECTURE.md;
## economics: notes/fund_economics.md. Never fund betting from the leveraged brokerage account.

BASE SETUP (user 2026-10-27): bankroll $2,000 unlevered on Kalshi (user to deposit); 1 unit = $50 (2.5%); TIERED sizing:
A proven (sealed pass: weather E1/E2/E4) <= 1u/night total; B paper-lead w/ forward evidence <= 0.5u/bet; C judgment <= 0.25u/bet,
monthly C loss cap; loss stops daily -3u, weekly -6u. All bets through one router (scripts/bet.py) incl. phone bets logged.
CAPITAL GATES (user offers up to $10k "if confident", 2026-10-27): start $2k Kalshi; -> $5k when tier-A live ROI stays inside
the sealed CI for ~60 days AND verified qualified volume > ~$250/day; -> $10k when a 2nd edge (LIP / Polymarket-US maker /
ForecastEx weather) passes sealed + forward. Venues: Kalshi (API, live); Polymarket US (connector being built, needs user API
key); ForecastEx/IBKR (feasibility); Robinhood = no event-contract API (manual only); CA sportsbooks not legal.
E5 ForecastEx weather: SEALED PASS-UNDERPOWERED 10-28 (219 bets, +6.9% ROI, p .0004; last-trade prices, thin). Next for E5:
IBKR account (user) -> forward paper on live bid/ask -> small live.
USER ASK queued: sealed test of Q003 Mentions maker NO (IN1_d6, events >= 2026-06-25) -- decaying edge, say "yes Q003".
TODO (10-28): sports/espn.py::parse_prop returns 0 rows -- ESPN dropped the over/under marker; first record of each pair = over
(586/586 matched DK "N+"). Fix + re-run collect.py props. Props scan 10-28: Kalshi vs DK fair median gap 0c -> no edge.
E5 LIVE AUTHORIZED (user 2026-10-28 ~21:45 PT: "no need to dry run mode only", "if it works it works"): ForecastEx executor
scripts/forecastex_exec.py live on IBKR LIVE account (gateway localhost:5050, auto-confirm on), frozen E5 rule, maker at best bid,
caps $5/order, 10 contracts, $25/day, daily loss stop -$10, KILL file. Tier B. Gateway needs daily re-login.
Next: L40 LIP paper test starts 10-29 (lip_paper.py); review L38 in-game longshot + live_board results; daily brief
07:30 (notes/briefs/); user asks pending: KALSHI_EXEC_STYLE=maker, fund shard 3, agent authority.

## HANDOFF #4 2026-10-25 -- START HERE: read `sports/notes/HANDOFF_2026-10-25.md` (scoreboard, every edge, systems,
## Robinhood, NBA action plan). Then EDGES_AND_AGENTS.md. User wants performance numbers + NBA action + keep researching.
Snapshot 10-25: LIVE Kalshi 29 fills / 14 settled 14 wins +$0.23 (bal $35.49 + 15 open); paper 68/69 wins +0.53u;
4 PASSED edges (E1, E2 live; E3 not tradeable from CA; E4 MAKER deployed but OFF until user adds KALSHI_EXEC_STYLE=maker).
Sealed this sprint: O1/E2/C1/M3 FAIL, MAKER PASS. Next: L28 NBA big-lead live paper logger (scripts/nba_live_scan.py),
NBA/NHL game-level fair-price logger, PredictIt before 11-03, F1 interim Tue 10-27.

**FULL REGISTRY of every edge, hypothesis, agent (IDs, briefs, findings, files, how to reach) and peer session:
`sports/notes/EDGES_AND_AGENTS.md` -- read it right after this block. Keep it updated after every agent/trial.**

## UPDATE 2026-10-25 ~10:30Z (user: "lots of betting tomorrow, find edges; comebacks, matchups, coaching, refs -- NFL and NBA; reports on all edges")
- Reports: notes/edge_report_2026-10-25.md (scoreboard), notes/slate_2026-10-25.md (10-25/26 card: 0 pregame flags, all within ~1c
  of Pinnacle; SNF KC@SEA rain rule borderline; in-game cheat sheets), notes/nfl_comebacks_matchups.md (0/70 survive),
  notes/nba_comebacks_matchups.md (sides efficient; leads L30 P-PAINT, L31 P-TOT3). Registry updated (section 4 NEW block).
- 10-27 SIZE UP (user "we can up that price"): .env BANKROLL 400, MAX_CONTRACTS 3, MAX_ORDER 3.10, DAILY 30, OPEN 40 -> orig7 $3.10
  / x1 $2 per pick; loss stop 24h -$20, 7d -$40; preflight 0 FAIL. Backup ~/.claude/quant-kalshi/.env.bak-2026-10-27. Still TAKER
  (maker E4 passed sealed; enable with KALSHI_EXEC_STYLE=maker).
- 10-27 DEPLOYED L39 monthly rain/snow paper logger (launchd com.jonathanwang.quant-kalshi-wx-monthly, hourly :05, decides 22:00
  local in a month's last 7 days): `scripts/kalshi_wx_monthly_logger.py report`. Preview 10-27: 22 qualifying NO brackets (15 cities).
- 10-27 research: NFL recency DEAD; NBA shooting luck no bet (in model already; cold-over seed 5/10 seasons); NHL luck + blind
  6.5 unders DEAD (16 seasons: 6.5 under -4.3%/1,160) -> CANCELLED tonight's 3 NHL under orders (unfilled); kept SJ/NJ ML.
  User NBA leans revised: drop POR@LAL over (luck-only); BKN@BOS under stays (logger pick).
- 10-27 RESULTS: NBA open-totals logger first picks 3-0 (CLV +1.5); manual DET@EDM U6.5 filled & LOST (7 goals, -$1.80); TOR@CGY U
  + PHI+4.5 unfilled (both would have lost); weather live 40/40 +$0.76. 10-27 MANUAL (resting, expire at puck drop, $9.36): BUF@SJ
  U6.5 NO@.45, OTT@VGK U6.5 NO@.50, CBJ@PHI U5.5 NO@.49, SJ ML YES@.43, NJ ML YES@.47 (x4 each). NBA leans for user (shard):
  BKN@BOS under ~223, POR@LAL over ~234, SAC@SA over (value gone), NY by >6.5 @.41. notes/tonight_nba_2026-10-27.md
- 10-26 MANUAL BETS (user-authorized $25 Kalshi bonus; scripts/manual_bets.py, post-only maker, expire at start, cap $25): resting
  DET@EDM U6.5 NO@.45x4, TOR@CGY U6.5 NO@.46x4, DAL@PHI PHI+4.5 NO@.50x4 ($5.64). NBA orders REJECTED: "insufficient_shard_balance"
  (NBA markets on a separate Kalshi shard; account not set up there) -> user places NBA on phone. Sheets: notes/tonight_nba_2026-10-26.md,
  notes/tonight_mnf_nhl_2026-10-26.md; scripts/nba_tonight_sheet.py re-runnable.
- 10-26 L37 SEALED FAIL: 477 bets -3.9c/contract (ROI -7.3%, p .97); leader -3.2c, trailer -5.3c; Kalshi Brier .1284 beats model
  .1319 -> in-game Kalshi is SHARPER than our mid-game model; model edges >= 5c are adverse selection. Descriptive (sealed, can't be
  a trial): Kalshi trailer mids run ~1.5c above realised in 0-15c bins (longshot bias) -> forward-only idea. L37 logger kept as a
  calibration recorder. NBA open-totals logger: BC (bias-corrected) variant pre-registered + redeployed. notes/MODELS_GUIDE.md written.
- 10-26 L28 SEALED FAIL (user "yes one shot tests"): 15 bets 13-2, ROI -8.6%, boot p .88, exact p .96 (window 2026-01-15+). 253/268
  trigger games never offered ask <= .96 (leaders won 99.6%); the cheap ones won 87% -> ADVERSE SELECTION: a cheap leader means the
  market knows something. L28 forward logger UNLOADED (dead). L37 mid-game scanner deployed (launchd com.jonathanwang.quant-nba-midgame,
  `scripts/nba_midgame_scan.py show|report`); L37 sealed prep running.
- 10-26 00:10 LOOP STOPPED (user). DEPLOYED L28 launchd com.jonathanwang.quant-nba-bigleads (2 min): scripts/nba_bigleads_logger.py
  (report: `cd ~/.claude/quant-kalshi && .venv/bin/python scripts/nba_bigleads_logger.py report`). Week of 10-26: earnings
  CLS/CDNS 10-26, V/HOOD/STX/NXPI/BA/UPS/KO/PYPL 10-27, MSFT/GOOGL/NOW/KLAC/GEV/VRT 10-28, AMZN/INTC/LLY/SNDK/COIN/MA/CAT 10-29,
  XOM/CVX/ABBV 10-30, AAPL 11-02. Midterm Kalshi vs Poly within ~0.5c; flagged "arbs" (NE/MT/SD/ID/LA Sen...) = third-candidate
  races, NOT arbs.
- 10-25m loop 8: Kalshi early NFL TOTALS not stale (47 bets: CLV +0.33 p .11, taker +1.7%/-1.1%); first tight ladder ~15 h after
  listing already within 1.1 pts of close, zero early volume. => soft opening totals are a SPORTSBOOK-opener effect; Kalshi NFL
  totals are no low-vig route. notes/kalshi_early_nfl_totals.md
- 10-25l loop 7: L36 CFB simple model (SBR 2008-22, 14 trials): TOTALS vs opener CLV +1.02 pts (3,014, p<.0002), no-vig +7.4% p .0004
  PASS, -110 +2.6% fail; spreads fail (model miscalibrated: cap/shrink, v2 needs new prereg). CROSS-SPORT: totals openers soft in
  NBA/NFL/NHL/CFB; money needs low-vig execution -> candidate: Kalshi totals ladders + forward CFB totals logger.
- 10-25k loop 6: L4 NFL props x game script DEAD (0/8 cells; only 2024 ESPN BET yardage props in dev; rush-yd unders low-total +7.7 pts p .039 = weak lead). notes/nfl_props_script.md
- 10-25j loop 5: Kalshi EARLY NFL prices stale: CLV +3.3c toward legacy model (109 games, p<.0002, both halves) but maker at first
  bid adversely selected (+16% p .09, fills when market moves against), taker at first ask -4.5% (placeholder 59c spread).
  L35 = take at first TIGHT quote (spread <= 10c, ~2 h after listing) -> forward only (don't re-mine the 109). notes/kalshi_early_nfl.md
- 10-25i DEPLOYED launchd com.jonathanwang.quant-open-lines (15 min): scripts/open_lines_logger.py NHL (ML >= .03, totals P(over) >= .04)
  + NFL (spread B >= 2 pts, total C >= 2.5) vs first-seen Pinnacle, Kalshi quotes logged; report `open_lines_logger.py report --league nhl|nfl`.
- 10-25h loop 4: L34 NHL simple model vs opener: CLV passes (ML +1.04 pts, totals +0.073 goals, p<1/4000) but no ROI trial passes at
  ESPN BET's ~4% vig (totals no-vig +7.1% p .0125). PATTERN across NBA/NFL/NHL: simple models lead openers, vig eats it ->
  the money question is EXECUTION at low vig (Kalshi maker on early markets). Next: extend open-totals logger to NHL/NFL.
- 10-25g loop 3: L6b ref-crew FT -> player points props DEAD (overs -8.1%/2,174; market moves lines toward crew; prop vig 6.9%). notes/nba_ref_ft_props.md
- 10-25f loop 2: L26 weather lock-in DEAD (below-max NO already $1.00 99.8%; above-max 460 bets -8.3%; traders see 1-5 min obs). notes/wx_lockin.md
- 10-25e loop 1: NFL bad-D unders DEAD vs opener; offence-EPA slope real but no bet (-1.6%). L28 logger building.
- 10-25d DEPLOYED launchd com.jonathanwang.quant-nba-open-totals (10 min): scripts/nba_open_totals.py (L32 forward paper:
  first-seen Pinnacle total + Kalshi KXNBATOTAL ladder vs model, pick at >= 3 pts, CLV/ROI). Report:
  `cd ~/.claude/quant-kalshi && .venv/bin/python scripts/nba_open_totals.py report` (also `show`). Redeploy:
  `bash sports/scripts/install_nba_open_totals_agent.sh`. NBA live model ready (nba_simple_live.py: ESPN paint/rosters/
  injuries; mean model-mkt +1.3). NFL model refit 2020-25 (`nfl_simple_model.py week --json`), L33 forward only.
- 10-25c L32 SEALED FAIL (narrow): +6.4% no-vig p .054, CLV +1.2, 48% vs close -> forward logger nba_open_totals.py.
- 10-25b simple models: L32 NBA totals vs opener +8.7% no-vig dev (p .001, fails ledger bar; sealed ask), L33 NFL leads opener CLV, no money.
- Next: build L28 + L31 forward-paper loggers; first-posted-totals logger for L32; fold paint-D into L2 model; ask user re sealed 2025-26 for L30/L31.

## UPDATE 2026-10-23 (user "yes M3 and yes MAKER")
- SEALED M3 FAIL (+0.45%/2,674 p .22). SEALED MAKER PASS (ROI +1.07% -> +2.22%, +30.6u, p<.001) = E4 in the registry.
- LIVE night 1: 14 fills, fees ~$0.02 total (exact, good), no KILL, balance $36.23 + 14 open NO positions.
- Maker mode DEPLOYED (commit d0c7d3a, 48 executor tests, preflight 0 FAIL) but OFF: default stays taker. Prod checks 10-23:
  expiration_time accepted (201), resting order visible after ~1.8 s, cancel OK, $0 cost. ENABLE: append
  KALSHI_EXEC_STYLE=maker to ~/.claude/quant-kalshi/.env (or KALSHI_AB=1 for a forward A/B). First maker night: confirm
  maker_fill_cost_dollars ~ q in `kalshi_exec.py status`; a mismatch writes KILL (fails safe).

## UPDATE 2026-10-22 (user: "keep backtesting, do the sealed tests" + spawn research/Robinhood/execution agents)
- SEALED O1, E2, C1 scored once each -> ALL FAIL (O1 EV -2.55%/554 though CLV +0.68 pts p .005; E2 -6.5%/97; C1 -0.69%/mo).
  No sealed asks remain open. Details: notes/results.md, EDGES_AND_AGENTS.md s3.
- EXECUTION 10-22 (notes/execution_tests_2026-10-22.md): prod order path PASS (post-only NO @$0.01 placed, dup 409, cancel);
  executor now cross-checks a fill via GET order before the convention KILL (31 tests, redeployed, preflight 0 FAIL).
  First real orders tonight 19:03-00:03 PDT. $50 sim: sealed edge -> +$29/yr median; low-CI -> +$7 (21% lose); if fees
  round UP to 1c/contract the edge is ~0 -> CHECK the first fill's fee (~$0.002 expected). Open cap $20 binds (64% of picks
  placed) -> suggest KALSHI_MAX_OPEN_USD=30. Polymarket US: read OK, needs keys. Robinhood agentic acct $0, approvals OFF.
- NEW SEALED ASK M3 (kalshi_mentions.py): buy NO on mention markets with YES <= 15c; dev +4.32%/293 p<.0003 (24h). Needs
  "yes M3". Sealed price pull (no P&L) started 10-22 in background, log in job tmp; rerun `prices --sealed` if it died.
- E1/E2 AS MAKER (kalshi_temp_maker.py, dev picks only): bid at mid +9.5u vs taker p .0003 (fill 72%, ROI +3.2% vs +1.78%);
  ask-2c also passes; ask-1c and rest-then-take fail; first 3 quarters negative. Next: sealed-period replay (needs "yes MAKER")
  or 4-wk live A/B; change spec (KALSHI_EXEC_STYLE, sweep must skip maker orders) in notes/kalshi_temp_maker.md. NOT deployed.
- K-FAV (kalshi_fav_maker.py) DEAD on dev: taker -0.32%/4,419 p .72 (fair before fees); maker 1c under ask -2.19%/2,829
  (filled bids win 90.5% vs 99.3% unfilled = adverse selection). Caution for MAKER weather: only the deeper bids passed there.
- REPORTS DONE: prior_research (L19-L23), social_edges (L24-L27, best = run E1 as MAKER: weather charges takers only),
  robinhood_strategies (DELEVERAGE FIRST; tickets in gitignored private file). Backtest agents RUNNING: kalshi_mentions.py,
  (all three finished 10-22).
- 4 background agents launched 10-22: social/forum edges -> notes/social_edges_2026-10-22.md; prior academic research ->
  notes/prior_research_2026-10-22.md; Robinhood strategies -> notes/robinhood_strategies_private.md (gitignored) +
  robinhood_strategies.md; $50 execution + payment routes -> notes/execution_tests_2026-10-22.md. If a file is missing the
  agent did not finish: re-launch with the brief in EDGES_AND_AGENTS.md s6.

## HANDOFF #3 2026-10-22 ~09:15Z (user clearing context; START HERE, then the LIVE block below)
KALSHI LIVE: weather executor trading the user's real $50 ($1 units, 1 contract/order, $15/day, $20 open; see LIVE block).
  First real orders = tonight's tick (~22-23 local). Morning check: `~/.claude/quant-kalshi/.venv/bin/python
  ~/.claude/quant-kalshi/scripts/kalshi_exec.py status`. The key file kalshi-demo.pem is really a PROD key pasted in chat
  -> user should revoke/replace (mv/sed commands are in the prior session; go_live.md has the .env reference).
  User plans ~$5k later (1% units = $50); then set KALSHI_BANKROLL_USD=5000 and drop the MAX_* overrides.
ROBINHOOD: session "connect-robinhood-mcp" has the robinhood-trading MCP (this session does not; cross-session messages
  now auto-accept: ~/.claude/settings.json crossSessionInbound=accept). Agent can TRADE ONLY the empty agentic account;
  main + Roth are read-only ("FORBIDDEN" even for review_*), so main-account changes are tickets the user enters.
  REPO IS PUBLIC: all balances/leverage/positions/plans are ONLY in gitignored sports/notes/private_account.md and
  sports/notes/robinhood_snapshot.md -- read them locally, never copy numbers into tracked files.
  BE 10-22 earnings DONE: EPS beat but stock fell ~11% on 10-23; user's short-dated BE/LRCX calls were closed at a loss and
  one BE Feb-2027 call opened. Latest full account recheck (weekend 10-25, Fri prices) = robinhood_snapshot.md section 8
  (gitignored; numbers live there only). Main account still leveraged with thin buying power -- recheck Monday at the open.
NBA DEN@OKC 10-22: Kalshi .26/.27; models: four-factor (scripts/nba_fourfactors.py) OKC -8.3, P(DEN) .280 -> NO BET by the
  rule committed before running (bet DEN only if >= .314); healthy-core ratings (scripts/nba_matchup.py) OKC -6.4..-9.6;
  2026-27 preseason starters +/-48 DEN +7.6 vs OKC +2.0 (collected data/espn/nba/*_2027) -> blend 10-25% weight gives
  P(DEN) .30-.35 = a lean, not an edge. User may say "place DEN $1" (3 YES @ .27 on Kalshi; executor only trades weather,
  so place via kalshi_exec.Kalshi client by hand).
AGENT RESULTS 10-22 (all in notes/): company graph 0/6 (graph = data product); venues: Polymarket US = own book +
  automated trading allowed, connector scripts/polymarket_us.py (read-only), weather rule on PMUS dev -0.14%/283 fail,
  no cross-venue arb; ForecastEx = different stations (new trial, needs IBKR). Player sim: simple model leads OPENER
  (+1.53 pts line move, t 8.8) -> SEALED ASK O1 still waits for "yes O1". Rumors/GDELT closed; optional $13 X test.
  Inference thesis memo + tracker (not installed). Pick'em logger not installed. Bovada declined (offshore; ToS).
USER ASKS OPEN: revoke Kalshi key; fund Kalshi; Polymarket US keys -> sports/.env; X test yes/no;
  install thesis tracker / pickem logger (scripts/install_*.sh); free disk (97% full).

## LIVE 2026-10-22 ~08:30Z (user: "use the $50, $1 trades, grow to 100, then talk about 10k")
- Kalshi PROD key (user pasted it believing it was demo; it authenticated on prod, $50.00; advised to revoke + replace) in
  ~/.claude/quant-kalshi/keys/kalshi-demo.pem (misnamed: it is a PROD key). .env: EXEC_ENV=prod, EXEC_MODE=live,
  KALSHI_BANKROLL_USD=200 (only so tiers allow 1 contract), MAX_ORDER 1.05, DAILY 15, OPEN 20, 1 contract/order;
  loss stops 24h -$10 / 7d -$20 (KILL). Executor deployed via install_kalshi_paper_agent.sh; preflight 0 FAIL in LIVE.
- First live run: no fresh picks, 0 orders, exchange balance $50, 0 positions. First real orders = tonight's tick (~22-23 local).
- Check each morning: `~/.claude/quant-kalshi/.venv/bin/python ~/.claude/quant-kalshi/scripts/kalshi_exec.py status`.
- Expectation told to user: ~1% ROI on ~$1 risked, <= 15 bets/day -> roughly +$3-5/month; $50 -> $100 is ~a year, so the
  $50 phase validates execution, not growth. Robinhood MCP connected (new session needed). Bovada automation declined.

## UPDATE 2026-10-22 (agents round 2; reports in notes/)
- PAPER weather 10-22 07Z: 54 picks, 26 settled, 26 wins, +0.84u. Kalshi unfiltered /markets 503s; series-filtered calls fine.
- PLAYER SIM (notes/player_sim.md): simple ridge team rating beats player-sim everywhere; both lose to CLOSE. vs OPENER: simple
  model gap>=4 -> line moves toward model +1.53 pts/1,512 (t 8.8, passes ledger-wide bar); realised ROI at open +3.8% p.17 (no).
  SEALED ASK O1 (needs user "yes O1"): 2025-26, frozen simple model, gap>=6, primary = close-implied ROI at opening price.
  lines_2026 schema has *_open/*_close (checked names only).
- SOFT VENUES (notes/soft_venues.md): Sleeper 0/1,047 legs +EV; Polymarket US 0/116 (within .72c of Pinnacle); Novig/ProphetX need
  accounts. Pick'em in CA = grey (AG opinion). Logger built, not installed: scripts/install_pickem_logger_agent.sh.
- RUMORS (notes/rumor_signals.md): GDELT event study 134 Kalshi rumor mkts 2025: price moves BEFORE news (median 4h); 4 rules all
  lose -> CLOSED for GDELT. Optional X test ~$13 (needs X_BEARER_TOKEN + explicit spend yes).
- INFERENCE THESIS (notes/inference_thesis.md): demand evidence strong; compute share flat 29-39% (SemiAnalysis, secondhand);
  basket outperformance mostly SNDK+INTC. Tilt 1-3% of portfolio. Tracker not installed: scripts/install_thesis_tracker_agent.sh.
- ROBINHOOD: official Agentic Trading MCP (OAuth, dedicated funded account; stocks, maybe options/crypto; NO prediction markets
  yet). User to create the account + `claude mcp add --transport http robinhood <url>` + /mcp login; approvals ON. Bovada =
  offshore/illegal in CA -> refused. DraftKings Predictions legal in CA (class action pending), manual.
- RUNNING: venues + Polymarket US connection (notes/venues.md), company graph (notes/company_graph.md).

## UPDATE 2026-10-21 night (5 parallel agents; reports in notes/, all dev only)
- EARNINGS (notes/earnings.md, scripts/earnings_*.py): BE reports 10-22 AMC; judgment P(beat EPS) ~78%, rev ~65%; straddle
  +-9.8% vs 12-q mean move 10.4-10.8% -> fair, not +EV. NO BE contract on Kalshi/Polymarket. Kalshi has no beat markets
  (only earnings-call MENTION markets); Polymarket beat markets are intl-only (restricted). 971 resolved Poly beat mkts:
  market calibrated (px .732 vs 74.3% beat); cheap-yes <=.60 +11.4% p.12 (dev); 385 post-July mkts SEALED, needs yes.
  Next: log daily consensus; backtest Kalshi mention markets; log Yahoo straddles daily.
- NBA LATE-GAME (notes/late_game_calibration.md, scripts/nba_late_wp.py): Kalshi CALIBRATED in Q4<=6min |m|<=8 (7/7 bins);
  T1 -1.5%/311 p.55, T2 -5.4%/331 -> CLOSED. Pacers 10-21 23c was at down 8 (fair); Pistons -22 at 5-6c fair.
- TWINS (notes/pairs_twins.md, scripts/pairs_twins.py): 47 trials 0 pass; dual-class +3.8% gross -> -3.5% net -> CLOSED.
- EXECUTION (notes/go_live.md, kalshi_exec.py rewrite, 30 new tests): idempotent orders, reconcile, demo env, caps + loss
  stops + KILL file, preflight/status/kill/smoke. NOT deployed: user runs scripts/install_kalshi_paper_agent.sh, adds demo
  key + EXEC_ENV=demo + KALSHI_BANKROLL_USD to ~/.claude/quant-kalshi/.env, then go_live.md checklist.
- ROTOBOT (notes/rotobot_tracker.md, scripts/rotobot_tracker.py): @rotobot_ai, paid app, no record; X free tier has no
  reads. User to choose: X API pay-per-use (X_BEARER_TOKEN in sports/.env) or paste posts to data/rotobot/posts.txt.
- R_REB SEALED SCORED 2026-10-21 (user "yes R_REB"): 1,214 bets/291 games, -18.9u, ROI -1.56%, CI [-81,+44], p .72,
  win .540 @ avg dec 1.866 -> FAIL. Rebound unders CLOSED. (graded/games_2026 sha matched frozen MANIFEST.)
- PUBLIC NBA MODELS (notes/nba_public_models.md, scripts/nba_public_models.py): only 538 (Elo/CARM/RAPTOR) + ESPN have free
  archives; ALL worse than de-vigged close (log loss +0.008..+0.019, CIs exclude 0); 7 rules 0 pass (bar p<.0071); model+market
  stacking ties the market -> CLOSED vs close. Next cheap test: same models vs the OPENING line (ESPN openers 2024-25).
- Nuggets@OKC 10-22: Kalshi .26/.27, Pinnacle fair .258-.270 (OKC -8); preseason projections (ESPN/Opta) ~.29-.32 -> told user
  judgment-tier at most; public-models result says that gap is more likely model error than edge.
- USER ASKS OPEN: demo key; X access;
  disk is 97% full (28 GiB free) -- blocked pulling pre-2021 NBA pbp.

## HANDOFF 2026-10-21 (user is clearing context; fresh session starts HERE)
UPDATE 10-21 late (this session, after HANDOFF #1): LOOPS ARE OFF by user choice (memory feedback_loops_off) -- work only
when asked; the non-Claude launchd loggers (kalshi-paper hourly + dry executor, gap-logger, live-signals) still run.
Done since HANDOFF #1 (all in notes/results.md + ledger):
- NBA props P2 (nba_props_dist.py, empirical-distribution unders vs de-vigged open): +0.27%/60.8k p.26 vs -4.05% all-unders
  baseline (selection +4.3 pts, not past vig); skew mechanism REFUTED; rebounds +2.7% again -> R_REB sealed test
  PRE-REGISTERED below, waits for user yes.
- N1 day-of temp (wx_dayof.py, fixed a missing /historical price fallback): D1 +0.19% p.34, D2 -1.31% -> dead.
- Stat arb pairs (pairs_bt.py, KO/PEP + 6 pairs): +1.89%/yr, Sharpe .41, p .052; ~0 after 2009 -> not worth automating.
- Rotobot: a research assistant, NO public/audited pick record. To test "follow Rotobot": log its picks at post time with
  line + price + timestamp (user exports/pastes them) -> grade CLV vs close + ROI; a tracker is NOT built yet.
- NEW TOOL scripts/ev_check.py: verify any bet before placing -- EV after fees + fractional-Kelly stake per notes/risk.md tiers
  (temp, temp_x1, soccer, judgment, paper); refuses EV<=0 or below-tier edges. E.g. Abbott YES .72 vs fair .75 -> $7.47 on $1k.
- Midterm scan 10-21: LA Senate gap persists (2-venue cost .9752); NV Gov 3c diff no arb; TX Gov .705/.698.
USER ASKS STILL OPEN: (1) yes/no on R_REB sealed; (2) LA Senate: on Polymarket US? + Kalshi rule says "Kentucky";
(3) Kalshi key in ~/.claude/quant-kalshi/.env to go live; (4) user's state (soccer S12 needs soft-book accounts);
(5) Robinhood: Agentic Trading = stocks only, event contracts likely Kalshi-routed (unverified) -- nothing stock-side has
passed a backtest yet (14 ETF rules + pairs all failed), so nothing to automate there yet.
NEXT if asked to work: (a) Rotobot pick tracker (CSV in -> CLV/ROI grading); (b) NFL prop unders by category x game script
on ESPN 2022+ props (research_prop_models #7, pre-register, sealed season 2025); (c) F1 in-game fade interim Tue 10-27;
(d) shrink P2 toward the market (it claimed +17 pts, delivered 0) as a new pre-registered trial.
USER DIRECTIVES (latest, 10-21):
- "Keep looking at races and other edges we've saved" + "make sure it's stuff that survives backtests" -> only models that
  pass dev -> sealed (user yes) -> paper count; a dev number alone is never an edge. Keep the ledger/Bonferroni discipline.
- Wants many correlated situational variables for player props (location/arena, schedule, back-to-backs, travel, time,
  weather, teams, nightlife-city effects e.g. "Harden worse in cities with strip clubs") -> a feature-correlation model,
  backtested over time. EV first: fade what the public chases (books shade OVERS: NBA prop overs -4..-7%, unders ~-2..0%).
- Two research agents launched 10-21 (background; they write their reports to files -- read them first):
    sports/notes/research_prop_models.md      (deep research: how to build player-prop / NBA+NFL models, who has had luck)
    sports/notes/research_situational_vars.md (situational variables: arena, b2b, travel, altitude, nightlife, refs, etc.)
  If a file is missing the agent did not finish -> re-launch with the same brief (see bottom of this block).
- User wants the Kalshi account live (funding now) + Polymarket US + Robinhood connected; cross-venue arb scans.

LIVE / SAVED EDGES (check every session):
- PASSED SEALED: (1) Kalshi temp consensus-reject orig 7 cities +1.45%/1,844; (2) X1 17 new daily-high cities +0.78%/2,464;
  (3) SOCCER S12 soft books vs Pinnacle 2019-26 CLV +3.63%, ROI +8.71%/2,004 (needs non-CA soft-book accounts; ask user's state).
- Paper (launchd L1, 24 cities) 10-21: 31 picks, 9 settled, 9 wins, +0.10u; dry executor 10 orders $198.83 (caps bind; see 000.)
- F1 NFL in-game maker-fade FORWARD paper: games 2026-10-22..11-30; `scripts/nfl_ingame_fade.py interim` (Tuesdays); score once after 12-02.
- N1 DONE 10-21: day-of temp D1 +0.19%/1,279 p.34, D2 new-volume -1.31%/364 -> no edge (market prices overnight runs).
  Was: N1 day-of temp (wx_dayof.py): build bv0ajs1qh was on the LAST model (NBM) at 10-21 ~03Z; if the session died, re-run
  `../.venv_quant/bin/python scripts/wx_dayof.py build` (cached, resumes; Open-Meteo free tier ~10k loc-calls/day) then `dev --ledger`.
- MIDTERMS (election 11-03): run `scripts/xvenue_elections.py` every session; LA Senate arb candidate (R@Poly .93 + D@Kalshi .033)
  waits on user: is LA Senate on Polymarket US? Kalshi SENATELA-26 rule text says "Kentucky". TX Gov Abbott: market .72 vs
  poll-model ~.75 -> judgment bet <= 1% B (notes/risk.md).
- SEALED ASK -- SCORED 10-21, FAIL (see UPDATE block): R_REB = NBA REBOUND unders, P2 rule (empirical last-20 P(under) minus
  de-vigged same-book opening P(under) >= 0.08, >= 20-min players, best opening under price), season 2026 only; PASS = one-sided
  bootstrap p < 0.05 over games AND ROI > 0. Dev evidence: P1 rebound unders +2.8%/14.5k, P2 rebound unders +2.7%/16.2k (same
  seasons). Needs user's explicit yes; then add `sealed --confirm R_REB` to nba_props_dist.py and run once.
- POST-HOC LEADS (need sealed data = user yes, pre-register as NEW trials): NBA rebound UNDERS at opening price +2.8%/14,535
  (dev 2023-25; sealed season 2026); NFL in-game overshoot fade (being forward-tested as F1).
CLOSED THIS SPRINT (no edge, don't redo): NFL in-game H1-H3, NBA in-game H1-H3 (Kalshi beats nflfastR/ESPN WP; lag gone
<1 min), Kalshi soccer 1X2, Kalshi CPI vs Cleveland nowcast (0 bets), gas band (0 bets), hourly temp (thin), TX/other
cross-venue elections + sports (Kalshi~Polymarket within 0.5-1c), rain R1 (sealed FAIL), lows X2 (dev -6.4%), NBA props
rolling-form P1 (-3.3%/130k). Traps: datetime64[us] epoch bug (memory feedback_datetime_unit_epoch); ESPN prop
last_updated is post-tip -> use *_open only; 3-candidate races are not R/D arbs.
RESEARCH DONE 10-21 (both reports written): key synthesis -> (a) skew: prop stats are right-skewed, lines ~ median, so a
MEAN projection leans OVER -- likely part of P1's -3.3%; model the MEDIAN / full distribution (NB/Poisson) instead;
(b) no public market-tested prop edge exists; our rebound-unders +2.8%/14.5k is stronger than anything published;
(c) NBA 2026-10-16 told licensed books to RESTRICT unders on <12-min players, same-player multi-under parlays and big alt
unders (Kalshi/Polymarket not covered) -> execution risk for unders at books; (d) referee crews (posted ~9am ET) = cleanest
structural test per the situational report, deprioritized by the models report -> test both, cheap; (e) The Odds API has
5-min prop snapshots from 2023-05 (paid) if we need timestamps.
NEXT (in order): read the two research reports -> design ONE pre-registered NBA prop feature model (situational
variables from the reports, walk-forward by season, unders-first per EV), dev 2023-25 only -> if it passes dev, ask user
for sealed 2026; keep L1-L4 + midterm scan + F1 running.
Research-agent brief (for re-launch): see the two report files' header; each agent writes its own file, cites sources,
separates evidence (market-tested) from anecdote, and lists concrete testable variables + data sources we can get free.


## Goal
Make money legitimately. User upgrades plan if we prove > $100/month. Path agreed (2026-10-14):
**confident backtest on frozen data -> buy Odds API (time-stamped) -> paper-trade fake units live -> real money, small caps.**

## Hard rules (never break)
- Sealed/lockbox seasons are scored ONCE and only on the user's explicit "yes". Never re-score.
- Never ask for / handle brokerage or Kalshi credentials. User puts keys in `sports/.env` themselves.
- Never bypass geo-blocks (Pinnacle league-info endpoint is US-blocked; odds endpoints answer — fine).
- Push CODE to GitHub, never data (`sports/data/` is gitignored).
- Automated execution only via Kalshi official API (sportsbooks have no API; automating them breaks ToS).

## Where things stand (2026-10-19)
- PASSED sealed (3): SOCCER S12 best3_pow3 2019-26 CLV +3.63% p<.001, ROI +8.71%/2,004 p .010 (user yes 10-20) -- needs
  UK-style soft-book accounts (not CA). Plus: X1 = same rule on 17 new KXHIGHT* series, +0.78%/2,464 p .0047 (2026-03..10, user yes 10-19);
  and the original Kalshi temp "consensus-reject" (+1.45% ROI, 1,844 bets) -- paper-trading (9 picks, 1 settled, +0.01u).
- BEST NEW LEAD (dev only): SOCCER soft books vs Pinnacle, scripts/soccer_fd.py, football-data.co.uk 22 divs.
  S12 best3_pow3 = best of B365/WH/bwin, power de-vig of Pinnacle pre-close, edge >= 3%, price < 10:
  dev 2012-19 CLV +2.87% (p<.001, 7/7 seasons), ROI +8.5%/2,604 (p .008); CLV monotonic in threshold.
  Trap: PROPORTIONAL de-vig overstates longshots -> fake CLV (S6 withdrawn). Sealed 2019-26 awaits yes.
  10-19 robustness (not a trial): S12 dev CLV +2.9..+3.7% under power/Shin/odds-ratio/proportional close de-vig,
  positive in 22/22 divs, 7/7 seasons, H/D/A, every price bin, all 3 books -> not a de-vig artifact.
- KALSHI SOCCER (10-19, scripts/kalshi_soccer.py, 15 leagues, $1-6M/EPL outcome): dev 0/3 (draws -6.5%,
  fav60 -1.9%, tail-NO +1.9% p.087); mids sum 1.005. Live snapshot: Kalshi mid within 0.34c (median) of
  Pinnacle power fair, 0/296 quotes >= 1% EV -> the soft-book soccer edge does NOT exist on Kalshi (US-legal
  route is closed; S12 needs a non-US-style soft book account). Sealed Kalshi soccer (2026-03+) untouched.
  Data trap: Kalshi soccer occurrence_datetime = kickoff+3h; price at close_time-4h (close = full time).
- ~45 trials 10-16..10-18, all dev, all in lab/ledger.jsonl: every NFL/NBA/MLB closing-line angle loses ~the vig;
  Kalshi sports longshots <10c lose ~45% (paper replicated) but no Kalshi rule past the bar; maker-side
  edge = mostly spread capture and a symmetric ladder test showed it does NOT survive adverse selection.
- Data traps caught this round (check every new dataset!): NFL weather joined by row position (53% wrong);
  NBA sqlite Spread is UNSIGNED (fake +7.2%); Kalshi 2026 expected_expiration = start+6h for some games
  (exp-4h decisions land after kickoff; 2026 MLB start time is IN the ticker, e.g. 26AUG171340STLCING1).
- Live: live_signals.py (NFL pre-kickoff rain -> Kalshi totals paper) evaluated 7 games on 10-18, 0 fired.
- Frozen snapshot `data/frozen/2026-10-14/` (sha256 MANIFEST). Sealed: nba 2026, nhl 2026, mlb 2025, nfl 2025.
- Older: pattern-mine sealed confirm 0/10 (10-15); NBA DK high-line under untested (needs Odds API);
  live_edges.py Pinnacle-vs-Kalshi scanner 0 flags (session-only).

## ORCHESTRATION (user 2026-10-19: "take control of orchestration... more loops"; goal $100/month revenue)
Durable loops (launchd, all from ~/.claude/quant-kalshi; reinstall with the scripts/install_*.sh after edits):
  L1 quant-kalshi-paper  hourly :03  kalshi_paper.py tick (temp consensus-reject, 24 cities = orig7 + X1 17, + settle) THEN
                         kalshi_exec.py run --live = SHADOW execution: sizes each fresh pick vs the live book,
                         logs data/kalshi_exec.jsonl. DRY unless ~/.claude/quant-kalshi/.env has EXEC_MODE=live (user).
  L2 quant-live-signals  15 min      live_signals.py tick (NFL pre-kickoff rain -> Kalshi totals, paper)
  L3 quant-gap-logger    2 min       gap_logger.py tick: every Kalshi game contract vs Pinnacle (6 US + 9 soccer
                         leagues, ~400 quotes/tick) -> episodes with EV >= 1% after fee; `gap_logger.py report`.
                         Answers "Kalshi moves around -> opportunities?". Judge after ~1 week of ticks.
  L4 research loop       this session (/loop, self-paced) + usage-watch auto-resume: works the QUEUE below,
                         one item per wake, dev data only, every trial in the ledger, STATE + push after each.
Each wake, first: `cd ~/.claude/quant-kalshi && .venv/bin/python scripts/kalshi_paper.py report;
  .venv/bin/python scripts/kalshi_exec.py report; .venv/bin/python scripts/gap_logger.py report;
  .venv/bin/python scripts/live_signals.py report` -> note numbers here.

$100/MONTH MATH (from the sealed record, 2026-10-19): consensus-reject sealed = 1,844 bets over 2025-07..2026-10
(~15 mo, ~121 bets/mo), ROI +1.45% per $ risked after fees (95% CI on units +10.9..+42.6 -> ROI +0.6..+2.3%).
$100/mo needs ~$57 per bet (CI at that stake: ~$40-$160/mo).
UPDATED 10-19 after X1 PASS: orig7 ~121 bets/mo x 1.45% + X1 ~325 bets/mo x 0.78% = ~$4.3/mo per $1 staked per bet
-> $100/mo at ~$23/bet (rough range ~$30-$165/mo at that stake, from the two sealed CIs). X1 books are thinner
(validation 10-19: LV 11, SDF 35, OKC 82 contracts within 1c) -> some fills partial at $23. Median resting depth 617 contracts within 1c ->
$57 is ~10% of depth. ~4 picks/night -> ~$230 out at once; keep ~$1,000 bankroll (one loss at NO 0.95-0.99
costs the whole stake = 20-90 wins). Profit was lumpy (2026Q1 = 16.7 of 26.8 units; 2 of 6 quarters negative).
Exec default caps ($25/order, $100/day, $300 open) -> ~$44/mo; $100/mo needs ~$60/order, $250/day, $500 open.
Capacity lever: Kalshi now lists 48 live US temp series (21 daily highs, 27 daily LOWS); we trade 7 highs.
SETTLEMENT SOURCE (checked 10-19): every Kalshi temp series switched NWS -> The Weather Company on 2026-08-14;
TWC = official NWS value on 3,080/3,081 days (scripts/wx_source_check.py --all) -> benign.

QUEUE v2 (2026-10-19, user: "I need other edges as well"; research loop works top-down, dev only, pre-register first):
  USER 2026-10-20 WIDE ASK: connect Kalshi + Polymarket (US product only) + Robinhood; cross-venue arb vs sportsbooks;
     confirm NBA/NFL edges (weather, players, totals); midterms (TX Gov Abbott); unit sizing -> notes/risk.md (written).
  M1 midterms: scripts/xvenue_elections.py (Kalshi vs Polymarket, 63 races): median R gap 0.5c; flagged "arbs" are
     3-candidate races (NE Osborn etc.), NOT arbs; LA Senate +4.3c likely a mismatched market (fix matching: Kalshi -26
     event only, Polymarket title must contain state). Run each wake until 11-03. Next: fair value from polling averages
     + historical poll error -> list races where |market - poll model| >= 5 pts (judgment bets, 1/8 Kelly per risk.md).
     TX Gov: Kalshi .715 / Poly .728; Pollsmax avg R+3.5 -> ~75% by a 5-pt error model (unvalidated).
  M2 DONE 10-21: no arb (NFL median 1c, NBA 0.5c, min round trip $1.021). Was: Kalshi vs Polymarket SPORTS game markets (NFL/NBA/soccer) -> extend gap_logger with a Polymarket leg.
  M3 Robinhood event contracts: verify whether they are Kalshi-routed (same book) before building anything.
  M4 props DONE 10-21: P1 form model -3.3%/130k (0/3); post-hoc rebound unders +2.8%/14.5k -> sealed 2026 needs yes.
  M4 DONE 10-20 (in-game half): NBA in-game 0/3 (-4.9%, -7.8%, -7.6%); Kalshi Brier .171 < ESPN .180. Props half still open.
    Was: NBA 2026-27 season: in-game (nba_api pbp wall clock + Kalshi KXNBAGAME 1-min) and props (ESPN 2022+).
  N0 DONE 10-20: NFL in-game 0/3 (H1 -5.6%, H2 -7.2%, H3 6 bets); Kalshi reprices big plays within ~1 min and beats
     nflfastR WP; only lead = post-hoc maker fade of a 1.8-pt overshoot (needs fresh data). Was: TOP PRIORITY (user 2026-10-20: wants IN-GAME sports edges -- "game scenarios when odds can't update fast enough",
     model on past info backtested on real in-game quotes). scripts/nfl_ingame.py pre-registered (commit ea02bef): Kalshi
     KXNFLGAME 1-min in-game candles (bid/ask, ~$13M/side/game) vs nflfastR vegas_home_wp after each play (time_of_day),
     act at play+60 s. H1 model gap hold, H2 lag after big plays (|dWP|>=.08, exit +6 min), H3 late favourite. DEV = 2025
     wk 1-12; SEALED = 2025 wk13+/POST + 2026. 280/285 2025 games match Kalshi. Build running. Next: NBA version (nba_api
     pbp has wall clock) if NFL shows anything.
  F1 FORWARD PAPER (user yes 10-20): scripts/nfl_ingame_fade.py -- maker fade of the post-big-play overshoot, NFL games
     2026-10-22..11-30 only (pre-registered before they were played). `interim` = descriptive look (weekly, Tuesdays);
     `score` = ONE shot after 2026-12-02. Smoke-tested on 10-08..10-13 (outside window, 55 fills, no totals looked at).
  N1 IN PROGRESS: wx_dayof.py build running in background (Open-Meteo free tier 429s -> ~2 days; resumable, cached).
     DAY-OF consensus-reject: same frozen rule but decide 06:00 station-local ON the day, lead-0 forecasts over
     hours <= 18 (all issued before 06:00? -> NO: lead-0 hour h is issued ~h; use model runs issued <= 06:00 via
     previous_day1 shifted? -- design the leak-free forecast FIRST, write it in the docstring). Kalshi temp markets
     trade all day; tighter forecasts reject more brackets -> more volume. Dev = orig kalshi_temp dev quarters
     (2024Q1-2025Q2) + X1 series dev (< 2026-03-01). Counted trials on new dataset kalshi_temp_dayof.
  N2 DONE 10-19: 0 bets (all priced rungs inside the 3-sd nowcast band) -> no edge. Was: Kalshi CPI (KXCPI*) monthly brackets vs the Cleveland Fed inflation nowcast (public, archived daily): buy NO on
     brackets the nowcast puts far outside; same consensus-reject logic. Check history depth + liquidity first.
  N3 DONE 10-19: too thin (median 1-71 contracts/market). Was: Kalshi hourly / directional temperature markets (KXTEMP*H*): same 5-model idea at 1-2 h lead. Liquidity first.
  N6 DONE 10-19: 0 bets (3sd band wider than the ladder; dev only 25 single-rung days). Was: Kalshi gas-price markets (AAA national average, daily) vs a random-walk band from AAA history.
  N7 DEPRIORITIZED (same ladder structure as N2/N6 -> band rules don't fire). Was: Kalshi TSA checkpoint-throughput markets vs TSA's published daily counts (seasonal, autocorrelated).
  N4 gap_logger report (>= 3 days of ticks by 10-22): episode counts/durations; if draw gaps dominate, re-check the
     3-way de-vig (Shin) before calling anything an edge.
  N5 soccer_fd variants (dev; bar p < 0.005) -- only matters if user's state allows soft books.
DONE (v1): Q1 X1 PASS sealed; X2 lows dead; Q4 rain R1 sealed FAIL; Q2-Q3/Q5 folded into N4/N5.

## USER DIRECTION (2026-10-15) -- go wide, everything that can earn >= $100/month
User wants: stocks, Polymarket, sportsbooks ("bet until limited; limits come ~$50k won" -- user's number,
unverified), 2-leg high-probability parlays to look recreational, in-game weather + location features for
sports, stock rules (trim on rises / buy dips, tech->industries graph model) with a hard daily loss cap.
"Start backtesting all of these." Same protocol everywhere: dev -> sealed (user yes) -> paper -> small real.
- Robinhood: official "Agentic Trading" (May 2026, MCP, dedicated funded account) is STOCKS ONLY in beta;
  event contracts "coming". No unofficial login libraries. Re-check when event contracts land.
- Polymarket: only the US-regulated product, never the intl site via VPN. Weather markets there = try the
  consensus-reject rule. Earlier Kalshi-Polymarket arb scan: <1c.
- Sportsbooks: manual bets only (no bot, ToS). Sign-up promos are the one sure +EV (one-time per book; legal
  only in states with online books -- NOT California). Parlays: book margin compounds per leg (2 legs at
  -110 ~ 9% hold) so each leg needs its own edge; testable on our prop/game data -> next sports task.
- PARLAYS TESTED 2026-10-16 (scripts/parlay_test.py, dev only, 2 trials): 0/2. Props legs at dec<=1.50 lose
  -4.7% each -> 105k cross-game 2-leg parlays -9.3% +-0.3% (compounding predicts -9.2%; hit rate 56% feels good,
  payout doesn't cover it). NFL ML favourites -1.2% single -> parlay -1.35% (n=885, ns). Parlays square the
  legs' edge; they can't create one. Favourite-longshot bias is real (prop favs -5% vs longshots -17.6%) but
  favourites are still negative. Same-game parlays untested (no SGP prices).
- USER 2026-10-16: "weather" = weather AT THE STADIUM as a sports feature, NOT betting on weather markets
  (Polymarket/Kalshi weather). Open question to user: keep or stop the Kalshi temp paper trader + Polymarket step 4?
- 2026-10-16 BUG: all NFL weather results before today used weather joined to the WRONG game (53%). Fixed.
  Corrected dev (2006-19): pre_rain_unders +8.0%/219 p=.11 (bar .0083) NOT past; rain+wind pre-kickoff -10.2% dead;
  oracle (observed in-game) rain+wind +21.4%/70, rain -1.9 pts beyond close. The sealed score I ran (+13.3%/101,
  FAIL) is INVALID (buggy data); not re-scored -- would need user's yes, and corrected dev doesn't earn it.
- USER 2026-10-16: wants iteration (many small edges) + real-time system -> eventually a wallet to auto-bet.
  Built scripts/live_signals.py: signal registry -> Kalshi pricing -> paper picks + settle. First signal
  nfl_pre_rain (forward paper test on fresh 2026 games). Kalshi KXNFLTOTAL = liquid (1c spread, 1e4+ size).
  Bovada etc: NOT automated (offshore/unlicensed in US + ToS). Wallet = Kalshi API key in sports/.env, only after
  forward paper shows an edge, with caps + kill switch.
- In-game weather/location = existing NFL leads (rain+wind unders oracle, west_coast_night, div_unders_late):
  sealed lockbox needs user's explicit yes; rain/wind also needs a FORECAST-based (not observed) version.
- QUESTIONS OUT to user (unanswered): (1) which US state? (2) total capital available?

## STOCKS (started 2026-10-15)
- Frozen: `scripts/stocks_freeze.py` -> data/stocks/frozen_2026-10-15.parquet (18 tickers 2005-2025, sha256
  097a14686d31; gitignored). Backtester `scripts/stocks_bt.py dev`: DEV 2006-2018, SEALED 2019-2025 (dropped
  at load; no confirm command written yet). Close-to-close, 2 bps/trade, alpha vs own buy-and-hold, HAC t.
  Ledger lab/stocks_ledger.jsonl (bar = 0.05/#trials; 11 so far -> p < 0.0045).
- DEV RESULT: 0/11. dip_rip (SPY/QQQ x band 3/5/10%) alpha -1.1..+2.1%/yr, best QQQ 5% p=0.05 (not near bar);
  tech_gate (hold 8 sectors only if tech rose) -6.3%/yr t=-2.6 (do NOT flip it post-hoc); graph_leadlag
  (ridge lead-lag, top-3) -1.0%/yr; 2% daily loss cap on SPY/QQQ: ~+1.3%/yr ns, smaller DD, fills optimistic.
- 2026-10-17 +3 trials (14 total, bar p < 0.0036): sector_momentum 12-1 top-3 +0.6%/yr p=.40; sector_reversal
  weekly bottom-3 -0.6%/yr; vol_target SPY 10% (Moreira-Muir) alpha +2.7%/yr t=1.97 p=.024 (not past bar),
  Sharpe 0.67 vs 0.47, maxDD -25% vs -55%, but CAGR 6.4% vs 7.5% (75% avg exposure) -> RISK CONTROL, not more
  money; the principled form of the user's daily-loss-cap idea. Only beats B&H in dollars with leverage.
- Remaining stock ideas: supplier->customer lead-lag (Cohen-Frazzini) needs a supplier graph we don't have.
  Honest read after 14 trials: simple ETF timing rules don't beat holding. Baseline: index fund ~7%/yr =
  $100/mo on ~$17k.

## AUTONOMOUS LOOP (user 2026-10-17/18: "keep looking and backtest as much as you can"; only backtests count)
Done (scripts): nfl strategies in lab/strategies (5 online angles); mlb_ml_open.py, mlb_ml_move.py (SBRO);
nba_angles.py, nba_model.py (data/nba_odds/OddsData.sqlite); kalshi_longshot.py, kalshi_maker.py,
kalshi_maker_ladder.py, kalshi_start_times.py (ESPN starts -> leak-free sealed prices); soccer_fd.py.
Done 10-19: kalshi_soccer.py (0/3), S12 de-vig robustness. Next ideas (unblocked): soccer_fd variants are the vein -- per-book volume at US-relevant books, Shin de-vig check,
min-price / draw-only splits (each = a counted trial; bar for soccer_fd is now p < 0.005 after 10 trials).
Blocked: tennis-data.co.uk 403s automation (don't bypass); NHL/college odds sources not found free;
Pinnacle O/U in soccer only from 2019 (sealed era).

## WAITING ON USER
0. NEW 2026-10-20 LA SENATE CROSS-VENUE ARB CANDIDATE (needs user checks, NOT riskless yet): Kalshi SENATELA-26 D ask .033
   (R .968/.969) vs Polymarket LA Senate R ask .93 (D .07/.08) -> R@Poly + D@Kalshi ~= $0.965 + Poly fee for $1 if R or D
   wins (~2.5-3.5%, capital locked to Jan 2027 swearing-in). Blockers: (1) Kalshi SENATELA-26 rules text says "Senator of
   KENTUCKY" (copy error -> resolution risk); (2) is the LA Senate market on Polymarket US (regulated)? never use intl;
   (3) resolution mismatch: Kalshi = party sworn in, Poly = election winner incl. runoff, independents win -> both lose;
   (4) Poly US fee unverified; (5) depth ~$50k. User decides; Claude places nothing.
(RESOLVED 10-19: user said yes; X1 sealed PASS, R1 sealed FAIL -- see results.md) 00000. Q4 RAIN: R1 (all 5 models forecast < 0.01 in for the day, lead-2 -> buy NO on Kalshi KXRAIN)
   PASSED dev: +3.42%/361 bets p<.001, 3 losses, 9/10 weeks and 20/22 cities positive. Dev = 56 summer days only.
   ~40 cities x ~25% of days dry-consensus -> roughly 200+ bets/mo at ~96c. Sealed 2026-09-15..10-18 (~5 weeks, one
   shot) needs the user's yes; add `sealed --confirm R1_dry_no` (not written yet). Forward paper of R1 can start
   without a yes (paper is not sealed scoring) -- propose to user.
(RESOLVED 10-19, X1 PASS) 0000. Q1 -- the BEST capacity lever: consensus-reject rule FROZEN, applied to 13 new daily-HIGH series
   (wx_expand.py X1): dev +1.22%/444 bets p .057 (dev only 2025-12..2026-02; sealed rule passed at +1.45%). ~300 bets/mo
   -> with the 7 traded cities ~420 bets/mo -> $100/mo at ~$17/bet. Sealed 2026-03-01..2026-10-18 (one shot) needs the
   user's yes: add `sealed --confirm X1_new_highs` to wx_expand.py (not written yet: same score(), df.date >= SPLIT,
   lockbox.jsonl, PASS = p < 0.05 and units > 0). LOWS (X2) failed dev -6.4% -> do not trade lows.
000. NEW 2026-10-19 REAL MONEY SWITCH -- user said "let's go" (10-19): opening a Kalshi account + funding.
   Executor re-checked vs docs.kalshi.com 10-19 (Ed25519 default key, V2 /portfolio/events/orders, sell-YES = buy-NO,
   KALSHI_MAX_CONTRACTS=1 default for live). After the first live fills: confirm average_fill_price ~ 1 - NO price.
   10-20 shadow night 1 (24 cities): 24 picks, dry executor sized 5 ($99.74) then skip_cap on the rest -- the default
   $100/day cap allows ~4 orders/night. For ~$100/mo after the 1-contract check: KALSHI_MAX_ORDER_USD=25,
   KALSHI_MAX_DAILY_USD=300, KALSHI_MAX_OPEN_USD=600 (user sets them). Loop reports 10-20 04Z: paper 24 picks/3 settled
   +0.04u; gap logger 342 ticks, 11 episodes (LaLiga draws persist ~5.5 h -> de-vig artifact suspected); live_signals 8 games 0 fired.
   Original text: (only route to revenue with evidence): executor is built + shadowing (dry).
   To go live the USER: (1) creates a Kalshi API key, saves the private key file, and writes to
   ~/.claude/quant-kalshi/.env: KALSHI_KEY_ID=..., KALSHI_PRIVATE_KEY_PATH=..., KALSHI_MAX_ORDER_USD=25,
   KALSHI_MAX_DAILY_USD=100, KALSHI_MAX_OPEN_USD=300; (2) runs `cd ~/.claude/quant-kalshi && .venv/bin/python
   scripts/kalshi_exec.py check`; (3) adds EXEC_MODE=live. Claude never touches the key. Kill: `touch
   ~/.claude/quant-kalshi/KILL`. First live order: verify fill vs log (order-body fields unexercised).
00. NEW 2026-10-18 BEST LEAD -- soccer soft-book vs Pinnacle (scripts/soccer_fd.py): S7 b365_pow3 dev CLV +2.57%
   (p<.001), ROI +9.8%/1,623, 6/7 seasons; S8 wh_pow3 CLV +2.81%, ROI +10.4%/910. Passed dev bar on CLV, but the
   power de-vig was chosen after seeing dev calibration -> needs the sealed 2019-20..2025-26 score (user's yes).
   Real-world: B365/WH limit winners; not available in California; ~230 B365 bets/yr across 22 leagues.
   S12 best3_pow3 (best of B365/WH/bwin): CLV +2.87% p<.001, ROI +8.5%/2,604 p .008 -> ~370 bets/yr.
   Sealed (one shot, on yes): `python scripts/soccer_fd.py sealed --confirm best3_pow3 [b365_pow3 wh_pow3]`.
   Recommend scoring best3_pow3 alone (fewer promotions = lower bar). tennis-data.co.uk blocks automation (403).
0. NEW 2026-10-18 -- Kalshi sports sealed test (one shot, 2026 events, ~13.5k with verified start times,
   leak-free prices built, nothing scored): candidates G1_fav85 (dev +2.0%/383 p .062; verified-start +1.9%/179)
   and G4_dog2050 (POST-HOC, dev +1.4%/4,429; verified +2.4%/3,462 p .12). Neither passed the dev bar (.0083).
   Command if user says yes: `python scripts/kalshi_longshot.py sealed --confirm G1_fav85 [G4_dog2050]`.
   ALSO maker_dog2050 (dev +9.8%/898 fills p .009, bar .0063; robust to fill rule; partly post-hoc; mostly
   spread capture -> fill realism is the risk). Sealed candles: `kalshi_maker.py build-sealed` (prices only);
   score on yes: `python scripts/kalshi_maker.py sealed --confirm`.
   10-18 symmetric ladder maker test: YES -2.6%, NO +2.7% (net ~0) -> spread capture does NOT survive
   adverse selection on ladders; maker_dog2050 is weaker than its +9.8% suggests.
0. Keep the Mac awake for kickoffs/22:00-local picks? (pmset wake schedule) -- see Live signal engine RISK.
1. Odds API decision (pre-registered outcome of the sealed test, no Class-B pass): buying ($119, 1 month,
   `ODDS_API_KEY=` in `sports/.env`) is a $119 test of ONE lead -- the NBA high-line under @DraftKings
   (+1.9% dev, untestable on ESPN data). User's call; do not frame it as a likely winner.
2. Older pending lockbox decisions: west_coast_night, div_unders_late, pre_rain_unders (causal, NFL 2020-25).
   NOTE for west_coast_night: its circadian MIRROR (west team at early ET kickoff) lost -10.6% on dev 10-16 --
   if body clock were the cause both halves should show it, so +21.5%/107 is likelier noise. Tell the user.

## Next steps (in order) -- a fresh session starts at the first unblocked one
1. Kalshi weather paper trading runs itself (launchd, see below). Each session: `cd ~/.claude/quant-kalshi &&
   .venv/bin/python scripts/kalshi_paper.py report` and note picks/P&L here. Judge after 4-6 weeks.
2. Sports, user's ideas: (a) DONE 10-16 parlays 0/2 (see above); (b) DONE 10-16 (see above; lockbox needs yes); (c) DONE 10-16 NFL: west_early_east -10.6%/110,
   den_altitude_home -7.4%/94 (both ns, wrong sign). MLB/NBA/NHL game tables have NO moneylines (all NaN), so
   location tests there need prices first (Odds API or another source).
3. Stocks: next ideas in the STOCKS section; add trials via TRIALS in stocks_bt.py (ledger counts them).
4. Polymarket US: check legal access for the user's state; if yes, scan its weather markets with the
   consensus-reject rule (paper only).
5. Kalshi real-money execution path (dry-run default, user-set caps, kill switch; key in sports/.env by user).
6. If user buys Odds API: time-stamped puller -> re-test DK-reference NBA under only.
7. live_edges.py scanner is session-only; low priority.

## Kalshi paper trader (durable)
2026-10-17 FIX #2: the 10-16 minute-based dispatch (:00-:04 -> pick, else settle) lost EVERY pick on 10-16 because
launchd ran late after sleep (19:09, 20:17, 21:10, 22:13). Now one `tick` per hourly run = pick each city in its
decision window 22:00-23:59 CITY-local (once per city-night, data/kalshi_paper_done.json) + settle. Each pick logs
late_min. Lesson: never infer which job to run from the wall clock when the scheduler can be late.
Paper so far: 1 pick (DEN 10-16, NO @0.99) settled WIN +0.01u. Real record starts 2026-10-17 evening.
2026-10-16 FIX: the old single 22:03-PT run looked up the day AFTER tomorrow for ET/CT cities (5/7 "no open event"
on 10-15; only DEN logged, 1 pick). Backtest decides at 22:00 CITY-local. Now `pick-due` runs hourly at :03 and
picks each city in its own 22:xx hour (NY/MIA/PHL 19:03 PT, CHI/AUS 20:03, DEN 21:03, LAX 22:03); log is deduped
by ticker. Paper record effectively starts 2026-10-16 evening.
launchd com.jonathanwang.quant-kalshi-paper: `tick` hourly at :03 from ~/.claude/quant-kalshi
(code copy + own venv; launchd can't read Desktop). Canonical log ~/.claude/quant-kalshi/data/kalshi_paper.jsonl,
run log ~/.claude/quant-kalshi/run.log. After editing kalshi_paper.py or sports/: `bash sports/scripts/install_kalshi_paper_agent.sh`.

## Live signal engine (durable, paper)
launchd com.jonathanwang.quant-live-signals: `live_signals.py tick` every 15 min from ~/.claude/quant-kalshi.
Logs ~/.claude/quant-kalshi/data/live_signals.jsonl (picks+settles), live_signals_eval.jsonl (every game evaluated),
live_signals.log (non-empty ticks only). Report: `cd ~/.claude/quant-kalshi && .venv/bin/python scripts/live_signals.py report`.
After editing live_signals.py / sports/: `bash sports/scripts/install_live_signals_agent.sh`. Add signals to SIGNALS.
Rain fires ~15-20x/season, so judging it takes seasons -- the engine's value is hosting MORE signals.
RISK (2026-10-17): the signal window is 10-50 min before kickoff, but launchd has run 10-70 min late while the Mac
sleeps (14 runs in ~21 h instead of ~84). Games will be MISSED (sample loss, not bias: sleep is unrelated to rain).
Fix needs the Mac awake at kickoffs (`sudo pmset repeat wake ...` or Amphetamine/caffeinate) -- USER's call.
First real NFL slate since install: Sun 2026-10-18; check live_signals_eval.jsonl after it.

## Dies with the session (restart by hand)
- Live paper loop: `cd sports && ../.venv_quant/bin/python scripts/live_edges.py loop --every 10` (run_in_background).
  Not running after a clear unless restarted. Flags/snapshots persist in data/live/.

## Recovery if the worktree is gone
- Code + notes + ledgers are on GitHub: `git -C <main repo> fetch` then
  `git -C <main repo> worktree add .claude/worktrees/quant-trader worktree-quant-trader`; venv: `.venv_quant` (python 3.14, pandas, sklearn, scipy, pyarrow, requests, yfinance, statsmodels).
- Data is NOT on GitHub: rebuild with `collect.py all --league <lg> --seasons ...` (cached, hours).
  A re-collected snapshot will NOT match `data/frozen/2026-10-14/MANIFEST.json` checksums (committed copy:
  sports/lab/frozen_manifests/2026-10-14.json) -- if it differs, sealed results are no longer comparable; say so.
- Trial counts that set significance bars: sports/lab/ledger.jsonl, lockbox.jsonl, reports/prop_loop_ledger.jsonl,
  reports/pattern_survivors.json + pattern_cells.csv (all committed).

## Detail notes (nested)
- notes/data.md — sources, coverage, frozen snapshot, known data quirks
- notes/results.md — every strategy tried + verdict (the ledger in words)
- notes/live.md — live scanner, execution plan, Kalshi API plan
- notes/protocol.md — how to run backtests honestly (lockbox, trial counts, freeze)
