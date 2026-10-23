# EDGES + AGENTS REGISTRY (read with STATE.md; last full rewrite 2026-10-22 ~09:30Z)

Every edge, lead, hypothesis and research agent from the quant-trader work, with numbers, status, files and how to reach
or re-run each agent. Status words mean exactly this:

- **PASSED SEALED**: passed dev, then passed a one-shot sealed/lockbox test on the user's yes. The only things that count as edges.
- **LIVE / PAPER**: running forward (real money or paper).
- **SEALED ASK**: passed dev; a pre-registered sealed test waits for the user's explicit "yes <name>".
- **LEAD**: interesting dev or descriptive signal, not past the bar. Needs a new pre-registered trial.
- **UNTESTED**: hypothesis written down, no test yet.
- **DEAD**: failed (dev or sealed). Don't redo without a new mechanism.

Protocol (notes/protocol.md): pre-register in the script docstring -> dev only -> ledger (lab/ledger.jsonl, 178 lines on
2026-10-22) with a Bonferroni bar per dataset -> sealed once on user yes -> paper -> small real. A dev number is never an edge.
Bet sizing: notes/risk.md (fractional Kelly on the conservative edge; scripts/ev_check.py checks any bet).

---------------------------------------------------------------------------------------------------------------------
## 1. EDGES THAT PASSED SEALED (real)

| # | Edge | Sealed result | Where it trades | Status / how to run |
|---|---|---|---|---|
| E1 | **Kalshi temp consensus-reject, orig 7 cities**: buy NO on daily-high brackets that all 5 forecast models miss by > 3F, decided ~22:00 local the night before | lockbox 2025Q3-2026Q4: +1.45% ROI, 1,844 bets, CI +10.9..+42.6u, p<.0001, 4/6 quarters + | Kalshi KXHIGH* | **LIVE on real $50 since 2026-10-22** ($1 units, 1 contract/order, $15/day). Paper: launchd L1 hourly. `~/.claude/quant-kalshi/.venv/bin/python ~/.claude/quant-kalshi/scripts/kalshi_exec.py status` |
| E2 | **X1 = same rule on 17 more daily-high cities** (KXHIGHT*) | sealed 2026-03..10: +0.78%, 2,464 bets, p .0047 | Kalshi | LIVE with E1 (cohort x1, 0.5% tier) |
| E3 | **Soccer S12 soft books vs Pinnacle**: best of B365/WH/bwin vs power-de-vigged Pinnacle pre-close, edge >= 3%, price < 10 | sealed 2019-26: CLV +3.63%, ROI +8.71%, 2,004 bets, p .010 | UK-style soft books; NOT Kalshi (0/296 quotes >= 1% there) | Not tradeable from CA. **User is often in Colorado**: DK/FanDuel/BetMGM are legal there in person -> NEXT: re-test S12 logic with US-book prices before trusting it (US books differ from B365/WH). scripts/soccer_fd.py |

Paper record 2026-10-22 07Z (E1+E2): 54 picks, 26 settled, 26 wins, +0.84u. Expected money: ~1% on $ risked; at $1 units
roughly +$3-5/month; at a $5k bankroll (1% = $50 orig7, $25 X1) roughly $75-150/month, lumpy (2 of 6 sealed quarters < 0).

## 2. LIVE / FORWARD PAPER

| # | What | Status | Script / check |
|---|---|---|---|
| F1 | NFL in-game maker fade of the post-big-play overshoot (1.8-pt overshoot found post-hoc) | forward paper on games 2026-10-22..11-30 (pre-registered); weekly `interim` Tuesdays; score ONCE after 2026-12-02 | scripts/nfl_ingame_fade.py |
| F2 | NFL pre-kickoff rain -> totals unders (dev +8.0%/219, p .11 after the join-bug fix) | forward paper via launchd L2 | scripts/live_signals.py report |
| F3 | Kalshi vs Pinnacle gap logger: any Kalshi game contract with EV >= 1% after fee | logging every 2 min since 10-19; judge after ~1 week | scripts/gap_logger.py report |
| F4 | Polymarket US weather consensus-reject (exact port of E1; dev -0.14%/283, p .53 -> too few bets, history is display prices) | paper logger built, NOT installed | scripts/venues_weather.py `paper tick` |
| F5 | Pick'em PK1: Sleeper legs where Pinnacle's conservative fair P beats 1/multiplier by >= 2 pts (pass: >= 200 legs, ROI > 0, p < .05) | logger built, NOT installed (0 qualifying legs in the first snapshot) | `bash scripts/install_pickem_logger_agent.sh` |
| F6 | Inference-vs-training thesis baskets (rules hash b745e0d3aa0e) | daily tracker built, NOT installed | `bash scripts/install_thesis_tracker_agent.sh` |
| F7 | Thesis v2 "HPC for AI agents": MEMORY / COMPUTE-INF / ENERGY-GEN / ENERGY-EQUIP sleeves (rules hash 7e822f587214, frozen 10-22, monthly EW) | same tracker logs v1+v2 (tracker_v2.jsonl), NOT installed; history hindsight-only | notes/thesis_v2.md; install script as F6 |

## 3. SEALED ASKS -- ALL THREE SCORED 2026-10-22 (user "do the sealed tests"), ALL FAIL; none open now

| # | Name | Sealed result | Lesson |
|---|---|---|---|
| S1 | O1 NBA simple model vs opener (gap >= .06; opener amended PRE-SCORE in commit 1c04d34: ESPN BET where present, else DraftKings, since ESPN BET ended 2025-12) | 554 bets: close-implied EV at open -2.55% [-4.1,-1.0] p .999 FAIL; realised ROI -10.2%; CLV +0.68 pts p .005 (ESPN BET subset +1.37, DK +0.50) | model still leads the opener but by half the dev size, far below ~4% overround; DK openers are sharper; gap>=.06 mostly picks longshots (avg dec 3.44). DEAD for betting |
| S2 | E2 Polymarket earnings cheap-YES (p <= .60) | 97 bets / 40 days: ROI -6.5% [-30,+19] p .71 FAIL | market calibrated; dev +11.4% was noise |
| S3 | C1 graph-smoothed momentum minus plain momentum | 46 months 2023-01..2026-09: -0.69%/mo t -1.96 FAIL (graph net -0.25%/mo, plain +0.44%/mo) | graph neighbours add noise; decay seen in dev continued negative |

NEW SEALED ASK (10-22):

| # | Name | Dev evidence | Sealed set | Say |
|---|---|---|---|---|
| S4 | **M3 Kalshi MENTION markets: buy NO when YES <= 15c** (24h primary, 6h secondary before the event anchor) | dev 2025 (<10-01): 24h +4.32%/293 bets/79 events p<.0003; 6h +2.99%/394 p .0027 (bar .0083); positive in all groups, 7/8 months; only 6 losses at 24h -> insurance-like (break-even loss rate ~6%) | ~3,600 events 2025-10..2026-10, pass p < .025; prices pulled first (`kalshi_mentions.py prices --sealed`, no P&L) | "yes M3" -> `sealed --confirm M3_d24 M3_d6` |

Older never-scored sealed candidates (none passed their dev bar; not recommended): Kalshi sports G1_fav85 / G4_dog2050 /
maker_dog2050 (kalshi_longshot.py, kalshi_maker.py); NFL lockbox west_coast_night / div_unders_late / pre_rain_unders.

## 4. LEADS + UNTESTED HYPOTHESES (each needs a NEW pre-registered trial)

NBA / NFL / sports
- L1 **Opening-line models generally**: public models (538, ESPN) lose to the CLOSE but were never tested vs the OPEN; ESPN openers exist for 2024-25 (nba_public_models.md s8). Same idea as O1.
- L2 **Healthy-roster + preseason ratings for single games** (user idea, 10-22): scripts/nba_matchup.py (opponent-adjusted ORtg/DRtg with healthy-core splits and date windows) and scripts/nba_fourfactors.py (four-factor matchup regression). DEN@OKC 10-22: four-factor OKC -8.3, P(DEN) .280 vs Kalshi .27 -> no bet by the pre-committed rule; preseason starters +/-48 (data/espn/nba/*_2027) DEN +7.6 vs OKC +2.0 -> P(DEN) .30-.35 depending on a judgment weight = lean only. To make it an edge: freeze one model + a threshold, log every game forward vs the Kalshi/Pinnacle price at a fixed time, grade CLV + ROI after ~200 games.
- L3 **NBA rebound unders** were the best prop lead (dev +2.7-2.8%) but **R_REB FAILED sealed** (-1.56%/1,214, p .72). Dead as specified.
- L4 NFL prop unders by category x game script on ESPN 2022+ props (research_prop_models #7), sealed 2025. UNTESTED.
- L5 Shrink P2 toward the market (it claimed +17 pts, delivered 0). UNTESTED.
- L6 Referee crews: TESTED 10-22 for TOTALS (notes/nba_referees.md, scripts/nba_referees.py, 7 trials, bar .00714).
  Crews move fouls/FTA (positive control p .002), but on points vs the close only excess R^2 .005 (p .017, fails), split-half
  r .22; walk-forward bet-by-crew -1.9%/2,222 (p .83) -> DEAD for game totals. Untested: crew FTA vs FT/points PROPS at the open.
  Same study (user ask "prove bias / fixing"): no home/favourite L2M error bias, no playoff series-extension effect (n 125),
  Donaghy-style line-move screen within chance; L2M errors favour the team trailing at the time 58% (p<.0001, PASS) --
  bias not fixing. Post-hoc: partly opportunity (trailing teams commit 57.5% of infractions), but within call type their
  infractions are missed +5.7 pp more often (CI 3.7-7.7) -> leniency toward whoever is behind. No betting use.
- L7 Situational variables for props (arena, b2b, travel, altitude, nightlife/"city effects", time zone): user wants a feature-correlation model, unders first (books shade overs: NBA prop overs -4..-7%). UNTESTED as a combined model.
- L8 Player-level simulation (RotoBot-style) LOST to the simple model everywhere (too much variance: implied SD 23.7 vs real ~12.5) and books already price this way (DraftKings deck). Only reopen with a historical injury feed (none free).
- L9 Soccer S12 at US books in Colorado (see E3). UNTESTED.
- L10 Kalshi earnings-call MENTION markets (the only legal US earnings contracts; Tesla's had 117k contracts). UNTESTED.


From prior-research sweep 2026-10-22 (notes/prior_research_2026-10-22.md; Burgi-Deng-Whelan Kalshi paper read in full):
- L19 K-FAV: Kalshi favourite side 88-97c as MAKER in non-sports daily categories (paper: >70c earn small + after fees;
  makers -9.6% vs takers -31.5%). TESTED 10-22 -> DEAD (see s5; notes/kalshi_fav_maker.md).
- L20 Maker execution for E1/E2 picks (resting NO vs taking the ask) -- price improvement vs adverse selection. UNTESTED.
- L21 PredictIt midterm NO-bundle sums > 110 (NBER abstract); verify current fees/caps first. UNTESTED.
- L22 Kalshi politics underconfidence (arXiv; contradicted by Burgi et al.) -- weak prior. UNTESTED.
- L23 Mention markets buy NO (SSRN via Quantpedia, UNVERIFIED) = L10 with a direction. UNTESTED.
- Also: Kalshi crypto vs Deribit implied probs; Kalshi parlays above leg product. Stock anomalies (overnight, pre-FOMC,
  index adds, PEAD large-cap, 0DTE) reported dead for retail.

From social/forum sweep 2026-10-22 (notes/social_edges_2026-10-22.md; Reddit unreadable by the tool, X barely indexed):
- L24 E1/E2 AS MAKER (strongest): weather series charge TAKERS only (checked via API 10-22); Becker 72M trades: weather
  takers -1.29% / makers +1.29% pre-fee; taker fee ~0.7% of stake at NO .90 vs +1.45% edge. Executor is a pure taker
  (IOC at ask). Test = replay picks vs trade history (fill + adverse selection), then live A/B. Same as L20.
  -> TESTED 10-22 DEV (scripts/kalshi_temp_maker.py, notes/kalshi_temp_maker.md, 9 trials bar .0056): resting NO bid at ask-2c or
     mid (to close) beats taker on 1,713 dev picks: M3_N10 +9.5u [+4.2,+14.4] p .0003, M2_N10 +8.1u p .001; ask-1c and
     rest-10h-then-take fail. All losers fill (adverse sel. 4.1%->5.7% loss rate). Sealed replay needs user yes; exec change spec in note.
- L25 Fade YES in single-name/Mentions/Entertainment/World (Bartlett-O'Hara: takers buy YES 60%, 31% settle YES) ~ L23.
- L26 Weather late-day lock-in from OBSERVED station highs after the daily peak (not the dead 06:00 forecast test).
- L27 Kalshi liquidity-reward programs (6,926 active) -- negligible at 1 contract; revisit at ~$5k.
- Warning: free forecast-ensemble weather bots are public -> E1 edge may compress; watch paper/live ROI.

Markets / stocks
- L11 **Earnings straddles**: one paper (Gao, Xing & Zhang 2018) found straddles 3 days before earnings +3.34%; our implied-vs-realised check (implied ~1.33x past mean move) is biased. Robinhood now gives daily option OHLC history + IV -> a real backtest is possible. UNTESTED. BE 10-22: implied +-10% vs historical 10.4-10.8% -> fair.
- L12 Earnings consensus logging: log daily consensus so a model can price each contract's actual threshold. Not started.
- L13 Inference-vs-training tilt (F6): demand evidence strong, compute share flat 29-39% (SemiAnalysis, secondhand); basket outperformance mostly SNDK + INTC. A forward record, not a backtest.
- L14 Company graph as a DATA product (12,917 dated customer filings, ~96% precision; data/company_graph/edges.parquet), e.g. "customer guided down -> suppliers reporting in 30 days" screens. Not an alpha.
- L15 ForecastEx (IBKR) weather: different stations (LGA etc.) and Weather Underground source -> a NEW trial of the E1 idea; needs an IBKR account for live quotes.

Information / social
- L16 X post counts vs Kalshi rumor markets: GDELT timestamps are late, so it can't rule out X leading. One-off test ~$13 (`rumor_signals.py xcounts --confirm-spend`) needs X_BEARER_TOKEN in sports/.env + user "yes, spend". Live X monitoring would cost ~$7k/month for 100 names.
- L17 RotoBot pick record: tracker built (scripts/rotobot_tracker.py); needs posts (X API ~$5/1k posts, or the user pastes posts into data/rotobot/posts.txt). Grade on CLV (140-370 picks resolve a 2% edge), not win rate.
- L18 Midterms (election 11-03): run scripts/xvenue_elections.py; LA Senate 2-venue cost .9752 (Kalshi rule text says "Kentucky" -> verify); TX Gov Abbott .70-.72 vs poll-model ~.75 = judgment bet <= 1% of B.

## 5. DEAD (do not redo without a new mechanism)

| Idea | Result | Report |
|---|---|---|
| **L19 K-FAV** Kalshi fav side 88-97c, non-sports daily (scripts/kalshi_fav_maker.py, dataset kalshi_fav_maker) | DEV 2021-08..2025-06: F1 taker -0.32%/4,419 p .72; F2 maker (bid 1c under ask) -2.19%/2,829 p .999. Adverse selection: filled win 90.5% vs unfilled 99.3% (breakeven 92.5%). No sealed ask | notes/kalshi_fav_maker.md |
| NBA in-game lag (H1-H3) and NFL in-game (H1-H3) | Kalshi reprices within ~1 min, beats ESPN/nflfastR WP | STATE CLOSED list |
| **NBA big leads** (user 10-22: "teams up 20 win more than priced") | true they win a lot (first 20-pt lead Q2-Q3 .941, 20+ at half .968, 11 seasons) but Kalshi charges ~that; DEV (86 playoff games) B1 -8.9%/37 p .93, B2 15/15 but exact p .71 and 14/29 ask=$1, B3 -4.3%/23; favourites 90-99c realised .893 vs mid .939; ESPN WP overconfident 2-3.5 pts | notes/nba_big_lead.md; scripts/nba_big_lead.py |
| **NBA late-game comeback calibration** (user: "Pacers down 3 at 23%") | Kalshi calibrated in 7/7 bins (369 games); T1 -1.5%/311 p .55, T2 -5.4%/331. Pacers 23c was at down 8 = fair | notes/late_game_calibration.md |
| Public NBA models vs close (538 Elo/CARM/RAPTOR, ESPN) | all worse than the close (log loss +.008..+.019); 0/7 rules; stacking ties the market | notes/nba_public_models.md |
| NBA rebound unders sealed (R_REB) | -1.56%/1,214, p .72 | ledger, STATE |
| NBA props P1 rolling form / P2 empirical distribution | P1 -3.3%/130k; P2 +0.27%/60.8k p .26 | results.md |
| Parlays | legs' negative edge compounds (-9.3%/105k) | results.md |
| Kalshi vs Polymarket arbitrage (sports, elections, weather) | within 0.5-1c; weather 0/48 brackets under $1 after fees | venues.md |
| Polymarket US game winners vs Pinnacle | 0/116 +EV, within .72c | soft_venues.md |
| Sleeper pick'em vs Pinnacle | 0/1,047 legs past break-even (Sleeper margin ~12.4%) | soft_venues.md |
| Earnings beat markets (Polymarket) | calibrated (price .732 vs 74.3% beat, 971 mkts); beat-rate models disagree only through threshold/GAAP mismatches | earnings.md |
| Rumors/news (GDELT) -> Kalshi rumor markets | price moves BEFORE news (median 4 h); 4/4 rules lose | rumor_signals.md |
| Twin stocks / pairs (dual-class, ETF twins, dual-listed, S&P screen) | 47 trials 0 pass; dual-class +3.8% gross -> -3.5% net | pairs_twins.md |
| Classic pairs KO/PEP + 6 | +1.89%/yr p .052, ~0 after 2009 | results.md |
| Customer-supplier momentum + earnings spillover (A1, A2, B1-B3) | all fail net of costs | company_graph.md |
| Kalshi soccer 1X2, CPI, gas, hourly temp, lows (X2), daily rain (R1 sealed fail), day-of temp (N1) | see STATE CLOSED list | results.md |
| 14 ETF rules (stocks_bt.py) | all failed | results.md |

---------------------------------------------------------------------------------------------------------------------
## 6. AGENTS: who did what, where the output is, how to reach them

How to reach: the IDs below are background subagents of the session "automated trading system development" (peer id
f9a9b4). While THAT session is alive, `SendMessage(to="<agent id>")` resumes the agent with its full context. **After a
/clear or a new session, treat them as unreachable**: re-launch a general-purpose Agent with the brief below (each
report's header also restates its brief). Their outputs are durable files (code + notes in git; data gitignored).

| Agent (ID) | Brief | Key findings | Outputs |
|---|---|---|---|
| Earnings beat EV research (a8e77c7f039f5ec13) | earnings calendar; BE; which contracts are +EV | BE 10-22 AMC, est EPS ~$0.65; P(beat EPS) ~78% / rev ~65% (judgment); no Kalshi beat markets (only MENTION markets); Polymarket beat markets intl-only + calibrated; E2 sealed ask; straddles fair | notes/earnings.md; scripts/earnings_poly.py, earnings_moves.py, earnings_kalshi_scan.py; data/earnings/ |
| NBA late-game calibration (af10a2fe8963b5ac5) | P(win given down x with t left) vs Kalshi | model beats ESPN WP but adds 0 over Kalshi; Kalshi calibrated; T1/T2 lose -> DEAD | notes/late_game_calibration.md; scripts/nba_late_wp.py |
| Kalshi trade automation (a4bd1e4141e2b1688) | production-harden the executor | idempotent client_order_id, reconcile, demo env, caps, loss stops, KILL, preflight/status/kill/smoke; 30 tests | scripts/kalshi_exec.py, tests/test_kalshi_exec.py, notes/go_live.md, scripts/install_kalshi_paper_agent.sh |
| RotoBot X tracker (adcdb9063c5f2364d) | find RotoBot, X access, grade picks | @rotobot_ai, paid app, no public record; X has no free reads; tracker built (CLV grading) | notes/rotobot_tracker.md; scripts/rotobot_tracker.py; tests/test_rotobot_tracker.py |
| Twin-company pairs (a3088c103e49d74d3) | dual-class/ETF/dual-listed twins + screen | 47 trials 0 pass, net negative -> DEAD | notes/pairs_twins.md; scripts/pairs_twins.py |
| Public NBA models vs market (a010c1d7bfadaa9d2) | DARKO/EPM/538/ESPN/RotoBot vs close | only 538 + ESPN archived; all lose to close; suggests the OPENER test | notes/nba_public_models.md; scripts/nba_public_models.py; data/fte_nba/ |
| Venue expansion + Polymarket connection (a676a8ee8d7f0a373) | all US venues; port E1; Polymarket US API | Polymarket US = own book, automated orders allowed (Ed25519 key), weather rule dev -0.14%; no arb; ForecastEx different stations; Robinhood/Coinbase/Webull route to Kalshi | notes/venues.md; scripts/venues_weather.py, polymarket_us.py |
| Pick'em/exchanges vs Pinnacle (ac7c557ba06b23cd3) | soft-price idea at pick'em + P2P exchanges | Sleeper 0/1,047, PMUS 0/116; CA legality (pick'em grey, sweepstakes banned); PK1 pre-registered | notes/soft_venues.md; scripts/pickem.py, pmus_vs_pin.py, install_pickem_logger_agent.sh |
| Player-sim game models (a518265a341bceffc) | latest research; simple vs player sim; vs open and close | simple beats sim; both lose to close; simple leads the OPENER -> O1 | notes/player_sim.md; scripts/nba_player_sim.py |
| Rumor/social signals (a2edacd4c35612780) | do X/Reddit/news lead prices? | GDELT: price leads news; 4 rules lose; X test ~$13 optional | notes/rumor_signals.md; scripts/rumor_signals.py; data/rumor/ |
| Company relationship graph (a1d8d0577727268bd) | EDGAR graph; customer momentum; spillovers; graph momentum | 6 trials 0 pass; C1 near-miss decaying; graph = data product | notes/company_graph.md, company_graph_lit.md; scripts/company_graph.py; data/company_graph/ |
| Inference-vs-training thesis (a431ad01a84c61944) | user thesis: inference > training | demand strong, share flat; tickers classified; baskets fixed; tilt 1-3% | notes/inference_thesis.md; scripts/thesis_tracker.py, install_thesis_tracker_agent.sh |
| Thesis v2 HPC-for-agents (10-22; 3 web sub-agents: memory, energy, compute) | user thesis update: memory + inference compute + energy, de-emphasize networking/training | evidence + bear cases cited; 4 sleeves frozen; compute corr 0.95 / memory 0.84 to SMH, energy-gen 0.49 = only diversifier; top-2 = 35-48% of each sleeve's gain | notes/thesis_v2.md (public), notes/thesis_v2_private.md (gitignored); scripts/thesis_tracker.py rules-v2/history-v2 |
| Claude Code docs lookup (af0e08030d14425c0) | auto-accept cross-session messages | `crossSessionInbound: accept` (now set in ~/.claude/settings.json) | (memory reference_robinhood_mcp_and_cross_session) |
| Prior academic research sweep (a21559133abdc2479, 10-22) | Kalshi/PM/sports/equity literature 2015-26 | Kalshi longshot bias + maker advantage; weather mkt beats NBM; retail stock anomalies dead; leads L19-L23 | notes/prior_research_2026-10-22.md |
| Robinhood strategies (ad99111bd1ca1bb20, 10-22) | strategies for the user's real book | deleverage first (margin-call risk), tax-lot selection, wash-sale traps, index core; covered calls/CSPs no | notes/robinhood_strategies.md (public), robinhood_strategies_private.md (gitignored), scripts/rh_strategies_bt.py |
| Social/forum edge sweep (a9ae4292efc10313e, 10-22) | X/Reddit/HN/GitHub/Substack claims | online claims mostly unverified vendors; trade-level studies agree maker>taker, NO>YES, favs>longshots; L24-L27 | notes/social_edges_2026-10-22.md |
| Kalshi mention-NO backtest (add08d9ac940e74b9, 10-22) | L23 buy NO on mention markets | M3 (YES<=15c) passes dev at 24h + 6h; M1/M2 fail; sealed ask S4 | scripts/kalshi_mentions.py, notes/kalshi_mentions.md |
| Earlier research agents (10-21, pre-this-session) | prop-model research; situational variables | see report headers (re-launch brief at the bottom of each) | notes/research_prop_models.md, notes/research_situational_vars.md |

Work done directly by the lead (no agent): R_REB sealed scoring (scripts/nba_props_dist.py `sealed --confirm R_REB`, run
once, FAIL); DEN@OKC single-game models (scripts/nba_matchup.py, scripts/nba_fourfactors.py); go-live of E1/E2 on $50.

### Peer Claude sessions (reach with SendMessage(to="<name>"); messages auto-accept since 2026-10-22)
- **connect-robinhood-mcp** [3cc68f]: has the `robinhood-trading` MCP loaded. Route every Robinhood request there,
  read-only unless the user says otherwise. The agent can trade ONLY the (empty) agentic account; the main account and
  the Roth are read-only (even review_* is FORBIDDEN), so main-account changes are tickets the user enters in the app.
  Its output: sports/notes/robinhood_snapshot.md (sections 1-7; personal, GITIGNORED, never commit).
- automated trading system development [f9a9b4] = the quant-trader lead session (this work).
- Others on the machine (not quant): separate-voc-noise-metrics [cd2253], long-lived agent architecture [d21638],
  loop-harness-pi-check [a8eb37].

### Launchd loops (non-Claude, keep running)
L1 quant-kalshi-paper (hourly :03; paper + LIVE executor), L2 quant-live-signals (15 min), L3 quant-gap-logger (2 min).
Deployed copy: ~/.claude/quant-kalshi (launchd cannot read ~/Desktop). Redeploy after code edits:
`bash sports/scripts/install_kalshi_paper_agent.sh`. Emergency stop: `touch ~/.claude/quant-kalshi/KILL`.

## 7. Personal account context
THIS REPO IS PUBLIC. Balances, leverage, positions, account suffixes and the deleveraging plan live ONLY in the
gitignored files sports/notes/private_account.md and sports/notes/robinhood_snapshot.md. Read them locally; never copy
their numbers into tracked files.

## 8. Open asks to the user
(O1/E2/C1 scored 10-22, all FAIL); revoke + replace the pasted Kalshi key (it is a PROD key named kalshi-demo.pem); fund
Kalshi (~$5k later); Polymarket US keys -> sports/.env; X test yes/no; install F4/F5/F6 loggers; free disk (97% full);
confirm home state (CA vs CO) for venue legality.
