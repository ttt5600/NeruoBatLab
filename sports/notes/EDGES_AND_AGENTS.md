**MASTER INDEX = `notes/FUND_BOOK.md` (2026-10-29): mandate, live ops, every edge by status, research system, tools,
lessons, pending asks. This file keeps the long-form detail (numbers, agent IDs, briefs); update both.**

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
| E4 | **E1/E2 executed as MAKER** (resting post-only NO bid at the mid, floored, <= ask-1c, until close; no taker fallback) | sealed 2025-07..2026-10, 4,308 picks: ROI +1.07% -> +2.22%, +30.6u [+26.2,+34.9] vs taker, p<.001; fill 80%; filled loss rate 3.07% vs 2.46%; both cohorts pass; +27.4u even with a maker fee | Kalshi | executor maker mode being implemented (KALSHI_EXEC_STYLE=maker); scripts/kalshi_temp_maker.py |
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
| S4 | **[SCORED 10-23: FAIL -- d24 +0.45%/2,674 p .22; d6 +0.72%/4,524 p .061; edge shrank as volume grew ~10x]** M3 Kalshi MENTION markets: buy NO when YES <= 15c** (24h primary, 6h secondary before the event anchor) | dev 2025 (<10-01): 24h +4.32%/293 bets/79 events p<.0003; 6h +2.99%/394 p .0027 (bar .0083); positive in all groups, 7/8 months; only 6 losses at 24h -> insurance-like (break-even loss rate ~6%) | ~3,600 events 2025-10..2026-10, pass p < .025; prices pulled first (`kalshi_mentions.py prices --sealed`, no P&L) | "yes M3" -> `sealed --confirm M3_d24 M3_d6` |

Older never-scored sealed candidates (none passed their dev bar; not recommended): Kalshi sports G1_fav85 / G4_dog2050 /
maker_dog2050 (kalshi_longshot.py, kalshi_maker.py); NFL lockbox west_coast_night / div_unders_late / pre_rain_unders.

## 4. LEADS + UNTESTED HYPOTHESES (each needs a NEW pre-registered trial)

NBA / NFL / sports
- L1 **Opening-line models generally**: public models (538, ESPN) lose to the CLOSE but were never tested vs the OPEN; ESPN openers exist for 2024-25 (nba_public_models.md s8). Same idea as O1.
- L2 **Healthy-roster + preseason ratings for single games** (user idea, 10-22): scripts/nba_matchup.py (opponent-adjusted ORtg/DRtg with healthy-core splits and date windows) and scripts/nba_fourfactors.py (four-factor matchup regression). DEN@OKC 10-22: four-factor OKC -8.3, P(DEN) .280 vs Kalshi .27 -> no bet by the pre-committed rule; preseason starters +/-48 (data/espn/nba/*_2027) DEN +7.6 vs OKC +2.0 -> P(DEN) .30-.35 depending on a judgment weight = lean only. To make it an edge: freeze one model + a threshold, log every game forward vs the Kalshi/Pinnacle price at a fixed time, grade CLV + ROI after ~200 games.
- L3 **NBA rebound unders** were the best prop lead (dev +2.7-2.8%) but **R_REB FAILED sealed** (-1.56%/1,214, p .72). Dead as specified.
- L4 NFL prop sides by category x game script: TESTED 10-25 DEV, **0/8 cells pass** (notes/nfl_props_script.md, scripts/nfl_props_script.py, ledger dataset nfl_props_script, bar .00625). Only 2024 has yardage props (ESPN BET, openers); 2022-23 = TD/INT/tackles only. Best: C7 low-total rush-yds U +7.7pp p .039 q .31; C3 dog RB rush U +9.5pp p .078. Blind unders ~0 vs de-vig, -6 to -13% ROI at 6.4% vig. DEAD as specified; no sealed ask.
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
  -> TESTED 10-25 DEV (scripts/wx_lockin.py, notes/wx_lockin.md, 6 trials bar .0083): DEAD as taker. At 17:00, 99.8% of
     brackets below the METAR max already have NO at $1.00 (6 bets in 4 yrs); ABOVE_2 460 bets -8.3% p 1.0 (market sees
     1-min peaks the hourly METAR misses). Maker trials NOT SCORED (tapes 2,076/11,176). Official max < METAR max on
     4/5,365 days (sensor spike, `$` flag, garbled METAR, 1 unexplained); LST climate day matters (+19 traps if local midnight).
- L27 Kalshi liquidity-reward programs (6,926 active) -- negligible at 1 contract; revisit at ~$5k.
- Warning: free forecast-ensemble weather bots are public -> E1 edge may compress; watch paper/live ROI.

NEW 2026-10-23/25 (from the user's live NBA checks):
- L28 NBA big-lead LIVE rule: leader up >= 25 in Q3 (or >= 20 in Q4 > 6 min) with Kalshi ask <= 0.96; history 99.3%
  (25-40 pt leads mid-Q3, 702 games) vs break-even ~96.3%. DAL up 31 @ 96c 10-23 (won). Dev Kalshi sample tiny ->
  forward paper via scripts/nba_live_scan.py --log; sealed 2025-26 in-game candles untouched (needs pre-reg + yes).
- L29 Miami thin-depth thesis: MIA under its closing spread over next 20 games (0-2, two home blowouts). Track forward.
- Scanner: scripts/nba_live_scan.py (Kalshi ask vs historical win rate in the same period/clock/margin; paper only).

Markets / stocks
- L11 **Earnings straddles**: one paper (Gao, Xing & Zhang 2018) found straddles 3 days before earnings +3.34%; our implied-vs-realised check (implied ~1.33x past mean move) is biased. Robinhood now gives daily option OHLC history + IV -> a real backtest is possible. UNTESTED. BE 10-22: implied +-10% vs historical 10.4-10.8% -> fair.
- L12 Earnings consensus logging: log daily consensus so a model can price each contract's actual threshold. Not started.
- L13 Inference-vs-training tilt (F6): demand evidence strong, compute share flat 29-39% (SemiAnalysis, secondhand); basket outperformance mostly SNDK + INTC. A forward record, not a backtest.
- L14 Company graph as a DATA product (12,917 dated customer filings, ~96% precision; data/company_graph/edges.parquet), e.g. "customer guided down -> suppliers reporting in 30 days" screens. Not an alpha.
- L15 ForecastEx (IBKR) weather: different stations (LGA etc.) and Weather Underground source -> a NEW trial of the E1 idea; needs an IBKR account for live quotes.
  -> TESTED 10-28 DEV (scripts/forecastex_wx.py, notes/forecastex_wx.md, 2 trials bar .025): T1 D-1 last trade 159 bets
     +20.9u ROI +13.2% p .0004; T2 day-high price +14.4u p .001 -- but n < pre-reg floor 500 (= not resolvable), 128/159 bets
     in Jul-Aug (D-1 trading ~absent May-Jun), trade prices not quotes (mean cost .88). WU max 1-2F below NWS on 64% of days.
     SEALED ASK FX1 (08-15..10-26) awaits user yes; IBKR account only after.

Information / social
- L16 X post counts vs Kalshi rumor markets: GDELT timestamps are late, so it can't rule out X leading. One-off test ~$13 (`rumor_signals.py xcounts --confirm-spend`) needs X_BEARER_TOKEN in sports/.env + user "yes, spend". Live X monitoring would cost ~$7k/month for 100 names.
- L17 RotoBot pick record: tracker built (scripts/rotobot_tracker.py); needs posts (X API ~$5/1k posts, or the user pastes posts into data/rotobot/posts.txt). Grade on CLV (140-370 picks resolve a 2% edge), not win rate.
- L18 Midterms (election 11-03): run scripts/xvenue_elections.py; LA Senate 2-venue cost .9752 (Kalshi rule text says "Kentucky" -> verify); TX Gov Abbott .70-.72 vs poll-model ~.75 = judgment bet <= 1% of B.

### NEW 2026-10-25: comebacks / matchups / coaching / refs (user after DEN 23-pt comeback; NFL + NBA)
Agents: NFL study (notes/nfl_comebacks_matchups.md, scripts/nfl_comebacks.py; dev 2006-2019 + nfl_ingame dev 2025 wk1-12),
NBA study (notes/nba_comebacks_matchups.md, scripts/nba_comebacks.py, results/nba_comebacks.json; dev 2015-16..2024-25),
slate card (notes/slate_2026-10-25.md: 0 flags, in-game cheat sheets). Re-launch with the brief at the top of each note.
- NFL: 70 tests, 0 survive. vegas_wp calibrated within ~1pp; "better team comes back" +2.6pp/SD is mostly the spread and
  Kalshi already prices it (+3.9pp over vegas_wp for trailing favs). PROE/pace/4th-down/coach/head-ref flags: nothing.
  Leads: CB1 fade trailing dogs in-game (Kalshi +1.8pp over vegas_wp, ~1% after fees -> too small to test);
  bad-D x bad-D unders -0.79 pts/SD p .0049; offence EPA ATS +0.56 p .04.
- NBA: sides efficient (net rating +5.7 margin, -0.03 ATS). Comebacks depend only on deficit x time x spread (0/69 trait
  cells). ESPN WP under-rates trailers 30-50% (Kalshi is calibrated, so not an edge). Survivors over all 152 tests:
  **L30 P-PAINT** high paint-pts-allowed teams -0.54 ATS/SD (p 2.5e-5, 10/10 seasons) -> feed into L2 game model;
  **L31 P-TOT3** season 3P% (own/allowed) -> unders, bigger spreads -> overs; walk-forward 1,417 bets 53.6% ROI +2.3%
  (chosen post hoc, not sig) -> forward paper; sealed 2025-26 only on user's yes. Timeout-after-run = 0 (naive -0.92 was selection).

### NEW 2026-10-25b: SIMPLE player+style models (user: "simple model, players, playing style; not xgboost")
- **L32 NBA totals vs OPENER** (scripts/nba_simple_model.py, notes/nba_simple_model.md; dev 2023-24/2024-25, 8 trials):
  team ridge + player value (box-regressed +/-) x projected minutes + 5 style terms. Model >= 3 pts off opening total:
  1,190 bets 55.5%, no-vig ROI +8.7% [+3.1,+14.3] p .001, both seasons + both opener books (DK/ESPN BET). Fails
  whole-ledger bar (.00024); vs CLOSE only 52.4% -> stale-opener edge. Expect ~+3-4% true. SEALED ASK: frozen rule
  once on 2025-26 (results/nba_simple_model_frozen.json) -- needs user yes; ~25% power. FORWARD: log first-posted totals.
  Spreads: CLV +0.79 pts vs open (p<1/4000) but 52.1% ATS -> same as O1, no money. Absences: CLV +1.75 vs +1.03.
  Style: paint-D -1.1 pts/SD sides; 3P% for/against -1.9/-2.3 totals; pace/ORB/TOV/3PA volume = 0.
  **SEALED 2026-10-25 (user "yes L32"): FAIL** -- 575 bets 2025-26, no-vig ROI +6.4% [-1.6,+14.4] p .054 (bar .05); 53.2% vs
  open, -110 ROI +1.6%, CLV +1.20 pts, 48.2% vs close; DK +6.0%/500, ESPN BET +9.1%/75; 480/575 overs; months -10.8..+26.8u.
  Same sign/size as dev but not significant (pre-registered ~25% power). Never re-score. Continues as FORWARD paper via
  scripts/nba_open_totals.py (CLV is the leading indicator). Flag: box_2026 hash not in frozen MANIFEST (re-pulled).
- **L33 NFL simple model** (scripts/nfl_simple_model.py, notes/nfl_simple_model.md; SBR openers 2007-19, 14 trials):
  pass/run EPA + QB + style OLS. CLV vs open spread +0.76 / total +0.81 pts (6/6 pass); ROI at -110 none pass (totals C
  +6.6% p .014). QB-change weeks spread CLV +1.15. Matchup interactions all null. Live `week` output disagrees with
  current lines on 11/14 (fit on 2007-19; totals 3.1 under) -> do NOT bet live output until refit; sealed 2020-25 on yes.

- **L34 NHL simple model vs OPENER** (scripts/nhl_simple_model.py, notes/nhl_simple_model.md; ESPN BET opener, dev
  2023-24/2024-25, 8 trials bar .00625): ridge + starting-goalie GSAA + b2b + shots/PP/sh% -> Poisson. CLV passes (ML
  +1.04 prob pts, totals +0.073 implied goals, p<1/4000, both seasons, not blind drift) but NO money: ML -1.1% offered,
  totals +7.1% no-vig p .0125 (fails; ~half is a blind 6.5-under lean) / +2.6% offered; close-implied EV at open -1.8%.
  Goalie-change games CLV +1.23 vs +0.98, ROI ns. DEAD as a bet at ESPN BET (4% overround). No sealed ask.

- **NFL leads vs OPENER (loop 1, notes/nfl_leads_open.md, scripts/nfl_leads_open.py, dev 2010-19, 10 trials bar .005):**
  bad-D x bad-D unders DEAD (214 bets, CLV -0.16, -7.7% at open, both halves negative; market raises these totals).
  Offence-EPA: margin slope vs open +0.98 pts/SD p .0003 (real) but rule |z|>=1 994 bets 51.5%, -1.6% at -110 -> NO BET;
  mostly already inside model B (+0.71/SD incremental p .0077, fails).

- **L6b ref crew FT -> props (loop 3): DEAD.** notes/nba_ref_ft_props.md; 4 trials; crew effect real (+0.8 FTA/z) but lines move toward crew (+0.135/z open->close); overs -8.1%/2,174; no FT-made props in dev.

- **L35 Kalshi early NFL prices (loop 5, notes/kalshi_early_nfl.md):** first-listed KXNFLGAME prices move +3.3c toward the out-of-sample legacy model by kickoff (109 games, p<.0002). Maker at first bid adversely selected; taker at first ask -4.5% (59c placeholder spread). NEXT: forward-log the first tight quote (spread<=10c) vs model.

- **L36 CFB totals vs opener (loop 7, notes/cfb_simple_model.md):** CLV +1.02 pts, no-vig +7.4% p .0004 (passes), -110 +2.6% (fails). Spreads model miscalibrated (v2 needs fresh prereg). Cross-sport: opening TOTALS are the soft spot (NBA L32, NFL L33, NHL L34, CFB L36).

- **Kalshi early NFL totals (loop 8): not stale** (notes/kalshi_early_nfl_totals.md; 3 trials fail; first tight ladder ~6 days pre-kick already near close). Low-vig route for the totals-opener edge must be elsewhere (sportsbook openers / exchanges).

- **L28 NBA big-lead buy: SEALED FAIL 2026-10-26** (results/nba_l28_sealed.json): 15 bets 13-2, -8.6%; 253 trigger games never <= .96 (99.6% wins); cheap leaders = adverse selection. DEAD.
- **L37 NBA mid-game comeback rule** (scripts/nba_midgame_scan.py, frozen 2c29645): forward paper live; sealed prep in progress (user yes).

- **L37 NBA mid-game comeback rule: SEALED FAIL 2026-10-26** (results/nba_l37_sealed.json): 477 bets -3.9c/contract p .97; Kalshi in-game sharper than model (Brier .1284 vs .1319). Descriptive: trailer longshot bias ~1.5c in 0-15c bins -> forward-only L38 candidate (fade trailing longshots), fee-limited.

- **NFL recency/overreaction (10-27, notes/nfl_recency.md, 12 trials): DEAD.** Fades 48.7-51.7% ATS; line drifts toward last-week coverer open->close (+0.275/SD p 3e-13, EPA-driven = sharp) but outcomes don't reward it.

- **NBA shooting luck (10-27, notes/nba_luck.md, 14 trials): no bet.** Hot 3P -> slight unders (-0.31/luck pt p .002, 9/10 seasons) but quintile bet -1.1%/4,719; early-season +3.7% p .18; cold-cold-over seed only 5/10 seasons; already inside the model (L31 terms). Sheet luck column = context only.

- **Kalshi winter weather: monthly rain/snow ladders, unreachable-NO (10-27, notes/kalshi_wx_winter.md, scripts/kalshi_wx_winter.py, dataset kalshi_wx_winter, 6 trials, bar .0083): NO PASS.** ACIS reproduces 1,259/1,259 settlements. DEV best W1m rain NO maker +3.6%/146 bets, 1 loss, market-null p .016 (1/city-month .12); taker W1 +1.6%/167 p .10; snow S1/S1m 36/30 bets 0 losses, p .35/.24; locked-YES never trades (ask $1.00). Lead L39 = forward-log W1m this winter, then one sealed shot (rain Jul26-Jan27 + snow) ~Feb 2027. Sealed now = underpowered, not asked.

- **Weather confidence-based Kelly sizing (10-27, notes/kalshi_sizing.md): NOT ADOPTED.** Per-pick confidence beyond price is unmeasurable (features add nothing); YES hits ~0.69x the NO-ask-implied rate (sealed 0.665) -> Kelly ~ flat 4.4% of bankroll/pick (k .25, 80% LCB); Kelly sizing did not beat flat in dev backtest. Lever = flat unit size + maker.

- **Small/long-tail market calibration scan (10-27, notes/smallmarket_calibration.md, scripts/smallmarket_*.py; EXPLORATORY, no trials):**
  Q1 "small = less efficient" is a LOOK-AHEAD artifact of lifetime volume (upsets attract volume); ex-ante Kalshi 24h volume shows no
  gradient, only wider spreads (3c -> 12c). Longshot bias +1.5-4pp venue-wide but = crossing cost (FAV mid +2-4.5%, net ~0).
  Polymarket negRisk sums median 1.01-1.02; "overround legs" +10-14% is a DISPLAY-PRICE artifact (live median spread 63c).
  Worst calibrated: mentions (both venues, YES +4-7pp) and PM culture/awards. Leads: H1 PM mention+culture NO at real ask (~4.5k
  legs/~90 days forward; tens of $/day), H2 PM longshot-NO as MAKER via data-api tape on the untouched 2026-05..10 holdout.
  Holdout 2026-05-01..10-20 never priced -> reserve for pre-registered tests.

- **L40 Kalshi Liquidity Incentive Program on weather brackets (10-27, notes/kalshi_lip.md):** $20/bracket/period (~1.6 d) on all 48 daily temp series; 75% of temp brackets pay nobody (YES side < 1,000); 1,000 YES @1c (~$10) could claim up to ~$10/period vs fill cost ~$0.5-1.3/bracket-night. 10-day PAPER test 10-29..11-08 pre-registered; program ends 2027-01-01.

- **L38 in-game trailing-longshot fade (10-27, notes/ingame_longshot.md, 8 trials): DEAD.** 3,051 dev snapshots NBA/NFL/NHL; L37-sealed 1.5c bias does not replicate (15-25c trailers +1.6c UNDER-priced); taker -0.2..-3.6c/contract; maker n.s. Weak: NHL 1-5c trailers 0/38. In-game Kalshi = efficient; live board kept as decision support/recorder only.

- **NBA live totals model vs Kalshi KXNBATOTAL (10-27, notes/nba_live_totals.md, 3 trials): DEAD.** Kalshi Brier .1196 < model .1240; rule -2.1%/336; close-Q4 -41.5%/46. Game pace carries ~no signal beyond pregame total/time/margin. Lead only: overs realised 2-6 pts below Kalshi mid in 5-55c bins (not prereg).

- **E5 ForecastEx weather consensus-reject: SEALED PASS-UNDERPOWERED 2026-10-28** (user "yes ForecastEx"; results/forecastex_wx_sealed.json): 219 bets 10 losses (4.6% vs b/e 10.2%), +15.05u, ROI +6.9% p .0004; T2 high-trade +5.5% p .0036; halves +4.3/+10.8. Caveats: last-trade (not quote) prices, thin (median 29/contract), <500 floor. WU settlement (independent of Kalshi NWS). NEXT: forward paper on live IBKR bid/ask (needs IBKR account).

- **L41/Q003 Kalshi Mentions maker NO (10-28, notes/q003_retail_maker.md, 4 trials):** IN1_d6 859 fills +7.08% p<.0002 (LEAD), MID_d6 +5.8%; d24 DEAD (2nd half negative). Adverse selection small (0.3-1.0c). DECAYING: halves +11.5%/+2.9% (cf. M3 death). Sealed ask: IN1_d6 on events >= 2026-06-25 (needs user yes).

- **NFL kicker over 3.5 "90%" system (social, 10-29, notes/nfl_kicker_system.md): DEAD.** 84.7% hit (n 354) vs 83.0% base; matched -0.7pp; needs ~-555 odds; ESPN BET lines open 4.5-9.5 and price 4+ at 89% (main-line over +0.6%/52).

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
| Econ consensus-reject 10-28 (claims, U3, payrolls, GDP vs GDPNow, core PCE; scripts/econ_reject.py) | DEV: T1 k=3 33 bets 0 losses +8.6% but market-price null p .08 (driven by 2021 revised-vintage claims); tails +2.7%/27 null p .48; T2 k=2 p .030 FAIL. Ladders don't quote stale tails -- family closed | econ_reject.md |
| 14 ETF rules (stocks_bt.py) | all failed | results.md |
| E1/E2 decided EARLIER on D-1 (11/14/18:00 local, lead-2 mixed forecasts, margin 3/5; scripts/wx_timing.py, dataset wx_timing, 12 trials, 10-27) | 0/6 taker pass, all deltas vs 22:00 negative (-2.1..-15.2u); NO 4-5c cheaper earlier but loss rate 7.6% vs 4.1%; 6 maker trials NOT SCORED (tapes stopped) | notes/wx_timing.md |
| NHL PDO/shooting luck + blind unders by opening total (scripts/nhl_luck.py, dataset nhl_luck, SBR 2007-08..2022-23 + NHL API shots) | 0/17 pass. Under at open 6.5 -4.3%/1,160 (L34 +5.8% no-vig lead does not replicate); luck bets lose ~vig; close prices ~90% of luck; totals drift TOWARD luck open->close | notes/nhl_luck.md |

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
| Venue expansion 2 (10-28; ForecastEx sub-agent acdb9da8c46a1675e) | PMUS trading client; Kalshi-vs-PMUS map; ForecastEx feasibility | PMUS client built (read-only works; trading gated by --live + keys; 13 mocked tests); 6/24 weather cities overlap 48/48 brackets, PM depth >= Kalshi; ForecastEx 18/24 same airports, WU settlement, $0.01/contract fee, public daily price files since 2026-03 -> proxy backtest possible without account | notes/venue_expansion.md; scripts/venues/; tests/test_polymarket_us_client.py |
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
