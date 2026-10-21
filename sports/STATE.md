# quant-trader / sports — STATE (read this first; overwrite at the end of every work chunk)

Last updated: 2026-10-21 (HANDOFF before /clear -- read the HANDOFF block first). . Branch `worktree-quant-trader` (repo ttt5600/NeruoBatLab), worktree
`.claude/worktrees/quant-trader`. Run Python as `../.venv_quant/bin/python` from `sports/`.
Push: `git -C <worktree> -c credential.helper='!gh auth git-credential' push -q https://github.com/ttt5600/NeruoBatLab.git worktree-quant-trader`

## HANDOFF 2026-10-21 (user is clearing context; fresh session starts HERE)
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
- N1 day-of temp (wx_dayof.py): build bv0ajs1qh was on the LAST model (NBM) at 10-21 ~03Z; if the session died, re-run
  `../.venv_quant/bin/python scripts/wx_dayof.py build` (cached, resumes; Open-Meteo free tier ~10k loc-calls/day) then `dev --ledger`.
- MIDTERMS (election 11-03): run `scripts/xvenue_elections.py` every session; LA Senate arb candidate (R@Poly .93 + D@Kalshi .033)
  waits on user: is LA Senate on Polymarket US? Kalshi SENATELA-26 rule text says "Kentucky". TX Gov Abbott: market .72 vs
  poll-model ~.75 -> judgment bet <= 1% B (notes/risk.md).
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
