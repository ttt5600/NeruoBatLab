# quant-trader / sports — STATE (read this first; overwrite at the end of every work chunk)

Last updated: 2026-10-17 am (Kalshi paper 2nd timing bug fixed: late launchd runs; stocks 3 more trials 0/3). Branch `worktree-quant-trader` (repo ttt5600/NeruoBatLab), worktree
`.claude/worktrees/quant-trader`. Run Python as `../.venv_quant/bin/python` from `sports/`.
Push: `git -C <worktree> -c credential.helper='!gh auth git-credential' push -q https://github.com/ttt5600/NeruoBatLab.git worktree-quant-trader`

## Goal
Make money legitimately. User upgrades plan if we prove > $100/month. Path agreed (2026-10-14):
**confident backtest on frozen data -> buy Odds API (time-stamped) -> paper-trade fake units live -> real money, small caps.**

## Hard rules (never break)
- Sealed/lockbox seasons are scored ONCE and only on the user's explicit "yes". Never re-score.
- Never ask for / handle brokerage or Kalshi credentials. User puts keys in `sports/.env` themselves.
- Never bypass geo-blocks (Pinnacle league-info endpoint is US-blocked; odds endpoints answer — fine).
- Push CODE to GitHub, never data (`sports/data/` is gitignored).
- Automated execution only via Kalshi official API (sportsbooks have no API; automating them breaks ToS).

## Where things stand
- Only strategy past a sealed test: Kalshi temp "consensus-reject" (+1.45% ROI, 1,844 bets). Paper trader: `scripts/kalshi_paper.py`.
- NFL/MLB game-line situational edges: priced in or too small (see notes/results.md).
- Frozen snapshot `data/frozen/2026-10-14/` (sha256 MANIFEST). Sealed: nba 2026, nhl 2026, mlb 2025, nfl 2025.
- Pattern mining: 10 dev survivors -> SEALED CONFIRM RUN 2026-10-15 (done, never re-run): 0/10 pass.
  `lab/reports/pattern_confirm.jsonl`. mlb hits over/line-down dev +36% -> sealed -5.4%/3,655 (t=-4.4):
  the line-move artifact reading was right. NHL unders: +2.8%, -2.3%, -9.4% -> noise. NBA DK under: 0 sealed
  bets (sealed NBA = ESPN BET only) -> still UNTESTED, the only open prop lead.
  Overs heavily overpriced in dev (NBA -11.9% vs unders -2.4%) -- descriptive, not a strategy.
- Live paper scanner Pinnacle no-vig vs Kalshi game contracts: `scripts/live_edges.py loop|report`
  (runs only while a session keeps it alive; 0 flags at EV>=2% so far).

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

## AUTONOMOUS LOOP (user 2026-10-17: "keep searching ... loops while I'm out ... only backtest matters")
Done 10-17 (all dev, all in ledger): NFL online angles 0/5; MLB fav-at-open (SBRO 2010-18) -2.7%;
NBA (new data: data/nba_odds/OddsData.sqlite, 2007-26, SPREAD COLUMN IS UNSIGNED -> sign from ML fav)
angles 0/5 + Elo/logit models 0/2. Running: scripts/kalshi_longshot.py build (all Kalshi sports markets,
1 h pre-start price) -> then `dev --ledger` (5 pre-registered trials). kalshi_longshot sealed is BLOCKED
until 2026 start times come from schedules (Kalshi changed expiration offsets in 2026).
Idea queue: Kalshi maker-side (post bids on longshots); NBA totals model; NHL (no odds data yet);
college (needs collegefootballdata key from user).

## WAITING ON USER
0. NEW 2026-10-18 -- Kalshi sports sealed test (one shot, 2026 events, ~13.5k with verified start times,
   leak-free prices built, nothing scored): candidates G1_fav85 (dev +2.0%/383 p .062; verified-start +1.9%/179)
   and G4_dog2050 (POST-HOC, dev +1.4%/4,429; verified +2.4%/3,462 p .12). Neither passed the dev bar (.0083).
   Command if user says yes: `python scripts/kalshi_longshot.py sealed --confirm G1_fav85 [G4_dog2050]`.
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
