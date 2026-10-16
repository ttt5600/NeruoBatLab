# quant-trader / sports — STATE (read this first; overwrite at the end of every work chunk)

Last updated: 2026-10-16 pm (weather JOIN BUG found+fixed; corrected rain edge +8% p=.11; live_signals.py paper engine on launchd). Branch `worktree-quant-trader` (repo ttt5600/NeruoBatLab), worktree
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
- Next stock ideas (each = new ledger trials): cross-sectional momentum/short-term reversal on sectors,
  supplier->customer lead-lag (Cohen-Frazzini economic links) needs a supplier graph we don't have,
  vol-targeting with loss cap (risk control, not alpha). Baseline truth: index fund ~7%/yr = $100/mo on ~$17k.

## WAITING ON USER
1. Odds API decision (pre-registered outcome of the sealed test, no Class-B pass): buying ($119, 1 month,
   `ODDS_API_KEY=` in `sports/.env`) is a $119 test of ONE lead -- the NBA high-line under @DraftKings
   (+1.9% dev, untestable on ESPN data). User's call; do not frame it as a likely winner.
2. Older pending lockbox decisions: west_coast_night, div_unders_late, pre_rain_unders (causal, NFL 2020-25).

## Next steps (in order) -- a fresh session starts at the first unblocked one
1. Kalshi weather paper trading runs itself (launchd, see below). Each session: `cd ~/.claude/quant-kalshi &&
   .venv/bin/python scripts/kalshi_paper.py report` and note picks/P&L here. Judge after 4-6 weeks.
2. Sports, user's ideas: (a) 2-leg parlay test on existing prop/game data (does combining two +EV-looking legs
   survive the compounded hold?); (b) DONE 10-16 (see above; lockbox needs yes); (c) venue-location features (altitude, travel, timezone) on NFL/MLB/NBA dev.
3. Stocks: next ideas in the STOCKS section; add trials via TRIALS in stocks_bt.py (ledger counts them).
4. Polymarket US: check legal access for the user's state; if yes, scan its weather markets with the
   consensus-reject rule (paper only).
5. Kalshi real-money execution path (dry-run default, user-set caps, kill switch; key in sports/.env by user).
6. If user buys Odds API: time-stamped puller -> re-test DK-reference NBA under only.
7. live_edges.py scanner is session-only; low priority.

## Kalshi paper trader (durable)
launchd com.jonathanwang.quant-kalshi-paper: pick 22:03 / settle 10:07 daily from ~/.claude/quant-kalshi
(code copy + own venv; launchd can't read Desktop). Canonical log ~/.claude/quant-kalshi/data/kalshi_paper.jsonl,
run log ~/.claude/quant-kalshi/run.log. After editing kalshi_paper.py or sports/: `bash sports/scripts/install_kalshi_paper_agent.sh`.

## Live signal engine (durable, paper)
launchd com.jonathanwang.quant-live-signals: `live_signals.py tick` every 15 min from ~/.claude/quant-kalshi.
Logs ~/.claude/quant-kalshi/data/live_signals.jsonl (picks+settles), live_signals_eval.jsonl (every game evaluated),
live_signals.log (non-empty ticks only). Report: `cd ~/.claude/quant-kalshi && .venv/bin/python scripts/live_signals.py report`.
After editing live_signals.py / sports/: `bash sports/scripts/install_live_signals_agent.sh`. Add signals to SIGNALS.
Rain fires ~15-20x/season, so judging it takes seasons -- the engine's value is hosting MORE signals.

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
