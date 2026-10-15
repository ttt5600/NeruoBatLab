# quant-trader / sports — STATE (read this first; overwrite at the end of every work chunk)

Last updated: 2026-10-15 (after sealed pattern confirm). Branch `worktree-quant-trader` (repo ttt5600/NeruoBatLab), worktree
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

## WAITING ON USER
1. Odds API decision (pre-registered outcome of the sealed test, no Class-B pass): buying ($119, 1 month,
   `ODDS_API_KEY=` in `sports/.env`) is a $119 test of ONE lead -- the NBA high-line under @DraftKings
   (+1.9% dev, untestable on ESPN data). User's call; do not frame it as a likely winner.
2. Older pending lockbox decisions: west_coast_night, div_unders_late, rain+wind unders (NFL 2020-25).

## Next steps (in order)
1. Kalshi temp consensus-reject paper trading (the only passed strategy) -- keep `kalshi_paper.py` running,
   accumulate paper P&L; this is the realistic path to the >$100/month bar.
2. Make the live loop durable (code outside ~/Desktop: launchd cannot read Desktop) and add Kalshi paper execution.
3. If user buys Odds API: time-stamped puller (NBA props 2023+, 5-min snapshots) -> re-test DK-reference under only.
4. Fix pattern_mine.py label: confirm prints two-sided p under a "one-sided" header (cosmetic; verdict unaffected).

## Dies with the session (restart by hand)
- Live paper loop: `cd sports && ../.venv_quant/bin/python scripts/live_edges.py loop --every 10` (run_in_background).
  Not running after a clear unless restarted. Flags/snapshots persist in data/live/.

## Recovery if the worktree is gone
- Code + notes + ledgers are on GitHub: `git -C <main repo> fetch` then
  `git -C <main repo> worktree add .claude/worktrees/quant-trader worktree-quant-trader`; venv: `.venv_quant` (python 3.14, pandas, sklearn, scipy, pyarrow, requests).
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
