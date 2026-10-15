# quant-trader / sports — STATE (read this first; overwrite at the end of every work chunk)

Last updated: 2026-10-15. Branch `worktree-quant-trader` (repo ttt5600/NeruoBatLab), worktree
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
- Pattern mining (`scripts/pattern_mine.py discover`): 10 survivors in `lab/reports/pattern_survivors.json`.
  5 look like line-move price/line artifacts; plausible: NBA under high-line @DraftKings +1.9%/54k,
  NHL blocked-shots/saves/mid-line unders +6.7–13%. Overs heavily overpriced (NBA -11.9% vs unders -2.4%).
- Live paper scanner Pinnacle no-vig vs Kalshi game contracts: `scripts/live_edges.py loop|report`
  (runs only while a session keeps it alive; 0 flags at EV>=2% so far).

## WAITING ON USER
1. "yes" to run `python scripts/pattern_mine.py confirm --confirm` (one shot on sealed seasons).
2. Odds API key (5M plan $119/mo) in `sports/.env` as `ODDS_API_KEY=` -> then write time-stamped puller.
3. Older pending lockbox decisions: west_coast_night, div_unders_late, rain+wind unders (NFL 2020-25).

## Next steps (in order)
1. On approval: run pattern confirm; record result here + notes/results.md.
2. When key arrives: Odds API puller (NBA/NFL props 2023+, 5-min snapshots) -> re-test DK-reference consensus.
3. Make the live loop durable (code outside ~/Desktop: launchd cannot read Desktop) and add Kalshi paper execution.

## Detail notes (nested)
- notes/data.md — sources, coverage, frozen snapshot, known data quirks
- notes/results.md — every strategy tried + verdict (the ledger in words)
- notes/live.md — live scanner, execution plan, Kalshi API plan
- notes/protocol.md — how to run backtests honestly (lockbox, trial counts, freeze)
