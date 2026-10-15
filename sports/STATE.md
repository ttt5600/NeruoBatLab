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
   User said (2026-10-15) they will clear context and then say yes. RUNBOOK when they do:
   - `cd sports && ../.venv_quant/bin/python scripts/pattern_mine.py confirm --confirm` (~5-15 min; reads
     data/frozen/2026-10-14, sealed rows only; writes lab/reports/pattern_confirm.jsonl; refuses to run twice).
     Code path verified 2026-10-15 on discovery rows: all 10 survivors reproduce their dev ROI exactly.
   - Bar: one-sided p < 0.005 (0.05 / 10 survivors) AND sealed ROI > 0.
   - PRE-REGISTERED reading (written before any sealed number existed):
     Class A, suspected price/line-mismatch artifacts (a pass here is NOT an edge): the 5 cells containing
       `line_move` (mlb hits over/down, nba under >2.3/up, mlb over down+drifted, mlb K under/up, nfl tackles under/up).
     Class B, plausible: nba under high-line @DraftKings (expect ~0 sealed bets: sealed NBA is ESPN BET only ->
       UNTESTABLE here), nba under >2.3 & high line, nhl blocked-shots under 1.8-1.87, nhl under 1.6-1.8 mid,
       nhl saves under 1.6-1.8.
   - PRE-REGISTERED Odds API decision: if any Class-B cell passes -> recommend buying ($119, 1 month) to re-test it
     on time-stamped multi-book prices before paper trading. If none pass -> the only open lead is the DK under
     (untestable on ESPN data); buying is then a $119 test of that single lead -- present it as the user's call.
   - Afterwards: commit pattern_confirm.jsonl, add the verdict to notes/results.md, update this file, push.
2. Odds API key (5M plan $119/mo) in `sports/.env` as `ODDS_API_KEY=` -> then write time-stamped puller.
3. Older pending lockbox decisions: west_coast_night, div_unders_late, rain+wind unders (NFL 2020-25).

## Next steps (in order)
1. On approval: run pattern confirm; record result here + notes/results.md.
2. When key arrives: Odds API puller (NBA/NFL props 2023+, 5-min snapshots) -> re-test DK-reference consensus.
3. Make the live loop durable (code outside ~/Desktop: launchd cannot read Desktop) and add Kalshi paper execution.

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
