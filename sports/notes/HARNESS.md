# HARNESS -- game-day paper loggers + headless digest under launchd (deployed 2026-11-01)

Why a deployed copy: launchd cannot read ~/Desktop (macOS TCC: bash exits 126, python cannot list it), and the old
session shell runners (scripts/week_runner.sh, scripts/prop_runner.sh) died with the session or with sleep. They were
stopped 2026-11-01 ~18:46 PT after the launchd versions were verified writing.

## What runs where
Deployed root: `~/.claude/quant-sports` (own venv `.venv`, python 3.14, packages pinned to ../.venv_quant).
All jobs are paper / read-only: no orders, no keys. Live executors (quant-kalshi-paper, quant-forecastex-exec) and
`~/.claude/quant-kalshi/.env` are untouched; L32b and daily_ticket only READ ~/.claude/quant-kalshi.

| label (com.jonathanwang.) | does | schedule | output |
|---|---|---|---|
| quant-sports-early-form | `nba_early_form_logger.py tick` (first tick per day refreshes ESPN, later `--no-refresh`) | 600 s, active 08:00-22:00 PT | data/nba_early_form_log.jsonl |
| quant-sports-l32b | `l32b_logger.py tick` | 600 s, 08:00-22:00 PT | data/l32b_log.jsonl, l32b_state.json |
| quant-sports-props | `kalshi_prop_logger.py grade` (first tick per day) + `tick` | 600 s, 08:00-22:00 PT | data/kalshi_prop_log.jsonl, kalshi_prop_grades.jsonl |
| quant-sports-ticket | `daily_ticket.py --date <PT date>` | 600 s, 08:00-22:00 PT | data/tickets/<date>.md |
| quant-sports-lock-queue | `lock_queue_logger.py tick` (READ-ONLY, no orders/keys; real 99c queue for lock_t2_nba_maker99: locks from ESPN STATUS_FINAL (moneyline winner) and score sum >= K + 3.5 (over rungs), book at lock +0/15/30/60/120/300 s, public tape to +2400 s; not an edgelab trial) | **60 s, 24 h**; no network outside [start + 75 min, start + 6 h] of a cached schedule; stays up polling every 15 s only while a game is in period >= 4 or book snapshots are owed; idle runs write only `state/lock_queue_heartbeat`. `report` = per-series queue and joiner share | data/live/lock_queue.jsonl, lock_queue_state.json |
| quant-sports-q017 | `q017_logger.py tick` | 600 s fire, runs if >= 25 min since last, **24 h** | data/q017/fwd/{vintages,quotes}.jsonl |
| quant-sports-nhl-ingame | `nhl_ingame_logger.py tick` (NHL score/now + KXNHLGAME quotes, Poisson/empty-net model; settles tickers after FINAL) | **60 s, 24 h**; exits with no network outside [start - 75 min, start + 5 h] of a cached schedule; idle runs write only `state/nhl_ingame_heartbeat` | data/live/nhl_ingame_paper.jsonl, nhl_ingame_results.jsonl, nhl_ingame_paper_state.json |
| quant-sports-nfl-fade | `nfl_fade_live.py tick` (F1L: live proxy of F1 -- ESPN WP big plays >= 0.08, YES bid at t+60..120 s; F1 candle grade after t+780 s) | **60 s, 24 h**; 15 s sub-ticks for ~50 s inside [start - 15 min, start + 5 h]; idle otherwise | data/live/nfl_fade_live.jsonl, nfl_fade_live_state.json |
| quant-sports-digest | `qs_digest.sh` (fund_status.py + `claude -p`) | 09:00 and 23:30 PT | digests/<date>-<am\|pm>.md + notification |

Q017 is deliberately not windowed: its frozen decision times are 06:00 / 10:00 CT (= 04:00 / 08:00 PT) and need a
quote in [T-45 min, T]; an 08:00-22:00 PT window would make both unobservable.
NHL in-game (added 2026-11-06): forward paper for edgelab trial `nhl_ig_t1_late_taker` (dataset kalshi_nhl_ingame,
forward from 2026-11-08; rule edgelab/rules/nhl_ingame.py decides bets from the log, the logger's `signal` is only a
flag). Pregame anchor = home mid seen pre-puck-drop, else the last Kalshi 1-min candle <= scheduled start (so a Mac
asleep at puck drop still logs the game). Any API error -> `exit 1` line in logs/nhl-ingame.log, state kept. If the
Mac sleeps through the last 15 min of a game, that game simply has no late rows (missing, not wrong).
REAL-MONEY HOOK (11-07, user "small amounts of real money for NHL and NFL"): nhl-ingame evaluates the FROZEN trial rule
live each tick (edgelab loader + rule on the game's ticks so far -> data/live/nhl_ingame_rule_signals.jsonl) and
nfl-fade emits F1L signals; both pipe them (and housekeeping) to ~/.claude/quant-kalshi/scripts/ingame_exec.py in the
quant-kalshi venv. DRY until .env has EXEC_MODE=live + LEAD_EXEC_MODE=live + LEAD_LIVE_NHL=1 / LEAD_LIVE_NFLF=1.
Caps, guards and kills: ingame_exec.py docstring. Journal: quant-kalshi data/lead_exec.jsonl (rows keyed `key`).
`ingame_exec.py status` = per-strategy mode / spend / P&L. The paper rows stay the edgelab record.
Run logs: `~/.claude/quant-sports/logs/<job>.log` (every run: header line + script output + `exit N`; rotated at 5 MB);
launchd's own stdout/stderr: `logs/launchd-<job>.{out,err}`. Per-day stamps: `state/`.
Entry point: `scripts/qs_job.sh <job>` (window check, `caffeinate -i` around every run, STOP file check).

## Commands (from sports/ in the worktree)
- Install / refresh everything: `scripts/install_quant_sports_agents.sh` (idempotent; never overwrites deployed logs)
- Redeploy code after edits: `scripts/qs_sync.sh push` (scripts/, sports/ pkg, lab/*.py, results/*.json, STATE.md copy,
  digest prompt; also copies data/user_calls.jsonl worktree -> deployed, because the worktree is that file's writer)
- Bring logs home for analysis/commits: `scripts/qs_sync.sh pull` (deployed -> worktree data/, run logs ->
  data/quant_sports_logs/). `scripts/qs_sync.sh diff` shows which logs differ.
- **Stop everything (one command):** `scripts/qs_stop.sh` (pause: every run no-ops; `scripts/qs_stop.sh resume`).
  Remove the agents entirely: `scripts/uninstall_quant_sports_agents.sh` (keeps the deployed data).
- Verify one agent: `launchctl kickstart -k gui/$(id -u)/com.jonathanwang.quant-sports-<job>` then check that
  `logs/<job>.log` grew (exit 0 alone proves nothing).
- Status bundle by hand: `cd ~/.claude/quant-sports && .venv/bin/python scripts/fund_status.py` (`--digest` = no-LLM digest).

**The DEPLOYED copy is the source of truth for these logs from 2026-11-01.** Do not run the ticks from the worktree any
more (two writers -> duplicate rows); run reports in the worktree only after `qs_sync.sh pull`.

## Digest
`scripts/qs_digest.sh [am|pm]`: (1) `fund_status.py --out digests/<date>-<tag>` writes the deterministic bundle
(.status.md: kalshi_exec status head, bet.py pnl head, F1 / L32b / L32 / props / user_calls one-liners, today's ticket,
log ages + last exit codes, `launchctl list | grep quant`, errors in the last 24 h, pending asks from the STATE.md
copy), a no-LLM digest (.fallback.md) and a headline; (2) `claude -p --model sonnet --max-turns 15
--setting-sources project --strict-mcp-config --no-session-persistence`, allowed tools Read + tail/grep/ls/launchctl
list, Edit/Write/Web disallowed, prompt = `digest_prompt.md` (source scripts/qs_digest_prompt.md) + the bundle on
stdin; the result becomes digests/<date>-<tag>.md; (3) if claude fails or returns no `HEADLINE:` line, the fallback
digest is used instead; (4) `osascript display notification` with the headline. Digests hold dollar amounts and live
under ~/.claude only (never in the public repo).
Token budget (measured, first launchd run 2026-11-01 pm, digests/usage.jsonl): 1 turn, 10.9 s, ~20.7k input tokens
(10.2k cache-write + 10.5k cache-read), 1.5k output, $0.057 -> ~$0.11/day at two runs. Auth: the OAuth keychain
login works under launchd (verified); `--bare` would NOT work (API-key only).
`QS_DIGEST_NO_LLM=1` forces the fallback.

## Sleep / wake
Every job run is wrapped in `caffeinate -i`, so the Mac cannot idle-sleep mid-run, but launchd does not wake a sleeping
Mac: StartInterval jobs missed during sleep fire once on wake. To guarantee the morning window, the user can run (sudo,
not applied by the agent): `sudo pmset repeat wakeorpoweron MTWRFSU 07:55:00` (check with `pmset -g sched`;
remove with `sudo pmset repeat cancel`). Only one repeat wake is allowed by pmset; the evening game window relies
on the Mac being awake/plugged in (lid closed on battery = sleep = missed tip windows).

## Failure modes
- Mac asleep through a 60-90 min pre-tip window -> that game is never logged (loggers are idempotent and refuse
  tipped games, so it is a missing row, not a bad row). Digest "log age" column shows it.
- Code edited in the worktree but not pushed -> deployed jobs run stale code. Always `qs_sync.sh push` after edits.
- Data needed by a new job missing in the deployed copy -> add it to `seed()` in qs_sync.sh and re-run install.
  The HTTP cache (sports/data/cache, 40 GB) is NOT copied; the deployed copy starts cold and fetches only new games.
- claude CLI path moves (cask update) -> qs_digest.sh uses /opt/homebrew/bin/claude (symlink); a failure falls back to
  the no-LLM digest and the log says so. Logged-out claude -> same fallback.
- Pending asks come from the STATE.md copy taken at the last `qs_sync.sh push`, so they can be stale.
- Q017: at 18:45 PT on 11-01 the KXTXERCOTPEAKD series had events but 0 open markets; quotes accrue only while
  markets are open -- check `q017_logger.py report` after a morning.
