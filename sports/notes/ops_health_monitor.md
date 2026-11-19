# ops_health_monitor -- hourly launchd health check for every quant job (agent ops_health_monitor, 2026-11-18)

## Question
Two silent failures this week (E4 pick logger 0 picks 11-13/11-14: open-meteo timeouts, launchd exit 0;
Q013 RT logger 0 picks since 11-03: field rename). Build a monitor that would have caught both within a day.

Ops build, not an edge test: no success/kill rule on outcomes. Acceptance rule written before the first run:
(1) the monitor runs under launchd from ~/.claude/quant-sports (TCC: no ~/Desktop reads), hourly;
(2) per job it checks launchd load state + last exit, recent log errors, output freshness vs a per-job cadence with
season windows, last-row key-field sanity, and 0-pick streaks for pick loggers;
(3) replay check: on the E4 run.log it must flag the 11-13/11-14 ReadTimeout ticks (log-error check).

## Plan
- scripts/health_monitor.py (stdlib only) + scripts/health_jobs.json (per-job config) + install_health_monitor_agent.sh
- output: ~/.claude/quant-sports/digests/health-<date>.md (overwritten each hourly run), state/health_status.json,
  data/health/history.jsonl; macOS notification at most once per job per day (state/health_notified.json)
- scorecards.py: `card_health()` reads state/health_status.json (separate "Ops health" section; not a kill-rule card)

## Built + deployed (2026-11-18 19:43 PST)
- Source: `sports/scripts/health_monitor.py` (stdlib), `sports/scripts/health_jobs.json` (47 entries: 41 active,
  3 RETIRED forecastex-exec / ibc-gateway / lock-queue, 1 IDLE early-form, plus the monitor itself),
  `sports/scripts/install_health_monitor_agent.sh` (copies via tmp + mv, writes plist, bootstraps).
- launchd `com.jonathanwang.quant-sports-health`, StartInterval 3600 + RunAtLoad, runs the deployed venv python on
  `~/.claude/quant-sports/scripts/health_monitor.py`. Verified: `launchctl print` runs = 2, last exit code = 0; digest
  `~/.claude/quant-sports/digests/health-2026-11-18.md` (9.4 KB), `state/health_status.json`,
  `data/health/history.jsonl`, one-line summary per run in `logs/health.log`.
- Alerts: osascript notification (kalshi_exec.notify pattern), one per run listing jobs newly STALE/BROKEN, each job at
  most once per PT day (`state/health_notified.json`). WARN never notifies (digest + scorecard only).
- `scripts/scorecards.py`: `card_health()` in its own "Ops health" section after the cards (never in KILL/near-kill).
- Unconfigured-job detector: any `com.jonathanwang.quant-*.plist` not in health_jobs.json is listed in the digest.

## Acceptance check (replay `--now 2026-11-14T09:00:00-08:00`, prints only, writes nothing)
- kalshi-paper -> BROKEN "0 picks on the last 2 active day(s) (limit 1); per day 11-07:20 11-08:14 11-09:12 11-10:13
  11-11:12 11-12:0 11-13:0". E4 had no 0-pick PT day from 10-18 to 11-11 (kalshi_paper.jsonl per-day counts), so
  zero_days = 1 has no historical false alarm; it would have alerted the morning of 11-13 (one night lost, not two).
- pmus-weather-paper on the same replay: WARN, 5/8 logged runs with open-meteo ReadTimeout.
- q013 on the same replay: BROKEN, 0 picks 11-10..11-13 (limit 3). Q013 picks are event-driven (KXRT/KXMC T24/T6
  decisions on 10-30, 11-01, 11-02, 11-04, 11-05, 11-08, 11-09, 11-15, 11-16), so zero_days 3 is a compromise; a precise
  rule needs "markets closing that day" as the eligibility denominator (not built).

## First live run (2026-11-18 19:43 PST): 0 BROKEN, 0 STALE, 7 WARN, 35 OK, 3 RETIRED, 1 OFF-SEASON, 1 IDLE
WARN (all real, none paging):
1. kalshi-paper (E4): run.log 3/24 ticks exit 3 = FETCH_FAIL KXHIGHDEN (open-meteo ReadTimeout, read timeout 10 s) on
   every retry in the 11-17 22-23h window; kalshi_paper.jsonl has 0 KXHIGHDEN-26NOV18 rows (could also be "no qualifying
   pick" -- not distinguished). Picks per PT day 11-11..17: 12,0,0,14,16,20,11.
2. e4p: 3/31 runs exit 3, e.g. `FAIL KXHIGHNY: ReadTimeout ... open-meteo (read timeout=15)` -- same upstream.
3. forecastex-paper: 23/48 logged runs `VenueError('UHBKF not in the public category tree')` -- BKF product missing from
   ForecastEx's tree; other cities fine (58-76 picks/day). Proposal: skip BKF in code.
4. open-lines: `ERR pinnacle HTTPError` 2/3 logged runs (pins 0); NHL picks still flowing.
5. kx-openers: 1/87 runs `pin err nhl HTTPError` (same Pinnacle source).
6. econ-events: `MISSED KXAUDUSDEVENT-18NOV262030|p30` slot at 01:11Z.
7. nhl-ingame stderr: `qs_job.sh: line 236: syntax error near unexpected token ;;` (nfl-fade too, 11-15). Mechanism: bash
   reads a script lazily while it runs; installers `cp` over qs_job.sh in place, so a job running at that moment reads a
   half-old/half-new file. Transient (one tick lost) but recurs at every install. PROPOSAL: all installers copy
   qs_job.sh via tmp + `mv -f` (atomic rename; the running bash keeps the old inode). Not changed here (other agents').
OFF-SEASON: kalshi-wx-monthly (active in the last 6 days of a month; next 11-25). RETIRED: forecastex-exec, ibc-gateway
(user stopped IBKR 11-15), lock-queue (ACT_LIST). IDLE: early-form (F1 window closed ~11-07).
Q013: OK now (snapshots 4 min old; picks 35 on 11-15 and 11-16).

## Wrong turns
- First launchd run flagged the monitor BROKEN for its own not-yet-written status file and spent today's notification
  on itself; removed that self-output check and reset health_notified.json.
- The digest job's log contains LLM-written sentences quoting other jobs' errors ("e4p.log still erroring (exit 3)"),
  which the generic error regex matched; the digest job now has its own err_re.

## Learnings
### launchd exit 0 is not evidence a job worked
All 41 active quant jobs showed last exit 0 in launchd on 11-18 while 7 had real errors in their own logs
(digests/health-2026-11-18.md). Wrappers exit with their last command's status; judge health from outputs.
### A per-day pick count with zero_days=1 would have caught the E4 outage one night earlier
Replay at 2026-11-14 09:00 PT flags kalshi-paper BROKEN (11-12:0, 11-13:0); E4 had no 0-pick PT day 10-18..11-11.
Event-driven loggers (Q013) need a looser threshold or an eligibility denominator.
### Overwriting a running bash script in place corrupts the running job
qs_job.sh syntax errors in launchd-nhl-ingame.err / launchd-nfl-fade.err; bash reads scripts lazily. Deploy shell
scripts with tmp + mv (atomic rename), never cp onto the live path.
