#!/bin/bash
# launchd entry point for com.jonathanwang.quant-sports-<job> (deployed copy ~/.claude/quant-sports; installed by
# scripts/install_quant_sports_agents.sh). launchd fires every 600 s; THIS wrapper decides whether to do anything:
#   early-form  08:00-22:00 PT: nba_early_form_logger.py tick (first successful tick of the day refreshes ESPN data,
#               later ticks --no-refresh)
#   l32b        08:00-22:00 PT: l32b_logger.py tick
#   props       08:00-22:00 PT: kalshi_prop_logger.py grade (first tick of the day) + tick
#   ticket      08:00-22:00 PT: daily_ticket.py --date <PT date>  (writes data/tickets/<date>.md)
#   q017        24 h, every >= 25 min: q017_logger.py tick. Deliberately NOT windowed: its frozen decision times are
#               06:00 and 10:00 CT (= 04:00 / 08:00 PT) with a quote needed in [T-45 min, T].
#   nhl-ingame  24 h, launchd every 60 s: nhl_ingame_logger.py tick (exits without network when no NHL game is in
#               [start - 75 min, start + 5 h]; idle runs leave only state/nhl_ingame_heartbeat, not log lines)
#   nfl-fade    24 h, launchd every 60 s: nfl_fade_live.py tick (F1L proxy; idle exit without network outside game windows)
#   digest      scripts/qs_digest.sh (09:00 / 23:30 PT via StartCalendarInterval)
# Every run is wrapped in `caffeinate -i` so the Mac cannot idle-sleep mid-run. A file named STOP in the deployed
# root makes every job a no-op (scripts/qs_stop.sh). Paper only: none of these jobs place orders or read keys.
# Exception by design (11-07): nhl-ingame and nfl-fade pipe their signals to ~/.claude/quant-kalshi/scripts/
# ingame_exec.py (separate venv + .env); it sends orders only when .env arms EXEC_MODE + LEAD_EXEC_MODE + LEAD_LIVE_<S>.
set -u
QS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$QS" || exit 1
export TZ=America/Los_Angeles PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
PY="$QS/.venv/bin/python"
job="${1:?job}"
mkdir -p logs state
LOG="logs/$job.log"
d=$(date +%F); hm=$(date +%H%M)
[ -f "$QS/STOP" ] && exit 0

# rotate at 5 MB (keep one old copy)
if [ -f "$LOG" ] && [ "$(stat -f %z "$LOG")" -gt 5000000 ]; then mv "$LOG" "$LOG.1"; fi

in_window() { [ "$((10#$hm))" -ge 800 ] && [ "$((10#$hm))" -lt 2200 ]; }
done_today() { [ "$(cat "state/$1" 2>/dev/null)" = "$d" ]; }
run() {   # run <python script + args>: log header, caffeinate, exit code
  echo "=== $(date '+%F %T %Z') $job: $*" >> "$LOG"
  /usr/bin/caffeinate -i "$PY" "$@" >> "$LOG" 2>&1
  local rc=$?
  echo "exit $rc" >> "$LOG"
  return $rc
}

case "$job" in
  early-form)
    in_window || exit 0
    if done_today early_form_refresh; then run scripts/nba_early_form_logger.py tick --no-refresh
    else run scripts/nba_early_form_logger.py tick && echo "$d" > state/early_form_refresh; fi ;;
  l32b)
    in_window || exit 0
    run scripts/l32b_logger.py tick ;;
  props)
    in_window || exit 0
    if ! done_today props_grade; then run scripts/kalshi_prop_logger.py grade && echo "$d" > state/props_grade; fi
    run scripts/kalshi_prop_logger.py tick ;;
  ticket)
    in_window || exit 0
    echo "=== $(date '+%F %T %Z') ticket: daily_ticket.py --date $d" >> "$LOG"
    /usr/bin/caffeinate -i "$PY" scripts/daily_ticket.py --date "$d" > /dev/null 2>> "$LOG"
    rc=$?; echo "exit $rc ($(wc -c < "data/tickets/$d.md" 2>/dev/null || echo 0) bytes data/tickets/$d.md)" >> "$LOG" ;;
  q017)
    now=$(date +%s); last=$(cat state/q017_last 2>/dev/null || echo 0)
    [ $((now - last)) -ge 1500 ] || exit 0
    run scripts/q017_logger.py tick && echo "$now" > state/q017_last ;;
  tsa)      # E1X-T KXTSAW maker paper (notes/e1x_tsa.md): fills+settle once a day; tick Mondays 09:00-17:00 PT
    if [ "$((10#$hm))" -ge 700 ] && ! done_today tsa_fills; then run scripts/e1x_tsa_logger.py fills && echo "$d" > state/tsa_fills; fi
    if [ "$(date +%u)" = 1 ] && [ "$((10#$hm))" -ge 900 ] && [ "$((10#$hm))" -lt 1700 ]; then run scripts/e1x_tsa_logger.py tick; fi ;;
  nhl-ingame)   # forward paper for edgelab trial on kalshi_nhl_ingame (notes/HARNESS.md); log only non-idle runs
    out=$(/usr/bin/caffeinate -i "$PY" scripts/nhl_ingame_logger.py tick 2>&1); rc=$?
    echo "$(date '+%F %T %Z') rc=$rc" > state/nhl_ingame_heartbeat
    if [ $rc -ne 0 ] || ! printf '%s' "$out" | grep -q ' idle: '; then
      { echo "=== $(date '+%F %T %Z') $job: nhl_ingame_logger.py tick"; printf '%s\n' "$out"; echo "exit $rc"; } >> "$LOG"
    fi ;;
  nfl-fade)     # F1L live proxy paper logger (scripts/nfl_fade_live.py, 11-07): 15 s sub-ticks for ~50 s while an NFL
                # game is in [start - 15 min, start + 5 h]; hands signals to ingame_exec.py (DRY unless LEAD_LIVE_NFLF=1)
    out=$(/usr/bin/caffeinate -i "$PY" scripts/nfl_fade_live.py tick 2>&1); rc=$?
    echo "$(date '+%F %T %Z') rc=$rc" > state/nfl_fade_heartbeat
    if [ $rc -ne 0 ] || ! printf '%s' "$out" | grep -q ' idle: '; then
      { echo "=== $(date '+%F %T %Z') $job: nfl_fade_live.py tick"; printf '%s\n' "$out"; echo "exit $rc"; } >> "$LOG"
    fi ;;
  digest)
    /usr/bin/caffeinate -i /bin/bash scripts/qs_digest.sh "${2:-}" >> "$LOG" 2>&1 ;;
  *) echo "unknown job $job" >&2; exit 2 ;;
esac
exit 0
