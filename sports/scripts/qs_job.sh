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
#   kx-k1 / kx-knfl / kx-kcfb / kx-openers / lx1   READ-ONLY Kalshi-execution paper trials + scanners (11-13,
#               notes/ops_forward_wiring.md; installed by scripts/install_kx_forward_agents.sh)
#   nhl-om-a / nhl-goalie   READ-ONLY NHL opener-model forward paper + goalie-confirmation logger (11-15,
#               notes/ops_nhl_goalie.md; installed by scripts/install_nhl_agents.sh)
#   ot1         READ-ONLY OT1 Deribit-vs-Kalshi crypto tails forward paper (11-15, notes/ops_ot1_forward.md;
#               installed by scripts/install_ot1_agent.sh)
#   e4p         READ-ONLY E4 probabilistic rule C / C-agree maker forward paper (11-16, notes/ops_e4p_forward.md;
#               installed by scripts/install_e4p_agent.sh)
#   snow-markets READ-ONLY TG1/SD1/SD2 snow paper, daily ~15:30Z (11-16, notes/snow_markets.md;
#               installed by scripts/install_snow_markets_agent.sh)
# Every run is wrapped in `caffeinate -i` so the Mac cannot idle-sleep mid-run. A file named STOP in the deployed
# root makes every job a no-op (scripts/qs_stop.sh). Paper only: none of these jobs place orders or read keys.
# Exception by design (11-07): nhl-ingame and nfl-fade pipe their signals to ~/.claude/quant-kalshi/scripts/
# ingame_exec.py (separate venv + .env); it sends orders only when .env arms EXEC_MODE + LEAD_EXEC_MODE + LEAD_LIVE_<S>.
# The whole body is ONE brace group ending in `exit` (ops_robustness_1118): bash parses a { ... } group completely
# before running it, so even an in-place overwrite of this file mid-run cannot make a running job read half-new
# bytes ('syntax error near unexpected token ;;', nhl-ingame / nfl-fade 11-15..11-18). Installers also deploy it
# with tmp + mv (scripts/atomic_deploy.sh). Keep new code INSIDE the braces.
{
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
  lock-queue)   # READ-ONLY queue measurement for lock_t2_nba_maker99 (scripts/lock_queue_logger.py): stays up polling every
                # 15 s only while an NBA game is in period >= 4 or snapshots are owed; idle runs print nothing
    /usr/bin/caffeinate -i "$PY" -u scripts/lock_queue_logger.py tick >> "$LOG" 2>&1; rc=$?
    echo "$(date '+%F %T %Z') rc=$rc" > state/lock_queue_heartbeat
    [ $rc -eq 0 ] || echo "$(date '+%F %T %Z') lock-queue exit $rc" >> "$LOG" ;;
  # ---- ops_forward_wiring (11-13, notes/ops_forward_wiring.md): READ-ONLY Kalshi-execution paper trials + scanners.
  # All Kalshi GETs share scripts/kx_throttle.py (one cross-process slot, <= ~3.3 req/s total). Outputs append-only.
  kx-k1)        # L32b-K1: price new L32b live picks at the tick candle; settle finals. 24 h, every >= 25 min
    now=$(date +%s); last=$(cat state/kx_k1_last 2>/dev/null || echo 0)
    [ $((now - last)) -ge 1500 ] || exit 0
    run scripts/kx_forward.py k1 && echo "$now" > state/kx_k1_last ;;
  kx-knfl)      # K-NFL + K50-NFL: price new L33 decisions on Kalshi; settle finals. 24 h, every >= 2 h
    now=$(date +%s); last=$(cat state/kx_knfl_last 2>/dev/null || echo 0)
    [ $((now - last)) -ge 7000 ] || exit 0
    run scripts/kx_forward.py knfl; r1=$?
    run scripts/kx_forward.py k50nfl; r2=$?
    [ $r1 -eq 0 ] && [ $r2 -eq 0 ] && echo "$now" > state/kx_knfl_last ;;
  kx-kcfb)      # K-CFB: once a day from 07:00 PT, the pre-registered forward pipeline on games final through yesterday
    [ "$((10#$hm))" -ge 700 ] || exit 0
    done_today kx_kcfb && exit 0
    hi=$(date -v-1d +%F)
    X=results/kalshi_openers_xleague
    export KX_LO=2026-11-12 KX_HI="$hi" KX_TAG=fwd
    run $X/q2_cfb_preds.py --refetch && run $X/q1_fetch.py cfb && run $X/q1_timing.py cfb && run $X/q2_cfb.py \
      && run scripts/kx_forward.py kcfb && echo "$d" > state/kx_kcfb ;;
  kx-openers)   # openers timing logger (Kalshi 7 ladder series + Pinnacle), one tick per 10 min, 09:00-24:00 PT
    [ "$((10#$hm))" -ge 900 ] || exit 0
    run results/kalshi_openers_xleague/tonight_logger.py --once ;;
  lx1)          # LX1 logical-arb scanner: scan every >= 2 h until the judge window closes (2026-11-20 08:00Z);
                # once a day from 06:00 PT: persist --hours 48 + report (report keeps running after the window)
    now=$(date +%s); last=$(cat state/lx1_last 2>/dev/null || echo 0)
    if [ "$now" -lt 1795161600 ] && [ $((now - last)) -ge 7000 ]; then
      run scripts/lx1_arb_scan.py scan && echo "$now" > state/lx1_last
    fi
    if [ "$((10#$hm))" -ge 600 ] && ! done_today lx1_daily; then
      run scripts/lx1_arb_scan.py persist --hours 48 && run scripts/lx1_arb_scan.py report && echo "$d" > state/lx1_daily
    fi ;;
  # ---- ops_nhl_goalie (11-15, notes/ops_nhl_goalie.md): READ-ONLY; installed by scripts/install_nhl_agents.sh
  nhl-om-a)     # NHL-om-A forward paper (nhl_om_a_kft-754cfc46): MoneyPuck refresh once a day from 05:00 PT (current
                # season summaries re-pulled for new goalies), then decide/settle every run (launchd 1800 s, 24 h)
    if [ "$((10#$hm))" -ge 500 ] && ! done_today nhl_mp; then
      yr=$(date +%Y); [ "$(date +%m)" -lt 8 ] && yr=$((yr - 1))
      rm -f results/nhl_opener_model/mp/summary_teams_$yr.csv results/nhl_opener_model/mp/summary_goalies_$yr.csv
      run results/nhl_opener_model/fetch_mp.py --refresh && echo "$d" > state/nhl_mp
    fi
    run scripts/nhl_om_forward.py tick ;;
  nhl-goalie)   # goalie-confirmation timestamps + Kalshi NHL ladder snapshots every 120 s; idle ticks only heartbeat
    out=$(/usr/bin/caffeinate -i "$PY" scripts/nhl_goalie_logger.py tick 2>&1); rc=$?
    echo "$(date '+%F %T %Z') rc=$rc" > state/nhl_goalie_heartbeat
    if [ $rc -ne 0 ] || ! printf '%s' "$out" | grep -q ' idle: '; then
      { echo "=== $(date '+%F %T %Z') $job"; printf '%s\n' "$out"; echo "exit $rc"; } >> "$LOG"
    fi ;;
  # ---- ops_ot1_forward (11-15, notes/ops_ot1_forward.md): READ-ONLY; installed by scripts/install_ot1_agent.sh
  ot1)          # OT1 forward paper: decide once/day at the first run in [11:00, 12:00) ET (Deribit then Kalshi live book),
                # settle events after 17:05 ET. launchd 120 s; idle runs print nothing -> only the heartbeat file
    out=$(/usr/bin/caffeinate -i "$PY" scripts/ot1_forward.py tick 2>&1); rc=$?
    echo "$(date '+%F %T %Z') rc=$rc" > state/ot1_heartbeat
    if [ $rc -ne 0 ] || [ -n "$out" ]; then
      { echo "=== $(date '+%F %T %Z') $job: ot1_forward.py tick"; printf '%s\n' "$out"; echo "exit $rc"; } >> "$LOG"
    fi ;;
  # ---- ops_series_watch (11-15, notes/ops_series_watch.md): READ-ONLY; installed by scripts/install_series_watch_agent.sh
  series-watch) # daily Kalshi new-series diff + E4-pattern score; launchd 07:00 PT (+ hourly retry until it succeeds)
    [ "$((10#$hm))" -ge 700 ] || exit 0
    done_today series_watch_done && exit 0
    export SW_NO_MIRROR=1 PYTHONUNBUFFERED=1   # never open ~/Desktop from launchd (TCC open() hangs)
    run scripts/series_watch.py run --date "$d" && echo "$d" > state/series_watch_done ;;
  # ---- ops_state_gas_book (11-15, notes/ops_state_gas_book.md): READ-ONLY; installed by scripts/install_state_gas_books_agent.sh
  state-gas-books)  # KXAAAGASCD* order books (all open state events, all 5 buckets, full depth), ~100 GETs/snapshot.
                    # launchd every 900 s; slot am = first run in [14:00, 18:00) UTC, slot close = first run in
                    # [02:15, 03:05) ET (events close ~03:15-03:40 ET when AAA prints). Once per slot per day.
    ud=$(TZ=UTC date +%F); uhm=$(TZ=UTC date +%H%M); ed=$(TZ=America/New_York date +%F); ehm=$(TZ=America/New_York date +%H%M)
    if [ "$((10#$uhm))" -ge 1400 ] && [ "$((10#$uhm))" -lt 1800 ] && [ "$(cat state/state_gas_am 2>/dev/null)" != "$ud" ]; then
      run scripts/state_gas_books.py snap --slot am && echo "$ud" > state/state_gas_am
    fi
    if [ "$((10#$ehm))" -ge 215 ] && [ "$((10#$ehm))" -lt 305 ] && [ "$(cat state/state_gas_close 2>/dev/null)" != "$ed" ]; then
      run scripts/state_gas_books.py snap --slot close && echo "$ed" > state/state_gas_close
    fi ;;
  # ---- ops_econ_event_logger (11-15, notes/ops_econ_event_logger.md): READ-ONLY; scripts/install_econ_event_books_agent.sh
  econ-events)  # KX*EVENT ladder books at release-5/+5/+30 min (+ Deribit BTC/ETH trades at +5). launchd 60 s, 24 h;
                # idle ticks make no network call (schedule re-pulled <= every 6 h) and print nothing -> heartbeat only
    out=$(/usr/bin/caffeinate -i "$PY" scripts/econ_event_books.py tick 2>&1); rc=$?
    echo "$(date '+%F %T %Z') rc=$rc" > state/econ_events_heartbeat
    if [ $rc -ne 0 ] || [ -n "$out" ]; then
      { echo "=== $(date '+%F %T %Z') $job: econ_event_books.py tick"; printf '%s\n' "$out"; echo "exit $rc"; } >> "$LOG"
    fi ;;
  # ---- ops_e4_0600 (11-16, notes/ops_e4_0600.md): READ-ONLY; installed by scripts/install_e4_0600_agent.sh
  e4-0600)      # E4 06:00-D add-on maker paper: tick decides per orig-7 city in [06:00, 06:40) station-local (idle runs
                # log nothing but the heartbeat); settle (tape + result) at most once per hour. launchd 300 s.
    out=$(/usr/bin/caffeinate -i "$PY" scripts/e4_0600_paper.py tick 2>&1); rc=$?
    echo "$(date '+%F %T %Z') rc=$rc" > state/e4_0600_heartbeat
    if [ $rc -ne 0 ] || ! printf '%s' "$out" | grep -q '^idle:'; then
      { echo "=== $(date '+%F %T %Z') $job: e4_0600_paper.py tick"; printf '%s\n' "$out"; echo "exit $rc"; } >> "$LOG"
    fi
    now=$(date +%s); last=$(cat state/e4_0600_settle_last 2>/dev/null || echo 0)
    if [ $((now - last)) -ge 3600 ]; then run scripts/e4_0600_paper.py settle && echo "$now" > state/e4_0600_settle_last; fi ;;
  e4p)          # E4 probabilistic rule C / C-agree maker paper (notes/ops_e4p_forward.md): tick decides per city in
                # [22:00, 22:50) station-local D-1 (idle runs log nothing but the heartbeat); settle at most once per hour.
    out=$(/usr/bin/caffeinate -i "$PY" scripts/e4p_paper.py tick 2>&1); rc=$?
    echo "$(date '+%F %T %Z') rc=$rc" > state/e4p_heartbeat
    if [ $rc -ne 0 ] || ! printf '%s' "$out" | grep -q '^idle:'; then
      { echo "=== $(date '+%F %T %Z') $job: e4p_paper.py tick"; printf '%s\n' "$out"; echo "exit $rc"; } >> "$LOG"
    fi
    now=$(date +%s); last=$(cat state/e4p_settle_last 2>/dev/null || echo 0)
    if [ $((now - last)) -ge 3600 ]; then run scripts/e4p_paper.py settle && echo "$now" > state/e4p_settle_last; fi ;;
  # ---- crypto_barrier (11-16, notes/crypto_barrier.md): READ-ONLY; installed by scripts/install_crypto_barrier_agent.sh
  crypto-barrier)   # CB1 monthly crypto one-touch paper: decide once per UTC day at the first run >= 16:00 UTC (idle runs
                    # print nothing -> heartbeat only); settle picks at most once per hour. launchd 300 s.
    out=$(/usr/bin/caffeinate -i "$PY" scripts/crypto_barrier_forward.py tick 2>&1); rc=$?
    echo "$(date '+%F %T %Z') rc=$rc" > state/crypto_barrier_heartbeat
    if [ $rc -ne 0 ] || [ -n "$out" ]; then
      { echo "=== $(date '+%F %T %Z') $job: crypto_barrier_forward.py tick"; printf '%s\n' "$out"; echo "exit $rc"; } >> "$LOG"
    fi
    now=$(date +%s); last=$(cat state/crypto_barrier_settle_last 2>/dev/null || echo 0)
    if [ $((now - last)) -ge 3600 ]; then run scripts/crypto_barrier_forward.py settle && echo "$now" > state/crypto_barrier_settle_last; fi ;;
  # ---- ops_snow_logger (11-16, notes/snow_markets.md 'Ops'): READ-ONLY; installed by scripts/install_snow_markets_agent.sh
  snow-markets) # TG1/SD1/SD2 snow paper: one `log` snapshot per UTC day at the first run >= 15:30 UTC through 2027-04-01;
                # `settle` (fired tickers' Kalshi result / scalar value) once per UTC day from 2026-11-27. launchd 1800 s.
    ud=$(TZ=UTC date +%F); uhm=$(TZ=UTC date +%H%M)
    if [ "$((10#$uhm))" -ge 1530 ] && [[ "$ud" < "2027-04-02" ]] && [ "$(cat state/snow_log_day 2>/dev/null)" != "$ud" ]; then
      run scripts/snow_markets.py log && echo "$ud" > state/snow_log_day
    fi
    if [[ ! "$ud" < "2026-11-27" ]] && [ "$(cat state/snow_settle_day 2>/dev/null)" != "$ud" ]; then
      run scripts/snow_markets.py settle && echo "$ud" > state/snow_settle_day
    fi ;;
  # ---- ops_quake_logger (11-16, notes/quake_monthly.md 'Ops'): READ-ONLY; installed by scripts/install_quake_monthly_agent.sh
  quake-monthly)  # QM1/QM2 KXBIGGESTQUAKEMONTH paper: USGS GET every run; Kalshi snapshot at the first run >= 15:00 UTC
                  # (decision iff in [15:00, 15:45)), 23:00 UTC on the month's last 5 days, and within 6 h of any USGS
                  # M>=6.5 (launchd 1800 s -> <= ~30 min lag); settle/fills hourly. Idle runs print nothing -> heartbeat only
    out=$(/usr/bin/caffeinate -i "$PY" scripts/quake_monthly_logger.py tick 2>&1); rc=$?
    echo "$(date '+%F %T %Z') rc=$rc" > state/quake_monthly_heartbeat
    if [ $rc -ne 0 ] || [ -n "$out" ]; then
      { echo "=== $(date '+%F %T %Z') $job: quake_monthly_logger.py tick"; printf '%s\n' "$out"; echo "exit $rc"; } >> "$LOG"
    fi ;;
  # ---- esports_pinnacle (11-18, notes/esports_pinnacle.md 'Ops'): READ-ONLY; installed by scripts/install_esports_pin_agent.sh
  esports-pin)  # ESP1 Kalshi esports books vs Pinnacle esports odds (+ quote age) for matches starting within 6 h.
                # launchd 120 s, 24 h; every run writes a fetch row (coverage denominator); stdout only on matches/bets.
                # settle paper bets at most once per hour.
    out=$(/usr/bin/caffeinate -i "$PY" scripts/esports_pin_logger.py tick 2>&1); rc=$?
    echo "$(date '+%F %T %Z') rc=$rc" > state/esports_pin_heartbeat
    if [ $rc -ne 0 ] || [ -n "$out" ]; then
      { echo "=== $(date '+%F %T %Z') $job: esports_pin_logger.py tick"; printf '%s\n' "$out"; echo "exit $rc"; } >> "$LOG"
    fi
    now=$(date +%s); last=$(cat state/esports_pin_settle_last 2>/dev/null || echo 0)
    if [ $((now - last)) -ge 3600 ]; then run scripts/esports_pin_logger.py settle && echo "$now" > state/esports_pin_settle_last; fi ;;
  digest)
    /usr/bin/caffeinate -i /bin/bash scripts/qs_digest.sh "${2:-}" >> "$LOG" 2>&1 ;;
  *) echo "unknown job $job" >&2; exit 2 ;;
esac
exit 0
}
