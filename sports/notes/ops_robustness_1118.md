# ops_robustness_1118 -- open-meteo robustness (E4/e4p), forecastex BKF skip, atomic installers (2026-11-18)

Ops fixes, not an edge test: no outcome success/kill rule. Acceptance rule written before changes:
(1) E4 / e4p: same frozen inputs (open-meteo /v1/forecast, same 5 models, same params); a city is lost only if
    open-meteo fails across the whole 22-23h local window; every lost city-night raises an alert; tests pass.
(2) forecastex-paper: a station missing from the public category tree is skipped with a logged reason, other
    stations proceed, run does not error; tests pass.
(3) every install_*_agent.sh / qs_sync copy step / qs_job.sh deploy uses tmp-in-same-dir + mv; each touched job
    runs once afterwards without a syntax error.
User approval: STATE.md 2026-11-18 "you have my approval on everything"; lead brief says ops fixes approved.
Backups before deploy go to ~/.claude/<deploy>/backups/ops_robustness_1118/.

## FIX 1 -- open-meteo (E4 real money, e4p paper)
### Diagnosis
- Lost on 11-17 PT: BOTH KXHIGHNY-26NOV18 (19:03 + 20:03 PT ticks = 22/23 ET) and KXHIGHDEN-26NOV18 (21:03 + 22:03
  PT = 22/23 MT): 3 tries x 2 ticks each, all `ReadTimeout ... (read timeout=10)` (~/.claude/quant-kalshi/run.log
  lines 11075-11117). kalshi_paper.jsonl has 0 rows for either event (11 rows for 26NOV18, all other cities). The 23h
  window retry DID run for both and failed too. The health monitor named only DEN; NY was lost the same night.
- Phase: the code passes timeout=(10, 45) but the error says "read timeout=10" = the CONNECT value. urllib3 2.8.0
  connectionpool.py:465-472 raises ReadTimeoutError with conn.timeout (= connect timeout) from `_validate_conn`, i.e.
  the TLS handshake stalled after TCP connected. e4p (timeout=(15, 60)) says "read timeout=15": same phase. Slow
  server compute would have shown 45/60.
- Pattern: in every failing tick the failed cities are the FIRST due cities in dict order (NY first ET; DEN first MT;
  e4p 11-16 20:06 CHI+AUS failed while DAL/HOU/MIN/NOLA/OKC/SATX succeeded in the same tick; e4p 11-17 22:04 LAX+LV).
  Requests later in the same process succeed. E4 NY succeeded at 19:03:04 tonight while e4p NY failed at 19:03:10.
  So: intermittent, clears within minutes; 3 tries in ~1 min is too short. Root cause (open-meteo edge / network path
  at the start of a burst) is SUSPECTED, not confirmed. Probe now (results/ops_robustness_1118/probe_om.py, 14
  requests): all 200, TLS done 0.09-0.13 s, total 0.13-0.18 s, one IP 51.81.167.44.
### Fix (same endpoint, same 5 models, same params; test_live_forecasts_inputs_unchanged asserts it)
1. `scripts/om_fetch.py` (shared): timeout (30, 60) so a stalled handshake gets 30 s. Each success caches the raw
   response per exact param set in ~/.claude/quant-shared/om_cache (atomic write). On failure, a caller with `since`
   (= 22:00 city-local, the window start) may use a cached response fetched in [since, now]: same source, same
   params, fetched inside the decision window (E4 and e4p fetch identical URLs minutes apart; coords match). Pick
   rows / e4p city_day rows now carry `fc_source` ("live" or "cache@<ts>").
2. kalshi_paper.py `retry` + run.sh: when tick exits 3, AFTER kalshi_exec has acted on the cities that worked, retry
   the pending cities every 90 s for up to 20 min, then run kalshi_exec again. Tries now span ~22:03-22:25 and
   ~23:03-23:25 local instead of ~1 min each.
3. LOST alert: a city still unfetched once its window is in its last hour (or shut) -> `LOST ...` line in run.log, a
   row in data/kalshi_paper_lost.jsonl, a macOS notification (once per city-night); health_jobs.json kalshi-paper
   warn_re "^LOST " (deployed). e4p: macOS alert on every `missed` city_day row.
   No fallback source: if open-meteo is down for the whole window the city is skipped and alerted.
- Tests: tests/test_kalshi_paper_retry_cache.py (6), tests/test_e4p_om_robust.py (2); the 3 existing resilience
  tests updated (fakes accept **kw). Full suite: 572 passed, 2 failed, both in tests/test_bet.py (bet.py untouched
  here; not caused by this change).
- Deployed 19:55 PST via tmp + os.replace (results/ops_robustness_1118/deploy.py). Backups:
  ~/.claude/quant-kalshi/backups/ops_robustness_1118/{run.sh,dispatch.sh,kalshi_paper.py},
  ~/.claude/quant-sports/scripts/backups/ops_robustness_1118/e4p_paper.py. Files: quant-kalshi scripts/kalshi_paper.py,
  scripts/om_fetch.py, run.sh; quant-sports scripts/e4p_paper.py, scripts/om_fetch.py. Read-only checks:
  `kalshi_paper.py validate` ran the live path (cache files written); `e4p_paper.py dry --series KXHIGHNY --day
  2026-11-19` decided (tradeable 4, C 0).

- First live tick on the new code, 20:03 PST: 3 picks (MIN, NOLA, SATX), all `fc_source` "live", exit 0, so no retry
  was needed. The retry and LOST paths have not run live yet; only the tests cover them.

## FIX 2 -- forecastex-paper UHBKF
- venues/forecastex_client.strikes() raises `VenueError('UHBKF not in the public category tree')` on every tick in
  BKF's window (forecastex_paper.log lines 2758-2769, 11-17), each logged as ERROR (health monitor: 23/48 runs).
- Fix: cmd_tick reads the tree once; a station absent from it -> `SKIP UH<code> is not listed ...` once per
  station-day (state `skip_logged`), not marked done (picked up again if it reappears). Tree fetch failure -> old
  per-station path (real error logged). Tests: tests/test_forecastex_paper_missing_station.py (2). Deployed (backup
  quant-kalshi/scripts/backups/ops_robustness_1118/); live `forecastex_paper.py validate`: "BKF 2026-11-19: SKIP
  UHBKF ..." and no ERROR lines.

## FIX 3 -- atomic installers
- Mechanism reproduced: tests/test_atomic_deploy.py::test_inplace_cp_corrupts_running_script (cp over a running
  script -> it never prints old-ok); the acp version prints old-ok.
- scripts/atomic_deploy.sh: `acp` (cp semantics, tmp in the target dir + mv -f, keeps the source mode) and `awrite`
  (heredoc -> tmp + mv). results/ops_robustness_1118/atomicize.py rewrote 30 installers + qs_sync.sh (originals in
  results/ops_robustness_1118/orig/): every cp at a command position -> acp, every `cat > "$DST/x.sh" <<` -> awrite.
  install_kalshi_paper_agent.sh done by hand (acp + run.sh/dispatch.sh via tmp + mv, run.sh now has the retry step).
  Health monitor installer was already atomic. Plists untouched (launchd reads them only at bootstrap). rsync steps
  already do tmp + rename. All 35 shell scripts pass bash -n. Installers were NOT re-run (not needed).
- qs_job.sh: whole body wrapped in one `{ ... exit 0 }` group; bash parses it whole before running, so even a plain
  cp mid-run is safe (test_brace_group_survives_inplace_cp). Deployed atomically 19:58 (backup
  quant-sports/scripts/backups/ops_robustness_1118/qs_job.sh). nhl-ingame and nfl-fade each ran 3 times by 20:01
  (launchctl runs 14387->14390, 13006->13009), last exit 0, no new stderr.

## Open / proposals
- Root cause of the TLS stalls is unconfirmed. If LOST alerts recur after this fix, log per-request timing (connect /
  TLS / first byte) in om_fetch to separate an open-meteo edge problem from the local network.
- tests/test_bet.py has 2 failures that came before this work (bet.py untouched); the lead should look at them.

## Learnings
### urllib3 "Read timed out (read timeout=<connect value>)" means the TLS handshake stalled
E4 passed timeout=(10, 45) but the logs said "read timeout=10". urllib3 2.8.0 connectionpool.py:465-472 reports the
connect timeout when `_validate_conn` (the TLS handshake) times out. Read the number in the message to tell which phase hung.
### open-meteo failures hit the first requests of a burst and clear within minutes
11-16..11-18: the cities that failed were always the first due cities in a tick (NY, DEN, CHI+AUS, LAX+LV), while
later cities in the same process succeeded (run.log, logs/e4p.log). Retries spread over ~20 min beat 3 quick tries.
Both NY and DEN were lost on 11-17, not only DEN.
### A cached response from the same window keeps a frozen rule's inputs unchanged
E4 and e4p request the same open-meteo URL minutes apart. Sharing a response fetched inside the decision window
(same endpoint, same params) adds redundancy without switching to a different forecast source.
### Wrap long-running bash entry points in one { ... exit } group
bash parses a brace group completely before it runs it, so overwriting the file mid-run can't corrupt the job.
tests/test_atomic_deploy.py reproduces the cp failure and shows that the brace group and acp (tmp + mv) both avoid it.
