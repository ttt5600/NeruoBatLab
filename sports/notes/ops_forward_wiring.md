# ops_forward_wiring -- wire the unwired forward paper trials (2026-11-13)

Ops agent for the quant-trader lead. User approved 11-13 ("lets keep it running") installing READ-ONLY paper loggers /
scanners under launchd. No orders, no keys, no git, no .env / kill files, live executors untouched.

## Problem
Registered forward trials with nothing accumulating:
- L32b-K1 `l32b_k1_kalshi_taker-8d6fb950` (notes/nba_l32b_exec.md, rule "Forward-paper rule K1")
- K-NFL `knfl_l33_kalshi_taker-445b123e` and K-CFB `kcfb_l36_kalshi_taker-613b7d62` (notes/kalshi_openers_xleague.md)
- LX1 logical-arb scanner (notes/lx1_logical_arb.md: scan every 2 h, daily persist --hours 48 + report, judge 11-19)
- openers timing logger (results/kalshi_openers_xleague/tonight_logger.py) one night only so far.

## Success / kill for this ops task (written before deploying)
- Success: each job (a) runs from ~/.claude/quant-sports under launchd, (b) after one kickstart its output file has new
  rows with plausible content (tickers that exist, prices in (0,1), times before tip), (c) `edgelab evaluate <id>
  --split forward` runs without error for K1 / K-NFL / K-CFB (0 settled bets is an acceptable verdict today).
- Fail: any job needs Desktop paths, writes outside its own files, or can exceed 4 Kalshi req/s together.
- The trading rules are NOT re-decided here: each logger re-implements the pre-registered rule text line by line
  (K1 = q2_score.py one(); K-NFL = q2_nfl.py; K-CFB runs the dev scripts themselves on forward dates).

## Plan
1. scripts/kx_throttle.py: one cross-process Kalshi throttle (fcntl lock + shared timestamp file, >= 0.30 s between
   requests from ALL these jobs + 0-0.1 s jitter => <= 3.3 req/s total), 429 back-off.
2. scripts/kx_forward.py k1|knfl|kcfb: append-only data/kx_forward/<trial>.jsonl, `pick` row once per decision
   (strike S, ticker, side, ask, fee, quote candle end, decided_at, event_start), `settle` row once per final.
3. edgelab/adapters/kx_fwd.py + vault.yaml forward_adapters for the three specs.
4. qs_job.sh cases + scripts/install_kx_forward_agents.sh (copies only the files these jobs need; no full push).
5. scorecards.py cards for K1 / K-NFL / K-CFB / LX1.

## Log
- 08:05Z lead addition: wire K50-NFL (`k50_nfl_k50_kalshi-d7b7da0f`, notes/k50_strike.md "Forward rule") like K-NFL.
  NBA/CFB K50 deliberately NOT wired (lead: dev ~0 edge vs Kalshi's own line).
- Built scripts/kx_throttle.py (shared fcntl slot, 0.30 s + U(0,0.10) jitter => <= ~3.3 req/s across kx_forward,
  lx1_arb_scan and kx.py/q1_fetch; LX1 and kx.py fall back to their old per-process sleeps if the module is absent).
- scripts/kx_forward.py k1|knfl|kcfb|k50nfl|status -> ~/.claude/quant-sports/data/kx_forward/<trial>.jsonl (append-only;
  one `pick` per decision, one `settle` per bet).
- Fidelity checks vs the pre-registered dev scripts (code checks on DEV rows, written to results/ops_forward_wiring/devcheck*,
  NOT the forward log):
  - K1: the 2 rows in q2_k1_rows_forward.csv reproduce exactly (S 223.5 / ask .54 / fee .0174; S 214.5 / ask .52).
  - K-NFL (KX_FWD_FROM=2026-10-26): 21/21 decisions identical to q2_nfl_rows_dev.csv (status, S, ticker, ask);
    14/14 settles identical (won, clv); my yes/no result = Kalshi's own settlement 14/14.
  - K50-NFL (KX_K50_FROM=2026-10-26) vs results/k50_strike/rows_nfl.csv: decision hour T identical 46/47 games, S50
    identical 45/47, ask 15/16 shared bets; 21/21 settles agree with Kalshi's result. The 2 S50 differences come from
    the stated deviation (forward fetches EVERY strike; dev fetched +-4/+-5 around the DK open, unknown pre-game).
    My event match (via open_lines_nfl_tight.json) also finds 4 games dev marked "no kalshi event"/"never tight".
  - Adapter on settled replays (results/ops_forward_wiring/adapter_check.py): PIT clean, outcomes resolve; knfl 14 bets,
    k50nfl 21, k1 3; Kalshi-result disagreements 0.
- K50-NFL forward set = kickoff after registration (2026-11-13T08:08Z). Caveat: its first 9 bets (week 10) were DECIDED
  11-10 (model cache 17:58Z + first tight), before registration; outcomes were unknown at registration. The k50 agent
  scored dev only through 11-11 finals, so these decisions were not in its dev rows.
- edgelab: adapters/kx_fwd.py; vault.yaml gets forward_adapters for l32b_k1_kalshi_taker (nba_totals_open) and two NEW
  datasets kalshi_openers_nfl (knfl + k50nfl) / kalshi_openers_cfb (kcfb) -- the specs referenced them but vault.yaml
  had no entry (forward split from 2026-11-12, no dev/sealed; loader name deliberately unregistered so only the adapter
  path works). Tested `evaluate --split forward` for all 4 on a SANDBOX registry copy
  (results/ops_forward_wiring/registry_sandbox.jsonl) so the real registry has no access rows and no adapter hash is
  pinned by me -- the lead's first real forward evaluate pins it.
- close_price for edgelab = same-side MID of the last candle <= event start (biased low by ~half a spread vs our ask);
  the pre-registered judge's points-CLV is in the settle rows (clv_S / clv).
- Deploy: scripts/install_kx_forward_agents.sh (copies only the files these jobs need into ~/.claude/quant-sports, seeds
  data/sbr_cfb, data/lab/l36_espn_to_sbr_names.json, results/kalshi_openers_xleague/cfb_espn and the 11-12 LX1 logs with
  --ignore-existing; installs lxml 6.1.3 + bs4 4.15.0 into the deployed venv -- the first K-CFB run died on
  `ImportError: Import lxml failed` in pandas.read_html of the SBR archive). Wrapper cases in scripts/qs_job.sh.
- Edits to shared scripts (behaviour unchanged except rate): kx.py get() and lx1_arb_scan.py get() take the shared slot;
  tonight_logger.py gained `--once` (one tick -> ticks_fwd/<UTC date>.jsonl.gz + first_seen_fwd.json; ~20 KB/tick gz).

## Jobs (all READ-ONLY, launchd, deployed root ~/.claude/quant-sports, logs logs/<job>.log)
| launchd label (com.jonathanwang.quant-sports-*) | does | schedule | output |
|---|---|---|---|
| kx-k1 | K1 pricing + settle | every 600 s, acts >= 25 min apart, 24 h | data/kx_forward/k1.jsonl |
| kx-knfl | K-NFL + K50-NFL pricing + settle | every 600 s, acts >= 2 h apart | data/kx_forward/knfl.jsonl, k50nfl.jsonl |
| kx-kcfb | K-CFB forward pipeline (preds --refetch, q1_fetch, q1_timing, q2_cfb on games final through yesterday) | daily from 07:00 PT | data/kx_forward/kcfb.jsonl (+ results/kalshi_openers_xleague/*fwd*) |
| kx-openers | openers timing logger tick | every 600 s, 09:00-24:00 PT | results/kalshi_openers_xleague/ticks_fwd/, first_seen_fwd.json |
| lx1 | LX1 scan / persist --hours 48 + report | scan >= 2 h apart until 2026-11-20 08:00Z; persist+report daily from 06:00 PT | results/lx1_logical_arb/ |

Stop one: `launchctl bootout gui/$(id -u)/com.jonathanwang.quant-sports-<label>` (and remove its plist from
~/Library/LaunchAgents to keep it off after reboot). Stop every quant-sports job: `touch ~/.claude/quant-sports/STOP`.

## Verification (11-13 ~08:30Z)
- kx-k1 (launchd kickstart, exit 0): k1.jsonl 10 pick rows (9 bets, 1 "gap<3 at S"), 3 settles (11-12 games, all W,
  Kalshi result agrees 3/3); asks 0.50-0.55, quote 1.3-6.6 min before the tick, all before tip.
- kx-knfl (launchd, exit 0): knfl 0 rows (no L33 forward pick with t >= 11-12 in open_lines_nfl_picks.jsonl yet --
  correct, the logger's last pick is 11-10); k50nfl 26 decisions for kickoffs after registration, 9 bets / 17 skip gap.
- lx1 (launchd, exit 0): scan 2026-11-13T08:12:59Z, 449 s, 200,363 markets, 1 violation row (scans.jsonl appended).
- kx-openers: outside its window at deploy time -> ran the identical command with the deployed venv by hand:
  627 rows (411 Kalshi, 216 Pinnacle), 20.5 KB gz. First launchd tick due 09:00 PT.
- kx-kcfb: outside its window -> ran the job body by hand (results/ops_forward_wiring/kcfb_manual.sh): 693 of 696 2026
  games final, 2 ESPN finals 11-12, 1 Kalshi total event, 1 bet (KXNCAAFTOTAL-26NOV12MEMUSF-50 over 49.5 at 0.48, won,
  CLV +2.0). Its decision (first tight 11-09) predates registration (11-12 17:37Z) -- forward is by game date per the
  note's KX_LO=2026-11-12; outcomes were unknown at registration. First launchd run due 07:00 PT.
- edgelab evaluate --split forward: runs for all four on the sandbox registry (verdict fail at n 0 = fail-closed, expected).
- scorecards.py: cards K1 / K-NFL / K-CFB / K50-NFL / LX1 added; notes/SCORECARDS.md regenerated (all too-early).

## For the lead
- Run the first REAL `python -m edgelab evaluate <id> --split forward` yourself (it pins the adapter sha256).
- vault.yaml changed (new datasets + adapters) -> vault_sha256 changes for every later evaluation (visible, by design).
- deployed qs_job.sh now = worktree copy (it was byte-identical before my cases were added), so a later full
  `qs_sync.sh push` stays consistent.

## Learnings

### Replay the forward logger on dev rows before trusting it
Running the forward K-NFL logger with its start date moved back reproduced the pre-registered dev script exactly:
21/21 decisions, 14/14 settles (results/ops_forward_wiring/devcheck2/knfl.jsonl vs q2_nfl_rows_dev.csv). A forward
logger that cannot reproduce its own dev rows is scoring a different rule.

### Hidden outcome columns are stripped before an adapter's rows()
edgelab drops HIDDEN columns before rows() runs, so a check comparing the exchange result to ours must live in load();
the first sandbox evaluate died on KeyError 'result'. Every outcome-bearing column (kalshi_result) belongs in HIDDEN.

### Test edgelab forward evaluations on a registry copy
`--registry <copy>` exercises `evaluate --split forward` without writing access rows or pinning an adapter hash in
lab/edgelab_registry.jsonl (the first scored forward evaluation locks the adapter).

### One Kalshi throttle across processes, not per script
Each script had its own 0.25 s sleep (LX1, kx.py) and q1_fetch ran 4 threads, so jobs together could exceed 4 req/s on
the IP the live bots use. scripts/kx_throttle.py: an fcntl-locked timestamp file, 0.30 s + jitter -> <= ~3.3 req/s total.

### Run each new launchd job body once with the deployed interpreter
The launchd venv (~/.claude/quant-sports/.venv) is not ../.venv_quant: the K-CFB model's SBR html parse failed only there
(no lxml). Exit 0 in the worktree proves nothing about the deployed run.
