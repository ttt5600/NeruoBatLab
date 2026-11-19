# forward_review_1115 -- forward review of every paper trial + live strategy (2026-11-15/16)

Review subagent for the quant-trader lead. Read-only (Kalshi GETs, log reads). No orders, no git, no .env/kill changes.

## Question
After NFL week 10 (Sun 11-15) and the NBA slate settle: where does each forward paper trial and live strategy stand vs
its PRE-SET kill rule, scored from EXCHANGE fills/settlements (Kalshi GET /portfolio/fills, /settlements, market
`result`), not from our own journals? And did every launchd job run and write rows on 11-15?

## Pre-registered judging (written 2026-11-15 16:10Z, before any outcome looked at)
- Status per trial = the kill rule already written for it (no new rules here). Sources:
  scripts/scorecards.py card docstrings + notes/ACT_LIST.md + notes/ops_ot1_forward.md + notes/ops_nhl_goalie.md.
  - K1 (l32b_k1): pass ROI > 0 p<.10 AND CLV > 0 at 100; kill ROI CI upper < 0 after 60 or CLV <= 0 after 100.
  - K-NFL / K-CFB / K50-NFL / nhl_om_a: same 100 / 60 rule as K1 (scorecards.py card text).
  - OT1 / OT1b: kill ROI CI entirely < 0 after 150 bets (spec min_n 150).
  - L32b: CLV kill at 100 / judge 200. L33: CLV judge at 100.
  - NHL late (B5): 10 losses in first 30 live, or preset after 100. DUL (B2): after 40 bets ROI < 0 OR mean CLV < 0.
  - TSA (B4): 2 live losses in first 30 fills; paper: ROI <= 0 or 3 losses in first 30, judge >= 60.
  - E4: whatever rule the E4 card / e4 notes quote (check before scoring).
  - ForecastEx paper: its card's rule if one exists; else descriptive only.
- Every ROI comes with a bootstrap 95% CI; a CI crossing 0 = "not distinguishable". n < 10 -> "too early", no ROI claim.
- Journal vs exchange disagreements are reported as bugs, not reconciled silently.
- Health: a job is STALE if its output file has no row/heartbeat dated 11-15 local when it should have acted
  (schedule from ops docs), or its log shows a non-zero exit / traceback on 11-15.

## Plan
1. Wait until 2026-11-16T05:30Z (background loop).
2. `scripts/scorecards.py --kalshi` (refreshes exchange fill cache), each trial's report command, `edgelab evaluate
   <trial> --split forward` where wired (NOTE: the lead was told to run the FIRST real forward evaluate to pin the
   adapter hash -- check registry for prior access rows; if none, use a sandbox registry copy and say so).
3. Health sweep of ~/.claude/quant-sports/logs + data/kx_forward/*.jsonl + quant-kalshi run.log.
4. Table + headline below.

## Log
- 16:08Z started; inbox `started` row. Waiting for settles.
