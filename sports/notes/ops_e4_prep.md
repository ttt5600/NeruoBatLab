# ops_e4_prep -- X1 restart cap, E4 price-band cap, TSA high-rung skip (agent ops_e4_prep, 2026-11-15)

Ops prep only: worktree code + tests + exact apply commands. Nothing deployed, no .env/launchd change, no orders.
Each change is default OFF: copying the new scripts changes nothing until the .env lines are added.

## Question / acceptance rule (written before coding)
Q: can each of (1) X1 restart at minimum size with its own contract cap + pre-set kill, (2) E4 price-band cap
(24 ct at NO >= .97, 19 ct at <= .96; notes/e4_evidence.md T3, SIZING_RULE.md), (3) TSA live skip of high-rung NO
posts (notes/tsa_fill_review.md) be applied in one step? Accept = flag defaults reproduce today's behaviour (the existing
51 kalshi_exec + lead_exec tests pass unchanged), new tests cover the on-state, full suite shows only the 2 known
tests/test_bet.py failures.

## Starting point (read 2026-11-15)
- Worktree scripts/kalshi_exec.py and scripts/lead_exec.py were byte-identical to ~/.claude/quant-kalshi/scripts/ (diff).
- Deployed .env (non-secret lines): BANKROLL 400, MAX_CONTRACTS 24, MAX_ORDER 24.80, MAX_DAILY 120, MAX_OPEN 120,
  ORIG7_FRAC .062, NIGHT_FRAC .30, COHORTS=orig7, maker, EXEC_MODE=live.
- kalshi_exec had NO per-cohort contract cap: max_contracts is global; X1 sizing is TIER_FRAC x1 = 0.005 x B (hard-coded).
  So re-enabling X1 today with no code change gives $2.00/order = 2 ct (results/ops_e4_prep/preview_cfg.py output).

## Wrong turn caught: X1 would crowd orig7 out of the shared day/open caps
kalshi_paper.jsonl since 11-04: orig7 1-7 picks/night, X1 8-15 picks/night, both logged 03-06 UTC interleaved (one-off
count printed in this session). At 6 ct X1 is up to ~$87/night; orig7 alone already reaches the $120 day cap at ~5 picks.
With only KALSHI_X1_MAX_CONTRACTS, X1 picks logged first would take orig7's room. Added KALSHI_X1_DAY_USD (sub-cap on
both X1 spend today and X1 open, inside the shared caps) and recommend raising the shared caps by the same amount.

## What was implemented (worktree)
scripts/kalshi_exec.py (patches: results/ops_e4_prep/patch_kx.py, patch_kx2.py):
- KALSHI_X1_FRAC (default 0.005 = old TIER_FRAC), KALSHI_X1_MAX_CONTRACTS (default none), KALSHI_X1_DAY_USD (default none).
- KALSHI_PRICE_BAND_CAPS="0.97:24,0:19": highest floor <= order price (maker q / taker ask) caps contracts; default off.
- contract_cap() = min(global, X1 cap, band); dry stays uncapped as before; journal gets ct_cap only when it differs.
- X1 kill (only evaluated while x1 is in KALSHI_COHORTS): settled X1 fills placed >= KALSHI_X1_KILL_SINCE -> >= 2 losses
  in the first 30, or ROI < 0 once >= 60 -> journal row event=cohort_kill (sticky per SINCE), notification, X1 picks
  -> skip_cohort_killed; orig7 unaffected. Does NOT write KILL.
- preflight prints a "cohorts / contract caps" line.
scripts/lead_exec.py (patch_lx.py): .env LEAD_TSA_SIDES (default yes,no). =yes -> TSA paper posts with side no are
journaled skip_tsa_side_off at $0 (not counted in the $8 week cap). Paper logger untouched. L39/DUL untouched.
Tests: tests/test_kalshi_exec_caps.py (16), tests/test_lead_exec_tsa_sides.py (5).
Full suite: 516 passed, 2 failed = the known tests/test_bet.py pair.

## Offline preview (results/ops_e4_prep/preview_cfg.py: deployed .env + overrides, no network)
- Today: orig7 24 ct at every q; x1 (if enabled) $2.00 -> 2 ct.
- Change 1 lines: x1 $6.00 -> 6 ct at q .90-.98; orig7 unchanged 24 ct; day/open $150.
- Change 2 lines: orig7 19 ct at q .90/.94/.96 ($17.10-$18.24), 24 ct at .97/.98.

## APPLY COMMANDS (user's yes; each change needs the script copy once, then only its .env lines)
Step 0 (once, covers all three; behaviour unchanged until .env lines are added):
  cd ~/.claude/quant-kalshi/scripts && cp kalshi_exec.py kalshi_exec.py.bak-2026-11-15-precaps && cp lead_exec.py lead_exec.py.bak-2026-11-15-precaps
  cp ~/Desktop/vocalizations_lab/.claude/worktrees/quant-trader/sports/scripts/{kalshi_exec,lead_exec}.py ~/.claude/quant-kalshi/scripts/
  cd ~/.claude/quant-kalshi && .venv/bin/python scripts/kalshi_exec.py preflight   # -> "cohorts / contract caps ... price bands off"
Change 1 (X1 restart, 6 ct, own $30 room, kill armed): in ~/.claude/quant-kalshi/.env
  KALSHI_COHORTS=orig7,x1        (replace orig7)
  KALSHI_X1_FRAC=0.015           ($6 at B 400)
  KALSHI_X1_MAX_CONTRACTS=6
  KALSHI_X1_DAY_USD=30           (~5 X1 orders/night)
  KALSHI_X1_KILL_SINCE=2026-11-16
  KALSHI_MAX_DAILY_USD=150 ; KALSHI_MAX_OPEN_USD=150 ; KALSHI_NIGHT_FRAC=0.375   (orig7 room unchanged at $120)
  Verify: preflight caps line shows x1 $6.00 and "x1 max ct 6 (kill since 2026-11-16), x1 day/open $30.0"; after the
  next night: grep '"cohort": "x1"' data/kalshi_exec.jsonl | tail -> count <= 6, ct_cap 6. Revert: KALSHI_COHORTS=orig7.
  Minimum-size alternative with no new code: only KALSHI_COHORTS=orig7,x1 -> 2 ct/$2 (no kill, no sub-cap).
Change 2 (E4 price bands): KALSHI_PRICE_BAND_CAPS=0.97:24,0:19
  Verify: preflight "price bands >=0.97:24, >=0.00:19"; journal maker rows with maker_q <= .96 show count <= 19, ct_cap 19.
Change 3 (TSA live NO legs off): LEAD_TSA_SIDES=yes
  Verify: next TSA post week, data/lead_exec.jsonl NO rows -> status skip_tsa_side_off, cost 0; YES rows post as before.
  Already-resting 26NOV15 high-rung NO orders are NOT cancelled by this (they expire at market close).
Rollback any: delete the .env line(s); full rollback: cp the .bak-2026-11-15-precaps files back.

## Learnings
### kalshi_exec had no per-cohort contract cap; X1 sizing was a hard-coded 0.5% of bankroll
TIER_FRAC x1 = 0.005 -> $2 at B 400 = 2 ct; KALSHI_MAX_CONTRACTS is global (scripts/kalshi_exec.py config/caps).
Re-enabling a cohort silently uses the hard-coded fraction, so check per-cohort $ caps before any restart.
### Adding a cohort under a shared daily cap can starve the proven cohort
X1 logs 8-15 picks/night vs orig7 1-7, interleaved 03-06 UTC (data/kalshi_paper.jsonl since 11-04); orig7 alone nearly
fills the $120 cap. Give a new cohort its own sub-cap and raise the shared cap by that amount.
### A sticky kill keyed on a restart date lets a cohort be re-armed deliberately
cohort_kill rows carry KALSHI_X1_KILL_SINCE; a new SINCE = a new restart, so pre-pause fills neither trip nor excuse it.
