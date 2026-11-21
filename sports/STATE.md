# fund STATE -- Updated: 2026-11-21 ~12:00Z (session 11-19..21; previous STATE: notes/STATE_HISTORY_2026-11-19_to_21.md)
Next: merge the 4 running agents (below) from notes/findings_inbox.jsonl; score settlements: Tulsa DK credit (11-21, decides
  SR vs SNR credit policy, notes/money_now.md), KFG1 first live $2 picks (Sun 11-22), OpenRouter T238 (11-23 10:00 ET).

## Live money (Kalshi; launchd from ~/.claude/quant-kalshi)
- E4 orig7 maker + X1 ($30 sub-cap): live 133 settled / 130 wins, P&L +$19.52, 7d +$14.44 (`kalshi_exec.py status`, 11-21).
  orig7 not cap-bound since 11-03; X1 picks hit its $30 cap 6-9/day by design. DUL/TSA/NHL-late as before; DUL took U45 IND@HOU.
- EXPERIMENT BUDGET $50 (user 11-21, "be aggressive"; notes/EXPERIMENT_BUDGET.md; journal ~/.claude/quant-kalshi/data/exp_budget.jsonl):
  OpenRouter KXTOKENUSE-26NOV23-T238 YES 180 @ .0735 = $14.08 (likely loses: Kalshi followed NEW definition 4/5, source_shift);
  KFG1 $2 IOC per pick via scripts/exp_exec.py inside the kfg1 launchd job (cap = $50 total spend). Stop: touch data/EXP_STOP.
- DraftKings Predictions (user's own account, side-hustle session operates it): $50 credit on Tulsa 52 ct, UNHEDGED by user choice;
  3 more $50 credits ~11-28/12-05/12-12. Policy: SR -> favourite unhedged; SNR -> .10-.25 dog hedged on Kalshi (money_now.md).

## Monitoring / jobs added this session
- kfg1 (KFG1 forward paper + exp_exec live), installed by user. source-shift watcher: job case deployed; launchd NOT installed
  (user: `bash sports/scripts/install_source_shift_agent.sh`). health: gas-books max_gap 14->20h.
- odd-lot tender watcher scripts/oddlot_watch.py built, not scheduled.

## Running agents (notes/RESEARCH_QUEUE.md)
e4_data_v2 (E4 headroom, CLI DGP, forecast backfill, above-forecast losers 66/70), esoccer_tennis, props_forward_1121, ufc_decision_gap.

## Session results (EDGES_AND_AGENTS.md, LEARNINGS_2026-11.md, STRATEGY_v2.md, IDEA_PATTERNS.md)
- ~25 ideas tested, all dead except KFG1 (dev +23% [-0.7,+47.3] n56, forward). Only free-official-source nowcasts survive sealed
  (sealed/dev 0.664 vs 0.162 for models; winners_transfer). Fastest public page sets price. Pool census: Kalshi loser money is
  in in-play sports + 15m crypto ($659M/30d); weather $1.49M; wide far rungs -> pool goes to the quoting maker.
- PM US taker fee is .095 (docs corrected). DK Predictions fee schedule in notes/venues.md.

## Waiting on user
1. Re-lock GOAL.md (hub doctor FAIL; sha in notes/STATE_HISTORY). User asked for 2x in 1 month: explained ~1-3% base rate;
   12-month goal unchanged until they re-lock. 2. Install source-shift watcher. 3. vault.yaml windows for forward trials.
4. Branch not pushed (push blocked by permission classifier): `! git push origin worktree-quant-trader`.

## Standing authorizations + rules
"Approval on everything" (prepared/tested/reversible) + $50 experiment budget; never identity-acting steps; never bring
browser windows forward (memory feedback_no_tab_focus_during_user_activity). Traps: Kalshi cash per shard; close_time != event
time; /historical `close` dollars vs live `close_dollars`; first-print settlements; robots.txt (Vercel /api/ disallowed).
