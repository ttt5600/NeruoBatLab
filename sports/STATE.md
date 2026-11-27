# fund STATE -- Updated: 2026-11-27 ~00:30Z (session 11-19..27; previous STATE: notes/STATE_HISTORY_2026-11-19_to_21.md)
Next: both 11-24 agents merged (E4C paper arm live from 11-25; I4 dead). Then: KFG1 Thanksgiving (11-26) + Sun
  11-29 live picks (auto, exp_exec); DK claim ~11-28 (one favourite .94-.97); weekly forward scorecards; next agents from
  STRATEGY_v2 / IDEA_PATTERNS "where to look next".

## 2026-11-27 00:30Z
- NEEDLE SETTLED EARLY: KXDHOUSESEATS-27-249 finalized YES 11-26 16:18Z, 87 ct x $1 = $87.00 paid (bought 11-04 @ .963 -> ~+$3.2).
  No sale needed. Exchange 00:20Z: balance $296.97 (shard 0 $213.10, shard 3 $83.87) + open exposure $46.68 = real bankroll
  $343.65 -> step-3 gate ($370) short by $26.35. Deposit ~$30 -> apply the pre-registered SIZING_RULE step-3 diff.
- STEP 3 READY: sports/scripts/step3_apply.py (check / --go / --wait). hub verify change_size = human (exit 3) -> the USER runs
  the --go line after depositing ~$30 (00:40Z bankroll $342.15, short $27.85). Script refuses below $370; backs up .env first.
- KFG1 Thanksgiving: CHI@DET CHI2 NO won (+$0.52); PHI@DAL DAL2 NO x4 @.41 + PHI2 YES x4 @.45 filled live (open); KC@BUF skipped (edge < .04).
- User 11-26: never had Kraken/Coinbase/eToro/Rakuten; will do the bonuses. Rakuten offers re-verified 11-27 00:20Z (Kraken $150,
  eToro $50, Coinbase $12.50). Playbook with fixed exits: notes/PROMO_TRACKER.md "Playbook handed to the user 11-26".
- 11-26 reconcile_mismatch on KXHIGHDEN-26NOV25-B53.5/B55.5: benign ordering at settle (journal filled 24 each, settle rows
  +$0.72/+$0.48 written same tick; exchange fills = our NO maker buys at .97/.98).

## Live money (Kalshi; launchd from ~/.claude/quant-kalshi)
- E4 live (journal, mode=live, 11-25 04:25Z): 142 settled / 139 wins, +$22.34 (orig7 62 / 1 loss / +$26.01 = +6.18%; X1 80 / 2 /
  -$3.67). Since step 2 (11-12) orig7 13 settled, 0 losses, +$22.64, 14/28 maker orders filled. X1 since 11-18: 10, 1 loss, kill not
  fired. (Portfolio's +$37.27 = live + 15 dry settles.) Needle KXDHOUSESEATS-27-249 87 ct ($83.78) open to Feb 2027, outside journal P&L.
- STEP 3 PRE-REGISTERED (SIZING_RULE.md 11-25): 24 -> 48 ct at c >= .97 from 11-26 if depth check passes (q5_depth_step3, runs
  06:12Z 11-25) AND real bankroll >= $370 (now $245; user must add cash). User 11-25: 'time to up the weather ... be aggressive'.
- 11-24 06:47Z: shard 0 funded ($113.57) after the user set target_balance_allocation 0=100 and deposited; $83.87 still
  sits in shard 3 (allocation did not move existing cash this time; new money lands in 0). Sizing per SIZING_RULE.md: NO
  step-up now (24 ct = depth ceiling; real E4 bankroll ~$250 < configured KALSHI_BANKROLL_USD 400; next step >= 11-26 and
  needs new depth evidence + bankroll >= ~$370 for 48 ct at 1/4 Kelly).
- EXPERIMENT BUDGET $50 (notes/EXPERIMENT_BUDGET.md): spend $24.66. OpenRouter T238 LOST -$14.08 (Kalshi settled on the
  displayed BYOK-excluded total 210.11T). KFG1 live $2 IOC per pick via exp_exec in the kfg1 job: 11-22 7 fills, 5 won, +$2.01.
  Stop: touch ~/.claude/quant-kalshi/data/EXP_STOP. Settlement check: sports/scripts/exp_settle_probe.py.
- DraftKings Predictions (user's account, side-hustle session operates it): credit trades pay the FULL payout (Tulsa $49.92 PD
  -> $52.00 cash). Claims ~11-28 / 12-05 / 12-12 (168 h expiry): one favourite .94-.97 each, unhedged.

## Jobs / ops
- source-shift watcher installed (hourly); AAA rollover guard; Q014 Roll Call endpoint fixed (count 58 = RSS 58).
- DNS outage 11-23 ~10:00-18:51 PT broke every launchd job (NameResolutionError); recovered at the 19:03 PT tick. No code fault.
- LX1 scanner window over (KILLED); its job still writes a daily report (user may unload). early-form idle (optional unload).
- MC1: CK API now Cloudflare-blocked (11-25) -> scanner runs SIGNAL-ONLY (n_signal 18 on 11-25 data); naming needs another route.
- MC1 T-MTG paper phase LIVE 11-25 (launchd quant-sports-mc1, daily >= 06:30 PT; data/mc1/daily.jsonl): judge 12-02 on median
  n_signal >= 15/day, then the USER spot-checks 10 flags (>= 6/10 real NM listings) before any <= $100 batch. notes/small_capacity_census.md.
- odd-lot tender watcher SCHEDULED 11-25 (launchd quant-sports-oddlot, daily >= 07:15 PT; open offers show in the
  SessionStart hook). MYPS Dutch $4.30-5.50 open to 12-10, price $4.93 (11-24) > floor -> not riskless; watch.

## Agents (notes/RESEARCH_QUEUE.md) -- running 11-25: boost_hedge (existing-customer promos + cross-venue arb), pokemon_cards
  (Triumph/Rips pack EV, MSRP vs secondary). promo_hunt_2 DONE: Kraken via Rakuten $150 (if new), eToro $50; PROMO_TRACKER priority table.
- e4_coldmargin DONE 11-24: LEAD +4.07u [-0.002,+8.22] n217 extra picks, placebo 81.6th pct (fail). PAPER arm deployed in the
  quant-sports e4p job (e4c_cold.jsonl, from 11-25); kill roi_ci_below_zero after 60 fills; decide at 150 fills (~3 mo).
- wx_bot_flow DONE 11-24: I4 dead (bot flow not identifiable; bot-BUY brackets fair; E4's premium = diffuse YES demand).

## Results since 11-21 (EDGES_AND_AGENTS.md "Lead scoring 2026-11-23"; IDEA_PATTERNS P-1/P-4)
- ~30 ideas tested this month; only E4 (live) and KFG1 (forward, n tiny) alive. Dead 11-21..23: UFC JSON latency (JSON trails the
  broadcast ~20 s), LX1 logical arb (1 distinct set vs bar 5), OpenRouter definition bet, props, E4 data_v2 T1-T3, tennis, eSoccer.
- P-4: Kalshi follows the NEW source definition 5/6 [0.442, 0.981] -> bet with the display, not with history.

## Waiting on user
0. Metaculus bot BUILT (sports/metaculus_bot/README.md, $0, tests pass): user creates bot account + token, credits form, AskNews;
   then smoke -> dry run -> bot-testing-area -> MiniBench. Needle exit ticket ready (sell 87 KXDHOUSESEATS-27-249 @ .996 -> +$86.62
   to shard 0; hub verify: user's call) -- user reply 'sell it'.
1. Re-lock GOAL.md (hub doctor FAIL). 12-month goal unchanged until re-lock. 2. vault.yaml windows for forward trials (incl. kalshi_temp_e4c_fwd text in notes/e4_coldmargin.md).
3. Optional: unload lx1 + early-form jobs; promos needing sign-up (money_now.md: Gemini ~$38,
   ProphetX ~$36.5, PM US $15.30, Novig $45 unverified, Webull ~$18.7) -- identity steps are the user's.

## Standing authorizations + rules
"Approval on everything" (prepared/tested/reversible) + $50 experiment budget; never identity-acting steps; never bring
browser windows forward (no `open <url>`); no DK trades / Kalshi hedges without the user's OK. Push over HTTPS
(`git -c http.version=HTTP/1.1 -c http.postBuffer=157286400 push https://github.com/ttt5600/NeruoBatLab.git worktree-quant-trader`;
SSH 22 blocked; plain HTTP/2 push dies with 'unexpected disconnect while reading sideband packet' on 11-25); never force-push.
Traps: Kalshi cash per shard; close_time != event time; /historical `close` dollars vs live `close_dollars`; first-print
settlements; robots.txt (Vercel /api/ disallowed).
