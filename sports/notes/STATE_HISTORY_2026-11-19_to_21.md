# fund STATE -- Updated: 2026-11-19 (handoff before /clear; full history: notes/STATE_HISTORY_to_2026-11-19.md)
Next: launch the next 3-4 research agents from notes/RESEARCH_QUEUE.md (user: keep agents running; approval on everything);
  fed_vs_futures + mortgage_rates finished and are merged (both dead, LEARNINGS_2026-11.md). Check health digest first.

## Live money (Kalshi; launchd from ~/.claude/quant-kalshi; .env backups *.bak-2026-11-18-precaps)
- E4 weather maker orig7: 24 ct >= .97 / 19 ct <= .96 (KALSHI_PRICE_BAND_CAPS). X1 (17 cities) RESTARTED 11-18: 6 ct, $30
  sub-cap, kill since 11-18 (2 losses/first 30 or ROI<0 after 60). Caps day/open $150, NIGHT_FRAC .375. notes/SIZING_RULE.md
- lead_exec: DUL NFL div unders $2 (KXNFLTOTAL), L39 monthly rain NO (live Nov 24-30), TSA YES-only. NHL late $1 (ingame_exec).
- OFF: NFLF (data/LEAD_DISABLE_NFLF; needs order_guard fix cec89a0 re-check before re-arm), IBKR gateway + ForecastEx exec
  (disabled; user withdrawing ~$65.50 IBKR cash, account stays open).
- Kalshi 11-19: cash $514.64, shard target 0=80 2=15 3=5. USER's own positions (not bots): KXNVDACEO-29JAN-TCOO 651 ct,
  KXNBAGAME-26NOV18BOSDEN-DEN 35 ct. Status: `cd ~/.claude/quant-kalshi && .venv/bin/python scripts/kalshi_exec.py status`.

## Monitoring
- Health monitor hourly (com.jonathanwang.quant-sports-health) -> ~/.claude/quant-sports/digests/health-<date>.md; 47 jobs.
- Scorecards: `cd sports && ../.venv_quant/bin/python scripts/scorecards.py [--kalshi]` (SPRT decision-support panel incl.).
- Stop one job: launchctl bootout gui/$(id -u)/com.jonathanwang.<label>; stop all quant-sports: touch ~/.claude/quant-sports/STOP.

## Forward paper trials logging (read-only launchd; rules + kills pre-registered in each notes file)
K1/K-NFL/K50-NFL/K-CFB (Kalshi openers), NHL-om-A + nhl-goalie, OT1/OT1b (crypto tails), CB1-3 (crypto barriers), E4P-C/CA
(NBM rule C, best lead: +1/3 E4 picks), E4-0600 add-on, ESP1 esports vs Pinnacle, QM1/2 quake, TG1/SD1/SD2 snow, state-gas
books, econ-events books, LX1 (self-stops 11-20), Q013/Q014 RT, series-watch (daily 07:00 PT -> RESEARCH_QUEUE).

## Latest results (detail + numbers: notes/LEARNINGS_2026-11.md, notes/EDGES_AND_AGENTS.md)
- Only proven edge: E4 (sealed E1 +2.87% [+1.98,+3.71]); live maker SPRT LLR ~0 -> years to certify; sizing = judgment.
- Kalshi sports efficient after cost (models ~0.85 pt CLV vs 1.2-1.4 needed); maker fills adversely selected (6 markets).
- 11-18 dead: tropical/NHC, NYC hourly dir, snow (maker prices recent decade), RT review lag, PM skilled wallets, OT1 ext.
- Q004-NHL + L34 killed by rule. K1 n20 +18.7% early. edgelab SCORING v2 + BAR v2 amendment adopted 11-18 (GOAL.md).

## Waiting on user
1. Re-lock GOAL.md: sha f405c49ecec8c6349b08eca9e8f3ba18ed1f7dae50223840b9d1f708e3047db7 into ~/agent-harness/goals/fund.yaml,
   then `sudo ~/agent-harness/install.sh --lock`.  2. Withdraw IBKR cash.  3. Optional: real-time odds feed ($99-198/mo),
   ask Kalshi if retail can quote parlays (identity-acting steps not covered by blanket approval).

## Standing authorizations + traps
User 11-18 "approval on everything" (memory feedback_quant_approval_everything): apply prepared/tested/reversible changes,
log + back up; never identity-acting outward steps. Traps: Kalshi cash is per shard; fills `side` = outcome acquired;
order list lags 1-2 s (trust POST response); journals record intent not fills; check output growth not exit codes.
