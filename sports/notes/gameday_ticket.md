# gameday_ticket: readiness check for 2026-11-15 (subagent, ops; no orders, no git)

Question: will every live and paper strategy that can fire on the 11-15 NFL slate and the 11-15/16 NBA slates actually run,
and what will each one do? No pre-registration is needed, because this is an ops audit and estimates no edge.
Output: notes/tickets/2026-11-15.md (per-game table + user list).

## What I checked (all read-only)
- launchd: `launchctl list`. Loaded: lead-exec, kalshi-paper (runs kalshi_exec), live-signals, open-lines,
  sports-{kx-knfl,kx-k1,l32b,early-form,nfl-fade,nhl-ingame,ticket,kx-openers,lx1}.
  **Not loaded: ibc-gateway, forecastex-exec** (forecastex_exec.log still updates, but every run says "NOT TRADING").
- Logs (tail): lead_exec.log 01:23 PT; exec.log 01:03; open_lines.log 00:39; quant-sports/logs/*. Heartbeats for nfl-fade and nhl-ingame read 01:28 PT.
- Kalshi GET only: results/gameday_ticket/ro_balance.py (balance, positions, resting orders), ro_shards.py / ro_shards2.py
  (market exchange_index, fills since 11-13), ro_dul_preview.py (imports the deployed lead_exec functions: dul_games at
  kick-60, pick_strike now, budget_left; writes nothing). f2_rain_forecast.py (Open-Meteo forecast for outdoor stadiums).
- Deployed lead_exec.py == worktree copy (diff clean).

## Findings
1. **Shard-0 cash $4.81 out of $604.39 total.** $570.73 sits on exchange_index 2 and $28.84 on 3. Every NFL, weather, TSA and
   NHL market is on index 0; NBA is on 3; KXBTC15M is on 2. The drop is at 11-13 21:55Z ($595.70 -> $6.73 in lead_exec.log)
   and matches a ~$500 KXBTC15M-26NOV131700-00 buy/sell round trip (GET /portfolio/fills) that is not in any bot journal.
   Consequences, confirmed in the journals:
   - E4 `insufficient_balance` x2 at 11-15 05:03Z (kalshi_exec.jsonl).
   - NHL `skip_cash` x6 (lead_exec.jsonl, 11-14 22:53Z to 11-15 05:45Z).
   - DUL preview: budget_left -$75.19, binding=cash, on all 6 division games.
2. Today is NFL **week 10** (schedule), not 11. The division games are MIN@GB, CAR@NO, BUF@NYJ, JAX@TEN (1 pm), LA@ARI
   (4:05), PIT@CIN (SNF). DUL strikes nearest 50c at 09:28Z: 42.5/48.5/50.5/46.5/38.5/49.5, with NO asks .49-.52 (all <= .56).
3. DUL first live bet: WAS@NYG 11-12, 3 NO @ .47 on 41.5. Final 24-21 (45), so it lost.
4. nfl_ingame_fade.py `interim` crash: the deployed root has no data/pbp, so p.write_bytes raised FileNotFoundError. Fixed in
   the worktree with `p.parent.mkdir(parents=True, exist_ok=True)`. Tested with PBP pointed at a missing tmp dir:
   23,302 rows. The worktree interim run gives 114 bets / 36 games, units -5.07 [-8.30, -1.99], ROI -10.1%. F1 is heading to FAIL.
5. K-NFL has 0 rows by design (FWD_FROM 11-12 > week-10 L33 pick times 11-09/10). K50-NFL decided week 10 on 11-13:
   9 paper bets. K1 has 3 paper bets tonight, plus 2 with no quote.
6. F1 NBA early-form is finished: every team has played more than 8 games, and the last pick was 11-06.
7. F2: HOU@CLE has a 1.1 mm forecast in the 3 h before kick, which could fire a paper NO. Every other outdoor game forecasts 0.0 mm.
8. The lead-exec plist fails strict XML parsing (`--` inside a comment), but launchd loaded it anyway.
9. IB Gateway has refused connections since 11-14 23:45 PT, so E5 is not trading.

## Proposals (user / lead)
- User funds shard 0 before 08:30 PT. Consider changing `shard0_cash` gating to warn loudly, e.g. a notify when
  shard-0 < reserve while other shards hold cash. Today the bots skip silently.
- Deploy the nfl_ingame_fade.py fix (paper).

## Learnings
### Kalshi cash is split by exchange shard, and our bots only see shard 0
GET /portfolio/balance balance_breakdown showed $4.81 on index 0 against $570.73 on index 2 (11-15 09:27Z, ro_balance.py).
NFL, weather, TSA and NHL markets are index 0, NBA is 3, and KXBTC15M is 2. A trade on another shard settles there, so total
balance can look healthy while every bot skips for cash. Check the breakdown, not balance_dollars.
### A cash shortfall shows up as quiet skips, not errors
E4 logged `rejected` (insufficient_balance) and NHL logged `skip_cash`. Nothing alerted for about 36 h after the 11-13 21:55Z drop.
Add a balance-by-shard line to the morning brief or scorecards.
### Deployed roots lack data dirs that the worktree has
nfl_ingame_fade.py wrote into ROOT/data/pbp, which exists in the worktree but not in ~/.claude/quant-sports. It worked in
dev and crashed in deploy. Create output dirs with mkdir(parents=True) before writing.
