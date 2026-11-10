# ACT LIST -- what we can put money on now, including the not-proven ones (2026-11-05)

User 11-05: "add to our list of edges even if some aren't 100% -- we should be able to act on something."
This is NOT an edgelab verdict. edgelab (GOAL.md) passes exactly one strategy; everything in tier B is a LEAD that the
USER chooses to bet at judgment size under the 10-31 bet policy ($1-3/bet, own strategy tag in bet.py, never sized up
on a hot streak). The point of tier B is that small real bets ARE the forward test: each has its kill rule written
here BEFORE the first bet, and is judged on its own tag, never pooled.

Numbers below come from the deployed logs read 11-05 (~/.claude/quant-kalshi/data/, ~/.claude/quant-sports/) and the
11-04 pm digest; file per line in EDGES_AND_AGENTS.md (11-05 entry).

## Tier A -- passed edgelab, live
| ID | Action | Evidence | Size / gate |
|---|---|---|---|
| E4 orig-7 | Kalshi daily-high weather NO, maker, orig-7 cities (executor does it; nothing to do by hand) | sealed 1,581 fills +2.87% [+1.85,+3.84]; live 102 settled 100-2, -$1.10 (fills >= .90: 97-1 +$0.90; < .90: 3-1 -$2.00) | caps bind, not cash ($177.52). Bigger size needs the user's A/B choice (HANDOFF 11-03 addendum) |

## Tier B -- act small (user's call, judgment tier)
| # | Lead | Exact action | Why it makes the list | Stake / cap | Kill (pre-set 11-05) |
|---|---|---|---|---|---|
| B1 | L39 monthly rain/snow "unreachable" NO, MAKER | Nov 24-30, 22:00 station-local: logger prints picks (`kalshi_wx_monthly_logger.py report`, picks file). Post NO bid at min(floor NO mid, NO ask - 1c); rest to close; no taker fallback | dev +3.6%/146, 1 loss, p .016; forward 22-0, maker +3.9% (15/22 filled), taker +1.3% | $1-3 per bracket, <= $15 per month (capacity ~$7.5/mo anyway) | any 2 losses in the first 100 picks (same as its prereg) |
| B2 | DUL NFL division-game UNDERS, week >= 10 | Week 10 = 11-12..11-16: WAS@NYG (Thu), MIN@GB, CAR@NO, BUF@NYJ, JAX@TEN, LA@ARI, PIT@CIN. Bet the under at the best total available the day of the game; log CLV vs close | dev 2009-19 623 bets +9.81%, 8/11 seasons positive, p .0098 -- but post hoc and edgelab family bar says dead (registry lines 49-50) | $2 per game, ~7/week, <= $60 season | after 40 bets: ROI < 0 OR mean CLV < 0 -> stop |
| B3 | F1 NBA early-form sides | picks appear 60-90 min before tip (launchd early-form logger / daily ticket) | forward 28 bets +3.6% [-28,+36] (not distinguishable); user journal 5-3 +$3.22 | $2 per pick | judge at 100 bets: ROI CI upper < 0 or CLV < 0 p<.05 |
| B4 | TSA weekly MAKER NO (KXTSAW) | same rule as tsa_t2_maker paper (8 orders/week posted by the TSA logger); mirror them as real $1 bids | dev 89 fills +8.4%, sealed 24 fills 0 losses +6.1%; edgelab: dead on concentration (one shutdown week) | $1 per order, <= $8/week | 2 losses in first 30 fills |

**Execution (user 11-05 "Both"):** B1/B2/B4 are placed on Kalshi by `scripts/lead_exec.py` (LIVE since 11-05 17:25Z,
launchd com.jonathanwang.quant-lead-exec every 15 min, deployed ~/.claude/quant-kalshi; gate .env LEAD_EXEC_MODE=live;
journal data/lead_exec.jsonl; stop all = touch data/LEAD_KILL, one = data/LEAD_DISABLE_<L39|TSA|DUL>). Lead open <= $40,
keeps >= $80 cash for weather. DUL: series KXNFLTOTAL, NO on the strike nearest 50c, 30-90 min pre-kickoff, NO ask <= .56.
TSA mirrors the paper side (YES or NO). bet.py does NOT see these bets yet. B3 F1 stays user-placed on RH.
E4 sized 2x 11-05 (edgelab early step; caps $6.20/6 ct/day $60/open $80; loss stops unchanged).

## Tier C -- watch only (forward data says no for now)
- L33 NFL vs opener: 3 settled (2-1). Too small; keep the logger.
- L34 NHL totals slice 19-17-1 +6.0% is POST HOC (ML side 13-26 -24.8%); not betting a slice picked after looking.
- E5/FX1 ForecastEx weather: STOP PROPOSED 11-10 (notes/e5_forecastex_audit.md). Maker fills adversely selected (every
  losing station-day filled, 0/200 unfilled days lost); asks on 6.9% of picks. Q008 KILLED by rule (0 qualifying/56).
  Also: bet.py loss stop blocked 373 FX picks on OTHER strategies' ~2.9u open risk (still daily after 11-07 fix);
  26 orders from 10-29 have no fill/settle rows -> check IBKR statement.

## Off the list 11-05 (forward data negative)
Kalshi props (CLV -1.5c [-2.5,-0.5], n 115) | L32 totals vs opener 17-20 -8.9% | Q004 first-tight-quote NHL -27%/39,
NBA -24.6%/25 | L34 NHL ML -24.8% | L40b LIP interim kill both fill models | Q008 0 qualifying quotes (kill < 25%).

## Open ops items found 11-05
- 3 ForecastEx LAS/LAX maker orders from 11-04 have no cancel/settle row; executor lost IB Gateway :4001 from
  11-04 23:48 PT. Check at IBKR, re-login gateway (`launchctl kickstart -k gui/$UID/com.jonathanwang.ibc-gateway`).
- 11-05 ForecastEx picks all refused by the loss stop on "open B+C risk 2.92u" (non-FX positions, likely needle bets).
- DUL status conflict: registry dead (historical_import) vs FUND_BOOK LEAD. Listed here as a user-judgment lead only.

## In-game small real money (user 11-07: "small amounts of real money for our NHL and other NFL games") -- ARMED 11-07
- B5 NHL last-15-min taker (mirrors paper trial nhl_ig_t1_late_taker): $1/bet, $5/day, $40 season; kill 10 losses in first 30 or preset after 100.
- B6 NFLF live big-play maker fade (scripts/nfl_fade_live.py = live PROXY of F1, ESPN WP swing >= .08; not the pre-registered F1): $2/order, $10/game-day, $40 total; kill P&L <= -$15 or 8 losses in first 20.
- Executor scripts/ingame_exec.py (be3bded), flags LEAD_LIVE_NHL=1 / LEAD_LIVE_NFLF=1 in ~/.claude/quant-kalshi/.env (backup .env.bak-2026-11-07-preingame).
  Stop one: touch ~/.claude/quant-kalshi/data/LEAD_DISABLE_NHL (or _NFLF); stop all: ingame_exec.py kill "reason".
