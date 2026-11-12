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

## Forward paper -- next Kelly candidate (11-11)
- **L32b-K1 NBA totals vs opener ON KALSHI** (edgelab `l32b_k1_kalshi_taker-8d6fb950`, notes/nba_l32b_exec.md): Kalshi
  KXNBATOTAL ladders open ~3.4 h before Pinnacle and price the opener (0.33 pts from open, 1.82 from close). Rule: each L32b
  pick, Kalshi strike nearest the Pinnacle open, bet only if model - c - strike >= 3 pts our side; YES over / NO under at
  the ask + taker fee. Dev-on-logged 27 picks ROI +1.9% [-32.6,+36.3], CLV +1.37 pts [+0.35,+2.43]. Judge at 100
  (~4-5 wk at ~23/wk): pass ROI > 0 p<.10 and CLV > 0; kill ROI CI < 0 after 60 or CLV <= 0 after 100.
  Score with every scorecard refresh: `cd sports && ../.venv_quant/bin/python results/nba_l32b_exec/q2_fetch.py &&
  ../.venv_quant/bin/python results/nba_l32b_exec/q2_score.py forward`. Live money only after it passes (Kelly rule).

- **K-NFL / K-CFB (11-12, notes/kalshi_openers_xleague.md):** Kalshi lists early + sits on the opener for CFB (~6 d),
  CBB (~20 h), NHL (~32 h); NFL gate fails vs ESPN's lookahead "open". K-CFB (frozen L36 totals) dev n63 CLV +1.49 pts
  [+0.86,+2.13], ROI -2.1% [-23.9,+20.3] (avg cost .577 -- strike at the DK open is already off 50c); K-NFL n14 CLV +0.79
  [+0.18,+1.43]. Edgelab knfl_l33_kalshi_taker-445b123e / kcfb_l36_kalshi_taker-613b7d62 (unwired, forward paper). Next
  prereg: model gap vs Kalshi's 50c strike. CBB totals = best timing gate but no model yet.

## Tier C -- watch only (forward data says no for now)
- L33 NFL vs opener: 3 settled (2-1). Too small; keep the logger.
- L34 KILLED by its pre-set rule 11-10 (110 settled, CLV -0.0005 [-0.0044,+0.0033]); totals slice stays post hoc.
- SCORECARDS (11-10): `cd sports && ../.venv_quant/bin/python scripts/scorecards.py [--kalshi]` -> notes/SCORECARDS.md,
  one card per strategy vs its pre-set kill. Near-kill: F1 (CLV vs ASK -0.0068 [-0.0120,-0.0016] n29; vs mid n.s.;
  rule doesn't say which CLV -> user to pin), Q004 (NHL -27.6%/76). Lock-in 99c: DEAD 11-11 (notes/nba_lockin_fill.md):
  0/188 open books with queue <= 5k (the "21%" were closed markets); EV +0.58c vs 1c fee; lock-queue logger unloaded
  (plist kept; reload: launchctl bootstrap gui/$UID ~/Library/LaunchAgents/com.jonathanwang.quant-sports-lock-queue.plist).
  Proposal: retire edgelab lock_t2_nba_maker99 (sealed stays closed).
- **E5 LIVE PROBE with 06:00 cutoff (user yes 11-12):** bids cancelled at 06:00 station-local on the event day (before the
  07:30-13:00 forecast-update sellers; notes/e5_fix.md F1). Deployed 11-12 (forecastex_exec.py STATION_TZ/local_cutoff;
  backup .bak-2026-11-12-pre0600; revert: FX_CUTOFF_LOCAL_HOUR=off). KILL (preset 11-12): any loss in the first 15 fills,
  or ROI < 0 after 30 fills -> stop E5 live. ~0.5 fills/day -> ~2 months to 30. IBKR started ~$60, $65.50 on 11-12.
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
