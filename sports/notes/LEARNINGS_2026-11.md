# Quant-trader learnings, November 2026 (one heading per learning, for the memory vault)

Source session: quant lead "automated trading system development", 2026-11-09..11. Each learning cites its evidence.
Detail lives in notes/EDGES_AND_AGENTS.md ("11-10 wave 5" and later lines) and the files named below.

## Execution and market microstructure

### Resting maker bids on event-day markets fill only when the event is going against you
ForecastEx weather (E5): every losing station-day filled, 0 of 200 unfilled days lost (p 7.4e-6 at station-day level);
Kalshi daily rain (RD2): 9/9 losses filled vs 0/70 unfilled (p .0014). A forecast edge at mid/last-trade is not a trading
edge. Before going live with any maker strategy, compare outcomes of filled vs unfilled orders. E4 daily-high survives
because its brackets are far from reachable. Sources: notes/e5_forecastex_audit.md, notes/raind_daily_rain.md.

### Kalshi's order list lags a new order by about 1-2 seconds
27/27 live entry POSTs were missing from GET /portfolio/orders on the immediate lookup and present 1-2 s later. The POST
response already carries `fill_count` (no `_fp` suffix). Reading "not listed" as "not filled" caused the NFLF runaway
exit (new client_order_id every tick). Rule: take fills from the POST response and position change; a lookup miss is
UNKNOWN; never issue a new coid for an unconfirmed action. Source: notes/incident_find_order_miss.md; fix cec89a0.

### Executor journals record intent, not fills -- reconcile from the exchange
The NFLF runaway ran on 11-08 as well as 11-09 (70 sells vs 8 intended exits; 49 vs 3), hidden for a day because the
journal booked clean exits (-$2.80 "record"). Rebuilt from exchange fills: strategy 4-7 -$0.23, runaway +$63.47 by luck.
Replicated independently. Sources: notes/SCORECARDS.md, notes/post_trade_review_2026-11-10.md, results/replicator_nflf/.

### A shared exchange account lets one bot cancel another bot's orders
6 E4 weather orders were cancelled externally on 11-04 at the needle bot's cancel-all times; all 6 would have won.
Give each bot its own order scope, never cancel-all. Source: notes/e4_live_audit.md.

### One ForecastEx fill is booked in two journals
UHATL_110626_75 (+$5.40) is in both bets.jsonl (E5) and forecastex_exec (E5X); cross-journal P&L sums double-count it.
Source: memory-vault session report 2026-11-10.

### Kalshi fills: `side` is the outcome you acquired; `action` is not buy/sell from your view
E4 NO buys appear as action=sell side=no. Rule verified 11-12: YES delta = +count if side=='yes' else -count reproduced
3/3 open positions (HOUSEME2 -46, KXHIGHCHI -12, KXHIGHPHIL -12) and nets JT-IND to 0; reading `action` literally gave
+142 for a -46 position. Each fill = acquire `side` at `<side>_price_dollars`; P&L per fill = count*(payoff(side)-price)
-fee, which reproduced the exchange's election settlements exactly (-$143.34). (NFLF replicator used action on
KXNFLGAME buy-yes/sell-yes fills and also matched -- check per series before trusting either reading.)

### ForecastEx informed sellers trade on morning forecast updates, not on the thermometer
IBKR reqHistoricalTicks gives timestamped ForecastEx trades (contracts vanish ~1 week after expiry; walk 12 h windows).
All 6 tick-covered losers were sold through our bid 07:30-13:00 local while obs were 3-17F short -> an observation-
triggered cancel is too late (-10.8%); lower bids only drop winners (losers trade to ~2c). Kalshi same-city markets are
not a proxy (fill 67-77% vs ~8%). Source: notes/e5_fix.md.

### The election needle lost by SELLING, not by entering
Rebuilt from all 655 fills: settled election P&L -$143.34; had every closing fill been skipped (hold each opening buy
to settlement) it would have been about +$52.74 -> exits cost ~$196. Kalshi would be ~$822 vs $731 net in. Exits fired on
a stale needle model during the count. Lesson: a pre-committed hold-to-settlement plan beats reacting to a model mid-count.

## Edge status (wave 5, 2026-11-10/11)

### E4 weather maker holds but is underpowered
Maker-only live 39 fills 38-1 ROI +1.72% [-4.33,+6.11]; sealed +2.22% [+1.57,+2.84] inside; ~530-930 fills needed to
separate from 0. Binding limit is 6 ct/order and depth, not cash. Source: notes/e4_live_audit.md.

### Kelly step-ups: size by quarter Kelly on the CI lower bound, capped by depth and same-night correlation
User 11-10 authorized raising size on a strategy's OWN record. At 96-98c contracts full Kelly is 30-150% of bankroll and
one point of win-rate error flips the sign, so depth and correlation bind first. Step 1: E4 6 -> 12 ct/order, night cap
20%. Rule: notes/SIZING_RULE.md.

### Daily rain and ForecastEx weather are dead at executable prices
RD1 taker n140 +0.07% [-4.53,+4.28]; RD2 maker n69 < min; R1 sealed fail earlier. E5: asks on 6.9% of picks.
Sources: notes/raind_daily_rain.md, notes/e5_forecastex_audit.md.

### Video-game mega-releases do not lower game totals
14 launch clusters 2008-22 (>= $500M week one): unders days 0-6 -4.97% [-8.63,-0.95] n1,766; scoring vs control weeks
+0.020 SD [-0.032,+0.077]. Rules out league-wide slumps > ~0.6 NBA pts. Source: notes/game_release_slump.md.

### Bigger game launches do not bring bigger scoring slumps
Dose-response over 22 launches / 9,880 games: +0.042 SD per ln(launch-week revenue) [-0.046,+0.188], wrong sign. At
GTA-VI scale (assumed $3B, an extrapolation) the most negative plausible effect is ~-0.07 SD, about break-even at -110.
Hyped-only rule grh7 dev -7.90%/1,129. Source: notes/game_release_dose.md.

### A real, partly unpriced NBA effect can still fail to beat the vig
P-PAINT (teams allowing paint points): -0.537 pts/SD vs the spread, p 2.5e-5, leak-free; market prices ~30% of it. Yet
the fade bet is flat (-0.01%/2,972 at -110; +0.66%/1,160 at real close) because +0.86 pts over close only lifts covers to
52.4%. Effects under ~1-1.5 pts belong inside a model, not as standalone bets. Source: notes/nba_l30_paint.md.

### Known gamer athletes do not slump after big game launches
Difference-in-differences, 50 sourced gamers (list frozen + hashed before outcomes, evidence dated to avoid hindsight):
NBA days 0-6 +1.02 pts/game vs matched controls [-0.39,+2.38] (player bootstrap), minutes +1.1. Gamer evidence mostly
dates from 2018+, so ~5 usable launches; young improving players likely explain the positive sign. Three GTA tests
(league totals, dose, player) all null or wrong-signed. Source: notes/gamer_athletes.md.

### Kalshi NBA totals price the book opener, so opener edges are executable there
Kalshi KXNBATOTAL ladders list a median 3.4 h before our first Pinnacle sighting (95% of 114 games), already ~3c tight;
at the Pinnacle-open tick Kalshi's implied total is 0.33 pts from the open vs 1.82 from the close. L32b picks at the
Kalshi ask + fee: CLV +1.37 pts [+0.35,+2.43], ROI +1.9% [-32.6,+36.3] (n27). Kalshi taker cost ~ -110, so compare to
the -110 dev number (+4.8%), not no-vig. Source: notes/nba_l32b_exec.md.

### Kalshi opens early on the opener for CFB, CBB and NHL too -- but CLV keeps failing to become ROI
Timing gate D = |Kalshi-open| - |Kalshi-close| at first tight quote: CFB total -0.67, CBB total -1.07, NHL -0.048 (all
CIs < 0); NFL fails vs ESPN's lookahead open. K-CFB n63 CLV +1.49 [+0.86,+2.13] but ROI -2.1%: buying the strike at the
book open costs ~58c once Kalshi has moved. Third CLV-without-ROI case (L36 sealed, P-PAINT). Buy near 50c or the CLV
is spent on price. Kalshi occurrence_datetime = kickoff + 3 h. Source: notes/kalshi_openers_xleague.md.

### Kalshi sports first-listing staleness is real but small and not capturable
1,578 events / 40 series (dev 10-20..11-11): first mid moves median 2.5c by T-30 vs 6c first spread; first price less
accurate (Brier .1974 vs .1889); 0/40 series pass (no listing-time predictor below p<.001). Taker at first ask -10.0%
[-15.2,-4.8]; maker at first bid -11.0% with filled orders 9.3c [4.4,14.5] worse than unfilled -- adverse selection holds
in every sport (E5/L35/RD2 lesson). Only untested angle: a sharp price at listing (needs a forward Pinnacle logger).
Source: notes/kalshi_first_listing.md.

### Kalshi logical arbs are nearly absent at executable size
Read-only scanner (scripts/lx1_arb_scan.py), 3 valid scans 11-12: 1 distinct violation >= 2c net at >= 10 ct (Burgum
leave-date ladder, +2.1c/set, ~$1 on $49 locked to 2028); 0 in Fed, 1.4-1.6M threshold-ladder pairs, exclusive-set NO.
7-day 1-min replay: 5 leg sets ever >= 2c at best price, most < 10 ct depth. Fake arbs came from a comma parser bug (+98c
ETH), "exactly k" tagged as "below k", shared-word titles -- verify every arb leg's rules before trusting it.
Source: notes/lx1_logical_arb.md.

### Q004 lost because its totals model was biased, not because of timing
Raw model +3.85 pts above the Kalshi strike -> 47/48 picks were overs. De-biasing (L32b's trailing-100 correction) is
what separates CLV +1.11 (kept picks) from -1.44 / ROI -36% (dropped). Source: notes/nba_l32b_exec.md.

### NBA prop situational factors are priced, and a better prop model still loses to the vig
L7: 0/11 schedule/geography features (B2B, rest, travel, altitude, nightlife, time zones) move prop results vs the
de-vigged open on 141,746 dev props. L5: the P2 distribution adds real information (+0.094 / +0.112 coef, replicated)
but its under picks return -0.17% [-2.51,+2.31] over 5,569 at the open; all unders lose 4.3-6.8%. A prop edge must beat
~4.5% vig, not 50%. Source: notes/props_situational.md.

### The NBA back-to-back penalty grew, and the market grew with it
21,454 games 2007-08..2024-25: B2B margin penalty ~-1.5 (2007-12) -> ~-2.7 (2021-24), trend -0.053/season
[-0.117,+0.011]; closing-spread B2B markdown -1.0 -> -2.1 (-0.058/season [-0.073,-0.042]); B2B ATS residual trend
+0.004 [-0.060,+0.069], level -0.63 [-0.97,-0.30]. Back B2B ATS -8.06%/1,787; B2B overs -3.78%/2,113. "Recovery science"
shows up as stars resting, not better B2B play (untested explanation). Source: notes/b2b_recovery_trend.md.

### Kalshi's in-game price does not misprice "clutch" teams either way
673 games / 2,507 trailing snapshots (dev): calibration gap top-minus-bottom comeback-record tercile -1.2c [-7.1,+4.8];
trait coefficient given Kalshi mid +0.02 [-0.18,+0.21]; trailers won 19.7% when priced 19.4%. Four bet rules dead (buy
top-tercile trailer -17.6%/258). Comeback record mostly measures team quality (.07 WSH to .65 OKC), already priced.
Source: notes/clutch_market.md.

### E4 is a small edge; the only extra weather capacity is X1, and its forward record is negative so far
Census 11-12: 505 Kalshi series, no new qualifying weather series. Tape-through capacity (291 forward picks, 21 nights):
orig-7 ~$0.32/night modelled at 24 ct (real +$1.03), depth-capped; X1 $1.78/$3.26/$5.70 per night at 24/48/96 ct but real
-$0.82/night (5 losses) and 96 ct ties up ~$328 of ~$383. X1 maker failed dev on concentration (one week 36% of P&L, 7-week
window), not a bug; its sealed evidence is the user-approved 10-23 lockbox (+1.67% maker). 73% of picks sit at 99c NO
ask with little volume -- the money is in ~3.7 cheaper picks/night. Source: notes/wx_capacity.md.

### "Clutch" / comeback ability is not a team trait
NBA comebacks depend only on deficit x time x spread (0/69 trait cells, 152 tests); Kalshi in-game calibrated 7/7 bins.
Source: notes/nba_comebacks_matchups.md, notes/late_game_calibration.md.

### L34 NHL vs opener and Q008 killed by their pre-set rules
L34 n110+ CLV ~0; Q008 0 qualifying quotes/56. Source: notes/SCORECARDS.md.

### NBA 99c lock-in maker is dead: the queue forms as one block before you can join
CORRECTED 11-11: the earlier "21% of locks <= 5,000" counted already-closed markets with empty books; among open books
0/188 had queue <= 5,000 (moneyline min 109,590). The 99c bid appears ~2 min before the horn (median 126 s, lead 9, WP
.986) as one block (~80k within 40 s); first-in-queue happened in 0-3.4% of 12,259 near-locked states. Front-of-queue EV
+0.58c [-0.24,+0.86] vs a 1c maker fee. Logger unloaded 11-11. Lesson: a depth statistic must exclude closed/empty books.
Source: notes/nba_lockin_fill.md.

## Process

### Agents must write findings to disk as they go
Subagent replies die at /clear. Protocol notes/AGENT_PROTOCOL.md: findings_inbox.jsonl rows + own notes + a Learnings
section with one heading per learning; the lead merges, commits, and ingests into the vault.

### Check the do-not-redo list before briefing an agent
The daily-rain agent was briefed on a lead that had already sealed-failed (R1); caught from memory and redirected to
the only new mechanisms. Brief agents with FUND_BOOK section 3d.

### A critic changes the wording, not just the verdict
E5 "dead" became "no executable channel shown profitable" (both ROI CIs crossed 0); a p-value of 1.5e-14 at strike
level was 7.4e-6 at station-day level (12 losses = 6 weather events). Cluster at the level where outcomes are shared.
