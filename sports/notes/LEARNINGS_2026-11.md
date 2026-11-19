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

### A public index that tracks the settlement 1:1 gets priced in: Kalshi weekly mortgage rates
KX30YMORTW (Freddie Mac PMMS, Thursdays; 10 events since 2026-09-17): nowcast = last PMMS + 0.981 x change in weekly-avg
Optimal Blue OBMMI (ALFRED first-published) -> sd 1.8bp vs 9.3bp no-change, max miss 8.5bp, 0/180 PMMS revisions. Rule
(NO > 11bp from nowcast): 59/59 strikes settled far side but all already 99-100c -> 0 taker bets; maker 3 posts, 0 fills.
Market's two-sided strikes within ~+-5bp of the nowcast. ALFRED via python requests times out; curl works. Source:
notes/mortgage_rates.md.

### Kalshi Fed-decision tails are not richer than fed funds futures imply -- busy book, no E4-style staleness
22 meetings 2023-05..2025-12 (ZQ=F front month + EFFR; 44/66 cells usable): futures 0% -> Kalshi mid 1.0c (n89, 0 occurred);
futures 5-20% -> Kalshi cheaper 5.0c vs 8.5% (n14, 0 occurred); near meetings within a few cents. Taker NO rule 7 bets (< min
20), maker 12 -> dead. Flagged quotes 1c wide, 33k-1.17M contracts/24h. Source: notes/fed_vs_futures.md.

### Silent failures need an output-based health monitor, not exit codes
Second and third silent losses this week: Q013 RT logger wrote 0 picks 11-03..11-18 (RT renamed criticsScore.score ->
scorePercent), and E4 lost NY + DEN on 11-17 (open-meteo TLS handshake stalls on the first cities of a run; read
timeout=10 was the connect setting). Fixes: hourly health monitor over 47 jobs (output growth, row sanity, 0-pick streaks;
scripts/health_monitor.py), E4 fetch with 30 s handshake + in-window retries + LOST alerts (same source/params, rule
unchanged), atomic installers (overwriting a running bash script crashed nhl-ingame/nfl-fade). Sources:
notes/ops_health_monitor.md, ops_robustness_1118.md, q013_rt_scores.md.

### A single API timeout silently cost E4 two whole nights
11-13/11-14 UTC: every decision-hour tick crashed on an open-meteo ReadTimeout; cmd_pick had no per-city try/except, so
no city was logged and E4 placed 0 orders. launchd showed exit 0 because run.sh's last command (the executor) exits 0.
Fix deployed 11-15: retries, per-city isolation, failed cities retried next hour, tick exits 3 on failure. Lesson: check
for missing OUTPUT rows per night, not exit codes. Source: notes/e4_ops_diag.md.

### Kalshi cash is per exchange shard; an order sees only its own shard's cash
Weather/NFL = shard 0, crypto/commodities = 2, basketball/tennis/baseball = 3. 11-13 a ~$500 BTC 15-min trade moved
cash to shard 2, leaving shard 0 at $4.81 of $604 -> E4 rejects, NHL skips, DUL would skip everything. Executor now
sizes on shard-0 cash (balance_breakdown). Rebalance: `bet.py allocate 0=95 3=5 --confirm` (moves money; user's call).

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

### Long-dated tail NO on Kalshi earns about the cash rate, not an edge
T-90: 768 bets / 280 events, NO won 97.5% at mean 97.2c, ROI +0.15% [-1.01,+1.18] (+0.94% with 3.5% APY) -- annualized
~ Kalshi's cash APY, which idle cash also earns. T-180: -1.90% [-4.63,+0.48]. Only climate positive (+3.0%, 36 events, 0
losses = 3 hurricane seasons; post hoc, needs sealed 2025 on user yes). Kalshi APY on positions started ~Oct 2024.
Combo/parlay series KXMVE* page caches reached ~61 GB -- filter at ingest. Source: notes/lx2_tail_no_carry.md.

### Kalshi soccer vs free Pinnacle odds: the "gaps" are stale Pinnacle, not slow Kalshi
KS12 (S12 rule, Kalshi as soft book, taker): 37 bets / 30 matches -26.2% [-63.4,+12.7]; CLV vs Kalshi close -1.15 pts;
bigger gaps worse (EV >= 5%: -78%, n18). Free Pinnacle odds are cached up to ~15 min, so a "gap" is usually Pinnacle
behind. Team matcher bug: gap_logger._first() maps West Ham and West Bromwich both to "west" (fake +146% EV). F3's
draw-heavy flags came from counting repeated rows; by first tick draws are 12%. Source: notes/kalshi_soccer_sharp.md.

### Against Kalshi's own line, our opener models' edge is ~0.85 pts -- below taker break-even
K50 (buy at Kalshi's ~50c strike): cost ~0.53, CLV vs the strike bought NBA +0.88 [+0.56,+1.31] n25, CFB +0.82
[+0.40,+1.35] n71; ROI NBA -2.2%, CFB -7.3%. Taker break-even needs ~+1.40 (NBA) / +1.16 (CFB) pts; CLV predicts NBA -2.1%,
observed -2.2%. Earlier CLV (+1.37/+1.49) partly measured Kalshi's own move off the book opener, already in its price.
Ladder-optimal strike picking fails by design (model-as-truth chases cheap far strikes). Source: notes/k50_strike.md.

### At our volume, statistics cannot certify small edges inside the mandate -- sizing rests on judgment within risk limits
Bar review + calibration v1/v2 + critic (notes/bar_review.md, bar_calibration*.md): current edgelab rule is mis-specified
for near-certain contracts (false-passes 96-98c placebos ~5x its bar at 0 losses; Sharpe/30%-share checks block real
small edges). A calibrated forward SPRT (H1 loss ratio .70, alpha .005 -- month-level loss shocks inflate nominal .05 to
.08) accepts an E1-sized edge in a median ~21-25 months and kills it 10-17%; a +2% sports edge cannot be accepted within
12 months; only ~+5% at 200 bets/month does (median 14 mo). Forward-after-sealed-pass gives ~.011 false accepts/yr.
Amendment text (user's call) in bar_calibration_v2.md.

### Net CLV cannot be certified as a primary statistic at our scale; keep ROI primary
Kalshi close calibration vs outcomes: 95% half-width 3.3-8.5c per league (1.34c pooled) vs the 0.25c needed; ~50-70k games
per league to certify. Kalshi close vs DK de-vigged close at the book line: pooled +0.01c [-0.07,+0.08] n711 (NBA +0.25,
NFL -0.46, CBB -0.17 differ). ROI on net CLV for our rows: slope 1.25 [0.28,2.31] n398 -- unresolved. Bias under our
selection undetectable. CLV stays a diagnostic. Source: notes/clv_calibration.md.

### Kalshi Rotten Tomatoes markets reprice within minutes of the review embargo -- traders read the outlets too
KXRT (Tomatometer 'above X', settles Monday after wide release 10:00 ET): volume jumped from 10 to 10,107 contracts within
4 min of the embargo hour. Outlet-review nowcast 30 min after embargo: Brier .236 vs Kalshi .136, gap +0.1008 [+0.061,
+0.142] n46 films -> killed. First-wave reviews run +5.5 Metacritic points above the final Metascore. RT per-review lists
return 401 anonymously (recorded, not bypassed). Q013 logger silently broke 11-03 (criticsScore.score -> scorePercent):
a silent-zero-picks health check is needed on every logger. Source: notes/rt_review_lag.md.

### 'Skilled' Polymarket wallets don't stay skilled, and Kalshi reprices before you could copy them
Sports game winners (12,176 matched games): 53/2,681 wallets 'skilled' on 2025 (~17 by chance); their own 2026 PM fills +0.75%
[-9.2,+11.1], median wallet -6%. Copying on Kalshi (2,276 bets): their trades were contrarian dip buys; Kalshi moved 1.69c
against before and bounced 0.64c by our fill (kill K2: repriced < 60 s); ROI -2.36% [-5.19,+0.32], CLV -5.10c. Forum
survivorship again. PM data-api v2 cursor pages need the original params re-sent. Source: notes/pm_wallet_signal.md.

### Season-settled contracts can never pass the dev gate: profit lands in one week by construction
LX2 climate T-90 tail NO (lx2c_a_t90_climate_tail_no): dev 70 bets / 36 events, 0 losses, +3.01% [+2.61,+3.42] but dead on
concentration (one week 48% -- all hurricane markets settle ~Nov 30), no close prices for CLV, and Sharpe 6.3 too-good guard;
exact loss CI [0, 5.13%] vs 2.92% break-even (n.s.). Sealed shot refused by edgelab (needs dev_pass) -- correctly not
bypassed. Route: forward-only over coming seasons or a gate change for season-settled contracts (user). Source:
notes/lx2_climate_sealed.md.

### Monthly biggest-quake ladder: model matches the bid; watch whether the maker's quotes decay with time
KXBIGGESTQUAKEMONTH (started 11-10; settles on USGS magnitude shown Dec 1 10:00 ET, Mww preferred): empirical-window model
(USGS since 2000, current-month max pinned) within 0-5c of the YES bid on all 9 rungs; asks 4-7c rich -> taker 0 bets now.
If quotes don't decay as the month runs down without an M6.6+, the taker rule fires ~11-24 -- the E4-style staleness to watch.
One event/month -> >= 12 months to test. Source: notes/quake_monthly.md.

### Kalshi city home values settle on Zillow's first-published page value, which leads the public CSVs and re-levels history
18 *HOME series (98 mkts, none settled; Nov events close 12-17 15:00Z), active (~42.7k ct on 11-16), strikes within +-1 sd.
Each Zillow release shifted a city's history by a constant % (one vintage pair: p95 2.88% vs ~0.15% monthly moves); real-time
nowcast p95 error 7.8 sd units vs 2.1 on revised data. The page showed October while research CSVs ended at September, so
apparent edges vs the CSV vanish. A script call to the page API got the IP 403-blocked -- don't scrape it. Open question:
does the page show the settling value before trading closes (check by browser ~12-15). Source: notes/home_values.md.

### Check which climate baseline the market uses: Kalshi's snow-days maker prices the recent low-snow decade
KXSNOWDAYS (scalar, Dec 1-Apr 1, >= 1.0 in for snowy cities / 0.1 in elsewhere): vs 30-winter climatology our values sit
+.085 above mid (5 YES 'signals'), vs the last 10 winters only +.006 -> the signals are one correlated bet on a snowier
winter, not mispricing. Thanksgiving snow (trace = YES): ECMWF-ENS 50/50 with trace climatology inside the 3-20c spreads.
Forward-only trials TG1/SD1/SD2 registered. Source: notes/snow_markets.md.

### Kalshi NYC hourly temperature settles on a proprietary 8-station index the market sees and we can only approximate
KXTEMPNYCHS (= HT1's series, not new): index = per-minute average of 8 ASOS airports with weekly station offsets; our rebuild
misses it by ~0.5F sd, as much as 26 min of temperature change. Dev 488 events: taker n80 +5.1% [-7.8,+18.8] (picks lost 13.8%
vs 1.7% claimed); market Brier .0145 vs model .0259; maker filled lost 11/64, unfilled 0/33. Don't trade a settlement index
you can't observe at the market's resolution. Series watcher flagged it 'new' from first-seen opens -- fixed to use history.
Source: notes/nyc_hourly_dir.md.

### Crypto monthly max/min (barrier) ladders: models don't beat Kalshi's mid; MAX YES looked rich (post hoc)
Per-minute 20/20-trimmed BRTI monitoring (early closes median 0.66 h after first Coinbase crossing). 3,105 market-days /
21 events / 12 months: Brier M1 (reflection, t4, DVOL) .0605, Kalshi mid .0634, M2 bootstrap .0653 -- n.s. (13/21 events).
MAX YES mid .093 vs realised .045 (rich); MIN mid .124 vs .123 (right) while both models underpriced MIN (flat vol misses
downside skew). CB1 forward trial live (verdict needs ~12 months). Untested idea: skew-aware smile model; NO on MAX rungs
(post hoc, would need its own forward prereg). Source: notes/crypto_barrier.md.

### A calibrated NBM probability forecast adds ~1/3 more E4 picks -- forward paper, not size (critic: weakened)
Rule C (NO when calibrated P(hit) <= 2%, NEW brackets only): later dev 544 taker picks, 5 losses (0.92%, CI .30-2.13% vs
b/e 2.50%); maker 331 fills +11.58u [+6.0,+16.5], 0/213 unfilled lost. Critic: no leak (NBS 19Z D-1, error 1.85F vs 4.97F
when shifted a day), fits strictly prior, no sealed overlap; but taker with per-order cent fee rounding at 1 ct -> +4.51u
[-0.79,+8.99] n.s.; edgelab family bar fails (E1 p .0014); 3/5 losses where market YES >= 5c (model-vs-market disagreement
again). Where the model disagrees most with the price (rule PA), the market is right. Source: notes/e4_probabilistic.md.

### OT1's options-implied rule does not extend to other crypto ladders; Kalshi tails are only 1-2c rich
Range brackets (KXBTC/KXETH) n206 +0.07% [-4.51,+4.22]; Friday events 1-4 days ahead n73 +0.43% (< min n; since 2025-07
-1.7%); SOL 0 bets (Deribit SOL smile on 71/465 days). Calibration n867: implied .073 / Kalshi mid .087 / realised .073 --
the richness is eaten by spread + fee. Monthly max/min ladders are one-touch barriers (different model) -> forward-only
trial being built. Source: notes/ot1_extension.md.

### Other E4 decision times add no independent capacity; 06:00 add-size on filled brackets is a forward-only lead
Daily-high markets list at 10:00 ET on D-1 (D-2 impossible). D-1 10:00 (4F margin fixed from error climatology): pooled
taker +11.1u p .047 / maker +14.8u p .0125 but brackets not also picked at 22:00 lose (-1.7u/206); orig-7 alone +0.94% p .31
-- the pooled number is X1-driven. D 06:00: 90% same brackets; extra size where the 22:00 order already filled +11.2u
[-0.5,+22.4] p .031, ~23 fills/month (no leak-safe overnight runs before 2026-04). Unfilled orders never lost at any
time (0/1,160). Source: notes/e4_timing.md.

### NHC's intensity forecast is a central estimate, not a ceiling -- Kalshi prices rapid-intensification risk
KXHURCAT (storm reaches category N; settlement = NHC advisory peak wind, 236/236 matched): NO when NHC 5-day peak forecast
is >= 20 kt below threshold -> 106 bets / 36 storms, 18 losses (17%, exact CI 10.4-25.5%) vs 13.1% break-even, ROI -4.48%
[-13.8,+3.3]. Losses = rapid intensification beating NHC by 23-48 kt (Beryl, Milton, Idalia...), which Kalshi priced at
10-50c correctly. Post-advisory NO: 15 bets 0 losses but ~5/season. Capacity a few hundred $/season. Unlike E4's model
consensus, a single official forecast's point estimate understates tails. Source: notes/tropical_nhc.md.

### New Kalshi data/energy series arrive with a market maker already holding the certain side
State AAA gas (KXAAAGASCD*, daily change in 1c buckets): ~700-900 ct quoted per level, far buckets NO at .99-1.00.
ERCOT hub monthly averages (KXERCOT*M): late-month pinned-average rule 0 losses / 5,562 flagged rungs 2012-23, but the
certain side is bid 94-95c and nobody sells (~0 fills/month). Econ-event ladders (16 KX*EVENT families, Pyth settlement):
1/1,349 tradable quotes, single maker at 2/99. E4's stale tails were in an OLD neglected book; new series are seeded by
makers. Re-check new series after makers thin out. Sources: notes/state_gas.md, energy_monthly.md, econ_event_ladders.md.

### Kalshi politics favourites at 60-85c are priced; edgelab mis-scores scalar settlements
L22 (Le arXiv 2602.19520): T7 favourites 2026-07..11-15 -0.56% [-6.28,+5.07] n337/154 races (won 75.5% at 74.6c ask);
T14 -9.26%; paper period -9.09%. Mid-price calibration slope 1.23 [1.08,1.42] (weak) vs paper 1.83 (trade-weighted);
spreads 6-13c below 80c put the ask 3-6c over mid, ~ the bias. BUG: edgelab result_equals_side scores scalar
settlements (e.g. 0.25) as losses -- 109 politics markets; fix proposed (edgelab locked). Source: notes/pol_favourites.md.

### Crypto daily tails vs Deribit options: a lead (critic pending); maker and stale-input index versions fail
OT1 (11:00 ET, Deribit-implied tail <= 10%, Kalshi >= 3c richer after fee, take the other side at ask+fee): dev n387 / 195
crypto-days +4.60% [+1.07,+7.62] p .0058 (bar 1.04e-3), both halves +, CLV +2.9c, monotone in gap (-6.9 .. +11.5%),
options Brier .04568 < Kalshi .04617; 2025 Q3-Q4 negative; spreads narrowed 3-5c -> 1c. Maker variants adversely
selected (filled won 88.4% vs unfilled 99.8%). Index arm failed: model input (10:30) older than the quote (11:00) --
inputs must be at least as fresh as the market. Deribit history via history.deribit.com. Source: notes/options_implied_tails.md.
CRITIC 11-15 (critic_ot1): no bug (timestamps/DST/fee/settlement/clustering clean; drop top-10 days still +2.6%) but
WEAKENED: pre-2025-07 +7.3% [+3.7,+10.4] n197, since +1.7% [-4.6,+6.7] n190 (n.s.); one-bet-per-day +3.0% [-1.8,+7.1];
rests on 24 losses; CLV measured 1 h before settlement mostly reflects the outcome. Forward paper only.

### Kalshi rules differ from books in golf ties and cricket no-results, but prices follow Kalshi's rules
4,750 open series classified: sports totals/spreads match books (OT counted; NHL shootout = 1 goal). Golf top-N pays every
tied player $1 (top-10 pays 1.147 [1.105,1.191] x N), yet the field's prices sum to 11.08 vs 11.45 paid (n.s.); buy YES at
ask -18.0%/2,376. Cricket no-result = $0.50 both sides (T20 11.9% of events); favourites calibrated incl. 50c; underdog
-14.7%/216. Check "sum of prices vs actual payout" before building a rules-gap trade. Source: notes/rules_quirks.md.

### A loss-free certain-side strategy at 96-98c cannot pass the bar on short history: null p = price^n
L39 early-month version (P1m): 94 maker fills, 0 losses, +4.28% [+3.61,+5.00], market-null p .021 > bar .00455 -> dead;
.96^94 = .0216. Passing needs ~132 loss-free fills at 96c, ~213 at 97.5c. With zero losses a bootstrap CI only reflects
entry-price spread; the evidence is the run length. Judge E4-like ideas by expected fills-to-bar before testing. Also:
L39 is a forecast-margin rule (2x wettest model + 0.5 in), not "unreachable"; NOAA matched Kalshi 1,259/1,259; the one
loss was real rain. Source: notes/l39_expansion.md.

### Billboard #1 on Kalshi is priced by Wednesday, like album units
HDD Tuesday #1 = Billboard 200 #1 in 0.941 of 68 dev weeks; rule picked 802 Wednesday rows with 0 losses, but the leader's
YES median bid was .98 with no seller and rivals had no YES bid on 702/754 rows. Taker n7, maker 22 fills (cap ~$1.7k
over 20 months even at full capture). Title matching nearly bought NO on the real #1 ("MAN'S" vs "Mans", "I AM MUSIC") --
verify string joins before pricing. Source: notes/billboard_top.md.

### Kalshi NFL totals do not react to weather-forecast updates, so there is no lag to trade
261 outdoor games, 5,326 NBM/GFS-MOS run-to-run updates, 32 qualifying events (28 rain-chance, 4 wind): buy the 50c strike
5 min after the update -> net CLV -1.99c [-3.17,-0.86] n31 (fee + spread); implied total moved ~0 at +15/+60/+180 min and
+0.05 [-0.27,+0.36] by kickoff. A timing trade needs information the market actually prices. Books already move totals
-0.22 pts per mph forecast wind open->close. Source: notes/nfl_wx_updates.md.

### A great public forecast is no edge when the book is made from the same forecast
Census 11-15: only KXALBUMEQUIV (album first-week units, 95 settled) has a public timestamped forecast (HitsDailyDouble
midweek). Early rule picked 333 dev brackets, 0 losses -- but our side's ask was > .99 on 300/319, so 7 bets (< min 30).
Box office not listed on Kalshi; app rank / streams have no forecast; air quality / Lake Mead / snow too few events.
E4 works because weather tails are stale (no one re-prices them); album brackets are priced by forecast readers.
Source: notes/consensus_reject_census.md.

### NHL opener model on Kalshi: real CLV, below taker cost; goalie news is priced 6-12 h before puck drop
nhl_om_a (k .40 goals, Kalshi 50c rung at first tight ~32 h out, 5.6 h before Pinnacle): n38 ROI +5.8% [-25.6,+37.2];
CLV +1.67c [+0.75,+2.67] vs 2.5c over-mid cost -> ~-0.9c/contract expected. Placebo moves 0.04 of gap vs model 0.34.
Goalie trial at tip-60 min (actual starter): CLV +0.10c n36 -- Kalshi prices the confirmed starter 12-6 h before tip, so
any goalie edge needs confirmation timestamps (logger being built). Source: notes/nhl_opener_model.md.

### No free real-time sharp feed from California; cheapest reseller ~$99/mo
Pinnacle API closed to the public 2025-07-23 and refuses US customers; Betfair bars US residents; The Odds API's Pinnacle
odds come from the public website with delay (would repeat the stale-feed failure). Cheapest real-time: pinnapi reseller
$99/mo REST (10 req/s) or $198 with WebSocket, free tier 100 req/day to verify freshness; SharpAPI $399+$149 as a cross-
check. Novig/ProphetX are free in CA but likely follow Kalshi. Latencies are vendor claims. Source: notes/sharp_feed_research.md.

### NFL inactive lists are priced before they are published
2025 season: buying 3 min after the inactive list (QB Q/D ruled out or played) net CLV -1.52c [-2.13,-0.85] n18; post-list
drift +0.47c [-0.17,+1.17] < ~2.75c entry cost; no jump at the list across 279 games (+0.034c vs placebo). 57% of the
24-h move came before the list (insider reports). Same as Q011 NBA injury PDFs. Official releases are not information
edges on Kalshi. Forward logger scripts/news_lag_logger.py ran 11-15. Source: notes/news_lag.md.

### Maker execution does not rescue the opener models: fills pick the losers even with a model choosing the side
180 dev bets (K50 NBA/CFB/NFL + CBB), trade-tape fills at mid-1c: 56% fill; filled CLV -0.90 vs unfilled +2.21, gap -3.10
[-3.82,-2.58] in every league; CLV-implied filled ROI -3.6% vs taker -4.0%. Maker fee is only 0.44c at 50c, but the
fills arrive when the line is moving against us. Fifth case (E5, RD2, first-listing, L35, here). At ~100 bets win rate
cannot show adverse selection; CLV can. Source: notes/maker_fairfill.md.

### Opener models beat the line but not the cost -- 4 sports now; Kalshi taker costs more than -110
CBB totals model (walk-forward ridge, de-biased): dev 2023-25 n2,213 CLV +0.83 pts [+0.69,+0.97], no-vig +3.2% p .054
(fail), at DK open price -1.4%. Kalshi 50c strike costs 0.541 after fee vs 0.524 at -110 -> ~-3.5% expected as taker.
Same as NBA L32b/K50, CFB, P-PAINT: our models carry ~0.8-0.9 pts of real information; taker break-even needs ~1.2-1.4.
The binding constraint is execution cost, not prediction. Historical ESPN core API has DK NCAAB open/close since 2023-24.
Source: notes/cbb_opener_model.md.

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

### Paid-panel settlement sources are the opposite of E4 (altdata_kpi, 11-19)
Kalshi's Carbon Arc series settle on a paywalled daily panel: subscribers ARE the informed side. Persistence nowcast CA1
dev +2.05% [-10.5,+13.2] n74 (one calendar month). Rule: before modelling a series, check who sees the settlement data first.
