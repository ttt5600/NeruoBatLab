# esoccer_tennis -- eSoccer/eBasketball calibration + tennis pre-match vs Pinnacle (agent esoccer_tennis, started 2026-11-21T06:56Z)

Brief (lead, from pool_census write-ups A + C): (A) KXESOCCERGAME / KXEBASKETBALLGAME (drafted.gg Valhalla/Valkyrie cups):
descriptive calibration of pre-start mid/ask by price bucket + blind cheap-side seller P&L at the ask incl. fee, pre-start and
in-play separately, on a DISCOVERY window; only if a robust miscalibration shows, pre-register ONE rule on the later half with a
base-rate-shape placebo (sealed = latest ~30%, unopened). (B) tennis pre-match: Kalshi vs Pinnacle no-vig gap distribution and
whether buying the side Kalshi prices cheaper by >= X after fee makes money (forward-only if Pinnacle history is not available).
Rules: AGENT_PROTOCOL (no git, no orders, no launchd). Scratch: results/esoccer_tennis/.

Prior (from FUND_BOOK 3d + STRATEGY_v2): BIAS/FLB rows 0/21 passed; Kalshi sports longshots overpriced but no rule past the bar
(G2/G3/L1/L2); Kalshi soccer 1X2 made off Pinnacle (S1-S3); live Kalshi-vs-Pinnacle scanners 0 flags; CS2 public-stats model
lost to the book. Expect: (A) calibrated or a longshot tilt that does not clear spread + fee; (B) Kalshi within 1-2c of Pinnacle.

## Data-shape facts (probe.py / probe_tennis.py / probe_pin.py, 2026-11-21 ~07:00Z; no outcomes analysed)
- KXESOCCERGAME: 3 markets per event (team A, team B, TIE), mutually exclusive; eFootball 2x4-min halves; "outcome after 90
  minutes plus stoppage"; cancelled/postponed > 2 weeks -> all $0.33. KXEBASKETBALLGAME: 2 markets; tie/draw or cancel -> both
  $0.50. Fee: quadratic taker (0.07 p(1-p)), fee_multiplier 1. Source DRAFTED.GG.
- Timing: event ticker HHMM = scheduled start in ET (rules text "originally scheduled for Nov 21, 2026 at 4:13 AM EST" for ticker
  ...0413); occurrence_datetime = start + 3 h (as in derived_ladders); markets open ~2.3 h before start (04:22Z for a 06:40Z
  start) with placeholder quotes (0.01/0.98, 0.15/0.76 at listing, 0 volume); close_time = when the result is declared (~10 min
  after start: 06:50:20Z for 06:40Z); settled ~4 min later.
- Tennis: no start time in the Kalshi ticker (date only, 26NOV21); occurrence_datetime is NOT a reliable start (ITF PLUMAL
  closed 06:20Z with occurrence 10:30Z; ATPMATCH ZVESHE occurrence 13:30Z, closed 21:19Z). Markets open ~13-15 h before close,
  trade in-play. Fee types: Challenger/ITF/WTA-Challenger quadratic; KXATPMATCH / KXWTAMATCH quadratic_with_maker_fees.
- Pinnacle guest API tennis = sport 33: 159 matchups (96 parents) on a Saturday, covers ITF Men/Women, ATP Challenger,
  WTA 125k, ATP Finals, BJK Cup; league name carries the round ("ITF Women Heraklion - SF"). Cache max-age 905 s
  (Age 559 s / 223 s at read). Period-0 moneylines 86; maxRiskStake e.g. 2,400.
- Kalshi live REST cutoff /historical/cutoff = 2026-09-21 (everything here is in the live API: use `*_dollars` fields).

## PRE-REGISTRATION Part A (written 2026-11-21 ~07:05Z, BEFORE any outcome is computed)
Universe: settled markets of KXESOCCERGAME and KXEBASKETBALLGAME, with S = scheduled start parsed from the event ticker (ET).
Windows by S (UTC), chronological:
- DISC  [2026-11-07 00:00Z, 2026-11-12 00:00Z)  5 days  (~36%)  -- the only window used for the descriptive pass
- TEST  [2026-11-12 00:00Z, 2026-11-17 00:00Z)  5 days  (~36%)  -- NOT pulled unless a rule is pre-registered after DISC
- SEALED [2026-11-17 00:00Z, 2026-11-21 00:00Z) 4 days  (~29%)  -- not pulled; opened only on the user's explicit yes
Quotes: batch GET /markets/candlesticks, period 1 min, from S - 30 min to close_time + 1 min (kx_throttle). Quote at time T =
yes_bid / yes_ask close of the latest candle with end_period_ts <= T (stale guard: that candle must end within 10 min of T,
else "no quote"). Two-sided = bid > 0 and ask < 1. mid = (bid + ask) / 2.
Outcome v = 1 (yes) / 0 (no) / settlement_value (scalar: 0.33 eSoccer cancel, 0.50 eBasketball tie/cancel).
Fee per contract = ceil(100 * 0.07 * 10 * p * (1 - p)) / 1000 (10-lot taker, cent-rounded; same as ESP1/CS2F).
Taker P&L per contract: YES at ask a -> v - a - fee(a); NO at 1 - b -> (1 - v) - (1 - b) - fee(1 - b). ROI = sum pnl / sum cost.
CIs: 95% bootstrap over EVENTS (cluster; 3 or 2 markets of an event move together), 5,000 resamples; day-level split shown.

Descriptive pass on DISC (decision time PRE = S - 5 min; secondary S - 1 min and S - 15 min shown, not used for selection):
1. Calibration table: mid bucket -> n markets, n events, mean mid, outcome rate, gap = rate - mid with event-clustered CI; for
   all two-sided quotes and for spread <= 0.05. Split by series and (eSoccer) team vs TIE market.
2. Taker cells (the K cells used for discovery): series (2) x market type (eSoccer team, eSoccer TIE, eBasketball team = 3)
   x side (YES at ask, NO at 1 - bid) x price-paid bucket edges [0.01, .05, .10, .15, .25, .40, .60, .75, .85, .90, .95, .99]
   (11 buckets) -> K <= 3 x 2 x 11 = 66 cells at PRE.
3. Blind cheap-side seller (the brief's headline), one trade per market, 1 contract:
   - PRE: every market with yes_ask <= 0.15 and yes_bid >= 0.01 at S - 5 min -> buy NO at 1 - yes_bid.
   - INPLAY: first 1-min candle with end in (S, close_time] at whose close yes_ask <= 0.15 and yes_bid >= 0.01 -> buy NO at
     1 - yes_bid of that close (latency-optimistic); also the 1-min-delayed version (next candle's close quote) to show how
     much of the result is latency. Markets already cheap at S - 5 are included (first in-play touch).
   - Also the pool_census trade-price version (TAIL - MIRROR from candle volume x mean price, S - 30 min to close) for
     comparability with pool_census (+$1.06M / +$0.50M per 30 d there).
   -> 2 series x 3 variants = 6 more cells; K_total = 72 (Bonferroni alpha .05 / 72 = 6.9e-4 for DISC flags).
"Robust miscalibration" in DISC (ALL required): (i) >= 200 events in the cell; (ii) ROI event-clustered CI at the Bonferroni level
(two-sided 1 - 6.9e-4) excludes 0 on the profitable side, after fee; (iii) ROI > 0 on >= 4 of the 5 DISC days (level-drift guard,
notes/derived_ladders.md); (iv) not a placeholder artefact: >= 50% of the cell's quotes have spread <= 0.05.
If NO cell qualifies -> Part A verdict "no taker-robust miscalibration pre-start / in-play"; TEST is not pulled; stop.
If >= 1 qualifies -> pick ONE (highest Bonferroni lower bound), write rule + base-rate-shape placebo + success/kill here and an
edgelab spec (family kalshi_sports_pregame for PRE, kalshi_ingame for INPLAY) BEFORE pulling TEST. Placebo principle (fixed
now): same side and same market type, applied to the adjacent price band(s) outside the rule band in the same events/days, so a
level shift (e.g. more ties this week, favourites winning more) earns the placebo the same money; the rule must beat it
(rule ROI - placebo ROI, event-clustered CI > 0). TEST success: ROI CI > 0, p < family bar, both TEST halves > 0, beats placebo.
Kill: ROI point <= 0, or placebo difference CI includes 0.

## PRE-REGISTRATION Part B (written 2026-11-21 ~07:05Z, before any Kalshi-vs-Pinnacle number is computed)
Question: at a fixed pre-match time, what is the distribution of Kalshi price minus Pinnacle no-vig on tennis match winners, and
does buying the Kalshi YES of a player when conservative Pinnacle fair - (ask + fee) >= X make money?
History check: Pinnacle tennis history is not stored locally (gap_logger = US leagues + 9 soccer leagues; esports_pin = sport 12
only); tennis-data.co.uk is declined (403, FUND_BOOK 3d) and covers main tour only, not Challenger/ITF. -> FORWARD ONLY.
Trial TB1 (forward; edgelab family cross_venue). Logger scripts/tennis_pin_logger.py (read-only, no launchd; run by hand):
- Series <-> Pinnacle league prefix: KXATPCHALLENGERMATCH = "ATP Challenger"; KXITFMATCH = "ITF Men"; KXITFWMATCH = "ITF Women";
  KXWTACHALLENGERMATCH = "WTA 125k"; KXATPMATCH = other "ATP ..."; KXWTAMATCH = other "WTA ...". Singles only (no "/" names).
- Match: one-to-one player match (normalized accent-free token sets: equal, or the smaller set has >= 2 tokens and is inside the
  larger; never substrings, never a first-name-only match) on BOTH players, AND each Kalshi title surname ("Hassan vs Watanuki")
  inside its matched Pinnacle name, AND Pinnacle start within [Kalshi ticker date 00:00 ET - 12 h, + 36 h], AND exactly one
  Kalshi event qualifies and is claimed by one Pinnacle matchup only.
- Pinnacle fair = conservative de-vig of the side bought = min(multiplicative, power); quote age_s = Date - last-modified;
  fresh = age_s <= 120 s.
- Decision window: Pinnacle matchup not live, tick time in [startTime - 90 min, startTime - 10 min]. Signal: fresh, Kalshi YES
  ask in [0.03, 0.97] with ask size >= 1, fair - (ask + fee) >= 0.03 (fee as Part A). FIRST qualifying tick per match; either
  player may fire. Outcome = Kalshi settlement (scalar fair-price / $0.50 at value).
- Descriptive (not trials): gap = Kalshi mid - Pinnacle multiplicative no-vig at the first fresh tick in the window, by series;
  thresholds X = 1/2/5c; stale-quote (age > 120 s) version as the cache-artefact control; Pinnacle close CLV.
- SUCCESS (-> propose a small forward money trial to the user): n >= 300 bets AND ROI CI lower > 0 AND both halves > 0 AND mean
  Pinnacle-close CLV > 0 AND p < cross_venue bar. KILL: ROI point <= 0 at n 300; or over the first 7 logger days median
  |gap| <= 1c and < 1 signal/day (priced off the sharp line -> park).

## Log
### 2026-11-21 07:00-07:15Z setup (no outcome computed yet)
- DISC fetch (results/esoccer_tennis/fetch_a.py DISC, fetch_DISC.log): 7,719 markets / 2,718 events (eSoccer 2,283 events,
  eBasketball 435), 98 batch candle calls, 199,935 1-min candles, 7,698 markets with candles.
- Timing check (markets parquet): close - S median 12.3 min eSoccer [p5 10.1, p95 14.8], 25.5 min eBasketball [22.6, 30.0];
  S - open median 159 / 138 min. Ticker time = real start (eSoccer matches ~12 min wall-clock). 21 eSoccer events (63 mkts)
  closed BEFORE S = cancellations settled scalar 0.33.
- AMENDMENT A1 (before outcomes; mechanics only): Kalshi 1-min candles are emitted only when something changes (0.49 of
  minutes present; 66% of candles 0 volume = quote-change candles; one market's full-life pull = 48 candles over 169 min).
  So the quote at T = carry-forward of the latest candle <= T with NO 10-min stale guard (the guard would drop unchanged
  quotes). Markets with no candle <= T in the pulled window are "no quote" (2.0% of eSoccer at T-5, 0% eBasketball).
  A market enters a decision only if close_ts > T (cancelled-before-T events drop out; cancels after T count at 0.33/0.50).
- Part B logger written (scripts/tennis_pin_logger.py) and started by hand: `TENNIS_PIN_OUT=tennis_pin nohup ... loop 600 120`
  from results/esoccer_tennis/ (bounded 10 h, read-only, no launchd; log tennis_pin_loop.log). First tick: Pinnacle 63 tennis
  singles parents with a moneyline, Kalshi 79 open events, 52 matched (all exact token sets); unmatched audit: Pinnacle
  "Alejandro Garcia" vs Kalshi "Garcia Carbajal" rejected by the surname check (conservative, as intended); BJK Cup was
  unmapped -> mapped to KXWTAMATCH (Davis Cup -> KXATPMATCH) before the loop; 61 matched after.
