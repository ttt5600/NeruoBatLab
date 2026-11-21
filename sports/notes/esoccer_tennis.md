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

## RESULT Part A -- DISC descriptive pass (2026-11-21 ~07:09Z; results/esoccer_tennis/analyze_a.py DISC -> analyze_DISC.log,
## a_DISC_calib.csv, a_DISC_cells.csv, a_DISC_blind.json, a_DISC_frame.parquet)
- Numbers below are from the FINAL run (analyze_DISC.log / a_DISC_*.csv/json; bootstrap draws shift by <= 0.5 pt vs the
  first run because the bug fix changed the RNG order). Bug caught before reading the tight table: `~tight` on a Python bool is -1/-2 (truthy), so the first run's "spread <= .05"
  calibration rows equalled the all-quote rows; fixed to `(spread <= .05) if tight else True` and rerun.
- Book at T-5 (S - 5 min): 7,659 markets quoted, 98.5% two-sided, median spread 3c, 72% with spread <= 5c (eSoccer team 74%,
  TIE 73%, eBasketball 59%). Volume: 99.8% of market volume trades inside S - 30 min .. close; IN-PLAY = 83% of eSoccer and 90%
  of eBasketball contracts (5.70M of 6.82M; 2.55M of 2.82M) -> these are in-play products.
- Calibration (mid, spread <= 5c, event-clustered 95% CI): no bucket with n >= 100 misses by more than ~3c. eSoccer team
  [.25,.40) gap -0.001 [-0.031,+0.027] n1,014; [.40,.60) -0.014 [-0.038,+0.009] n1,403; TIE [.15,.25) +0.028 [+0.004,+0.051]
  n1,073 (ties slightly cheap at mid) but TIE [.10,.15) -0.016 [-0.044,+0.013] n495; small longshot cells noisy (team [.05,.10)
  +0.103 [+0.013,+0.211] n56). eBasketball [.40,.60) -0.004 [-0.025,+0.018] n312.
- Taker cells (48 non-empty of K = 66, a_DISC_cells.csv): ROBUST cells = 0. Every cell with >= 200 events has ROI < 0 after
  fee: e.g. eSoccer team YES [.40,.60) -13.1% [-16.7,-9.4] n1,889; TIE YES [.15,.25) -8.3% [-17.9,+1.2] n1,658; TIE NO
  [.85,.90) -3.5% [-6.0,-1.1] n1,075; eBasketball YES [.40,.60) -6.7% [-10.2,-3.1]. The TIE mid gap (+2.8c) does not survive
  the 3c spread + fee (YES at ask -8.3%).
- Blind cheap-side seller, TAKER (buy NO at 1 - yes_bid when yes_ask <= .15 and yes_bid >= .01; 1 ct per market):
  eSoccer PRE -4.19% [-6.80,-1.66] n651/472 ev, 0/5 days > 0 (cheap side won 11.5% vs mean bid .082 / ask .124);
  eSoccer IN-PLAY first touch -2.78% [-3.66,-1.87] n4,184/2,169 ev, 0/5 days; +60 s delayed -3.06% [-4.09,-2.08];
  eBasketball PRE -3.4% [-22.3,+9.4] n17 (too few); eBasketball IN-PLAY +2.25% [-0.31,+4.65] n402/390 ev, 4/5 days, Bonferroni
  CI [-2.1,+6.6] -> not distinguishable (fails rule ii); +60 s delayed +1.67% [-1.10,+4.27].
- Blind seller at TRADE prices (pool_census TAIL - MIRROR form, S-30 min .. close): eSoccer TAIL $155,321 - MIRROR $128,130 =
  +$27,190 per 5 d = +$163k/30 d (pre-start -$13.6k/30 d, in-play +$176.7k/30 d); eBasketball +$11,801 per 5 d = +$70.8k/30 d.
  pool_census's +$1.06M / +$0.50M (18 sampled events x212) overstated the net ~6.5x / ~7x: TAIL roughly agrees ($0.93M/30 d here
  vs $1.07M) but MIRROR (comebacks) was nearly missing from the 18-event sample.
- Reading: the trade-price "longshot buyers overpay" money is the half-spread. Longshot YES buyers lift the ask (~.12) and the
  cheap side wins ~10-12%: the PASSIVE seller (the provider-style MM quoting both sides) earns it; a taker who sells at the bid
  (~.08) loses 3-4%. Same mechanism as amendment 6 (gas_daily_v2) and the TSA/E5 maker lessons.
- VERDICT Part A (per pre-registration): no taker-robust miscalibration in DISC, pre-start or in-play -> no rule registered, TEST
  not pulled, SEALED untouched. Not an edgelab trial (descriptive only, no rule).
- Untested follow-up (maker class, low prior: MAKER 0/7, fills = losers): a resting seller of the cheap side at the ask in-play.
  Needs the trade tape (taker_side) to measure filled-vs-unfilled and queue position vs the provider; not run here.
- Row check (4 random eSoccer in-play first touches, raw candles): paths are coherent (e.g. Hoffenheim (Krocs) .18/.21 at S+2
  -> .07/.09 at S+3 -> .01/.02, result no); in-play quotes blink wide (0.01/0.74) when the MM pulls around goals, so candle-close
  executions are approximate. First touch median S+4 min [p10 1, p90 8]; 61% of eSoccer markets hit ask <= .15 in-play.

### Part A add-on: who earns the cheap-side money? exact trade tape (DISC only; descriptive, outside the K cells)
- results/esoccer_tennis/tape_maker.py: random 500 eSoccer + 200 eBasketball DISC markets, every trade (GET /markets/trades,
  taker_side) -> tape_DISC.jsonl (700 markets), tape_DISC_summary.json. TRAP CAUGHT: pandas parsed created_time to
  datetime64[us, UTC]; `astype(int64) // 10**9` gave 1,794,152 instead of 1,794,152,693 (1000x off, the datetime64[us] trap).
  The script's range assert fired; fixed to `(ts - epoch) // pd.Timedelta(seconds=1)`.
- In-play, trades with the cheap side c = min(yes, no) <= .15 (contract-weighted, event-clustered 95% CI):
  eBasketball, taker BOUGHT cheap: 120,143 ct, mean c .042, cheap side won 1.27% -> passive seller +2.97c/ct [+1.93,+3.90]
  (173 ev), taker -3.28c [-4.22,-2.19]; positive 5/5 days (1.08..4.79c). eBasketball taker SOLD cheap: passive -1.85c [-4.65,+1.73].
  eSoccer, taker bought cheap: 97,556 ct, mean c .046, won 3.63% -> passive +0.92c [-1.99,+3.09] (361 ev); one comeback event
  (OLIVMIA) cost the passive side $970 on 11-08 (-16c/ct that day). eSoccer taker sold cheap: passive +4.32c [-2.49,+12.41].
- Base-rate-shape placebo (maker_band.py -> maker_band_DISC_eb.json / _es.json): ROI on capital at risk of the passive seller
  of whatever the taker bought in-play. eBasketball: rule band c <= .15 ROI +3.10% [+2.00,+4.08] (181 ev) vs placebo band
  (.15,.30] +11.44%; diff -8.34% [-16.61,+1.57]. eSoccer: rule +0.96% [-2.17,+3.25] vs placebo +4.21%.
  -> the passive side earns MORE on mid-priced sides than on longshots: in-play takers lose at every price (spread + speed),
  so this is generic market-making P&L, not a longshot pricing error. Fails the placebo already in DISC -> no rule, TEST not
  pulled. Capturing it needs queue priority and quote-pulling at feed speed (rubric R6 = 0).

## Part B log
- 07:10Z TB1 registered in edgelab: tb1_kalshi_tennis_pin_fresh-0ac23199 (spec edgelab/specs/tb1_kalshi_tennis_pin_fresh.yaml;
  cross_venue now 13 trials, bar p < 3.85e-03; "no vault entry" like ESP1 -> forward only). Pinnacle history: none (see prereg).
- 07:10Z descriptive snapshot, ALL matched matches at any lead time (results/esoccer_tennis/gap_snapshot.py ->
  gap_snapshots.jsonl; not the pre-registered T-90..T-10 sample): 61 matched pre-match matches (all two-sided); Pinnacle quote
  age 784 s (stale-ish); gap = Kalshi home mid - Pinnacle multiplicative no-vig: median -0.0003, median |gap| 0.0093, p90
  0.0209, max 0.0494; Kalshi spread median 2c; side edge (conservative fair - ask - fee) median -0.0315, sides >= +1c: 0 of 122.
  By series median |gap|: ITF M .007 (27), ITF W .0107 (16), WTA/BJK .0135 (9), ATP Ch .0086 (7), ATP .019 (2).
  Snapshot loop every 30 min x 20 (gap_snapshot_loop.log) + the logger (10 h) keep collecting.
- Rules caveat for any Kalshi-vs-Pinnacle gap: Kalshi ITF = retiring player loses / ATP Ch "after a ball has been played";
  Pinnacle voids or settles retirements by its own set rule -> Pinnacle no-vig is not exactly Kalshi fair (retirements ~2-3%,
  rules_quirks l.74). Matters only for gaps of ~1c.

- STATUS at handback (2026-11-21T07:29Z; results/esoccer_tennis/score_b.py -> b_summary.json): 14 ticks, fresh-tick share
  0.143; pre-registered sample (first fresh tick in [start-90, start-10]) n = 4 matches: median gap
  -0.0032, median |gap| 0.0108, max 0.0225, Kalshi spread median 0.015, side edge
  median -0.0317, sides with edge >= 0: 0 of 8; TB1 bets 0. Too few to judge; the logger
  keeps running by hand until ~17:05Z (loop 600 min) and the snapshot loop for 10 h. Lead: rerun score_b.py, then
  `TENNIS_PIN_OUT=results/esoccer_tennis/tennis_pin python scripts/tennis_pin_logger.py settle`. TB1's kill needs 7 logger days ->
  deployment (qs_job.sh case + installer, like esports-pin) is the lead's/user's call; nothing installed.
- READING so far: Kalshi tennis sits on Pinnacle's no-vig line (|gap| ~1c, inside its 2c spread; 0/122 snapshot sides and 0/8
  in-window sides with positive edge after fee). Expect TB1 to die on kill (b) ("< 1 signal/day") rather than produce bets.

## Learnings
### Kalshi tennis match winners sit on Pinnacle's no-vig line pre-match
61 matched matches (07:10Z snapshot): median |Kalshi mid - Pinnacle no-vig| 0.93c, p90 2.1c, max 4.9c, spread 2c; 0 of 122 sides
had conservative fair - ask - fee >= +1c. Source: results/esoccer_tennis/gap_snapshots.jsonl. Forward TB1 logger checks it in-window.
### A positive TAIL - MIRROR at trade prices is the half-spread, not a taker edge
eSoccer trade-price net +$163k/30 d (S-30 min..close, DISC 11-07..11-11), but the taker who sells the cheap side at the bid loses:
pre-start -4.19% [-6.80,-1.66] n651, in-play -2.78% [-3.66,-1.87] n4,184, 0/5 days positive. Longshot buyers lift a ~.12 ask, the
cheap side wins ~10-12%, a seller at the ~.08 bid loses. Re-price any loser pool at the bid before calling it harvestable.
Source: results/esoccer_tennis/a_DISC_blind.json, analyze_a.py.
### Small Horvitz-Thompson event samples miss the MIRROR (comebacks) and overstate a pool's net
pool_census put eSoccer at +$1.06M/30 d net from 18 sampled events (x212); the full DISC window gives TAIL $0.93M/30 d (close to
pool_census's $1.07M) but MIRROR $0.77M/30 d, net +$163k (~6.5x lower); eBasketball +$70.8k vs +$0.50M. Comebacks are rare,
large events: sample them fully. Source: a_DISC_blind.json vs notes/pool_census.md write-up C.
### Kalshi eSoccer / eBasketball are in-play products with a calibrated pre-start book
83% of eSoccer and 90% of eBasketball contracts trade after kickoff (5.70M of 6.82M; 2.55M of 2.82M, DISC). At S-5 min the book
is 98.5% two-sided, median spread 3c; mid calibration within ~3c in every bucket with n >= 100 (eSoccer team [.40,.60) -0.014
[-0.038,+0.009] n1,403). 0 of 48 taker cells robust. Source: a_DISC_calib.csv, a_DISC_cells.csv.
### In-play takers lose at every price; a maker profit in one band is not a band-specific mispricing
Exact tape (700 DISC markets): passive sellers of in-play eBasketball longshots earned +2.97c/ct [+1.93,+3.90], 5/5 days, but
the placebo band (.15,.30] earned more (ROI +11.4% vs +3.1%, diff -8.3% [-16.6,+1.6]). Run the adjacent-band placebo before
spending a TEST window. Source: results/esoccer_tennis/maker_band_DISC_eb.json, tape_DISC_summary.json.
### pandas to_datetime of ISO strings gives datetime64[us]; astype(int64) // 1e9 is 1000x off
tape_maker.py first produced t = 1,794,152 for 2026-11-08 (should be 1,794,152,693): the [us] trap again (memory:
datetime64[us] epoch trap). A one-line range assert on epoch seconds caught it; use (ts - epoch) // pd.Timedelta(seconds=1).
### Kalshi 1-min candles exist only for minutes where something changed
0.49 of the minutes in a window have a candle; 66% of candles have 0 volume (quote-change candles). The quote at T = the latest
candle <= T carried forward; a "stale if > 10 min" guard would drop unchanged quotes. Source: fetch_DISC.log, notes A1.
