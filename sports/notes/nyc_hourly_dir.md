# nyc_hourly_dir -- KXTEMPNYCHS "Hourly Directional NYC Temperature" (agent started 2026-11-16)

Lead: notes/series_watch/2026-11-15.md (score 8/10, "newly active"). Prior: HT1 (same series family) dead,
notes/kalshi_hourly_temp.md.

## Census (read-only GETs, 2026-11-16 ~10:30Z)
- NOT new: KXTEMPNYCHS has run since 2026-09-10 (1,319 events / 13,199 markets to 11-06 in
  data/kalshi_hourly_temp/markets.parquet). HT1 already traded it (consensus-reject maker at H-45, dead). The
  series_watch "first market created 11-15" only saw the currently-open window. Title "Directional" = same contract.
- Series meta (GET /series/KXTEMPNYCHS): hourly, fee_type quadratic, fee_multiplier 1, source "Synoptic Data",
  contract terms NYCWINDEX.pdf (saved /tmp/nycdir_dl/terms.pdf).
- Event = one hour H (ET). ~10 strikes, 1F apart, "above K.99" (strictly greater). Market opens H-60 min, closes at H.
  Example KXTEMPNYCHS-26NOV1605: strikes 30.99..39.99, expiration_value 35.60.
- **Underlying is NOT one station.** Kalshi Weather Index NYC (nyc-temperature-v1.0): equal-weight (0.125) average of
  8 ASOS stations' 1-minute HF-ASOS temps (KLGA KEWR KTEB KCDW KHPN KFRG KSMQ KISP; NOT Central Park/KJFK), each
  offset-adjusted z_i = x_i - b_i + B_c (b: LGA/EWR/TEB +1.0C, HPN/SMQ -1.0C, others 0; B_c +0.125C; offsets
  re-estimated weekly, step clipped to +-0.5C), computed in C, published in F to 0.01. Quorum >= 7 members.
  Fallback per member: prior-minute HF-ASOS / METAR T-group up to 75 min old. Receipt deadline t+5 min.
- Settlement value for hour H = canonical index at event minute H (or the latest minute <= H, max age 60 min), fixed at
  H+5 min. Kalshi publishes every minute's index publicly on its website -> the market sees the index with <= 5 min lag.
- Volume (HT1 markets file, 09-10..11-06): per-market median 19 contracts, mean 606; 55% of markets ever trade.

## Implication
The forecasting problem is a 20-60 minute nowcast of an 8-station average whose current value is public. The
information edge (if any) is: (a) the market not updating to the latest index minutes, (b) a better short-range trend
(1-min station data, diurnal shape). HT1 used only the H-2 settled value + day-old NWP -- much weaker information.

## Data
- Station 1-min temps: IEM asos1min (integer F, == ASOS 1-min record that Synoptic relays). Coverage Sep 10-30:
  LGA 20.6k, EWR 18.6k, TEB 16.7k, CDW 1.7k (!), HPN/FRG/SMQ/ISP ~25.6k of 30.2k minutes. IEM lags ~15-20 h (backtest
  only; live would need Kalshi's own index page or Synoptic).
- Quotes: Kalshi event candlesticks, 1-min (yes_bid/yes_ask OHLC) per event. Tape: /historical/trades.

## PRE-REGISTRATION (written 2026-11-16 before any price or outcome of the test window was joined)
Question: at T = H - 20 min, does a public-information nowcast of the NYC index at H find the market overpricing the
other side by >= 3c after fee?
Split: same boundaries as vault kalshi_temp_hourly (event-ticker local date): dev 2026-09-10..09-30 (NYCHS only),
sealed 10-01..11-05 (NOT opened), forward 11-07+.
Reconstruction R(t): for each station take its 1-min tmpf (carry forward <= 10 min), C = (F-32)/1.8, z = C - b_i + B_c
with v1.0 offsets; R = mean of available z, need >= 5 stations; F out.
Nowcast at T: t0 = T - 6 min (index public with <= 5 min lag), L = H - t0 = 26 min.
  slope = (R(t0) - R(t0-30)) / 30; Xhat = R(t0) + beta * slope * L.
  beta and the residual distribution e = R(t0+L) - Xhat are fit ONLY on station data 2026-08-20..09-09 (no market
  data exists for NYCHS then), separately for 3 local-hour-of-H bins: night (19-07), morning (08-12), afternoon (13-18).
  Assumed reconstruction error: add N(0, 0.3F) to the residuals (fixed now, not tuned).
  p_yes(K) = P(Xhat + e > K) from the empirical CDF.
Rule A (primary, taker): quote = last 1-min candle with end <= T. Need 0 <= yes_bid < yes_ask <= 1, spread <= 10c.
  Side S = yes if p_yes >= 0.95, no if p_yes <= 0.05. Price a = S ask (no ask = 1 - yes_bid).
  Bet iff p_S - a - fee(a) >= 0.03 (fee = Kalshi quadratic taker 0.07*a*(1-a), exact). One bet per event: the market
  with the largest edge. Hold to settlement. Cluster = event date.
Rule B (secondary, maker): same selection with a' = a - 0.01 and fee 0; filled iff >= 10 contracts print at <= a' in
  (T, H]. Report filled vs unfilled loss rate.
Success: edgelab dev pass (p < family bar, ~7e-4) AND day-clustered net-ROI CI lower bound > 0, n >= 100.
Kill: CI crosses 0, or n < 100 in dev (-> forward logger proposal only), or CLV negative.
Diagnostics (not tuning): recon R(H) vs expiration_value MAE; nowcast vs market-mid Brier at T on all markets.

## Log
- 2026-11-16 station 1-min pulled (results/nyc_hourly_dir/raw/*.csv, IEM asos1min 08-19..10-01). R.parquet: R defined
  on 85% of minutes (>= 5 stations). CDW only ~54% coverage.
- Nowcast fit (scripts/nyc_hourly_dir.py model -> results/nyc_hourly_dir/model.json): beta 0.722, n 504 hours,
  residual sd 0.50 / 0.51 / 0.50 F (afternoon/morning/night); persistence-only sd 0.73 F.
- Recon diagnostic (dev events, R(H) vs expiration_value, n 341): bias +0.04 F, MAE 0.37, sd 0.50 (sd 0.44 even with
  7 stations). Pre-registered recon noise 0.3 F was too small -- kept as registered.
- Candles: 488 dev events, 125k 1-min candle rows (results/nyc_hourly_dir/candles.parquet). Frame
  data/nyc_hourly_dir/frame.parquet: 4,880 markets, p_yes defined on 69%, quote on 99%. Maker tape 1,981 prints.
- edgelab (family kalshi_weather_temp, bar 6.85e-4 at 73 trials):
  - nd1_nyc_nowcast_taker-01df6e59: schema_rejected (price_source label 'taker_ask' bug) -> retired, re-registered as
  - **nd1b_nyc_nowcast_taker-3f70866c dev FAIL -> dead**: n 80 (17 days), net ROI +5.1% [-7.8, +18.8], p 0.25,
    halves +13.0% / -2.8%, CLV +1.1c (p_neg .72). Also below prereg n >= 100.
  - **nd1_nyc_nowcast_maker-d0b3948a dev FAIL -> dead**: n 64 filled, net ROI +2.8% [-10.4, +20.2], p 0.40,
    halves +7.2% / -1.6%, CLV -0.5c.
- Loss rates (exact binomial): taker 11/80 = 13.8% [7.1, 23.3] vs model-claimed 1.7% and market-implied 16.6%.
  Maker FILLED 11/64 = 17.2% [8.9, 28.7]; UNFILLED 0/33 [0, 10.6] -> adverse selection.
- Calibration on all 3,072 quoted markets at T: Brier model 0.0259 vs market mid 0.0145; day-clustered diff
  +0.0110 [0.0079, 0.0141] (bootstrap over days) -> the market is clearly sharper than our nowcast.

## Verdict: DEAD (dev). No sealed run.
Why: the settlement index is published by Kalshi every minute (<= 5 min lag), so at H-20 the market already knows the
true current value; our IEM reconstruction misses it by ~0.5 F sd (integer-F 1-min, gaps, weekly offset drift) -- the
same size as the 26-min physical change. The "near-certain" picks were where our recon disagreed with the true index.

## Forward-logger proposal (low prior, NOT built)
Only a test with the SAME information as the market is meaningful: log Kalshi's published index minutes (web endpoint
still to be discovered) + 1-min quotes at T in {H-20, H-10}; nowcast = true index(T-1) + trend. Physical change over
10-20 min is ~0.3-0.5 F, the market already prices off that feed, per-market volume median 19 contracts -> small EV.
Recommend not building unless a cheap index feed turns up.

## Learnings
### KXTEMPNYCHS settles on an 8-station Kalshi Weather Index, not Central Park
Kalshi Weather Index NYC v1.0: equal-weight 1-min HF-ASOS average of KLGA KEWR KTEB KCDW KHPN KFRG KSMQ KISP with
per-station offsets (refreshed weekly), value at minute H (latest <= H, max age 60 min); source NYCWINDEX.pdf. Any NYC
hourly model on KNYC/KJFK models the wrong underlying.
### When the settlement value is published live, the market knows the present; a reconstruction loses
IEM reconstruction error sd 0.50 F (n 341) vs 26-min nowcast residual sd 0.50 F; model Brier 0.0259 vs market 0.0145
(diff +0.011 [0.008, 0.014], 3,072 markets). Taker picks lost 13.8% vs model-claimed 1.7%. Check whether the
underlying is public in real time before building a nowcast edge.
### Maker fills on hourly temperature are adversely selected (again)
ND1 maker: filled 11/64 lost (17.2% [8.9, 28.7]) vs unfilled 0/33. Same pattern as HT1/E4 maker diagnostics.
### series_watch "newly active" can mean only the currently-open window
KXTEMPNYCHS was flagged first-created 11-15 but has run since 09-10 (1,319 events) and was already HT1's series.
Check the markets archive before treating a series as new.
