# X2 lows: is the forecast window the cause? (2026-10-30)

X2 = the frozen consensus-reject rule (scripts/wx_expand.py) on the KXLOW* daily-LOW series. DEV result
(ledger dataset kalshi_temp_expand, wx_expand_X2_lows): 224 bets, -14.28 u, ROI -6.38%. Suspected cause, never
checked: the forecast min is taken over local hours <= 18:00 of the target day, but the settlement low is the
minimum over the whole local-STANDARD-time climate day (00:00-24:00 LST).

## Step 1 -- diagnosis (DEV only, dates < 2026-03-01)

Script: `results/x2b_lows_window/diag.py`. Hourly obs = IEM ASOS METARs (routine + specials, same fetch and
T-group decoding as scripts/wx_lockin.py), cut to the LST day of each station-day. DEV for lows covers only
7 series (AUS, MDW, DEN, LAX, MIA, NYC, PHL), 436 station-days, 2025-12-13..2026-02-28.

Things that change how to read it:
* **DEV contains no daylight-saving days at all** (Dec 13 - Feb 28). LST = local time throughout, so the
  "DST-shifted hour" half of the hypothesis cannot be tested in dev. The only window defect dev can show is
  the 18:00 cutoff (hours 19-24 are not in the forecast).
* Hourly obs are a slightly warm proxy for the settlement low: settle - (min hourly ob) = 0 on 209/436 days,
  -1 on 140, -2 on 41, -3 on 24, <= -4 on 8 (the true minimum falls between reports). The hour of the min is
  therefore approximate (+-1 h), which is fine for "before vs after 19:00".

Where the observed low falls (hour of first hourly ob equal to the day's min, LST):

| hour LST | 0-3 | 3-6 | 6-9 | 9-18 | 18-21 | 21-24 |
|---|---|---|---|---|---|---|
| all 436 station-days | 102 | 148 | 72 | 7 | 10 | 97 |
| 35 losing X2 bets | 1 | 7 | 4 | 0 | 3 | 20 |

* 24% of all station-days set the low at or after 19:00 LST; among the **35 losing bets it is 23/35 (66%)**.
  20 of those 23 have the min at the LAST ob of the day (23:51-23:54 LST): temperature still falling through
  midnight (cold front / cold advection), exactly the hours the forecast never looked at.
* **All 23 late-low losers would have WON if the day had ended at 18:59 LST** (obs min over 00-18 was outside
  the bracket in every one). They are window losses, not forecast misses.
* Winners are also enriched for late lows (105/189 = 56%): the rule selects falling-temperature days, because a
  warm 00-18 forecast min relative to the bracket is what triggers a "below" rejection. So the defect is not a
  random leak -- it sits on the rule's own selection.
* Side: 34/35 losses are brackets BELOW the forecasts ("less" 16, "between" below 18); only 1 is an "above"
  reject. That is the side a truncated window biases (min over a sub-window >= min over the whole day, so it can
  only make the forecast low too warm). "greater": 7 bets, 0 losses.
* The other 12 losses are pre-dawn lows (00-09 LST) that the models simply forecast too warm: 9 of them KAUS
  (Austin Bergstrom, a radiational-cooling site). By station: AUS -8.58/69, DEN -4.98/29, MIA -1.39/22,
  NYC -0.54/23, MDW +0.68/34, PHL +0.19/36, LAX +0.34/11. By month: Dec +0.03/3, Jan -7.96/120, Feb -6.36/101.
* Loss rate 15.6% vs break-even 9.7% (avg NO price 0.898). Removing only the 23 window losses would leave
  12 losses / 224 = 5.4%, below break-even -- but that is an upper bound, not a forecast: a full-window forecast
  also changes which brackets get picked.

**Diagnosis: timing explains the majority (23/35, 66%) of X2's dev losses**, all through the 18:00 cutoff (no
DST days exist in dev). The remaining third is a genuine model warm bias on pre-dawn lows, concentrated at KAUS.
Proceeding to one pre-registered fix.

## Step 2 -- PRE-REGISTRATION: X2b lows, full-settlement-window forecast (written before any X2b number exists)

Honesty note: the dev outcomes above were seen while diagnosing, and the fix was chosen because of them, so a
dev pass is weak evidence. **The real test is SEALED (2026-03-01 .. 2026-10-18, the last settled date in
data/wx_expand.parquet), which has never been loaded for lows** (only the count of series/station-days was read
for the power check: 24 low series, 4,428 station-days; no prices, forecasts or outcomes). Sealed also contains
the DST months, the half of the hypothesis dev could not test.

**Single change vs X2.** Per model, the forecast min is taken over the full settlement window = hourly values
at **LST 00:00 through LST 24:00 inclusive** of the target day (25 hourly values; in DST that is local 01:00 on
the target day through 01:00 the next day). The 24:00 value is the only buffer, chosen from Step-1 physics:
20/23 late losers set their min at the day's last ob, i.e. still falling at the boundary, and the next-midnight
value is the best hourly proxy for the 23:59 temperature. No other buffer, margin or filter changes.

**Leak control (pre-decision only).** The existing rule uses `temperature_2m_previous_day1` (run ~24 h before each
valid hour) only up to 18:00 local on the target day = decision (22:00 local the day before) + 20 h. X2b keeps
exactly that: valid times <= decision + 20 h use previous_day1; later valid times (local 19:00 .. LST 24:00) use
`temperature_2m_previous_day2` (run ~48 h before the valid hour, i.e. initialised >= ~21 h before the decision).
A model counts as available if >= 20 of the 25 hourly values exist. Times are fetched in GMT and converted with
the station's fixed standard offset, so no wall-clock/DST ambiguity. Alignment check before scoring: recomputing
the old local-hours<=18 previous_day1 min with the new code must reproduce the cached `fcsafe_*` columns.

**Frozen (identical to X2):** models {gfs_seamless, ecmwf_ifs025, icon_seamless, gem_seamless, ncep_nbm_conus}
with their start dates; >= 3 available; margin 3F with the same greater/less/between reject formulas; decision
22:00 local the day before (cached hourly candle prices); tradeable quotes (yes_bid > 0, yes_ask < 1,
spread <= 10c, yes_ask > 0.01); buy NO at 1 - yes_bid; fee 0.07*p*(1-p); 1 unit of cost per bet; drop
result_mismatch rows. Station coordinates/timezones from api.weather.gov as in wx_expand build.

**Trials (dataset kalshi_temp_expand):**
1. `wx_expand_X2b_lows_window` DEV (dates < 2026-03-01): PASS iff one-sided p < 0.05 (bootstrap over station-days,
   4000 resamples, seed 0, p = share of resampled total units <= 0) AND units > 0 in both halves by date
   (split at 2026-01-21, second half = dates >= split; the calendar midpoint of 2025-12-13..2026-02-28).
2. ONLY if trial 1 passes: `wx_expand_X2b_lows_window_SEALED`, scored ONCE on 2026-03-01..2026-10-18:
   PASS iff p < 0.05, ROI > 0, and units > 0 in both halves (split at 2026-06-24, calendar midpoint; second half = dates >= split).
   If dev fails, sealed stays unloaded and X2b is dead.
Reported: n, observed ROI with bootstrap 2.5/97.5 interval of ROI (the observed value, not the resample mean),
p, halves, loss rate vs break-even loss rate, avg NO price, picks/month, by station and by side.

**Power.** Per-bet P&L SD at these prices ~0.30-0.33 u (win ~+0.11, loss -1, break-even loss rate ~10%).
DEV: X2 had 224 bets on 436 station-days; X2b should be similar (~150-250) -> SE(ROI) ~ 2.2%, so dev can only
detect ROI >~ +3.6% at p < 0.05. An E1-sized edge (+1.45% sealed) would most likely FAIL dev: the dev gate is
under-powered and its failure means "not shown", not "absent". SEALED: 4,428 station-days, ~10x dev; at dev's
pick rate ~1,500-2,300 bets -> SE ~ 0.7-0.8% (more with station-day clustering), detects ROI >~ +1.3%.
**Too-good guard:** ROI > +10% on > 200 bets -> inspect rows (settlement vs forecast day alignment, UTC/LST
offsets, previous_day1/day2 issue times) before reporting.

## Step 3 -- results

`python scripts/x2b_lows_window.py dev --ledger` (2026-10-30; ledger row wx_expand_X2b_lows_window, sha
2ac558c9a920; bets in results/x2b_lows_window/bets_dev.csv).

| trial | n | units | ROI (obs) | ROI 2.5/97.5 | p (1-sided) | half 1 (<01-21) | half 2 | loss rate | break-even loss rate | avg NO | picks/mo | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| X2 (old window, reference) | 224 | -14.28 | -6.38% | -- | -- | -- | -- | 15.6% | 9.7% | 0.898 | -- | dead |
| **X2b DEV** | 116 | -9.94 | **-8.57%** | [-15.2%, -2.0%] | 0.995 | -2.71 / 38 | -7.23 / 78 | 14.7% | 7.4% | 0.921 | 50 | **FAIL** |
| X2b SEALED | not run -- gated on a dev pass; sealed lows remain unloaded | | | | | | | | | | | |

By station: AUS -6.40/39, DEN -1.05/23, NYC -2.40/10, PHL -0.82/17, MDW +0.50/14, LAX +0.12/5, MIA +0.11/8.
By side: below -7.50/86, above -2.44/30. Versus X2 on the same rows: 94 picks kept, 130 dropped (their units
-6.02), 22 new.

**What happened.** The fix did what it was built to do: late-evening (>= 19:00 LST) losses fell from 23 to 2.
It also removed most of the X2 bets (the fuller window rejects fewer "below" brackets) and the survivors trade at
a thinner price (avg NO 0.921, break-even loss rate 7.4%). The 17 remaining losses are forecast misses, not
timing: 2 are late lows, 12 are the same pre-dawn losers as X2 (9 at KAUS, where even the coldest model forecast the low 4-12F too warm),
and 3 are new "above" losses at NYC/PHL where every model forecast the low too cold (4 "above" losses in all). A 14.7% loss rate against a
7.4% break-even is not a window problem.

**X2 stays DEAD; X2b DEAD at the dev gate (p 0.995, both halves negative). Cause of X2's loss = two things:**
(1) the 18:00 window cutoff (23/35 losses; fixed by X2b), and (2) the 3F margin is not enough for daily lows --
lead-1 model lows miss by > 3F far more often than highs do (radiational cooling at KAUS/DEN, cold-air bias at
NYC/PHL). (2) is what kills it. Sealed was never loaded, so it remains clean for any future lows idea.

**Something off found on the way (affects all wx forecast caches, not the live rule's leak-safety).** Open-Meteo
labels a whole response with ONE `utc_offset_seconds` -- the offset in force on the REQUEST date -- not the
DST-correct offset per date (a Chicago request made in October returns GMT-5 for January dates). wx_expand's
build ran 2026-10-19 (DST), so every winter "local hour <= 18:00" window in data/wx_expand.parquet is really
23:00 previous day .. 17:59 standard time. The alignment check reproduced the cached fcsafe columns with 0
mismatches only after emulating that fixed offset (6% of KAUS GFS bracket-rows differed with DST-correct
times). For highs this is harmless (earlier cutoff = still pre-decision); a cache built in winter would shift
summer windows to <= 19:00 local, still before the 22:00 decision. sports/wx.py `forecast_daily_max` uses the
same `timezone=tz` request and inherits it. X2b requests GMT and converts itself, so it is unaffected.
