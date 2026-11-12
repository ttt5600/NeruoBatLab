# e5_fix -- can E5 ForecastEx maker bids be saved from adverse selection? (2026-11-11)

## Question
E5 (FX weather consensus-reject) maker bids rest from ~22:00 local D-1 to cutoff ~03:44Z D+1 and fill only when an
informed day-D seller arrives (e5_forecastex_audit.md: filled loss rate 12/53 = 22.6% vs 0/623 unfilled). Do any of
three execution fixes give executable ROI > 0 with a CI?
- F1 time cutoff (cancel at a fixed local time), F2 observation-triggered cancel (running max near our threshold),
  F3 lower bid price.

## Data
- Timestamped PROXY: Kalshi KXHIGH* consensus-reject picks already cached for L24/E4:
  data/kalshi_temp_maker_{picks,trades}.parquet (DEV E1+X1, 1,713 picks, 2024-03..2026-02, full public tape in (T, close]).
  data/kalshi_temp_maker_{picks,trades}_sealed.parquet = E1/X1 sealed-period picks; that lockbox was already opened
  10-23 for the maker replay, so it is used here only as a SECOND REPLICATION sample (not a clean holdout).
- Observations: IEM ASOS archive (METAR tmpf, incl. specials) for the settlement stations (sports/wx.py CITIES,
  wx_expand station column). Kalshi settles NWS CLI; FX settles WU daily observations (hourly METAR max) -- so the
  METAR running max is exactly FX's settlement input and a slight under-estimate of Kalshi's.
- FX itself: results/e5_forecastex_audit/settled_picks.csv (693 picks, 10-29..11-08; daily OHLC only, low_d = day-D low).
  FX dev = event_date < 2026-11-01 (185 picks), FX holdout = 11-01..11-08 (508 picks).
- IBKR intraday history for FX contracts: probe (results/e5_fix/ib_hist_probe.py, read-only clientId 93).

## Pre-registered design (written BEFORE any fill/outcome by time, obs or price was computed)
Analog of the FX order on Kalshi: post a NO bid at q0 = Kalshi NO best bid at T (= 1 - yes_ask at the decision candle,
~22:00 local D-1); FX rests at the best bid too. Fill = cumulative count of prints with no_price <= q in (T, cancel]
>= N. PRIMARY N = 10 (size-through, the L24 standard); N = 1 reported. Cost = q + $0.01 (FX commission) per contract;
pnl per $1 cost = 1/cost - 1 if win else -1. CI = cluster bootstrap over calendar dates, 4000 draws, seed 0.
Side: Kalshi brackets are mapped to "above-side" (losing region above the consensus: greater-strike or between-bracket
with floor > settlement-free reference) vs "below-side" by comparing the bracket to the event's highest-priced
bracket at T (market mode, known at T). FX: NO picks = above-side, YES picks = below-side.
Baseline B0 = rest until close (today's E5 analog).

Trials (13, bar p < 0.05/13 = 0.0038 one-sided, ROI > 0):
- F1 cancel at event-day local 00:00, 06:00, 09:00, 12:00, 15:00 (5 trials).
- F2 above-side: cancel at the first METAR (obs time + 15 min assumed latency) whose running max since local midnight
  of D >= L - X, L = the lowest losing temperature, X in {2, 4, 6} F (3 trials). Below-side picks under F2: cancel at
  local 12:00 on D unless the running max already clears the losing region (then keep resting; outcome decided).
- F3 bid lower: q = q0 - k, k in {2, 5, 10, 15, 20} cents, rest to close (5 trials).
Combos (F1xF3, F2xF3) are DESCRIPTIVE only, not trials.

Success (per fix): proxy ROI on filled bets > 0, date-cluster 95% CI lower bound > 0, p < 0.0038; AND the proxy is
validated on FX where FX data allows it (F3: FX fill iff low_d <= q0 - k; F1: FX fills before day D = D-1-dated
trades, last_d1; F2: needs intraday FX prints -> only if the IBKR probe returns history), with FX ROI point > 0.
Kill: no fix passes on the proxy, OR the passing fix has FX point ROI <= 0 / FX fills ~0 (capacity < $1/day) ->
recommend STOP E5, not fixable with these levers.
Known non-transferability (stated before scoring): Kalshi has uninformed flow that FX lacks, so proxy fill RATES do not
transfer; what transfers is the TIMING/obs-state profile of informed fills (when do losers fill vs winners).

## Log
### 1. Design note (before scoring): side assignment uses the 5-model fcsafe consensus at T (the rule's own forecast)
  instead of the market mode -- both known at T; consensus is what the rule uses. No outcome seen when changed.
### 2. FX daily-file timing (results/e5_fix/fx_picks_d1.csv)
- File D-1 has a print for 0/693 picks (so a cutoff before day D's FX session = ~0 fills). File D: 152. File D+1
  (evening of D, after the high is known): 65, of which 3 below our bid, all winners (1 not already filled via low_d).
### 3. F3 on FX directly (results/e5_fix/fx_check.py -> fx_check.json; fill iff day-D/D+1 low < bid - k)
- all 693: k=0 54 fills 12 losses ROI -14.4% [-30.8, +3.2]; k=5c 36 fills 12 losses -22.1%; k=10c 25 fills -33.5%;
  k=20c 20 fills -41.0%; k=40c 16 fills -46.1%. Every loser fills at every price (a losing contract goes to ~0);
  lowering the bid only sheds winners. F3 FAILS on FX (point ROI falls monotonically with k).
- Split: dev (10-29..10-31, 183 picks) had 0 losses (k=0 +5.3% n10); all 12 losses are holdout (6 station-days).
### 4. Kalshi proxy (results/e5_fix/proxy.py -> proxy_dev.json, proxy_sealed.json; N10 primary, fee $0.01)
- DEV 1,713 picks: B0 rest-to-close at the NO best bid fills 66.9%, loss rate filled 6.1% (all 70 losers fill),
  ROI +3.0% [+1.4, +4.6]. EVERY trial passes the bar, including the baseline -- Kalshi has uninformed flow.
  F1 00/06/09/12/15h: ROI +3.8/+4.0/+3.6/+3.7/+3.0% (loss rate filled 6.6/5.1/5.4/5.3/5.9%). F2 X=2/4/6: +3.5/+3.7/+3.5%.
  F3 -2/-5/-10/-15/-20c: +4.5/+6.2/+10.0/+13.9/+17.7% (fill rate 47->17%; losers still all fill).
- Opened sealed-period sample (4,308 picks, 2025-07..2026-10): B0 +1.4% [+0.7, +2.0]; F1/F2 within +-0.1pt of B0
  (F1 00h fails the bar, p .0095); F3 -5c +4.0%, -10c +5.6%, -20c +8.0%.
- Timing of loser fills (B0 N10, hours after local midnight D; bins -6,0,6,9,12,15,18,30): dev losers 34/7/10/4/10/1/4,
  sealed 63/27/6/4/4/2/0. ~half to 60% of Kalshi losers fill BEFORE local midnight (overnight model runs), not on
  day-D observations. So on Kalshi neither a time cutoff nor an obs trigger removes much adverse selection; F1/F2 move
  loss-rate-filled by <= 1 pt. What makes Kalshi work is the volume of uninformed winner fills (67-77% of picks).
- NON-TRANSFER (stated in prereg): FX winner fill rate is 42/681 = 6% vs Kalshi 67-77%. The Kalshi passes cannot be
  carried to FX; the proxy's role is the timing profile only.
### 5. F2 on FX (results/e5_fix/fx_f2.py -> fx_f2.json, fx_f2_picks.csv)
- METAR (IEM routine+special) daily max reproduces FX settlement on 693/693 picks (WU = hourly METAR max).
- FX fills have no timestamp -> bounds. Optimistic (a fill survives only if the pick never triggers):
  X=2: 22 fills, 0 losses, ROI +8.3%, +1.82 units over 11 days; X=4: 6 fills +0.80u; X=6: 2 fills +0.53u.
  Pessimistic (every fill before its trigger) = B0: 54 fills, 12 losses, -14.4%, -7.79u.
- Loser trigger times (X=2, local h on D): 8.1, 9.1, 9.1, then 13.1-14.1 (9 of 12). Outcome decided ~1-2 h later.
- The rule's margin is 3F, so X >= 4 fires on most NO picks whenever the forecast verifies (X=4: 233 of 534 NO
  picks) -- X=4/6 are badly chosen for FX, X=2 is the only sensible level.
### 6. IBKR intraday FX history EXISTS (results/e5_fix/ib_ticks.py, read-only clientId 93)
- reqHistoricalTicks(TRADES) on FORECASTX contracts returns timestamped prints (price, size). Gotchas: (a) HMDS only
  searches a short window after startDateTime -- a start 2 days early returns 0 ticks; walk 12-h windows;
  (b) contract definitions vanish ~1 week after expiry (10-29..11-02 conids: "No security definition"); 11-05+ resolve.
  First puller run had bug (a) and reported "no ticks" everywhere -- not a data absence.
- 11-10/11-11 events (results/e5_fix/fx_tick_timing.csv; 36 traded picks, all winners by METAR): 10 had a print
  through our bid. Local time of first through-print: 02:47 (a YES), 08:02, 09:12, 11:20, 11:21, 13:15, 13:15, 14:00,
  14:31, 19:20. Gap (losing temp - running max) at that moment: 3, 3, 3, 4, 4, 7, 9, 16, 20 F.
  -> FX winner fills are day-D "scare" fills, half of them when the running max is within 3-4F of losing. A cutoff
  before ~09:00 keeps 1-2 of 10; an X>=4 trigger cancels 5 of 10. Winners and losers fill on the same kind of day.
### 7. FX-NATIVE validation on ticks incl. 6 of the 12 losers (fx_tick_score.py -> fx_tick_score.json, _rows.csv;
  fx_tick_timing_old.csv). Events 11-05..11-08 + 11-10..11-11, 199 picks, 32 through-bid fills.
| variant | fills | losses | ROI | units |
|---|---|---|---|---|
| B0 rest to close | 32 | 6 | -8.9% | -2.86 |
| F1 cancel 00:00 | 1 | 0 | +40.9% | +0.41 |
| F1 cancel 06:00 | 3 | 0 | +26.5% | +0.79 |
| F1 cancel 09:00 | 7 | 1 | +1.5% | +0.10 |
| F1 cancel 12:00 | 17 | 4 | -11.4% | -1.93 |
| F1 cancel 15:00 | 31 | 6 | -9.5% | -2.95 |
| F2 X=2 | 29 | 6 | -10.8% | -3.13 |
| F2 X=4 | 19 | 5 | -15.4% | -2.93 |
| F2 X=6 | 17 | 4 | -11.8% | -2.00 |
- Loser first-through-print, local h on D: 7.5, 9.7, 9.8, 10.6, 12.8, 13.1. Gap to losing temp at that moment:
  14, 17, 6.4, 7, 3, 5 F. The METAR X=2 trigger fired at 13:07-14:08 -> ALL 6 losers filled BEFORE the obs trigger,
  and before the outcome was decided. The informed sellers trade on day-D FORECAST updates (morning model runs),
  not on observations. F2 cannot fix this.
- Winner fills (26): -0.6, 0.3, 2.8, 6.2, 7.1, 8.0, 9.2, then 10-14.5 h (17 of them), 19.4. Clean zone = before
  ~07:00 local D (no loser fill earlier than 07:31) but only 3 fills in 6 event-days live there.
## Verdict (against the pre-registered rules)
- F3 (lower bid): DEAD. Passes on the Kalshi proxy, fails on FX (fx_check.json: losers fill at every price; ROI falls
  -14.4% -> -46% as k goes 0 -> 40c).
- F2 (obs-triggered cancel): DEAD. Passes on the proxy but on FX ticks every loser filled before the trigger;
  X=2 -10.8% on 29 fills vs B0 -8.9%.
- F1 (time cutoff): 00:00/06:00 point ROI > 0 on FX but on 1 and 3 fills (no CI possible; not distinguishable);
  09:00+ ~0 or negative. Capacity of the clean window: 3 fills / 6 event-days = 0.5 fills/day; at 20 contracts x ~$0.85
  = ~$17 per fill x +26% = ~$2/day best case. Proxy timing agrees only partly (Kalshi losers often fill before
  midnight -- on FX there are no D-1 prints at all, the FX session effectively starts on D).
- OVERALL: no fix makes executable ROI > 0 with a CI. The FX book has almost no uninformed sellers; the few fills a
  resting bid gets are mostly day-D forecast-informed. Kill rule (capacity < $1/day or FX fills ~0) is met for
  everything except F1 06:00, which is at ~$2/day best case on n=3.
## PROPOSAL for the lead (do not deploy without the lead/user)
1. Stop resting E5 bids through day D now (B0 is -8.9% on ticks, -14.4% [-30.8, +3.2] on daily lows).
2. If the lead wants a cheap live probe instead of a full stop: executor change = cancel every E5 bid at 06:00
   station-local on event day D (place as now at ~22:00 local D-1; set cutoff = local 06:00 D instead of ~03:44Z D+1).
   No price change, no obs feed needed. Forward kill (pre-set): any loss in the first 15 fills, or ROI < 0 after 30.
   Expect ~0.5 fills/day -> it takes ~2 months to reach 30 fills. Honest expectation: not worth the ops cost.
3. Evidence logger (read-only, free): weekly pull of IBKR HMDS ticks for every E5 pick (results/e5_fix/ib_ticks.py,
   12-h windows; contract defs survive ~1 week after expiry) so any future E5 variant is judged on real FX fill times.
4. The remaining untested idea is a FORECAST-update cancel (re-run the 5-model consensus on the 06Z D runs; cancel if
   the margin to X shrinks below m). Loser fills came 7.5-13 h local with 3-17F gaps -- consistent with sellers using
   morning model runs. Needs its own prereg (Open-Meteo previous-runs API, lead-0 06Z runs).

## Learnings
### FX weather bids are filled by day-D forecast traders, not observation watchers
On IBKR ticks, all 6 E5 losers (11-05..11-07) were hit 7.5-13 h local with the running max 3-17F below the losing
temperature, before a 2F obs trigger fired (results/e5_fix/fx_tick_score.json). Obs-triggered cancels cannot fix it.
### Lowering a maker bid never sheds losers in a binary that goes to zero
A losing contract trades down to ~0.02, so it fills at any bid; lowering the FX bid 0 -> 40c only removed winners
(ROI -14.4% -> -46%, results/e5_fix/fx_check.json). Price fixes adverse selection only when losers settle near the bid.
### IBKR HMDS serves ForecastEx tick history, but only with a near start time
reqHistoricalTicks(TRADES) on FORECASTX works for ~1 week after expiry (older conids: "No security definition"), but a
startDateTime 2 days before the prints returns 0 ticks -- walk 12-h windows (results/e5_fix/ib_ticks.py).
### Kalshi temp tape is a poor proxy for ForecastEx fills
Kalshi NO bids at the best bid fill 67-77% of picks with loss-rate-filled 3-6% and pass every fix; FX fills 8% with
22% losers. Uninformed flow differs by venue; transfer only timing, and even timing differed (Kalshi losers fill
before midnight, FX has no D-1 prints) (results/e5_fix/proxy_dev.json, proxy_sealed.json).
### METAR max reproduces ForecastEx settlement exactly
IEM ASOS routine+special tmpf daily max (local day) matched FX settlement on 693/693 picks (results/e5_fix/fx_f2.json):
outcomes can be scored the same evening from METARs without waiting for the daily file.
