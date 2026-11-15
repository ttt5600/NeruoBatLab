# nfl_wx_updates -- does Kalshi's NFL total ladder lag weather-forecast UPDATES? (agent nfl_wx_updates, 2026-11-15)

Question: when the as-issued forecast for an outdoor NFL game's kickoff window changes a lot between two model runs,
does the KXNFLTOTAL ladder reprice within minutes, or is there an executable lag (buy the side the update favours at
ask + fee within 10 min of the update's public time and beat the Kalshi close)?
Different from the dead items (F2 rain unders, kicker x wind A1, in-game weather oracle): those traded the weather
LEVEL; this trades the TIMESTAMP of a forecast change vs the price path.

## PRE-REGISTRATION (written 2026-11-15 before any forecast change was joined to any Kalshi price)

Data
- Games: nflverse schedule (`data/nfl_schedule.csv`), roof == "outdoors", season 2025 (REG+POST) and 2026 games with
  kickoff < 2026-11-01 00:00 ET. HOLDOUT: 2026 games kicking off on/after 2026-11-01 are not fetched or scored.
- Kickoff UTC = gameday + gametime localised America/New_York per date -> UTC. Differences via // Timedelta(seconds=1).
- Forecast vintages: IEM MOS archive (`mesonet.agron.iastate.edu/api/1/mos.json`), station = nearest airport ASOS
  (mapping from scripts/nfl_kicker_wx.py STATIONS, extended for new stadium ids). Two models, each 4 runs/day:
  NBM text NBS (runtimes 01/07/13/19Z; NWS gridpoint forecasts are NBM-based) and GFS MOS MAV (00/06/12/18Z).
- Public time (when a trader could have it) = runtime + 1.5 h (NBS) / + 4 h (GFS MAV). These latencies are my
  estimate, not measured; a later public time only makes the test more conservative for the trade.
- Window value per run: wind = mean `wsp` (kt x 1.15078 -> mph) over forecast times in [KO - 1.5 h, KO + 3 h];
  precip = max `p06` over 6-h periods (ftime - 6 h, ftime] overlapping [KO, KO + 3 h]. A run counts only if KO is
  inside its horizon.
- Prices: Kalshi KXNFLTOTAL 1-minute candles (historical / live candlestick endpoints via scripts/kx_throttle.py),
  forward-filled; a strike's quote is valid when 0 < bid, ask < 1 and ask - bid <= 0.10. YES = total > strike.
  Implied total = PAVA-monotone interpolation of valid mids (same method as kalshi_early_nfl_totals).
- Settlement: the market's own `result`.

Event (fixed now)
- Consecutive runs of the SAME model (6 h apart). dW = new - old wind (mph), dP = new - old p06 (pct pts).
- Wind event: |dW| >= 5 mph and max(old,new) >= 12 mph. Precip event: |dP| >= 30 and max(old,new) >= 40.
- Direction: UNDER if dW >= +5 or dP >= +30; OVER if dW <= -5 or dP <= -30. Wind and precip in opposite directions
  in the same update -> dropped.
- Public time must lie in [KO - 72 h, KO - 20 min].
- T1 (the one counted trial): first qualifying event per game (earliest public time, either model).

Trade (T1)
- At t* = public time + 5 min, take the forward-filled ladder; on the event's side (UNDER = buy NO, OVER = buy YES)
  choose the valid strike whose side ask is nearest 50c. Entry = side ask; fee = 0.07 p (1-p) exact.
- Net CLV = side mid of the same strike at its last valid quote before KO - ask - fee. ROI = (payout - ask - fee)/(ask + fee).
- PASS: n >= 20 AND mean net CLV 95% game-bootstrap CI lower > 0 AND mean ROI 95% CI lower > 0 (5000 draws).
- KILL: mean net CLV <= 0, or n < 20 (=> "dev thin", write the forward-logger proposal).
- Power (before data): expected n ~ 30-80. Per-bet ROI SD ~ 1 -> SE 11-18%, so ROI CI lower > 0 needs ROI ~ +25-35%.
  The CLV leg is the sensitive one (SD ~ 4-6c -> SE ~ 0.6-1c). A fail is "not shown", not "zero".

Descriptive (not counted): signed implied-total move (toward the event side, pts) from the last quote before
runtime and before public time to +15 / +60 / +180 min, vs placebo updates (same games, |dW| < 2 and |dP| < 10).
Also: wind-only and precip-only splits, GFS vs NBS. Uncounted -> leads only.

Edgelab: family nfl_lines, id nfl_wxupd_t1_taker, registered before scoring.

## Log
- 2026-11-15: prereg written; edgelab registered `nfl_wxupd_t1_taker-71a18425` (family nfl_lines now 136 trials ->
  family bar p < 3.68e-04; my own pass rule is the prereg CI rule, the family bar is stricter and would also apply).
- Script `scripts/nfl_wx_updates.py` (mos / events / kalshi / score). Kalshi KXNFLTOTAL market list: 9,640 markets,
  487 events (historical + live settled); 604 rows for events dated >= 2026-11-01 dropped from
  `results/nfl_wx_updates/kmarkets.parquet` (holdout). 1-min candles exist on the historical endpoint (sparse: one
  candle per minute with activity; forward-filled).
- Data: 261 outdoor dev games (2025 REG+POST, 2026 KO < 11-01), 260 matched to a KXNFLTOTAL event; IEM NBS + GFS
  MOS for 25 stations (`results/nfl_wx_updates/mos/`, log `mos.log`). 5,326 consecutive-run updates inside
  [KO-72h, KO-20m]: 45 events in 32 games, 4,710 placebo (`updates.parquet`). First events: 28 precip / 4 wind,
  17 OVER / 15 UNDER, median 50 h before KO (min 1 h). Mostly GFS runs 2-3 days out.
- Wrong turns fixed before scoring T1: (a) live candlestick endpoint names fields `close_dollars` / `volume_fp`
  (historical uses `close` / `volume`) -> all 2026 games first parsed as NaN quotes; re-fetched. (b) the KO-minute
  candle is already pulled wide (e.g. 2025_01_LV_NE bids .01 / asks .99), so "close" = the strike's last VALID quote
  with end < KO, as the prereg says; the first draft used the raw KO-minute state. (c) dropped a 6-h staleness
  filter: Kalshi omits 1-min candles with no change, so the last candle stands.

## RESULT (2026-11-15) -- T1 DEAD (pre-registered kill: mean net CLV <= 0)
Source: `results/nfl_wx_updates/results.json`, bets `t1_bets.csv`, reaction rows `reaction.parquet`.

| item | n | estimate [95% game bootstrap] |
|---|---|---|
| T1 net CLV (close side mid - ask - fee) | 31 | **-1.99c [-3.17, -0.86]** |
| T1 ROI to settlement | 31 | -5.4% [-37.5%, +27.7%] (win 48.4% at mean ask 50.1c) |
| gross (close mid - ask); fee alone | 31 | -0.24c ; fee 1.75c |
| by dir: OVER / UNDER net CLV | 16 / 15 | -3.1c / -0.8c |
| by season 2025 / 2026 net CLV | 15 / 16 | -2.6c / -1.4c |

Descriptive (signed toward the event side, implied total in pts; 42 event updates, NOT game-clustered; 4,646 placebo):
| window | event signed move | placebo mean abs move |
|---|---|---|
| runtime -> public | -0.13 [-0.23, -0.04] | 0.07 |
| public -> +15 min | -0.02 [-0.07, +0.02] | 0.015 |
| public -> +60 min | -0.005 [-0.07, +0.04] | 0.045 |
| public -> +180 min | -0.01 [-0.10, +0.07] | 0.09 |
| public -> KO-1 min | +0.05 [-0.27, +0.36] | 0.62 |
| 6 h before runtime (anticipation) | +0.05 [-0.05, +0.14] | -- |

Reading: the ladder does not move toward the forecast change at all -- not in minutes, not by kickoff. There is no
lag to harvest because there is no repricing. Gross close-mid minus ask is ~0 (-0.24c), so the taker just pays the
fee + half-spread (~2c). Most dev events are precip-probability swings 2-3 days out, which the market (consistent
with F2 rain unders dead) treats as noise. The 4 wind events are too few to say anything. The runtime->public
-0.13 pts is against the event side and not game-clustered; treat it as noise.
Caveats: MOS public latencies (+1.5 h NBS, +4 h GFS) are estimates; airport MOS, not stadium; hourly HRRR runs not
tested (IEM has no NBH; NBS only 01/07/13/19Z); not critic-checked (a kill, cheap to accept).
Edgelab: `nfl_wxupd_t1_taker-71a18425` retired -> dead.

## Forward-logger proposal (weeks 11-18) -- LOW priority, only for a wind-only follow-up
Dev was not thin (n 31) but wind events were (4). A wind-only forward test would need: hourly HRRR/NBM (Open-Meteo
`models=hrrr` or the NWS gridpoint JSON, polled hourly Thu-Mon for outdoor games); log (issue time, kickoff-window
wind); event = |dWind| >= 5 mph with max >= 15 inside KO-24h; 1-min KXNFLTOTAL snapshot at event +0/+15/+60/+180 and
KO-1; same kill rule. Expected ~1-3 events/week Nov-Jan -> ~15-25 by week 18. Evidence against: books already move
the total -0.22 pts/mph open->close (notes/nfl_kicker_wx_matchups.md), and this ladder showed no reaction on 31
games. Recommend NOT building it unless an existing logger already polls Open-Meteo hourly (E4 does).

## Learnings

### Kalshi NFL total ladders do not reprice on MOS forecast updates, so there is no lag to trade
32 outdoor dev games with a large GFS/NBS run-to-run change (mostly PoP): implied total moved -0.02 / -0.005 / -0.01
pts toward the update at +15/60/180 min (CIs span 0), +0.05 [-0.27, +0.36] by KO. Taker at public+5 min: net CLV
-1.99c [-3.17, -0.86], n 31 = fee + spread. Source results/nfl_wx_updates/results.json. An information-timing edge
needs information the market actually prices; a forecast change it ignores is not a lead, however fast you are.

### Kalshi live vs historical candlestick schemas differ (close vs close_dollars)
`/historical/markets/{t}/candlesticks` returns yes_bid.close / volume; `/series/{s}/markets/{t}/candlesticks`
returns yes_bid.close_dollars / volume_fp. A parser written for one silently yields NaN quotes for the other (all 16
2026 games skipped on the first pass). Parse both keys.

### The KO-minute candle on KXNFLTOTAL is already pulled wide; take the close from the last valid quote before KO
At end_ts == scheduled KO many strikes show bid .01 / ask .99 (2025_01_LV_NE). A "state at KO" close drops games or
biases CLV; use the last valid (spread <= 10c) candle with end < KO.

### IEM archives NBM text (NBS) only at 01/07/13/19Z and has no NBH
mesonet mos.py with model=NBS returns 4 runs/day; NBH queries return nothing. Hourly forecast vintages need another
source (Open-Meteo single-runs, HRRR grib). IEM mos.py rate-limits ("slow down"): ~30 s per station-season request.
