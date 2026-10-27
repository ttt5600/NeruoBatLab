# Kalshi winter weather: monthly rain / snow ladders, "unreachable" NO (DEV only), 2026-10-27

**Verdict: NO PASS (0/6 trials clear the bar).** The best variant, W1m (rain-unreachable NO as a resting maker bid),
earns +3.6% ROI on 146 dev bets with one loss. That is in the right direction and positive in both halves, but its
market-null p is .016, which misses the bar of .0083, and it drops to p .12 when counted as one bet per city-month.
Locked-YES brackets (W2/S2) never trade, because their YES ask sits at $1.00. **The sealed test is NOT worth asking for
yet.** The sealed window (rain Jul-Sep 2026, snow Mar 2026) would give about 40-50 rain bets and about 10 snow bets.
At a 2-4c edge per bet, even a loss-free run of that size can't reach p < .05 against the market null. Better plan:
log forward through this winter for free, then ask for one sealed shot on Jul 2026-Feb 2027 together.
Money scale: about +0.6 units/month (rain, maker) and about +0.5 units/month in snow months, at $1 per bet. That is
pocket change even if it is real.

Script: `scripts/kalshi_wx_winter.py`. The pre-registration in its docstring was written before any price, forecast
or observation was pulled. Data: `data/kalshi_wx_winter{,_obs,_dev_picks}.parquet`. Ledger: 6 trials, dataset
`kalshi_wx_winter`, bar p < 0.0083.

## Step 1: Kalshi weather census (public /series, category "Climate and Weather", 458 series)
| family | series | structure / settlement | history (settled events) | volume |
|---|---|---|---|---|
| Daily high (E1/E2) / low (X2 dead) | KXHIGH*, KXLOWT* | brackets, NWS CLI / TWC | years | traded already |
| **Monthly rain** | KXRAINNYCM (2024-03+), KXRAINCHIM (2025-07+), LAX HOU SEA SFO MIA DAL DEN AUS (2025-11+); ~25 more listed 10-2026 with 0-1 settled | ladder of `greater` strikes 1..10 in, "strictly greater"; CLIxxx station; TWC/NWS | 142 city-months total | NYC $2.8M lifetime, LAX $2.1M, others $0.7-1.5M; currently open LAX $157k |
| **Monthly snow** | NYC DC PHIL CHI BOS DET SEA DENMB SLC LAX DAL HOU SFO AUS MIA (*SNOWM*, 2025-12..2026-03), legacy KXSNOWNYM 2021-25; Dec-2026 ladders open for NYC/DC/PHIL/CHI/BOS/DET/SEA/DEN/MKE/PIT/MSP/TTN/BigSky | ladder 0.1..35 in; TWC (Denver: NWS threaded "Denver Area") | 72 city-months | NYC $6.6M, DC $1.7M, PHIL $1.3M, CHI $1.1M |
| Daily rain (R1 dead) | KXRAIN (multi-city), KXRAINNYC (2021-26) | threshold > 0 | long | NYC $30M |
| Weekly | KXAVGTK* (8 cities, consecutive-days-above-X, 8 weeks Aug-Oct 2026), KXWEEKHT*, KXRAINWKND, KXRAINNYCW | | 8-10 events each, none open now | $12-150k per series |
| Seasonal / one-off | ski-resort snowfall (Alta, Mammoth, Park City... 0 settled), White Christmas, KXSNOWSTORM ($10.6M, 6 events), KXHDDCVG (Cincinnati HDD), KXTORNADO monthly (39 settled), KXHMONTH | custom | too few events to backtest | |
| Hourly temp (KXTEMP*H) | 16 series | | | too thin (N3, earlier) |

I picked monthly rain and monthly snow for testing. They are the only non-daily families with tens of settled events
and real money, and month-to-date accumulation is a hard state variable.

**Data audit:** NOAA ACIS daily obs (keyless), summed over the month, reproduce the Kalshi settlement on **1,259/1,259
settled markets (194/194 city-months)**. SFO-snow and Jackson have no ACIS snowfall and were dropped by the
pre-registered missing-data rule.

## Step 2: rule and results (DEV: rain months < 2026-07, snow months < 2026-03)
Rule (frozen in the docstring): decide at 22:00 local on each of the month's last 7 days. A bracket "total > X" is
unreachable if X minus the observed month-to-date total exceeds 2 x the wettest model's forecast for the rest of the
month, plus 0.5 in (rain) or 1.0 in (snow). Snow is forecast as 10 x the liquid precipitation. Forecasts come from
Open-Meteo previous runs, lead j+1, 5 models, >= 3 required. When a bracket qualifies, buy NO with a tradeable quote
(spread <= 10c). One bet per market (the first day it qualifies), held to settlement.

| trial | bets (city-months) | losses [95% CI] | break-even loss rate | units | ROI | cluster boot p | market-null p (1/city-month) | halves (by date) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| W1 rain NO, taker | 167 (72) | 1 [0, .033] | .022 | +2.74 | +1.64% | .015 | .105 (.34) | +1.2% / +2.1% | fail |
| W2 rain locked YES | 0 | - | - | 0 | - | - | - | - | no trades |
| S1 snow NO, taker | 36 (17) | 0 [0, .097] | .028 | +1.07 | +2.99% | 0* | .35 (.50) | +2.0% / +4.0% | fail |
| S2 snow locked YES | 0 | - | - | 0 | - | - | - | - | no trades |
| **W1m rain NO, maker (E4 M3)** | 146 filled of 167, fill 87% (69) | 1 [0, .038] | .041 | +5.23 | **+3.58%** | .0003 | **.016** (.12) | +3.2% / +4.0% | fail (bar .0083) |
| S1m snow NO, maker | 30 of 36, fill 83% (15) | 0 [0, .116] | .046 | +1.48 | +4.93% | 0* | .24 (.41) | +5.0% / +4.8% | fail |

\*The cluster bootstrap returns p = 0 for any sample with no losses, so it says nothing there. Before the ledger
write I amended the pass rule to need BOTH the bootstrap p and the market-null p below the bar. In the market null,
each bet wins with probability equal to its all-in cost. This is strictly harder than the bootstrap alone. Both
p-values are in the ledger.

- **The one loss is the R1 lesson again.** KXRAINSFOM-25DEC-4: 3.45 in had fallen by 12-26, and all 5 models
  forecast about 0.01 in for Dec 27-31. The market had YES at 9-16c, and 0.94 in fell. The market knew something
  the models did not. Part of the gap is built into the test: previous-runs lead-(j+1) values are older than the
  18Z run that is live at 22:00. A live version could use fresher runs, but the backtest can't measure that.
- **Pricing.** The average NO ask is .976, so each win earns about 2.4c and one loss wipes out about 40 wins. The
  market already prices most unreachable brackets at a $1.00 NO (yes_bid = 0, no trade). Only 22% of rows in the
  window had a two-sided quote.
- **Maker helps here too** (+3.6% vs +1.6%). The one loser filled, the same adverse selection as E4.
- **Spread of results:** W1m is positive in 9/10 rain cities (SFO, which holds the loss, is the exception) and on
  every days-left value except 4 (the loser).
- **Capacity:** median hourly-candle volume in the pick window is about 2.8k contracts (rain) and 6.4k (snow). That suggests
  $1-10 units would fill (no historical depth, so this is unverified). At 1 unit per bet that is about 17 rain bets/month; the edge size, not liquidity, is the
  binding limit.
- **Locked YES:** 1,967 open, quoted rows where the month-to-date total already exceeded X. In none of them did YES
  trade at <= 98c (asks sit at $1.00), so no lag arb.

## Next (cheap, no sealed spend)
1. Forward paper logger, not installed: at 22:00 local on each of the month's last 7 days, use live Open-Meteo runs
   plus ACIS/CLI month-to-date. Log W1m picks for the rain ladders (about 36 monthly-rain series open now) and the Dec-Feb snow
   ladders.
2. After about Feb 2027, ask the user for ONE sealed shot of W1m on rain Jul 2026-Jan 2027 plus snow Mar 2026 and
   Dec 2026-Feb 2027 (rows never loaded by `dev`). Frozen now: W1m as specified, same pass rule.
