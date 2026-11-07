# ForecastEx physical-measurement census + FC1/FC2 monthly climate nowcast (2026-11-06)

## Verdict
**edgelab: both FAIL on dev, so both trials are dead and nothing is sealed.**
- **FC1 GTM:** n 2. The only failed check is `d_min_n`; every other check passed.
- **FC2 MACD:** 0 bets.

**Cause:** the structural wall from `forecastex_wx2` again. Once the input that makes a strike unreachable is public, those strikes barely trade, so the daily files have almost no price for the rule to buy at.

## Census: ForecastEx physical/mechanical families
Sources: live `/api/products` catalog (392 products), plus a ticker-only weekly sample of daily files from 2025-06 to 2026-11. Only tickers, file dates and expiries were read from the sample; no prices.

| family | what | cadence / events in files | covered by | status |
|---|---|---|---|---|
| UH* (24 US cities) | daily high, WU | daily | E5 / FX1 | live |
| UL* (24 US cities) | daily low, WU | daily, 2026-02 onward (17 cities from 05) | forecastex_wx2 F1 | FAIL: 77 D-1 trades |
| SH* (8 intl: FPO, DDF, JTT, SSS, SSY, YHC, YUL, YYZ) | daily high in °C, WU | daily, 2026-02 onward | forecastex_wx2 F2 | FAIL: 7 bets |
| DH/DL legacy | daily high/low | Nov 2025 to Feb 2026, delisted | wx2 F3 | 0 bets |
| HRU* (LAX, LGA, SFO, LAS, MIA) | hourly temperature, WU | 2026-09 onward | none | event-day market; daily files can't price it |
| TR* / OA* | monthly city rain / mean temp | TR 2025-07 onward; OA volume 0 | none | thin |
| GTM, USTM | NOAA monthly global / US temperature | ~20 months | **FC1 (GTM)** | dead |
| MACD | Mauna Loa monthly CO2 | ~19 months | **FC2** | dead |
| GT, UST, GTTA, GTTM, ACD, GSL, USDR, AMOCW | annual or multi-year (tickers 12/26, 12/30, 12/35) | ≤ 6 events | none | untestable |
| HCAB, HLF, HCAT4, MHCMA, TROPA, SWTUS | hurricane / storm / tornado counts | seasonal | none | few events |
| GCE, USCE, GCY* | emissions, crop yields | annual / monthly | none | slow |
| EM*, ELGP*, CMUS*, CMCAN, NGP, OP | EIA monthly electricity, gas, oil | ~22 months | none | next candidates (EIA-930 nowcast) |

## FC1 / FC2 (pre-registered and committed before any price pull)
**Rule:** `edgelab/rules/fx_climate.py`. Builder: `scripts/fx_climate.py`. Vault split `fx_climate` by expiry date: dev 2024-08-01 to 2026-03-31, sealed 2026-04-01 to 2026-11-05.

**Prediction (pred):**
- **GTM:** ERA5 Climate Pulse monthly mean anomaly + c.
- **MACD:** mean of Mauna Loa daily values + c.
- c and sd were fit on 2015-01 to 2024-06, which ends before the dev window starts (see `data/fx_climate/inputs.json`):

  | | c | sd |
  |---|---|---|
  | GTM | +0.582 °C | 0.052 °C |
  | MACD | -0.0004 ppm | 0.032 ppm |

**Bets:**
- Buy NO when pred + 3sd ≤ X. Buy YES when pred - 3sd > X.
- Entry is the earliest CT trade-day that starts after the input is public and falls at least 2 days before expiry.
- Price is that day's HIGH trade + $0.01. The files have no bid/ask, so this is a proxy for the ask.

**Data:**
- 414 contracts. 258 have an exact 0/1 settlement and are kept.
- 156 were dropped:
  - 68 had not expired yet.
  - 88 belong to whole months where the contract disappears from the files after its expiry day, leaving only a non-final mark. These are GTM 06/08/09-26 and MACD 01/02/04/07/08-25 and 10/12-25.
  - Which months were lost depends on what was published, not on outcomes.

**Dev funnel:**

| | candidates | picked side ever traded | traded after month end | bets |
|---|---|---|---|---|
| GTM | 71 | 56 | 23 | 2 |
| MACD | 28 | 24 | 7 | 0 |

**Lesson:** the same wall as wx2. The daily files only show trades, and an unreachable strike stops trading once the information that makes it unreachable exists. The only way to measure this edge is a forward logger on the live IBKR bid/ask, the same step E5 took.
