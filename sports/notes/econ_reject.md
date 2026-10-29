# Econ-data consensus-reject on Kalshi (2026-10-28) -- DEV only, no orders

## Verdict: NO EDGE. Do not use the sealed split.
The weather mechanism (stale far tails that every forecast rules out, still quoted at 2-5c) does not carry over to
economic-data ladders. The tails that are quoted settle at about the rate their prices imply. They are also too rare
to trade: about 6 qualifying rungs a year across all six series.

- **T1 (k=3, primary): 33 bets over 27 events, 0 losses, ROI +8.6%.** This clears the pre-registered event-bootstrap
  test (p = 0), but that pass means nothing. With 0 losses every resample is positive, so the bootstrap returns p = 0
  for any loss-free sample. Under the right null (each bet wins with probability equal to its price plus fee) p is
  **.08**. Also, the ROI comes almost entirely from 6 mid-priced YES bets (cost .68-.79) on 2021-22 KXJOBLESS. Their
  anchor is FRED's current-vintage initial claims (ICSA), and 2021 seasonal factors were heavily revised, so that
  anchor likely contains information the market did not have. The other **27 true tail bets: +2.7% ROI, 0 losses
  against 0.71 expected at market prices, null p .48.** That is exactly what fair pricing predicts.
- **T2 (k=2): 123 bets, 7 losses, ROI +5.75%, bootstrap CI [-0.2%, +11.4%], p .030 > bar .025 -> FAIL.** Market-price
  null p .053. Restricted to tail-priced bets (cost >= .90) it is +0.9% over 87 bets.
- Bar: p < .05/2 = .025. Both trials are logged to `lab/ledger.jsonl` (dataset `econ_reject`) with `passed: false` and
  a verdict string. The registered-bootstrap outcome is kept in its own field.

## Why (same pattern as CPI/gas, now on 6 more series)
Kalshi econ ladders list strikes around the consensus, about ±2.3 sd of our anchors (10-90% of rung z-scores:
-2.33..+2.34). Of 816 priced and anchored DEV rungs, only 84 sit beyond 3 sd and just 33 of those had a tradeable quote.
Temperature ladders span the whole outcome range and leave stale 3c tails. Econ ladders put their outside rungs at a
1c bid, or don't quote them at all. When a quote does exist, it is fair.

## Setup (pre-registered in `scripts/econ_reject.py` docstring before any price was pulled)
| series | DEV events (close < cut) | anchor | sd (pre-Kalshi window) |
|---|---|---|---|
| KXJOBLESS 2021-22 (single rung) | 48 (< 2022-07-06) | FRED ICSA week S-7 (current vintage) | 9,878 (weekly change 2015-19) |
| KXJOBLESSCLAIMS | 43 (< 2026-06-25) | same | same |
| KXU3 | 44 (< 2025-04-04) | FRED UNRATE M-1 | 0.128 pp |
| KXPAYROLLS | 30 (< 2025-10-03) | previous month's Kalshi first print (leak-free) | 117k |
| KXGDP | 15 (< 2025-04-30) | GDPNow final pre-advance (TrackRecord) | RMSE 0.73 (2011Q3-2019Q4) |
| KXPCECORE | 18 (< 2026-04-09) | Cleveland Fed core-PCE nowcast <= decision-1d | 0.082 pp |

Decision is close minus 24 h, using the last hourly candle. NO costs 1 - yes_bid, YES costs yes_ask, plus the exact taker
fee ceil(0.07·C·p·(1-p)) on a 50-contract order. DEV is the first ~70% of events per series. SEALED was never priced.
Excluded: KXPCEHEAD (1 event), KXCONTCLAIMS (9 events, median volume 25), Bloomberg consensus (not public).

## Post-hoc sensitivities (not trials; `/Users/jonathanwang/.claude/jobs/588c8b26/tmp/econ_sens.py`)
| cut | bets | losses | ROI | market-null p |
|---|---|---|---|---|
| T1 excl 2021-22 KXJOBLESS | 27 | 0 | +2.7% | .48 |
| T1 NO side only | 16 | 0 | +4.1% | .53 |
| T2 NO side only | 55 | 0 | +7.1% | .028 |
| T2 excl 2021-22 KXJOBLESS | 104 | 3 | +3.4% | .08 |
| claims k=3, anchor = prior Kalshi first print instead of FRED | 7 | 0 | +4.1% | .76 |
| claims k=2, same anchor swap | 44 | 2 | +0.0% | .58 |

- The **maker-NO upper bound** in the script output (+30-48%) is an artefact. It prices NO at 1 - yes_ask, and the asks
  on dead rungs are stale (e.g. U3-23SEP T4.1 had ask .90 against bid .01). Ignore it.
- T2 NO-only (55 bets, 0 losses, null p .028) is the only number that looks interesting. It is post hoc, it misses the
  bar anyway, and its profit comes from a few mid-priced NOs (payrolls 24NOV after the strike/hurricane October print;
  claims Jan-26), not from 2-5c tails. At most it is a forward-paper curiosity, not worth a sealed shot.
- Claims anchor: FRED current vintage and the prior week's Kalshi first print differ by up to 10k (median 0). The
  leak-free anchor makes claims weaker, not stronger.

## Caveats
- The GDPNow "right before advance" value is occasionally published on release eve, possibly after the close-24h decision.
- No public consensus history was available, so the anchors are nowcasts or random walks. A better centre would narrow
  the band, but Diercks et al. show Kalshi already beats Bloomberg on the CPI centre.
- Do not reopen this as a family: CPI (N2), gas (N6) and these six series all fail for the same structural reason.
