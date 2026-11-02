# E1X-T -- Kalshi weekly TSA throughput (KXTSAW) consensus-reject, taker and maker

PRE-REGISTERED 2026-11-01, committed BEFORE any KXTSAW candle or trade was joined to a forecast or outcome.
Script `scripts/e1x_tsa.py` (`audit` -> `build` -> `candles` -> `trades` -> `score --split dev`). Ledger dataset
`e1x_tsa`. Data `data/e1x/` (gitignored).

## Disclosed before writing
- KXTSAW market list (277 events; 15-22-rung ladders 50k apart since 2024-03, 21 rungs 1.80M-2.80M since 2025).
- Liquidity peek (`e1x_quotecheck.py`, hourly candle quotes, no outcomes) on 6 events at Monday +2 h / Tuesday: far
  rungs are one-sided -- low rungs YES bid .88-.97 / ask 1.00; high rungs YES bid 0 / ask .04-.10. So the taker
  far side is mostly NOT buyable (N7/Q016 structure); a maker inside the spread is the realistic route. Expect T1 ~0 bets.
- TSA daily numbers 2023-01-01..2026-10-29 (tsa.gov/travel/passenger-volumes/<year>, fetched with the WebFetch tool
  because tsa.gov returns 403 to curl; I did not try to defeat the block. The tool transcribes the page, so the data is
  validated against Kalshi settlements by the audit gate below.)
- Quake family E1X-Q (same day) died on adverse selection of maker fills; that is why maker fills here are modelled
  from the trade tape with no cancel.

## Contract
"If weekly average TSA airport screenings are above K for the week ending <Sunday>, according to the TSA, YES."
strike_type greater, 50k spacing. Lists Monday ~10:00 ET (14:00/15:00 UTC), last trade Sunday 23:59 ET.
fee_type quadratic: taker 0.07 p(1-p) unrounded, maker 0.

## Audit gate (must pass before scoring)
A_W = mean of the 7 TSA daily numbers Monday..Sunday of the week ending on the ticker date. On every ladder event,
the settled results must be consistent with A_W (A_W > K for every YES rung, <= K for every NO rung) on >= 95% of
events. If Mon..Sun fails and Sun..Sat passes, use Sun..Sat (window fixed by the audit, not by P&L). Fail -> stop.

## Forecast (frozen, point-in-time)
Decision T = event open_time + 2 h (Monday ~12:00 ET). Data used: daily numbers through the SATURDAY before T (Sunday's
number may not be published by Monday noon).
- R = sum Y(d) / sum Y(d - 364 d) over the 14 days ending that Saturday.
- F_W = R x mean over the 7 days d of week W of Y(d - 364 d).
- Residuals r = ln(A/F) for all earlier weeks W' (same construction) whose week ended <= 8 days before T, starting
  with weeks ending 2024-01-21 (first week with a full 364-day lag in the 2023+ data). Need >= 20 residuals.
- Consensus-reject (E1 analogue: every past forecast error misses by a margin): with r_min, r_max over that set and
  margin m = 0.01, rung K is REJECTED-HIGH if K >= F e^(r_max + m) (bet NO) and REJECTED-LOW if K <= F e^(r_min - m)
  (bet YES on "above K").

## Trials (k = 2), one contract per rejected rung
- T1 TAKER: buy the far side at the candle-close ask at T (NO ask = 1 - yes_bid, YES ask = yes_ask); buyable iff ask
  <= 0.99; cost ask + 0.07 a(1-a).
- T2 MAKER: post 1c inside on the far side at T. High rung: NO bid q = 1 - yes_ask + 0.01 (requires yes_ask <= 0.99 and
  spread >= 2c); filled iff a later trade (before close) has taker_side = yes and yes_price >= 1 - q. Low rung: YES bid
  q = yes_bid + 0.01 (requires yes_bid >= 0.01 and spread >= 2c); filled iff a later trade has taker_side = no and
  yes_price <= q. Cost q, fee 0, held to settlement, no cancel. Also report fill rate and filled vs unfilled loss rate.

## Split, metric, bar
- Test events: ladder events (>= 10 rungs) with week ending 2024-07-07 .. 2026-10-25. DEV = earliest 70% of those
  events by date; SEALED = the rest, scored once only after a DEV pass.
- ROI per $ staked, event-clustered bootstrap 10k (seed 20261103), one-sided p = share of resampled ROI <= 0.
- PASS = p < 0.025 AND both chronological halves of DEV positive. Too-good guard: > +10% on > 200 bets -> inspect
  (decision time vs data publication, week window, LY alignment over holidays).

## Power
Far-side tickets cost ~.93-.99, so ROI per bet is small (E1: +1.45%); at +2% and cost .95, edge_math says ~870 bets.
DEV ~ 80 events x (rejected rungs that also have a quote/fill) -- likely a few hundred maker fills at most -> power is
modest (< 50% at +2%); a fail at small n is weak evidence, a pass would need to be large.

## Kill
T1 and T2 both ROI <= 0 on DEV -> DEAD. Positive but not past the bar -> LEAD (forward logger only).

## AUDIT RESULT + AMENDMENT A1 (2026-11-01, before any price was joined)
Mon..Sun 129/139 = 92.8% consistent (Sun..Sat 58%) -> window = Mon..Sun; the 95% gate as written FAILS.
Diagnosis (printed, `audit` + a one-off check): every one of the 10 mismatches is a single rung adjacent to the value,
|A - K|/A <= 1.25%; where Kalshi shows an expiration_value the median relative gap to my series is 0.02% (largest
-2.0% TSAW-24SEP08, -1.2% TSAW-24MAR03). So the transcription is right and the gaps are first-published vs revised
TSA numbers (Kalshi settles on first publication). Amendment (disclosed goalpost change, made before any P&L exists):
gate = ">= 95% consistent OR every mismatch is one rung and <= 2% from the value" -> PASS. Scoring uses Kalshi's own
results, so revisions cannot leak into P&L; they leak only as small (<~1%) noise in the forecast inputs, which the
consensus margin (>= max past error + 1%) is meant to absorb. Legacy TSAW-* ladders (same contract) are included; rung
K parsed from the ticker where floor_strike is missing.

## RESULTS (scored 2026-11-01; DEV once, then SEALED once on the T2 DEV pass)
Forecast check: rejected rungs settle on the forecast's side 99.27% of the time on DEV (6/827 wrong), 100% on SEALED (368).
| split | trial | bets | events | losses | ROI | 95% CI (event bootstrap) | p | halves | note |
|---|---|---|---|---|---|---|---|---|---|
| DEV 2024-07-07..2025-12/2026-02 | T1 taker | 178 | 30 | 6 | -0.0% | [-1.6, +1.2]% | .47 | +1.3% / -1.9% | FAIL |
| DEV | T2 maker | 89 fills / 681 posted (13%) | 32 | 6 | **+8.36%** | [+1.0, +20.7]% | .008 | +4.2% / +13.4% | PASS; filled loss 6.7% vs unfilled 0% |
| SEALED (last 30% of weeks, through 2026-10-25) | T2 maker | 24 fills / 325 posted (7%) | 9 | 0 | **+6.10%** | [+3.3, +11.8]% | 0.000 | +5.2% / +8.0% | PASS by the frozen bar |
| SEALED | T1 taker | 163 | 17 | 0 | +0.96% | -- | -- | -- | not eligible (failed DEV) |

How much to believe it (read before acting):
- DEV is fragile: KXTSAW-25NOV16 (shutdown / FAA flight-cut week, market bid low rungs at .05-.50) is +3.50 of the +6.40
  units. Without it DEV is +3.9% p .023; restricted to q >= .90 it is +2.8% p .055 (post-hoc sensitivities, not gates).
- SEALED has 0 losses in 24 fills, so the bootstrap is degenerate (p = 0 means "no losing resample", not certainty).
  Fills cluster by week (9 events). Exact 95% upper bound on the loss rate from 0/24 is 11.7%; break-even at mean cost
  .942 is 5.8%. So the sealed half is consistent with the edge and also with no edge. Calling this a weak pass.
- Adverse selection is present (DEV filled 6.7% vs unfilled 0% losses) but the discount (mean cost .861 DEV) covered it.
- Capacity is tiny: ~0.7 fills/week at 1 contract; depth behind the fills is unknown (fill = any qualifying trade).
- After 2026-03 the residual max (+14.7%, a holiday week) pushes the high band above the top rung, so only the low
  (YES) side ever fires now.
Verdict: PASSED SEALED (weak, low capacity) -> forward paper logger `scripts/e1x_tsa_logger.py` (lead deploys), judge
at >= 60 settled paper fills: KILL if ROI <= 0 or any 3 losses in the first 30.
