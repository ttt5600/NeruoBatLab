# L26: late-day lock-in from observed station highs (DEV only), 2026-10-25

## Verdict: DEAD as a taker strategy. The market does not lag the observed max; it is AHEAD of it.
- **BELOW (NO on brackets entirely below the 17:00 observed max):** at 17:00 local, 5,520 of 5,529 such brackets
  (99.8%) already had yes_bid = 0, i.e. NO cost the full $1.00 and there was nothing to buy. Only 6 taker bets in
  ~4 years of dev. Their one loss was the one bracket where the market disagreed with the METAR, and the market was right (below).
- **ABOVE (NO on brackets >= 3F above the 17:00 max):** 460 taker bets, 53 losses (11.5%, 95% CI 8.8-14.8%) vs a
  break-even loss rate of 10.5%: -38.3 units, ROI -8.3%, p 1.00, both halves negative. **FAIL.**
  Many losers had NO at 1c (YES at 99c). The market already knew the high was above every hourly METAR. Traders
  watch 1-/5-minute ASOS data, which catches peaks that the hourly METAR misses.
- The 3 MAKER trials are **NOT SCORED: data incomplete**. The tape pull was stopped at 2,076 of 11,176 tickers on the
  lead's instruction. They stay in the trial count, so the bar is still p < .0083.
- Settlement risk (the whole risk on a 99c bet) is real but small. The official max came in BELOW a temperature the
  station had already reported on **4 of 5,365 dev station-days (0.075%)**. That is 1 of 5,529 BELOW brackets
  (0.018%, CP 95% CI up to 0.10%). All 4 cases were bad or flagged observations, not a slow market.

## Pre-registration (scripts/wx_lockin.py docstring, written before any price/tape/obs pull)
- 24 paper cities, using the CLI station from the contract rules.
- Dev only: orig7 2021-08..2025-06 (kalshi_temp dev quarters), X1 < 2026-03-01. Sealed rows are dropped on load and
  their observations were never requested.
- Decision T = 17:00 local clock time on the event day. Price = close of the last hourly candle ending at or before T.
- M = max whole-F temperature over METARs (IEM routine + specials) from 00:00 local STANDARD time to T-10 min.
  The T-group (tenths of a degree C) is converted exactly. Without a T-group the conversion is conservative-low. At least 12 obs are required.
- Rules: BELOW_m = top YES value <= M-1-m (m = 0, 1). ABOVE_2 = bottom YES value >= M+3.
- Taker = NO ask (1 - yes_bid), needs yes_bid > 0 and spread <= 10c, fee `live.kalshi_fee` unrounded (cent-rounded
  fee as a sensitivity row).
- Maker = post-only NO bid at 1 - yes_ask, filled iff >= 10 contracts print at or below it after T (the E4 N=10 proxy).
- Truth = Kalshi `result`. The bootstrap resamples whole dates. PASS needs p < .05/6, units > 0, and both halves > 0.
- Ledger: 6 rows, dataset `wx_lockin`. The 3 taker rows are `ok`; the 3 maker rows are `not_scored_data_incomplete`.
- **Amendment (dated, before scoring, rules unchanged):** tapes come from the HTTP cache only. Maker trials are not scored, and tape-based numbers are descriptive.

## Results (dev: 5,365 station-days with >= 12 obs; 11,246 candidate brackets with a 17:00 quote)
| trial | bets | losses | loss rate [95% CI] | break-even | avg entry | units | ROI | p | halves | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| BELOW_0_taker | 6 | 1 | 16.7% [0.4, 64] | 12.0% | 0.877 | -0.91 | -15.2% | .66 | -0.98 / +0.07 | fail (no volume) |
| BELOW_1_taker | 2 | 1 | 50% | 33% | 0.660 | -0.97 | -48.6% | .76 | -1.00 / +0.03 | fail (no volume) |
| ABOVE_2_taker | 460 | 53 | 11.5% [8.8, 14.8] | 10.5% | 0.893 | -38.31 | -8.3% | 1.00 | -23.4 / -14.9 | **FAIL** |
| BELOW_0/1_maker, ABOVE_2_maker | -- | -- | -- | -- | -- | -- | -- | -- | -- | NOT SCORED (tapes 2,076/11,176) |

The cent-rounded fee sensitivity makes every taker row worse (ABOVE_2 -41.9 units).

**Descriptive only (tape subset = 2,076 E1 tickers, 2023-05..2025-06, not a trial):**
- BELOW_0 maker at 99c: 275 of 996 eligible filled, 0 losses, +2.83 units (~1c per contract).
- ABOVE_2 maker: 175 filled, 12 losses, +102.75 units. One 1c fill that settled NO alone is +99 units, so that number is
  noise, not a signal.
- Capacity proxy (contracts printed at or below the entry after 17:00, all brackets, NOT our queue share):
  ~2,600 contracts/day at 99c for BELOW, ~2,400/day for ABOVE. Historical displayed depth is not available publicly.

## Settlement-definition audit (diag; no prices involved)
Official max minus the full-day METAR max: 0F on 2,221 days, +1F on 2,578, +2F on 487, +3 to +6F on 75, below 0 on 4.
The hourly METAR almost always UNDER-states the official max. The CLI uses higher-frequency data, so a METAR max is a
lower bound in 99.9% of days. The same fact is why ABOVE loses: the market sees the peak that the hourly obs miss.

The 4 days the official max came in below the METAR max:
| day | METAR max | official | cause (from the raw METAR) |
|---|---|---|---|
| KNYC 2024-12-29 | 68 | 60 | single 17:51Z jump of +4C in an hour (sensor spike, QC'd out of the CLI). **Kalshi YES was 65c at 17:00: the market knew.** |
| KNYC 2025-06-02 | 70 | 66 | obs carried the `$` maintenance flag |
| KMDW 2021-10-01 | 86 | 84 | not explained by the METAR text (T03000144 = 86F); possibly a CLI or legacy-settlement difference |
| KNYC 2022-09-16 | 76 | 75 | corrupted METAR: another station's report (KNFE, 24.4C) was concatenated into the KNYC text |

Other definitional traps, measured:
- **Climate day is local STANDARD time.** Starting the window at local midnight instead adds 19 more "official < M"
  days. Every evening-before warm reading at 00:00-00:59 LDT belongs to the previous climate day.
- Kalshi `result` never disagreed with `expiration_value` on these 26,559 dev brackets.
- Obs without a T-group (1-5% per station) never set M on a mismatch day.
- Late warming after 17:00: on ~1.9% of days the post-17:00 METAR max beats the 17:00 max by >= 3F (cold-season warm
  advection, up to +17F). So an "above" rule needs a forecast or a later decision time, not just obs.

## What would be worth testing next (each needs its own pre-registration; NOT recommended now)
1. BELOW as a MAKER at 99c with guards (skip `$`-flagged or garbled obs; require M confirmed by 2 obs or by the 6-hour
   max group).
   - Expected edge is ~1c per contract minus a ~0.02-0.1% loss rate, at 99c. One loss wipes out ~99 fills.
   - Queue priority at 99c is unknown and probably deep. Score it only with complete tapes (`build` resumes from the
     cache), and it would still be a tiny-margin, tail-risk trade.
2. Use 1-/5-minute ASOS data (what the market uses) instead of hourly METARs. Even then, the BELOW evidence says the
   book is already at $1.00 by 17:00, so the lag, if any, is minutes, not hours.

Files:
- Script: scripts/wx_lockin.py (`obs`, `diag`, `build`, `tapes-cached`, `dev`).
- Data: data/wx_lockin_{obs,days,quotes,trades,tape_tickers}.parquet.
- Ledger: lab/ledger.jsonl, dataset `wx_lockin`.
