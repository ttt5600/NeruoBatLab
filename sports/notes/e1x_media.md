# E1X-M -- E1 far-bracket pattern on Kalshi weekly MEDIA series (YouTube / Netflix / charts)

PRE-REGISTERED 2026-11-06, before any candle or trade was joined to an outcome. Script `scripts/e1x_media.py`
(`survey` -> `markets` -> `yt_audit` -> `yt_build` -> `yt_candles` -> `yt_trades`), rule `edgelab/rules/media_yt.py`,
spec `edgelab/specs/ytv_m1_maker.yaml` (trial ytv_m1_maker-55fb9db5, family kalshi_media_weekly, 1 trial), vault
dataset `kalshi_ytviews`. Data `data/e1x_media/` (gitignored).

## Step 1 -- data availability (decides which families survive)
| family | Kalshi settled | structure | free point-in-time partial observation? | verdict |
|---|---|---|---|---|
| G-YOUTUBE KXYTVIEWSW | 181 artist-weeks, 2026-08-02.. (2,853 mkts) | "greater" ladder on max daily artist views Mon..Sun | YouTube Charts/Kworb: NO per-day artist history (kworb shows only "yesterday"). BUT rungs `can_close_early`: a rung closes publicly when the weekly max passes it (767/1,492 YES rungs closed before the week ended) -> the still-trading ladder at time t bounds the observed max | SURVIVES (Kalshi-observed bound) |
| G-NETFLIX KXNETFLIXRANKSHOW(+RUNNERUP, GLOBAL) | 95 / 49 / 46 weeks | per-show "#1 on the Tudum chart" | Tudum all-weeks TSV is free (settlement only); the daily partial obs (FlixPatrol) returns 403 to curl; not defeated | DROPPED |
| G-CHARTS KXALBUMEQUIV / KXTOPSONG / KXTOPALBUM / KXBBSINGLE | 91 / 177 / 102 / 8 | units ladder; #1 per title; rank buckets | HDD projections exist only as editorial prose in Wayback captures (no structured numbers); Kworb has current charts only; KXBBSINGLE n=8 | DROPPED |

Consequence of early closes: the "locked YES" state of the E1 pattern is never tradable (Kalshi closes it). Only the
far HIGH side (NO on rungs the remaining days cannot plausibly reach) is left.

Disclosed before registering: audit-type looks at outcomes (early-close hours/weekdays; count of events with numeric
expiration_value 105/181, all-YES 23). No price or P&L was looked at.

## Trial ytv_m1_maker (k = 1 in family)
- T = Sunday 16:00Z of the week (data through ~Fri/Sat published, Sunday's not).
- U = lowest rung listed at T still trading (close > T); early close <= T = public YES lock.
- History: every artist-week ended >= 7 days before T: r = ln(M / U_that_week), M = max numeric expiration_value
  (if none and all rungs YES: top rung, a LOWER bound -- this understates spikes; disclosed). Need >= 20 residuals.
- Reject rung K (listed and trading at T) if K >= U * exp(r_max + 0.02) -> NO.
- Maker: q = floor_cent(min(NO mid, NO ask - 1c)) at the last hourly candle <= T; 0.50 <= q <= 0.99; filled iff a later
  trade has taker_side yes and yes_price > 1 - q (prints THROUGH us), no cancel, held. Fee 0 (quadratic).
- event_start = Monday 00:00Z; CLV close = NO ask at event_start. Cluster = week.
- Split (vault, by week end): DEV 2026-08-01..2026-10-11, SEALED 2026-10-12..2026-11-01, FORWARD 2026-11-08+.
- Pass = edgelab rule 5 (bar p < 0.05/1, both halves > 0, concentration <= 30%, n >= 40, CLV >= 0).
- Taker variant NOT registered (far rungs are expected one-sided like TSA; added only if spreads prove tight -> new trial).
- Power: dev ~10 weeks x ~16 artists; expected fills tens, so a fail at small n is weak evidence.

## AMENDMENT A1 (2026-11-06, after the audit, before any quote/tape join)
Audit gate (weekly max vs every rung's result): 99/105 = 94.3% consistent. All 6 mismatches have a numeric
expiration_value BELOW a rung that settled YES (e.g. FUE26OCT04 20.15M but 23M YES), so the numeric value is not the
weekly max. M := max(numeric value, highest YES rung) -- a hard lower bound. Selection code only; spec/rule unchanged.

## RESULTS (dev scored once 2026-11-06; sealed NOT pulled)
Rejected rungs 197 in 41 artist-weeks (89 in week 10-11); quoted 178; q in [.50,.99] 159, mean q 0.809 (wide,
one-sided books: YES asks sit 10-30c above a 0-1c bid). Tape printed through our NO bid on only 20 rungs ALL-TIME.
edgelab evaluate ytv_m1_maker --split dev: verdict "fail", n 18, net_roi +0.145 [0.055, 0.252], p 2.5e-05, bar 0.05,
half1 +0.055 half2 +0.252, concentration 0.794 (top week 2026-10-11), CLV mean +0.116, reasons
[c_concentration_ok, d_min_n], state dead. Rows inspected: all fills after T and before close; 8 of 18 are one
late-Tuesday YES sweep through the TAT26OCT04 ladder at 0.97-0.98.
Capacity (contracts printed through q after T, all weeks 08-23..11-01): 0 in 7 of 10 weeks; 33 / 85 / 31 contracts
($31 / $69 / $28) in the other three; median week 0 contracts, $0. Even if real, this is < $50/week.
Reading: the point estimate is positive but it is 18 bets in 2 weeks; the harness fails it on n and concentration and
it would not move the fund at this capacity. Do not re-run with tweaks (new trial = higher bar for nothing).
