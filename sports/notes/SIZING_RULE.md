# SIZING RULE -- per-strategy Kelly step-ups (written 2026-11-10, BEFORE applying any change)

**Authority:** user 2026-11-10: "if you have an edge and see in the paper test it's working well (split out what our
return is on just those trades ...) then you can bump up the bet size. You can do kelly sizing." fund.yaml lists
change_size / raise_caps as human gates; this message is the user exercising that gate, scoped by the rule below.
Proposed to user: record this as a standing amendment in fund.yaml (agents cannot edit it).

## Eligibility (all must hold, judged on the strategy's OWN trades only, never pooled)
1. Out-of-sample evidence: sealed lockbox PASS, or >= 100 settled live/forward trades with ROI 90% CI lower bound > 0.
   Dev/backtest numbers never count.
2. Live record (real fills, from exchange fills -- not journals) not significantly negative: ROI 95% CI upper > 0, and
   none of its pre-set kill rules has fired.
3. No open incident / reconciliation gap on that strategy's executor.

## Size
- Edge r = lower bound of the 90% CI of ROI from the eligible out-of-sample sample (sealed if that is the source).
- Per bet at price c: Kelly fraction f* = c*r/(1-c) of bankroll; we use 1/4 Kelly. (Rationale: near-certain contracts
  give huge f*, and one point of win-rate error flips the sign -- see memory feedback_longshot_kelly_stale_model.)
- Same-night / same-event bets are ONE correlated basket: basket exposure <= min(sum of 1/4-Kelly, 20% of bankroll).
- Depth: per order <= 50% of resting depth within 1c (executor KALSHI_DEPTH_FRAC), and only step up where the live tape
  shows volume through our price at >= 2x the new size on most fills.
- Steps: at most 2x per step, at most one step per 14 days, each step needs the live record SINCE the last step not
  significantly negative. Revert one step if the strategy loses >= 10% of bankroll since the step, or a kill fires.
- Not eligible -> stays at minimum units ($1 / 1 contract).

## Applied 2026-11-10
| Strategy | Eligible? | Evidence (own trades) | Action |
|---|---|---|---|
| E4 maker orig7 (kalshi_exec) | YES | sealed maker 4,308 picks ROI +2.22% CI [+1.57,+2.84]; live maker 45 filled, 1 loss, +$2.32 (`kalshi_exec.py status` 11-10); audit maker-only 39 fills ROI +1.72% [-4.33,+6.11] -> not significantly negative; tape through-price >= 2x on 32/39 fills | STEP 1: 6 -> 12 ct/order |
| E4 X1 cities | no | sealed taker +0.78% only; maker trial needs_audit; already paused (KALSHI_COHORTS=orig7) | none |
| B5 NHL late taker | no | live 3-3, -$1.92, ROI -39% [-80,+2.7]; paper 5-4 -25.7% | stays $1 |
| L39 / TSA / DUL | no | 0 live settled | stay at preset stakes |
| E5 ForecastEx | no | adverse selection (stop proposed) | none |
| NFLF | no | disabled | none |

E4 math (r = 0.0157): at c .96/.97/.98 (fill quantiles .10/.25-.50/.75-.90, results/e4_live_audit/q4_tape.csv),
1/4-Kelly per bet = 6.0% / 12.7% / 19.2% of bankroll. Bankroll $384.56 (exchange 11-10). Step 1 per order 12 ct x ~.97
= ~$11.70 = 3.0% of bankroll, well under 1/4 Kelly -> depth and correlation bind, not Kelly. Night basket cap 20% x
$400 = $80 (was 10% = $40; busiest night at 6 ct was ~$41 so 12 ct needs ~$82).

.env changes (backup .env.bak-2026-11-10-preKelly1):
KALSHI_MAX_CONTRACTS 6 -> 12; KALSHI_MAX_ORDER_USD 6.20 -> 12.40; KALSHI_ORIG7_FRAC 0.02 -> 0.031 (12.40/400);
KALSHI_NIGHT_FRAC (default .10) -> 0.20; KALSHI_MAX_DAILY_USD 60 -> 80; KALSHI_MAX_OPEN_USD 80 -> 100.
Unchanged: daily loss stop 5% of B ($20), weekly 10%, depth_frac .5, maker style, X1 paused.
Next earliest step: 2026-11-24, if the live record since 11-10 is not significantly negative.

## Applied 2026-11-12 -- E4 step 2 (USER WAIVED the 14-day spacing: "we could increase past 11/24 unless you really see
## some probability saying no")
Record since step 1 (`kalshi_exec.py status` 11-12 03:10Z): maker 47 filled, filled loss rate 2.2% of 45, P&L +$2.98
(+$0.66 24 h); total live +$2.56; balance $382.72. Not significantly negative.
Size: 12 -> 24 ct/order (= 4x the original 6; tape through-price at 4x on 29/39 fills -> depth ceiling; no step 3 above
24 without new depth evidence). Per order ~$23 = ~6% of bankroll vs 1/4 Kelly ~12% at c .97. Worst plausible night
(2 losses) ~-$47 = 12% of bankroll. Daily loss stop 5% ($20) unchanged: one loss at 24 ct trips it (24 h pause) -- an
intended brake.
.env (backup .env.bak-2026-11-12-preKelly2): MAX_CONTRACTS 24, MAX_ORDER_USD 24.80, ORIG7_FRAC .062, NIGHT_FRAC .30,
MAX_DAILY_USD 120, MAX_OPEN_USD 120. Revert to step 1 if E4 loses >= 10% of bankroll (~$38) since 11-12 or a kill fires.
