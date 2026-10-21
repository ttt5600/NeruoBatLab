# Risk + unit sizing (2026-10-20; user: "managed risk on units per expected edges and return")

Rule: size every bet by fractional Kelly on the CONSERVATIVE edge (lower end of the evidence), never on the point
estimate, and cap correlated exposure. Bankroll B = money set aside for this and nothing else.

Kelly for a binary contract bought at cost c (incl. fee) with true win prob p:  f* = (p - c) / (1 - c).
Kelly for decimal odds o with edge e (expected return per $1):                    f* = e / (o - 1).

| strategy (evidence) | conservative edge used | full Kelly | we bet | caps |
|---|---|---|---|---|
| Kalshi temp consensus-reject, orig 7 (sealed +1.45%, CI +0.6..+2.3%) | ROI +0.6% at NO ~0.96 -> p .966 | ~14% | 1/4 Kelly -> ~1% of B per pick (cap by depth) | <= 10% of B open per night (same-night picks share weather) |
| Kalshi temp X1 17 new highs (sealed +0.78%, CI +0.19..+1.34%) | ROI +0.2% | ~5% | ~0.5% of B per pick | inside the 10%/night cap |
| Soccer S12 soft-book vs Pinnacle (sealed CLV +3.63%, ROI +8.7%) | edge 2% (half of CLV), avg odds ~4 | ~0.7% | 1/2 Kelly ~0.3% of B per bet | needs non-CA soft-book accounts; books limit winners |
| Judgment bets, e.g. TX Gov Abbott YES at 0.72 vs poll-implied ~0.75 | p - c = 0.03 (model unvalidated) | ~11% | 1/8 Kelly -> <= 1% of B | one unit max per race; total political <= 5% of B |
| Anything only dev-tested or paper (F1 in-game fade, rain R1, etc.) | 0 | 0 | paper only | -- |

Hard limits (all strategies together): daily loss stop 5% of B; weekly 10% -> pause and review; never add money to
chase losses; a strategy whose live CLV/ROI after 200 bets is below its sealed CI lower bound is turned off.
Executor caps (kalshi_exec.py, ~/.claude/quant-kalshi/.env) must agree with this table: e.g. B = $1,000 ->
KALSHI_MAX_ORDER_USD=10-25, KALSHI_MAX_DAILY_USD=100, KALSHI_MAX_OPEN_USD=100.

Why fractional: the edge estimates are noisy (sealed CIs are wide), bets on the same night/race are correlated, and
full Kelly on an overestimated edge loses money. Quarter Kelly gives up ~44% of growth for ~1/16 of the variance.
