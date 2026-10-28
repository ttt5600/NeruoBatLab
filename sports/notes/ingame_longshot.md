# L38 In-game trailing longshots on Kalshi: sell them? (2026-10-27, DEV ONLY)

Script: `scripts/ingame_longshot.py` (build-nhl -> snap -> dev). Its docstring holds the full pre-registration, written
before any outcome was joined. Ledger dataset `ingame_longshot`: 8 trials, bar p < 0.00625. Results in
`results/ingame_longshot_dev.json`. No orders were placed and nothing was committed.

## Verdict
**DEAD in dev. 0 of 8 trials pass, and no trial even has a positive point estimate in both halves with p < .3.**
In the dev windows, Kalshi in-game trailer prices are **not** overpriced by enough to pay for the spread plus the fee.
In the 15-25c bin trailers actually won **more** often than their mid (+1.6c pooled, +3.3c NBA). The ~1.5c longshot
bias seen descriptively in the L37 sealed games (NBA 2026-01-15+, never read here) **does not replicate** in the
NBA 2025-10-21..2026-01-14 dev games. The best cell is the 1-5c bin: trailers were -0.5c below the mid (CI -1.8 to
+0.9), and taker breakeven needs about -0.9c. Taker -0.24c per contract, maker +0.32c (p .33). Both are noise.

## Data (dev only; sealed rows dropped at load)
| sport | dev window | games | snapshots |
|---|---|---|---|
| NBA | 2025-10-21..2026-01-14 | 589 | 2,192 |
| NFL | 2025 REG weeks 1-12 | 177 | 643 |
| NHL | 2025-04-19..2025-06-17 (end of 2024-25 + playoffs) | 80 | 216 |

**NHL deviation from the brief.** The frozen MANIFEST seals NHL 2025-26 (`sealed nhl: 2026`;
notes/nhl_simple_model.md), so I did **not** use "the first 40% of 2025-26". Kalshi KXNHLGAME has markets from
2025-04 onward. NHL dev therefore = the 83 Kalshi games of Apr-Jun 2025, all matched; 80 have a snapshot. It is
small, so the NHL rows are descriptive only.

The snapshot is each game's first in-game 1-min candle (between ESPN/nflverse first and last play wallclock) in which
the trailer's own-market mid falls in the bin. It needs a two-sided quote with spread <= 3c (median spread 1c).
There is one trade per game per bin, held to settlement. Kalshi settlement results are used; there were 0 ties and 0
missing results.

## Break-even (explicit)
Buying the leader at ask `a` costs `a + 0.07a(1-a)`. The leader ask sat at about 1 - mid + half the spread (median
leader ask = 1 - trailer bid exactly). Mean break-even leader win rate vs the needed trailer overpricing:

| bin | mean trailer mid | break-even leader win | trailer must win < | needed overpricing vs mid |
|---|---|---|---|---|
| 1-5c | 3.80 | 96.93% | 3.07% | ~0.7c (half-spread ~0.5c + fee 0.25c) |
| 5-10c | 8.26 | 92.89% | 7.11% | ~1.2c |
| 10-15c | 13.14 | 88.38% | 11.62% | ~1.5c |
| 15-25c | 21.77 | 80.26% | 19.74% | ~2.0c |

## Calibration + taker (pooled over sports; cents per contract; game-bootstrap 95% CI)
| bin | n | mid | realised | real - mid [CI] | taker P&L [CI] | p | H1 / H2 |
|---|---|---|---|---|---|---|---|
| 1-5c | 725 | 3.80 | 3.31 | -0.49 [-1.8, +0.9] | -0.24 [-1.61, +0.99] | .62 | -0.74 / +0.26 |
| 5-10c | 747 | 8.26 | 8.30 | +0.04 [-1.9, +2.1] | -1.19 [-3.20, +0.71] | .88 | -1.16 / -1.22 |
| 10-15c | 762 | 13.14 | 13.39 | +0.24 [-2.1, +2.7] | -1.77 [-4.23, +0.58] | .93 | -1.08 / -2.47 |
| 15-25c | 817 | 21.77 | 23.38 | +1.61 [-1.1, +4.5] | -3.64 [-6.54, -0.81] | .99 | -2.63 / -4.66 |

By sport (real - mid in c; taker P&L in c):
- **NBA:** 1-5c -0.98 / +0.27; 5-10c +0.11 / -1.22; 10-15c +1.17 / -2.62; 15-25c +3.26 / -5.25.
- **NFL:** 1-5c +2.07 / -2.72; 5-10c +0.48 / -1.55; 10-15c -1.54 / +0.08; 15-25c -0.84 / -1.09.
- **NHL** (small n): 1-5c 0 of 38 trailers won (-3.74 / +2.48); all bins pooled -4.0c [-7.5, -0.2] / +1.86c
  (p .16; H2 -0.39). This is the only sport where the direction matches the hypothesis. With 216 snapshots from 80
  playoff-heavy games it is a weak lead, not a result.

Taker sensitivities (pooled, c): buying trailer-NO at 1 - bid instead gives -0.39 / -1.24 / -1.68 / -3.46.
Adding 1-min latency gives -0.13 / -1.58 / -1.91 / -3.61. A rounded 1-lot fee gives -1.01 / -1.70 / -2.01 / -4.34.

## Maker (sell trailer YES at ceil(mid) for 10 min; fill = strict print-through on the public tape)
Tapes were fetched for 100% of snapshots. Fills were easy, at 84-91%. Adverse selection was mild: the trailer win rate
given a fill was 3.8% vs 3.3% overall in 1-5c, and about equal in the other bins.

| bin | fills | P&L c [CI] | p | H1 / H2 | '>=' fill variant |
|---|---|---|---|---|---|
| 1-5c | 610 | +0.32 [-1.28, +1.79] | .33 | -0.47 / +1.11 | +0.79 (719 fills) |
| 5-10c | 642 | -0.61 [-2.90, +1.51] | .71 | -1.29 / +0.06 | +0.02 |
| 10-15c | 672 | -0.52 [-3.24, +2.02] | .65 | +0.46 / -1.50 | -0.24 |
| 15-25c | 740 | -1.55 [-4.59, +1.43] | .84 | -0.12 / -2.91 | -1.73 |

The maker fee is 0.0175p(1-p) (repo convention, conservative). The 10-min print-through rule ignores queue position,
so the real fill rate would be lower and more adversely selected.

## Lessons
- The L37 sealed observation (trailer mids ~1.5c rich in 0-15c) is one window's descriptive number. Dev shows the
  opposite sign in 10-25c. Do not build a forward rule on it.
- Even where a bias exists (1-5c), it is about the size of half the spread plus the fee. Selling a 4c longshot as a
  taker gives up ~0.5c of spread to earn ~0.5c of bias.
- Possible remaining angle (NOT a trial; it would need a fresh pre-registration): NHL low-price trailers. 0/38 won
  in 1-5c, but n is tiny. A forward-only paper logger would be the only clean test, since NHL 2025-26 is sealed.
