# NFL comebacks / matchups -- generated tables (scripts/nfl_comebacks.py)

## A. Comeback base rates (dev 2006-2019, one snapshot per game per checkpoint)

Cell = trailing team's actual win rate (ties 0.5) / mean nflfastR vegas_wp at the same states / n games.

| time left | down 1-3 | down 4-7 | down 8-10 | down 11-14 | down 15-17 | down 18-21 | down 22-28 | down 29+ |
|---|---|---|---|---|---|---|---|---|
| end Q1 (45:00 left) | 39% / 41% / 748 | 33% / 31% / 1436 | 21% / 23% / 303 | 14% / 15% / 367 | 10% / 12% / 39 | 12% / 10% / 34 | 0% / 1% / 10 | - |
| half (30:00) | 42% / 39% / 668 | 30% / 30% / 1071 | 17% / 19% / 492 | 12% / 12% / 558 | 7% / 7% / 242 | 8% / 5% / 243 | 2% / 1% / 145 | 0% / 0% / 40 |
| mid Q3 (22:30) | 42% / 40% / 635 | 31% / 28% / 953 | 18% / 17% / 490 | 11% / 10% / 616 | 6% / 6% / 267 | 4% / 3% / 261 | 0% / 0% / 221 | 0% / 0% / 76 |
| end Q3 (15:00) | 39% / 39% / 604 | 27% / 25% / 889 | 16% / 15% / 468 | 8% / 8% / 574 | 2% / 3% / 293 | 1% / 1% / 266 | 0% / 0% / 278 | 0% / 0% / 147 |
| 10:00 Q4 | 39% / 36% / 598 | 22% / 22% / 855 | 15% / 12% / 459 | 5% / 5% / 554 | 1% / 2% / 271 | 1% / 1% / 309 | 0% / 0% / 288 | 0% / 0% / 204 |
| 5:00 Q4 | 38% / 34% / 590 | 20% / 19% / 865 | 7% / 8% / 434 | 3% / 3% / 500 | 1% / 1% / 281 | 0% / 0% / 321 | 0% / 0% / 321 | 0% / 0% / 229 |
| 2:00 Q4 | 31% / 31% / 566 | 17% / 16% / 888 | 3% / 4% / 389 | 0% / 1% / 466 | 0% / 0% / 286 | 0% / 0% / 347 | 0% / 0% / 336 | 0% / 0% / 253 |

## B. Same, split by the trailing team's PREGAME spread (actual / vegas_wp / n)

| time left | down | trail fav >3 | within 3 | trail dog 3.5-7 | trail dog >7 |
|---|---|---|---|---|---|
| end Q1 (45:00 left) | 4-7 | 56% / 54% / 350 | 34% / 33% / 527 | 21% / 19% / 329 | 11% / 11% / 230 |
| end Q1 (45:00 left) | 8-10 | 41% / 44% / 70 | 20% / 25% / 98 | 16% / 15% / 80 | 4% / 7% / 55 |
| end Q1 (45:00 left) | 11-14 | 36% / 34% / 66 | 13% / 17% / 111 | 11% / 8% / 102 | 3% / 5% / 88 |
| end Q1 (45:00 left) | 15-17 | 33% / 38% / 6 | 20% / 14% / 10 | 0% / 6% / 12 | 0% / 3% / 11 |
| end Q1 (45:00 left) | 18-21 | 60% / 25% / 5 | 9% / 15% / 11 | 0% / 3% / 12 | 0% / 2% / 6 |
| half (30:00) | 4-7 | 47% / 46% / 274 | 29% / 31% / 377 | 22% / 20% / 268 | 14% / 14% / 152 |
| half (30:00) | 8-10 | 32% / 33% / 110 | 18% / 20% / 181 | 10% / 11% / 115 | 5% / 7% / 86 |
| half (30:00) | 11-14 | 31% / 25% / 117 | 12% / 13% / 190 | 4% / 6% / 137 | 2% / 4% / 114 |
| half (30:00) | 15-17 | 19% / 19% / 31 | 8% / 9% / 83 | 3% / 4% / 75 | 4% / 2% / 53 |
| half (30:00) | 18-21 | 25% / 16% / 24 | 11% / 7% / 85 | 4% / 3% / 71 | 2% / 1% / 63 |
| mid Q3 (22:30) | 4-7 | 46% / 43% / 242 | 30% / 30% / 358 | 25% / 20% / 218 | 14% / 13% / 135 |
| mid Q3 (22:30) | 8-10 | 35% / 30% / 102 | 22% / 20% / 163 | 9% / 11% / 140 | 2% / 6% / 85 |
| mid Q3 (22:30) | 11-14 | 27% / 23% / 112 | 12% / 11% / 231 | 4% / 6% / 153 | 3% / 3% / 120 |
| mid Q3 (22:30) | 15-17 | 8% / 14% / 38 | 9% / 7% / 96 | 3% / 3% / 66 | 1% / 2% / 67 |
| mid Q3 (22:30) | 18-21 | 16% / 9% / 32 | 4% / 5% / 73 | 1% / 2% / 83 | 3% / 1% / 73 |
| end Q3 (15:00) | 4-7 | 37% / 36% / 227 | 30% / 26% / 335 | 20% / 19% / 202 | 12% / 12% / 125 |
| end Q3 (15:00) | 8-10 | 30% / 25% / 96 | 18% / 16% / 181 | 12% / 10% / 111 | 1% / 6% / 80 |
| end Q3 (15:00) | 11-14 | 22% / 17% / 100 | 8% / 8% / 204 | 3% / 5% / 169 | 4% / 3% / 101 |
| end Q3 (15:00) | 15-17 | 4% / 7% / 47 | 4% / 4% / 107 | 1% / 2% / 71 | 0% / 1% / 68 |
| end Q3 (15:00) | 18-21 | 0% / 3% / 28 | 1% / 2% / 78 | 1% / 1% / 86 | 0% / 0% / 74 |
| 10:00 Q4 | 4-7 | 30% / 31% / 196 | 25% / 23% / 351 | 17% / 17% / 192 | 11% / 12% / 116 |
| 10:00 Q4 | 8-10 | 31% / 20% / 109 | 14% / 12% / 146 | 9% / 8% / 111 | 4% / 6% / 93 |
| 10:00 Q4 | 11-14 | 10% / 9% / 97 | 6% / 6% / 195 | 3% / 3% / 159 | 1% / 2% / 103 |
| 10:00 Q4 | 15-17 | 3% / 4% / 36 | 0% / 2% / 102 | 0% / 1% / 75 | 0% / 1% / 58 |
| 10:00 Q4 | 18-21 | 0% / 1% / 34 | 1% / 1% / 109 | 1% / 0% / 92 | 0% / 0% / 74 |
| 5:00 Q4 | 4-7 | 29% / 25% / 210 | 21% / 20% / 341 | 15% / 14% / 201 | 10% / 11% / 113 |
| 5:00 Q4 | 8-10 | 15% / 11% / 86 | 9% / 8% / 149 | 5% / 7% / 123 | 0% / 4% / 76 |
| 5:00 Q4 | 11-14 | 6% / 4% / 79 | 2% / 3% / 179 | 4% / 2% / 147 | 2% / 2% / 95 |
| 5:00 Q4 | 15-17 | 5% / 1% / 39 | 1% / 1% / 98 | 0% / 0% / 81 | 0% / 0% / 63 |
| 5:00 Q4 | 18-21 | 0% / 0% / 40 | 0% / 0% / 100 | 0% / 0% / 86 | 0% / 0% / 95 |
| 2:00 Q4 | 4-7 | 24% / 23% / 203 | 17% / 17% / 330 | 13% / 13% / 230 | 11% / 9% / 125 |
| 2:00 Q4 | 8-10 | 5% / 7% / 74 | 4% / 3% / 148 | 1% / 2% / 103 | 0% / 2% / 64 |
| 2:00 Q4 | 11-14 | 2% / 1% / 84 | 0% / 1% / 164 | 0% / 1% / 121 | 0% / 0% / 97 |
| 2:00 Q4 | 15-17 | 0% / 0% / 45 | 0% / 0% / 96 | 0% / 0% / 85 | 0% / 0% / 60 |
| 2:00 Q4 | 18-21 | 0% / 0% / 41 | 0% / 0% / 106 | 0% / 0% / 109 | 0% / 0% / 91 |

## C. Possession matters (2nd half+Q4 checkpoints): trailing team HAS the ball vs NOT

| time left | down | trailing team has ball | opponent has ball |
|---|---|---|---|
| end Q3 (15:00) | 1-3 | 46% / 47% / 298 | 31% / 32% / 306 |
| end Q3 (15:00) | 4-7 | 31% / 31% / 429 | 23% / 20% / 460 |
| end Q3 (15:00) | 8-10 | 20% / 19% / 238 | 12% / 11% / 230 |
| end Q3 (15:00) | 11-14 | 12% / 11% / 297 | 4% / 4% / 277 |
| end Q3 (15:00) | 15-17 | 3% / 4% / 149 | 1% / 2% / 144 |
| 10:00 Q4 | 1-3 | 50% / 46% / 293 | 28% / 27% / 305 |
| 10:00 Q4 | 4-7 | 29% / 29% / 408 | 16% / 15% / 447 |
| 10:00 Q4 | 8-10 | 20% / 16% / 246 | 9% / 7% / 213 |
| 10:00 Q4 | 11-14 | 7% / 7% / 276 | 3% / 3% / 278 |
| 10:00 Q4 | 15-17 | 1% / 3% / 129 | 0% / 1% / 142 |
| 5:00 Q4 | 1-3 | 52% / 47% / 284 | 24% / 22% / 306 |
| 5:00 Q4 | 4-7 | 29% / 28% / 397 | 12% / 11% / 468 |
| 5:00 Q4 | 8-10 | 12% / 12% / 211 | 3% / 4% / 223 |
| 5:00 Q4 | 11-14 | 5% / 5% / 253 | 1% / 1% / 247 |
| 5:00 Q4 | 15-17 | 2% / 1% / 133 | 0% / 0% / 148 |
| 2:00 Q4 | 1-3 | 48% / 46% / 318 | 10% / 12% / 248 |
| 2:00 Q4 | 4-7 | 28% / 26% / 491 | 3% / 4% / 397 |
| 2:00 Q4 | 8-10 | 4% / 6% / 190 | 2% / 1% / 199 |
| 2:00 Q4 | 11-14 | 1% / 1% / 225 | 0% / 0% / 241 |
| 2:00 Q4 | 15-17 | 0% / 0% / 143 | 0% / 0% / 143 |

## D. Calibration of vegas_wp for the trailing team (all checkpoints, dev)

| vegas_wp bin | n states | actual | vegas_wp | 95% CI actual |
|---|---|---|---|---|
| (0.0, 0.02] | 6168 | 0.2% | 0.4% | 0.1-0.3% |
| (0.02, 0.05] | 2377 | 3.3% | 3.4% | 2.6-4.1% |
| (0.05, 0.1] | 2743 | 7.0% | 7.3% | 6.1-8.0% |
| (0.1, 0.2] | 4028 | 15.1% | 14.8% | 14.1-16.3% |
| (0.2, 0.3] | 3133 | 25.9% | 24.8% | 24.4-27.5% |
| (0.3, 0.4] | 2360 | 35.2% | 34.5% | 33.3-37.2% |
| (0.4, 0.5] | 1533 | 49.3% | 44.7% | 46.8-51.8% |

## E. How big a comeback is possible? Winner's largest deficit at any point (3,738 dev games)

| winner trailed by at least | games | share of all games |
|---|---|---|
| 7 | 1136 | 30.39% |
| 10 | 500 | 13.38% |
| 14 | 206 | 5.51% |
| 17 | 74 | 1.98% |
| 21 | 26 | 0.70% |
| 24 | 8 | 0.21% |
| 28 | 1 | 0.03% |

## F. Every hypothesis test (70 counted; positive controls listed but not counted)

coef = effect per 1 SD of the trait (pp of win prob for family 1/3c; points for ATS/total). early/late = same coefficient fit on 2006-12 / 2013-19 (family 1: logit units, sign is what matters).

| family | test | effect | 95% CI | p | BH q.05 | Bonf | n | sign 06-12 / 13-19 | note |
|---|---|---|---|---|---|---|---|---|---|
| 1 comeback vs vegas_wp | EPA-margin gap (trail - lead) vs vegas_wp (spread-aware) | +2.564 pp per 1 SD | [+0.887, +4.241] | 0.00273 | - | - | 2034 | +0.232 / +0.272 (both +) | states 5864 |
| 1 comeback vs vegas_wp | trailing offense EPA/play [T_off_epa] | +1.602 pp per 1 SD | [+0.006, +3.197] | 0.0491 | - | - | 2202 | +0.090 / +0.230 (both +) | states 6363; top-bottom tercile resid +1.5 pp |
| 1 comeback vs vegas_wp | trailing pass rate over expected [T_proe] | -0.238 pp per 1 SD | [-1.717, +1.241] | 0.753 | - | - | 2202 | -0.017 / -0.028 (both -) | states 6363; top-bottom tercile resid +0.7 pp |
| 1 comeback vs vegas_wp | trailing no-huddle (pace) rate [T_nohuddle] | +0.115 pp per 1 SD | [-1.340, +1.569] | 0.877 | - | - | 2202 | +0.057 / +0.008 (both +) | states 6363; top-bottom tercile resid -0.2 pp |
| 1 comeback vs vegas_wp | trailing no-huddle rate when behind in 2H [T_urgency] | -0.434 pp per 1 SD | [-1.953, +1.084] | 0.575 | - | - | 2190 | -0.045 / -0.041 (both -) | states 6327; top-bottom tercile resid -2.1 pp |
| 1 comeback vs vegas_wp | trailing QB (pass EPA/db + CPOE) [T_qb] | +1.518 pp per 1 SD | [-0.052, +3.088] | 0.0581 | - | - | 2202 | +0.097 / +0.208 (both +) | states 6363; top-bottom tercile resid +1.3 pp |
| 1 comeback vs vegas_wp | trailing Q4-when-behind offense EPA [T_off_late_trail_epa] | -0.211 pp per 1 SD | [-1.625, +1.203] | 0.77 | - | - | 2163 | -0.055 / +0.011 (FLIPS) | states 6238; top-bottom tercile resid -0.3 pp |
| 1 comeback vs vegas_wp | trailing pass EPA/dropback [T_off_pass_epa] | +1.913 pp per 1 SD | [+0.336, +3.490] | 0.0174 | - | - | 2202 | +0.136 / +0.245 (both +) | states 6363; top-bottom tercile resid +2.5 pp |
| 1 comeback vs vegas_wp | leader defense pass EPA allowed [L_def_pass_epa] | +1.211 pp per 1 SD | [-0.147, +2.569] | 0.0804 | - | - | 2251 | +0.165 / +0.076 (both +) | states 6500; top-bottom tercile resid +2.0 pp |
| 1 comeback vs vegas_wp | leader defense rush EPA allowed [L_def_rush_epa] | +0.303 pp per 1 SD | [-1.066, +1.672] | 0.664 | - | - | 2251 | +0.097 / -0.045 (FLIPS) | states 6500; top-bottom tercile resid +0.7 pp |
| 1 comeback vs vegas_wp | leader Q4-when-ahead D EPA allowed [L_def_late_lead_epa] | +0.309 pp per 1 SD | [-1.112, +1.730] | 0.67 | - | - | 2092 | +0.141 / -0.067 (FLIPS) | states 6055; top-bottom tercile resid +0.4 pp |
| 1 comeback vs vegas_wp | leader offense EPA/play [L_off_epa] | -0.823 pp per 1 SD | [-2.370, +0.724] | 0.297 | - | - | 2251 | -0.089 / -0.073 (both -) | states 6500; top-bottom tercile resid -3.8 pp |
| 1 comeback vs vegas_wp | trailing coach 4th-down aggressiveness [T_coach_aggr] | -0.220 pp per 1 SD | [-1.444, +1.004] | 0.725 | - | - | 2546 | -0.105 / +0.108 (FLIPS) | states 7320; top-bottom tercile resid -0.6 pp |
| 1 comeback vs vegas_wp | pregame total (vegas_wp ignores it) [total_line] | -0.020 pp per 1 SD | [-1.190, +1.150] | 0.973 | - | - | 3155 | -0.030 / +0.035 (FLIPS) | states 9183; top-bottom tercile resid +0.4 pp |
| 1 comeback vs vegas_wp | POST-HOC: trailing team's pregame spread (fav +) [trail_spread] | +1.538 pp per 1 SD | [+0.247, +2.829] | 0.0195 | - | - | 3155 | +0.127 / +0.177 (both +) | states 9183; chosen after seeing table B |
| 5 team traits vs spread (ATS, home-away) | EPA margin/play | +0.313 pts per 1 SD | [-0.223, +0.849] | 0.252 | - | - | 2416 | +0.099 / +0.578 (both +) |  |
| 5 team traits vs spread (ATS, home-away) | offense EPA/play | +0.557 pts per 1 SD | [+0.020, +1.093] | 0.042 | - | - | 2416 | +0.355 / +0.777 (both +) |  |
| 5 team traits vs spread (ATS, home-away) | defense EPA allowed | +0.157 pts per 1 SD | [-0.369, +0.683] | 0.559 | - | - | 2416 | +0.244 / +0.065 (both +) |  |
| 5 team traits vs spread (ATS, home-away) | pass offense EPA | +0.480 pts per 1 SD | [-0.053, +1.013] | 0.0776 | - | - | 2416 | +0.164 / +0.844 (both +) |  |
| 5 team traits vs spread (ATS, home-away) | rush offense EPA | +0.417 pts per 1 SD | [-0.129, +0.963] | 0.135 | - | - | 2416 | +0.552 / +0.297 (both +) |  |
| 5 team traits vs spread (ATS, home-away) | pass rate over expected | +0.190 pts per 1 SD | [-0.353, +0.733] | 0.494 | - | - | 2416 | +0.233 / +0.142 (both +) |  |
| 5 team traits vs spread (ATS, home-away) | QB composite | +0.485 pts per 1 SD | [-0.047, +1.017] | 0.0738 | - | - | 2416 | +0.213 / +0.816 (both +) |  |
| 5 team traits vs spread (ATS, home-away) | no-huddle rate | +0.080 pts per 1 SD | [-0.476, +0.635] | 0.779 | - | - | 2416 | +0.981 / -0.203 (FLIPS) |  |
| 5 team traits vs spread (ATS, home-away) | turnover margin/game | +0.261 pts per 1 SD | [-0.266, +0.788] | 0.331 | - | - | 2416 | +0.020 / +0.505 (both +) |  |
| 5 team traits vs spread (ATS, home-away) | one-score-game record luck | +0.047 pts per 1 SD | [-0.447, +0.541] | 0.852 | - | - | 2822 | -0.161 / +0.277 (FLIPS) |  |
| 5 team traits vs spread (ATS, home-away) | point margin/game | +0.404 pts per 1 SD | [-0.094, +0.901] | 0.112 | - | - | 2822 | +0.243 / +0.610 (both +) |  |
| 5 team traits vs spread (ATS, home-away) | penalties/game | -0.055 pts per 1 SD | [-0.571, +0.461] | 0.835 | - | - | 2416 | -0.090 / -0.019 (both -) |  |
| 5 team traits vs spread (ATS, home-away) | coach 4th-down aggressiveness | +0.263 pts per 1 SD | [-0.249, +0.774] | 0.314 | - | - | 2745 | +0.313 / +0.192 (both +) |  |
| 5 team traits vs spread (ATS, home-away) | turnover margin | EPA margin | +0.116 pts per 1 SD | [-0.549, +0.782] | 0.731 | - | - | 2416 | -0.068 / +0.281 (FLIPS) |  |
| 5 team traits vs spread (ATS, home-away) | one-score luck | EPA margin | +0.057 pts per 1 SD | [-0.503, +0.618] | 0.841 | - | - | 2416 | -0.156 / +0.276 (FLIPS) |  |
| 5 team traits vs spread (ATS, home-away) | point margin | EPA margin | +0.760 pts per 1 SD | [-0.986, +2.506] | 0.393 | - | - | 2416 | +0.254 / +1.241 (both +) |  |
| 5 team traits vs total | both offenses EPA | -0.074 pts per 1 SD | [-0.636, +0.487] | 0.795 | - | - | 2416 | +0.098 / -0.192 (FLIPS) |  |
| 5 team traits vs total | both defenses EPA allowed | -0.793 pts per 1 SD | [-1.345, -0.241] | 0.00486 | - | - | 2416 | -0.983 / -0.514 (both -) |  |
| 5 team traits vs total | both PROE | +0.145 pts per 1 SD | [-0.414, +0.705] | 0.611 | - | - | 2416 | +0.068 / +0.254 (both +) |  |
| 5 team traits vs total | both no-huddle | -0.106 pts per 1 SD | [-0.676, +0.463] | 0.715 | - | - | 2416 | -1.294 / +0.542 (FLIPS) |  |
| 5 team traits vs total | both plays/game (pace) | -0.070 pts per 1 SD | [-0.629, +0.489] | 0.806 | - | - | 2416 | -0.391 / +0.468 (FLIPS) |  |
| 5 team traits vs total | both penalties/game | -0.015 pts per 1 SD | [-0.570, +0.540] | 0.957 | - | - | 2416 | +0.249 / +0.164 (both +) |  |
| 5 team traits vs total | both coaches 4th-down aggr | +0.082 pts per 1 SD | [-0.417, +0.581] | 0.748 | - | - | 2745 | +0.530 / -0.540 (FLIPS) |  |
| 2 matchup -> ATS | pass-heavy O x pass-weak D | -0.106 pts per 1 SD | [-0.645, +0.434] | 0.701 | - | - | 2416 | +0.085 / -0.336 (FLIPS) |  |
| 2 matchup -> total | pass-heavy O x pass-weak D | -0.361 pts per 1 SD | [-0.917, +0.196] | 0.204 | - | - | 2416 | -0.282 / -0.549 (both -) |  |
| 2 matchup -> ATS | pass-efficient O x pass-weak D | -0.570 pts per 1 SD | [-1.097, -0.043] | 0.034 | - | - | 2416 | -0.377 / -0.777 (both -) |  |
| 2 matchup -> total | pass-efficient O x pass-weak D | -0.159 pts per 1 SD | [-0.722, +0.404] | 0.58 | - | - | 2416 | -0.114 / -0.218 (both -) |  |
| 2 matchup -> ATS | run-heavy O x run-weak D | +0.242 pts per 1 SD | [-0.298, +0.781] | 0.38 | - | - | 2416 | -0.139 / +0.663 (FLIPS) |  |
| 2 matchup -> total | run-heavy O x run-weak D | -0.101 pts per 1 SD | [-0.643, +0.441] | 0.716 | - | - | 2416 | -0.256 / +0.025 (FLIPS) |  |
| 2 matchup -> ATS | run-efficient O x run-weak D | -0.348 pts per 1 SD | [-0.892, +0.197] | 0.211 | - | - | 2416 | -0.422 / -0.292 (both -) |  |
| 2 matchup -> total | run-efficient O x run-weak D | +0.192 pts per 1 SD | [-0.355, +0.740] | 0.491 | - | - | 2416 | +0.470 / -0.154 (FLIPS) |  |
| 2 matchup -> ATS | pressure-sensitive QB x pressure D (2016-19) | +0.003 pts per 1 SD | [-0.907, +0.914] | 0.994 | - | - | 744 | +nan / +0.003 (n/a) |  |
| 2 matchup -> total | pressure-sensitive QB x pressure D (2016-19) | -0.008 pts per 1 SD | [-0.941, +0.925] | 0.987 | - | - | 744 | +nan / -0.008 (n/a) |  |
| 2 matchup -> ATS | blitz-sensitive QB x blitz-heavy D (2016-19) | -0.968 pts per 1 SD | [-1.736, -0.201] | 0.0133 | - | - | 749 | +nan / -0.968 (n/a) |  |
| 2 matchup -> total | blitz-sensitive QB x blitz-heavy D (2016-19) | +0.819 pts per 1 SD | [-0.164, +1.802] | 0.102 | - | - | 749 | +nan / +0.819 (n/a) |  |
| 2 matchup -> ATS | QB better vs man x man-heavy D (2018-19) | +0.882 pts per 1 SD | [-0.158, +1.923] | 0.0966 | - | - | 380 | +nan / +0.882 (n/a) |  |
| 2 matchup -> total | QB better vs man x man-heavy D (2018-19) | -1.415 pts per 1 SD | [-2.803, -0.028] | 0.0456 | - | - | 380 | +nan / -1.415 (n/a) |  |
| 2 matchup grid (side ATS, both sides stacked) | run-lean O vs good pass D | +0.375 pts ATS (side) | [-0.790, +1.540] | 0.529 | - | - | 554 | +0.492 / +0.201 (both +) | stacked home+away rows: games appear twice across cells |
| 2 matchup grid (side ATS, both sides stacked) | run-lean O vs mid pass D | +0.716 pts ATS (side) | [-0.394, +1.826] | 0.206 | - | - | 552 | +1.419 / +0.120 (both +) | stacked home+away rows: games appear twice across cells |
| 2 matchup grid (side ATS, both sides stacked) | run-lean O vs bad pass D | -0.195 pts ATS (side) | [-1.310, +0.921] | 0.732 | - | - | 506 | -0.974 / +0.465 (FLIPS) | stacked home+away rows: games appear twice across cells |
| 2 matchup grid (side ATS, both sides stacked) | mid O vs good pass D | -0.662 pts ATS (side) | [-1.798, +0.474] | 0.254 | - | - | 516 | +0.619 / -2.101 (FLIPS) | stacked home+away rows: games appear twice across cells |
| 2 matchup grid (side ATS, both sides stacked) | mid O vs mid pass D | -1.574 pts ATS (side) | [-2.685, -0.463] | 0.0055 | - | - | 528 | -2.665 / -0.671 (both -) | stacked home+away rows: games appear twice across cells |
| 2 matchup grid (side ATS, both sides stacked) | mid O vs bad pass D | -0.173 pts ATS (side) | [-1.262, +0.915] | 0.755 | - | - | 566 | +0.087 / -0.329 (FLIPS) | stacked home+away rows: games appear twice across cells |
| 2 matchup grid (side ATS, both sides stacked) | pass-heavy O vs good pass D | +0.804 pts ATS (side) | [-0.302, +1.909] | 0.154 | - | - | 542 | +0.326 / +1.357 (both +) | stacked home+away rows: games appear twice across cells |
| 2 matchup grid (side ATS, both sides stacked) | pass-heavy O vs mid pass D | +1.020 pts ATS (side) | [-0.097, +2.137] | 0.0736 | - | - | 530 | +0.468 / +1.505 (both +) | stacked home+away rows: games appear twice across cells |
| 2 matchup grid (side ATS, both sides stacked) | pass-heavy O vs bad pass D | -0.389 pts ATS (side) | [-1.603, +0.824] | 0.529 | - | - | 538 | -0.228 / -0.530 (both -) | stacked home+away rows: games appear twice across cells |
| 4 referee | ref flags/game -> total resid | -0.399 pts per 1 SD | [-0.843, +0.044] | 0.0775 | - | - | 3411 | -0.584 / +0.136 (FLIPS) |  |
| 4 referee | ref DPI/game -> total resid | -0.371 pts per 1 SD | [-0.816, +0.074] | 0.102 | - | - | 3411 | -0.189 / -0.127 (both -) |  |
| 4 referee | ref holding/game -> total resid | -0.365 pts per 1 SD | [-0.813, +0.083] | 0.11 | - | - | 3411 | -0.303 / -0.010 (both -) |  |
| 4 referee | ref home-minus-away penalty yds -> home ATS | +0.025 pts per 1 SD | [-0.416, +0.466] | 0.911 | - | - | 3411 | -0.169 / +0.168 (FLIPS) |  |
| 4 referee | ref flags/game -> home ATS | +0.093 pts per 1 SD | [-0.350, +0.535] | 0.682 | - | - | 3411 | +0.225 / -0.083 (FLIPS) |  |
| 3 coaching | coach trailing-at-half resid, odd vs even seasons (persistence r) | +0.063 Pearson r across coaches | [-0.277, +0.403] | 0.719 | - | - | 35 | +nan / +nan (n/a) | coaches with >=15 trailing halves in both parities |
| 3 coaching | coach ATS, odd vs even seasons (persistence r) | +0.208 Pearson r across coaches | [-0.075, +0.491] | 0.165 | - | - | 46 | +nan / +nan (n/a) | >=24 games in both parities |
| 3 coaching | trailing coach 4th-down aggr -> trailing-at-half resid vs vegas_wp | -0.228 pp per 1 SD | [-1.667, +1.211] | 0.756 | - | - | 2779 | -0.671 / +0.433 (FLIPS) |  |
| 1b Kalshi dev check (2025 wk1-12) | Kalshi minus vegas_wp for trailing team ~ trailing pregame spread | +0.737 pp per 1 SD | [+0.099, +1.374] | 0.0236 | - | - | 168 | +nan / +nan (n/a) | states 587; + means Kalshi already gives trailing favourites more than vegas_wp |
| 1 comeback vs vegas_wp | CONTROL: EPA-margin gap (trail - lead) vs nflfastR wp (no spread) | +6.947 pp per 1 SD | [+5.306, +8.588] | 1.08e-16 | - | - | 2034 | +0.614 / +0.718 (both +) | states 5864 |
| 5 team traits vs spread (ATS, home-away) | CONTROL: EPA margin -> RAW home margin (no line) | +5.277 pts per 1 SD | [+4.719, +5.836] | 1.42e-76 | - | - | 2416 | +4.919 / +5.715 (both +) |  |
| 4 referee | POSITIVE CONTROL: ref prior flags/game -> flags this game | +0.685 flags per 1 SD | [+0.551, +0.820] | 1.9e-23 | - | - | 3411 | +0.637 / +0.401 (both +) |  |
