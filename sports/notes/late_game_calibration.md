# NBA late-game calibration: does Kalshi misprice close late states? (2026-10-21)

Script: `scripts/nba_late_wp.py` (states -> fit -> kalshi -> dev). Ledger dataset `nba_late_wp`.

## Pre-registration (written 2026-10-21 BEFORE any Kalshi price was joined to a late-game state)

- Model: probit, P(home wins) = Phi(b0 + b1 m/sqrt(tau) + b2 z0 sqrt(tau) + b3 poss/sqrt(tau) + b4 m);
  m = home margin, tau = (period seconds left + 4)/2880, z0 = Phi^-1(ESPN pregame home win prob), poss from a play-type
  heuristic. Fit on all Q4/OT plays of games dated < 2025-04-15 (2021-22 .. 2024-25 regular season).
- Snapshot: each 1-min Kalshi candle in Q4/OT; state = last play <= candle end - 60 s; two-sided, spread <= 5c; price
  = close ask; fee 0.07 p (1-p).
- T1 trail_buy: Q4 <= 6:00 or OT, trailing by 1-8, buy trailer YES if model - ask - fee >= 0.04; first per game, hold.
- T2 any_side: same window, |margin| <= 8, either side, same threshold; first per game, hold.
- DEV = Kalshi games 2025-04-15 .. 2026-01-14; SEALED = 2026-01-15 onward incl. 2026-27 (user's yes only).
- One-sided bootstrap over games; 2 trials -> bar p < 0.025.
- Calibration curve (descriptive): trailing team's Kalshi mid by bin vs realised, game-cluster bootstrap CIs; and
  outcome ~ logit(mid) + logit(model).

## Verdict (short)
**No edge. Kalshi is calibrated in late, close NBA states.** Kalshi's price for the trailing team matches how often
the trailing team actually wins in every bin from 0 to 60c (all within the 95% CIs). Our model, which beats ESPN's own
win probability out of sample, adds **no** information beyond the Kalshi price (coefficient -0.06, CI -0.25..+0.17).
Both pre-registered rules lose: T1 -1.5% (p .55), T2 -5.4% (p .85). The Pacers bet was a fair-priced ~24c ticket that
hit. The Pistons were priced at about 5-6c when down 21 in the 3rd; the historical comeback rate from there is about 5%.
Both prices were fair. Single games prove nothing either way.

## 1. The win-probability table (2021-22 .. 2024-25, 5,191 games < 2025-04-15, ESPN play-by-play)
Possession heuristic check: the inferred team in possession took the next field-goal attempt 98.9% of the time
(7,069 shots, 40 games).

Final coefficients (probit; m = home margin, tau = (period seconds left + 4)/2880, z0 = Phi^-1(ESPN pregame p0)):
b0 -0.021, m/sqrt(tau) 0.0448, z0*sqrt(tau) 1.062, poss/sqrt(tau) 0.0156, m 0.0176.

Out-of-sample validation (fit on 2021-22 + 2022-23, scored on 2023-24 + 2024-25 regular season, 2,549 games):

| subset | plays | model log loss / Brier | ESPN WP log loss / Brier |
|---|---|---|---|
| all Q4/OT | 315,867 | 0.2553 / 0.0809 | 0.2608 / 0.0826 |
| Q4 <= 6:00 or OT | 169,271 | 0.1995 / 0.0632 | 0.2031 / 0.0642 |
| ... and abs(margin) <= 8 | 79,408 | 0.3939 / 0.1280 | 0.4006 / 0.1303 |

OOS calibration (late, abs(margin) <= 8, home-side probability; plays are correlated within games):
0-5%: 0.012 model vs 0.019 actual; 5-10: .075/.074; 10-20: .148/.170; 20-30: .248/.238; 30-40: .349/.356;
40-50: .448/.469; 50-60: .551/.585; 60-70: .650/.678; 70-80: .752/.759; 80-90: .853/.857; 90-95: .925/.946;
95-100: .988/.989. Mostly within ~2 pts. The 50-70% band runs ~3 pts low. The probit tails are too thin at the
extremes (e.g. the model says 0.000 for down 8 with 30 s left; the empirical rate is ~0.4%).

### Empirical P(trailing team wins), Q4, all games (one observation per game per cell)
| deficit | 0:00-0:30 | 0:30-1:00 | 1-2 min | 2-3 | 3-5 | 5-7 | 7-10 | 10-12 |
|---|---|---|---|---|---|---|---|---|
| 1 | .212 | .325 | .323 | .396 | .421 | .404 | .434 | .411 |
| 2 | .131 | .240 | .288 | .331 | .361 | .365 | .423 | .403 |
| 3 | .067 | .137 | .192 | .235 | .294 | .336 | .335 | .352 |
| 4 | .023 | .072 | .131 | .162 | .244 | .287 | .312 | .326 |
| 5 | .007 | .034 | .087 | .133 | .199 | .229 | .271 | .293 |
| 6 | .006 | .016 | .045 | .067 | .151 | .187 | .239 | .244 |
| 8 | .000 | .004 | .013 | .035 | .080 | .127 | .163 | .208 |
| 10 | .000 | .000 | .000 | .016 | .031 | .069 | .098 | .125 |
| 15 | .000 | .000 | .000 | .000 | .006 | .012 | .030 | .046 |
| 20 | .000 | .000 | .000 | .000 | .000 | .000 | .007 | .009 |
Cell sizes: down 3 has 593-1,309 games per cell, down 10 has 431-1,115, down 20 has 152-579.
Pregame strength matters: down 3 with 3-5 min left, the trailer wins .254 if it was the pregame underdog (p0<.40) and
.376 if it was the pregame favourite (p0>.60). Down 10 with 10-12 min left: .060 for an underdog vs .238 for a favourite.
Full tables by strength: `scripts/nba_late_wp.py fit`.

Model lookup, down 3, trailer has the ball / opponent has the ball:
| trailer pregame p0 | 6 min | 4 min | 3 min | 2 min | 1 min | 30 s |
|---|---|---|---|---|---|---|
| .30 | .272/.244 | .260/.226 | .247/.210 | .223/.181 | .170/.123 | .110/.065 |
| .50 | .342/.310 | .315/.278 | .293/.252 | .259/.213 | .192/.140 | .122/.073 |
| .70 | .417/.383 | .375/.335 | .343/.299 | .298/.248 | .215/.160 | .134/.082 |
A 23% price for a team down 3 is about right with about 2-3 min left, or earlier for an underdog without the ball.

Down 20-22 with 7:00-10:00 left in the **3rd quarter** (ad hoc count, games < 2025-04-15): the trailer won 35 of 713
games = **4.9%** (Wilson 95% CI about 3.5-6.8%).

## 2. The user's two games (2026-27 = SEALED period; one-game descriptions only, not scored, no rule touched)
- **Pacers @ Pelicans, 2026-10-21, IND won 123-120** (ESPN 401909839, KXNBAGAME-26OCT21INDNOP-IND). Pregame Kalshi
  IND .55/.56 (ESPN pregame .62). Kalshi asked 23-25c for IND in Q4 at about 9:40-8:20 left, when IND trailed by
  **8, not 3**. Our model said .22-.23 there and ESPN said .18-.19. Kalshi was a little *above* ESPN and level with
  our model. When IND was down 3 (Q4 7:21, and several times in Q3) Kalshi had them at 39-48c. I could not find a
  23c moment at a 3-point deficit in this game on Kalshi. The user may have seen another venue (Polymarket) or
  remembered the margin from another moment.
  **Fair-priced ~24% ticket that hit.** A 24% shot wins about 1 time in 4.
- **Celtics @ Pistons, 2026-10-20, DET won 121-113** (401909088). DET was down 22 at Q3 8:32 and down 21 at 7:26.
  Kalshi DET was 5-6c there and ESPN said 2.1%. History says ~4.9% from that spot. **Kalshi was fair** and ESPN was too
  low. With 9:36 left in Q4, down 9: Kalshi 13-14c, ESPN .099. For an even matchup the model gives .13-.15 for
  down 9 with 9-10 min left.

## 3. Pre-registered Kalshi test (DEV = 2025-04-15 .. 2026-01-14: 2024-25 playoffs + 2025-26 regular season before 01-15)
369 games with late, close states (Q4 <= 6:00 or OT, abs(margin) <= 8, two-sided, spread <= 5c); 11,959
side-snapshots. Latency 60 s, as pre-registered. Sanity: median abs(Kalshi mid - ESPN) 0.062, abs(mid - model) 0.043.

### Kalshi calibration curve (trailing team; first snapshot per game per bin; 95% CI = game bootstrap)
| Kalshi mid bin | games | mean mid | realised [95% CI] | our model | ESPN |
|---|---|---|---|---|---|
| 0-5c | 278 | .034 | .043 [.022, .068] | .094 | .110 |
| 5-10c | 267 | .077 | .071 [.041, .101] | .132 | .147 |
| 10-20c | 292 | .153 | .178 [.137, .226] | .190 | .185 |
| 20-30c | 253 | .250 | .241 [.190, .300] | .258 | .247 |
| 30-40c | 224 | .348 | .335 [.277, .397] | .315 | .313 |
| 40-50c | 174 | .446 | .460 [.385, .534] | .370 | .360 |
| 50-60c | 122 | .543 | .557 [.475, .648] | .394 | .380 |
Kalshi's mid is inside the CI in every bin. In the bins where the two disagree most, Kalshi is the one that is right.
Example: in the 0-10c bins the models say 9-15% and trailers won 4-7%.

Information test, outcome ~ logit(Kalshi mid) + logit(model), every 3rd minute, 4,244 rows / 369 games, game-cluster
bootstrap: Kalshi coefficient **0.99 [0.75, 1.22]**, model coefficient **-0.06 [-0.25, +0.17]**. The price already
holds everything the score/time/possession/strength model knows.
Brier on all late side-snapshots: Kalshi mid .1624, model .1711, ESPN .1735.

### Pre-registered trials (ledger dataset nba_late_wp, 2 trials, bar p < 0.025; one-sided bootstrap over games)
| trial | bets (=games) | units | ROI | 95% CI units | p | win rate vs model vs ask |
|---|---|---|---|---|---|---|
| T1 trail_buy | 311 | -0.91 | -1.47% | [-14.1, +12.6] | .552 | .196 / .295 / .189 |
| T2 any_side | 331 | -8.32 | -5.43% | [-23.4, +7.3] | .852 | .438 / .548 / .451 |
On the bets it picked, the model expected about 10 points more than the price, and the realised win rate sat on the
price. Picking spots where the model and the market disagree selects the spots where the model is wrong.

### Diagnostic (not a trial): how stale is the state?
| state lag | Brier Kalshi | Brier model | median abs(mid - model) |
|---|---|---|---|
| 60 s (pre-registered) | .1624 | .1711 | .043 |
| 20 s | .1637 | .1665 | .032 |
| 0 s | .1654 | .1648 | .028 |
| -30 s (peeks 30 s ahead) | .1670 | .1618 | .038 |
With zero latency the model only ties Kalshi. It beats Kalshi only when it is allowed to see 30 s of the future. This
also confirms the clocks are aligned. So the market prices late states as well as a good statistical table does,
within about one candle. This is the same lesson as the closed in-game H1-H3: the score and clock are public, and the
price already uses them.

## Why the intuition felt right (and the lesson)
Games that swing feel mispriced because you remember the swing, not the ~3 in 4 times a 24% trailer loses. The check
is calibration: across many games, do 24c teams win about 24%? On Kalshi they do. The table does answer "how often do
teams come back from X with Y left", and the market already knows those numbers. Strength context (pregame favourite
vs underdog) moves a 3-point deficit by about 12 points of probability, and Kalshi's pregame price carries that
context into the game.

## Not done / caveats
- Sealed (2026-01-15 onward incl. 2026-27) not scored. There is no reason to ask for it: dev shows no signal.
- Training data starts in 2021-22, the cached seasons (2015-2020 summaries were not pulled; disk is 97% full). Extreme cells (down 20)
  have thin samples. Possession is a heuristic (98.9% agreement). Clock = last play's clock (slightly stale).
- Polymarket was not tested. Kalshi and Polymarket NBA game markets sit within ~0.5c (M2, 10-21), so the result very
  likely carries over, but this is untested.
