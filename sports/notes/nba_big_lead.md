# NBA big leads: do teams up 20 win more than expected and priced? (2026-10-22)

Script: `scripts/nba_big_lead.py` (pbp -> empirical -> dev). Ledger dataset `nba_big_lead` (3 trials, bar p < 0.0167).
Data: ESPN play-by-play for 14,170 games, 2015-16 .. 2025-26 (`data/nba_big_lead_plays.parquet` + `_meta`); Kalshi
KXNBAGAME 1-min candles for the 86 games of the 2025 play-in + playoffs (`data/nba_late_candles_2025po.parquet`).
Calibration JSON: `data/nba_big_lead_calib.json`.

## Verdict (plain language)
**The first half of the hypothesis is true. The second half is not supported.** Teams up 20 do win a lot: 94% from
the moment they first go up 20 in the 2nd or 3rd quarter, 97% when up 20+ at halftime, 99.7% when up 20+ after
three quarters. But Kalshi already charges about that much. In the only Kalshi games we may score (2025 playoffs),
big favourites actually won **less** often than their price (realised 89.3% vs mid 93.9%, CI -11.2 to +1.3 points).
ESPN's public win-probability model is also too **high** on big favourites, not too low (it says 91-97%, they win
87-95%). All three pre-registered bets fail. Late in the game the ask is often $1.00, so there is nothing to buy.

Why it feels true: blown 20-point leads are rare (about 1 in 20 games where someone leads by 20) and memorable,
so "they almost always win" sounds like an edge. But at a 95c price, a win pays 5c and one loss costs 95c. You need
the leader to win more than about 95.3% (price + fee) just to break even. The market only has to be right to within
a point or two, and it is.

## 1. How often does the leader win? (one row per game per checkpoint; Wilson 95% CI; all 11 seasons)
| checkpoint | lead >= 15 | lead >= 20 | lead >= 25 |
|---|---|---|---|
| end Q1 | .874 [.854, .892] n 1,177 | .926 [.889, .951] n 283 | .983 n 57 |
| halftime | .921 [.911, .931] n 2,828 | .968 [.956, .976] n 1,211 | .985 [.970, .993] n 470 |
| end Q3 | .983 [.979, .987] n 4,161 | .997 [.994, .999] n 2,282 | .998 n 1,149 |
| Q4, 6:00 left | .996 [.994, .998] n 4,563 | .9996 n 2,650 (1 loss) | 1.000 n 1,424 |

Seasons <= 2023-24 alone give the same numbers to within half a point. Pregame strength matters a lot, and the
market knows it. Lead >= 15 at halftime: the leader wins .974 if it was the pregame favourite by 5+, .894 near pick'em,
and .750 if it was the underdog by 5+.

The rate at each rule's own trigger (first time the condition holds):
| event | all seasons | <=2023-24 reg | <=2023-24 playoffs | 2024-25 reg | 2024-25 playoffs |
|---|---|---|---|---|---|
| first lead >= 20 in Q2-Q3 | .941 (n 4,482) | .939 | .950 | .942 | .854 (n 41) |
| first lead >= 20 in Q4 > 6 min | .995 (n 3,934) | .996 | 1.000 | .990 | 1.000 (n 32) |
| lead >= 15 at halftime | .921 (n 2,825) | .920 | .906 | .927 | .920 (n 25) |

The 2025 playoffs were unusually bad for Q2-Q3 20-point leaders (6 lost of 41). That is the same handful of games that
sinks B1 below.

## 2. Comeback (blown-lead) rate by season: a team led by >= L at ANY point and lost
| season | 15+ | 20+ | 25+ |
|---|---|---|---|
| 2015-16 | .103 | .030 (15/492) | .004 |
| 2016-17 | .116 | .036 | .014 |
| 2017-18 | .124 | .056 | .015 |
| 2018-19 | .137 | .054 | .026 |
| 2019-20 | .135 | .051 | .017 |
| 2020-21 | .116 | .040 | .014 |
| 2021-22 | .125 | .054 | .020 |
| 2022-23 | .140 | .058 | .015 |
| 2023-24 | .128 | .064 (40/627) | .011 |
| 2024-25 | .121 | .054 | .022 |
| 2025-26 | .116 | .036 | .008 |
Pooled: 15+ 12.4%, **20+ 4.9% [4.4, 5.5]** (298/6,107), 25+ 1.5%. 2015-19 vs 2021-26 at 20+: 4.4% vs 5.3% (z 1.37,
p .17). Any 3-point-era rise in comebacks is small and not statistically clear. 2025-26 was the lowest since 2016-17.
Twenty-point leads have also become more common (492 games in 2015-16, 638 in 2025-26).

## 3. Market check
### Pre-registration (in the script docstring, written before any Kalshi price was joined)
Buy the leader's YES at the close ask of the first 1-min candle that starts >= 60 s after the pbp event. Fee =
ceil_to_cent(0.07 x 100 x P x (1-P)) on 100 contracts. Hold to settlement. B1 = first lead >= 20 in Q2-Q3; B2 = first
lead >= 20 in Q4 with > 6:00 left; B3 = lead >= 15 at halftime. One-sided bootstrap over games, bar 0.05/3.
DEV = 2024-25 season games (on Kalshi that means only the 86 play-in + playoff games). SEALED = 2025-26 and later,
not scored.

### Results (DEV)
| trial | bets | won | mean ask | break-even | units | ROI | 95% CI units | boot p | exact p | ask $1 = no bet |
|---|---|---|---|---|---|---|---|---|---|---|
| B1 lead20 Q2-Q3 | 37 | .838 (6 lost) | .914 | .919 | -3.02 | -8.9% | [-7.47, +0.89] | .930 | .978 | 2 |
| B2 lead20 Q4 > 6 min | 15 | 1.000 | .976 | .978 | +0.34 | +2.3% | [+0.20, +0.52] | .000* | .710 | **14** |
| B3 lead15 half | 23 | .913 (2 lost) | .951 | .954 | -0.94 | -4.3% | [-3.90, +1.13] | .720 | .916 | 2 |
*B2's bootstrap p = 0 is not evidence. With zero losses, every resample is profitable. If every bet won at exactly its
break-even probability, 15 out of 15 wins would still happen 71% of the time. This exact check was added **after**
seeing B2 (flagged in the docstring). B2 also covers only half its events: in 14 of 29 games the leader's ask was
$1.00 (bid ~99c), so nobody would sell. **No trial passes.**

### Calibration of big favourites on Kalshi (DEV, favourite's mid, first in-game candle per game per bucket)
| bucket | games | mean mid | mean ask | break-even | realised [95% CI] | losses |
|---|---|---|---|---|---|---|
| 90-93c | 75 | .912 | .923 | .928 | .853 [.773, .933] | 11 |
| 93-96c | 75 | .940 | .950 | .953 | .920 [.853, .973] | 6 |
| 96-99c | 74 | .965 | .975 | .977 | .905 [.838, .973] | 7 |
| pooled | 224 / 84 games | .939 | | | .893; realised - mid -4.6 pts [-11.2, +1.3] | 24 |
All three buckets come in **below** the price, which is the opposite of the hypothesis. But the CIs are 13-16 points wide,
so this cannot measure calibration to within 1 point. Resolving a 1-point edge at 95c takes about 3,700 independent bets
(80% power).

### Historical rate vs Kalshi ask at the same events (descriptive, post hoc, NOT strength-adjusted)
For each DEV bet, the matching rate is the win rate of first-trigger events in 2015-16 .. 2023-24 in the same period
with the clock within ±2 min (B3: lead within ±2):
B1 history .932 vs break-even .920 (+1.3 pts); B2 .996 vs .978 (+1.9 pts, on the buyable half only); B3 .938 vs
.954 (-1.6 pts). These gaps are small, they point in mixed directions, and they ignore team strength, which the price
includes. Realised DEV results went the other way for B1 (.838). This is not a lead worth a sealed ask: B1 failed dev
at p .93.

### Public model (ESPN WP, 2017-18 .. 2025-26, favourite in Q2-Q4, first play per game per bucket)
90-93: ESPN .910, realised .875 [.868, .881] (n 10,959); 93-96: .939 vs .908; 96-99: .966 vs .948. ESPN is
**overconfident** on big favourites by 2-3.5 points. Our own score-only table and Kalshi both sit closer to the truth
than ESPN.

## Caveats
- Kalshi DEV is only 86 playoff games (Kalshi listed no NBA regular season before 2025-04). The 2025-26 Kalshi season
  (about 1,200 games) is SEALED for this question, so the market half of the test is small. Note that 2025-26 before
  01-15 was already used as dev by the earlier nba_late_wp / nba_ingame tests. Those tests had different rules.
- Empirical tables use 2025-26 scores (pbp only, no prices). That is descriptive, and no trial touches that season.
- ESPN sequenceNumber is not chronological in some games. Plays are ordered by game clock (bug fixed before the
  ledger run). Two truncated pbp feeds: outcomes come from the official final score.
- Entry lag median ~150 s after the event (rule: candle starts >= 60 s after). A faster bettor would see a slightly
  different price, but the earlier test showed Kalshi reprices within about a minute.

## L28 sealed pre-registration (2026-10-26; user: "yes one shot tests")
Script: `scripts/nba_l28_sealed.py` (full pre-registration in its docstring; sha256 at writing 2c1d536f54ccc0a7...). Not yet run.
- **Rule (= forward logger L28):** state = last ESPN play with wallclock <= candle end - 60 s. Trigger = the market's team
  leads by >= 25 in Q3, or by >= 20 in Q4 with > 6:00 left (no OT). Buy leader YES at the candle close ask if bid > 0,
  ask < 1, spread <= 5c and ask <= 0.96. First such candle per game, 1 contract, hold. Fee 0.07 p(1-p) exact; the
  rounded-up 1c fee is a sensitivity check. Settled on the Kalshi result, cross-checked against the ESPN final.
- **Sealed window:** data/nba_ingame.parquet games dated **>= 2026-01-15** only (713 games: 626 regular, 6 play-in,
  81 playoffs). The 590 games dated 2025-10-21 .. 2026-01-14 are left out because nba_ingame.py and nba_late_wp.py
  already used them as DEV.
- **Pass rule:** ROI > 0, game bootstrap p < .05 (10k, seed 0, one-sided), AND exact Poisson-binomial p < .05 against
  each bet's break-even. Without the exact check, the bootstrap passes any sample with zero losses.
- **Secondary:** win rate vs mean break-even. Trigger games with no trade, split into ask > 0.96 and no tight quote.
  Splits by arm (Q3 / Q4) and by season type. Every loss is listed.
- **Power:** expect about 50 bets (713 x 0.329 trigger share x 0.214 buyable share, all from dev). Treat this number as
  rough. At that n, only a season with **zero losses** can pass, and only if mean break-even <= ~0.94. Power is
  5-22% for a true edge of 0-3 pts. A FAIL is weak evidence.
- **Known leak:** the 2025-26 comeback rates computed from scores alone (20+ blown 3.6%, 25+ 0.8%) were already shown
  above, so we already know this was a low-comeback season. That favours a pass. Prices were never joined.
- **Dev check (2025 play-in + playoffs, 86 games):** 28 trigger games, 6 bets, 5-1. The loss was MEM, up 26 at the
  start of Q3 at 89c, against OKC (the real 29-point comeback). Units -0.645, ROI -11.4%, boot p .67, exact p .96 ->
  FAIL. The other 22 trigger games had no trade, all because the ask was > 0.96 (first-trigger ask 0.97-1.00). Those
  leaders won 22/22. Kalshi result agreed with ESPN in 6/6. Median entry lag 73 s.
- Run: `../.venv_quant/bin/python scripts/nba_l28_sealed.py sealed --confirm-lockbox` (from sports/). It aborts if
  results/nba_l28_sealed.json or a lockbox row nba_L28_biglead exists.
