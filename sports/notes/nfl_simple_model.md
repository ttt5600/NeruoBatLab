# NFL simple model (team EPA -> + QB -> + style), scored against the OPENER (2026-10-25)

Script: `scripts/nfl_simple_model.py` (`dev [--ledger]` about 10 s; `week [--refresh] [--week N]` for live numbers).
Generated tables with every trial, stacking and coefficient row: `notes/nfl_simple_model_tables.md`.
JSON: `data/lab/nfl_simple_model_results.json`. Ledger: 14 rows, dataset `nfl_simple_model` in `lab/ledger.jsonl`.
Only DEV seasons were scored: features from 2006-2019 pbp, walk-forward test seasons 2010-2019 (2,667 games).
Openers come from the sportsbookreviewsonline (SBR) NFL archive, 2007-2019 pages only, saved to `data/sbr_nfl/`.
The 2020-2025 lockbox and 2026 were not scored. 2025 pbp is used only as the live shrinkage prior in `week`.

## Verdict (plain language)

1. **The simple model knows things the opening line does not.** For every version, when the model disagreed with the
   open by at least 2 points (spread) or 2.5 points (total), the line then moved toward the model by **+0.57 to +0.81
   points** before kickoff. Seven of the 14 pre-registered trials pass the bar (p < .0036, positive in 2010-14 and in
   2015-19). All seven are CLV trials. This is the NBA result again (`player_sim.md`): a simple model leads the opener.
2. **It does not yet make money at -110 at the open.** ROI at the opening line and -110 was +0.4% to +1.1% on spreads
   and +2.1% to +6.6% on totals. None passes. The best is totals model C, +6.6% [+0.8%, +12.5%] with p .014 over 1,017
   bets, positive in both halves, but short of the .0036 bar. Against the close the model adds nothing significant:
   stacking coefficients given the close are +0.04 to +0.17, and every CI includes 0.
3. **Without the vig it would pay** (secondary, not a counted trial). Graded at the open at even money, which is
   roughly what a Kalshi maker order costs, ROI was +5.0% to +5.8% on spreads and **+6.8% to +11.6% on totals**. Totals
   C was +11.6% [+5.5%, +17.7%], positive in both halves. Two caveats: nobody bets the opener at no-vig, and SBR's
   "open" is an undocumented book and time.
4. **QB changes are where the opener is most wrong, and the model's QB term catches it.** In QB-change games the
   model B CLV is **+1.15 points** (210 bets, CI +0.77 to +1.55; T14 passes). In the same games the team-only model A
   gets +0.36, so most of that comes from the QB term. Lines also move further in those games: the mean |close - open|
   is 1.91 points against 1.20 in other games. The opener's excess error in QB-change games (T13) is +0.08 points, CI
   -0.17 to +0.35, which does not pass. So the open does move a lot, but it does not end up measurably less accurate
   at full time.
5. **Control: a noisy opener explains only a small part.** A model that knows nothing (home edge only, or the league
   average total) also shows CLV of +0.14 (spreads) and +0.22 (totals), because some openers are stale or noisy. The real
   model's +0.75 is about 4-5 times that. Corr(model - open, close - open) is 0.35 for model B and 0.06 for the
   constant. The CLV holds with week 1 and playoffs removed (B spread +0.75, C total +0.78). Those games have
   openers posted weeks early.

**Bottom line:** this is a real information edge at the opener and no edge at the close. Whether it is money
depends entirely on getting near-opener prices at low vig. That is the same open question as NBA S3. The NFL's n is
too small to prove the money part on dev alone (see power).

## Playing style: what the coefficients say (fit on all dev 2007-2019; points per 1 SD of the feature)

Spread model B (home margin; R2 0.160, RMSE 13.6; the opener's RMSE on the same games is 13.4, the close's 13.3):

| term | what it is | pts per SD | read |
|---|---|---|---|
| off_pass | offence EPA/dropback, home minus away | **+4.0** | passing offence is the biggest driver of who covers margins |
| def_pass | pass defence EPA allowed, away minus home | +1.9 | pass D counts half as much as pass O (it is noisier week to week) |
| off_run | designed-run EPA, home minus away | +1.7 | running efficiency matters, less than passing |
| def_run | run defence | +1.2 | smallest of the four |
| qb_adj_diff | today's starter minus the QB mix behind the team's stats | **+1.5** | +1.6 pts per 0.1 EPA/db; a backup starting typically moves 3-6 pts |
| home field | constant | +2.3 pts | 2007-19 era; the modern HFA is smaller, so expect live spreads to lean home |

Totals model B: offence pass EPA +3.0/SD, pass D allowed +1.2/SD, QB sum +1.1/SD, run terms ~0.
Model C (style) adds, for **totals**: pass rate over expected **+1.65 pts/SD** (p 1e-9; pass-heavy teams play higher
totals), explosive-play rates +1.2/SD, pace +0.8/SD. For **spreads** style adds ~nothing: PROE +0.2/SD, pressure
+0.3/SD, explosive net -0.7/SD (p .02, with the wrong-looking sign once EPA is held).
**None of the six pre-chosen matchup interactions does anything** (|effect| <= 0.26 pts/SD, every p > .27).
That includes "pass-heavy O vs weak pass D", "pass rush vs leaky line" and "bad D x bad D" for totals. The comeback
study's bad-D unders lead (vs the close) does not appear as a scoring effect here (I5 -0.04/SD, p .88).
In short, style helps totals through *level* (pass-heavy and explosive teams score more) and not through matchups.

## Pre-registered trials (bar p < 0.05/14 = .00357; week-cluster bootstrap)

| trial | n | mean | 95% CI | p | 2010-14 / 2015-19 | pass |
|---|---|---|---|---|---|---|
| T01-T03 spread CLV A/B/C (pts) | 1239-1305 | +0.57 / **+0.76** / +0.66 | all lo > +0.45 | <.001 | all + | YES x3 |
| T04-T06 spread ROI@open -110 A/B/C | 1239-1305 | +0.4% / +1.1% / +0.7% | ~[-4%, +5%] | .32-.42 | mixed | no |
| T07-T09 total CLV A/B/C (pts) | 1017-1098 | +0.66 / +0.75 / **+0.81** | all lo > +0.54 | <.001 | all + | YES x3 |
| T10-T12 total ROI@open -110 A/B/C | 1017-1098 | +2.1% / +3.9% / **+6.6%** | C: [+0.8%, +12.5%] | .23 / .09 / .014 | all + | no |
| T13 opener excess error, QB-change games | 417 | +0.08 pts | [-0.17, +0.35] | .27 | -0.13 / +0.28 | no |
| T14 model B spread CLV in QB-change games | 210 | **+1.15 pts** | [+0.77, +1.55] | <.001 | +0.99 / +1.31 | YES |

Secondary (not counted): stacking of outcome on (model - open) gives spread B +0.35 [+0.15, +0.55] and total C
+0.35 [+0.15, +0.54]. Given the close: spread B +0.15 [-0.03, +0.34], total C +0.17 [-0.02, +0.36].
Blind drift, open to close: home spread -0.18 pts and total -0.14 pts. Model bets are 50% home and 56% overs, so
drift does not explain the CLV (if anything it works against the overs).

## Power (n needed for 80% power at the .0036 bar, at the observed effect)
- CLV: 60-160 bets. That is why CLV passes easily.
- ROI at -110: spread B (+1.1%) needs about 85,000 bets, which will never happen. Totals C (+6.6%) needs about
  2,500, against 1,017 here. That is about 25 NFL seasons at ~100 qualifying bets a season. NFL dev data alone cannot
  prove the money. The 2020-25 lockbox (about 600 more bets) cannot prove it either.

## Proposed forward rule (paper; nothing is staked without the user's yes)
- Model **B** for spreads, model **C** for totals (C was best on totals, but its 2026 totals look biased low by
  regime drift, so check `week` output; fall back to B totals if the mean gap vs line is < -1.5 pts across a week;
  week 7 2026 was -3.1, so **B totals for now**).
- Trigger: |model - OPENER| >= 2.0 pts (spread) or >= 2.5 pts (total). Take the model's side at the first available
  price, ideally a Kalshi maker order at or near the opener. Log the opener, our fill price and the close.
- Primary forward metric: CLV in points (expect about +0.75). Money metric: ROI at the fill price. Kill it if CLV is
  <= 0 after 100 bets.
- QB-change weeks are the highest-value slice (+1.15 pts CLV). Check the starter against the injury report before
  trusting `week`. Its QB comes from the nflverse schedule's projected starter, which can be stale.
- Sealed option (needs the user's explicit yes): score T02 (spread CLV B) and T12 (total ROI C at open) once on
  2020-2025, using SBR 2020-21 and 2021-22 plus ESPN BET `*_open` 2022-25. That would be about 6 seasons.

## Caveats
- **SBR opener provenance is unknown** (book and timestamp undocumented). It agrees with nflverse at the close
  (MAE 0.25 pts, r .997). About 30 games have open-to-close moves above 6 points (week 17 rest decisions, QB news,
  early-posted playoff and week 1 openers). Two games lacked an open. Limits at the open are usually small.
- Spreads are assumed to be -110 both sides at the open (SBR gives no open odds).
- Key-player availability (WR1/LT/edge) is **not built**. The injury and snap data are not local, and they are not
  clean back to 2007.
- Live `week`: QB ratings see only 2025-26 dropbacks (2020-24 pbp is lockbox and was not downloaded), so veterans'
  ratings are noisier than in dev. Coefficients are 2007-19, including HFA +2.3 and the totals level, so treat
  `week` numbers as directional. The first live run (week 7) showed model C totals 2-10 pts under the lines in
  several games (mean gap -3.1 over 14 games), which is regime drift (league pass rates and pace changed). Model B
  totals were unbiased (mean gap +0.3). So use B totals live until C is refit (dev-only refit is not possible for
  2026 levels; this is a known limitation).
- Shrinkage constants (K values, QB prior -0.10, CPOE weight 0.02, decay 0.6) and thresholds were fixed before the
  market join and never tuned. The control/diagnostic runs (constant model, timing splits) were run AFTER the
  14 trials and are not counted.
