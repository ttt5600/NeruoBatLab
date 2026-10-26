# CFB simple model vs the OPENER (2026-10-25) -- DEV ONLY

## Verdict

1. **Totals: the NBA/NFL/NHL pattern holds again.** When the simple points model disagreed with the opening total by
   at least 3.5 points, the total then moved toward the model by **+1.02 points** before close (3,014 bets, CI +0.87 to
   +1.17, positive in both halves). Bet at -110 at the open, ROI was **+2.6% [-1.3%, +6.8%], p .10. That FAILS** the
   bar (p < .00357). At even money (no vig) it was **+7.4% [+3.3%, +11.8%], p .0004**, positive in both halves (+4.5%,
   +10.2%), and that passes. Same as NBA/NFL/NHL: the model leads the opener, and the vig takes the money.
2. **Spreads: FAIL, and the cause is the model.** The pre-registered margin model is compressed. Its predictions vs
   the open have slope 0.58, and the outcome vs its predictions has slope 1.59, so it under-rates favourites
   at every spread size. The 3-point rule took 6,334 bets (58% of games), 96% of them on the dog. Results: CLV -0.12
   points, ROI -7.8% at -110. Cause: the +-28 blowout cap plus ridge shrinkage (lambda 4, prior 0.6 x last season) in
   a sport where spreads reach 50. This says the model is badly calibrated. It says nothing about the market.
3. Passing trials (3 of 14): T02 total CLV, T06 total no-vig ROI, T14 total CLV in weeks 1-4 (+1.03 points). No -110
   ROI trial passes. Early-season totals were not better than later ones (-110 ROI +3.0% vs +2.5%).
4. Stacking (not counted): total (model - open) predicts the line move (b +0.19, p 1e-47) and the outcome vs the open
   (b +0.25, p 1e-7), but **not the outcome vs the close** (b -0.004, CI -0.09 to +0.08). The close already contains
   everything the model knows.

**Action: none to bet.** The forward use would be a CFB totals open-line logger. That pays only at roughly no-vig
prices (a Kalshi maker order, if CFB totals are listed). The spread side needs a v2 with a fresh pre-registration:
a walk-forward slope recalibration fit on prior seasons' outcomes, no market data, or a weaker cap. I did NOT run
it, because scoring a post-hoc fix on these same games would be a forking path.

## Data

- **Sealed: CFB is not in the frozen MANIFEST** (sealed covers nba/nhl/mlb/nfl only), so 2025+ is treated as sealed
  and was never read (`data/espn/cfb/*_2025.parquet` were not opened).
- ESPN `data/espn/cfb/lines_*`: the opener columns are empty in 2022, cover about 450 games in 2023 (several books
  with prices), and about 530 in 2024 (ESPN BET only). The 2024 game feed is partial (548 games). That is too thin
  for a walk-forward test, so ESPN was not used.
- Used: the **sportsbookreviewsonline NCAAF archive, 2007-2022**, downloaded to `data/sbr_cfb/cfb_odds_<season>.html`
  (new directory). Columns used: finals, Open, Close. Limits: SBR gives no prices (-110 assumed), and it does not say
  which book "Open" comes from or when. The 2022 page stops on 11-15. 6-digit rotation rows (FCS-vs-FCS in 2019/2021)
  were dropped.
- FBS = at least 6 games per season, from results only (2020: at least 3 and FBS in 2019 or 2021). The counts match
  reality (120 rising to 131; 127 in 2020, with UConn/NMSU/ODU sitting out). An alias map fixes SBR's team renames.
  Both fixes were made BEFORE the market join, as an amendment in the docstring.
- Scored: 2008-2022 (2007 is burn-in), FBS-vs-FBS only, about 10,830 games with an open and a close. Halves: 2008-14
  and 2015-22.

## Model (pre-registered in `scripts/cfb_simple_model.py` docstring before the join)

- Margin: date-by-date weighted ridge on this season's games, with a 6-week recency half-life. Priors are 0.6 x last
  season's rating, and new FBS teams start at -10. Non-FBS teams are pooled as one "FCS" team. HFA = mean of prior
  seasons' estimates. Margin capped at +-28. QB availability was skipped (no cheap data).
- Totals: offence/defence points ridge with the same machinery; mu is shrunk toward last season's mean. Pace proxy
  not available (SBR has no plays).
- Outcome-only check (no market): margin MAE about 13.6, total MAE about 13.3, total bias within +-1 per season.
  Against the open: model MAE 13.58 vs open 12.43 (spread), 13.37 vs 13.25 (total).

## Trials (bar p < 0.05/14 = 0.00357; one-sided season-week cluster bootstrap; pass needs both halves > 0)

| trial | n | mean | 95% CI | p | 08-14 | 15-22 | pass |
|---|---|---|---|---|---|---|---|
| T01 spread CLV (pts) | 6334 | -0.12 | -0.19, -0.06 | 1.00 | -0.00 | -0.23 | no |
| T02 total CLV (pts) | 3014 | +1.02 | +0.87, +1.17 | <.0002 | +0.95 | +1.08 | **yes** |
| T03 spread ROI -110 | 6334 | -7.8% | -10.1, -5.6 | 1.00 | -7.9 | -7.8 | no |
| T04 total ROI -110 | 3014 | +2.6% | -1.3, +6.8 | .101 | -0.2 | +5.2 | no |
| T05 spread ROI no-vig | 6334 | -3.5% | -5.9, -1.2 | 1.00 | -3.6 | -3.5 | no |
| T06 total ROI no-vig | 3014 | +7.4% | +3.3, +11.8 | .0004 | +4.5 | +10.2 | **yes** |
| T07 spread -110, on fav | 240 | -7.3% | -18.9, +4.2 | .90 | -4.5 | -9.7 | no |
| T08 spread -110, on dog | 6094 | -7.9% | -10.2, -5.6 | 1.00 | -8.0 | -7.7 | no |
| T09 spread -110 wk 1-4 | 1768 | -10.0% | -13.6, -6.3 | 1.00 | -8.6 | -11.5 | no |
| T10 spread -110 wk 5+ | 4566 | -7.0% | -9.8, -4.2 | 1.00 | -7.6 | -6.5 | no |
| T11 total -110 wk 1-4 | 814 | +3.0% | -4.6, +10.8 | .21 | -1.9 | +8.7 | no |
| T12 total -110 wk 5+ | 2200 | +2.5% | -2.1, +7.1 | .15 | +0.6 | +4.1 | no |
| T13 spread CLV wk 1-4 | 1768 | -0.15 | -0.32, +0.03 | .95 | -0.07 | -0.24 | no |
| T14 total CLV wk 1-4 | 814 | +1.03 | +0.74, +1.31 | <.0002 | +1.09 | +0.96 | **yes** |

Totals bets won 53.8% at the open. Total CLV was positive in all 15 seasons (+0.22 to +2.17). Total ROI at -110 was
positive in 9 of 15 seasons. Caveat: the SBR "open" may be an early or overnight number from an unknown book.

Files: `scripts/cfb_simple_model.py` (`model`, `dev [--ledger]`, about 15 s), `data/lab/cfb_simple_model_results.json`,
ledger: 14 rows, dataset `cfb_simple_model`, in `lab/ledger.jsonl`.
