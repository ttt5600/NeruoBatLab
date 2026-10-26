# Kalshi early NFL TOTALS ladders vs the simple model (DEV 2025 wk 1-12), 2026-10-25

Script: `scripts/kalshi_early_nfl_totals.py` (`level`, `build`, `dev [--ledger]`). The pre-registration is the
docstring. It was written before any candle for a dev totals market was pulled. A frozen copy was taken at
2026-10-26 04:03Z, sha256 `f5b751b6...`, and is kept in the job scratch as `prereg_frozen.py`. Results are in
`data/lab/kalshi_early_nfl_totals_results.json`. Per-game bets are in
`data/lab/kalshi_early_nfl_totals_bets_{as_of_listing,std_week_features}.parquet`. Candles are in
`data/kalshi_early_nfl_totals_{candles,markets}.parquet`. The ledger has 3 rows under dataset `kalshi_early_nfl_totals`.

## Verdict

**DEAD as a dev edge. Kalshi's totals ladders are not stale in the way the game-winner market was.**
- **Trial results.** All three trials fail.
  - **T1, CLV.** +0.33 pts, CI -0.20 to +0.86, p .108, n 47.
  - **T2, taker at the strike nearest the model.** ROI +1.7%, CI -30% to +35%, p .47.
  - **T3, taker at the strike nearest the Kalshi implied total.** ROI -1.1%, CI -28% to +26%, p .56.
  - **Bar.** p < .0167.
- **Why the totals differ from L35.**
  - **First tight ladder vs the sportsbook close.** By the time a ladder first has a tight two-sided quote, its
    implied total is already within **1.08 pts** of the sportsbook closing total (mean absolute error; bias +0.14).
    At that point the game is a median 6.1 days from kickoff, and the ladder later moves a mean of only 1.1 pts.
  - **Model vs the sportsbook close.** The legacy model is 2.1 pts from the close.
  - **Model vs the ladder.** It disagrees with the first ladder by >= 2.5 pts in only 47 of 178 games, and
    corr(gap, later move) is only .20. In L35 that correlation was .42.
- **Power.** The test is underpowered, as stated before the join. n = 47 is below the 50-90 expected, and only a
  mean CLV of about 0.6 pts or an ROI above about 35% could have passed. So "no evidence" here is weak evidence of
  no edge, not proof of none. The one hint is in the uncounted slices: CLV at |gap| 4-6 pts is +1.0 (n 18), and
  under bets show +1.0 (n 17). Both were seen after the trials were scored, so both are leads for forward data
  only and must not be mined further on these games.

## Pre-registered decisions (fixed before prices)
- **Model choice.** Legacy model B totals: an OLS fit on 2007-19, so it is out of sample for 2025. C was not used
  because of its known level bias. Features are as of the listing time, following L35.
- **Level rule, run before `build`.** The rule compared the model with the sportsbook closing total over 2025 weeks
  1-4 (n 64). The bias was **+0.86 pts**, within the 1.5-pt tolerance, so **no correction** was applied and all of
  weeks 1-12 were scored. MAE was 1.88.
- **Bet set.** |model - first tight implied| >= 2.5 pts. The bet takes the model's side (over or under).
- **Taker fee.** ceil(7·p(1-p)) cents, i.e. a 1-contract order rounded up. Bets are held to settlement.

## Trials (as-of-listing features; one-sided game bootstrap, 5000 draws)

| trial | n | estimate | 95% CI | p | wk1-6 / wk7-12 | pass |
|---|---|---|---|---|---|---|
| T1 CLV (pts toward model) | 47 | +0.33 | [-0.20, +0.86] | .108 | +0.64 (18) / +0.14 (29) | no |
| T2 taker, model-nearest strike | 47 | +1.7% | [-30%, +35%] | .47 | +1.0% / +2.1% | no |
| T3 taker, implied-nearest strike | 47 | -1.1% | [-28%, +26%] | .56 | -16.5% / +8.3% | no |

- **T2 bets.** They won 44.7% at a mean price of 41.9c. The chosen strike sat a median 3.2 pts from the implied
  total. With the fee unrounded, ROI is +2.4%.
- **T3 bets.** They won 53.2% at a mean price of 51.8c.

## Uncounted diagnostics (after the trials)
- **Week-feature version.** These features use information from after the listing. T1 +0.41 (p .055, n 49),
  T2 -7.8%, T3 -8.9%. Corr(gap, move) is .29.
- **CLV by gap size.** 2.5-4 pts: -0.19 (26). 4-6 pts: +1.01 (18). >= 6 pts: +0.71 (3).
- **CLV by side.** Overs -0.07 (30), unders +1.02 (17). Blind drift of the implied total from first tight to last
  pre-kick is -0.18 pts, so totals drift down. The model leans over (64% overs, mean gap +0.50), which pulls against
  it. Part of the under-side CLV may simply be that downward drift.
- **Accuracy vs the final score (RMSE, 178 games).** First tight 13.16, last pre-kick 13.22, sportsbook close
  13.23, model 13.49. On final scores, the early ladder is as good as the close.

## How Kalshi lists NFL totals (2025 dev)
- **Lead time.** Events were listed a median 8.2 days before kickoff (IQR 5.7-10.7). Unlike the game-winner market,
  there were no May listings. Each event has 9-81 strikes, median 17, added over time.
- **First quotes.** The first two-sided quote anywhere on the ladder appears a median 0.6 h after listing. The first
  tight, bracketing ladder appears a median **15 h** after listing (mean 35 h). At that point a median 8 strikes
  qualify, and the strike nearest the implied total has a median spread of 3c.
- **Volume.** Median volume in the first 24 h after listing is **0** contracts per event (mean about 3k). Total
  pre-game volume is a median 118k per event. The first tight quote is therefore probably market-maker quoting into
  an empty book. It is already well calibrated, and the available size is unknown.

## Caveats
- **Bar resolution.** Hourly bars hide intra-hour quotes. The taker trade assumes the candle-close quote was still there.
- **No order book.** There is no historical depth data.
- **Interpolation.** The implied total is a linear interpolation between the bracketing strikes, after a monotone
  (pool-adjacent-violators) fit.
- **QB term.** It uses the schedule starter of the as-of game, as in L35.
- **Sealed data.** None was read. Weeks 13+, POST and 2026 were never fetched or scored. The 2025 pbp and schedule
  rows for week >= 13 were dropped before the features were built.
