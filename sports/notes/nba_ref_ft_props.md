# L6b: referee crew FT tendency -> player points props (DEV only, 2026-10-25)

## Verdict: DEAD (0/4 pre-registered trials pass, bar p < 0.0125)

The mechanism is real, but the market prices it, and the leftover is smaller than the prop hold.
- Crews that call more FTs give high-FTA players more FTAs and points: +0.17 FTA/z (p .001) and +0.28 pts/z (p .006).
- **Points lines move toward the crew between open and close**: +0.135 pts/z (CI +0.04..+0.23, p .003). Crews
  are announced about 9am ET, and the market reacts to them.
- At the **opening** line, the hit-rate residual vs the de-vigged market is +0.8 pp per z (p .17). It is not
  stable: +1.3 pp in the first half, -0.3 pp in the second, and 2025 is negative. That is far below the ~6.9% mean hold at open.
- No free-throws-made prop exists in any DEV season (ESPN), so only points was testable.

Script / pre-registration: `scripts/nba_ref_ft_props.py` (docstring written before lines were joined).
Results: `results/nba_ref_ft_props.json`. Ledger: `lab/ledger.jsonl`, dataset `nba_ref_ft_props`, 4 trials.
SEALED 2025-26 was never loaded.

## Data

- Points props with an opening line and price: 2022-23 and 2023-24 at DraftKings, 2024-25 at ESPN BET (the only
  book). One observation per player-game (protocol rule 4): for each player-game, the over/under pair whose open
  is closest to 50/50, with proportional de-vig.
- Crews come from `refs_dev.parquet` (ESPN). The walk-forward referee rating uses prior dates only, with ridge
  shrinkage K=100. 2021-22 is burn-in, and 2022-23..2024-25 are scored (3,930 games).
- High-FTA players are those with prior FTA/g >= 5 over at least 15 prior games: 54 players, 5,952 player-games
  with quotes.
- Caveat: ESPN's "open" has no timestamp, so I cannot verify whether it predates the crew announcement. The
  significant open-to-close move suggests at least part of it does. "Current" is only approximately the close.

## Results

| Trial | Result | p | Pass |
|---|---|---|---|
| T1 over, high-FTA, crew z >= +1, at open | 2,174 bets, win 49.3% vs market 50.0%, ROI -8.1% (halves -4.9%, -13.0%) | 1.00 | no |
| T2 under, high-FTA, crew z <= -1, at open | 347 bets, win 53.6% vs market 50.3%, ROI +1.9% (halves +2.6%, +1.1%) | 0.37 | no |
| T3 slope (over - p_open) on z, high-FTA | +0.0079 per z, CI -0.003..+0.019 (halves +0.013 / -0.003) | 0.17 | no |
| G2 game total - close total on z | +0.42 pts per z, CI -0.09..+0.94 | 0.10 | no |

Descriptive:
- **G1 positive control**: total FTA residual is +0.80 FTA per z (CI 0.47..1.13). The walk-forward rating
  carries real signal, so the null is not a power failure in the feature.
- **P0**: high-FTA player FTA +0.17/z, points +0.28/z (about twice the a-priori +0.16 guess).
- **CLV**: line move +0.135 pts/z (p .003), i.e. the close absorbs about half of P0's +0.28.
- T3 at the close: +0.0045/z, p .45. T3 over all players: -0.0024/z, p .21.

**Pre-registration flaw (found after the run):** the crew z was not centred. Its mean is +0.58, because
league FTA drifts against the lagged 500-game baseline, so 35% of games hit z >= +1 and only 7% hit z <= -1. The
slope trials (T3, G2, P0, G1) do not depend on centring. T1 and T2 do.
POST-HOC, not a trial: with z centred on the prior-date crew mean, overs at zc >= +1 go 986 bets, win 50.9% vs
market 50.0%, ROI -5.1%. Unders at zc <= -1 go 1,278 bets, win 52.8% vs market 50.3%, ROI -1.4%. Both still lose.

## Why it fails (the lesson)

The effect chain is crew -> about 0.8 more FTA per game -> about 0.17 FTA for a star -> about 0.3 points. That is
about 1 pp of hit rate at best. Half of it is already in the close, and an opener costs about 7% hold. Even a perfectly known
crew effect cannot clear the vig on single props. A free-throws-made prop would have a larger relative effect,
but no such market exists in this data.

## If revisited

You would need a timestamped opener before 9am ET plus a free-throws-made market. The Odds API has props from
2023-05 with 5-minute snapshots (paid). With those, the test is the price change in the hour after the crew
announcement. The current data cannot isolate that.
