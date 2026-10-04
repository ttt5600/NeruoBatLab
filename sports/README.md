# sports — game and player data for NFL, NBA, MLB, NHL, built for backtesting

Free public sources, one schema, and a timing tag on every column so a
pre-game model cannot see the final score by accident.

```bash
# from sports/, using the repo venv
../.venv_quant/bin/python -m pytest tests/ -q          # 14 offline tests
../.venv_quant/bin/python scripts/pull.py              # game tables, 2015 on
../.venv_quant/bin/python scripts/pull.py --nfl-players
../.venv_quant/bin/python scripts/pull.py --nba-boxes 2024 2025
../.venv_quant/bin/python scripts/baseline.py          # sanity + Elo + priced-in
```

## Sources (all free, no keys)

| league | games from | what it carries | requests |
|---|---|---|---|
| NFL | [nflverse](https://github.com/nflverse/nfldata) | stadium, roof, surface, temp/wind, both coaches, referee, starting QBs, **closing spread/total/moneylines**; weekly player stats (150 cols) | 1 file + 1/season |
| MLB | [MLB Stats API](https://statsapi.mlb.com) | venue lat/lon, elevation, roof and turf, weather at first pitch, **announced probable pitchers**, plate umpire, day/night | ~8/season |
| NHL | [api-web.nhle.com](https://github.com/Zmalski/NHL-API-Reference) | venue, neutral site, OT/SO; **starting goalies** from boxscores | ~35/season + 1/game |
| NBA | ESPN site API | venue, attendance, neutral site, OT; per-player box scores | 1/day + 1/game |

stats.nba.com refused every request from this machine, with or without browser
headers, so the NBA comes from ESPN. ESPN does not accept date ranges for the
NBA; a season is ~270 daily requests, cached forever once the day is final.

## The one design decision that matters: timing tags

Every column in `sports/schema.py` is tagged with when it becomes knowable:

| tag | known | examples |
|---|---|---|
| `pre` | days ahead | schedule, venue, roof, coaches, rest |
| `lineup` | ~1 hour ahead | starting QB / pitcher / goalie, officials, start-time weather |
| `close` | at the closing line | spread, total, moneylines |
| `post` | after the game | scores, attendance |

`schema.pregame(df, as_of="lineup")` returns only what a bettor could have
seen. `conform()` refuses any column that has no tag. Derived features
(`features.py`: rest, back-to-backs, rolling form, Elo) use strictly earlier
games, and two tests prove it, each catching a different leak:

- `test_features_are_causal` recomputes every feature on history truncated
  just before the game. Catches **future-game** leaks (a full-season average).
- `test_features_do_not_see_their_own_result` flips one game's score and
  requires its own features to be unchanged. Catches **own-result** leaks
  (Elo stored after the update, a rolling mean without `shift(1)`).

Both were checked by planting each leak and watching the right test fail. The
truncation test is blind to own-result leaks by construction — it keeps the
game itself — which is why there are two.

## First results (`scripts/baseline.py`, NFL + MLB)

**The data reproduce what everyone already knows**, which is the check that
the pipeline is right before anything is read from it:

- NFL home teams won **50.4%** in 2020 (empty stadiums) vs 52–61% otherwise.
- 2022 has **271** NFL regular-season games, not 272: Bills–Bengals was never finished.
- MLB 2020 has **898** games: the 60-game season, two never made up.
- MLB scoring rises monotonically with first-pitch temperature, **8.46 → 9.93**
  runs/game from <55°F to 85°F+. Coors Field is the top park at **11.46**.

**Elo beats a coin flip and loses badly to the market.** NFL 2016–2026, a
priori FiveThirtyEight parameters, nothing tuned:

| forecast | log loss |
|---|---|
| coin flip | 0.6931 |
| prior seasons' home-win rate | 0.6893 |
| Elo | 0.6470 |
| **devigged closing moneyline** | **0.6097** |

Elo minus market: **+0.037 per game, 95% CI [+0.028, +0.046]**. Learning that
good teams win is not an edge; the closing line already knows it. This is the
buy-and-hold lesson from `quant` again — the benchmark is the market, not zero.

**Is it priced in?** Each angle shown two ways: raw outcome, then outcome net
of the closing line. Only the second is money.

- **Rest advantage:** home teams with 4+ more days of rest win by +3.2, but
  beat the spread by −0.2 [−1.9, +1.5]. Priced in.
- **Home underdogs:** covered **50.8%** (575–556, pushes excluded), below the
  52.4% needed to beat -110. Home favourites covered 47.9%. The famous angle
  leans the right way and does not pay.
- **Wind:** totals fall with wind (47.0 → 41.5 points), and the residual
  against the closing total looks non-zero: **−2.6 [−4.5, −0.8]** at 15–20 mph,
  **+1.8 [+0.4, +3.1]** at 0–5 mph. **This is very likely not an edge.**
  nflverse records the wind *observed* at kickoff; the market prices the
  *forecast*. Binning by observed wind selects the games where the wind beat
  its forecast, which produces exactly this pattern even if the market priced
  the forecast perfectly. Settling it needs archived forecasts —
  Open-Meteo's Previous Runs API has them, but only from January 2024
  (about 450 outdoor games). The schema's `lineup` caveat about weather
  exists for this reason.

## Published angles, tested as money (`scripts/angles.py`)

Rules taken from the literature, not from this data, scored by what flat
1-unit bets would have paid at the NFL closing price, 2016–2026:

| angle | bets | ROI | 95% CI | Holm p |
|---|---|---|---|---|
| away ML in close games (p_home 0.3–0.7) | 1834 | −0.53% | [−5.33%, +4.51%] | 1.000 |
| week 1: fade last year's playoff teams (ATS, −110 assumed) | 73 | +1.99% | [−18.93%, +22.91%] | 1.000 |
| all favourites, moneyline | 2808 | −2.88% | [−5.49%, −0.25%] | 1.000 |
| all underdogs, moneyline | 2808 | −5.27% | [−10.52%, +0.05%] | 1.000 |
| big underdogs only (p < 0.25) | 590 | −11.57% | [−27.44%, +5.00%] | 1.000 |

None pays. The week-1 holdover angle's CI is ±21% because there are only 73
qualifying bets in eleven seasons; it cannot be confirmed or refuted here.

The devigged closing moneyline is close to calibrated. There is a mild
favourite–longshot tilt (home teams priced at 0.28 won 0.239 ± 0.022; those
priced at 0.85 won 0.882 ± 0.023), which is why favourites lose less than
underdogs — but less than the vig, so neither side is a bet. That is what an
efficient market with a bookmaker's margin looks like.

## Honest limits

- **NFL weather is patchy:** temp/wind present for only 38–75% of games by
  season, and not just domes (2022 is the worst). Missingness may not be random.
- **Starters differ in meaning by league.** MLB gives the announced probable
  pitcher (truly pre-game). NFL and NHL give who actually started.
- **No coaches outside the NFL**, and no venue coordinates for NFL/NBA/NHL yet.
- **Closing lines only for the NFL** so far. The other three leagues need the
  odds step before any "priced in?" question can be asked of them.
- **Four undocumented APIs.** The parser tests run against captured real
  responses; re-run `scripts/capture_fixtures.py` and the tests will say what
  changed when an endpoint moves.
