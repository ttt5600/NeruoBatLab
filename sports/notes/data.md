# Data — sources, coverage, quirks

## Collect (free, no keys)
`python scripts/collect.py all --league {nfl,cfb,nba,wnba,nhl,mlb} --seasons A B`; `status` prints coverage.
Writes `data/espn/<league>/{games,lines,props,box,weather,graded}_<season>.parquet`. HTTP cached in data/cache.
Modules: sports/espn.py (games/lines/props/box), sports/settle.py (grading map), sports/gamewx.py (ERA5 + forecasts).

## Coverage (2026-10-14)
| league | props | notes |
|---|---|---|
| NBA | 2022 small; 2023 & 2024 ~900k sides, 7-8 books; 2025-26 ESPN BET only | best depth |
| NHL | 2024-2026 (ESPN BET) | 2022-23 none |
| MLB | 2024, 2025 (ESPN BET; many markets unmapped: total bases, singles, runs...) | |
| NFL | yardage O/U only 2024-25; 2022-23 almost all TD-scorer markets | |
| CFB/WNBA | games/lines/box; props ~none | |
Other: nflverse games 1999+, player weeks 2015+, pbp + coverage charting (sports/nflpbp.py, 2016-25),
MLB box 2015+, SBR MLB odds 2010-21, Kalshi temp markets 2021+.

## Quirks (verified)
- Prop `last_updated` is a settlement stamp (~6h after kickoff); "current" ~= close, NOT verified.
- Lines that moved (line != line_open): stored price may belong to the old line -> fake edges (MLB hits overs +20-36%).
- MGM prices least informative (Brier .2418); DraftKings sharpest (.2280) — use DK as prop fair reference.
- Multiplicative de-vig inflates longshots; use min(multiplicative, power).
- Team codes: ESPN WSH/LAR vs nflverse WAS/LA. Kalshi names teams by city prefix.

## Frozen snapshot
`python scripts/freeze.py` -> `data/frozen/<date>/` read-only + MANIFEST.json (sha256, sealed seasons).
Current: 2026-10-14, sealed {nba:2026, nhl:2026, mlb:2025, nfl:2025}.

## Paid options (researched 2026-10-12, reports/Sports betting edge research.md)
The Odds API: lines from Jun 2020, props from 3 May 2023, 5-min snapshots; historical costs 10 credits/region/market
(props per event). 5M plan $119/mo fits ~1-2M credits for 2023-25 props. EndZone NFL props+Pinnacle $19/mo.
No vendor has 10+ seasons of per-game props.
