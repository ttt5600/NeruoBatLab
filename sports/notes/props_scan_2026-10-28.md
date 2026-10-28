# Player-prop cross-venue price check, 2026-10-28 (DK via ESPN vs Kalshi)

**This is a price check between two venues. It is not a tested edge.** Our own prop backtests (notes/nba_ref_ft_props.md,
notes/nfl_props_script.md) found no model edge, and the "fair" used here is DraftKings' own price with the vig removed
by a formula. No orders were placed. Snapshot taken ~16:00 UTC (noon ET), before any tip-off. Scripts and raw data:
`~/.claude/jobs/588c8b26/tmp/{scan.py,summ.py,write_md.py}`, `tmp/scan/{matched.csv,dk_pairs.csv,books.json}`. Nothing committed.

## Method
- **Kalshi**: public `/markets` for every series ticker with `26OCT28`. Open today: KXNBAPTS 444, KXNBAREB 498, KXNBAAST 381,
  KXNBA3PT 539, KXNBAPRA 194, KXNBASTL 175, KXNBABLK 70, KXNBA2D 47, KXNHLGOAL 252, KXNHLPTS 165, KXNHLAST 127.
  KXNBAPR/PA/RA, the ladder/escalator series, and KXNHLSOG have nothing open. Every series uses the `quadratic` fee type:
  taker fee is 0.07·p·(1−p), rounded up to the cent (computed per 1-lot, which is conservative), and makers pay nothing.
- **Book**: ESPN core `propBets` for all 12 NBA and 4 NHL games. Only one provider exists, **DraftKings** (14.5k records).
  **Format change:** a record no longer says whether it is the over or the under. A two-way line is now two adjacent
  records with the same total, so `sports/espn.py::parse_prop` **returns 0 rows now**, and `collect.py props` would
  silently collect nothing. To work out which record is the over, I compared each pair against DK's own one-sided
  "Milestones N+" price at N = line+0.5 (n=586). The first record matched it (mean |diff| 0.002, corr 1.00) and the
  second did not (0.126), so **first = over**. 674 two-way pairs.
- **De-vig**: proportional, and power (the k with p_o^k + p_u^k = 1). Median gap between them is 0.6c and the largest is 2.8c.
  The flag edge uses the **worse** of the two for each side.
- **Matching**: same player name, and a Kalshi "N+" strike (floor_strike N−0.5) equal to the DK x.5 line gives an **exact**
  match (370). Where a DK line falls between two Kalshi strikes I interpolated the bid/ask (**interp**, 227). These are
  flagged as interp: they are not tradable as one contract and have no depth.
- **Flags**: taker YES if fair − ask − fee ≥ 3c. Taker NO if (1−fair) − NO ask − fee ≥ 3c. Maker if some cent price strictly
  inside the spread sits ≥ 3c inside fair. Depth is the number of contracts at the best opposing bid, from `/orderbook`.

## How close the venues are (exact matches, n=370)
- Kalshi mid minus DK fair: mean **−1.4c**, median |gap| **1.5c**, 90th percentile |gap| 3.7c. The median Kalshi spread is
  2c and the median DK overround is 6.8%. The median of (YES ask − fair) is 0.0c: **Kalshi's YES ask sits about at
  DK's de-vigged fair**, so after the taker fee almost nothing clears.
- The gap leans one way. Kalshi prices the over **below** DK's fair in every series (3PM −1.8c, PTS −1.6c, REB −1.5c,
  AST −1.3c, PRA −5.1c (n=10), NHL AST −0.9c, NHL PTS −0.4c). The likely cause is that DK puts more of its vig on the over.
  Proportional and power de-vig both assume the vig is spread evenly, so both overstate the fair over. **So nearly every
  flag below is a "buy YES / rest a YES bid" flag (4/4 taker, 31/36 maker), which is just what that de-vig bias would
  produce.** Treat these as de-vig model error until a graded sample shows otherwise.
- Interpolated rows are looser (median |gap| 2.9c, spread 6c), as you would expect.

## Flags (taker: all of them; maker: top 20 of 36 exact)
"edge c" is net of the taker fee for taker rows and is fair minus the resting price for maker rows (no maker fee). Both use the
conservative de-vig. "rest at" is the best price strictly inside the current spread.

| flag | player (game) | stat | line | match | DK over/under (US) | DK fair over: prop / power | Kalshi YES bid / ask | rest at | edge c | depth at ask | Kalshi vol | injury |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| taker YES | Franz Wagner (ORL @ TOR) | NBA PRA | 37.5 | interp | -108/-122 | 0.486 / 0.484 | 0.25 / 0.42 |  | 4.2 | n/a | 266 |  |
| taker YES | Karl-Anthony Towns (NY @ CHI) | NBA PRA | 39.5 | exact | -112/-118 | 0.494 / 0.493 | 0.38 / 0.44 |  | 3.3 | 41 | 149 |  |
| taker YES | Bam Adebayo (MIA @ ATL) | NBA PTS | 18.5 | interp | -117/-109 | 0.508 / 0.509 | 0.42 / 0.46 |  | 3.0 | n/a | 0 |  |
| taker YES | Klay Thompson (MIA @ ATL) | NBA 3PM | 2.5 | exact | -120/-110 | 0.510 / 0.511 | 0.45 / 0.46 |  | 3.0 | 10 | 19 |  |
| maker YES | Karl-Anthony Towns (NY @ CHI) | NBA PRA | 39.5 | exact | -112/-118 | 0.494 / 0.493 | 0.38 / 0.44 | 0.43 | 6.3 | 41 | 149 |  |
| maker YES | Devin Vassell (SA @ UTAH) | NBA AST | 3.5 | exact | -129/-102 | 0.527 / 0.530 | 0.45 / 0.48 | 0.47 | 5.7 | 1634 | 122 |  |
| maker YES | Jared McCain (OKC @ DAL) | NBA 3PM | 3.5 | exact | 126/-166 | 0.415 / 0.406 | 0.29 / 0.36 | 0.35 | 5.6 | 7 | 7 |  |
| maker YES | Daniel Gafford (OKC @ DAL) | NBA REB | 7.5 | exact | -112/-117 | 0.495 / 0.494 | 0.43 / 0.45 | 0.44 | 5.4 | 5 | 49 |  |
| maker YES | Dejounte Murray (DEN @ NO) | NBA PRA | 29.5 | exact | -110/-120 | 0.490 / 0.489 | 0.40 / 0.45 | 0.44 | 4.9 | 29 | 11 |  |
| maker YES | Evan Mobley (WSH @ CLE) | NBA AST | 3.5 | exact | -143/109 | 0.552 / 0.557 | 0.49 / 0.52 | 0.51 | 4.2 | 85 | 0 |  |
| maker YES | Trae Young (WSH @ CLE) | NBA 3PM | 2.5 | exact | -119/-110 | 0.509 / 0.510 | 0.45 / 0.48 | 0.47 | 3.9 | 38 | 54 |  |
| maker NO | Yaxel Lendeborg (GS @ MIN) | NBA 3PM | 1.5 | exact | -110/-119 | 0.491 / 0.490 | 0.51 / 0.54 | 0.47 | 3.9 | 407 | 61 | Day-To-Day |
| maker YES | Tyrese Maxey (PHI @ IND) | NBA PRA | 34.5 | exact | -110/-120 | 0.490 / 0.489 | 0.41 / 0.46 | 0.45 | 3.9 | 244 | 403 |  |
| maker YES | Jared McCain (OKC @ DAL) | NBA REB | 3.5 | exact | 120/-158 | 0.426 / 0.418 | 0.37 / 0.40 | 0.38 | 3.8 | 9 | 95 |  |
| maker YES | James Harden (WSH @ CLE) | NBA AST | 7.5 | exact | 103/-135 | 0.462 / 0.458 | 0.41 / 0.46 | 0.42 | 3.8 | 3960 | 12 |  |
| maker YES | William Nylander (TOR @ SEA) | NHL PTS | 1.5 | exact | 180/-245 | 0.335 / 0.317 | 0.14 / 0.34 | 0.28 | 3.7 | 787 | 0 |  |
| maker YES | Giannis Antetokounmpo (MIA @ ATL) | NBA PRA | 49.5 | exact | -118/-111 | 0.507 / 0.508 | 0.46 / 0.49 | 0.47 | 3.7 | 414 | 55 |  |
| maker YES | Ace Bailey (SA @ UTAH) | NBA REB | 5.5 | exact | -113/-116 | 0.497 / 0.497 | 0.45 / 0.48 | 0.46 | 3.7 | 900 | 129 |  |
| maker YES | P.J. Washington (OKC @ DAL) | NBA REB | 7.5 | exact | 112/-147 | 0.442 / 0.436 | 0.18 / 0.43 | 0.40 | 3.6 | 9 | 10 |  |
| maker YES | Lauri Markkanen (SA @ UTAH) | NBA AST | 1.5 | exact | -146/111 | 0.556 / 0.562 | 0.50 / 0.55 | 0.52 | 3.6 | 1440 | 0 | Day-To-Day |
| maker YES | Bam Adebayo (MIA @ ATL) | NBA 3PM | 1.5 | exact | -104/-126 | 0.478 / 0.475 | 0.41 / 0.45 | 0.44 | 3.5 | 12 | 50 |  |
| maker YES | Peyton Watson (WSH @ CLE) | NBA REB | 5.5 | exact | 117/-154 | 0.432 / 0.425 | 0.38 / 0.40 | 0.39 | 3.5 | 2549 | 1 |  |
| maker NO | Cale Makar (NJ @ COL) | NHL AST | 0.5 | exact | -105/-130 | 0.475 / 0.472 | 0.50 / 0.53 | 0.49 | 3.5 | 960 | 1861 |  |
| maker NO | Alex Tuch (PHI @ WSH) | NHL PTS | 0.5 | exact | -120/-115 | 0.505 / 0.506 | 0.52 / 0.55 | 0.46 | 3.4 | 2332 | 106 |  |

## Injury / staleness notes (ESPN injuries, 2026-10-27/28)
- **Mattias Ekholm (EDM, Out, "will miss at least a week" since 10-23)**: both venues still quote him. DK AST 0.5 fair 0.28
  vs Kalshi 0.28/0.29, and DK PTS 0.5 fair 0.32 vs Kalshi 0.29/0.33. Neither venue is stale relative to the other. Both are
  stale relative to the news, unless he has been activated. Check the void/DNP rules on both venues before touching it.
- **AJ Dybantsa (UTAH, listed Out, but the comment says he was at shootaround)**: the status conflicts. The venues agree
  (3PM 1.5 fair 0.37 vs 0.32/0.36; AST 1.5 0.61 vs 0.58/0.63).
- Questionable: Irving, Markkanen, Lendeborg, Haliburton, Nembhard, Ajay Mitchell, Wiggins, and Anthony Black (missed
  shootaround). Both venues quote all of them, mostly within 0–4c. No case looks like one venue hasn't reacted while the
  other has. Marcus Smart (questionable) is on Kalshi only and has no DK line.
- ESPN `lastUpdated` stamps run ahead of the snapshot time (for example 18:24Z when the snapshot was ~16:00Z). They are not
  a usable freshness signal (see notes/data.md), so staleness was judged from the injury news only.

## Bottom line
Kalshi's player-prop ladders track DraftKings closely: about 1.5c typical gap and a 2c spread. Only 2 exact taker flags clear
3c, and both are barely over (3.0–3.3c, 10 and 41 contracts deep). The flags lean the same way the de-vig bias leans.
Nothing here is actionable as an edge without a graded study. The cheap way to test "Kalshi YES ladders under DK fair" would
be to log this snapshot daily and grade it against box scores (our ESPN box pipeline already does the grading). That needs a
pre-registration first.
