# Tonight's NBA information-edge sheet, Mon 2026-10-26 (9 games)

User ask: "run the numbers for tonight's games; try not to use exact model prediction, we need to see what no one else
is seeing." Built at ~12:40 ET by `scripts/nba_tonight_sheet.py`. The tables below the AUTO marker regenerate on every
re-run, and this read-out stays. Descriptive only, NO orders. The model (`nba1026.json`, run 00:37 PT) is used ONCE,
as a cross-check (bias-corrected total = model - 2.3).

**Re-run:** `cd sports && ../.venv_quant/bin/python scripts/nba_tonight_sheet.py --model-json <model json>`
(1) after **1 pm ET**: that is the injury-report deadline for teams on the 2nd night of a back-to-back. Five teams
are on one tonight: MIN, IND, OKC, MEM, UTAH. (2) Again ~6 pm ET. Referees are already posted (official.nba.com, 9 games).

## Bottom line (blunt)

- **No price edge anywhere.** Every Kalshi ML, total and spread contract is at or above the Pinnacle no-vig price
  after the taker fee. Best is ATL ML at +0.002, and nothing reaches the 2c bar. DraftKings (the only book ESPN lists)
  is within -4%..+0.6% EV of Pinnacle on every side. The apparent DK "UTAH +1.5 -120 = +3%" vanishes (+0.2%) once the
  margin distribution also matches Pinnacle's ML. Its +/-1-point masses were too fat.
- **No public betting %** exists in ESPN's feeds, so reverse-line moves are NOT computed.
- **Early-season 3P luck is the "not seen" angle, but it is weak.** I checked it on DEV (2022-23..2024-25,
  `scripts/nba_luck_early_calib.py`, descriptive, not a ledger trial). In games where both teams had played 1-5 games,
  the luck metric's slope on (actual - closing total) is -0.14 pts/pt (se 0.23, p .55). Games with luck <= -4 ("both
  sides shot cold") went **+7.0 pts over the close** (n 59; 66% overs; positive in all 3 seasons). But the effect is
  NOT dose-responsive (<= -6: +3.6, n 35; <= -8: +4.4, n 21), and hot games went over too (+3.9, n 27). The market
  does not move its line on luck either (open->close slope ~0). So read it as a weak over-tilt for cold games, not a signal.
  The whole league is cold: 3P% is .340 this season vs .360 last, which is why 5 of 9 games show luck <= -4.
- **The model's biggest disagreements tonight are its known blind spot (STRICT availability)**, not information.
  GS@DEN: the model counts Curry, Green and Podziemski OUT because they sat the 2nd night of a back-to-back. They are
  not on the injury report. ATL@HOU: it counts VanVleet OUT for the same reason. IND@MIL: its IND prior rating predates
  Haliburton's return. Ignore those model sides.
- The L32 open-totals logger's three model picks tonight (over MIN@CLE, over PHX@OKC, under IND@MIL) all have
  positive CLV so far (Pinnacle 233->234, 219.5->221.5, 240.5->239). Logged in that system, not counted here.

## Per game: what the market may be missing

| Game | Lean | Tier | Reason (and what argues against it) |
|---|---|---|---|
| MIN@CLE | CLE -4.5 | weak | About 17 of MIN's +23.3 pts/g margin is 3P luck: opponents shoot 23.6% (-11.8 pts/g) and MIN 42.4% (+5.5). MIN is on a back-to-back (hosted TOR last night). Paint +1.0 to CLE. Against: the market already went CLE's way (ML -170 -> -180, spread flat). Total: none (+1 already moved; model BC +1.1). |
| WSH@CHA | over 228.5 | weak | Slate-high game luck -10.4: CHA 29.6% on 47 3PA/g, WSH 28.6%, WSH opponents 28.8%. Model BC 229.9 (+1.4) already includes CHA's outs. Against: the 1-pt drop is real news (Knueppel, Coby White and Grant Williams out), and the DEV luck effect is weak. |
| CHI@BOS | CHI +14 / over 232.5 | weak (best on slate) | Clearest case of a line moving on shooting noise. CHI shoots 22.6% from 3 on 41/g (luck -13.9 pts/g), so its -13.3 margin is almost entirely shooting. Since open, the spread went 1 pt toward BOS and the total dropped 2 (Pin 234.5 -> 232.5, Kalshi 234 -> 232.8). BOS plays again 10-27 (front of a back-to-back), so its starters may sit in a blowout. Model: CHI side +1.7. Crew Zarba/Goble/Mulla is +1.5 on totals vs the close (DEV, shrunk, context only). Against: paint +1.1 to BOS, model BC total says under 1.7, and CHI has P. Williams and Essengue day-to-day. |
| DAL@MIA | none | none | Total +5 everywhere (DK 235.5 -> 240.5, Pin 240, Kalshi 240.8). Pinnacle moved too, so this is informed money (DAL without Lively, Aldama, Marshall and Martin; Flagg day-to-day). The model's under (-3.4) is the "model vs a moved market" trap. Kyrie sat the 2nd night of the back-to-back and is not listed, so expect him to play. |
| IND@MIL | watch: MIL +4.5 only if IND rests a starter | none now | IND is a 4-4.5-pt road favourite on the 2nd night of a back-to-back (@BKN last night, 1,182 km). Haliburton, back from his Achilles injury, played 31 min last night. Last season Siakam, Nembhard and Nesmith sat 58% of 12 second nights (confounded by IND's lost 2025-26 season). IND's report is due 1 pm ET. If someone rests and the line lags, that is the edge. The model's MIL +5.9 / under -5.6 is an artefact of its prior rating. |
| PHX@OKC | PHX +10.5 | weak | OKC is on a back-to-back AND lost Jalen Williams (Out, posted 10-26; Wallace and Jaylin Williams day-to-day). Yet DK moved toward OKC (-9.5 -> -10.5) while Pinnacle stayed -9.5. PHX's -16 margin over 2 games is 3P noise at both ends (29.0% for, 41.9% against = 13.6 pts/g). Against: DK +10.5 -110 is only fair vs Pinnacle (EV -0.3%), not a price edge. Total: none (moved +2 toward the L32 model pick). |
| ATL@HOU | over 229.5 | weak | Game luck -6.0 (opponents shoot 29.0% vs HOU and 30.2% vs ATL). Model BC +1.9 even while wrongly counting VanVleet out. Against: Smart is out (posted 10-26) and Adams is out. Side: none (the model's ATL lean is the VanVleet artefact). |
| MEM@UTAH | UTAH (Pin -1.5 / DK +1.5) | weak | Both teams on a back-to-back, but MEM travelled (@SAC last night, +1h). Altitude home +1.0 ATS (lead, p .055). DK swung 3 pts toward MEM (UTAH -1.5 -> +1.5) while Pinnacle holds UTAH -1.5 / 52.3%. Probably Edey moving from Out to day-to-day: check his status. Crew Foster +1.3 totals (context). |
| GS@DEN | over 236.5; DEN -7 | weak | DEN shoots 29.9% from 3 vs 39.6% last season, and opponents shoot 27.2% against them. Game luck -9.7. Altitude +1.0 to DEN. Curry, Green and Horford sat the 2nd night @PHX and are not listed, so expect them back. Pin total 235.5 -> 233.5 overnight -> 236.5 now (probably that news). Against: the model's DEN +10 / under are STRICT artefacts, so no model support. |

**Top 3 (all weak; size as paper or $1 units at most):** (1) CHI +14 / over 232.5, the line that moved on CHI's
22.6% 3P. (2) WSH@CHA over 228.5, slate-high cold shooting plus model BC. (3) MIL +4.5 conditional on an IND rest at
the 1 pm ET report. GS@DEN over 236.5 is the alternate.

Sources that worked: ESPN core odds (DK only), Pinnacle guest API, Pinnacle first-seen totals (L32 logger state),
Kalshi KXNBAGAME/KXNBATOTAL/KXNBASPREAD, ESPN injuries/rosters/box/paint, official.nba.com refs. None failed on the 12:40 ET run.
Caveats: prior-season 3P% priors ignore roster turnover. Travel uses the previous game's home arena. Paint z blends
this season with last at gp/(gp+20) and uses the DEV per-SD-of-difference coefficient.

<!-- AUTO: everything below is regenerated by scripts/nba_tonight_sheet.py -->

# Tonight's NBA information-edge sheet, 2026-10-26 (auto section)

Generated 2026-10-26 12:44 ET by `scripts/nba_tonight_sheet.py`. Descriptive only; no orders. League 3P%: prior season 0.360, this season so far 0.340.

ESPN lists no public betting % in its feeds -> reverse-line-move flags are not computed (not invented).

## 1. Line movement

| Game | Tip ET | DK spread open->now (home) | DK total open->now | DK ML open->now (H/A) | Pin open total (first seen) | Pin now: spread / total / home no-vig | Kalshi ML ask H/A | Kalshi implied total (open->now) |
|---|---|---|---|---|---|---|---|---|
| MIN@CLE | 19:00 | -4.5 -> -4.5 | 232.5 -> 233.5 | -170/+142 -> -180/+150 | 233.0 | -4.5 / 234.0 / 0.610 | 0.63/0.39 | 232.7 -> 233.8 |
| WSH@CHA | 19:00 | -2.5 -> -2.5 | 228.5 -> 227.5 | -142/+120 -> -135/+114 | 229.5 | -2.5 / 228.5 / 0.574 | 0.58/0.43 | 229.0 -> 228.0 |
| CHI@BOS | 19:30 | -13.5 -> -14.5 | 233.5 -> 231.5 | -850/+575 -> -750/+525 | 234.5 | -14.0 / 232.5 / 0.848 | 0.87/0.15 | 234.0 -> 232.8 |
| DAL@MIA | 19:30 | -7.5 -> -8.5 | 235.5 -> 240.5 | -290/+235 -> -298/+240 | 235.5 | -8.0 / 240.0 / 0.734 | 0.73/0.28 | 235.7 -> 240.8 |
| IND@MIL | 20:00 | +4.5 -> +4.5 | 239.5 -> 238.5 | +145/-175 -> +140/-166 | 240.5 | +4.0 / 239.0 / 0.387 | 0.39/0.62 | 240.5 -> 238.7 |
| PHX@OKC | 20:00 | -9.5 -> -10.5 | 218.5 -> 220.5 | -380/+300 -> -410/+320 | 219.5 | -9.5 / 221.5 / 0.776 | 0.79/0.22 | 219.7 -> 221.1 |
| ATL@HOU | 20:30 | -5.5 -> -4.5 | 229.5 -> 229.5 | -218/+180 -> -205/+170 | 229.5 | -5.0 / 229.5 / 0.632 | 0.66/0.35 | 229.7 -> 229.7 |
| MEM@UTAH | 21:00 | -1.5 -> +1.5 | 237.5 -> 237.5 | -115/-105 -> -110/-110 | 239.0 | -1.5 / 238.5 / 0.523 | 0.52/0.49 | 238.3 -> 238.5 |
| GS@DEN | 22:00 | -7.5 -> -7.5 | 235.5 -> 236.5 | -290/+235 -> -270/+220 | 235.5 | -7.0 / 236.5 / 0.697 | 0.71/0.30 | 235.7 -> 236.6 |

Books listed by ESPN: DraftKings.

## 2. Kalshi vs Pinnacle (edge = Pinnacle no-vig - Kalshi ask - taker fee; flag >= +0.02)

| Game | ML home ask/fair/edge | ML away ask/fair/edge | best total contract | best spread contract | best 2 book prices vs Pinnacle (EV/$1) |
|---|---|---|---|---|---|
| MIN@CLE | 0.63/0.610/-0.036 | 0.39/0.390/-0.017 | over 232.5 @ 0.53: -0.016 ~ | yes: Cleveland wins by over 6.5 points @ 0.42: -0.006 ~ (pmf P(home win) 0.612 vs Pin ML 0.610) | DraftKings MIN ML +150 p 0.390 EV -2.5%; DraftKings o233.5 -112 p 0.509 EV -3.7% |
| WSH@CHA | 0.58/0.574/-0.023 | 0.43/0.426/-0.021 | over 228.5 @ 0.49: -0.021 | yes: Charlotte wins by over 3.5 points @ 0.47: -0.017 ~ (pmf P(home win) 0.573 vs Pin ML 0.574) | DraftKings CHA ML -135 p 0.574 EV -0.1%; DraftKings o227.5 -105 p 0.508 EV -0.7% |
| CHI@BOS | 0.87/0.848/-0.030 | 0.15/0.152/-0.007 | over 233.5 @ 0.49: -0.020 ~ | yes: Boston wins by over 16.5 points @ 0.43: +0.010 ~ (pmf P(home win) 0.850 vs Pin ML 0.848) | DraftKings o231.5 -112 p 0.531 EV +0.6%; DraftKings BOS -14.5 -105 p 0.513 EV +0.1% |
| DAL@MIA | 0.73/0.734/-0.010 | 0.28/0.266/-0.028 | under 239.5 @ 0.47: -0.017 ~ | yes: Miami wins by over 10.5 points @ 0.43: +0.009 ~ (pmf P(home win) 0.736 vs Pin ML 0.734) | DraftKings MIA -8.5 -105 p 0.515 EV +0.5%; DraftKings MIA ML -298 p 0.734 EV -2.0% |
| IND@MIL | 0.39/0.387/-0.020 | 0.62/0.613/-0.023 | over 240.5 @ 0.46: -0.017 ~ | no: Indiana wins by over 4.5 points @ 0.49: +0.011 ~ (pmf P(home win) 0.388 vs Pin ML 0.387) | DraftKings IND ML -166 p 0.613 EV -1.7%; DraftKings MIL +4.5 -112 p 0.519 EV -1.8% |
| PHX@OKC | 0.79/0.776/-0.026 | 0.22/0.224/-0.008 | under 220.5 @ 0.49: -0.019 ~ | no: Oklahoma City wins by over 7.5 points @ 0.41: +0.001 ~ (pmf P(home win) 0.772 vs Pin ML 0.776) | DraftKings PHX +10.5 -110 p 0.522 EV -0.3%; DraftKings o220.5 -112 p 0.512 EV -3.1% |
| ATL@HOU | 0.66/0.632/-0.043 | 0.35/0.368/+0.002 | over 229.5 @ 0.51: -0.021 | yes: Houston wins by over 7.5 points @ 0.40: +0.006 ~ (pmf P(home win) 0.633 vs Pin ML 0.632) | DraftKings ATL ML +170 p 0.368 EV -0.7%; DraftKings o229.5 -112 p 0.507 EV -4.1% |
| MEM@UTAH | 0.52/0.523/-0.014 | 0.49/0.477/-0.031 | over 238.5 @ 0.51: -0.025 | yes: Utah wins by over 3.5 points @ 0.43: +0.006 ~ (pmf P(home win) 0.523 vs Pin ML 0.523) | DraftKings UTAH +1.5 -120 p 0.546 EV +0.2%; DraftKings o237.5 -110 p 0.524 EV +0.1% |
| GS@DEN | 0.71/0.697/-0.027 | 0.30/0.303/-0.012 | over 237.5 @ 0.48: -0.015 ~ | yes: Denver wins by over 9.5 points @ 0.42: +0.003 ~ (pmf P(home win) 0.696 vs Pin ML 0.697) | DraftKings DEN -7.5 -105 p 0.498 EV -2.8%; DraftKings GS ML +220 p 0.303 EV -3.0% |

Flags: **none >= 2c**

## 3. Availability and schedule

| Game | Team | GP | Rest days (prev game) | B2B? | Travel km / tz shift | Next game (B2B front?) | Injury report (status, date) | Missed last game, NOT listed | B2B 2nd-night sitters (prev+this season, >=20 mpg) |
|---|---|---|---|---|---|---|---|---|---|
| MIN@CLE | MIN | 3 | 0 (vs TOR 10-25) | YES | 1014 / +1h | 10-28 (no) | Donte DiVincenzo Out (2026-10-05) | - | - |
| MIN@CLE | CLE | 2 | 1 (vs BKN 10-24) | no | 0 / +0h | 10-28 (no) | none | - | - |
| WSH@CHA | WSH | 3 | 1 (@ CHI 10-24) | no | 949 / +1h | 10-28 (no) | AJ Dybantsa Out (2026-10-25); Tristan Vukcevic Out (2026-10-01); Jamir Watkins Out (2026-07-30) | - | - |
| WSH@CHA | CHA | 3 | 1 (vs MIA 10-24) | no | 0 / +0h | 10-28 (no) | Naz Reid Day-To-Day (2026-10-26); Kon Knueppel Out (2026-10-25); Coby White Out (2026-10-25); Grant Williams Out (2026-10-24); Kylan Boswell Out (2026-10-20) | - | - |
| CHI@BOS | CHI | 3 | 1 (vs WSH 10-24) | no | 1369 / +1h | 10-28 (no) | Patrick Williams Day-To-Day (2026-10-25); Noa Essengue Day-To-Day (2026-10-25); Nic Claxton Out (2026-10-19) | - | - |
| CHI@BOS | BOS | 2 | 2 (vs NY 10-23) | no | 0 / +0h | 10-27 (yes) | Devin Carter Out (2026-10-18) | - | - |
| DAL@MIA | DAL | 3 | 1 (vs SA 10-24) | no | 1788 / +1h | 10-28 (no) | Cooper Flagg Day-To-Day (2026-10-26); Caleb Martin Out (2026-10-25); Naji Marshall Out (2026-10-25); Tarik Biberovic Out (2026-10-24); Santi Aldama Out (2026-10-24); Dereck Lively II Out (2026-10-22) | Kyrie Irving (24 mpg) | - |
| DAL@MIA | MIA | 3 | 1 (@ CHA 10-24) | no | 1052 / +0h | 10-28 (no) | none | - | - |
| IND@MIL | IND | 3 | 0 (@ BKN 10-25) | YES | 1182 / -1h | 10-28 (no) | Ivica Zubac Day-To-Day (2026-10-26); T.J. McConnell Out (2026-10-24); Ben Sheppard Out (2026-10-23); Johnny Furphy Out (2026-02-09) | - | Pascal Siakam sat 58% of 12 2nd-nights vs 11% other; Andrew Nembhard sat 58% of 12 2nd-nights vs 17% other; Aaron Nesmith sat 58% of 12 2nd-nights vs 34% other |
| IND@MIL | MIL | 3 | 1 (@ PHI 10-24) | no | 1117 / -1h | 10-28 (no) | none | Ousmane Dieng (18 mpg) | - |
| PHX@OKC | PHX | 2 | 1 (vs GS 10-24) | no | 1352 / +2h | 10-30 (no) | Oso Ighodaro Day-To-Day (2026-10-25); Mark Williams Out (2026-09-11) | - | - |
| PHX@OKC | OKC | 3 | 0 (vs LAC 10-25) | YES | 0 / +0h | 10-28 (no) | Jalen Williams Out (2026-10-26); Jaylin Williams Day-To-Day (2026-10-26); Cason Wallace Day-To-Day (2026-10-26); Otega Oweh Out (2026-10-23); Nikola Topic Out (2026-10-21); Thomas Sorber Out (2026-10-21) | - | - |
| ATL@HOU | ATL | 3 | 1 (vs HOU 10-24) | no | 1128 / -1h | 10-28 (no) | Mouhamed Gueye Out (2026-10-19); Henri Veesaar Out (2026-09-21) | - | - |
| ATL@HOU | HOU | 3 | 1 (@ ATL 10-24) | no | 1128 / -1h | 10-28 (no) | Marcus Smart Out (2026-10-26); Steven Adams Out (2026-10-25) | Fred VanVleet (31 mpg) | - |
| MEM@UTAH | MEM | 3 | 0 (@ SAC 10-25) | YES | 856 / +1h | 10-29 (no) | Karim Lopez Day-To-Day (2026-10-26); Zach Edey Day-To-Day (2026-10-26) | - | Kris Murray sat 47% of 15 2nd-nights vs 26% other; Taylor Hendricks sat 47% of 19 2nd-nights vs 19% other; Cedric Coward sat 43% of 14 2nd-nights vs 19% other |
| MEM@UTAH | UTAH | 3 | 0 (vs LAL 10-25) | YES | 0 / +0h | 10-28 (no) | Trey Alexander Out (2026-10-24); Jusuf Nurkic Out (2026-10-07) | - | - |
| GS@DEN | GS | 3 | 1 (@ PHX 10-24) | no | 942 / +1h | 10-28 (no) | Gui Santos Day-To-Day (2026-10-25); Brandon Williams Day-To-Day (2026-10-25); Charles Bassey Day-To-Day (2026-10-25); Kristaps Porzingis Out (2026-10-21); Moses Moody Out (2026-10-16); Jimmy Butler III Out (2026-10-16) | Al Horford (18 mpg), Stephen Curry (26 mpg), Brandin Podziemski (33 mpg), Draymond Green (26 mpg) | - |
| GS@DEN | DEN | 2 | 1 (vs NO 10-24) | no | 0 / +0h | 10-28 (no) | Alpha Diallo Day-To-Day (2026-10-25); Cameron Johnson Day-To-Day (2026-10-25); Spencer Jones Out (2026-10-25); Julian Strawther Out (2026-10-17) | - | - |

## 4. Early-season noise (3P% shrunk with 750 attempts of prior)

| Team | GP | 3P% (prev / shrunk) | 3PA/g | off luck pts/g | opp 3P% (prev / shrunk) | def luck pts/g | margin (prev) | opp quality (opp prev margin) | pace (prev) |
|---|---|---|---|---|---|---|---|---|---|
| MIN | 3 | 0.424 (0.370 / 0.377) | 39.3 | +5.5 | 0.236 (0.355 / 0.344) | -11.8 | +23.3 (+3.4) | +1.9 | 103.3 (103.0) |
| CLE | 2 | 0.343 (0.360 / 0.358) | 33.5 | -1.5 | 0.351 (0.372 / 0.359) | -0.9 | -2.0 (+4.1) | -5.1 | 110.5 (102.2) |
| WSH | 3 | 0.286 (0.359 / 0.350) | 32.7 | -6.3 | 0.288 (0.368 / 0.350) | -7.3 | +2.0 (-12.0) | -2.9 | 106.7 (103.8) |
| CHA | 3 | 0.296 (0.378 / 0.365) | 47.3 | -9.9 | 0.387 (0.359 / 0.363) | +2.7 | -11.7 (+4.8) | -1.7 | 102.8 (99.5) |
| CHI | 3 | 0.226 (0.356 / 0.338) | 41.3 | -13.9 | 0.355 (0.368 / 0.359) | -0.5 | -13.3 (-5.2) | -5.1 | 110.8 (104.5) |
| BOS | 2 | 0.395 (0.367 / 0.370) | 40.5 | +3.1 | 0.388 (0.358 / 0.362) | +3.0 | -2.0 (+7.7) | +7.3 | 97.4 (97.3) |
| DAL | 3 | 0.360 (0.344 / 0.346) | 33.3 | +1.4 | 0.345 (0.350 / 0.358) | -1.4 | -6.0 (-5.5) | +1.8 | 102.6 (104.5) |
| MIA | 3 | 0.303 (0.361 / 0.353) | 39.7 | -6.1 | 0.420 (0.358 / 0.369) | +6.7 | -15.7 (+2.3) | +5.4 | 105.7 (105.5) |
| IND | 3 | 0.352 (0.356 / 0.355) | 35.0 | -0.3 | 0.359 (0.356 / 0.360) | -0.0 | +4.7 (-8.0) | -6.7 | 105.2 (103.1) |
| MIL | 3 | 0.372 (0.387 / 0.385) | 40.3 | -1.6 | 0.316 (0.375 / 0.354) | -4.4 | -10.0 (-6.2) | -5.8 | 108.5 (99.5) |
| PHX | 2 | 0.290 (0.361 / 0.355) | 34.5 | -6.8 | 0.419 (0.347 / 0.366) | +6.8 | -16.0 (+1.5) | -0.4 | 100.7 (99.6) |
| OKC | 3 | 0.337 (0.365 / 0.362) | 32.7 | -2.4 | 0.314 (0.369 / 0.354) | -4.2 | +10.3 (+11.1) | +4.8 | 102.0 (101.9) |
| ATL | 3 | 0.346 (0.371 / 0.367) | 43.3 | -2.8 | 0.302 (0.355 / 0.351) | -6.3 | +15.7 (+2.4) | +3.6 | 104.9 (103.7) |
| HOU | 3 | 0.417 (0.364 / 0.370) | 32.0 | +4.5 | 0.290 (0.351 / 0.350) | -7.4 | +9.7 (+5.2) | +1.7 | 100.9 (99.3) |
| MEM | 3 | 0.381 (0.353 / 0.356) | 37.7 | +2.7 | 0.330 (0.364 / 0.356) | -2.7 | +11.3 (-6.0) | -6.3 | 101.3 (103.8) |
| UTAH | 3 | 0.333 (0.345 / 0.343) | 34.0 | -1.0 | 0.383 (0.372 / 0.363) | +2.2 | -9.3 (-8.4) | -2.9 | 103.5 (105.1) |
| GS | 3 | 0.336 (0.356 / 0.353) | 38.7 | -2.0 | 0.385 (0.360 / 0.363) | +2.3 | -11.3 (-0.6) | -0.9 | 101.3 (101.7) |
| DEN | 2 | 0.299 (0.396 / 0.387) | 38.5 | -10.2 | 0.272 (0.349 / 0.351) | -9.7 | -3.0 (+5.1) | +3.3 | 105.3 (101.5) |

Game luck = 0.5 x (both offences' + both defences' luck pts/g): points of a recent-form total that come from 3P shooting expected to regress (+ = recent form overstates scoring).

## 5-6. Structure, refs, model cross-check

| Game | game luck pts | paint diff z (H-A share allowed) -> ATS pts to home | proj pace | altitude | crew (chief, ref, ump) | crew FTA / total-vs-close (shrunk, DEV) | model total -2.3 vs Pin total | model margin vs Pin spread |
|---|---|---|---|---|---|---|---|---|
| MIN@CLE | -4.4 | -1.85 -> +1.0 | 103.9 | - | Sean Wright, Bill Kennedy, Tyler Ricks | +0.6 / -0.0 | 235.1 vs 234.0 (+1.1) | +4.0 vs +4.5 (-0.5) |
| WSH@CHA | -10.4 | -0.14 -> +0.1 | 102.6 | - | Ed Malloy, Eric Dalen, Biniam Maru | -0.8 / -0.0 | 229.9 vs 228.5 (+1.4) | +5.7 vs +2.5 (+3.2) |
| CHI@BOS | -4.1 | -2.11 -> +1.1 | 101.9 | - | Zach Zarba, Jacyn Goble, Klajdi Mulla | -0.5 / +1.5 | 230.8 vs 232.5 (-1.7) | +12.3 vs +14.0 (-1.7) |
| DAL@MIA | +0.4 | -1.48 -> +0.8 | 104.7 | - | Josh Tiven, Kevin Scott, Chelisa Painter | -1.0 / +0.3 | 236.6 vs 240.0 (-3.4) | +7.7 vs +8.0 (-0.3) |
| IND@MIL | -3.2 | -1.83 -> +1.0 | 103.0 | - | Marc Davis, Gediminas Petraitis, Julian Scott | -0.4 / +0.9 | 233.4 vs 239.0 (-5.6) | +1.9 vs -4.0 (+5.9) |
| PHX@OKC | -3.3 | -1.31 -> +0.7 | 100.9 | - | Tony Brothers, Eric Lewis, John Conley | +1.2 / +0.8 | 220.9 vs 221.5 (-0.6) | +7.9 vs +9.5 (-1.6) |
| ATL@HOU | -6.0 | +0.29 -> -0.2 | 101.9 | - | Ray Acosta, Brent Barnaky, Mousa Dagher | -0.3 / -0.5 | 231.4 vs 229.5 (+1.9) | -1.1 vs +5.0 (-6.0) |
| MEM@UTAH | +0.6 | -0.75 -> +0.4 | 103.8 | YES (+1.0 home ATS lead) | Scott Foster, Karl Lane, Nate Cearley | +0.2 / +1.3 | 236.3 vs 238.5 (-2.2) | -1.2 vs +1.5 (-2.7) |
| GS@DEN | -9.7 | -0.19 -> +0.1 | 102.1 | YES (+1.0 home ATS lead) | Tre Maddox, Marat Kogut, JT Orr | +0.4 / +0.0 | 233.7 vs 236.5 (-2.8) | +10.0 vs +7.0 (+3.0) |

Refs: posted (official.nba.com). Crew numbers = mean of the three refs' DEV residuals shrunk with 60 games of prior; DEV crew betting lost -1.9% (notes/nba_referees.md) -> context only.
