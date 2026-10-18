# Results — every strategy tried, with verdict (newest first)

| date | strategy | data | result | verdict |
|---|---|---|---|---|
| 10-18 | SOCCER price shopping vs Pinnacle (football-data.co.uk, 22 divs) | dev 2012-19, 52,177 matches | S1 Kaunitz -3.4%/873; S2 best-price +3.8%/14,905 p.018 (Max has palps); S3 B365 -0.8%/3,092; S4 fav<=1.30 at close +1.8%/2,333 p.03; S6 WH CLV +3.97% WITHDRAWN (proportional de-vig overstates longshots: hit/fair .80 at 10-20, .67 at 20+). POWER de-vig, price<10: S7 B365 CLV +2.57% p<.001 ROI +9.8%/1,623 (6/7 seasons); S8 WH CLV +2.81% p<.001 ROI +10.4%/910 | S7/S8 PASS dev bar (.0063) on CLV -- but power method chosen after dev calibration -> sealed 2019-26 must confirm. Classic soft-vs-sharp +EV: books limit winners; B365 not legal in CA |
| 10-18 | Kalshi ladder MAKER, symmetric (YES bid / NO bid on rungs 35-65c, trade-through fill) | 2025 dev, 3,810 events, 15,773 rungs | YES -2.6%/2,796 fills p.82; NO +2.7%/3,285 p.14 | resting orders do NOT earn the spread net of adverse selection -> weakens maker_dog2050 (its gain was mostly spread capture) |
| 10-18 | Kalshi MAKER: rest bid at T-3h, fill = trade-through by T-1h, maker fee | Kalshi winners 2025 dev, 5,029 events | MK1 dog 20-50c +9.8%/898 fills p.009 3/3q (bar .0063); MK2 fav 50-80c -5.4%/893. Robust: verified +11.3%, looser fill +9.4%/2,895, always-filled +7.6%; taker on same set +1.3-3.7% | NOT past bar; band chosen after seeing dev calibration (partly post-hoc); league-inconsistent (MLB/NBA +, NFL/NHL -); mostly spread capture; contradicts paper (favs >50c should be +). Sealed candidate |
| 10-18 | Kalshi sports favourite-longshot, 1 h pre-start, taker fees (6 trials, G4 post-hoc) | Kalshi 2025-04..12, 9,312 events | G1 fav>=85% +2.0%/383 p.062 3/3q; G2 fav65-85 -5.6%/1,346; G3 dog<=15% -45%/367; ladder tails +0.5%/377, +0.4%/88; G4 dog20-50 (post-hoc) +1.4%/4,429 p.21 | none past bar .0083. Longshots <10c badly overpriced (paper replicated); ladders well calibrated. Sealed (2026, ~140k rows) BLOCKED on start-time join |
| 10-17 | MLB open->close ML move >=4pp: follow vs fade, at close | SBRO 2010-18 | follow -3.0%/3,308; fade -2.3%/3,308 | NO EDGE: close efficient both ways |
| 10-17 | NBA decorrelated models: Elo disagree >=0.10 ML; walk-forward logit edge >=3% | NBA 2007-19 | +0.6%/3,916 p.40; -4.2%/744 | NO EDGE (Brier mkt .199 < Elo .207) |
| 10-17 | NBA angles x5: rest edge ATS, big-spread dog, heavy fav ML, b2b unders, home dog ML | NBA 2007-19, 15,519 g (kyleskom SBR sqlite) | -1.5%/4361; +0.6%/1218; -2.7%/4239; -3.7%/5469; -3.9%/4766 | NO EDGE. Caught: file Spread is UNSIGNED (fav line) -> pre-fix big-dog +7.2% p=.002 was an artifact; fixed via ML favourite |
| 10-17 | MLB fav >=60% at OPEN vs CLOSE (FLB-at-open thesis) | SBRO 2010-18, 22,140 games | open -2.70%/8,768, close -2.59%/8,768, 0/9 seasons each | NO EDGE: favourites lose ~the vig at both; no open-line FLB |
| 10-17 | NFL online angles x5: big_fav_fade, dome_team_cold, primetime_unders, elo_disagree_ml, heavy_fav_ml | NFL dev 2009-19 | -2.7%/383; +3.1%/33; +1.4%/542; -0.9%/1557; -1.1%/833 (all p>.36) | NO EDGE: popular angles priced in |
| 10-17 | stocks: sector momentum 12-1, weekly sector reversal, SPY vol-target 10% | ETFs dev 2006-18 (14 trials) | +0.6%/yr ns; -0.6%/yr; vol-target alpha +2.7%/yr p=.024 (bar .0036), maxDD -25% vs -55%, CAGR 6.4 vs 7.5% | NO EDGE; vol-target = risk control only |
| 10-16 | NFL location: west_early_east (Pacific team at ET venue <2pm ET, bet home) + den_altitude_home | nfl_spread dev 2009-19 | -10.6%/110 p=.86; -7.4%/94 p=.76 | NO EDGE; circadian mirror fails -> weakens west_coast_night |
| 10-16 | 2-leg cross-game parlays of high-prob legs (dec<=1.50), 2 trials | props non-sealed + NFL ML dev 2006-19 | props -9.3%/105k (legs -4.7%, compounding predicts -9.2%); NFL ML -1.35%/885 ns | NO EDGE: parlays square the legs' edge |
| 10-15 | stocks dev: dip/rip, tech->sectors gate, learned lead-lag graph, 2% daily loss cap (11 trials) | ETFs+mega-tech 2006-18 | 0/11; tech gate -6.3%/yr t=-2.6; best dip/rip QQQ 5% +2.1%/yr p=.05 | no edge; loss cap = risk control only |
| 10-15 | pattern-mine SEALED confirm (10 survivors, run once) | frozen 2026-10-14 sealed seasons | 0/10 pass (bar p<.005 & ROI>0). mlb hits over/down: dev +36% -> sealed -5.4%/3,655 (t=-4.4, artifact confirmed, reversed). NHL blocked-shots under +2.8%/223 t=0.45; NHL mid-line under -2.3%/839; NHL saves under -9.4%/158; MLB K under/up +0.6%/646; NBA DK under: 0 sealed bets (UNTESTABLE) | FAILED. Pre-registered: no Class-B pass -> Odds API only buys a test of the DK under; user's call. Note: script prints two-sided p (header says one-sided); verdict unaffected |
| 10-14 | backwards pattern mine (2,280 cells) | frozen props, non-sealed | 10 survivors; 5 line-move artifacts; NBA under high-line @DK +1.9%/54k; NHL unders +6.7-13% | sealed confirm 10-15: 0/10 |
| 10-14 | multi-book consensus deviation | NBA 2023 dev | +7.4%/23k, MGM +36% | stale-quote illusion; 2024 sealed untouched |
| 10-12 | Pinnacle vs Kalshi live | live | 0 flags >=2% EV; Kalshi within 1-2c | running, paper |
| 10-12 | 252-trial NFL prop model loop | NFL 2024 weekly WF | 0 at p<.05, median -7.5% | no edge |
| 10-12 | FG props (defense style + script) | NFL 2024 | -8.2u/53 | market wins |
| 10-12 | defense zone/blitz -> FGs allowed | 2022-25 | r=0.00 | no relation |
| 10-10 | edge report (shrinkage) | NFL dev | candidates shrink to -3..+4% | unproven |
| 10-09 | MLB bullpen/starter rest/day-after-night | 12,808 g | 0/21 BH | priced in |
| 10-09 | Kalshi-Polymarket arb | live | <1c | not worth it |
| 10-09 | NFL scan 72 tests | dev 2006-19 | div unders, wind unders survive | lockbox pending |
| 10-09 | west_coast_night | NFL dev | +21.5%/107 p.007 | lockbox pending |
| 10-09 | div_unders_late (post-hoc wk>=10) | NFL dev | +9.8%/623 | lockbox pending |
| 10-09 | rain+wind unders (oracle, observed wx) | NFL dev | ~~+32.7%/70~~ BUG (wx joined to wrong games, 53%) -> corrected +21.4%/70 p.033 | oracle only |
| 10-16 | pre_rain_unders (wx 3h pre-kickoff, causal) | NFL dev 06-19 | ~~+12.4%/217~~ BUG -> corrected +8.0%/219 p.11, 11/14 seasons | below bar .0083; forward paper (live_signals.py) |
| 10-16 | pre_rain_wind_unders (causal) | NFL dev 06-19 | ~~+16.0%/67~~ BUG -> corrected -10.2%/67 | dead |
| 10-16 | pre_rain_unders SEALED (user yes) | NFL 2020-25 | +13.3%/101 p.074 | INVALID: scored on misaligned wx. Not re-scored (needs user yes; corrected dev no longer earns it) |
| 10-08 | Kalshi temp consensus-reject | lockbox | +1.45%/1,844 | PASSED — only real strategy |
| earlier | MLB strikeout props model | 2025 lines | +1.1u/1,970 | break-even |

Research takeaways (reports/Sports betting edge research.md): no published lasting edge on US closing lines;
graph methods (GNN/BN/causal discovery) have no market-tested win; calibration + decorrelation from the book +
CLV evaluation are what work; t > sqrt(2 ln N) for N trials.

### 2026-10-16 bug: NFL game weather joined by row position
`nfl.from_raw()` re-sorts rows; nfl_game_weather.py attached `stadium_id` with `.to_numpy()` (by position), so
53% of games got another stadium's weather. Every rain/wind number before 2026-10-16 was wrong (oracle +32.7% was
really +21.4%; causal rain +12.4% -> +8.0%; rain+wind +16% -> -10.2%). Fixed by merging on game_id
(validate="1:1"); rebuilt data/nfl_game_wx.parquet, 100% stadium match. Corrected oracle: rain is -1.9 pts beyond
the close (se 0.86), wind -0.24 pts/mph (se 0.07) -- the market under-reacts to weather, but rain visible 3 h before
kickoff captures only part of it. Lesson: join on keys; assert alignment after any transform.
