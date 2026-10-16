# Results — every strategy tried, with verdict (newest first)

| date | strategy | data | result | verdict |
|---|---|---|---|---|
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
