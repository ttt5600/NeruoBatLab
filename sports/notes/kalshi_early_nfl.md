# Kalshi early NFL prices vs the simple model (DEV 2025 wk 1-12), 2026-10-25

Script: `scripts/kalshi_early_nfl.py` (`build`, `dev [--ledger]`); pre-registration is the docstring, written before
any dev-game price was pulled. Results JSON `data/lab/kalshi_early_nfl_results.json`; per-game bets
`data/lab/kalshi_early_nfl_bets_{as_of_listing,std_week_features}.parquet`. Ledger: 3 rows, dataset `kalshi_early_nfl`.

## Verdict

**Kalshi's first prices are stale, and the legacy model knows which way they will move. But a resting maker order
does not collect that move, because it fills mainly when the market moves against it.**
- **T1 CLV passes.** In the 109 of 178 dev games where |model - first mid| >= 4c, the mid moved +3.3c toward the
  model by kickoff (95% CI +1.6 to +5.1, p < .0002; bar .0167). It is positive in weeks 1-6 (+3.2c) and in weeks 7-12 (+3.5c).
- **T2, a maker order at the first bid, fails.** 69 of 109 orders filled. ROI to settlement was +16%, CI -8% to +40%,
  p .09. This trial is underpowered. A real effect would need ROI above about 30% to be detected.
- **T3, a taker order at the first ask, fails.** ROI -4.5%, CI -21% to +12%, p .69.
- **The mechanism is adverse selection.** In games that did not fill, the market moved toward the model by +8.0c,
  CI +5.5 to +10.7. In games that filled, CLV was only +0.6c, CI -1.4 to +2.6. Measured against the last pre-kick
  mid, a filled order gains +2.2c per contract. That is about half the 4c spread, i.e. roughly the spread itself and
  not the model's information. The positive settlement ROI in T2 is mostly settlement noise: filled bets won 52% at a
  mean price of 44.5c, n = 69.
- **Bottom line: this is not a maker edge yet.** The tradable form of the result would be to take at the first ask
  and exit before kickoff, or to post near the mid and accept fewer fills. Neither was pre-registered here. Any version
  of it would be a new trial on new data, meaning forward paper or the sealed set with the user's yes.

## Pre-registered trials (as-of-listing features; bar p < .05/3 = .0167; one-sided game bootstrap)

| trial | n | estimate | 95% CI | p | wk1-6 / wk7-12 | pass |
|---|---|---|---|---|---|---|
| T1 CLV (prob units) | 109 | **+0.033** | [+0.016, +0.051] | <.0002 | +0.032 (62) / +0.035 (47) | YES |
| T2 maker @ first bid, ROI | 69 filled / 109 | +16.2% | [-7.9%, +40.3%] | .091 | +9.2% / +27.6% | no |
| T3 taker @ first ask + fee, ROI | 109 | -4.5% | [-21.0%, +12.4%] | .69 | -1.9% / -8.2% | no |

Model: legacy model B spread fit on 2007-2019, Phi(margin / 13.62). Features are as of the Kalshi listing time.
For 89% of games that means a week older than the usual pre-game features.

## Uncounted diagnostics (after the trials)
- **Usual "as of that week" features.** These include a week of results that came after the listing, so they are not tradable.
  T1 +5.0c [+3.2, +6.8]. corr(gap, move) is .62, against .42 for the as-of-listing features. Maker ROI -5.7% (56 fills),
  taker -10.2%. More information gives a better forecast of the line move but no better money result.
- **Controls.** A constant model (p = .55/.57/.60) gets CLV -0.1c, -0.1c and +0.2c. Fading toward 50c gets -0.3c. So a
  noisy first mid alone does not produce the CLV. Blind drift of the home mid is +1.2c, CI -0.1 to +2.5. The model
  bets home 62% of the time. Home-side CLV is +4.3c and away-side CLV is +1.6c (p .095), so a small part of T1 may be home drift.
- **By gap size.** 4-8c: +1.7c (n 49). 8-12c: +1.2c (32). >= 12c: **+8.5c** (28). Regressing the move on the gap gives
  a slope of 0.42. Excluding weeks 1-2 (listed in May) gives +3.2c (n 90, p .0004).
- **Maker at first ask - 1c.** Fills 88% of the time. ROI -5.1%.
- **Against the close** (de-vigged closing moneyline, nfl_games). MAE of the first mid is 6.0c, of the last pre-kick mid 1.2c,
  and of the model 7.3c. Brier scores: first .208, last .204, close .204, model .210. The model is the least accurate
  of these, but it is right about the direction of the first price's error.
- **No Pinnacle or consensus opener comparison.** No 2025 opener outside the lockbox exists locally. The ESPN 2025 openers were not used.

## How Kalshi lists NFL games (2025 dev)
- **Lead time.** Weeks 1-2 were listed on 2025-05-20, 110 and 76 days before kickoff. Weeks 3-12 were listed a median
  of 8.6-13 days before kickoff, i.e. about 1.5 weeks ahead, usually before the previous week's games were played.
- **Spread at listing.** The first two-sided quote arrives within ~1 h, but its spread is a median 59c (placeholder
  quotes). The first quote with spread <= 10c comes a median 1.9 h after listing (mean 50 h; up to 41 days for the
  May listings). At that point the spread is a median 4c (mean 4.7c).
- **Depth.** The historical API has no order book, so depth at listing is unknown. Volume is the only liquidity
  proxy: in the first 24 h after listing, a median 74 contracts per market (IQR 1-484). Total pre-game volume is a
  median 1.24M contracts per market. Early markets are very thin, so the size available at the first quote is
  probably small, and the T2 fill assumption may be optimistic for more than a few contracts.

## Caveats
- **Hourly candles.** These hide intra-hour quotes. The fill rule requires a trade strictly below our bid in a later
  hour and ignores queue position. Orders are cancelled at kickoff.
- **The QB term in as-of-listing features** uses the starter of the team's next game after listing (from the
  schedule). That is a projected-starter proxy with a small amount of look-ahead.
- **Spreads and totals ladders** (KXNFLSPREAD and KXNFLTOTAL exist for 2025) were not tested.
- **Fees.** Maker fee 0.0175·p(1-p) applied everywhere (repo convention); taker fee 0.07·p(1-p); per-contract rounding ignored.
- **SEALED data was not read.** 2025 weeks 13+, POST and 2026 were never fetched or scored. 2025 pbp and schedule rows
  for week >= 13 were dropped before features were built.
