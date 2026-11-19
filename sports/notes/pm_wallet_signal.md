# pm_wallet_signal -- do persistently skilled Polymarket wallets lead Kalshi on the same game? (2026-11-18)

Lead: notes/forum_strategy_scout.md idea #3. Script: `scripts/pm_wallet_signal.py`. Data: `data/pm_wallet_signal/`
(gitignored), small outputs `results/pm_wallet_signal/`.

## Priors (respected, not re-tested)
- ~70% of PM wallets lose; top-wallet income is often rebates / non-trading (blockrotate 2026-09-13, scout C9).
  Mitigation here: skill measured on TAKER fills only (data-api v2 `taker_only=true` default), held to resolution,
  so rebates and maker spread capture never enter the score.
- Cross-venue lag tests DEAD: xvenue_ingame.md (contemporaneous, Kalshi trades first by ~5-10 s), pmus_lag.md.
- Information-timing lesson: news is priced before official releases -> a "skilled" wallet may just be fast, and
  fast is already in the Kalshi price. The kill rules below test exactly that.

## Scope decision (before data)
Sports game winners only (NBA, NHL, MLB, NFL, CBB, CFB), using the 12,176 games already matched uniquely by team
side in `data/xvenue_ingame/match.parquet` (scripts/xvenue_ingame.py match). Reason: it is the only category with
(a) hundreds of independent events per month and (b) careful, already-audited side matching with equivalent rules
(both venues settle on the game winner incl. OT). Politics/econ/crypto/weather would need fresh rule-by-rule
matching (L18 lesson) and give few independent events (Fed: 8/yr). Not covered here -> stated as a limit.

## Pre-registration (written 2026-11-18 BEFORE any Polymarket trade or Kalshi price of any split was pulled;
## only one probe game's trade count was looked at: nba-bos-tor-2025-10-10, 167 fills >= $100)
Splits by game date (slug date, UTC):
- train: 2025-04-01 .. 2025-12-31 -- wallet skill only (resolutions of train games only).
- dev:   2026-01-01 .. 2026-04-30 -- event study + edgelab dev evaluation.
- sealed: 2026-05-01 .. 2026-11-05 -- pulled only after a dev pass (edgelab, once). Skill for sealed is re-estimated
  with the same definition on all games dated < 2026-05-01 (all resolved before sealed starts).
- forward: 2026-11-19 .. (not built).

PM data: data-api v2 /trades, condition=<cond>, taker_only default, filter_type=CASH, filter_amount=100.
Resolution: Kalshi market result of the matched team ticker (cross-checked with PM last prices on a sample).

Skilled wallet (computed on the skill window only):
- fills: cash >= $100, price in [0.02, 0.98], timestamp < Kalshi close_time (proxy for game end).
- direction on the matched team: BUY team token or SELL opp token = bull; else bear.
- per fill pnl: BUY token: size*(v - p), risk size*p; SELL token: size*(p - v), risk size*(1 - p); v = token payout.
- per wallet: aggregate by game; ROI = sum pnl / sum risk; se = sqrt(sum_g (pnl_g - ROI*risk_g)^2) / sum risk.
- eligible: >= 20 games, >= $5,000 total risk, two-sided share (games with both bull and bear fills) <= 0.5.
- skilled: ROI > 0 and ROI / se >= 2.5.

Event (dev): a skilled wallet's single taker fill with cash >= $500 in a dev game; first such fill per (game,
direction) per 10 min; Kalshi side mid at the last 1-min candle <= t in [0.05, 0.95]; t <= Kalshi close - 120 s.
Kalshi data: 1-min candlesticks of the matched team ticker (yes_bid / yes_ask close), window t-30 min .. t+125 min.
Trade: side = bull -> buy YES team ticker at yes_ask; bear -> buy NO at 1 - yes_bid; quote = close of the first
1-min candle ending after t (0-60 s after the PM block time; "within 60 s"); ask in [0.03, 0.97]; + Kalshi taker
fee 0.07 p(1-p) rounded up to the cent per 1-contract order ($1-ish stake), hold to settlement.
CLV mark = side mid at the candle closest before min(t + 120 min, Kalshi close - 60 s).
Trial `pmw_t1` (family cross_venue; bar 0.05 / (trials ever in family) at evaluation): edgelab rule 5, min_n 200.
Also reported (same rows, not extra trials): mean CLV with game-clustered bootstrap CI; ROI CI.
Descriptive path: Kalshi side-mid change at -30, -5, -1, +1, +5, +30, +120 min relative to t (sign = trade direction).
KILL (any one, even if ROI passes):
  K1 already moved: mean side-mid change over [t-5 min, t] >= mean change over [t, t+5 min] (Kalshi moved first).
  K2 reprices within 60 s: mean change t -> entry quote time >= 0.8 x mean change t -> t+5 min.
  K3 dev rule-5 fail (p >= bar, a half <= 0, concentration > 30%, CLV < 0, n < 200).
Descriptive robustness (not trials): cash >= $100 and >= $2,000; skilled z >= 2 vs 3.

## Log
- 11-18: probe -- data-api v2 /trades supports filter_type=CASH&filter_amount, condition (<=20 ids), taker_only.
  nba-bos-tor-2025-10-10: 1000+ fills total, 167 fills >= $100, 56 >= $500 (results/pm_wallet_signal/probe.py).
- 11-18: BUG caught in my puller before any analysis: data-api v2 cursor pages return 0 rows unless the original
  params are re-sent with the cursor (probe2.py: cursor-only 0 rows vs cursor+params 718). First pull capped every
  20-game batch at 1,000 fills -> discarded and re-pulled.
- 11-18: registered `pmw_t1_taker-51422090` (cross_venue now 12 trials, bar p < 4.17e-03) + vault dataset
  `pm_wallet_signal` + loader + rule edgelab/rules/pm_wallet.py, BEFORE any dev Kalshi price was pulled.
- 11-18 data: train 1,005,065 taker fills >= $100 in 4,907 games, 48,791 wallets (trades_train.parquet; probe game
  count 167 reproduced); dev 1,337,723 fills in 3,982 games. Kalshi results 5,169 no / 3,753 yes (results.log).
  Resolution sanity: train fills at price >= .97 -> that token won 99.3% (n 93,085).
- 11-18 skill (results/pm_wallet_signal/skill_dev.log): 46,785 wallets with resolved train fills, 2,681 eligible.
  Eligible median ROI -1.65%, pooled -0.23%. Skilled counts vs one-sided normal chance expectation: z>=2: 93 vs 61;
  z>=2.5: 53 vs 17; z>=3: 31 vs 4 -> more tail than chance (heavy-tailed returns make the normal count optimistic).
  Several "skilled" wallets have tiny se (ROI 2-4%, se .002-.003) = buying near-certain tokens late; others ROI .5-1.0
  on few fills = likely in-game speed traders. 33 of the 53 trade in dev; 5,204 fills >= $500, 2,743 events after
  dedupe in 1,357 games.
- 11-18: first `edgelab evaluate` refused: 12 matched Kalshi tickers settled 'scalar' (no value column). Added hidden
  `sv` (YES settlement value, from Kalshi settlement_value_dollars) to results.parquet + vault value_col; rule and
  splits unchanged. Note: the skill window simply drops scalar games (v_team NaN).

## Results (dev 2026-01-01..04-30 only; sealed never pulled)
Source: results/pm_wallet_signal/describe_dev.log (script describe; 1 contract per bet, game-clustered bootstrap) and
`edgelab status` (pmw_t1_taker, $1 stake per bet, edgelab's own CI).
- Events 2,731 -> bets 2,276 in 1,162 games / 119 days, 29 wallets (nba 1,426, cbb 350, nhl 336, nfl 95, mlb 43, cfb 26).
- Kalshi side-mid path vs PM fill time t (cents, sign = wallet's direction): -30m -4.16, -5m -1.69, -1m -0.81,
  entry(<=60 s) +0.64, +1m +0.64, +5m +0.40 (CI [-0.07, +0.90]), +30m -0.46, +120m -2.10.
  Reading: skilled wallets BUY AFTER their side has fallen (contrarian / in-game dip buying); Kalshi bounces +0.6c by
  the time we could fill and gives nothing after that.
- K1 (Kalshi moved first): pre -1.69c vs post +0.40c -> ok (it moved AGAINST the wallet's side beforehand).
- K2 (reprices within 60 s): entry +0.64c >= 0.8 x post5 (0.32c) -> **KILL**. All of the +5 min move is in the price
  by the first candle we could hit.
- ROI held to settlement: -2.36% [-5.19%, +0.32%] (n 2,275) -> not distinguishable from 0, point negative.
  CLV vs side mid at min(t+120m, close-60s): -5.10c [-6.27, -3.88].
- edgelab verbatim: `pmw_t1_taker cross_venue (12, 4.2e-03) dead dev 2276 -4.42% [-8.21%,-0.61%] p 9.9e-01 CLV -0.0332
  dead (dev fail): 5a p>=bar,5b half<=0,5c conc>30%,5e CLV<0/null`. (Difference vs -2.36% = $1 stake per bet vs one
  contract per bet; both negative.)
- Robustness (pre-listed, not trials): fills >= $2,000: n 476, ROI -0.17% [-6.06, +6.16], CLV -6.98c [-9.25,-4.54],
  K2 KILL (describe_dev_c2000.log). z >= 3 wallets: n 1,279, ROI -2.09% [-5.53, +1.26], CLV -5.97c, K2 KILL
  (describe_dev_z3.log). $100 / z>=2 not run: more pulls, and K2 fires on every variant run.
- Skill persistence on Polymarket itself (diagnostic, inline python on trades_dev + wallets_for_dev): the 33
  train-skilled wallets' own dev fills (held to resolution) pooled ROI +0.75%, game-bootstrap CI [-9.24%, +11.08%]
  over 1,765 games; median wallet -6.05%. "Skill" did not persist distinguishably out of sample even on PM.
- Scalar settlements: 12 matched Kalshi game markets settled 'scalar' (NFL preseason ties at 0.50; postponed CBB/NBA
  at fair values 0.01-0.72). Resolution-mismatch risk vs Polymarket (L18 lesson): PM resolves postponements its own way.
VERDICT: DEAD at dev (edgelab rule 5 fail + pre-registered K2). No sealed pull. Family cross_venue now 12 trials.

## Limits
Sports game winners only (politics / econ / crypto / weather not matched: would need rule-by-rule matching and give
few independent events). Taker fills >= $100 only (maker-side skill ignored by design). Kalshi quotes from 1-min candle
closes (entry 0-60 s after PM block time; block time itself lags the off-chain match by a few seconds).

## Learnings
### Polymarket "skilled" wallets on game winners are contrarian dip buyers, and Kalshi reprices within 60 s
Side mid fell 1.69c in the 5 min before a skilled wallet's >= $500 fill, rose 0.64c by the first Kalshi candle we
could hit, and only 0.40c by +5 min (results/pm_wallet_signal/describe_dev.log). Following them on Kalshi at the ask:
ROI -2.36% [-5.19, +0.32], CLV -5.10c. Wallet-flow copying is not an information edge across venues.
### Train-period wallet skill does not persist out of sample on Polymarket sports
33 wallets with z >= 2.5 on 2025 games: dev pooled ROI +0.75% [-9.2%, +11.1%], median -6%. 53 "skilled" vs ~17
expected by chance from 2,681 eligible; the excess is mostly late near-certain buyers (tiny se) and few-fill streaks,
consistent with the survivorship prior (~70% of PM wallets lose).
### Polymarket data-api v2 cursor pages need the original params re-sent
`/v2/trades?cursor=X` alone returns 0 rows; send condition/filter params with the cursor (results/pm_wallet_signal/
probe2.py). Otherwise silent truncation at 1,000 rows. filter_type=CASH&filter_amount=100 plus up to 20 condition ids
per call make a full season pull take ~5 min.
### Kalshi game-winner markets settle 'scalar' on postponements at fair values, not only 0.50
12 of 8,934 matched tickers (e.g. KXNCAAMBGAME-25DEC15EWUUSI-EWU at 0.01, KXNBAGAME-26JAN08MIACHI-MIA at 0.69; NFL
preseason ties 0.50). Any cross-venue game study must carry the settlement value (edgelab value_col).
