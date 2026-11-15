# maker_fairfill -- rest a bid on the MODEL's side at Kalshi's 50c rung instead of paying the ask (agent, 2026-11-15)

## Question
The opener models carry ~0.8-0.9 pts of CLV vs Kalshi's own 50c strike (K50: NBA +0.88, CFB +0.82; notes/k50_strike.md)
but taker friction (ask + fee, cost ~0.53) needs ~1.2-1.4 pts. A maker bid at/below mid saves ~2-3c. The known danger
(E5, RD2, first-listing, L35, K-FAV): resting bids fill mostly when about to lose. Here the model chooses the side.
Does a model-side resting bid fill on a representative subset (no adverse selection) and earn net of maker fee?

## Facts gathered before pre-registering (no outcomes)
- Kalshi series fee_type (GET /series/{s}, 2026-11-15): KXNBATOTAL, KXNBASPREAD, KXNCAAFTOTAL, KXNFLTOTAL,
  KXNFLSPREAD, KXNCAAMBTOTAL all `quadratic_with_maker_fees` x1 -> maker fee 0.0175*p*(1-p) (0.44c at 50c, 100-ct
  order, rounded up to the cent per order) vs taker 0.07*p*(1-p) (1.75c).
- Public trade tape GET /markets/trades?ticker&min_ts&max_ts works on these (Nov 2026) markets: yes/no price, count,
  taker_side, created_time.
- Seen already (disclosed): K50 taker outcomes per league (notes/k50_strike.md table) and CBB T1 taker outcomes
  (notes/cbb_opener_model.md). This test reuses the SAME bets, so dev is not independent of K50/T1. NOT computed: any
  fill, any filled-vs-unfilled split, any maker P&L.

## Pre-registration (written 2026-11-15 ~09:30Z, before any trade tape was fetched)
Universe = the primary taker bets already defined (side and strike fixed by the model, NOT by the tape):
- NBA: results/k50_strike/rows_nba.csv rule k50_k3.0 primary (n 25). Start = q1_timing.csv `start`.
- CFB: rows_cfb.csv rule k50_k3.5 primary (n 71). Start = kalshi_openers_xleague/q1_cfb_rows.csv `occ`.
- NFL: rows_nfl.csv rules k50_k2.5/k50_k2.0 primary (n 17). Start = q1_nfl_rows.csv `occ`.
- CBB: results/cbb_opener_model/kalshi_rows.csv bet == True (n 67). Start = `tip`. Ticker = event-ceil(S).
Order: at the decision time T (the same T / quote as the taker bet), post a bid on the model's side of the bought
ticker (YES bid if the pick is the YES side, else NO bid) at b_x = floor_to_cent(side_mid - x), side_mid = side ask -
spread/2, x in {0, 1, 2} cents; capped at side ask - 1c (post-only). Rest from T until start - 5 min, then cancel.
FILL (primary) = at least one trade on the tape in (T, start - 5 min] printed STRICTLY through b on our side
(YES bid: yes_price < b; NO bid: no_price < b). Price-time priority means every order at b was exhausted before a
print below b, so a strict print implies our order would have been hit. Size not modelled in the primary.
Cost = b + maker fee (ceil per 100-ct order). Graded with the existing `won` column. ROI = (1-cost)/cost or -1.
Trials (edgelab, rule forward:unwired, price_model maker_tape): one per league family x x = 12 trials:
mf_{nba,cfb,nfl,cbb}_x{0,1,2}. Families nba_totals, cfb_lines, nfl_lines, cbb_totals.
Stats: mean with 95% CI from a bootstrap clustered by ET game date, 4000 resamples, seed 0; one-sided p = share <= 0.
Reported per league x x: n bets, fill rate, filled W, filled ROI net of maker fee, win(filled) - win(unfilled) [CI]
(adverse-selection gap), CLV of filled, capacity (contracts printed strictly through b in the window; median, sum).
Verdict per league x x (pre-registered):
- PASS dev (-> sealed / real-money ask) only if filled ROI p < edgelab family bar AND n_filled >= 15 AND filled CLV > 0.
- GO forward paper if filled ROI > 0 (point) and n_filled >= 15.
- KILL if filled ROI CI upper < 0, OR (AS gap CI upper < 0 AND filled ROI < 0 point).
- Otherwise NEEDS DATA.
Mechanism readout (descriptive, the main question): pooled all leagues at x = 1: AS gap [CI]. If its CI upper < 0, the
model's direction does NOT protect a resting bid from adverse selection -> maker lever dead for these models.
Sensitivities (not trials): touch fill (print <= b); full-size fill (>= 100 contracts strictly through b);
y = 30 min cancel.
Expectation if no adverse selection: cost ~0.494 at x=1 and the K50 CLV (~0.85 pt ~ +1.8c) -> ROI ~ +4-5%.

## Plan
results/maker_fairfill/fetch.py: pull tapes for the ~180 tickers (kx_throttle kget, cached per ticker JSON).
results/maker_fairfill/score.py: fills, ROI, AS gap, capacity -> rows.csv, summary.json.

## Results

### Registered before any tape was fetched (edgelab, forward:unwired, price_model maker_tape)
mf_nba_x0-cb422d9d, mf_nba_x1-32c88182, mf_nba_x2-316f9d2c (nba_totals, 37 trials, bar p < 1.35e-03);
mf_cfb_x0-bb672836, mf_cfb_x1-0532f23a, mf_cfb_x2-748c30c6 (cfb_lines, 21, bar 2.38e-03);
mf_nfl_x0-60059997, mf_nfl_x1-510a396e, mf_nfl_x2-d3fbe8be (nfl_lines, 135, bar 3.70e-04);
mf_cbb_x0-4aea3e4f, mf_cbb_x1-208fbe36, mf_cbb_x2-a8a460ab (cbb_totals, 4, bar 1.25e-02).
Specs edgelab/specs/mf_*.yaml (results/maker_fairfill/make_specs.py).

### Data
results/maker_fairfill/fetch.py -> bets.csv (180 bets: NBA 25, CFB 71, NFL 17, CBB 67; all with a start time) and
tapes/<ticker>.json (all 180 tickers returned trades; log fetch.log). Side mapping checked against the tape: median
our-side print within +/-2 h of T is 0.007-0.020 from our side mid vs 0.025-0.067 for the other side (bigger gap on
the 37 bets with mid off 50c), so YES/NO sides are right. Resting window median: NBA 22 h, CBB 20 h, CFB/NFL ~5.5 days.

### Pre-registered readout (results/maker_fairfill/score.py -> rows.csv, summary.json; date-clustered CIs)
| cell | fill rate | n filled | win filled / unfilled | avg cost | filled ROI [95%] p | AS win gap [95%] | verdict |
|---|---|---|---|---|---|---|---|
| NBA x0 | .84 | 21 | .43 / 1.00 | .507 | -15.9% [-47.1, +7.3] .91 | -.57 [-.73, -.44] | KILL |
| NBA x1 | .48 | 12 | .42 / .62 | .497 | -16.3% [-55.5, +13.3] .82 | -.20 [-.39, +.01] | NEEDS DATA |
| NBA x2 | .36 | 9 | .33 / .63 | .484 | -31.6% [-100, +9.6] .90 | -.29 [-.57, -.17] | KILL |
| CFB x0 | .59 | 42 | .50 / .48 | .500 | -0.4% [-27.9, +44.8] .51 | +.02 [-.24, +.27] | NEEDS DATA |
| CFB x1 | .45 | 32 | .50 / .49 | .491 | +1.6% [-28.4, +42.7] .46 | +.01 [-.33, +.27] | GO fwd paper |
| CFB x2 | .39 | 28 | .54 / .47 | .482 | +10.9% [-19.9, +48.4] .25 | +.07 [-.30, +.35] | GO fwd paper |
| NFL x0-2 | .82/.65/.41 | 14/11/7 | ~.55 / ~.65 | .48-.50 | +10-19%, 3 date clusters, CI degenerate | - | NEEDS DATA |
| CBB x0 | .82 | 55 | .60 / .75 | .509 | +18.1% [+4.4, +41.4] .005 | -.15 [-.36, +.02] | GO fwd paper |
| CBB x1 | .67 | 45 | .62 / .64 | .500 | +24.5% [+15.3, +44.6] .000 | -.01 [-.24, +.18] | GO fwd paper* |
| CBB x2 | .63 | 42 | .62 / .64 | .491 | +26.4% [+15.2, +46.8] .000 | -.02 [-.24, +.13] | GO fwd paper* |
| pooled x1 | .56 | 100 | .55 / .56 | .496 | +10.7% [-3.8, +25.7] .068 | -.01 [-.18, +.13] | (descriptive) |
*CBB x1/x2 p < bar 1.25e-02 and n >= 15, but PASS also needs filled CLV > 0: filled CLV is -1.09 / -1.14 -> not PASS.
Pre-registered mechanism readout (pooled x1 AS win gap) is not distinguishable from 0: CI [-.18, +.13].
Sensitivities (summary.json): touch fill, full-100-ct fill, 30-min cancel all move pooled ROI by < 4 pts.
Capacity: median contracts printed strictly through b while filled: 4-10k (CBB/CFB/NBA), 55k (NFL) -> a 100-ct order
is ~1-2% of through-volume; size is not the constraint.

### POST HOC (added after reading summary.json): CLV shows the adverse selection the win rate is too noisy to show
results/maker_fairfill/clv_check.py -> clv_check.json. Every league, every x: filled bets carry NEGATIVE CLV, unfilled
carry all of it. Pooled x1: CLV filled -0.90 vs unfilled +2.21, gap -3.10 [-3.82, -2.58]. Per league x1: NBA -2.49
[-4.56, -0.53], CFB -3.42 [-4.89, -2.59], CBB -3.45 [-4.79, -2.26], NFL -1.05 (3 clusters). Converting CLV to win
probability (p = 0.5 + CLV * phi(0)/sigma; sigma NBA 18.61, CFB 15.39, NFL 13.09/12.51, CBB 16.69 from
odds_2024 x games_2024): CLV-implied filled ROI pooled x1 -3.6% [-5.7, -1.9] (x0 -2.5%, x2 -2.9%) vs the TAKER on all
180 bets -4.0% [-5.7, -2.0]. So the maker saves ~2-3c of price and gives it all back in selection: no improvement.
The +24% CBB filled ROI is the same luck already seen in CBB taker (ROI +16%, CLV +0.04): CBB filled CLV is -1.09.
Mechanism: a strict print through our bid means the price moved toward the other side; that same drift is what moves
the closing line against us. The model side does not protect: the market moves against the model on about half the
bets, and exactly those fill. Unfilled bets (NBA example: DALMIN K1 forward, CLV +2.5, won, never filled) are the
ones the model was right about.
Caveat: CLV and fill are partly mechanically linked; CLV is only a proxy for win probability. But in K50 the CLV-implied
ROI matched the observed NBA ROI (-2.1% vs -2.2%), and here the win-rate gap is not distinguishable from 0 either way.

## Verdict
Pre-registered cell verdicts above stand as written (CBB x0-2, CFB x1-2 "GO forward paper"; NBA x0/x2 KILL; rest NEEDS
DATA; nothing PASSES). My reading, POST HOC and labelled so: the maker lever does not rescue the opener models. Fills
select the bets where the line moves against the model; CLV-implied filled ROI -2.5% to -3.6%, no better than taker.
Do not ask for real money on any mf_* cell. Recommend: retire mf_nba_*; keep the rest as a free forward SHADOW only,
judged on filled CLV as well as ROI (kill any cell whose filled CLV CI upper < 0 after 60 fills).

## Forward paper (built, not wired)
scripts/kx_maker_paper.py: shadows every taker `bet` row in kx_forward/{k1,k50nfl,kcfb}.jsonl with the same rule,
after the window closes, from the public tape (immutable; order fixed by the logged decision quote -> lookahead-free,
no live process needed, so no foreground run tonight was started). Output maker_<trial>.jsonl (append-only).
Tested on a copy (KX_FWD_DIR=results/maker_fairfill/fwd_test): K1 11 shadows, x1 6 filled 4 W; KCFB 2 shadows.
Wiring (lead/ops): run `kx_maker_paper.py run` after kx_forward in the same launchd job. Note K1 buys the strike nearest
the Pinnacle open, not S50; there is no forward K50-NBA/CBB logger yet.

## Learnings

### A model-chosen side does not protect a resting Kalshi bid from adverse selection
Bids on the model's side of the 50c rung (NBA/CFB/NFL/CBB opener models, 180 bets) filled 56% at x=1c; filled CLV
-0.90 vs unfilled +2.21, gap -3.10 [-3.82, -2.58]. Source results/maker_fairfill/clv_check.json. Matters: fifth
maker-adverse-selection case (after E5, RD2, first-listing, L35, K-FAV); direction from a model is not enough.

### Judge maker fills on CLV, not win rate, at small n
Win-rate gap filled vs unfilled was -.01 [-.18, +.13] (not distinguishable) while the CLV gap was -3.1 pts with a tight
CI. Filled ROI +10.7% was luck: CLV-implied -3.6% [-5.7, -1.9]. Source results/maker_fairfill/summary.json,
clv_check.json. Matters: outcome-based maker tests at n~100 cannot see adverse selection; CLV can.

### Kalshi sports totals/spreads charge maker fees
KXNBATOTAL, KXNBASPREAD, KXNCAAFTOTAL, KXNFLTOTAL, KXNFLSPREAD, KXNCAAMBTOTAL are fee_type quadratic_with_maker_fees:
0.0175 p(1-p) = 0.44c at 50c (vs taker 1.75c). Source GET /series (2026-11-15). Matters: maker saves ~1.3c fee + ~1-2c
spread, ~2-3c total -- which selection fully eats here.

### Maker fills can be paper-scored after the fact from the public tape
GET /markets/trades is immutable and the order is set by the logged decision quote, so a strict trade-through shadow
needs no live process. Source scripts/kx_maker_paper.py. Matters: any taker paper logger gets a free maker shadow.
