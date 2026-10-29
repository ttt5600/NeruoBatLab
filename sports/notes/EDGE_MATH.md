# EDGE MATH -- the standard every edge claim must pass (written 2026-10-29)

Calculator: `scripts/edge_math.py` (library + CLI; tests `tests/test_edge_math.py`, 13 pass). Companions:
`notes/TRADING_PRINCIPLES.md` (the playbook: why each step exists) and `notes/SITUATIONAL_FACTORS.md` (every
stadium / weather / rest / travel / ref / luck factor and whether the market prices it). Binding via
`notes/protocol.md` rule 8. Nothing here is new evidence: every input is copied from the file cited next to it; every
output is `edge_math.py` arithmetic on those inputs. Public repo -- no account data.

```
../.venv_quant/bin/python scripts/edge_math.py bet --wins 209 --n 219 --price 0.8877 --venue forecastex \
        --frac 0.25 --bankroll 2000 --picks-per-night 3 --liquidity 29          # one bet, every step below
../.venv_quant/bin/python scripts/edge_math.py devig --odds -120 +100            # 3 de-vig methods
../.venv_quant/bin/python scripts/edge_math.py clv --bet-prob ... --close-prob ...
../.venv_quant/bin/python scripts/edge_math.py power --mu 0.02 --sigma 1.0       # or --roi 0.02 --cost 0.5238
../.venv_quant/bin/python scripts/edge_math.py ruin --p 0.95 --price 0.90 --frac 0.25 --level 0.5
```

-----------------------------------------------------------------------------------------------------------------
## (a) THE STANDARD -- 12 steps, in order. A claim that skips one is not an edge claim.

Notation: a binary contract pays $1. c = all-in cost per contract (price + fee); for a sportsbook c = 1/decimal odds.
p = our probability the side wins; q = 1 - p; b = (1 - c)/c = net odds.

| # | Step | Formula | Kill / note |
|---|---|---|---|
| 1 | **Price** -- the price you can actually get | ask (taker), resting bid (maker), offered odds at the book you can legally use, at the time you would bet | display prices, last trades and "current" timestamps are not quotes (FUND_BOOK lesson 6) |
| 2 | **Implied probability** | contract: price; decimal d: 1/d; American +A: 100/(A+100), -A: A/(A+100) | -- |
| 3 | **De-vig** (books / multi-outcome markets only) | overround = sum(implied) - 1; proportional p_i = q_i/S; power p_i = q_i^k with sum = 1; Shin (1993) closed form in z | use **power or Shin** for anything with a longshot: proportional over-states longshots and faked soccer CLV (`scripts/soccer_fd.py` docstring: S6 withdrawn, hit/fair 0.80 at odds 10-20) |
| 4 | **Fees** | Kalshi taker 0.07 C p(1-p), rounded up per order; Kalshi maker 0 on `quadratic` series, 0.0175 C p(1-p) on `quadratic_with_maker_fees`; Polymarket US taker 0.0695 C p(1-p); ForecastEx $0.01/contract; book: in the odds | fee is part of the price: c = price + fee (notes/kalshi_temp_maker.md, notes/venues.md, notes/forecastex_wx.md) |
| 5 | **Break-even probability** | p_BE = c (contract) = 1/d (book). -110 -> 0.5238 | an "X% hit rate" means nothing until set against p_BE |
| 6 | **Our p with a CI** | from a record: Wilson 95% on wins/n; from a model: out-of-sample calibrated p, or the de-vigged sharp CLOSE (CLV) | p must come from DEV/sealed/forward data, never in-sample. Report the lower bound p_LCB |
| 7 | **EV** | per contract p - c; per $ staked p/c - 1; at the LCB p_LCB/c - 1 | point EV <= 0 -> stop. EV > ~5% -> distrust (protocol rule 6) |
| 8 | **Kelly on the LCB** | f* = (b p - q)/b = (p - c)/(1 - c), evaluated at p_LCB; stake = k f*, k = 1/4 default (1/2 max), per-pick cap 5% | f*(p_LCB) <= 0 -> no money (edge not established), paper only |
| 9 | **Per-night / correlation cap** | sum of same-night stakes <= 10% B (executor NIGHT_FRAC); same-weather / same-game picks are one exposure | (notes/kalshi_sizing.md) |
| 10 | **Liquidity cap** | stake <= depth at the price x c; maker: <= size that fills without extra adverse selection (~$75/pick weather) | (notes/fund_economics.md) |
| 11 | **Sample size / power** | per-$ sigma = sqrt(p q)/c; n = ((z_a + z_b) sigma / mu)^2, one-sided a .05, power .8 -> (1.645 + 0.842)^2 = 6.18 x (sigma/mu)^2 | if n is years away, the edge cannot be confirmed -- size as Tier C at most |
| 12 | **CLV check (forward)** | CLV_prob = p_close,devig(side) - p_paid,devig; mean, SE, z, one-sided p over clustered bets (`edge_math.py clv`) | positive CLV is necessary, not sufficient: it must exceed the vig you paid (O1 sealed fail, FUND_BOOK lesson 3) |

Extra outputs of `bet`: log-growth per bet g = p ln(1 + f b) + q ln(1 - f); risk of ruin P(ever at level x start)
= level^(2g/v) (diffusion approximation, Thorp 2006; full Kelly ~ P(ever halve) = 1/2); `--sim N` = Monte Carlo
max drawdown over N independent bets (independence is an assumption -- same-night picks are correlated, which is why
step 9 exists).

-----------------------------------------------------------------------------------------------------------------
## (b) WORKED MATH -- every passed, paper and top lead

Common settings: bankroll B = $2,000 (FUND_BOOK s1 mandate), k = 1/4 Kelly, per-pick cap 5%, night cap 10%. p_LCB =
Wilson 95% two-sided lower bound on the record shown. "Sized" = after pick, night and liquidity caps. Inputs in
column 2; all outputs from `edge_math.bet_table` (script used: scratch `edge_worked.py`, a loop over `bet_table`).

### Summary table

| Edge | Record (source) | c all-in | p / p_LCB | EV/$ point / LCB | Kelly full point / full LCB / 1/2 LCB / 1/4 LCB | Sized (B $2k) | n to detect EV |
|---|---|---|---|---|---|---|---|
| **E1** Kalshi wx taker | sealed 1,844 bets, 55 losses; mean NO ask .9535 + fee .0031 (`data/kalshi_temp_maker_picks_sealed.parquet`, cohort E1) | .9566 | .9702 / .9614 | +1.42% / +0.50% | .313 / .110 / .055 / **.0275** | 2.0% = $40 (night cap, ~4.5 picks/night) | 972 |
| **E2/X1** Kalshi wx taker | sealed 2,464 bets, 51 losses [LB]; ask .9700 + fee .0020 (same parquet, X1) | .9720 | .9793 / .9729 | +0.75% / +0.09% | .260 / .031 / .015 / **.0076** | 0.76% = $15 | 2,374 |
| **E4** maker (E1+E2) | sealed 3,456 fills, loss rate filled 3.067% (= 106 losses), ROI +2.22% [LB]; maker cost DERIVED c = p/(1+ROI) = .9483; fee 0 (`quadratic`) | .9483 | .9693 / .9630 | +2.22% / +1.55% | .407 / .285 / .143 / **.071** -> 5% cap | 1.25% = $25 (night cap at ~8/night) | 416 |
| **E5** ForecastEx wx | sealed 219 bets, 10 losses, mean cost .8977 incl. $0.01 fee, median 29 contracts [LB forecastex_wx_FX1 T1] | .8977 | .9543 / .9180 | +6.31% / +2.26% | .554 / .198 / .099 / **.050** | 3.3% night-capped; **liquidity: 29 x .8977 = $26** | 84 |
| **E3** soccer S12 | sealed CLV +3.63% (EV vs power-de-vigged Pinnacle close), ROI +8.71%, units CI lo +29.1/2,004 [LB]; dev median price 3.6 (dev picks rebuilt with `soccer_fd.bets`, 2,604 = ledger) | .2778 (3.6) | close-implied .2879 / .2818 (from ROI CI lo +1.45%) | +3.63% / +1.45% | .0140 / .0056 / .0028 / **.0014** | $2.79/bet -- **not tradeable from CA** | 12,465 |
| **L32** NBA open totals, Kalshi taker | sealed 575, win .5322 ex-push (n ex-push not stored; 575 used) [LB]; Kalshi .51 + .0175 (notes/fund_economics.md) | .5275 | .5322 / .4913 | +0.89% / -6.9% | .010 / **<0** | **0 -- paper only** | 70,250 |
| L32 at a -110 opener | same record | .5238 | .5322 / .4913 | +1.60% / -6.2% | .018 / **<0** | 0 | 22,001 |
| **L34** NHL totals at -110 | dev T5 910 bets, win .5275, no-vig +7.1% p .0125 [L:nhl_simple_model]; real offered price ~4% overround, offered ROI +2.6% p .20 (T6) | .5238 | .5275 / .4950 | +0.70% / -5.5% | .008 / **<0** | 0 | 114,847 |
| **L36** CFB totals at -110 | dev T04 3,014 bets, ROI at -110 +2.62% p .10 [L:cfb_simple_model]; p DERIVED = 1.0262/1.9091 | .5238 | .5375 / .5197 | +2.62% / -0.79% | .029 / **<0** | 0 | 8,160 |
| **L39** monthly rain NO maker | dev W1m 146 bets, 1 loss, avg cost .9589, fee 0 [L:kalshi_wx_winter] | .9589 | .9932 / .9622 | +3.57% / +0.35% | .833 / .081 / .040 / **.020** | 2.0% = $40 -> **liquidity cap $30/pick** (notes/fund_economics.md) | 36 |
| **Q003** Mentions maker NO (IN1_d6) | dev 859 fills, YES rate filled .1641, mean NO cost filled .7806, fee 0 [L:q003_retail_maker] | .7806 | .8359 / .8096 | +7.08% / +3.72% | .252 / .132 / .066 / **.033** | 3.3% = $66 (size per quote 10 contracts in the test: notes/q003_retail_maker.md) | 278 |
| **Q008** FX econ | dev exploratory 64 bets / 57 events, 0 losses, "FX cost .868" (QUESTIONS.md; treated as all-in, the conservative reading); median 46 pairs | .868 | 1.000 / .9434 | +15.2% / +8.7% | 1.00 / .571 / .286 / **.143** -> 5% cap | $100 -> **liquidity 46 x .868 = $40** | (sigma 0 at 0 losses -- undefined) |
| **L40** LIP | not a probability bet -- see below | | | | | | |

**Per-edge Kelly verdicts (quarter Kelly on the 95% Wilson LCB):** E1 2.75% · E2 0.76% · E4 7.1% (capped 5%) ·
E5 5.0% · E3 0.14% · L39 2.0% · Q003 3.3% · Q008 14.3% (capped 5%) · **L32 / L34 / L36: <= 0 -> no money.**
In every positive case a cap (night, pick or liquidity) binds before Kelly does.

### Step-by-step for the two live edges

**E5 ForecastEx (live Tier B).** 1 Price: mean last D-1 trade .8877 (sealed T1). 2 Implied .8877. 3 De-vig: n/a (we
buy one side of a binary at a traded price). 4 Fee $0.01 -> c = .8977. 5 Break-even p = .8977, i.e. loss rate
<= 10.23% (lockbox `breakeven_loss_rate` .1023). 6 p = 209/219 = .9543; Wilson 95% [.9180, .9750] (lockbox loss_ci
[.0250, .0820] -- identical). 7 EV +6.31%/$ point, +2.26%/$ at the LCB. 8 b = .1023/.8977 = .1139; f*(LCB) =
(.9180 - .8977)/(1 - .8977) = .198 -> 1/2 = 9.9%, 1/4 = 5.0%. 9 ~3 picks/night share the weather -> 10%/3 = 3.3%.
10 Median 29 contracts x .8977 = **$26 per pick: liquidity binds** (FX caps were raised to $15/order, 20 contracts,
FUND_BOOK s2.3, which binds even earlier). 11 n to detect the point EV at sigma .23: 84 bets; to detect the LCB EV
+2.26%: far more (`power --mu 0.0226 --sigma 0.23` ~ 640). 12 CLV: no closing quote exists on ForecastEx; the forward
check is FX1-fwd ROI on live bid/ask (FUND_BOOK 3b).
**Correction found while doing this:** FUND_BOOK s1 derived E5 half-Kelly ~5% with c = .898 + $0.01 = .908, but the
lockbox `mean_cost` .8977 ALREADY includes the fee (`scripts/forecastex_wx.py` line 210, `cost = price + FEE`). With
the fee counted once, full Kelly at the LCB is .198 (half 9.9%), not ~.11. It changes nothing operationally (liquidity
and the 5% cap bind first), and .908 happens to be close to the conservative T2 day-high cost .9085, where
f*(LCB) = .104. Caveat that the LCB does not cover: last-trade prices may not have been executable.

**E1 Kalshi taker (live).** Ask .9535, fee 0.07 x .9535 x .0465 = .0031 -> c = .9566 -> p_BE .9566 (loss rate <= 4.34%).
Record 1,789/1,844 = .9702 (my recompute of ROI from the parquet is +1.44% vs lockbox +1.45%: the fee rounding).
LCB .9614 -> EV +0.50%/$ -> f*(LCB) = .110 -> 1/4 = 2.75%. kalshi_sizing.md's per-pick number (4.4% at its 80% one-sided
ratio-model bound) is less conservative because an 80% bound sits higher than a 95% one; same conclusion: flat sizing,
caps bind. Live executor tiers (1% / 0.5%, kalshi_sizing.md) are below both.

### Drawdown / ruin illustration (`edge_math.simulate_drawdown`, 1,000 independent bets, 1,000 paths, seed 0)

| Case | f per bet | true p | E[max DD] | 95th pct max DD | median final x | P(final < 1) |
|---|---|---|---|---|---|---|
| E4 maker night-capped | 1.25% | .9693 | 4.4% | 7.0% | 1.33 | 0 |
| E1 taker night-capped | 2.0% | .9702 | 7.8% | 12.8% | 1.32 | 0.9% |
| E5 sized 3.3%, p = point | 3.33% | .9543 | 10.4% | 15.5% | 8.1 | 0 |
| E5 sized 3.3%, p really = LCB | 3.33% | .9180 | 20.8% | 33.1% | 2.0 | 1.2% |
| **E5 FULL Kelly on the point estimate, p really = LCB** | 55.4% | .9180 | 99.9% | 100% | 0.00 | 89.9% |

The last row is the whole case for "Kelly on the LCB, then fractional": full Kelly on an over-estimated p has
negative growth (g = -0.0100/bet) and P(ever halving) = 1.0 under the diffusion formula, with a real edge in hand.

### L40 LIP (rebate, not a probability edge)
Return = share of a fixed reward pool for resting size, minus fill losses; Kelly does not apply because the payoff is
not a binary bet on an outcome we forecast. From notes/kalshi_lip.md (one snapshot 10-28): 100 contracts at the best
bid pays median $1.4-2.3 per period per side, ~1.8-5%/day on capital tied up, before fills; fill losses -$0.5 to -$1.3
per bracket-night for a 1,000-lot at 1c (-$3 to -$7 at 2c). Standard applied: (1) price = bid we rest at; (4) fee 0
(maker); (7) EV = reward share - E[fill loss] -- **net unknown** until the 10-day paper test (ends 11-08) measures
fills; (10) capital cap = contracts x bid; (11) the pre-registered 10-day test is the power budget. Max loss per
contract at a 1c YES bid is $0.01, so ruin is not the risk; adverse fills and program end (2027-01-01) are.

-----------------------------------------------------------------------------------------------------------------
## (c) Teaching case -- the "kicker over 3.5 hits 90%" system, through the same 12 steps

Source: notes/nfl_kicker_system.md, ledger `nfl_kicker_system` (DEV 2018-24, 1 trial).

| Step | Kicker system |
|---|---|
| 1 Price | claim: -150 to -200. **Unverified** -- ESPN BET 2024 main lines open at 4.5-9.5, never 3.5 |
| 2 Implied | -200 -> .6667; -150 -> .600 |
| 5 Break-even | at -200, p_BE = .6667 |
| 6 Our p | qualifying 300/354 = .847 (Wilson .806-.881); **base rate, all eligible: .830**; matched baseline on team total predicts .855 -> residual -0.7pp, p .65 |
| 7 EV at the claimed price | .847/.6667 - 1 = +27% (the +27..+41% headline) |
| 8 Kelly at the claimed price | f*(LCB) = .419 -- a Kelly number that large is itself the tell: no book leaves 42%-of-bankroll edges on a main market |
| 3+5 Fair price | p .847 -> **-555** (empirical); market-implied P(>=4) for these games .890 -> **-810** |
| 7-8 at the real price | at -555: EV 0, f* 0; at -810 with p = .890 - .006: EV -0.69%/$, f* < 0 |
| 11 Power | irrelevant: the filter adds -0.7pp to the base rate |

Lesson (FUND_BOOK lesson 5): **a hit rate is compared to the price's break-even, and the filter is compared to the
base rate of the same bet without it.** 85% vs 50% is not an edge; 84.7% vs 83.0% base vs 89.0% priced is a loss.
Recipe for any social-media system: `protocol.md` "claim check".

-----------------------------------------------------------------------------------------------------------------
## (d) Template applied to two hypothetical bets

**Sportsbook -110 side, model says 55%, model record 275-225 (n 500, out of sample).**
`edge_math.py bet --wins 275 --n 500 --odds -110 --frac 0.25 --bankroll 2000`
c = .5238 (overround .0476 on -110/-110, de-vig .5000/.5000) -> p_BE .5238 -> p .55, Wilson [.506, .593] -> EV
+5.0%/$ point, -3.4%/$ at the LCB -> full Kelly .055 point, **-.037 at the LCB -> no money**. n to detect +5% at -110:
2,231 bets. 55% over 500 bets is not yet distinguishable from the 52.4% break-even. Paper it and track CLV.

**Polymarket US YES at 0.55 (taker), our p .60 from a 180-120 record (n 300).**
`edge_math.py bet --wins 180 --n 300 --price 0.55 --venue polymarket_us --frac 0.25 --bankroll 2000`
fee .0695 x .55 x .45 = .0172 -> c = .5672 -> p_BE .5672 -> p .60, Wilson [.544, .653] -> EV +5.8%/$ point,
-4.2% at the LCB -> f*(LCB) < 0 -> **no money**. Resting a maker bid at .54 (no fee assumed only if the market's
schedule says so; check) would make c = .54 and f*(LCB) = (.5436 - .54)/.46 = .008 -- the fee and the spread are
most of the edge at coin-flip prices, which is why our live edges are at the 90c+ tails.

-----------------------------------------------------------------------------------------------------------------
## Caveats
- Wilson treats bets as independent. Weather within-night design effect measured 1.00 (notes/fund_economics.md), but
  E5 picks overlap E1 picks heavily (46/62 station-days, notes/forecastex_wx.md) -- cross-venue exposure is one bet.
- Mean-cost Kelly is an approximation for bets at varying prices; the executor sizes per pick.
- Derived inputs are marked DERIVED (E4 maker cost, L36 p, E3 representative bet). L32's n ex-push is not stored.
- E3 p is the de-vigged Pinnacle close (CLV), not the realised win rate -- the standard's preferred p for book bets.
