# Twin-security pairs (2026-10-21/22) -- scripts/pairs_twins.py

User: "Maybe we can see if there are other twin companies." Follow-up to pairs_bt.py (KO/PEP + 6 classic pairs:
+1.89%/yr dev, Sharpe .41, p .052, ~0 after 2009).

**Verdict: nothing to automate.** 47 trials, Bonferroni bar p < 0.00106. **None pass, and none reach even
uncorrected p < .05** (best: U4 GGR screen p .054, KELYA/KELYB p .073). Real twins *do* misprice: dual-class
spreads earn about +3.8%/yr gross, flat across 2006-12, 2013-18 and 2019-24, so the effect has not decayed. But the
gross edge sits in the thin share classes, and their spreads eat it: universe net is -3.5%/yr. Liquid twins (GOOGL/GOOG,
BRK, SPY/IVV, GLD/IAU) have a gross edge of 0.0-1%/yr, about the size of their trading costs. This is the textbook
limits-to-arbitrage pattern: the mispricing survives exactly where it is too expensive to trade.

## Design (pre-registered in the script docstring before any P&L; sha256 of the pre-data version 59fb1d8e...)
- DEV = 2006-01-03..2024-10-18. SEALED = 2024-10-21..2026-10-20. `dev` drops sealed rows at load. Sealed is unscored
  and waits for the user's yes. (2019-24 is dev here. stocks_bt.py's 2019-25 sealed window covers different rules.)
- Twin rule: spread = log A - log B (hedge ratio 1, no fit); z on the trailing 60 days (strictly past); enter |z|>=2,
  exit |z|<=0.5 / 20-day time stop / |z|>=5 break stop; **signal at close t, trade at close t+1**; skip entry on a
  zero-volume day. GATED variant: enter only if the deviation is at least 2x the round-trip cost.
- GGR rule (U4): monthly; 252-day formation within a GICS Sub-Industry of today's S&P 500; take the lowest-SSD 20 pairs
  that pass Engle-Granger p<.05; trade 126 days; open at 2 sigma, close when the spread crosses zero or the period
  ends; 6 overlapping cohorts.
- Costs per leg per side, tiered by trailing 60-day median $ volume: >=$1B 0.5bp, >=$100M 1.5, >=$20M 4, >=$5M 10,
  >=$1M 25, else 60bp. Borrow 0.3%/yr (>=$20M ADV), otherwise 2%/yr. The tiers are my estimates, not measured quotes.
- Return = daily P&L per $1 per leg, equal-weighted over live pairs. Newey-West(10) HAC t, one-sided p.

## Results (DEV, annual P&L per $1/leg, %)
| trial | gross | net | net no-borrow (= long-only swap) | net 2x cost | net Sharpe | p | trips/yr/pair | % days in pos | cost %/yr |
|---|---|---|---|---|---|---|---|---|---|
| U1 dual-class (24 pairs) base | +3.82 | -3.47 | -3.16 | -10.75 | -0.92 | 1.00 | 5.9 | 22 | 6.98 |
| U1 dual-class gated | +3.73 | -0.89 | -0.65 | -5.51 | -0.25 | .92 | 4.4 | 18 | 4.38 |
| U2 ETF twins (8) base | +0.18 | -2.10 | -2.00 | -4.38 | -2.67 | 1.00 | 11.6 | 15 | 2.19 |
| U2 ETF twins gated | +0.13 | -0.07 | -0.06 | -0.27 | -0.10 | .74 | 2.5 | 3 | 0.19 |
| U3 Canada interlisted (8) base | +0.38 | -1.98 | -1.92 | -4.35 | -0.63 | 1.00 | 12.5 | 13 | 2.31 |
| U3 interlisted gated | +0.35 | -1.47 | -1.42 | -3.29 | -0.47 | 1.00 | 11.6 | 12 | 1.77 |
| U4 GGR S&P 500 screen (20 pairs/cohort) | +1.92 | +1.14 | +1.49 | +0.37 | +0.37 | .054 | 3.1 | 68 | 0.44 |

Pairs worth naming (BASE rule; all 40 pair trials are in the ledger and the script output):
| pair | gross | net | p | net by sub-period 06-12 / 13-18 / 19-24 |
|---|---|---|---|---|
| GOOGL/GOOG (real from 2014-04) | +0.27 | +0.11 | .33 | 0.0 / -0.4 / +0.7 |
| BRK-B/BRK-A | +1.06 | +0.21 | .34 | +1.5 / -0.9 / -0.2 |
| Z/ZG | +2.44 | +1.49 | .12 | - / +1.2 / +1.6 |
| FWONK/FWONA | +4.05 | +1.31 | .16 | - / +0.2 / +2.2 |
| LBTYK/LBTYA | +2.25 | +0.84 | .13 | +1.8 / -0.1 / +0.6 |
| KELYA/KELYB | +9.82 | +6.09 | .073 | +5.4 / +10.3 / +2.6 (KELYB had zero volume on 3,375 of 4,731 days, so this is not executable at size) |
| AGM, GTN, BELFB (thin) | +11 to +12 | -4.0 to +1.7 | >.36 | costs of 10-15%/yr |
| ETF twins (each) | -0.05 to +0.21 | -5.3 to -0.2 | >=.94 | negative in every period |

Sub-periods, universe gross: U1 +4.1 / +3.6 / +3.8 (no decay); U2 +0.4 / +0.0 / +0.1; U4 +3.3 / +0.6 / +1.5.

## How to read it
1. **ETF twins are arbitraged away.** The gross edge is 0.0-0.2%/yr because authorized participants keep SPY, IVV
   and VOO within cents of NAV. The rule trades about 12 times a year on noise smaller than the spread. Gating on costs
   just stops it from trading.
2. **Dual-class premia mean-revert, but only where you can't trade them.** The rule's gross P&L rises as liquidity
   falls (AGM-A, GTN-A, BELFA, KELYB, CRD-B), and so does the cost. The 60bp tier for those names is a guess. Closes
   on near-zero-volume days are partly stale prints, so the real cost is probably higher, not lower.
3. **Liquid dual-class is roughly break-even.** On GOOGL/GOOG and BRK, the 1-day-lagged reversion is about the size
   of the 2-6bp round trip.
4. **The GGR screen (+1.1%/yr net, p .054) is the same weak result as pairs_bt.py**, and it is *optimistic*. Today's
   S&P 500 members are all survivors, so pairs where one leg collapsed or was delisted are missing. 2013-18 is flat.
5. **Canadian interlisted lines lose net**, and the gross is likely partly FX-timestamp noise (CAD=X is not a 16:00 ET
   close). Robinhood can't buy the TSX leg anyway.

## Execution note (Robinhood Agentic Trading is stocks-only)
A pair needs a short leg, and I have **not verified** that Robinhood allows short selling. The "net no-borrow" column
is the long-only version: someone who already holds one class swaps to the other when it is cheap. That needs no
short and pays the same spread. It is still negative for every universe except GGR, and the positive single pairs
(Z/ZG +1.6, FWONK +1.6) were picked after seeing results and are not significant. Don't build it.

## Data / bugs / caveats (all disclosed in the script)
- Not tested: closed-end fund vs NAV (no reliable free daily NAV history), LBRDK/LBRDA and CWEN/CWEN-A (no yfinance
  data), and delisted twins (DISCA/K, VIAC/A, CMCSA/K, LSXMA/K, RDS-A/B). So U1 also has survivorship bias.
- Fixed before scoring: CNI.TO -> CNR.TO (ticker typo). yfinance back-fills GOOG before 2014-04-03 as an exact copy of
  GOOGL, so I added a guard that blocks entries when the spread sd is below 1bp.
- Bug after the first scoring run: the GGR leg picked 0 pairs because the cache index contains TSX/FX-only days, which
  are NaN for every US stock. U4 was restricted to SPY sessions and re-run with the rules unchanged. The bad ledger
  line (trial 148) is voided by an appended `void_bug` record. Study size stays 47.
- GGR also picks GOOGL/GOOG before 2014, when the two series are identical, about 100 times (CollinearityWarning). It
  trades float noise and pays costs, which biases U4 net slightly *down*. The bias is bounded by U4's total cost of
  0.44%/yr.
- Ledger: 48 lines `"source": "pairs_twins"` in lab/ledger.jsonl (47 trials + 1 voided) plus the void record. Earlier
  stock trials (stocks_bt.py, pairs_bt.py) are in lab/stocks_ledger.jsonl, so the stock-side trial count is about 62.
