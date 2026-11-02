# E1X-Q -- Kalshi daily biggest earthquake (KXBIGGESTQUAKE) vs USGS-catalog climatology

PRE-REGISTERED 2026-11-01, committed BEFORE any Kalshi candle was fetched or any quote joined to a forecast or an
outcome. Script: `scripts/e1x_quake.py` (`usgs` -> `clim` -> `candles` -> `build` -> `score --split dev` -> `settle`).
Ledger dataset `e1x_quake`. Cache `data/e1x/` (gitignored).

## Why this family (from the E1-pattern search, same day)
E1 = niche retail book + a better public forecast + a tradeable far side. Survey of ~20 weekly/daily non-sports families
(contract census + fresh /series scan; `scripts/e1_family_scan.py`, `e1x_quotecheck.py`): TSA weekly (KXTSAW, 277 events),
album units, Netflix, YouTube all show the Q016/N7 structure -- far rungs are one-sided (YES bid 0 on high rungs, YES ask
1.00 on low rungs at the Monday decision time), so the "safe" side cannot be bought. KXBIGGESTQUAKE is the exception: all
10 rungs (M 5.2..7.0 step 0.2) were two-sided at 5-9c spreads on the 11-02 ladder, and the outcome is
unforecastable beyond a stable public base rate (nobody can hold private information about tomorrow's quakes, except
aftershock sequences, which are public and handled by a guard).

## Disclosed before writing
- Market list incl. expiration_value per event (I saw that two of 61 days had M >= 7: 10-09 M7.6, 10-22 M7.0).
- One live (unsettled) ladder for 11-02: YES bid/ask 5.2 .89/.94, 6.0 .37/.45, 6.6 .15/.20, 6.8 .15/.19, 7.0 .08/.13.
- p_clim computed (below). No historical candle has been fetched.

## Contract (read 2026-11-01)
"If the highest USGS-reported earthquake magnitude worldwide during <D> 00:00:00-23:59:59 UTC is >= K, YES."
Determining value = USGS magnitude displayed at expiration (Mw preferred). Event opens 20:00 UTC D-1, closes
23:59:59 UTC D (early on >= 7.0). fee_type quadratic -> taker fee 0.07 p(1-p) per contract, unrounded.

## Forecast (frozen)
p_clim(K) = share of UTC days 2000-01-01..2025-12-31 whose max USGS ComCat magnitude (type earthquake, M >= 4.5 pull,
current catalog values) is >= K. Values: 5.2 .901, 5.4 .763, 5.6 .599, 5.8 .440, 6.0 .306, 6.2 .201, 6.4 .132,
6.6 .088, 6.8 .059, 7.0 .038. Stable across 5-year blocks (e.g. 6.6: .082/.091/.098/.090/.080).
ETAS guard: skip the whole day if any M >= 7.0 occurred in the 72 h before the decision time.

## Decision and entry (frozen)
- Decision time T = 23:00 UTC on D-1 (one hour before the UTC day starts; ~3 h after listing).
- Quote = close of the last hourly candle with end_period_ts <= T. NO ask = 1 - yes_bid. Buyable iff yes_bid >= .01.
- 1 contract per bet, fill at the candle quote assumed (depth unknown; capacity noted separately).
- edge = (1 - p_clim) - NO ask - 0.07 NOask(1-NOask). Bet NO iff edge >= 0.03 (margin fixed a priori).

## Trials (k = 2)
- T1: all rungs passing the edge rule.
- T2: same, rungs K >= 6.4 only (the lottery tail).
No maker variant (no trade tape to model fills).

## Split, metric, bar
- DEV = event days 2026-09-02..2026-10-14 (~70% of the 61 settled days at pull). SEALED = 2026-10-15..2026-11-01, scored
  ONCE, only if a DEV trial passes, after re-pulling results; only events with all rungs finalized are used (several
  late-Oct events were only partly finalized at pull time -- partial events would bias toward YES-resolved rungs).
- ROI per $ (cost = ask + fee), day-clustered bootstrap (10k, seed 20261101); one-sided p = share of resampled ROI <= 0.
- PASS = p < 0.05/2 = 0.025 AND both chronological halves of the split positive.
- Too-good guard: ROI > +10% on > 200 bets -> inspect rows (decision ts vs listing, UTC day boundary, settlement
  value vs USGS) before believing.
- Descriptive (not a gate): `settle` compares USGS daily max with Kalshi expiration_value.

## Power (from climatology only, no prices)
If the tail prices look like the 11-02 ladder (NO asks ~.85/.85/.92 on 6.6/6.8/7.0), truth = p_clim gives about
+0.18 units/day on ~2.6 units staked (+6.8%), day sd ~0.65 units. DEV ~43 days -> z ~1.8, power ~40% at the .025 bar;
SEALED ~17 days -> power ~20%. **Underpowered by construction (the series only started 2026-09-02).** A DEV fail is
therefore weak evidence; a DEV pass + sealed fail is likely. Whatever the result, the honest next step for this family
is a forward paper logger (~250 days to ~80% power).

## Kill
DEV ROI <= 0 on both trials -> DEAD (tail not overpriced vs climatology). DEV positive but p >= .025 -> LEAD, forward
logger only, no sealed spend.
