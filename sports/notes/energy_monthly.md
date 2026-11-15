# energy_monthly -- ERCOT hub monthly-average power + TTF monthly + NATGAS/BRENT daily (agent 2026-11-15)

Agent topic `energy_monthly`. Brief: census the series (rules, index, events settled, bracket spacing, depth), then
pre-register a late-month "already pinned" certain-side rule for monthly AVERAGES. Scratch: `sports/results/energy_monthly/`.

## Census (read 2026-11-15 from /series + /events, `results/energy_monthly/census.py` -> `census.json`)
| series | settles on | events settled | open event | strikes |
|---|---|---|---|---|
| KXERCOTHOUM / NORTHM / WESTM | simple avg of ALL 15-min RT Settlement Point Prices (NP6-905-CD) at HB_HOUSTON / HB_NORTH / HB_WEST over the calendar month (CT operating days), incl. negatives; first-published prices; equality = No | **0** | Nov 2026 only (opened 2026-11-01 05:00Z, closes 2026-12-01 05:59Z) | 7 "greater" rungs each: HOU 20/30/35/40/50/65/100; NORTH 20/25/30/35/45/60/100; WEST 0/15/25/30/40/60/100 |
| KXTTFMON | Pyth TGAS/USD 1-min candle close at 17:00 ET on the LAST day (Nov 30) -- a POINT price, NOT a monthly average | 0 | Nov 2026 (opened 2026-11-10) | 40 rungs 8..164 step 4 |
| KXNATGASD / KXBRENTD | Pyth index 1-min close 17:00 ET daily | 126 / 131 | daily | 20-110 rungs |

Consequences read from the rules alone (no outcomes):
- **TTF is not an average** -> no pinning mechanism; it is a terminal-price ladder of exactly the Q016 class
  (Pyth commodity index at 17:00 ET), which Q016 already found dead (far rungs lack YES bids; two-sided rungs within
  ~1c of futures). Not re-tested. NATGASD/BRENTD are literally Q016's series -> not re-tested (do-not-redo).
- **ERCOT is the only real average.** It has zero settled events -> no Kalshi price history to backtest prices on.
  What CAN be tested on history is the *certainty* half of the rule: how often a conservative bound on the
  remaining days' average is violated (a violation = a loss at ~96-99c). The *price* half (is the certain side
  offered below 1 - fee?) can only be measured forward from now.

## PRE-REGISTRATION (written 2026-11-15 BEFORE downloading any ERCOT price history or computing any bound)
Question: late in a month, does partial realized RT hub average + a conservative bound on the remaining days rule
out some Kalshi rungs with zero historical violations, and how many rungs/month would that flag?

Data: ERCOT NP6-785-ER "Historical RTM Load Zone and Hub Prices" yearly zips (free, MIS reportTypeId 13061),
15-min SPP for HB_HOUSTON/HB_NORTH/HB_WEST. DEV = months 2011-01..2023-12. SEALED = 2024-01..2026-10 (not opened
without the user's yes). Nov 2026 = forward paper.

Rule (decision at 10:00 CT on day d of month m, R = days remaining incl. d, R in {10,7,5,3,2,1}):
- Known: A = average of all intervals of days 1..d-1 (published by then; RT SPP posts within minutes). n_e of N.
- Bound on remaining-days average M: U_R = max over all PAST R-day windows (strictly before month m, any month of
  the year, from 2011) of the R-day rolling average at that hub, x 1.25; L_R = min of the same, x 1.25 if negative
  else x 0.75 ... simplified: L_R = min(past R-day rolling min, 0) - 25 $/MWh. No lookahead: windows ending before
  day 1 of month m only.
- F_hi = (n_e A + (N-n_e) U_R)/N, F_lo = (n_e A + (N-n_e) L_R)/N.
- NO on rung K (YES = F > K) if F_hi <= K; YES on rung K if F_lo > K. Price: taker at ask + 0.07p(1-p) fee;
  maker variant = resting bid at 1c below ask, counted only if a trade prints through it (forward only).
- Strike grid for history: Kalshi's union {0,15,20,25,30,35,40,45,50,60,65,100} (current grids; flagged rungs
  only count if they are on that hub's own Nov-26 grid).

Success (dev): zero violations (rule-flagged side loses) across all dev hub-month-decision rows AND >= 3 flagged
rungs per hub-month on average at R<=5. Kill: any violation at the pre-set multiplier (no re-tuning on dev; a
re-tune would be a new trial counted in edgelab), OR the market-null bar is unreachable: with ~96-98c costs the
L39 lesson says 132-213 loss-free fills are needed -> if expected fills/month x 12 < 132, verdict = "lead, forward
paper only, cannot pass the bar within a year" (economic kill, not a statistical one).

## Steps / numbers
(filled in below as they are produced)

### 1. Book + tape, 2026-11-15 (`results/energy_monthly/books.py` -> `books_2026-11-15.json`)
- ERCOT far rungs: a maker BIDS the certain side at 94-95c (e.g. HOU T65 NO bids .88/.94/.95, T100 NO .95; HOU T20 YES
  bids .95) and nobody offers it: no YES bids on HOU T65/T100, NORTH T100, WEST T60/T100 -> the certain side has NO ask.
  Far rungs (HOU T20/T50/T65/T100, NORTH T45/T60/T100, WEST T25/T60/T100) printed 0 trades in 14 days. All 3 events:
  ~129 trades total, almost all on the 2-3 near-money rungs (NORTH T25 39, T30 59; volume 3087 on NORTH T30).
- TTF: same shape -- low rungs YES bid .96-.99 with no NO bids, high rungs NO bid .93-.96 with no YES bids; 13 trades
  total, all T80-T112. Exactly the Q016 pattern; not pursued.

### 2. ERCOT history (`fetch_ercot.py` -> `hubs.parquet`, 2011-2023 + 2026; 2024-25 zips NOT downloaded = sealed)
Daily interval counts all within 92-100 (assert in pin_backtest.py). Nov 2026 through 11-14 (1348 intervals):
HOU 38.20, NORTH 36.88, WEST 45.53 $/MWh (Oct 2026: 38.82 / 33.70 / 38.04).

### 3. Pre-registered certainty test, DEV 2012-01..2023-12 (`pin_backtest.py` -> `pin_dev.csv`)
- 5562 flags, **0 losses** (425 hub-months). NO flags 534/436/271/195/176/32 at R=1/2/3/5/7/10; YES 927/832/738/595/470/356.
- Median |final - K| of flagged rungs: NO 67-82 $/MWh, YES 15-34 -> these are the obvious rungs (the ones the maker
  already bids 95c for), not close calls.
- Flags per hub-month at R=5: **1.83 < 3 pre-registered -> SUCCESS CRITERION FAILS on count.**
- Uri (Feb 2021) sets U = $9309 (1.25 x max 5-day avg) for every later month -> **no NO flag is possible after Feb
  2021** (5/30 x 9309 = 1551 > 100); post-2021 flags are all YES on low rungs. Pre-2021 U was $450-711, never lost.
- Expected fills: ~1.8 flagged rungs/hub-month x 3 hubs = ~5.5/month at R=5 -> <= ~66/yr even at 100% fill vs
  132-213 loss-free fills needed (l39_expansion.md). Realistic fills ~0: no ask on the certain side, 0 far-rung trades.
- **Verdict: DEAD as a trade (economic kill + count fail).** Certainty half is sound (0/5562) but unmonetisable.
- E4 far-tail earlier in the month: no free ERCOT forward curve (ICE North Hub monthly is paid); not run.
- edgelab: `em1_ercot_pinned-de072a1a` registered (family kalshi_data_ladders now 50 trials, bar p < 1e-3), rule
  forward:unwired. Dataset `kalshi_ercot_monthly` has NO vault.yaml entry (lead: retire it, or add an entry only if wired).

## Learnings
### A pinned monthly average is only worth trading if someone sells the pinned side
ERCOT hub monthly-average rungs that the partial average rules out had 0 losses in 5562 dev flags (results/energy_monthly/pin_dev.csv), but on 2026-11-15 a maker already bid those certain sides at 94-95c with no offers and 0 far-rung trades in 14 days (results/energy_monthly/books_2026-11-15.json). Check for an ask on the certain side before building a pinning model.
### One extreme event poisons a max-based bound forever
Winter Storm Uri set ERCOT's max 5-day hub average to ~$7447 (U = $9309), so after Feb 2021 no NO-side flag is possible on any rung up to $100. A max-of-history bound needs seasonal/regime conditioning or it silently stops firing.
### KXTTFMON is a point price, not a monthly average
KXTTFMON settles on the Pyth TGAS 1-min close at 17:00 ET on the last day of the month (rules in results/energy_monthly/census.json) -> no pinning mechanism; it is a Q016-class terminal ladder with one-sided far rungs.
### Kalshi ERCOT monthly series are tiny and new
KXERCOTHOUM/NORTHM/WESTM: 0 settled events (Nov 2026 is the first), 7 rungs each, ~129 trades in 14 days concentrated on 2-3 near-money rungs. Settles on the simple average of NP6-905-CD 15-min RT SPP; free yearly history at ERCOT MIS reportTypeId 13061.
