# Q009 -- SPY volatility-risk-premium (defined-risk short vol) after real bid/ask

Question (QUESTIONS.md Q009): is selling SPY index-ETF volatility through defined-risk structures +EV for a small
unlevered account after the natural bid/ask and fees? Mandate: $1-2k agentic Robinhood account, defined risk only.

## 1. Data (Step 1)

**Source used: DoltHub `post-no-preference/options`, table `option_chain`** (public Dolt database, queried through the
DoltHub SQL API, `https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master`). Columns: `date, act_symbol,
expiration, strike, call_put, bid, ask, vol, delta, gamma, theta, vega, rho`. **No underlying-price column, no
volume / open interest, no quote timestamp.** Licence: none stated on the repo page that I could verify (treat as
research use only; do not redistribute). Downloader `scripts/q009_fetch_dolthub.py` (one equality query per weekday,
<= ~1.4 req/s; range scans time out on the public API). Raw files: `data/options/dolthub/SPY_YYYY-MM.parquet`
(gitignored, untrusted, read with pandas under `python -I`).

Coverage found by probing (before any P&L):
- SPY yes; **QQQ, IWM not present** (0 rows on 2024-03-04). Single names (earnings straddles, L11) not checked in depth.
- Snapshots are **not daily**: Mon / Wed / Fri only in the probes (2024-03-05 and 03-07 return 0 rows).
- Only **3 expirations per snapshot** (e.g. on 2024-03-04: 03-18, 03-28, 04-30 = ~2 wk, ~4 wk, ~8 wk; 4 on 2026-10-28).
- A **sparse strike ladder**, ~18 puts per expiration chosen roughly by moneyness/delta, ~10 points (~2% of spot)
  apart near the money (e.g. 462/472/482/492/503/513 on the 03-28 expiry). Strikes differ between snapshots, so a
  position opened on one date often has no quote on a later date -> **only hold-to-expiry rules are testable**; 21-DTE
  management and 5-point-wide spreads are not.
- First SPY rows between 2019-09-04 (none) and 2020-01-06 (62 rows); last 2026-10-28.
- Underlying: Yahoo daily SPY OHLC, `auto_adjust=False` (`data/options/yahoo_spy/spy_daily.parquet`, 1,823 rows
  2019-08-01..2026-10-30) -- used only for the expiry settlement close and the strike-width scale.

Alternatives checked: OptionsDX (free SPY EOD files reportedly need an account login; not verified, not automatable
here); CBOE DataShop / iVolatility / historicaloptiondata.com / Intrinio = paid (bid/ask EOD for SPY sold as a product);
Yahoo = current chain only. Row-level audit results: section 4.

## 2. Pre-registration (Step 2) -- frozen before any P&L is computed

Literature basis: the variance risk premium (Carr & Wu 2009; Bakshi & Kapadia 2003) makes index puts expensive on
average; CBOE PUT / BXM-style benchmarks and practitioner 16-delta, ~45-DTE rules (no parameter here is fitted).
Known counter-evidence: most of the premium in OTM index puts is crash insurance, net of spreads it is small, and the
return is left-skewed (2020-02/03, 2022).

**Common rules (all trials).**
- Underlying SPY only; snapshots from DoltHub. Snapshot time of day unknown (see audit).
- **Entry dates:** the first snapshot date of each ISO week (one new position per week; positions overlap ~6 deep).
- **Expiration:** among those listed that day, DTE (calendar) in [21, 70], the one closest to 45 (tie -> shorter).
  No eligible expiry -> no trade that week.
- **Quote filter:** a leg is usable only if bid > 0 and ask >= bid. Crossed / zero-bid rows are not used.
- **Strike choice:** short leg = the listed option of that type and expiry with delta closest to 0.16 in absolute
  value (data's own delta column). Long (wing) leg = the listed strike further OTM whose distance from the short
  strike is closest to 2% of S_prev (S_prev = Yahoo SPY close of the trading day BEFORE entry, to avoid peeking at the
  entry-day close); the resulting width must lie in [1%, 4%] of S_prev or the side is skipped.
- **Fills:** natural prices only: sell at BID, buy at ASK. No mid fills, no price improvement.
- **Fees:** $0 commission (Robinhood) + $0.03 per contract per leg at the open (regulatory/ORF/OCC proxy). Held to
  expiration, settled at intrinsic on the Yahoo SPY unadjusted close of the expiration date (last close on or before
  it if it is not a trading day); no closing fee. Early assignment ignored (with a long wing the loss stays bounded by
  the width; flagged as a modelling caveat).
- **Unit:** $1 risk = max loss = width - credit + fees (per share basis x100). Trade ROI = P&L / max loss. Skip if
  credit <= 0.

**T1 -- put credit spread:** short ~16-delta put, long put ~2% of spot lower, the common rules above.

**T2 -- iron condor:** T1 put side + the mirror call side (short ~16-delta call, long call ~2% above). Both sides must
be valid or no trade. Credit = sum of both sides' natural credits; fees on 4 legs; max loss = max(width_p, width_c) -
credit + fees (only one side can finish ITM).

No third trial (a VIX / term-structure gate is not justified a priori strongly enough to spend a trial on it; the
earnings-straddle L11 idea needs single-name chains, out of scope here). **k = 2.**

**Split by date (sealed never loaded in dev).** List the ISO weeks that have at least one SPY snapshot from the fetch
log row counts only (no prices). Dev = the earliest 70% of those weeks (by entry date); sealed = the last 30%. A trade
belongs to the split of its entry week; dev trades whose expiry falls in sealed time are still scored in dev (their
settlement uses only the Yahoo close). The loader reads only monthly files whose month <= the dev cutoff month and
filters rows by date; the sealed months' option files are not opened in dev. Sealed is scored once, same session,
only for a trial that passes dev.

**Bar (each trial).** one-sided p < 0.05 / 2 = **0.025**; p = share of bootstrap resamples with mean ROI <= 0,
**moving-block bootstrap over entry weeks with block length 8** (covers the ~6-9 week overlap of positions), 10,000
resamples. AND mean ROI > 0 in both chronological halves of dev. Report the OBSERVED mean ROI with the 2.5/97.5
bootstrap percentiles; interval crossing 0 = "not distinguishable".

**Risk reporting (EDGE_MATH style).** Sizing f = 2% of bankroll max-loss per position (~6 concurrent -> ~12% of
bankroll at risk), plus f = 5% for contrast. Report: win rate, mean credit/width, worst trade, max drawdown of the
compounded equity path (positions realised at expiry), worst calendar month, and P(ever losing 50%) within 52 entries
from 10,000 block-bootstrap paths.

**Too-good guard.** Sharpe-like (mean/sd per trade x sqrt(52)) > 2, or ROI > +10%/trade over > 200 trades -> inspect
rows (bid/ask swap, settlement close, strike/expiry parse, survivorship) before reporting.

**Power (stated before results).** Expected dev ~ 0.7 x ~350 weeks ~ 240 trades. Prior per-trade ROI sd for a
16-delta spread ~ 0.6-0.8 (wins ~ +0.2-0.3, losses to -1); overlap inflates the SE by ~1.5x -> SE(ROI) ~ 0.06-0.07.
80%-power detection at one-sided 0.025 needs mean ROI ~ 2.8 x SE ~ **+17-20% per trade** -- far above the few-% edge
the literature suggests survives spreads. So this test can only find a large edge; a "not distinguishable" result
does NOT show the VRP is absent, only that 6 years of weekly SPY spreads cannot certify it. The actual SE is printed.

**Amendment A1 (2026-10-30, written BEFORE any P&L was computed; found while downloading).** Snapshots exist on
NYSE holidays (e.g. MLK day 2020-01-20, 2021-01-18 have full SPY rows) -- stale copies, not tradeable. Rule: the
entry snapshot of a week is the first snapshot dated on a Yahoo SPY trading day (`--fix-holidays` in the fetcher
refetches the next trading day; holiday files kept as `holiday_SPY_*.parquet` for the audit). Also: files are one
per week (`SPY_wk_<monday>.parquet`), not per month; the dev loader opens only weeks before the first sealed week.
Coverage starts 2020-01-06 (2019-09..12 empty or too slow to probe: empty dates take ~40 s each on the public API).

## 3. Results (Step 3)

(filled in after the prereg commit)
