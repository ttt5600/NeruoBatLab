# L11 -- earnings run-up straddle after real bid/ask

Question (notes/earnings.md L11; Gao, Xing & Zhang JFQA 2018 report ATM straddles bought ~3 days before earnings and
sold before the announcement earn ~+3.3%/event before costs): does buying an ATM straddle ~5 trading days before a
scheduled earnings release and selling it at the last close BEFORE the release earn > 0 after the natural bid/ask
(buy at ask, sell at bid)? No position is ever held through the release.

## 1. Coverage (Step 1, before any P&L)

**Options:** DoltHub `post-no-preference/options`, table `option_chain` (see notes/q009_options_vrp.md section 1).
Single names ARE covered: all 10 probed (AAPL MSFT NVDA AMZN META GOOGL TSLA NFLX AMD JPM) have ~130-160 rows on
2024-03-04. AAPL snapshots on 2024-03-04/06/08 (Mon/Wed/Fri) and none on Tue/Thu, first AAPL rows 2020-01-06 (66
rows). Per snapshot and name: only 2-4 expirations (e.g. AAPL 2026-07-24: 08-21, 09-04, 09-18; JPM 2020-10-05:
10-30, 11-20), i.e. ~3-8 weeks out, **the listed set rolls between snapshots** (AAPL 07-29: 08-28, 09-04, 09-18), and
a **sparse strike ladder that also differs between snapshots** (AAPL 09-04 expiry: 285/290/300/305/315/320/325/330 on
07-24 vs 285/295/305/310/320/325/330 on 07-29). So the same contract is often not quoted at exit; the rules below
handle this explicitly. Quote quality varies: AAPL 2024-03-04 near-ATM spreads ~$0.05-0.10, AAPL 2026-07-24
near-ATM spreads ~$2-4 on ~$15 mids (likely not the live intraday NBBO; spread by year is reported).
Range scans and multi-date queries time out (54 s limit); one query per date with `act_symbol IN (...)` works.

**Earnings dates + timing.** Two independent free sources, both required to agree within 3 calendar days:
(a) yfinance `get_earnings_dates` (date + ET time of the release; 2014+), (b) SEC EDGAR submissions JSON, 8-K with
item 2.02, `acceptanceDateTime` (true UTC, checked: AAPL 20:30Z = 16:30 ET, JPM 10:30Z = 06:30 ET; the copy in
data/company_graph/earn_events.parquet is shifted and is NOT used). Agreement is high: of 764 matched events the
hour cross-tab is ~all-diagonal (yf 16h vs EDGAR 16-19h: 506; yf 6-8h vs EDGAR 6-8h: 234); 5 date disagreements
(UBER 2020-05, PYPL 2020-05, COST 2022-03/09, UBER 2023-07) resolved conservatively by the rule below. 2 yf events
had no EDGAR match (dropped).

**Usable events (no prices yet):** 764 events, 28 names x ~27 quarters, release sessions 2020-01-14..2026-10-28,
522 AMC / 242 BMO. Well above the 300 bail threshold, so the test proceeds.

## 2. Pre-registration (Step 2) -- frozen before any P&L

**Universe (fixed, chosen by liquidity/size before any outcome, not from coverage of outcomes):** AAPL MSFT NVDA AMZN
META GOOGL TSLA NFLX AMD JPM BAC WMT DIS INTC BA CSCO ORCL CRM ADBE PYPL UBER COST PFE KO MU QCOM V GS (28). NKE
and XOM were in the draft list and were dropped before any chain was loaded because the prior EDGAR extract had no /
one item-2.02 8-K for them.

**Event and timing.** For each yfinance event with a reported EPS in [2020-01-06, 2026-10-28] matched to an EDGAR
item-2.02 8-K within +/-3 days: for each source time ts on day D, the latest allowed exit close is D if ts >= 16:00
ET (after the close, AMC), else D-1 (BMO or intraday; yf 00:00 = unknown -> treated as BMO). T0 = the last trading
day <= the EARLIER of the two sources' limits (so a release can never fall before the exit close). E = the first
session after T0 (the reaction session). Trading days = Yahoo SPY daily bars.

**Snapshots.** DoltHub snapshots exist Mon/Wed/Fri. Entry date = the last Mon/Wed/Fri trading day <= E - 5 trading
days; exit date = the last Mon/Wed/Fri trading day <= T0. If the name has no rows on that date, step back one more
Mon/Wed/Fri (once; exit must stay > entry). Holding period therefore 3-9 trading days. Holiday (non-trading) dates
are never used (q009 amendment A1).

**Contract.** Expiry = the nearest expiration strictly after E that is listed in BOTH the entry and exit snapshots
(the listed set is a mechanical DTE grid, so this uses data availability, not prices). Strike = among strikes with
call and put both quoted (bid > 0, ask >= bid) at entry AND both legs present at exit for that expiry, the one with
the smallest |C_mid - P_mid| at entry (the at-the-forward strike, no spot feed needed); it must be within 3% of the
entry parity forward F = K + C_mid - P_mid, else the event is dropped. Same strike and expiry for both legs. The
distribution of |K - F|/F and the drop counts are reported. Survivorship of exit quotes: drops are counted; as a
descriptive sensitivity only, events dropped for missing exit quotes on the entry-ATM strike are listed.

**Fills and fees.** Entry pays ASK (call) + ASK (put); exit receives BID (call) + BID (put) (a zero exit bid counts as
0). Exclude events with zero bids or crossed quotes at entry (by the strike rule). Fees $0.03/contract per leg per
side ($0.12 per round-trip straddle). ROI per event = (exit bids - entry asks - fees) / (entry asks). Mid-to-mid ROI
(same contracts, fee-free) is reported descriptively to show the cost of the spread.

**Trials (k = 2, bar one-sided p < 0.025 each).**
- **P1:** all events, ROI as above.
- **P2:** the subset whose entry IV (mean of the data's call and put `vol` at the chosen strike) is BELOW the median
  entry IV of that name's own previous >= 4 events in the same split (trailing, no future events; events without 4
  priors are not in P2).

**Split by date (sealed never loaded in dev).** Events sorted by E; dev = the first 70% (E < 2024-10-23, 534 events
pre-filter), sealed = E >= 2024-10-23 (230 events). The fetcher for dev requests only dates < 2024-10-23; sealed
chains are not downloaded unless a trial passes dev, then scored once.

**Bar.** For each trial: one-sided p < 0.025, p = share of 10,000 bootstrap resamples (resampling clusters = ISO
week of E) with mean ROI <= 0; AND mean ROI > 0 in both chronological halves of dev (by E). Report the OBSERVED mean
ROI with the 2.5/97.5 cluster-bootstrap percentiles (crossing 0 -> "not distinguishable"), mid-to-mid ROI, median,
win rate, worst week (sum of ROI x $1 per event that week), and size feasibility for a $1-2k account (median
straddle cost x 100, contracts per event, concurrent events per peak week).

**Too-good guard.** Mean ROI > +10%/event over > 200 events -> inspect rows (exit after release, BMO/AMC errors,
strike/expiry mismatch, stale exit quotes identical to entry) before reporting.

**Amendment A1 (2026-10-30 14:06 PT, before any P&L; budget).** The DoltHub fetch slowed to ~10 pairs/min. The
dev fetch is capped at 14:51 PT (45 min more). Dates are fetched in date order, so a capped dev set is the earliest
events up to the date reached; dev = every dev event whose entry and exit snapshots were fetched by the cap.
"Both dev halves" = the two chronological halves of the events actually scored. The fallback stage (step back one
Mon/Wed/Fri) only runs if the primary stage finishes. Failed queries (API timeouts) count as missing snapshots. The
date reached and n are reported. Sealed would be fetched under the same rules, only on a dev pass.

**Power (before results).** Expected dev n ~ 450-500 after drops, ~60-70 week clusters (events bunch in earnings
seasons). Prior per-event sd of a 5-day pre-earnings straddle ROI ~ 10-15% (natural spread adds a near-constant
negative shift); with clustering the SE is ~0.8-1.2%, so 80% power at one-sided .025 needs a true mean of ~+2.3-3.4%:
the GXZ-sized +3% gross is detectable only if the spread costs little; a net edge of ~+1% is not.

## 3. Results (Step 3) -- dev only; sealed NOT fetched (both trials failed dev)

Prereg commit 3e081d1; amendment A1 + scorer 68b46dd (both before any P&L). The fetch finished before the A1 cap
(primary + fallback stages done). 13 peak-season dates first timed out with the full ticker IN-list; they were
re-queried in 4-ticker chunks (no change to any rule), so no dev date is missing for API reasons.
`scripts/l11_earnings_straddle.py score dev`; rows `results/l11_earnings_straddle/trades_dev.csv`, numbers
`results_dev.json`; ledger rows l11_earnings_straddle_P1 / _P2 (split dev).

Dev events 534 -> 311 trades. Drops: no expiry after E listed in both snapshots 134, ATM strike > 3% from the
forward 48 (sparse ladder), no snapshot 35, entry strike not quoted at exit 6. Chosen strike differs from the
unconstrained entry-ATM strike in 56 trades. |K-F|/F median 0.72%, 90th pct 1.96%. Holding median 5 calendar days.
**Median DTE at entry = 53 days** (DoltHub lists only ~monthly expiries ~3-8 weeks out). 0 stale exit quotes.
Median relative spread of the straddle (ask-bid over mid): entry 5.0% (2020) falling to 2.5% (2023-24).

| Trial | n (weeks) | ROI natural bid/ask (observed) | 2.5 / 97.5 (week-cluster boot) | p (bar .025) | dev halves | win | ROI mid-to-mid [2.5, 97.5] | Verdict |
|---|---|---|---|---|---|---|---|---|
| P1 all | 311 (133) | **-3.95%** | [-5.34%, -2.53%] | 1.000 | -4.11% / -3.79% | 27.3% | +1.19% [-0.22%, +2.65%] | **FAIL: reliably negative** |
| P2 IV < own trailing median | 112 (56) | **-2.82%** | [-4.88%, -0.74%] | 0.997 | -4.77% / -0.87% | 30.4% | +1.70% [-0.35%, +3.73%] | **FAIL: reliably negative** |

P1 by year, natural (mid): 2020 -4.2% (+2.3%), 2021 -6.6% (-0.8%), 2022 -2.6% (+2.7%), 2023 -4.0% (-0.6%), 2024
-2.5% (+2.4%). AMC -3.7% (mid +1.4%), BMO -5.0% (mid +0.5%). sd per event 10.6%, cluster SE 0.71%. Worst event
-37.0%; worst week (sum of $1 per event) -0.90 (2022-W30, 9 events). Too-good guard not triggered.

**Mechanism.** At mid the pre-earnings run-up is small and not distinguishable from zero (+1.2%/event), the same sign
as GXZ but a third of their +3.3%, which is expected here because these are ~53-DTE options: the earnings-day
variance is spread over ~37 trading days, so the IV run-up in a far expiry is diluted (GXZ use the nearest expiry).
Crossing the spread twice costs ~5.1 points per round trip (entry ~3.4% of mid), which turns +1.2% into -4.0%.

**Size feasibility.** Median straddle cost $1,360 (1 contract each leg) -> one position is ~70-130% of a $1-2k
account; up to 9 events land in one week. Even with an edge it could not be diversified at this account size.

**Verdict: DEAD for the fund (dev fail, sealed untouched).** Not shown: whether a short-dated (weekly, nearest
expiry) version with limit orders near mid earns the GXZ premium; this data cannot test it (no weeklies, no intraday).
