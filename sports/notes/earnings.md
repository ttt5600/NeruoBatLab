# Earnings season 2026-10-21: Bloom Energy, the calendar, beat contracts, vol (user ask 2026-10-21)

Bottom line: **no earnings rule has passed dev -> sealed. Nothing here is a backtested edge.** Anything you trade is a
judgment or paper bet, sized by notes/risk.md. The one backtest I could run, on 971 resolved Polymarket "beat" markets
(dev), found the market **calibrated**: average price 0.732, realized YES rate 0.743. All three pre-registered rules
failed the bar. The market predicts beats better than the base rate does: Brier 0.155 vs 0.191 for a constant 0.75.
"75% of companies beat" is already in the price.

Sources: Yahoo via yfinance (scripts/earnings_moves.py); Kalshi public API (scripts/earnings_kalshi_scan.py); Polymarket
gamma + CLOB public data (scripts/earnings_poly.py); SEC EDGAR 8-K item 2.02 timestamps; a web-research agent (URLs
below). Raw data is in data/earnings/ (gitignored). Every number below comes from one of these sources. Anything marked
UNVERIFIED was not cross-checked.

## 1. Bloom Energy (BE): the specific answer

**Date.** Thu **2026-10-22, after the close**. The call is at 5:00 pm ET. The user's date is correct. Confirmed by the
company release (bloomenergy.com/news/bloom-energy-to-announce-third-quarter-2026-financial-results-on-october-22-2026/)
and Yahoo (16:00 ET). EDGAR shows every 8-K 2.02 since 2022 accepted at about 16:00-17:20 ET, so Bloom always reports
after the close.

**Consensus EPS (non-GAAP).** The figure depends on the source, and so does what counts as a "beat":
- Yahoo: **$0.672**, 17 analysts, range 0.59-0.80.
- Nasdaq/Zacks: **$0.56**, 5 estimates.
- MarketBeat/TipRanks: $0.69.

**Consensus revenue.** Yahoo **$1.072B** (range 0.910-1.127B); MarketBeat about $1.06B.

**Contracts.** **None on Kalshi or Polymarket.** Kalshi has no BE market of any kind, and Polymarket has no BE beat
market. Only the stock and options can be traded.

| Evidence | Value | Direction |
|---|---|---|
| EPS beat record (Yahoo street EPS, last 12 qtrs, oldest first) | BmmmmBBBBBBB: 8/12, **7 straight beats** | up |
| Size of the last 7 surprises | +40% to +457% (Yahoo) | up: the beats were large, not marginal |
| Q2-26 (Jul 28) | EPS 0.78 vs ~0.41; revenue $1,065M vs ~$826-851M; FY guide raised | up |
| Estimate revisions | Q3 consensus 0.52 -> 0.672 over 90 days, flat over the last 30 | down: the bar rose after Q2, the opposite of the "walk-down to a beatable number" pattern (Richardson-Teoh-Wysocki 2004) |
| Guidance vs consensus | FY26 non-GAAP EPS guide 2.55-2.85; H1 actual 1.22 + Q3 0.67 + Q4 0.83 = 2.72 ≈ guide midpoint | neutral |
| Revenue bar | FY guide $3.9-4.2B; H1 actual $1,816M + consensus Q3 1,072 + Q4 1,214 = $4.10B, near the **top** of the guide; Q3 consensus ≈ Q2 actual (flat QoQ) | down for revenue |
| Market-wide | FactSet Q3-26 to date (9% reported): 84% beat EPS, 67% beat revenue; 5-yr average 78% / 70% | base rate |

**P(beat EPS vs the Yahoo $0.67 consensus): about 0.78** (plausible range 0.70-0.85; low-moderate confidence). The prior
is the 78% FactSet base rate. The streak and the size of past beats push it up; the higher bar after the estimate jump
pushes it down. Persistence of beats is documented (Bernard-Thomas 1990), but I have not measured how far it should
move this one name.
Against the Zacks $0.56: about 0.85. **None of this is backtested.** It is judgment.

**P(beat revenue vs $1.07B): about 0.65** (range 0.55-0.75; low confidence). Revenue is lumpy and depends on project
timing (Q1-26 $751M came in below Q4-25 $778M). Consensus already sits near the top of the FY guide. I have no verified
series of past revenue surprises for BE.

**Volatility.**
- *Implied move.* The 10-23 ATM straddle (K 237.5) costs $23.45 mid, about **9.8% of spot $238.5**. Yahoo chain,
  10-21 close; bid/ask spread $0.20 in total. That straddle also covers the normal 10-22 session, and BE's normal days
  are large (last 60 days: mean |daily return| 4.3%). Netting out one normal day (straddle ≈ 0.8σ) leaves an
  **earnings-only implied move of about 8.9%**. That is an approximation.
- *Realized moves.* The last 12 report-day moves were +8.4, −17.5, −0.5, +9.0, +23.4, +4.3, −8.2, −1.8, +18.0, +4.7,
  +27.2, −1.9 %. Mean |move| 10.4%, median 8.3%. Up 7 of 12.
- *Date error.* Yahoo misdates the May 2024 report by a day. EDGAR shows 05-09 after the close, so the true move is
  −4.9%, not −0.5%. With that fixed, the mean is about 10.8%.
- *Comparison.* Implied ≈ 0.9-0.94 × realized history. BE is one of the few names where options are **not** rich against
  history. With n = 12 and a standard error of about 2.3 points on the mean, the gap is noise.
- *Direction.* A beat does not mean the stock goes up. Jul 28 was a 90% EPS beat with raised guidance, and the next-day
  move was −1.9% (EDGAR 16:07 ET; the +26.5% came on 07-30, two days later). FactSet this season: companies that beat
  have moved −0.2% on average.
- *What a "positive swing" requires.* For a long straddle to pay off, the move by Friday's close must exceed about
  **±9.8%** (stock below ~215 or above ~262). History shows 4 of 12 report-day moves beyond 9.8%, but the average
  |move| is ≈ the price because the big moves are huge. Expected value is about zero, minus spread and the risk of
  mistiming. A long call alone needs the move up **and** larger than its premium (~$11.2 for the 240 call, ≈ +5%).
  P(up) is about 0.58 from 12 points, which is near a coin flip.

**BE verdict:**
- No +EV contract exists, because there are no prediction-market contracts on BE.
- Options are priced about at historical realized moves. Long vol is a fair, high-variance coin, **not +EV on the
  evidence**. Selling vol here is not supported either: implied is not above history for this name.
- Confidence that there is no edge: moderate-high. Confidence in P(beat) 0.78: low-moderate.
- If you trade it anyway, it is a judgment bet at ≤1% of bankroll (B), all-in with a defined maximum loss, i.e. a
  long-premium structure only.

## 2. Calendar (next ~10 trading days)

Sources: Nasdaq calendar API (Zacks consensus, n = number of estimates), pulled by the agent; Yahoo for times and the
"street" consensus; Polymarket strike = the consensus "as of market creation".

| Date | BMO (before open) | AMC (after close) |
|---|---|---|
| Tue 10-21 | — | META $4.98, TSLA $0.22 (already out) |
| **Thu 10-22** | PG 1.89, UNP 3.47, LMT 7.49, HON 2.17, CBRE 1.98, PCG 0.41, AAL −0.28, BX, DOW, GD (Polymarket says 10-22; Yahoo says 10-30), VLO, SPOT, CMCSA, NSC | **BE 0.56 (Zacks) / 0.67 (Yahoo)**, SSNC 1.59, WDC, NEM |
| Fri 10-23 | AXP 4.50 (Yahoo) | — |
| Mon 10-26 | VZ 1.27 (Yahoo) | EA (Polymarket 10-26; Yahoo 11-03: conflict) |
| Tue 10-27 | BA −0.21, SPGI 4.43, HLT 2.34, PYPL 1.31, SOFI 0.17, IVZ 0.74, KO 0.88, RCL 6.38, UPS 1.62, ARCC, JBLU, PJT (Polymarket 10-27; Nasdaq 11-03: conflict) | V 3.43, CSGP 0.28, CZR −0.09, HOOD 0.55, NXPI 4.12, STX, WM, CFLT (UNVERIFIED, not on Nasdaq) |
| Wed 10-28 | ETSY, GRMN, LH, WING, GEV, APH | GOOGL 2.94, MSFT 4.71, EBAY 1.11, TDOC −0.23, CMG, CVNA, F, MGM, ORLY, TMUS, KLAC, NOW, ROKU (Polymarket 10-28; Yahoo 11-02: conflict) |
| Thu 10-29 | LLY 5.38 (Zacks, n = 3; Yahoo 6.67: large disagreement), MRK 2.12, CAT 7.12, ICE 1.97, HGV, HOG, ARES, TW, MA, LIN | AMZN 2.02, INTC 0.28, RDDT 1.30, AAPL (Kalshi mention market dates it 10-29; Nasdaq 11-02: conflict), RBLX, MSTR, SNDK, SBUX, RIVN, MNST, GILD |
| Fri 10-30 | XOM 3.93, CVX 4.78, CL 0.92, CBOE 3.40, ABBV, REGN | — |
| Mon 11-02 | — | PLTR 0.34, (AAPL per Nasdaq) |
| Tue-Thu 11-03..05 | KHC 0.43 (11-04), IMAX (date conflict: 10-22 / 10-28 / 11-05), TFX (conflict: 10-22 Polymarket, 10-29 Yahoo, 11-05 Nasdaq) | NET 0.02, DKNG −0.23 (11-05) |

Date conflicts between sources are common. Before trading anything, check the company IR release or the EDGAR 8-K
history pattern. Yahoo was wrong by one day for BE in May 2024.

## 3. Tradable instruments

- **Kalshi: no EPS-beat contracts.** I checked all 15,336 series on 2026-10-21. The earnings content is about 190
  `KXEARNINGSMENTION*` word-mention markets ("will the CEO say X on the call") plus annual company-KPI series
  (deliveries, subscribers, headcount).
  - Mention events in the window: TSLA (117k contracts traded), INTC 10-22 (15k), LMT/PG 10-22, AXP and ALK 10-23,
    WING/SFM/CVNA 10-28, AMZN/GOOGL/AAPL/RDDT 10-29, PLTR/HIMS 11-02, PGR/CAVA 11-03, MCD 11-04.
  - Median YES spread is 1-3c on the busy ones and 6-9c on the rest. Top-of-book size is tens of contracts (e.g. INTC
    "NVIDIA" bid 0.50 × 30 / ask 0.53 × 21).
  - None of these is a beat or volatility instrument.
  - Kalshi company-report pages show consensus EPS (kalshi.com/company-reports/MA). Whether EPS markets are listed there
    is UNVERIFIED.
- **Polymarket: "Will X beat quarterly earnings?" exists on the international venue.** There are 95 open markets
  (10-21..11-05) and 1,403 closed since 2025-09. Every one is flagged `restricted: true`. I found **no evidence that
  Polymarket US lists them**; it launched with sports, then politics and economics (defirate.com). Treat them as **data
  only, not tradable for us.** Never route around the geo-block.
  - Books are thin: median liquidity about $500, volume $0.5-4k per market. Ticks are 1c, and typical spreads are 1-5c
    on the liquid ones (LMT 0.56/0.59, KO 0.72/0.73, MSFT 0.75/0.79, GOOGL 0.32/0.36).
  - Contract terms: YES iff reported EPS > the strike. **An in-line result resolves NO.** The strike is the consensus
    when the market was created, on either a **GAAP or a non-GAAP** basis.
- **Options (BE and every name above):** Yahoo chains are free. These are the only volatility instrument.

## 4. Base rates and the backtest

- **Market-wide.** FactSet (Earnings Insight 10-16-26): Q3-26 to date 84% beat / 4% in-line / 11% miss on EPS; 67% beat
  revenue. 5-yr averages 78% and 70%; 10-yr 77% and 68%. Q2-26 final: 87% / 77%.
- **Our own sample.** 74 names, 877 name-quarters (Yahoo, strict >): **75.7%** beat EPS. This agrees with FactSet, so
  the "~70-80%" figure is verified.
- **Per-name EPS history** (last 12 quarters):
  - Consistent beaters: PG, HON, CBRE, HLT, EBAY, GOOGL, MSFT, CL, LH, TDOC, PJT, AAPL, ROKU: 12/12.
  - Frequent missers: MSTR 2/12, CZR 2/12, ARCC 3/12, ORLY 4/12, SBUX 4/12, ETSY 4/12, HGV 5/12.
  - Full table: data/earnings/moves.json.
- **Predictors in the literature** (agent, with URLs):
  - Estimate walk-down / guidance management: Richardson-Teoh-Wysocki 2004; Matsumoto 2002.
  - Crowd estimates (Estimize) add accuracy to IBES consensus: Jame et al. 2016 JAR.
  - Drift after surprises / persistence of surprises: Bernard-Thomas 1990 (market-tested).
  - Strings of beats occur more often than chance: Myers-Myers-Skinner 2007 (the counts are UNVERIFIED).
  - Zacks ESP: ~70% vs 52% (vendor claim, not independent).
  - None of these has been tested by us.
- **Backtest (scripts/earnings_poly.py, pre-registered in the docstring before any outcome was loaded, ledger trials 1-3
  on dataset `poly_earnings_beat`):**
  - Decision is the price at endDate − 24 h. Cost is price + 0.02. Dev = before 2026-07-01 (971 markets, 150 report
    days). **Sealed = 2026-07-01 onward (385 markets), UNSCORED.**
  - Calibration (dev):

    | price bucket | n | mean price | YES rate |
    |---|---|---|---|
    | < 0.40 | 108 | 0.272 | 0.324 |
    | 0.40-0.60 | 96 | 0.510 | 0.573 |
    | 0.60-0.75 | 172 | 0.684 | 0.657 |
    | 0.75-0.85 | 227 | 0.803 | 0.819 |
    | ≥ 0.85 | 368 | 0.904 | 0.902 |

  - E1 buy YES on every market: **+0.07%/971, p .50**. The base rate is priced in.
  - E2 buy YES when price ≤ 0.60: **+11.4%/206, CI −6.3..+32.3%, p .12**. Fails the bar of .0167. It is the only lead,
    with weak evidence: the cheap buckets resolve YES 5-6 points above their price.
  - E3 buy NO when price ≥ 0.85: **−18.6%/368, p .91**. Dead.
  - Descriptive, not trials: GAAP-strike markets average price 0.657 and resolve YES 0.635; non-GAAP average 0.777 and
    resolve YES 0.807.
- **Why a naive "per-name beat rate" model is wrong here.** For the open markets I shrank each name's 12-quarter beat
  rate toward 0.78. That model disagrees with the market by 20-70 points on names such as GOOGL, CL, MGM, ARES, TW, BX,
  CBOE and CVNA. In nearly every large gap, the **strike is not today's consensus**:
  - CL: strike 0.96 vs Yahoo 0.93.
  - MGM: 0.18 vs 0.10.
  - ARES: 1.34 vs 1.26.
  - BX: GAAP 1.40 vs street 1.26.
  - XOM: 3.87 vs 4.04, priced 0.95.

  A strike set above (or below) the current consensus, or a GAAP strike on a company whose GAAP EPS runs below its
  street EPS, explains the price. The market has already done this. The model has not.

## 5. Volatility: implied vs realized

- **Today's cross-section.** Over 48 names with an expiry ≤ 4 days after the report, the median implied/historical
  mean-|move| ratio is **1.33**; 90% of names have a ratio > 1. Caveats: the ratio is biased upward because the straddle
  also covers normal days, and the history is only 12 points. This is **not a test.**
- **The literature is mixed, and the user's premise needs a correction.**
  - Volatility risk premia in general make straddles lose money on average.
  - **But** Gao, Xing & Zhang (JFQA 2018) find that ATM straddles bought 3 days before earnings earned **+3.34%** on
    average (significant), especially for small, volatile firms. Options *under*-priced the earnings jump in their
    sample.
  - ORATS (vendor) reports earnings straddles averaging about −2% over the 12 quarters to Feb 2026.
  - Dubinsky et al. (RFS 2019) show the announcement-day risk options price is large and informative. Their abstract
    makes no claim that it is systematically over-priced.
  - So "buying vol into earnings is usually −EV" is the usual rule of thumb and holds for many large caps (IV crush:
    implied vol collapses once the number is out). It is **not** a settled result. I could not test it: historical
    option prices are not free.
- **What a "positive swing" needs.** Long straddle P&L ≈ |realized move| − straddle cost − spread. It is +EV only if the
  expected |move| exceeds the implied move. Long calls also need the direction to be right.

## 6. EV table (Polymarket prices are international and restricted: NOT TRADABLE by us; shown for completeness)

"Naive p" = the name's 12-quarter beat rate shrunk toward 0.78 (k = 8). That is exactly the base-rate model the dev
backtest shows is beaten by the market (Brier 0.191 vs 0.155). The **honest p is the market mid** until a model beats
it out of sample.

| Name (date) | strike / current consensus | YES bid/ask | naive p | ev_check (naive p, polymarket) | honest view |
|---|---|---|---|---|---|
| KO (10-27) | 0.91 / 0.88 | 0.72/0.73 | 0.862 | EV +16.5%; judgment tier would "BET $10" | strike is above consensus -> naive p overstates; **paper** |
| RCL (10-27) | 6.44 / 6.38 | 0.72/0.73 | 0.862 | EV +16.5% | strike is above consensus; **paper** |
| LLY (10-29) | 7.49 / Yahoo 6.67, Zacks 5.38 | 0.60/0.62 | 0.812 | EV +28.9% | consensus sources disagree; **paper** |
| SPGI (10-27) | 4.51 / 4.42 | 0.56/0.59 | 0.762 | EV +27.0% | strike is above consensus; **paper** |
| TFX | 1.92 / 1.89 | 0.70/0.72 | 0.862 | EV +18.1% | report date unclear (3 sources disagree); **paper** |
| SSNC (10-22) | 1.80 / 1.77 | 0.64/0.69 | 0.862 | EV +23.1% | strike is above consensus; **paper** |
| LMT (10-22) | GAAP 7.51 / 7.52 | 0.56/0.59 | 0.712 | EV +18.7%; at p = mid 0.575: **EV −4.2%, NO BET** | GAAP strike; **paper** |
| BE (10-22) | — | no contract | 0.78 (judgment) | n/a | options only, ~fair |

ev_check output, verbatim pattern:
`fair p 0.8620 | cost 0.7400 (decimal 1.351) | EV +16.49% per $ ...`
- `--tier paper`: `VERDICT: NO BET -- edge below the 'paper' minimum`.
- `--tier judgment`: `BET $10.00 (1.00% of $1,000)`.

**Read the judgment-tier "BET" lines as a warning, not a recommendation.** The tool trusts the p you give it, and here p
comes from a model the dev data rejects. **Correct tier for every row: paper.** No earnings rule has passed
dev -> sealed.

## 7. What would make this a real edge

1. **Model the strike, not the base rate.** For each market, record strike vs consensus at decision time, the
   GAAP/non-GAAP basis, the name's historical surprise distribution (% surprise, from Yahoo/Nasdaq), and the revision
   in the last 30 days. Then compute P(actual > strike) = P(surprise% > (strike − consensus)/|consensus|) from the
   name's empirical surprise distribution, shrunk toward the sector distribution.
   - Data needed: point-in-time consensus. Yahoo eps_trend gives 7/30/60/90-day snapshots *only going forward*, so start
     logging daily now. A cheaper path is Nasdaq's earnings-surprise API plus our own daily snapshot logger.
2. **Pre-register "E4: strike-aware model".** Buy the side where |model − ask| ≥ 0.08.
   - Dev on Polymarket closed markets before 2026-07-01. The point-in-time consensus can only be reconstructed
     approximately: the strike itself is the creation-date consensus, and the gap to the actual figure is unknown. This
     is the hard part.
   - **Sealed = 2026-07+ (385 markets, untouched): one shot on the user's yes.**
   - E2 (cheap YES, +11.4% p .12) can be re-registered as a sealed-only test if the user wants: it was pre-registered,
     so a sealed run is legitimate. The dev evidence is weak, though.
3. **Tradability first.** Confirm whether Polymarket US lists earnings-beat markets (docs.polymarket.us needs keys;
   unverified). If it does not, the beat-market work is research only. Kalshi mention markets are tradable and liquid on
   mega-caps (TSLA 117k contracts). A separate pre-registered backtest of settled mention markets (favourite-longshot on
   words) is the US-legal earnings angle. Not done yet.
4. **Volatility.** Historical option prices are needed to test straddles (ORATS/CBOE DataShop are paid). The free proxy
   is to snapshot Yahoo straddles daily from now on, before every report, for 2 seasons (~1,000 events), then score a
   pre-registered "buy straddle when implied/hist < 0.9" rule (BE qualifies today).
