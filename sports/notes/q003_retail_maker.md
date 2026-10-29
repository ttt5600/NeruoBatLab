# Q003 retail maker: resting NO bids in Kalshi Mentions (DEV, 2026-10-28)

## Verdict
**DEV LEAD at D6, DEAD at D24. The edge is decaying, so this is a sealed ask and nothing more.** fund_stats:
MID_d6 LEAD (BH q .003) and IN1_d6 LEAD (BH q < .001) both pass the pre-registered bar of p < .0125 with both halves
positive. MID_d24 and IN1_d24 are DEAD because their halves disagree in sign.
Unlike K-FAV, adverse selection is small (0.3-1.0c per contract). The money comes from YES being overpriced, plus the
spread a maker saves. The worry is the trend: the second half (anchor >= 2026-02-22) earns only +1.9% / +2.9% at D6,
against +10-12% in the first half. At D24 the second half is negative. That matches M3 dying in sealed. No orders
were placed and nothing is committed.

## Results (DEV sample, anchor 2025-02..2026-06-25, 1,240 tickers / 851 events, event-clustered bootstrap 5,000)
| trial | quoted | fills (rate) | ROI [95% CI] | p | half1 / half2 ROI (fills) | c/fill | YES rate filled / unfilled / break-even | adverse sel. | taker same mkts |
|---|---|---|---|---|---|---|---|---|---|
| MID_d24 | 748 | 673 (90%) | +5.85% [+1.9, +9.6] | .0014 | +12.2% (358) / **-1.2%** (315) | +4.60 | .168 / .147 / .214 | 0.30c | +1.63% |
| **MID_d6** | 1,108 | 897 (81%) | **+5.78%** [+2.7, +8.8] | .0002 | +10.1% (429) / +1.9% (468) | +4.57 | .164 / .142 / .210 | 0.27c | +1.83% |
| IN1_d24 | 748 | 643 (86%) | +6.57% [+2.5, +10.6] | .0010 | +13.8% (346) / **-1.7%** (297) | +5.09 | .174 / .114 / .225 | 0.96c | +1.63% |
| **IN1_d6** | 1,108 | 859 (78%) | **+7.08%** [+3.9, +10.2] | <.0002 | +11.5% (415) / +2.9% (444) | +5.52 | .164 / .145 / .219 | 0.32c | +1.83% |

- Adverse selection = P&L per contract if every quoted market had filled at q, minus the actual P&L per fill.
  Filled markets resolve YES 1-6 points more often than unfilled ones, but every filled YES rate (.16-.17) is well
  under break-even (.21-.22).
- Sensitivities (these are not trials): a 1-contract touch fill gives +6.2 to +7.7%. Resting until close (through the
  live event) gives +5.0 to +6.0%. The 0.0175 maker fee costs about 0.4 points of ROI.
- By group (IN1_d6): other +11.6% (352 fills), sports announcers +3.3% (333), earnings -1.1% (101), fed +12.9% (32),
  trump +15.4% (41). Earnings and sports are about flat, and those two groups are growing.

## Design (pre-registered in scripts/q003_retail_maker.py docstring before tapes or outcomes)
Universe: Mentions binary word markets. Quotes come from M3's frozen hourly candles at D24/D6, with the same filters.
YES mid had to be 5-35c. Split: DEV = anchor < 70th percentile (2026-06-25); Q003-SEALED is after that and was never
loaded. Sample: 1,500 random DEV tickers. Quote rules: MID = floor(NO mid); IN1 = NO best bid + 1c; both are capped at
NO ask - 1c. Size is 10 contracts, and the order is cancelled at anchor - 2h. Fill model, conservative about queue
position: a quote inside the spread fills if at least 10 contracts print at no_price <= q. A quote that joins the
existing best bid fills only if at least 10 contracts trade through it (strictly below q). Positions are held to
settlement. Maker fee is 0, since every Mentions series is 'quadratic'. Ledger: 4 trials, dataset q003_retail_maker.

## Caveats
- **Budget cap:** 258 of the 1,500 sampled tickers were *not scored: data incomplete* (pull stopped by the lead). They
  are the last ones alphabetically: KXTRUMPMENTION 179, KXWCMENTION 86, KXVANCEMENTION 35 and others. So Trump is
  under-represented in the scored set.
- A bug was caught before scoring: the first pull used `astype(int64)//1e9` on [us] timestamps, the known
  datetime-unit trap, and returned 0 trades. It was fixed and re-pulled. No P&L had been computed.
- Quotes are the last hourly candle at or before D, so they can be hours stale. An order "inside the spread" may in
  reality have joined or crossed the book.
- The tape fill is counterfactual: our order would have changed who traded.
- The DEV window overlaps the M3 sealed window (2025-10 onward), whose outcomes were already read for taker NO. So this
  is not virgin data for the "mention NO" idea.
- The halves are decaying, and the taker-NO baseline is positive here (+1.6-1.8%) in a band that M3 sealed did not
  cover.

## Sealed ask (needs the user's explicit "yes Q003")
IN1_d6 as the single promotion (MID_d6 reported alongside). The test set is the Q003-SEALED anchors >= 2026-06-25, the
same rules, a fresh random sample of up to 1,500 tickers, and one shot. It passes if ROI > 0 and the one-sided event
bootstrap gives p < .05.

## SEALED RESULT 2026-10-29 (user "yes Q003"; scorer 08e539b committed before tapes/outcomes)
SEALED FAIL 10-29 (lockbox q003_IN1_d6_sealed): IN1_d6 920/1,500 fills, ROI +1.56% [-1.86,+4.84] p .18 (bar .05); halves +3.30%/-0.01%; adverse selection 2.70c (YES filled .192 vs unfilled .131); MID_d6 +1.29% p .21. DEAD -- dev +7.08% decayed out; do not redo.
Lesson: a dev edge whose halves decay (+10-12% -> +2-3%) arrives at sealed at ~0; adverse selection grew from 0.3-1c (dev) to 2.7c.
