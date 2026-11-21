# money_now -- concrete money THIS MONTH (started 2026-11-21 04:35 UTC / 2026-11-20 PT)

Agent: money_now (research subagent of the quant-trader lead; protocol sports/notes/AGENT_PROTOCOL.md).
Scratch: sports/results/money_now/. No git, no orders, no sign-ups, no credentials.

## Question
User (California resident; has Kalshi, Robinhood, Schwab, Fidelity, Public) wants concrete money this month.
A. Which CFTC prediction-venue sign-up promos can a CA resident claim today, per the venue's OWN terms page, and how
   much can each be locked by hedging on Kalshi (after Kalshi fee ceil(0.07*C*p*(1-p)) and venue fees)?
B. Which odd-lot tender / exchange offers (SC TO-I / SC TO-T / SC 13E4) are open now with odd-lot priority, and what is
   the per-account profit on 99 shares after each broker's voluntary-corporate-action fee?
C. A daily EDGAR full-text watcher: sports/scripts/oddlot_watch.py -> ~/.claude/quant-sports/data/oddlot/filings.jsonl.

## Pre-registered inclusion rules (written before looking at any offer)
- A promo is LISTED only if its terms come from the venue's own page (URL + date read). Affiliate-only facts are
  marked UNVERIFIED and the offer is ranked below verified ones. Locked $ uses the worst case of the hedge (min over
  outcomes) with all fees; a promo whose lock is <= $0 is dropped.
- A tender is LISTED only if: offer price > current ask + per-share broker fee share, expiry >= 3 business days out
  (buy T, settle T+1, tender instruction cutoff usually 1 business day before expiry), and odd-lot priority (holders
  of < 100 shares accepted in full without proration) is stated in the filing. Dutch auctions: profit computed at the
  BOTTOM of the range (odd-lots tendered at the purchase price get priority but price is set by the auction).
- Kill: if no offer passes, say so; do not stretch.

## Do-not-redo (read 2026-11-21)
- trading_ways_survey.md: promos ranked #1 from affiliate pages only (defirate CA 2026-11-05); odd-lot #3; Fidelity
  voluntary-reorg fee ~$38 UNVERIFIED; Robinhood supports voluntary tenders.
- LEARNINGS.md: no prediction-venue promo has been claimed by the user (no key found); Webull needs 100-share min for
  $0.10-0.999 stocks (webull.min_shares_sub_dollar); Fidelity login lockout 2026-11-19 (fidelity.login.lockout).
- soft_venues.md: Polymarket US taker 0.0695*p(1-p) maker 0 (docs eff. 10-19); ProphetX fee UNVERIFIED.

## Log
- 2026-11-21 04:40Z: launched 2 general-purpose subagents for venue promo terms (A1: FanDuel/DK/Underdog/RH/Webull ->
  results/money_now/promos_A1.md; A2: Gemini/ProphetX/PM US/Crypto.com/others -> promos_A2.md). I do B + C + hedge math.

## B. Odd-lot tenders (EDGAR read 2026-11-21 ~04:40-05:10Z = 2026-11-20 PT)
Method: efts.sec.gov full-text search q="odd lot" / "odd-lot" / "odd lots" / "fewer than 100 shares", forms SC TO-I,
SC TO-T (+/A), 2026-08-01..2026-11-20; plus ALL SC TO-I/TO-T filings since 08-01 (489 TO-I docs, 42 TO-T) filtered to
filers with a ticker; plus 8-K/SC TO-C for "modified Dutch auction" / "intends to commence". SC 13E4: 0 hits (form
retired). Prices: yfinance daily close 2026-11-20 (.venv_quant).
| ticker | offer | odd-lot priority | expires | price 11-20 | status / verdict |
|---|---|---|---|---|---|
| MYPS | Dutch auction $4.30-$5.50, up to $15.0M, cash on hand, no financing cond., no min. (Offer to Purchase 11-12, ex99a1i) | YES (holders of <100 sh tendering all at/below Purchase Price) | 2026-12-10 midnight NYC | $4.94 (close 11-11 pre-offer $4.29) | OPEN but price > floor: at floor 99*(4.30-4.94) = -$63.36; at cap 99*(5.50-4.94) = +$55.44. NOT riskless -> EXCLUDED by prereg rule; speculative only |
| INVE | Dutch $2.85-$3.40, $20M | yes | 2026-11-19 5pm (EXPIRED; prelim results 11-20 SC TO-I/A No.3) | $2.19 | closed |
| UTMD | issuer tender (odd-lot text) | yes | last TO-I/A 10-29 | $69.49 | closed |
| MDT | MiniMed split-off exchange (odd-lot text) | yes | final results 425 2026-10-14 | $99.94 | closed |
| ABUS | Dutch tender, $230M returned "early October 2026" (8-K 11-10) | yes | closed | $4.02 | closed |
| SKYA | fixed $3.00 partial tender, pro rata, NO odd-lot priority | no | expired 2026-11-12, final results 11-17 | $2.18 | closed |
| GPUS | exchange 100 common -> $2 + 1 sh 12% Series I pref ($24 liq.), pro rata, min 100M sh; <100 sh may NOT participate | no | 2026-12-18 | $0.13 | pref unlisted -> value unknown; excluded |
| RWT | exchange 7.58 common -> 1 Junior Series B pref ($25 stated), up to 17.36M sh, pro rata | no | 2026-12-15 | n/a | new pref value unknown; excluded |
| ULBI | EnerSys any-and-all $16.00 cash; HSR + UK NSIA pending | n/a (no proration) | 2026-12-03 (extendable) | $15.58 | merger arb, deal/timing risk; 99*(16.00-15.58)=+$41.58 gross if closes; not odd-lot |
| AYTU | Rosemont $2.70 cash tender announced 11-16 (TO-C), not yet commenced | n/a | close "late Dec 2026" | not read | merger arb, not odd-lot |
| others (BMEZ CEF 98% NAV expired 11-18; PCRX expired 11-16; SSTI, RGR partial $44.80 expired 10-15; KDK employee option exchange; HSBC CoCo tender; BTCS pref; OTLY $4.13 vs $6.94; Cadeler redomicile) | | | | | not applicable |
**Verdict B: no open odd-lot tender passes the pre-registered rule today (2026-11-20).** MYPS is the only open odd-lot
offer and is above its floor.

Broker voluntary-corporate-action fees (official, read 2026-11-20):
- Robinhood: "Voluntary Corporate Action/Election ... $0" (RHF Standard Pricing Fee Schedule PDF,
  cdn.robinhood.com/assets/robinhood/legal/RHF+Fee+Schedule.pdf; PDFKit text; row mapping by column order).
- Schwab: "Security reorganization: Voluntary, mandatory, and post-effective $0" (schwab.com/legal/
  schwab-pricing-guide-for-individual-investors, "April 2026").
- Fidelity: "Voluntary corporate reorganization $0 Per Transaction" (fidelity.com/trading/commissions-margin-rates).
  The older $38 figure (trading_ways_survey) is superseded. Fidelity login is fragile (fidelity.login.lockout 11-19).
- Public: "Tender offer / reorg processing (domestic) $50" (public.com/disclosures/fee-schedule PDF, "updated as of
  October 13, 2026") -> never tender a 99-share odd lot from Public unless profit > $50.
