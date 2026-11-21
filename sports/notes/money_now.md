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
