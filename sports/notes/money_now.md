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

## C. sports/scripts/oddlot_watch.py (written + tested 2026-11-21 ~04:46Z)
- stdlib only; efts full-text search for "odd lot" / "odd-lot" / "odd lots" / "fewer than 100 shares" in root forms
  SC TO-I, SC TO-T, SC TO-C (amendments come via root_forms); dedupe by accession; for new originals with a ticker,
  fetches the offer doc and regex-extracts kind (dutch/fixed/exchange), price range, expiry date, odd-lot sentence,
  under-100-excluded flag, financing-condition flag; appends to ~/.claude/quant-sports/data/oddlot/filings.jsonl.
- WRONG TURN: listing "SC TO-I/A" in the efts forms filter made efts return ONLY amendments (the MYPS original
  vanished). Fixed by passing root forms only; comment left in the code.
- Test: `--lookback 60` -> 19 filings matched, 19 new (10 with a ticker), file has 19 lines; extraction matches my
  manual read for MYPS ($4.3-5.5, DECEMBER 10, 2026), INVE ($2.85-3.4, NOVEMBER 19, 2026), GPUS (DECEMBER 18, 2026,
  UNDER100-EXCLUDED). Second run (default lookback 4) -> 5 matched, 0 new (dedupe works). Exit 0 both.
- Not installed (no launchd per brief). Suggested: daily 07:15 PT, `python3 sports/scripts/oddlot_watch.py`.

## A. Hedge math (results/money_now/hedge_calc.py; Kalshi quotes GET 2026-11-21 ~04:50Z)
Kalshi series KXNFLGAME/KXNBAGAME/KXNCAAFGAME: fee_type quadratic_with_maker_fees, multiplier 1 -> taker
ceil(0.07*C*p*(1-p)). NFL Sun 11-22 quotes (yes bid/ask): BUF .91/.92 MIA .08/.09; KC .91/.92 ARI .08/.09; CHI .84/.85
NO .15/.16; DEN .82/.83 LV .17/.18; CAR .58/.59 BAL .41/.42; SF .61/.62 MIN .38/.39 (SNF, vol 1.08M); 1c spreads,
books 10^5-10^6 contracts. NFL = Kalshi shard 0 cash; NBA = shard 3 (kalshi.shards.rebalance).
Mechanism (why stake-returned matters): a stake-returned (SR) credit B at venue price q buys B/q contracts paying $1
each, so hedging N=B/q opposite contracts on Kalshi at a=1-q' locks B*q'/q - fees ~ B. A stake-not-returned (SNR)
credit only pays the winnings B/q - B, so the lock is ~B*(1-q) - fees: buy a longshot with it. Best SR play = venue
favourite + Kalshi longshot (tiny Kalshi capital); best SNR play = venue longshot + Kalshi favourite (big capital).
Venue price assumed = Kalshi ask for the same side (+0/+1/+2c sensitivity); venue quotes not observable without accounts.
- FanDuel $20 SNR tranche, 2c/contract fee: NO@.16/CHI@.85 -> +$12.83 (+1c: 11.99, +2c: 11.28), Kalshi capital ~$77;
  BAL@.42/CAR@.59 -> +$9.82 with $15 capital. => 5 tranches ~ $49-64 IF the offer is still live.
- DraftKings $50 credit if SR (unconfirmed): BUF@.92/MIA@.09 -> +$47.92 (+1c 47.02), Kalshi capital $4.77;
  CAR@.59/BAL@.42 -> +$45.59. If SNR: NO/CHI -> +$30.04 (capital $181). => 4 x $50: ~$180-192 SR, ~$110-120 SNR.

## A. Promo terms (subagents A1/A2, raw quotes: results/money_now/promos_A1.md, promos_A2.md; all read 2026-11-20)
Official-source method: most venue pages are JS/403; A2 used each exchange's CFTC Rule 40.6 promo filings (cftc.gov PDFs)
and docs sites; A1 used official pages and, where 403, search snippets of the official domain (marked "snippet").
The affiliate leads were stale on all three venues they covered: Gemini is $40, not "up to $100" (the 50% football match was replaced 10-23); ProphetX is $40, not $75 (changed 10-29);
Polymarket US taker is now tiered (0.0695 days 1-7 for new accounts, then 0.095 below $10k/30d; eff. 11-14).

Hedge locks (hedge_calc.py + inline variants; venue price assumed = Kalshi ask for same side; Kalshi quotes 04:50Z):
| # | venue | offer (official?) | reward type / stake | locked $ (worst case) | Kalshi capital | notes |
|---|---|---|---|---|---|---|
| 1 | DraftKings Predictions | spend $5 -> $50 now + 3x$50 claim d7/14/21 (snippets of draftkings.com; end 11-29 per SportsLine UNVERIFIED) | Predictions Dollars, 1-yr, pay fees; SR "probably" (payouts->cash, no deduction clause) | SR: 4x47.92-0.48 = ~$191 (BUF .92 / MIA .09); if SNR: 4x30.04-0.48 = ~$120 | $4.77/credit (SR) | T&C bans "trading both sides of a market" ON DK + "non-bona fide" sole discretion -> unclaimed $150 at risk |
| 2 | Gemini Predictions | $40 Predict Cash after ONE >=$20 buy at 0.20-0.80 held to settlement (CFTC rules10162634421.pdf) | SR explicit; 30-day expiry | credit 38.88; qualifier must stay UNHEDGED (explicit ban) -> net +18.24..+52.24, EV ~+38 | $3.87 | maker fee 0 to 12/31 -> post qualifier as maker |
| 3 | ProphetX | trade $40 own funds -> $10 d1, $10 d8, $20 d15 (own filing 10-15) | SR implied ("withdrawn upon settlement"); 2% of net winnings | 9.69+9.69+19.39-2.29 = ~$36.5 | ~$2-4/credit | monitors "contracts priced near $1.00" -> use .80-.85 favs, not .95 |
| 4 | Novig (Ludlow) | deposit $10 -> 5x$10 credits, 1/trade, price <=0.60 (CFTC rules09162626976.pdf) | SR explicit | 5x9.00 = $45 | $6.72/credit | LIVE STATUS + CA UNVERIFIED |
| 5 | OG.com (Crypto.com DNA exch.) | trade $25 (<0.80, settle <=7d) -> $25 coupon + 100% boost cap $25 (og.com page; filing shows only $10) | coupon SR NOT STATED | coupon ~19-24 + boost 11.4-13.5 - qual 1.4-1.9 = ~$29-36 | $11-36 | CA UNVERIFIED |
| 6 | FanDuel Predicts | $20/day x5 for $1 real trade daily (homepage) BUT terms page says period ended 2026-10-20 | SNR explicit, 24h expiry, 2c/contract fee | 5x12.83 = $64 (NO .16/CHI .85) or 5x9.82 = $49 (BAL/CAR, $15 capital) | $77 or $15 | CONFIRM IN APP first |
| 7 | Webull | "$20 Voucher" for new event-contract users (banner official; T&C JS-unreadable) | UNVERIFIED (cash vs stock vs credit) | ~20-1.26 = ~$18.7 if $20 cash-equivalent | $1.89 | Webull trades ARE Kalshi contracts |
| 8 | Polymarket US | deposit >=$10 within 7d -> $25 Bonus, no trade needed (docs.polymarket.us) | SNR; Bonus absorbs first $25 of losses on any trade, survives a win | $15.30 (NO .16 on PM / CHI .85 Kalshi, S=$35) + bonus survives if PM leg wins | $164 | expiry not posted (7/10/30d) |
Dropped: Robinhood (targeted, first Derivatives account only; user already holds RH event contracts per
notes/robinhood_snapshot.md -> ineligible); Underdog (bonus "cannot be used for prediction picks"); Crypto.com app (no US
promo found); Coinbase (routes to Kalshi, no promo); PrizePicks/tastytrade/Sporttrade (none/not live); Fanatics
(affiliate-only); Kalshi referral (amount not published; credits withdrawal-restricted).
Process flag: A1 briefly fetched fanduel.com via curl with a browser UA after WebFetch 403s; the permission classifier
blocked further such fetches as circumvention and A1 switched to WebFetch/WebSearch only (temp files deleted).

### Steps (proposal only; the user places every order; recompute with results/money_now/hedge_calc.py at execution)
Rule: SR credit -> spend it on the heavy favourite at the venue and buy the SAME count of the underdog on Kalshi; go only if
venue ask + Kalshi ask <= 1.02. SNR credit -> spend it on a ~.15-.20 underdog at the venue, hedge the favourite on Kalshi
with N = winnings-only contracts. NFL hedges need Kalshi shard-0 cash; NBA shard 3. Promo income is taxable.
- DK: sign up (no code) -> $5 cash on a heavy favourite + 5 underdog on Kalshi (-$0.48) -> $50 credit: BUF 53 @<=.92 on DK,
  MIA 53 @<=.09 on Kalshi (Sun 11-22 1pm ET) -> claim each later $50 within 168h, repeat. Never take both sides on DK.
- Gemini: sign up DIRECT (not via FCM/IB) -> ONE maker buy >=$20 at .20-.80, do NOT hedge, hold to settlement -> $40 Predict
  Cash ~7d later -> favourite on Gemini + underdog on Kalshi.
- ProphetX: >=$40 of opening trades within 30d (hedge on Kalshi) -> use each installment fully as collateral on .80-.85
  favourites, hedge underdog on Kalshi; d8/d15 installments follow.
- Polymarket US: deposit $10 -> $25 Bonus -> $35 on a .15-.20 underdog, Kalshi favourite N~193 (NO/CHI example).

## B/C result summary: see sections above.

## Learnings
### Prediction-venue promo terms live in CFTC Rule 40.6 filings, not on the JS-rendered promo pages
Gemini, ProphetX, Polymarket US, Novig and OG.com all self-certify promo terms at cftc.gov/filings/orgrules/*.pdf with
verbatim reward, stake and abuse clauses (results/money_now/promos_A2.md). Affiliate pages were stale on all three venues
whose terms we checked (Gemini $100 vs $40, ProphetX $75 vs $40, PM US fee). Read the filing, not the review site.
### Stake-returned vs not decides a promo's hedged value
SR credit converts ~95-98% (DK $50 -> $47.92 with Kalshi underdog hedge); SNR converts ~B*(1-q) minus fees (FanDuel $20
-> $12.83 best, 2c/contract fee hurts longshots). Source results/money_now/hedge_calc.py; notes/money_now.md section A.
### Gemini's new-user terms explicitly ban hedging the qualifying purchase on other venues
"Hedging or offsetting a Qualifying Purchase through other accounts or related positions" voids the $40 (CFTC
rules10162634421.pdf). The qualifier must ride unhedged; only the Predict Cash leg is hedgeable.
### Odd-lot tenders are rare: 0 qualifying on 2026-11-20 out of 5 public odd-lot offers in 60 days
EDGAR full-text 09-21..11-20: MDT, UTMD, ABUS, INVE odd-lot offers all expired; MYPS (Dutch $4.30-5.50, exp 12-10) open but
stock $4.94 > floor. Watcher: sports/scripts/oddlot_watch.py. Voluntary tender fees: Robinhood/Schwab/Fidelity $0,
Public $50 (official schedules read 2026-11-20).
### EDGAR efts forms filter: pass root forms only
forms=SC TO-I,SC TO-T returns originals + amendments (root_forms); adding SC TO-I/A returned ONLY amendments and hid the
MYPS original. Encoded in oddlot_watch.py.

## Credit policy (lead, 2026-11-21; user: 'up to you'). Model: S=$50, DK all-in = price + 1.5c, Kalshi opposite = 1-c+.01 + fee
```
price | SR unhedged EV | SR hedged lock | SNR unhedged EV | SNR hedged lock   (credit $50; Kalshi opposite at 1-c+.01)
0.10 | $ 43.48 | $ 36.64 | $ 38.48 | $ 32.42
0.20 | $ 46.51 | $ 41.68 | $ 36.51 | $ 32.72
0.30 | $ 47.62 | $ 43.74 | $ 32.62 | $ 29.96
0.50 | $ 48.54 | $ 45.87 | $ 23.54 | $ 22.25
0.80 | $ 49.08 | $ 47.75 | $  9.08 | $  8.83
0.92 | $ 49.20 | $ 48.36 | $  3.20 | $  3.14
```
- SR (win pays full contract value): use the credit on a heavy favourite (.80-.95). Hedged lock is ~$48; unhedged EV is ~$49.
  Default: unhedged (user is risk-tolerant and chose not to hedge Tulsa). Hedge only if DK is priced above Kalshi.
- SNR (stake excluded): use a .10-.25 underdog. Hedged lock is ~$32-33; unhedged EV is ~$36-38 with ~80-90% of getting $0.
  Default: hedge (the variance is large for a $4 EV gain).
- Always: pick the side where DK's all-in price is at or below Kalshi's ask for the same side (cross-venue check), never both sides on DK.
- Decide SR vs SNR from the Tulsa settlement (52 ct credit-funded): $52 cash = SR, ~$2 = SNR.
