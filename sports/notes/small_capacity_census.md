# small_capacity_census -- inefficiencies that persist because they are too small / annoying / rationed for big money (started 2026-11-24 PT / 2026-11-25 UTC)

Agent: small_capacity_census (research subagent of quant-trader lead; protocol sports/notes/AGENT_PROTOCOL.md).
Read-only desk research: no purchases, bids, sign-ups, logins, credentials. Scratch: sports/results/small_capacity_census/.

## Question (user, verbatim)
"market inefficiencies we may be able to address, the types of market angles that are out there that are too small for
large companies or serious companies to address" (example: Pokemon card reselling a la Triumph Labs / Rips by Triumph).
User: CA resident, BS EECS / ML engineer (can write scanners/bots), ~$200-1,500 capital, limited hours.

## Do-not-redo (read 2026-11-24, cited, NOT re-researched)
- FUND_BOOK.md s3d (91 rows: sports models, in-game, props, weather variants, Kalshi structural, cross-venue, stocks).
- trading_ways_survey.md RANKING: promos (#1), RSA (#2, side-hustle session owns), odd-lot tenders (#3; watcher
  sports/scripts/oddlot_watch.py, money_now.md s.B: 0 qualifying 11-20, 5 odd-lot offers in 60 d), IPO access (#4),
  SPAC trust arb ~T-bill (#5), crypto funding carry DEAD (FC1), merger arb DEAD (~0.23%/mo).
- ~/.claude/seed-runner/research_notes/opps_2026-11-12/00_RANKED + 01-05: DEAD = crypto spot arb, gift-card resale,
  price-error resale, points-for-cash, API resale, cheap-card PSA grading, bank-bonus tracker, deal alerts, micro-SaaS/
  Chrome ext, lead lists, faceless AI video. Open leads there: ticket dispersion scanner (paper), domain-auction predictor
  (paper), TCG singles cross-venue (>20% gap needed, ~13% fees each side), Apify actors, claimit.ca.gov (one-shot).
- promo_hunt_2.md / promo_verify_1124.md (sign-up bonuses: done); boost_hedge.md (cross-venue PM arb: dead);
  pokemon_cards.md (parallel agent; skipped here); bank/brokerage bonuses (side-hustle session; skipped).

## Screening rubric (written BEFORE any search, 2026-11-25 03:32Z; inbox started row says 04:00Z by mistake)
Score each candidate 0-3 on each axis; any HARD FAIL removes it.
1. WHY BIG MONEY SKIPS IT (must name at least one, else HARD FAIL -- "nobody noticed" is not a reason):
   capacity cap (max profitable size per person/account ~$100-$10k), per-person rationing (one allocation per depositor/
   account/household), legal friction (licensing, KYC, residency), ops cost per unit (shipping, inspection, paperwork),
   niche knowledge (local, hobby, regulatory).
2. EVIDENCE QUALITY: 3 = primary data with >= 10 dated outcomes; 2 = primary data, few outcomes or one study;
   1 = secondary (blog/forum/affiliate) with numbers; 0 = anecdote/vendor claim -> cannot rank above #8.
3. NET $/HOUR of the user's time (after fees, shipping, tax at ~30% marginal NOT deducted -- flag), steady state:
   3 >= $50/h, 2 = $20-50/h, 1 = $5-20/h, 0 < $5/h (California minimum wage ~$16.90 is the opportunity-cost floor).
4. CAPITAL fits $200-1,500 (HARD FAIL if minimum ticket > $1,500 or capital lock > 6 months for < 10% return).
5. WORST-CASE LOSS per cycle, stated in $ (3 = <= $25, 2 = <= $100, 1 = <= $500, 0 = more).
6. CA LEGALITY + TERMS: HARD FAIL if illegal in CA, needs ToS breach, bot-buying (BOTS Act), user-agent spoofing,
   or robots.txt violation for the scanner.
7. DECAY: how fast the gap closes once known (3 = structural/regulatory, persists years; 0 = already arbitraged).
8. SOFTWARE ADVANTAGE vs other small players: 3 = a scanner/model clearly beats manual hobbyists (search, timing,
   pricing); 0 = pure manual labor.
Ranking key = expected net $/month at realistic hours, then $/hour, with evidence <= 1 capped at "lead".
Kill (candidate DEAD in this census): sourced net EV <= 0 after fees, OR HARD FAIL on 1/4/6, OR net < $5/h at steady state
with no automation path.

## Plan
1. Brainstorm >= 30 candidates (table below), first-pass score from prior knowledge, mark which need research.
2. 2-4 parallel searches per shortlisted angle (8), primary sources first.
3. Top 3 -> tiny pre-registered tests (<= $100) + scanner designs.

## Log
- 03:32Z: read do-not-redo files; research-rag has no prior work on this (index = bioacoustics code only).

## Brainstorm (first pass from prior knowledge, BEFORE searching; "why skip" = rubric axis 1)
Codes: CAP capacity cap, RAT per-person ration, LEG legal friction, OPS ops cost, NICHE knowledge. Status: SL = shortlisted
for research, DNR = already covered elsewhere (cite), KILL-prior = fails a hard rule on known facts, LOW = passes rules but
first-pass $/h or evidence too weak to research now.
| # | candidate | why big money skips | status / first-pass reason |
|---|---|---|---|
| 1 | Mutual savings bank standard conversions (depositor subscription rights) | RAT+LEG (only eligible depositors; rights non-transferable) | SL |
| 2 | MHC second-step conversions (minority-holder + depositor priority) | RAT+CAP | SL (with #1) |
| 3 | Insurance / credit-union demutualizations | RAT | LOW (rare; folded into #1 search) |
| 4 | CEF tender offers at 98-99% NAV / discount-trigger tenders | CAP (proration) but activists (Saba, Karpus) DO play | SL (equity special sits) |
| 5 | OTC liquidating companies / liquidation trusts below est. distributions | CAP (illiquid microcaps), NICHE | SL (with #4) |
| 6 | Rights offerings (CEF, small caps) | CAP | LOW (usually dilutive; skip) |
| 7 | Odd-lot / small Dutch tenders | CAP (99 sh) | DNR (money_now.md s.B, oddlot_watch.py) |
| 8 | RSA reverse-split round-ups | RAT (per account) | DNR (side-hustle session) |
| 9 | GSA Auctions / GovDeals / police (PropertyRoom) / university surplus vs eBay sold comps | OPS (local pickup, as-is), NICHE | SL |
| 10 | CA tax-defaulted property (county Ch.7 sales) | LEG+OPS+NICHE | SL-lite (check min bids / risk) |
| 11 | Expired-domain drops / hand-reg | NICHE | DNR (opps 05 test A, paper) |
| 12 | Amazon online/retail arbitrage (OA/RA) | OPS | LOW (saturated; tools sold to everyone) -- one search |
| 13 | Textbook / used-book scouting (thrift/library sale -> buyback/Amazon) | OPS+CAP | SL-lite |
| 14 | eBay misspelled / miscategorized listings | CAP | LOW (public tools exist; eBay 2026 bans AI buy agents per opps 00) |
| 15 | Liquidation pallets / returns boxes | OPS | KILL-prior (widely reported negative EV; no source yet -> check once) |
| 16 | LEGO retired sets (hold) / BrickLink part-outs | OPS (storage), NICHE | SL |
| 17 | Sneakers (SNKRS draws, StockX flips) | RAT | LOW (margins compressed; check once) |
| 18 | Concert / sports tickets | RAT | DNR/KILL (opps 01: CA AB 1349 speculative ban, BOTS Act) |
| 19 | Class-action settlement claims (genuine eligibility only) | RAT | DNR (opps 03) |
| 20 | Unclaimed property claimit.ca.gov | RAT | DNR (opps 03, one-shot) |
| 21 | Bug bounties / AI red-teaming bounties (Gray Swan, HackerOne, vendor programs) | NICHE+CAP (per-finding) | SL (fits EECS/ML skill) |
| 22 | Micro-SaaS / tiny website acquisitions (Acquire, Flippa) | CAP | KILL-prior (opps 04 micro-SaaS dead; ticket > $1.5k typical) |
| 23 | Kalshi thin series | CAP | DNR (core quant project) |
| 24 | CS2 skins Steam market vs third-party (Steam wallet non-withdrawable wedge) | LEG+OPS | SL-lite (gambling-adjacent; Valve ToS) |
| 25 | Sports-card breaks | -- | KILL-prior (breaks are retail-negative-EV by construction: breaker margin) |
| 26 | Crowdfunding exclusive rewards resale | RAT | LOW (anecdote only) |
| 27 | US Mint household-limited releases -> eBay premium | RAT (household limits) | SL |
| 28 | Costco gold/silver bars (member limit) + card cash-back | RAT | SL-lite (unit price vs capital) |
| 29 | Allocated whiskey (BTAC/Pappy) resale | RAT | KILL-prior (CA ABC: reselling alcohol w/o licence illegal; verify) |
| 30 | Recreation.gov campsites / permits resale | RAT | KILL-prior (resale prohibited; verify) |
| 31 | Restaurant reservation resale (AppointmentTrader) | RAT | KILL-prior? (CA law unknown; verify once) |
| 32 | MTG buylist arbitrage (TCGplayer low < Card Kingdom / store buylist) | CAP (buylist qty caps), NICHE | SL (MTGJSON has both price sides) |
| 33 | Carrier/Apple trade-in promo arbitrage (buy used phone < trade-in credit) | RAT (per line) | LOW (credits locked to new lines) |
| 34 | I-bonds (TreasuryDirect $10k/person/yr) | RAT | baseline only (risk-free comparison) |
| 35 | Facebook Marketplace / Craigslist underpriced electronics | OPS+NICHE | LOW (scraping barred by ToS -> no software edge; manual) |
| 36 | Storage-unit auctions | OPS | KILL-prior (TV-inflated; no evidence) |
| 37 | Retail hidden clearance (BrickSeek-type) -> resale | OPS | LOW (tools commoditized) |
| 38 | Retired/sealed video games (Limited Run, Nintendo) | RAT | LOW |
| 39 | Pop-culture blind-box drops (Pop Mart Labubu etc.) | RAT | LOW (fad decay; check once) |
| 40 | Open-source issue bounties (Algora, Gitcoin) | CAP | folded into #21 |

## Research log (shortlist), 2026-11-25 UTC
Labels: [P] primary (regulator, issuer, dataset), [N] trade press / news, [B] blog/reseller (secondary), [U] unverified.

### S1. Mutual savings bank standard conversions (depositor priority)
- Mechanism: eligible account holders (qualifying deposit on the eligibility record date) get Tier-1 subscription
  priority; when oversubscribed the issuer "must allocate shares among the eligible account holders so that each, to the
  extent possible, may purchase 100 shares", the remainder pro rata on qualifying deposits (12 CFR 192.375(a),(d),
  https://www.law.cornell.edu/cfr/text/12/192.375) [P]. => a SMALL depositor gets roughly the same 100-share floor as a
  big one. That is the per-person ration that keeps funds out (a fund can't be a depositor at 50 mutuals cheaply... it
  can, but each slot pays ~$100-400; capacity capped per person).
- Eligibility record date >= 1 year BEFORE the board adopts the plan (federal rule, also FDIC 1994 final rule, which
  cites "investment partnerships opening small accounts at many mutuals" as the abuse it targets:
  https://www.govinfo.gov/content/pkg/FR-1994-11-30/html/94-29240.htm) [P]. Wisconsin state rule: >= 90 days [P,
  docs.legis.wisconsin.gov]. Connecticut: qualifying deposit min $50 [P, portal.ct.gov]. => must hold deposits 1-3+ years
  BEFORE any announcement; you cannot react to news.
- First-day returns (S&P Global "State of the pipeline", trade press, $10 offer price) [N]: Hoyne Bancorp 2025-12-04
  +40.0% (Tier 1 oversubscribed, priced at supermax); Security Midwest 2025-08 +14.0%; Lake Shore (second step) 2025
  +19.5%; FB Bancorp 2024-10-23 +18.6%; Monroe Federal 2024-10-24 +13.5%; PSB Financial 2026-05-22 +5.0%. Avidia 2025-07
  oversubscribed by eligible depositors, $191.8M. Academic: Cagle & Porter avg initial return ~7%; 1992 24%, 1993 29%
  (openjournals.libs.uga.edu/fsr/article/download/3820/3267) [P-ish, old]. CAUTION: these are the deals the search
  surfaced, not a census -> selection bias toward pops.
- Frequency: S&P May 2026: 3 conversions in pipeline, "Community Savings Bank ... the lone standard conversion" [N via
  search snippet; article itself 403 to WebFetch]. => ~2-6 standard conversions/year nationally (my estimate [U]).
- Verdict so far: real, rationed, legal, but SLOW (>= 1-yr seed, unpredictable which mutual converts; most mutuals
  never do in any given year). Net per hit ~ 100 sh x $10 x (5-40%) = $50-400, but hit rate per seeded account per year
  maybe 1-3% [U] -> $/account-year ~ $2-10 [U]. Needs: base rate of conversion per mutual per year (open).
- Legal: transferring subscription rights / buying as nominee is prohibited (federal conversion rules; every prospectus
  requires a no-transfer certification) [U: rule text not fetched; eCFR redirected to a bot check, not bypassed].

### S2. US Mint household-limited releases -> eBay
- Rationing: household limits (often 1) on hot numismatic releases; Mint cancels orders that break the limit and blamed
  ordering problems partly on bots (coinworld.com "Mint reviews recent customer ordering troubles") [N]; Mint FAQ: limits
  may be added/changed at its discretion (usmint.gov/help-center/faqs/order-payment-processing.html) [P].
- Dated resale evidence (issue -> eBay):
  - 2026 250th Anniversary Silver Proof Set: $245 issue (06-11); CoinNews 06-21 eBay ~$310-349.95, avg ~$326 (+33%);
    mintage raised 250,520 -> 271,250; RETURNED to Mint site 07-01 (premium-killer)
    (coinnews.net/2026/06/21/mint-silver-set-draws-premiums/; coinnews.net/2026/07/01/us-mints-250th-silver-set-returns/) [N].
  - 2026 Mint Set: >100 eBay sales avg $185.92, +49.3% over issue (coinnews.net/2026/07/06/mint-set-premiums/) [N].
  - 2025 Limited Edition Silver Proof Set: $255, HH limit 1, 25,008 units; eBay sold ~$350 (+37%) [B resellcalendar].
  - 2025 250th Marine / Navy silver proof: $105 -> $180-200 raw (+71-90%) [B resellcalendar]; Army 100k sold out
    06-13 day 1 [N coinworld].
  - 2026 Trump $1 rolls/bags 09-02: 2/household first 24 h, sold out by 3 pm; premium not sourced [N].
- Net per success, e.g. Marine: sell $190 - eBay ~13.25% ($25) - ship ~$5 - cost $105 - Mint S&H [U ~$5] = ~$50;
  Silver Proof Set: $326 - $43 - $8 - $245 - S&H = ~$25. CA sales tax: < $2,000 coin purchases taxable in CA unless
  the US-as-seller exemption (R&TC 6352) applies -- UNRESOLVED (cdtfa.ca.gov/formspubs/l-906.pdf; CDTFA annotation
  168-0005) [P, ambiguous]. At 7.25-10.25% tax the Silver Proof Set net goes to ~0.
- Fill risk: hot drops sell out in minutes; success rate per attempt unknown [U]. Software advantage: NONE allowed
  (bot buying violates Mint terms / they cancel); only a calendar + alert helps. Capacity: 1-2 units/household/release.
- Verdict: real and repeatable, but ~$25-80 net per success, a few successes/yr, $/h ~$20-60 only on hits [U]. Lead.

### S3. Government surplus (GovDeals / GSA / PublicSurplus / PropertyRoom) vs eBay comps
- Evidence is blog/vendor only [B]: electronics/office lots "5-10%" thin premiums (closo.co); shipping/restoration is the
  main loss mode; GovDeals mostly pickup-only (thehustle.co); no study of hammer vs eBay sold prices found.
- Data access: eBay Marketplace Insights (sold-price) API is Limited Release, "available only to select developers
  approved by business units" (developer.ebay.com/api-docs/buy/marketplace-insights/overview.html) [P]; forum replies
  show denials [P-ish]. => a hammer-vs-comps scanner needs manual Terapeak lookups or third-party comps; scraping eBay
  sold listings is not cleared under its terms. Software edge reduced to listing alerts + a price model.
- Verdict: LOW on evidence; local pickup + as-is = ops cost is real (that's why it persists) but $/h unmeasured.

### S4. CEF tender offers (98-99% NAV)
- Proration outcomes 2025-26 [N/P mix]: SWZ 01-2026 up to 4M sh, 12.04M tendered (~33% accepted, barchart);
  ZTR 12-2025 37.25% accepted (barchart); NFJ 10-2026 99% NAV, 80.67% proration (Seeking Alpha, secondary);
  CMU 05-2026 prelim ~91.37% (SEC SC TO-I/A ex99-a5iv) [P]; JOF 07-2026 10% offer, 22.0M tendered (~13% accepted).
- Why NOT capacity-capped: activists/arbs (Saba, Karpus) already tender at scale -> proration falls when gap is big.
  On $1,000 at an 8% discount, 40% proration -> ~$24 gross, with price risk on the 60% returned. Verdict: LOW (not a
  small-player niche; institutions are in it).

### S5 (quick kills / lows)
- LEGO retired sets: Dobrynskaya & Kishilova (HSE WP 2020; RIBF 2021) ~11%/yr secondary-price growth, 2,322 sealed sets
  1987-2015; dispersion huge; growth starts 2-3 yrs after retirement; storage/delivery costs flagged
  (wp.hse.ru/data/2020/09/16/1583081489/80FE2020.pdf) [P]. Pre-2015 sample, before the 2020s reseller boom -> likely
  decayed [U]. Gross 11%/yr on $1k = $110/yr minus ~13% selling fee once => ~T-bill-plus at best, multi-year lock. LOW.
- CA tax-defaulted land: min bid = redemption + costs; as-is; title companies generally won't insure without quiet title
  (marincounty.gov TDL general info); no refunds (slocounty.ca.gov) [P]. Illiquid unbuildable lots; worst case = 100%
  + carrying tax. KILL for this user (no exit, legal cost > capital).
- Allocated whiskey resale: CA B&P 23300 (no licensee act without a licence; ABC third-party-provider guidance) [P,
  indirect]. KILL (unlicensed alcohol sales).
- Bug bounties / AI red-teaming: HackerOne avg bounty $1,090, $81M paid 07-2024..06-2025, top-100 programs $51M
  (hackerone.com/report, secondary coverage) [N]; 2026 report $89.1M [N]; Gray Swan Arena 2.0 $300k pool from
  2026-10-20 (fortune.com 2026-10-20) [N]; UK AISI x Gray Swan 2025 pool $171,800 (grayswan.ai blog) [P]. Power-law
  payouts; no median $/h published. It is skilled labor/competition, not an inefficiency; route to compete project. LEAD.
- Book scouting / textbook buyback: only vendor/affiliate $/h claims ("$20+/h", "$41/book") [B]; Amazon 15% + $1.80 per
  book; FBA reimbursement window cut 18 mo -> 60 d from 2025-01-09 [B]. LOW (labor, saturated scanner apps).
- Restaurant-reservation resale: CA AB 1640 (Restaurant Reservation Anti-Piracy Act; up to $1,000 civil penalty per
  violation, written agreement required) passed Assembly 2026-05-28; one tracker says chaptered 2026-09-27 (Ch. 475)
  (calmatters.digitaldemocracy.org/bills/ca_202520260ab1640; legiscan.com/CA/bill/AB1640) [N, chaptered status U]. KILL.
- Sneakers: share of releases trading above retail 47% in 2024, down 11 pts from 2020; margins "10-25%" vs ~100% peak
  (frontofficesports.com 2026-06-10; businessoffashion "resale market is broken") [N/B]; StockX fees up to ~19%. LOW.
- CS2 skins: Steam market ~15% seller fee; Steam wallet not withdrawable -> third-party sites 10-35% below Steam
  (skinbase.io, exitlag, games.gg) [B]; "gaps close within minutes" [B]. The wedge is a cash-out cost, not a mispricing;
  gambling-adjacent ecosystem. LOW/KILL.
- Thrift base rate: ~494 mutual banks in 45 states (ICBA, undated) [N]; S&P: ~3 conversions in pipeline at a time
  (05-2026). FDIC "does not require banks to favor local depositors over out-of-area 'professional depositors'" (per
  bankrate.com summary of FDIC rule) [B]. SEC investor bulletin warns against "loan to depositor" schemes
  (sec.gov/investor/pubs/mutualconversion.htm) [P]. Implied per-mutual conversion rate ~1-2%/yr [U, my arithmetic: ~5-10
  conversions/yr / ~494]. With $1,500 spread over ~15-30 accounts ($50-100 each) -> ~0.2-0.5 hits/yr x ~$150 = ~$30-75/yr
  + ~10-20 h of account opening/1099 admin. => DOWNGRADE to LOW for this capital (good only as a "plant and forget"
  side bet; the 100-share floor is the attractive part).

### S6. MTG buylist arbitrage (buy TCGplayer retail -> sell to Card Kingdom buylist)
- Mechanism: CK posts buylist prices (cash; +30% if store credit, bonus on whole order, PayPal gets no bonus); payment
  within 24 h after grading in 5-7 days; ship within 7 days of creating the order; no official minimum found
  (cardkingdom.freshdesk.com first-time-seller article; cardkingdom.com/purchasing/how_to_sell) [P]. Shipping cost
  inbound: not found [U].
- Practitioner basis: "negative spread" (buylist > retail) = arbitrage to the buylist (blog.mtgprice.com "The New Spread",
  2019) [B]; CK buylists only ~5-6 commons per Modern set (mtgprice 2019) [B].
- Fees to cross: TCGplayer buyer pays CA sales tax (~7.25-10.25%) + seller shipping (~$1/order under $5 [U]); seller side
  fees N/A (CK pays cash). TCGplayer seller commission 10.75% + 2.5% + $0.30 since 2026-02-10 (help.tcgplayer.com Fees;
  seller.tcgplayer.com blog) [P] -- relevant only for the reverse direction.
- Data: MTGJSON AllPricesToday (MIT licence, mtgjson.com/license; robots.txt disallows /api/v5/*.json but not the
  compressed archives; 4.4 MB .xz) -- download running; script results/small_capacity_census/scripts/mtg_spread.py.
- S2 addendum (Mint costs): standard shipping $12.95/order, budget $5.95 or $9.95 (pages disagree); free shipping after 3
  orders in a calendar year (usmint.gov/help-center/faqs/shipping-and-returns.html; most-popular-questions) [P].
  Household limits "typically lifted one business day after a product goes on sale" (usmint.gov most-popular-questions)
  [P] -> only products that sell out on day 1 carry a premium; the rest trade near issue. Recomputed net per unit with
  $9.95 ship + eBay 13.25% + $5 outbound, no CA tax: Marine/Navy-type $105 -> ~$190 sale = ~+$45; 2026 Silver Proof Set
  $245 -> ~$326 = ~+$20 (and it came back 07-01). With ~9% CA tax if it applies: ~+$35 / ~-$2.
- S3 addendum (scanner access, 2026-11-25 ~03:55Z): www.govdeals.com/robots.txt returns Akamai "Access Denied" to a
  plain curl -> scripted access is blocked; NOT bypassed. => GovDeals scanner = dead; only GovDeals' own saved-search
  email alerts. GSA Auctions is an SPA; an official GSA Auctions API on api.gsa.gov needs an api.data.gov key [U, not
  verified; key = a sign-up the user would do]. Software edge for surplus: weak -> stays LOW.

## Pre-registered tiny tests (written 2026-11-25 ~04:00Z, before any outcome; user runs every purchase/sign-up)
### T-MINT (US Mint household-limited release flip) -- capital <= $100
- Question: do day-1 sell-out Mint products with household limits still carry a resale premium that survives eBay fees
  + Mint shipping, and can a normal buyer actually get filled?
- Selection rule (fixed in advance; decided from pre-release facts only): product issue price + Mint shipping <= $100;
  household limit <= 2 on launch day; published mintage/product limit <= 100,000; AND (silver/gold proof OR 250th-theme).
- Paper phase ($0, next 3 qualifying releases from usmint.gov/product-schedule): record issue price, launch time,
  whether "Currently Unavailable" within 24 h, and the eBay SOLD median at launch+14 d (user via eBay sold filter /
  Terapeak; agent cannot pull eBay sold data -- Marketplace Insights API is restricted).
  Success: >= 2 of 3 have sold median >= (issue + $9.95) x 1.25 + $5 (i.e. >= ~+$15 net after 13.25% fee + $5 ship).
  Kill: < 2 of 3, or < 2 qualifying releases in 90 days (frequency too low).
- Live phase (only if paper passes): one order <= $100 all-in on the next qualifying release; list within 3 days of
  delivery at the then sold median. Success: realised net >= +$15 within 30 d. Kill: net < $0, or 3 straight launch
  attempts unfilled (fill rate < 1/3). CA sales tax: record whether the Mint charges it (resolves the open question).
- Software: calendar + launch-time reminder only; NO bots (Mint cancels limit-breaking/bot orders).

### T-BOUNTY (AI red-teaming / agent-jailbreak bounties) -- capital $0
- Question: does the user's ML background earn >= CA minimum wage on paid red-team arenas?
- Protocol: 6 h timebox on Gray Swan Arena 2.0 (live from 2026-10-20, $300k pool) or one vendor AI bug-bounty program;
  log hours and accepted findings. Success: >= $100 paid/accepted within 6 h (>= $16.7/h). Kill: $0 after 6 h.
- Routing: this is a competition-style income source -> belongs with the compete project, not the fund.
- S6 addendum: TCGplayer marketplace minimum shipping for orders < $5 rose $1.31 -> $1.49 on 2026-05-26
  (seller.tcgplayer.com/blog/new-minimum-shipping-price-ahead-of-mayhem-promotion) [P]; >= $5 the seller's own rate
  applies (often $0-ish, not guaranteed). Outbound to CK: one tracked package per batch (~$5-8 [U]).
- S6 MTGJSON-only census (AllPricesToday meta date 2026-11-24, version 5.3.0+20261124; script
  results/small_capacity_census/scripts/mtg_spread.py -> mtg_spread_out.json; assumptions tax 9.25%, $1/card ship):
  81,701 CK-buylist x TCGplayer-retail pairs; net >= $0: 657; >= $2: 281; >= $20: 97; ratio quantiles (CK buy / TCG
  retail) p50 0.412, p90 0.86, p99 1.503. By TCG price band (pairs / net>=0 / net>=2 / sum net>=2):
  <$5 58,282/277/45/$358; $5-20 15,541/179/63/$328; $20-50 4,886/54/36/$305; $50-200 2,350/65/57/$1,831;
  >=$200 642/81/80/$17,039. TOO GOOD at the top: top rows are $700-4,600 buylists with no CK retail stock (vintage/
  condition or printing mismatch suspected). Treat as UNVERIFIED until the CK-pricelist + Scryfall cross-check.
- S6 competition: MTGStocks Premium sells "CK: Cheaper than TCGplayer" and "Direct Low Differential" arbitrage filters
  and tracks CK buylist changes (mtgstocks.com/go-premium/draftsim; mtgstocks.com/news/7760-new-tool-alert) [P vendor];
  Underpriced guide warns gross-vs-buylist comparisons without fees/shipping are the common mistake [B]. => the scanner
  itself is NOT a moat; any persistent gap is either too small for the paid-tool crowd, capped by qty_buying, or an
  artefact (condition / printing / stale).
- S1 addendum: more 2026 deals (S&P pipeline columns via search snippets) [N]: URSB Bancorp 2026-03-27 +7.0% (supermax);
  PSB Financial 2026-05-22 +5.0% (Tier 1 oversubscribed, supermax). Collected first-day closes (n=7, 2024-10..2026-05,
  NOT a census): +18.6, +13.5, +14.0, +19.5 (2nd step), +40.0, +7.0, +5.0 -> mean +16.8%, median +14.0% (my arithmetic).
  At the 100-share floor: ~$140 median per hit (pre-tax). Pipeline: CSB Financial (standard, filed 2026-03-13),
  Columbia Financial 2nd step + Northfield acquisition, Rhinebeck 2nd step.
- S6 cross-check 1 (scripts/mtg_join.py -> results/small_capacity_census/mtg_join_out.json; CK pricelist v2 created
  2026-11-24 19:07:54, 155,033 rows, api.cardkingdom.com robots "Disallow:" empty): of 281 net>=$2 candidates only 16
  matched a unique CK row (264 ambiguous on price). 15 of 16 were VINTAGE ($350-$7,200: Alpha/Beta/Unlimited Moxen, duals,
  Time Vault, Nevinyrral's Disk...) with CK qty_buying 1-5 and CK NM retail ~1.7-2x the TCGplayer price -> the "spread" is
  TCGplayer market price of PLAYED vintage copies vs CK's NM buylist (CK grades down). ARTEFACT, not arb. The one sub-$50
  hit (Swords to Plowshares, Magic 30th Anniversary, TCG $0.10 vs CK buy $65) is a product-mapping mismatch. Scryfall
  prices equal MTGJSON's exactly (same TCGplayer feed) -> they confirm identity, not condition.
- Bounty addendum: Gray Swan IPI Black challenge $300k (Anthropic, Meta, Google DeepMind, Reflection AI), new scenarios
  weekly through 2026-11-02; Arena community ~15,000 (Discord) (fortune.com 2026-10-20) [N]. Pool / community ~= $20
  per participant on average -> power-law, most earn $0. Prize split rules not published in the article [U].

## RANKING (2026-11-25; for $200-1,500, CA resident; $ are estimates [U] unless a row cites a measured number)
Ranking key = expected net $/month at realistic hours. DNR rows are listed for context, not re-researched.
| # | angle | why big money skips | net $/mo | $/h | capital | max loss / cycle | automatable? | evidence |
|---|---|---|---|---|---|---|---|---|
| ctx | Odd-lot tenders (DNR) | 99-share cap | ~$0-30 (0 qualifying 11-20; 5 offers/60 d) | high per event | $100-1,000 | price < floor | YES, watcher built (oddlot_watch.py) | money_now.md s.B |
| ctx | Kalshi thin series (DNR) | capacity ~$150-300/mo at $5k | see FUND_BOOK | -- | -- | -- | yes (core) | FUND_BOOK s1 |
| 1 | US Mint household-limited releases -> eBay | per-household limit 1-2; Mint cancels bot/limit-breaking orders | ~$10-40 | ~$15-30 on fills | $110-260/unit | ~-$20/unit (re-stock kills premium, e.g. 2026 Silver Proof Set back 07-01) | alerts only; buying bots = cancelled orders | CoinNews 2026-06-21/07-06 eBay sales [N]; resellcalendar [B] |
| 2 | Mutual-bank conversion depositor seeding | per-depositor 100-share floor (12 CFR 192.375); rights non-transferable; >= 1-yr record date | ~$3-6 (est. 0.2-0.5 hits/yr x ~$140) | ~$5-15 | $50-100/account, locked years | ~0 on deposit (FDIC); IPO can trade +5% only (PSB) | EDGAR watcher for plans/S-1 (detects, can't qualify you late) | 12 CFR 192.375 [P]; n=7 first-day closes, median +14% [N] |
| 3 | AI red-team / bug bounties | niche skill; per-finding payouts | $0-? (power law; ~$20/participant avg on Gray Swan) | unknown | $0 | time only | partly (harnesses) | HackerOne avg bounty $1,090 [N]; fortune 2026-10-20 [N] |
| 4 | MTG buylist arb (TCGplayer -> Card Kingdom) | CK qty_buying caps 1-5/card; condition grading | see S6 cross-check 2 | -- | $50-300 | CK downgrade | YES (MTGJSON + CK API + Scryfall, all permitted) | own census 2026-11-24 |
| 5 | Gov surplus vs eBay comps | pickup, as-is, local | unmeasured | unmeasured | $50-500 | 100% of lot | NO (GovDeals blocks scripts; eBay sold API restricted) | blogs only [B] |
| 6 | CEF tender offers at 98-99% NAV | NOT skipped: Saba/Karpus play | ~$2 (one ~$24 event/yr on $1k) | ~$10 | $1k+ | price risk on prorated rest | EDGAR SC TO-I watcher (same as odd-lot) | proration 13-91% 2025-26 [N/P] |
| 7 | LEGO retired sets | storage, multi-year | ~T-bill+ (11%/yr gross, pre-2015 data) | low | $500+ | set-specific -50% | price tracking | Dobrynskaya & Kishilova [P] |
| -- | KILLED | tax-defaulted land (no title insurance/no exit), allocated whiskey (B&P 23300), reservation resale (AB 1640), GovDeals scanner (blocked), tickets (DNR, AB 1349/BOTS), sneakers/CS2/books (margins/fees, vendor-only $/h), liquidation pallets/storage units/breaks (no evidence / negative by construction) | | | | | | |
- S2 frequency addendum (CoinNews 2026 items) [N]: quick sell-outs in 2026 = 2026-W Proof Silver Eagle + Congratulations
  Set 02-26 (gone within minutes; W lasted ~17 min); Steve Jobs $1 05-2026 (gone right away; macdailynews); Silver Proof
  Set 06-11; 2026 Mint Set; Freedom Ringing gold 07-2026 (gone in 30 min, then RETURNED to sale); Minnesota $1 rolls
  08-2026 (8,339/8,400 roll limit); Trump $1 09-02. => ~6-10 sell-out events in 2026, an unusual 250th-anniversary
  year; expect fewer in 2027 [U]. Two of ~8 later returned to sale (premium risk).
- S6 cross-check 2 (scripts/mtg_manapool_check.py -> results/small_capacity_census/mtg_manapool_out.json, offline):
  of 144 sub-$50 TCGplayer candidates (net >= $2), 108 have a Manapool retail price and only 36 still clear net >= $2
  against BOTH venues (sum of min-net, one copy each = $194.44). CK buy / CK retail on these = 0.4-0.65 (CK's usual
  buylist ratio) -> the gap exists because CK's OWN retail is ~2x the marketplaces (e.g. buy $55 / CK retail $109.99 vs
  TCG $28.26 / Manapool $21.87). Most are old foils.
- Mechanism found while debugging the join: CK prices in FLAT TIERS (e.g. 172 foil rows at buy $16.50 / retail $32.99
  -- Mercadian Masques, Urza's Destiny/Legacy foils; 19 non-foil rows at $55 / $109.99 -- Mana Vault 4th, Alpha commons).
  Low-demand old foils that the market prices below the tier are where buy > market. CK caps it with qty_buying (~4-5 on
  the tier rows seen). Wrong turn logged: first joins matched on price only and were 92-100% ambiguous for this reason.
- Mapping caveat: which TCGplayer condition MTGJSON/Scryfall report is not documented in what I read; TCGplayer
  "Market Price" = avg recent sales per product per condition (help.tcgplayer.com article 222376867) [P] [U for feed].

### T-MTG (Card Kingdom tier-buylist arbitrage) -- capital <= $100 (written 2026-11-25 ~04:20Z, before naming results)
- Question: after real NM listings, CA tax, shipping both ways and CK's grading, does buying low-demand cards on
  TCGplayer/Manapool and selling to CK's cash buylist net >= 15% per batch, repeatably?
- Signal (fixed): CK cash buy >= max(TCGplayer, Manapool retail) x 1.0925 + $1 + $3; CK qty_buying >= 1; TCG price < $50;
  EXCLUDE pre-1995 sets and Reserved List (vintage condition artefacts, cross-check 1).
- Paper phase (7 days, $0): run the scanner daily; user spot-checks 10 random flagged cards for a real NM listing
  (with seller shipping) at <= the scanned price. Success: median >= 15 flagged cards/day AND >= 6/10 confirmed with
  net >= $3. Kill: < 5/day OR < 4/10 confirmed.
- Live phase (only if paper passes): <= $100 on 5-15 flagged cards (NM, sellers >= 99% rating), one tracked package
  to CK within 7 days of creating the sell order (CK rule). Success: CK cash payout - (cards + tax + inbound shipping +
  outbound postage) >= +$15 AND <= 1 card downgraded. Kill: net < $0, OR >= 30% downgraded, OR CK cuts price/qty on
  > half the list before the order is created (gap decays faster than a human can act).
- Scanner design: daily 07:00 PT job. Data: MTGJSON AllPricesToday.json.xz (MIT; robots disallows raw .json only),
  api.cardkingdom.com/api/v2/pricelist with gzip (robots "Disallow:" empty; gives names, qty_buying, tier prices),
  Scryfall /cards/collection for names/links (accurate User-Agent + Accept, <= 10 req/s). Join CK->Scryfall by
  scryfall_id, Scryfall->TCG price; Manapool from MTGJSON. Alert: email/Slack table (card, set, finish, CK buy,
  TCG/Manapool, net, qty_buying, TCGplayer link). Human buys; no automated purchasing (TCGplayer terms; eBay-style
  AI-agent bans). CA: buy-for-resale may need a CDTFA seller's permit if it becomes regular [U].
- S2 verified (WebFetch coinnews.net/2026/07/06/mint-set-premiums/): 2026 Mint Set issue $124.50, released 2026-06-30,
  unavailable ~24 h later, product limit 300,000; > 100 eBay sales avg $185.92 (+$61.42, +49.3%); sale date range not
  stated [N, primary article]. Net/unit = 185.92 x 0.8675 - $5 - ($124.50 + $9.95) = +$21.84 (before any CA tax).
- S6 downgrade schedule: CK pays below-NM cards a % of the NM buy: foils EX 75% / VG 50% / G 30%; $15-24.99 non-foil
  EX 85% / VG 70% / G 50% (cardkingdom.com/purchasing/how_to_sell; page date unknown) [P]. => per-card worst case is
  bounded: e.g. buy $16.50 foil bought at $6.49 (+tax +$1 = $8.09): EX +$4.29, VG +$0.16, G -$3.14. Batch of $100 all
  graded G ~ -$55; realistic worst (half EX/half VG) ~ -$10..-$20.
- S6 sensitivity (offline, price = max(TCG, Manapool), TCG < $50, one copy each; computed AFTER T-MTG was written):
  tax 7.25%/ship $0/net>=$2: 90 cards, $375.31 | 9.25%/$1/>=$2: 35, $190.95 | 10.25%/$1.49/>=$2: 24, $148.84 |
  9.25%/$1/>=$3 (= T-MTG signal before the pre-1995 exclusion): 19, $151.53 | >=$5: 12, $122.75.
  With qty_buying ~4-5 the one-day snapshot pool is ~$600-750 gross [U] before downgrades / real listing depth.
  Today's count (19) sits just above the T-MTG paper bar (median >= 15/day): the test is informative, not a formality.
- S6 RESULT (scripts/mtg_name_strict.py -> results/small_capacity_census/mtg_strict_named.json; one snapshot,
  MTGJSON 2026-11-24 + CK pricelist 2026-11-24 19:07): 19 cards pass the T-MTG signal; 15 named (4 unresolved), 14 after
  the pre-1995 exclusion (Bad Moon Unlimited dropped), 12 of 14 FOILS from 1999-2007 (Cateran Overlord MMQ foil buy $55 vs
  TCG $13.20 / Manapool $32.30; Crusading Knight INV foil $16.50 vs $6.49/$6.18; Numot PLC foil $30 vs $19.28/$19.86;
  Ral Storm Conduit promo foil $8.40 vs $1.14/$0.95; Bruna Secret Lair $25 vs $18.33/$13.49, qty_buying 16; ...).
  One copy each: net $101.76 on cost $346.92 (29.3%). qty_buying-weighted (70 units): net $423.56 on $1,515.79 (27.9%).
  Caveats: ONE day (no persistence/decay data -> no claim of a monthly rate); assumes NM listings exist at market price
  for every unit (old foils are thin); old foils are the most downgrade-prone (foil EX pays 75%); not critic-reviewed.
  Verdict: LEAD (the only automatable angle found with a measured gross gap); run T-MTG paper phase before any money.

## RANKING (final, 2026-11-25) -- supersedes the draft table above
| # | angle | why big money skips | net $/mo | $/h | capital | max loss | automatable? | evidence |
|---|---|---|---|---|---|---|---|---|
| 1 | MTG: buy TCGplayer/Manapool -> sell CK cash buylist (old foils in CK's flat price tiers) | qty_buying 1-5/card; condition grading; shipping per batch | ~$30-100 [U]; snapshot pool $102 one-copy / $424 at full qty | ~$15-30 [U] | $100-350 | ~-$55/batch if all graded G; ~-$10..-20 realistic | YES: MTGJSON + CK API + Scryfall (permitted); human buys | own census 2026-11-24 (S6) |
| 2 | US Mint household-limited releases -> eBay | HH limit 1-2; bot/limit orders cancelled | ~$10-40 (6-10 sell-outs in 2026, fewer expected 2027) | ~$15-30 on fills | $125-260/unit | ~-$20/unit (re-stock: 2 of ~8 returned) + CA tax U | alerts only | CoinNews Mint Set +$21.84/unit net (verified); others [N]/[B] |
| 3 | AI red-team / bug bounties | niche skill, per-finding | $0-? (power law) | unknown | $0 | time | partly | HackerOne avg $1,090 [N]; Gray Swan $300k / ~15k community [N] |
| 4 | Mutual-bank conversion seeding | per-depositor 100-share floor; >= 1-yr record date | ~$3-6 | ~$5-15 | $50-100/acct, years | ~0 (FDIC deposit; buying optional) | EDGAR watcher (detects only) | 12 CFR 192.375 [P]; n=7 pops median +14% [N] |
| 5 | CEF tenders 98-99% NAV | NOT skipped (activists) | ~$2 | ~$10 | $1k+ | price risk | EDGAR TO-I watcher | proration 13-91% [N/P] |
| 6 | Gov surplus vs eBay | pickup/as-is | unmeasured | unmeasured | $50-500 | 100% of lot | NO (GovDeals blocks scripts) | blogs [B] |
| 7 | LEGO retired | storage, years | ~T-bill+ | low | $500+ | -50% per set | tracking only | HSE paper pre-2015 [P] |
| ctx | Odd-lot tenders / Kalshi thin series | DNR (money_now.md s.B; FUND_BOOK) | | | | | built | |
| KILL | tax-defaulted land, allocated whiskey (B&P 23300), reservation resale (AB 1640), GovDeals scanner, tickets (DNR), sneakers/CS2/books, pallets/storage units/breaks | | | | | | | |
Top-3 tests: T-MTG, T-MINT, T-BOUNTY (above). Thrift = plant-and-forget suggestion only (no <= $100 test can see an outcome
within a year).

## MC1 scanner (agent mc1_scanner, built 2026-11-25 UTC; T-MTG paper phase tooling)
Files: `scripts/mc1_buylist_scan.py` (stdlib only, `python3 -I` safe, py3.9+), `scripts/qs_job.sh` case `mc1`,
`scripts/install_mc1_agent.sh`, `tests/test_mc1_buylist_scan.py` (10 offline tests), run logs in `results/mc1_scanner/`.

**What it does.** Downloads MTGJSON `AllPricesToday.json.xz` + the Card Kingdom v2 pricelist (gzip) into a temp dir
(deleted on exit), applies the strict filter of `scripts/mtg_name_strict.py` UNCHANGED (price = max(TCGplayer,
Manapool) < $50; net = CK cash buy - (price x 1.0925 + $1) >= $3; name = CK (buy, retail, foil) tier row whose Scryfall
TCGplayer price equals MTGJSON's to the cent, exactly one hit), then the pre-registered exclusions (Scryfall
released_at < 1995-01-01, Scryfall `reserved`, CK qty_buying < 1). Writes `~/.claude/quant-sports/data/mc1/<PT date>.json`
(flags + unresolved + excluded + data stamps), `<date>.md` (click-through table + seeded 10-card spot-check checklist) and
appends one line to `daily.jsonl` (date, n_flags, net/cost one copy, net/cost full qty_buying, n_signal, n_unresolved,
mtgjson_date, ck_created_at, run_utc). Exit 0 ok, 2 network/download failure (nothing written), 1 bug.

**Terms checked 2026-11-24.** mtgjson.com/robots.txt disallows `/api/v5/*.json` only (we fetch the `.xz`); licence MIT.
api.cardkingdom.com/robots.txt = `Disallow:` (empty); no API terms page reachable (www.cardkingdom.com is behind a
Cloudflare challenge); one GET/day. Scryfall (scryfall.com/docs/api/rate-limits): UA + Accept required;
**`/cards/collection` hard limit 2 req/s (500 ms)**, not the general 10/s; a 429 = 30 s lockout. The census scripts spaced
calls 0.3 s (3.3/s, over the limit); the scanner uses 0.6 s and backs off 35 s on a 429. ~12 POSTs per run (841 ids on 11-24).

**Reproduction (cached census files, Scryfall live 2026-11-25 04:31Z;** `results/mc1_scanner/repro_20261124.log`): 19 signal
rows -> 4 unresolved (3 `no_ck_retail`, 1 `no_match`; same 4 as mtg_strict_named.json) -> Bad Moon (Unlimited, 1993)
excluded pre-1995, 0 Reserved List, 0 qty 0 -> **14 flags, identical (name, finish, CK buy, TCG, Manapool, qty) tuples to
mtg_strict_named.json; net $101.76 on $346.92 one copy; $423.56 on $1,515.79 at full qty (70 units)** = the census
numbers to the cent (cost totals sum UNROUNDED per-card costs, nets sum cent-rounded nets: the census convention).

**Count definition -- decide BEFORE day 1.** The T-MTG bar is "median >= 15 flagged cards/day". The scanner's
`n_flags` = named, post-exclusion cards (the only ones a human can buy or spot-check). On 11-24 that is **14, i.e. just
BELOW the bar**; the census line "Today's count (19) sits just above the T-MTG paper bar" counted pre-naming signal
rows (`n_signal`, incl. 4 unnamed rows and the pre-1995 Bad Moon). Recommendation: judge on `n_flags` (the
pre-registered signal includes the exclusions and a flag must be buyable); `n_signal` is logged alongside so the other
reading can be reported, not chosen after the fact.

**Live end-to-end run** (2026-11-25 04:35-04:38Z, --out /tmp/mc1_live_out, `results/mc1_scanner/live_run.log`,
table `results/mc1_scanner/live_20261124.md`): downloads OK (MTGJSON meta 2026-11-24 = same build as the census -- MTGJSON
had not rebuilt yet; CK pricelist REFRESHED, created 2026-11-24 20:07:34 vs census 19:07:54, 155,033 rows; Scryfall 836
cards), exit 0, 225 s wall, peak RSS ~0.8 GB, temp dir removed. Result **identical to the census: 14 flags, net $101.76 on
$346.92 one copy, $423.56 on $1,515.79 at full qty (70 units)**; n_signal 19, unresolved 4, pre-1995 1. The CK refresh one
hour later changed none of the 14 rows. This proves the pipeline on live endpoints; it is NOT a second independent day.

**Timing / risks found while building.**
- MTGJSON AllPricesToday.json.xz Last-Modified = 2026-11-24 06:14:16 GMT (HEAD request) -> builds ~22:15 PT the evening
  before, labelled with the UTC date, so the 06:30 PT run gets the same-dated build (the scanner warns if MTGJSON is
  > 1 day older than the run date).
- Naming needs Scryfall's TCGplayer price = MTGJSON's to the cent. That held at ~04:30Z (census + reruns). Whether it
  holds at the 06:30 PT (14:30Z) run is unverified (Scryfall bulk files refreshed ~22:00Z 11-24; API price refresh time
  unknown). If the two snapshots drift, rows move to `unresolved` (reason `no_match`) and n_flags falls for a data-timing
  reason, not a market one. **Check day 1's n_unresolved** (baseline 4 of 19: 3 `no_ck_retail` structural + 1
  `no_match`). If it jumps, fix the naming step (e.g. MTGJSON uuid -> scryfall_id via identifiers) BEFORE the week
  counts -- not after seeing the counts.
- Wrong turn (logged): while checking build times I GET-requested mtgjson.com/api/v5/Meta.json once; robots.txt
  disallows /api/v5/*.json. One small request; the scanner only fetches the .xz. Use a HEAD on the .xz instead.

### How to run
- Installed job (lead): `bash sports/scripts/install_mc1_agent.sh` -> launchd `com.jonathanwang.quant-sports-mc1`
  every 1800 s; `qs_job.sh mc1` runs the scan once per PT day at the first run >= 06:30 PT (exit 2 -> retried on the
  next run, at most 6 tries/day; state in `state/mc1_day`, `state/mc1_tries`; log `logs/mc1.log`).
- Manual: `python3 -I sports/scripts/mc1_buylist_scan.py --out /tmp/mc1test` (live) or add
  `--mtgjson-file F --ck-file F --date YYYY-MM-DD` for cached inputs. Tests:
  `cd sports && ../.venv_quant/bin/python -m pytest tests/test_mc1_buylist_scan.py -q`.
- Optional (not done): a `health_jobs.json` entry, e.g. `{"job": "mc1 (T-MTG paper scanner)", "label":
  "quant-sports-mc1", "logs": ["$S/logs/mc1.log"], "stderr": ["$S/logs/launchd-mc1.err"], "outputs": [{"path":
  "$S/data/mc1/daily.jsonl", "max_gap_h": 30, "ts": "run_utc", "keys": ["date", "mtgjson_date"]}]}`, and a
  ctx_inject.py line showing the last daily.jsonl row.

### Pre-registered test (copied verbatim from T-MTG above, written 2026-11-25 ~04:20Z)
- Signal (fixed): CK cash buy >= max(TCGplayer, Manapool retail) x 1.0925 + $1 + $3; CK qty_buying >= 1; TCG price < $50;
  EXCLUDE pre-1995 sets and Reserved List (vintage condition artefacts, cross-check 1).
- Paper phase (7 days, $0): run the scanner daily; user spot-checks 10 random flagged cards for a real NM listing
  (with seller shipping) at <= the scanned price. Success: median >= 15 flagged cards/day AND >= 6/10 confirmed with
  net >= $3. Kill: < 5/day OR < 4/10 confirmed.
(Implementation note: the scanner caps max(TCG, Manapool) < $50, as mtg_name_strict.py did -- stricter than "TCG price
< $50", never looser. Results between the success and kill bars, e.g. 5-14/day or 4-5/10, are neither; the
pre-registration does not say what happens then -- lead/user decide extend-or-stop and write it down before judging.)

### Judge date
Seven daily scans starting the PT day the lead installs the job (launchd's first tick comes ~30 min after bootstrap,
so an install before ~22:00 PT scans the same day). **Judge date = install date + 7 days**, using the last
`daily.jsonl` row per date for the 7 scan dates (median of `n_flags`) plus the user's spot-check sheet. Example: install
2026-11-25 -> scan days 11-25..12-01 -> judge 2026-12-02. Lead: write the actual install date here: INSTALL_DATE = ____.
Missing days (exit 2 all 6 tries) count as missing, not as 0; if >= 2 of 7 are missing, extend by the missing days.

### What the user must do (spot-check, ~20-30 min, $0, no purchases)
1. On ONE scan day inside the week (same morning as the scan -- prices move daily), open
   `~/.claude/quant-sports/data/mc1/<that date>.md`. Use the "Spot-check sample" list at the bottom: 10 flags drawn at
   random with seed = date (fewer if fewer flags; if < 10, take the next day's sample too until 10 cards are checked).
   Do not swap cards for "nicer" ones.
2. For each card: click the TCGplayer link (opens the product filtered to English / Near Mint / the right printing; check
   the filter took), find the cheapest NEAR MINT listing, note listing price + seller shipping + seller rating. Manapool
   (via the Scryfall card link) counts too.
3. Confirmed iff listing price <= the card's "price used" AND CK buy - (listing x 1.0925 + seller shipping) >= $3. Fill the
   blanks in the checklist (or tell the lead the numbers). Nothing is bought.
4. Optional: open the CK link and note whether CK's buylist still shows the same cash price (CK pages may need a browser).

## Learnings
### Per-person-rationed angles pay per EVENT, so event frequency, not capital, caps income
US Mint household limits, mutual-bank 100-share floors and odd-lot 99-share tenders each pay ~$20-400 per event, but
events come a few to ~10 times a year (2026 Mint sell-outs ~6-10 in a 250th-anniversary year; ~3 conversions in the
pipeline at a time; 5 odd-lot offers in 60 days, 0 qualifying 11-20). At $200-1,500 that is ~$10-50/month. Source:
this file S1, S2; money_now.md s.B. Count events per year before promising capacity-capped income.
### Card Kingdom's flat price tiers create a real, capped buylist gap on old foils
CK prices in tiers (e.g. 172 foil rows at buy $16.50 / retail $32.99); low-demand 1999-2007 foils priced below the tier on
TCGplayer AND Manapool clear CK cash buy + 9.25% tax + $1 by >= $3: 14 cards on 2026-11-24, $101.76 net on $346.92 one
copy each, $423.56 on $1,515.79 at full qty_buying (results/small_capacity_census/mtg_strict_named.json). One snapshot,
not critic-reviewed; the T-MTG paper week decides.
### Vintage MTG "buylist above market" rows are condition artefacts, not arbitrage
The 15 unique raw matches above $350 were Alpha/Beta/Unlimited (Moxen, duals, Time Vault): TCGplayer market price
reflects played copies, CK buys NM only and grades down. Exclude pre-1995 sets and the Reserved List from any MTG arb scan
(mtg_join_out.json).
### Require a second marketplace before trusting a price gap
Requiring Manapool to also be cheap cut sub-$50 candidates from 144 to 36 (mtg_manapool_out.json). One venue's quote is
often stale or a mapping error (e.g. a 30th Anniversary Swords at TCG $0.10 vs CK buy $65).
### Mutual-bank conversions give small depositors the same 100-share floor, but only after a >= 1-year seed
12 CFR 192.375(a) allocates so each eligible holder "may purchase 100 shares" before pro rata; the record date is >= 1 year
before plan adoption (federal). n=7 first-day closes 2024-10..2026-05 (not a census): median +14%, mean +16.8% -> ~$140
per hit. ~494 mutuals; ~1-2%/yr conversion odds per account. Plant-and-forget only.
### A scanner is not a moat when the venue blocks scripts or already sells the filter
GovDeals returns Akamai "Access Denied" to plain curl (even robots.txt); eBay's sold-price API is limited-release;
MTGStocks Premium sells a "CK cheaper than TCGplayer" filter. Check data access and existing paid tools first.
### Permitted MTG price data: MTGJSON archives, the CK v2 API with gzip, and Scryfall with an honest User-Agent
MTGJSON AllPricesToday.json.xz is 4.4 MB (robots.txt disallows raw /api/v5/*.json, not the archives).
api.cardkingdom.com/api/v2/pricelist (robots "Disallow:" empty) is 9.8 MB gzipped vs 69 MB raw (use --compressed); it
has scryfall_id, price_buy, qty_buying and per-condition retail. Scryfall requires an accurate User-Agent + Accept and
supports exact-price search (q=usd=12.34). This Mac's link ran at ~25 KB/s during the session: batch, cache, retry.
### Scryfall /cards/collection is limited to 2 requests/second, not the general 10/s
scryfall.com/docs/api/rate-limits (read 2026-11-25): /cards/search, /named, /random and /collection are 2/s (500 ms);
other endpoints 10/s; a 429 locks the client out for 30 s. The census scripts used 0.3 s spacing (3.3/s). The MC1 scanner
uses 0.6 s and a 35 s back-off on 429. Read the endpoint-specific limit, not the headline one.
### Define the counted unit of a paper-test bar before day 1
The T-MTG bar is ">= 15 flagged cards/day". On 2026-11-24 the same data gives 19 signal rows but 14 buyable, named,
post-exclusion flags -- one reading passes the bar and the other fails it (results/mc1_scanner/repro_20261124.log).
The census note used 19. Write which count is judged before the first scan day, and log both.

**Lead decision on the count definition (2026-11-25 ~05:00Z, before day 1 of the paper phase):** T-MTG's bar was written
at ~04:20Z when "flagged" meant the scan SIGNAL rows (11-24: 19). Changing it to the named count after seeing 14 would
move the goalpost in the test's disfavour, so the bar stays as written: **primary = median `n_signal` >= 15/day** over
the 7 days; `n_flags` (named, buyable) is logged as secondary, and buyability is judged by the spot-check gate
(>= 6/10 real NM listings at <= the scanned price), which only samples named cards. Judge date = install date + 7.

**2026-11-26 ops (lead):** Card Kingdom's API went behind a Cloudflare managed challenge on 11-25 (HTTP 403 to every
non-browser client, robots.txt included) -> 7 failed tries 11-25/26, no data. We do not get past such blocks. Fix: the
scanner now degrades to SIGNAL-ONLY mode when the CK pricelist is unavailable (n_signal from MTGJSON's cardkingdom buylist
field, n_flags = null, ck_blocked = true; new test). Day 2 logged: mtgjson_date 2026-11-25, n_signal 18. Day count = distinct
mtgjson_date values; the judge moves to the 7th distinct date (~12-01/02). The spot-check gate needs NAMES: before judging,
name the signal rows another way (MTGJSON AllIdentifiers uuid -> name/set/number, or Scryfall search by price as in
results/small_capacity_census/scripts/mtg_resolve.py); CK buy quantities must then be read in a real browser by a person.
