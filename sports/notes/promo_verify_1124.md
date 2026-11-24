# promo_verify_1124 -- re-verify CA prediction-venue new-customer promos (agent started 2026-11-24 06:45Z = 2026-11-23 22:45 PT)

Agent: promo_verify_1124 (research subagent of the quant-trader lead; protocol sports/notes/AGENT_PROTOCOL.md).
Read-only: no sign-ups, no logins, no credentials, no orders, no git. Scratch: sports/results/promo_verify_1124/.
Note: the brief says "TODAY (2026-11-24)"; the machine clock reads Mon 2026-11-23 22:41 PT (= 11-24 06:41 UTC). All
"read" dates below are 2026-11-23 PT / 2026-11-24 UTC.

## Question
For each venue (Gemini, ProphetX, Novig/Ludlow, Polymarket US, Webull, OG.com, FanDuel Predicts + any NEW live
prediction-venue promo): live today? CA eligible? exact qualifier, stake treatment, anti-hedge clauses, fees, source.
Then rank by expected net $ for THIS user (CA; already has Kalshi, Robinhood (holds RH event contracts -> RH promo
ineligible), Webull brokerage (RSA use), DraftKings Predictions (claimed 11-2x, SR confirmed: Tulsa $49.92 PD -> $52.00)).

## Pre-registered rules (written before re-reading any venue page)
- "Live" = an official source (venue domain, venue docs, or venue's own CFTC filing) dated/effective covering 2026-11-24,
  with no later official notice ending it. Affiliate-only = UNVERIFIED and ranked below verified offers.
- "CA eligible" = explicit official statement naming California, or an official restricted-state list that omits CA
  (weaker; labelled "by omission"). Affiliate-only = UNVERIFIED.
- Net $ = expected value with the method the terms allow: if the qualifier must be unhedged, EV of the qualifier is
  -fees - spread (fair-priced market) and its max loss is the full qualifier stake. Credit leg valued at the hedged
  lock (SR ~0.95-0.97 x credit, SNR ~B(1-q) - fees) from money_now.md hedge_calc.py; not recomputed with live quotes.
- Max loss = worst case of the qualifying trade (own cash at risk) assuming the credit leg is hedged or used on a
  favourite; forfeiture risk of the credit listed separately.
- Kill: if a venue's official terms cannot be read today, do not carry forward the 11-20 status as "live"; say
  "unverified since 11-20".

## Log
- 06:45Z research-rag: only bioacoustics corpus, nothing on promos. Prior: money_now.md A table, promos_A1/A2.md (11-20).
  User already claimed DraftKings Predictions (inbox 11-24 03:06Z: SR confirmed) -> DK excluded from this list.
- 06:50Z Gemini: re-downloaded CFTC rules10162634421.pdf (curl, plain UA; CFTC robots.txt "Allow: /", use=reference),
  text -> results/promo_verify_1124/pdfs/gemini_rules10162634421.txt. Terms identical to 11-20. NEW detail: s.1 "Any change
  to those terms or to the Promotion Term will be implemented only through a new Rule 40.6 self-certification filing";
  s.9 notice >=7 days before on gemini.com/titan/notices (JS-only, unreadable). Later Gemini filings found: 10-22
  (rules10232635843, liquidity incentive termination), 10-26 (rules10262636962, Liquidity Rewards eff. 11-10). None
  amends the New User Promotion. CFTC index page 503 (Cloudflare) -> not pursued.
- 06:55Z Polymarket US docs (user-programs.md, withdraw-promo.md, fees.md): unchanged vs 11-20. Campaign A "from
  October 16, 2026 | Deposit at least $10 | 7 days | Receive $25". Credit expiry still not posted.
- 06:55Z ProphetX filings page: newest promo filing still the 10-15 amendment ($40 -> $10/$10/$20); 11-12 Rewards &
  Loyalty (eff. no earlier than 12-01); no Nov amendment of the bonus. prophetx.co/lobby/news shows "TRADE $40 · GET $40".
- 07:00Z OG.com: signup page now "*Eligibility, qualifying-trade requirements and reward type vary by jurisdiction";
  terms page unchanged ($25 coupon + 100% boost cap $25; trades < $0.80, settle within 7d of verification).
- 07:00Z FanDuel: /predicts/new-customer-offer 403 to WebFetch; official CA research article still says "began on
  August 24, 2026, and will end October 20, 2026"; Rotowire (affiliate) 11-18 still advertises $100/5 days. No official
  source shows a period covering 11-24 -> UNVERIFIED-LIVE.
- 07:00Z Webull: banner unchanged; T&C (webull.com/ko-yield/1791775405665-d61651) JS-only; affiliates conflict (voucher vs
  cash vs fractional share) -> UNVERIFIED terms.
- 07:00Z Novig: novig.us -> 301 novig.com/landing -> 405 to WebFetch. No official Notice found; affiliates (Rotowire
  Nov 2026) say $10 -> $50. Filing rules09162626976 ($50, 5x$10, <=0.60, 30d, SR) is the only official source.
- 07:10Z Fanatics Markets (NEW): official terms fanaticsmarkets.com/legal/tradeandget (robots.txt allows; plain curl,
  saved results/promo_verify_1124/html/fanatics_tradeandget.{html,txt}); fees /legal/fcm-fee-disclosure; states
  /legal/availability. Offer 1 verified (details in table). Also OG help center (help.og.com) read: Bonus/Rebate/Boost.
- 07:20Z Crypto.com app: Covers 11-23 lists "up to $50" match; no official US page found -> dropped (unverified).
  Kalshi new-user $55/$60 (CBS/Covers 11-23): user is an existing Kalshi customer -> ineligible. Underdog: official
  homepage still fantasy "play $5, get $50"; 11-20 official T&C: bonus "cannot be used for prediction picks" -> dropped.
- 07:25Z arithmetic: results/promo_verify_1124/scripts/promo_math.py -> results/promo_verify_1124/promo_math_out.json.
  Assumptions: venue price = Kalshi price same side, Kalshi opposite at 1-p+0.01, Kalshi taker fee ceil(0.07*C*x(1-x)).
  No live quotes pulled (prices move by game day); recompute at execution.

## Per-venue verification (all read 2026-11-23 PT = 2026-11-24 UTC)
Legend: OFF = official (venue domain/docs or venue's own CFTC filing); SEC = affiliate/secondary; SNIP = search snippet.

### 1. Fanatics Markets (Morton St. Trading Investments LLC FCM -> CDNA exchange) -- NEW, LIVE, CA EXPLICIT
- Live: Offer 1 "Daily Trade Match (10-Day)": "The Promotion Period begins at 12:00 a.m. ET on November 16, 2026 and ends
  at 11:59 p.m. ET on December 13, 2026" (registration+KYC must fall inside). OFF fanaticsmarkets.com/legal/tradeandget.
- CA: "open to legal residents of the following jurisdictions: AL, AK, CA, DE, FL, GA, HI, ID, ME, MN, MS, NE, NH, NM,
  ND, OK, OR, RI, SC, SD, TX, ..." 21+, new customers only (OFF tradeandget); /legal/availability lists CA (OFF).
- Qualifier: must "elect this Promotion as your one new-customer sign-up offer" at registration (final). Each rolling
  24h "Day" (from KYC time, 10 Days): "the first trade placed during the Day ... that is a purchase of event contracts
  made using your cash balance and is priced at $0.90 or below". Reward = FanCash "equal to the U.S. dollar volume of the
  Qualifying Trade for that Day, up to a maximum of $10.00 per Day"; max $100; credited within 72h; missed Days forfeit.
- Credit: "FanCash can be applied to trades and to trading fees. Payouts from trades placed using FanCash will be
  transferred to the customer's cash balance, at which time they may be withdrawn". Expires "seven (7) days after
  issuance". Non-withdrawable itself. STAKE: same wording as DK ("Payouts ... transferred to ... cash balance"), and DK
  paid full $52.00 on a $49.92 PD trade (inbox 11-24 03:06Z) -> SR inferred, not stated.
- Anti-abuse (verbatim): excluded trades include "(c) trades involving self-matching, wash trading, coordinated or
  collusive trading, trading between accounts under common ownership or control, or other manipulative or disruptive
  trading practices"; "(a) trades funded in whole or in part by FanCash"; Sponsor may "withhold or reverse any Reward" if a
  person "tampers with or abuses the Promotion ... uses multiple or false accounts". NO hedging / opposite-side clause;
  nothing about other venues.
- Fees: "Total Trading Fee of between $0.00034 and $0.02 per contract", $0.016 @0.90, $0.018 @0.85, $0.014 @0.95 per
  contract (OFF /legal/fcm-fee-disclosure, "Effective September 2, 2026"); ACH deposit free, debit up to 2%.
- PITFALL: a smaller cash buy placed first in a Day becomes that Day's qualifier (match = its volume). Make the Day's
  FIRST cash purchase >= $10 at <= $0.90. FanCash-funded buys are disregarded for qualification.
- Corroboration (SEC): CBS 11-23 "Up to $100 in FanCash ... one real-money cash trade per day at $0.90 or below for 10
  consecutive days".

### 2. Gemini Predictions -- LIVE (no amending filing found), CA UNVERIFIED-OFFICIAL
- Live: "begins at 12:00:00 a.m. ET on Friday, October 30, 2026, and ends at 11:59:59 p.m. ET on January 15, 2027";
  "Any change ... will be implemented only through a new Rule 40.6 self-certification filing" (OFF CFTC
  rules10162634421.pdf). Later Gemini filings found (10-22 rules10232635843, 10-26 rules10262636962) do not touch it.
  Notices page gemini.com/titan/notices is JS-only (unread) -> early termination cannot be fully excluded.
- CA: only "Be located in a jurisdiction where Gemini Predictions and the Promotion are available" (OFF); support page
  says US users only (SNIP; direct fetch 403). No official CA statement either way.
- Qualifier: ONE purchase >= $20.00 excl. fees, price $0.20-$0.80 inclusive, cash-funded, "hold the entire position ...
  continuously from execution until the contract settles"; reward $40 Predict Cash "credited upon settlement regardless
  of whether the contract settles in the participant's favor", ordinarily within 7 days; void/cancelled contract -> no
  reward. Must be a Self-Clearing Member (not via FCM/IB).
- Credit: Predict Cash non-withdrawable, 30-day expiry; "When a contract purchased with Predict Cash is sold or settled,
  the USD proceeds ... will be credited to the participant as ordinary cash" -> SR explicit.
- Anti-hedge (verbatim): "Hedging or offsetting a Qualifying Purchase through other accounts or related positions in
  order to avoid market risk through settlement"; also "offsets it by purchasing the opposite outcome in the same
  market" forfeits. Bans hedging the QUALIFIER anywhere ("other accounts or related positions"); credit leg not covered.
- Fees: taker 0.07*C*P*(1-P), maker 0 through 2026-12-31 (11-20 read: gemini.com/fees/predictions;
  developer.gemini.com maker-fee-free promo) -- not re-read today.

### 3. ProphetX -- LIVE (button + no amendment), CA by implication
- Live: filings page newest bonus filing = Oct 15 amendment; Nov filings are Rewards & Loyalty (11-12), maintenance,
  products (OFF prophetx.co/lobby/t-c/filings). prophetx.co/lobby/news shows "TRADE $40 · GET $40" (OFF). Program ends
  "earlier of: December 31, 2026; and the date on which 50,000 Participants have satisfied the Qualifying Trade
  Requirement" (OFF Oct 15 filing, read 11-20).
- CA: Terms have "Notice for California Users Under Civil Code Section 1789.3"; "Live Nationwide" news (OFF, 11-20).
  Affiliates: excluded NV (Covers 11-23), UT+NV (CBS 11-23) (SEC).
- Qualifier: ">= $40.00 in the aggregate of funds" in opening trades within 30 days of approval; any price since 10-29.
  $10 Day 1, $10 Day 8, $20 Day 15; each next installment requires the prior one fully used as collateral; each
  installment must be used within 45 days. "may be withdrawn only upon position settlement" -> SR implied.
- Anti-abuse: monitors "concentrated trading in contracts priced near $1.00 and coordinated or offsetting activity among
  related accounts"; one account per person. No cross-venue hedging clause.
- Fees: "2% of net proceeds ... upon settlement" per single-event market (OFF fee filing, 11-20).

### 4. Novig (Ludlow Exchange) -- LIKELY LIVE, CA UNVERIFIED-OFFICIAL
- Official: CFTC rules09162626976.pdf (re-read today): min deposit $10; five $10 credits ($50); 30-day expiry; price
  $0.01-$0.60; one credit per trade, remainder forfeited; "Proceeds resulting from a Trade collateralized by a Deposit
  Credit are settled to the Member's account as funds and are withdrawable" (SR explicit); program "will expire one year
  from its implementation date"; effective period "announced by Notice on the Exchange website" (unread: novig.com 405).
  No later Ludlow filing amends it (checked 10-01, 10-05, 10-16, 10-22 filings).
- Live (SEC only): Rotowire/CBS/Covers Nov 2026 "Deposit $10, Get $50"; Covers 11-23 says "ten $5.00 Trade Credits"
  (conflicts with filing's five $10).
- CA (SEC only): Phandroid 10-27 "live in California"; Covers "Nationwide, 21+".
- Anti-abuse: "multi-accounting, identity duplication, or bonus abuse" -> clawback; "may not be combined with any other
  Incentive Program"; no programmatic trading. No hedging clause. Fees: UNVERIFIED (affiliates: $0 pregame).

### 5. Polymarket US -- LIVE, CA UNVERIFIED-OFFICIAL (no state list in docs)
- Live: "Campaign A (from October 16, 2026) | Deposit at least $10 | 7 days | Receive $25" (OFF docs.polymarket.us/
  incentives/user-programs.md). First deposit only; no trade required; credited within 24h.
- CA: official docs list no state restrictions (signup.md, trading-restrictions.md); SEC: Covers 11-23 "all states
  except Nevada"; defirate 11-18 "except Louisiana and Nevada".
- Credit mechanics (OFF withdraw-promo.md): "Your Cash is used first"; "Losses on any trade are taken from your Bonus
  first, then from your Cash"; Example #1 (cash trade loses, $10 Bonus -> "up to $10 added to your Cash"); Example #2
  (Bonus trade wins: profit to Cash, Bonus returns). Not withdrawable; expiry NOT posted (filing allows 7/10/30 days).
- Anti-abuse: "fraud, abuse, manipulation, self-dealing, self-referral, coordinated activity" (OFF). No hedging clause.
- Fees (OFF fees.md, eff. 11-14): new accounts days 1-7 taker 0.0695*C*p(1-p), then 0.095 (<$10k/30d); maker 0.
- KEY: Bonus is a loss absorber -> a cash trade of $25 is a FREE ROLL: lose -> Bonus refunds $25; win -> keep profit
  and Bonus recycles. EV per round 25(1-q); repeating until a loss gives E = $25 (minus fees), P($0) = 1-q.

### 6. OG.com (NADEX d/b/a OG.com Markets / CDNA) -- LIVE, CA via "available in all U.S. states"
- Live: og.com/posts/welcome-offer-k3r9p "Starts October 20, 2026 (limited time only)", no end date (OFF).
- CA: "OG.com is currently available in all U.S. states" (OFF help.og.com 12752911, updated 2026-07-27); Bonus reward
  type "All jurisdictions except the State of New York" (OFF help 15478570). SEC Covers 11-23: excludes NY, AZ.
- Qualifier: "$25 in cumulative volume, including fees", contracts "below $0.80", trades must settle within 7 days of
  identity verification, held to settlement, no rewards used. Reward: $25 Bonus + 100% boost cap $25 a week later.
- Credit: Bonus "Lowers the upfront cost of your trade ... Use a $20 bonus on a $100 trade. You only pay $80." ->
  SR implied (you own full contracts). Bonus only on contracts "below 80%"; excess forfeited if trade < bonus.
  EXPIRY CONFLICT: T&C "30 days from disbursement" vs help 16897429 "seventh day from issuance at 23:59 ET". Plan 7.
- AMOUNT CONFLICT: OG's own CFTC filing (rules10092632823) still says "up to $10" vs page $25+$25.
- Anti-abuse: "Wash trading," "Circular trading," "Any activity intended to manipulate promo eligibility without genuine
  market intent"; OG's determination "final". No hedging clause (a hedged qualifier arguably lacks "genuine market intent").
- Fees: CDNA taker 0.07*C*P*(1-P) (11-20 Crypto.com help; not re-read).

### 7. FanDuel Predicts -- UNVERIFIED LIVE (official terms ended 2026-10-20)
- /predicts/new-customer-offer 403 today; official CA research article: "began on August 24, 2026, and will end
  October 20, 2026" (OFF, read today). Rotowire 11-18 and CBS 11-23 still advertise $100/5 days (SEC); Covers 11-23
  itself notes it "ran from Aug. 24, 2026 to Oct. 20, 2026". CA named in terms. SNR explicit, 24h expiry, $0.02/contract.
  -> Only act if the app shows the offer with terms dated to cover today.

### 8. Webull "$20 Voucher" -- terms UNREADABLE
- Banner unchanged "New to Event Contracts? A $20 Voucher Awaits. Claim, Trade, Redeem." (OFF webull.com/prediction-
  markets); T&C link JS-only. Affiliates conflict (voucher / withdrawable $20 / fractional share). Existing Webull users
  "activate your Events account directly from your profile" (OFF) -> user (has Webull) can read the T&C in-app.
  Fees $0.01 Webull + $0.01 Kalshi per contract (OFF, 11-20). Webull contracts are Kalshi contracts.

### Dropped / ineligible
DraftKings (claimed, SR confirmed); Kalshi new-user (existing customer); Robinhood (holds RH event contracts);
Underdog (bonus fantasy-only per official T&C 11-20); Crypto.com app (no official US offer); Coinbase, PrizePicks (none
found 11-20, not re-checked); Kalshi referral (needs a different referee; self-referral banned).

## Ranking (results/promo_verify_1124/promo_math_out.json; venue price = Kalshi price, 1c spread)
| rank | venue | live? | CA? | expected net $ | max loss $ (qualifier worst case) | Kalshi capital |
|---|---|---|---|---|---|---|
| 1 | Fanatics Markets Offer 1 | yes (OFF, reg by 12-13) | yes, explicit | +88.5 hedged / +91.5 EV unhedged (10 Days) | 0.40/Day hedged; 10.99/Day unhedged | ~2.16/Day |
| 2 | Novig | likely (filing; SEC live) | unverified | +45.8 hedged (5 x 9.16) | ~0 (deposit withdrawable); 6.84/credit if a credit is voided after hedging | 6.84/credit |
| 3 | Gemini | yes (OFF) | unverified | +38.9 EV | 20.00 (qualifier must ride unhedged) -> worst net +18.9 if credit paid | ~3.1 |
| 4 | ProphetX | yes (OFF) | implied | +35.2 hedged | 1.29 hedged ($40 unhedged) | ~11 then ~1.9/credit |
| 5 | Polymarket US | yes (OFF) | unverified | ~+25 EV free roll (P($0)=1-q) or +17.1 locked | ~0 if docs hold (maker fee 0) | 0 (free roll) / 112 (lock) |
| 6 | OG.com | yes (OFF) | yes ("all U.S. states") | +23.3 (bonus lock 23.8 - qual 0.53) + boost EV ~12.5 | 25.65 qualifier unhedged; 25 boost bet | ~7.2 |
| 7 | FanDuel Predicts | UNVERIFIED (terms ended 10-20) | yes if live | +49..64 hedged if live | ~0.3/day | 15-77/day |
| 8 | Webull voucher | unknown terms | unknown | ~+18.7 if cash-like | ~1-2 | ~1.9 |

## Steps (proposal; the user does every sign-up/KYC/order; recompute prices at execution)
0. Kalshi hedges need cash in the right shard (commit 3bbf7a9: shard 0 "nearly empty"); NFL = shard 0, NBA = shard 3.
1. Polymarket US (max loss ~0): sign up, FIRST deposit $25 (>= $10 qualifies; $25 lets the Bonus absorb a full $25
   loss) -> wait for $25 Bonus (<=24h) -> post a MAKER buy of ~$25 on a side at q ~0.5-0.8 -> if it loses, Bonus
   refunds the cash; if it wins, repeat. Confirm the Bonus expiry shown in-app first.
2. Fanatics Markets (start soon; KYC at a convenient time of day since Days roll 24h from KYC): sign up, ELECT Offer 1
   (not Offer 2), ACH deposit ~$25 -> each Day, the FIRST cash buy = 12 contracts at <= $0.90 (>= $10 volume)
   [optionally hedge 12 opposite on Kalshi at ~0.11] -> FanCash ($10, 7-day expiry) on a ~0.94 favourite, hedge 10
   opposite on Kalshi. Repeat Days 1-10.
3. ProphetX: sign up direct, $40 of opening trades on ~0.80 favourites, hedge on Kalshi -> each installment fully as
   collateral on 0.80-0.85 favourites (not near $1.00), hedge on Kalshi; Day 8 and Day 15 follow.
4. Novig: deposit $10 -> each $10 credit on a side priced <= 0.60, hedge the opposite on Kalshi (one credit per trade).
5. OG.com: $25 qualifier (< 0.80, settles within 7 days of KYC), do NOT hedge it -> $25 Bonus on a < 0.80 side, hedge on
   Kalshi within 7 days -> boost a week later: optional EV bet (max loss = stake).
6. Gemini (largest at-risk leg, $20): sign up direct, ONE maker buy >= $20 at 0.20-0.80, never hedge it, hold to
   settlement -> $40 Predict Cash (30 days) on a heavy favourite + Kalshi underdog hedge.
7. Webull / FanDuel: open the app, read the in-app terms; act only if they cover today.

## Changes since 2026-11-20 (money_now.md section A)
- NEW #1: Fanatics Markets Offer 1 (registrations 11-16..12-13): $10/Day FanCash x 10 Days, CA explicit, no hedging
  clause, payouts -> cash (SR-style wording). Not on the 11-20 list (was "affiliate-only").
- OG.com: help center now gives CA coverage ("all U.S. states") and the reward type outside NY ("Bonus" = lowers cost,
  SR implied); new conflict on expiry (7 vs 30 days); filing still says "up to $10".
- Polymarket US: terms unchanged, but the official FAQ makes the $25 Bonus a loss absorber -> an unhedged maker trade
  is a free roll (max loss ~0) worth ~$25 EV; better for a cash-tight user than the 11-20 hedged $15.30 plan ($164
  Kalshi capital).
- FanDuel: still no official period covering today (Covers 11-23 itself says it ended 10-20). Gemini, ProphetX, Novig,
  Webull: unchanged; no amending filings found.

## Learnings
### Fanatics Markets Offer 1 pays up to $100 FanCash to new CA customers through registrations on 2026-12-13
$10/Day for 10 rolling 24h Days; each Day's first cash buy at <= $0.90 earns FanCash equal to its $ volume (cap $10);
FanCash expires 7 days; "Payouts from trades placed using FanCash will be transferred to the customer's cash balance";
no hedging clause. Hedged value ~$88.5 for ~$0.40/Day qualifier cost (results/promo_verify_1124/promo_math_out.json;
fanaticsmarkets.com/legal/tradeandget read 2026-11-24). Why it matters: largest verified promo left for this user.
### Polymarket US $25 Bonus is a loss absorber, so the unhedged play is a free roll
docs.polymarket.us/learn/faq/withdraw-promo.md: "Losses on any trade are taken from your Bonus first, then from your
Cash"; a winning trade returns the Bonus. A $25 cash maker trade at price q has EV 25(1-q) per round and ~$25 if
repeated until a loss, with max loss ~0 (maker fee 0). Read the mechanics clause, not just "non-withdrawable", before
choosing a hedge: hedging here needed $112 Kalshi capital to lock only $17.
### A venue's help center can contradict its own promo T&C and CFTC filing
OG.com: T&C says rewards last 30 days, help article 16897429 says they expire on the 7th day; its CFTC filing
(rules10092632823) says "up to $10" while the page says $25 + $25 boost. Plan for the stricter term and record each
source separately (notes/promo_verify_1124.md section 6).
### "First trade of the Day" qualifiers punish small test trades
Fanatics counts only the first cash purchase each Day; a $1 test buy placed first would earn a $1 match that Day.
Make the first cash buy the full-size qualifier (tradeandget "Qualifying Trade" definition).
### Gemini terms can change only by a new CFTC 40.6 filing, so searching for newer filings is the liveness test
rules10162634421.pdf s.1: changes "will be implemented only through a new Rule 40.6 self-certification filing"; no newer
Gemini filing touching the New User Promotion found through 2026-11-24 (10-22 and 10-26 filings are liquidity programs).
The CFTC filing index page returns 503 behind Cloudflare; PDFs download with plain curl (robots.txt Allow: /).
