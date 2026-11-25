# promo_hunt_2 -- more CA small-capital sign-up / referral bonuses (agent started 2026-11-25 02:31Z = 2026-11-24 18:31 PT)

Agent: promo_hunt_2 (research subagent of the quant-trader lead; protocol sports/notes/AGENT_PROTOCOL.md).
Read-only: no sign-ups, no logins, no credentials, no orders, no git. Scratch: sports/results/promo_hunt_2/.

## Question
Beyond notes/promo_verify_1124.md + notes/PROMO_TRACKER.md (prediction venues) and the seed-runner bank/broker work
(BONUS_LEDGER.md, brokers.md, LEARNINGS.md), which sign-up / referral / deposit bonuses can a CALIFORNIA resident collect
now with <= ~$100 at a time and capped risk? Categories: crypto exchanges/apps, small-deposit brokerages, new venue
launches, referral programs on venues the user already holds. Excluded: sweepstakes/social casinos, DFS pick'em in CA
(unless officially legal+offered), large deposits, payroll direct deposit.

## Already held by the user (=> new-customer offers ineligible) -- from local files
- Brokerage: Robinhood, Fidelity, Webull, Schwab (6 accts), Public (3 accts), Moomoo (funded; LEARNINGS moomoo.newuser.eligibility)
  (seed-runner/tracker/accounts.csv "open" rows 2026-11-03; LEARNINGS). IBKR (LEARNINGS status.ibkr.withdraw_nov2026).
- Banks: Wells Fargo, U.S. Bank (BONUS_LEDGER.md). Prediction: Kalshi, DraftKings Predictions.
- PLANNED by seed-runner (do not double-book; that session owns them): SoFi Invest via Finder ($1,250), tastytrade,
  E*TRADE, Merrill, SoFi Checking, Chase, Citi, Four Leaf. -> excluded here.
- Crypto accounts: NOT recorded anywhere local (grep of LEARNINGS/tracker/seed-runner for coinbase/kraken/crypto.com/
  uphold/cash app: no account rows). Robinhood Crypto = part of Robinhood (held). User must confirm Coinbase/Cash App/
  PayPal/Venmo status before any "new user" offer.
- Prior survey (LEARNINGS brokerage.bonuses.low_deposit_2026_11, [U], 11-11): only small leads Acorns $20, Stash $5-30,
  eToro US $30 referral for $100 -- re-verify here.

## Pre-registered rules (written 02:35Z, before reading any offer page)
- "Live" = official page (issuer domain, help center, official filing) dated or effective covering 2026-11-24 with no
  later notice ending it. Affiliate / aggregator / search snippet only = UNVERIFIED (listed, ranked below verified).
- "CA eligible" = official text naming CA, or official restricted-state list that omits CA ("by omission", weaker),
  or official "all 50 states / US residents" wording. Otherwise UNVERIFIED.
- Max loss = worst-case $ of the qualifying action if the price of what must be bought goes against you during the
  required hold, assuming the cheapest allowed low-volatility asset (stablecoin, money-market, cash) + all fees.
  If the qualifier forces a volatile asset (e.g. must buy BTC), max loss = 100% of that buy in principle; I report a
  realistic stress (-30% over the hold) separately and say so.
- Net $ = reward (cash value; stock/crypto reward at face; locked credit valued at what can be extracted) - fees -
  expected price drift (0 for stablecoin/cash) - any subscription fees over the hold. Reward paid in volatile crypto:
  face value, flagged.
- Minutes = my estimate of user time (sign-up+KYC ~10-15 min, funding 5 min, trade 5 min, withdrawal 5 min) [estimate].
- Kill: an offer is DROPPED if (a) CA explicitly excluded, (b) needs > ~$100 at once or payroll DD, (c) only affiliate
  evidence AND the official page contradicts or is silent on an end date that has passed, (d) user already a customer.
- Every number cites a URL (OFF/SEC/SNIP) + read date 2026-11-24 PT.

## Log
- 02:31Z started; research-rag not queried (corpus is bioacoustics only per promo_verify_1124 log 06:45Z).
- 02:40Z Coinbase: help.coinbase.com is behind a Cloudflare challenge (plain curl 403, html saved
  results/promo_hunt_2/html/cb_incentive.html = challenge page). Official-domain SNIPPETS only (WebSearch allowed_domains
  coinbase.com): new-user page "All new users will randomly receive a Reward when complete with the first purchase ...
  between $10 - $2,000", first purchase >= $50 single transaction, referred users NOT eligible; spin-the-wheel variant
  "$30 - $2,000" in BTC. No date on snippet -> UNVERIFIED-LIVE. User may already hold Coinbase (unknown locally).
- 02:45Z Gemini crypto: gemini.com robots.txt has "Crawl-delay: 600". I fetched robots + 2 legal pages back-to-back
  (02:44Z) before reading it -> breach of the 600 s crawl-delay on 2 requests; no further gemini.com fetches this run.
  Pages are Next.js; legal body not in HTML (client-loaded). Strings embedded in the page bundle (OFF, read 11-24):
  "Invite friends to Gemini and you'll both earn $50 in the crypto of your choice."; disclaimer "There is a $1500 limit
  to the ${REWARD_VALUE} Standard tier reward per year. Each referred friend must be new to Gemini and successfully
  complete account sign-up using the referral link. The reward will be credited to both accounts 32 days after the
  referred user's sign-up date, provided the referred user trades at least 100 USD ... within the first 30 days."
  (results/promo_hunt_2/html/gem_legal_gemini-referral-rewards-program-terms.clean.txt). Older strings in the same
  bundle say $15 -> the live value is the ${REWARD_VALUE} variable; $50 per the hero string. Search snippet of
  gemini.com/referrals: "$1875 limit to the $50 Standard tier". -> amount $50 likely, UNVERIFIED exact.
  STACKING: Gemini Predictions promo eligibility (rules10162634421 s.2) = "never executed a trade in any event contract
  on Gemini Predictions" -> a crypto referral sign-up does NOT disqualify it (no referral exclusion in s.2). Needs a
  real referrer who already has Gemini.
- 02:55Z Kraken OFFICIAL rules saved results/promo_hunt_2/html/kraken_promotions.{html,txt} (kraken.com/legal/promotions,
  robots allows; read 11-24): "Kraken New Customer Bitcoin Reward"; US "excluding Maine, New York, Washington and Florida"
  -> CA by omission; 1 Nov 2026 - 30 Apr 2027 or budget; "sign up ... through a participating partner's link", within 30
  days: Intermediate KYC, "deposit at least $100 in fiat", "trade at least $100 in total"; "Stablecoin-to-stablecoin and
  fiat-to-stablecoin trades do not count"; draw $5 33% / $10 32% / $15 15% / $25 12% / $50 6% / $100 2% (EV = $15.10,
  my arithmetic); paid in BTC within 14 days; "30-day withdrawal hold".
- 03:00Z Rakuten store pages (robots allows /shop/; plain curl, 3 s spacing; saved results/promo_hunt_2/html/rakuten/*.txt;
  page "Last Edited: November 18, 2026" on Kraken) [OFF for Rakuten's own cash-back offer]:
  * Kraken "$150 Cash Back was $50": "first-time U.S. customers who complete one qualifying trade with a transaction value
    of $200 USD or more, excluding fees, within 30 days of account sign-up"; excludes cash-to-stablecoin, stablecoin-to-
    stablecoin, staking, wash trades; "Applying a separate offer with Kraken (Referral Code, Discount, or Coupon) will
    negate your Cash Back"; posts "within a few months". Rakuten also lists "Buy crypto on Kraken, earn up to $100 in
    Bitcoin" as a Kraken deal -> Rakuten looks like a "participating partner" (UNVERIFIED that the BTC draw stacks).
  * Coinbase "$12.50": "first trade exceeds $50", new U.S. customers, "Offer not valid in the state of Hawaii".
  * eToro "$50": "new customer opens an eToro account and trades a minimum of $100.00".
  * SoFi Crypto "$30": new SoFi Crypto customers, "initial purchase of $30 or more within 45 days". CONFLICT RISK with the
    seed-runner's planned SoFi Invest (Finder) / SoFi Checking (Rakuten) -> sequence after those or skip.
  * MoneyGram "$15": first online transfer, "minimum $50 transfer". Remitly "$7.50" ($100 first transfer, intl).
    Xe "$11.25" ($300). Chime "$50" needs payroll DD $200+ (EXCLUDED: payroll). Current "$17.50" needs $1,500 DD (EXCL).
    J.P. Morgan SDI "$100" needs $5,000 (EXCL). Willow Wealth (accredited only, EXCL).
- 03:05Z Acorns OFFICIAL acorns.com/acorns-20-bonus-investment-promotion-terms (read 11-24 via WebFetch): "valid from
  10/05/26 through 12/31/2026 for First Time Acorns Users"; code ACORNS20; open Invest + "recurring investment of $5 or
  more"; $20 "Reward Shares" "typically deposited within 30-45 days"; no state restriction; no cash alternative. Fee not
  on the page (subscription fee UNVERIFIED here).
- 03:05Z Uphold OFFICIAL uphold.com/en-us/legal/uphold-new-user-first-trade-program: $30 BTC, but "ends on July 15, 2026"
  -> DEAD (no newer US page found).
- 03:15Z TopCashback: plain curl 403 (bot wall) -> not pursued; snippet only (SNIP topcashback.com/kraken): "$25 cash back
  ... single trade of $1,000 or more" -> worse than Rakuten; skip.
- 03:15Z Kraken fees (SNIP kraken.com/pro/fees): Pro Tier 1 spot "0.40% maker / 0.80% taker"; "ACH (Plaid) Deposit minimum
  1 USD Deposit fee Free"; withdrawal hold shown as "None" in one table and "7 days" in another -> plan for a 7-day hold.
- 03:20Z eToro US: live-states page (OFF etoro.com/en-us/trading/live-states, read 11-24) lists California; crypto not in
  NV/HI/PR/VI. Fees page (OFF etoro.com/trading/fees): "zero commission for ETF trades"; "For withdrawals from a USD
  investment account to external accounts, etoro charges a fixed $5 fee" ($30 min) vs US help snippet "etoro USA LLC
  does not charge a withdrawal fee" (SNIP) -> assume $5. eToro own deposit bonus (Jan-2026 PDF, SNIP): $200-499 -> $10 crypto;
  live status UNVERIFIED -> not counted.
- 03:20Z Acorns fee (SNIP acorns.com Program Agreement / Form CRS): Bronze $4/month for sign-ups after 2026-08-16.
- 03:20Z Rakuten welcome-bonus FAQ (OFF rakuten.com/help/article/welcome-bonus-new-member-faqs): "make a qualifying purchase
  of at least $25 within 90 days of becoming a member"; amount not stated; whether account sign-ups count NOT stated.
- 03:25Z PrizePicks: official CA page (SNIP prizepicks.com/states/california) offers code CAPicks "$150 in Bonus Lineups
  if you win your first $5+ Lineup" -> DFS pick'em; brief excludes unless official source confirms LEGAL in CA; none
  found -> EXCLUDED (also the bonus needs a winning lineup).
- 03:25Z Uphold $30 BTC ended 2026-07-15 (OFF); Binance.US Rare Evo ended 2026-08-07 (SNIP support.binance.us) -> DEAD.
- 03:35Z Rakuten Banking & Finance category (OFF rakuten.com/banking-finance, saved html/rakuten/banking-finance.html):
  38 stores. Small-capital, non-payroll ones kept: Kraken $150 (was $50), eToro $50, SoFi Crypto $30, SoFi Invest $50
  ($100 initial funding, new Active Invest), SoFi Plus $25 (pay $10 subscription), MoneyGram $15, Coinbase $12.50,
  Xe $11.25, Western Union up to $10 ($100 first transfer / $12.50 prepaid card funded $100), Remitly $7.50. Dropped:
  Alliant $200 (needs $500 payroll DD + $1,000 balance), Valley Direct $50 ($1,000 deposit), Chime/Current (payroll DD),
  JPM SDI ($5,000), Mercury/Grasshopper (business), Motley Fool/Quicken/Self (paid subscriptions), Acorns Early (kids).
  SoFi Invest/Crypto/Plus -> HAND OFF to seed-runner (it owns SoFi Invest via Finder + SoFi Checking via Rakuten; one
  Rakuten SoFi cash back may block another; opening SoFi Crypto first may make the user an existing SoFi customer).
- 03:40Z Fanatics Markets RAF OFFICIAL fanaticsmarkets.com/legal/raf (robots "User-agent: * Allow: /"; saved
  html/fm_legal_raf.terms.txt): "Effective as of November 9, 2026"; period "November 9, 2026 ... ends ... November 15,
  2026" -> the posted window is OVER (renewal unknown); Referrer gets "$25 in FanCash for each Referee who completes a
  Qualifying Trade of $25 or more"; FanCash expires 7 days; "Referees do not earn FanCash ... remains eligible for
  Fanatics Markets' then-current new-customer sign-up offer"; CA listed; "only share ... with people they know".
- 03:45Z Gemini referral (SNIP gemini.com): New User Incentive excludes "users who were referred to Gemini from the Gemini
  Referral Program"; referral "intended solely for individual users"; general promo terms: disqualify for "VPN, shared
  device, shared identification credentials". No shared-address rule found.
- 03:45Z DK Predictions RAF: official page predictions.draftkings.com/en/refer exists (SNIP), amounts not readable;
  affiliates say $25 PD per friend, cap ~5 (SEC). Kalshi RAF: help.kalshi.com says amounts vary, credits may be
  withdrawal-restricted (OFF, via promos_A2.md 11-20); affiliates $25 each up to 40 (SEC Action Network Oct 2026).
  Polymarket US RAF: "Campaign A | Deposit at least $10 | Receive $25" both sides, "limited to 50 friend referrals"
  (OFF docs.polymarket.us, via results/money_now/promos_A2.md line 148, read 11-20). Robinhood tiered referrals from
  2026-09-01 (OFF snippet robinhood.com/us/en/support/articles/tiered-referrals-program). Crypto.com (OFF
  help.crypto.com/en/articles/3124990): US Tier 1 referee milestones volume $100 -> $5, $500 -> $15, $1,000 -> $25,
  $5,000 -> $75 within 30 days of KYC. Cash App $5 each (OFF cash.app/legal/us/en-us/affiliate, eff. 2026-04-16).
  Venmo $5 each, max 10 (SNIP help.venmo.com P2P referrals). PayPal 1,000 points = $10 (page ~554 days old, stale).
- 03:50Z Fees: Kraken instant buy "1% trading fee" + spread (SNIP kraken.com/features/fee-schedule); Coinbase spread-based,
  fee shown at preview (SNIP, old pages) -> 4% round trip assumed; MoneyGram "$0 transfer fee on your first online
  transfer if you're a new customer" (SNIP moneygram.com).
- 03:50Z Sporttrade: prediction market not live in CA (SEC Action Network/FantasyLabs, pre-Nov) -> nothing to claim.
- 03:55Z Arithmetic: results/promo_hunt_2/scripts/rank.py -> results/promo_hunt_2/rank_out.json (assumptions in file).

## Ranking (solo offers the user can do alone; rank by net $/hour; results/promo_hunt_2/rank_out.json)
Read 2026-11-24 PT. OFF = issuer/portal's own page; SNIP = official-domain search snippet; SEC = affiliate.
| # | offer | net $ | max loss $ | capital $ | min | $/h | CA source | live? |
|---|---|---|---|---|---|---|---|---|
| 1 | Kraken new account via Rakuten: $150 cash back | ~146 | ~10 (fees + 3% move; sell same hour) | ~205, ~7 d ACH hold | 35 | ~250 | Kraken rules: US excl. ME/NY/WA/FL (OFF, by omission); Rakuten "first-time U.S. customers" | yes; elevated rate "was $50", END DATE NOT POSTED |
| 2 | eToro US via Rakuten: $50 cash back | ~45 | ~5.5 ($5 withdrawal fee) | ~101 | 30 | ~90 | etoro live-states lists California (OFF) | yes (Rakuten page 11-24) |
| 3 | Coinbase new user ($10-$2,000 BTC, single first buy >= $50) + Rakuten $12.50 | ~20 (at the $10 floor) | ~4.6 | ~51 | 25 | ~49 | Rakuten excl. Hawaii only (OFF); Coinbase "New US users" (SNIP) | Rakuten yes; Coinbase reward SNIP only |
| 4 | MoneyGram first online transfer via Rakuten: $15 | ~15 | ~5 | 50 (sent to a real recipient) | 20 | ~45 | Rakuten US; domestic-transfer availability UNVERIFIED | yes |
| 5 | Acorns ACORNS20: $20 Reward Shares | ~12 (after ~2 x $4/mo) | ~12 | 5 recurring | 30 | ~24 | no state limit, US residents (OFF) | yes, to 2026-12-31 |
| 6 | Western Union first transfer >= $100 via Rakuten: up to $10 | ~5 | ~5 | 100 | 20 | ~15 | Rakuten US | yes; fee UNVERIFIED |
Upside not counted: Kraken official BTC draw (EV $15.10, $5-$100, 30-day hold) if Rakuten is a "participating partner".

## Referral programs (need a real friend; the partner is the obvious one, acting on their own account/device/money)
| program | referrer (user) gets | friend must | source | status |
|---|---|---|---|---|
| Gemini crypto (user signs up for Gemini Predictions anyway) | $50 crypto (hero string; ${REWARD_VALUE} var) | trade >= $100 within 30 d; credited day 32; friend also gets $50 | OFF page bundle + SNIP | amount UNVERIFIED-exact |
| Polymarket US | $25 | deposit >= $10 (friend $25 too); max 50 | OFF docs (11-20) | live |
| DraftKings Predictions | ~$25 PD | first trade (SEC) | SEC; official page unreadable | partner's own DK welcome ends 11-29 (LEARNINGS) |
| Fanatics Markets | $25 FanCash per friend trade >= $25 | trade >= $25 | OFF legal/raf | posted window 11-09..11-15 ENDED; check in-app |
| Kalshi | in-app ($25 each per SEC) | trade (SEC: $25) | OFF help (amounts vary) | user is a customer -> referrer side OK |
| Robinhood / Moomoo / Public / Webull | stock / $100 coupons | seed-runner owns partner brokerage (s018, mm01) | LEARNINGS | hand off |
| Cash App / Venmo / PayPal | $5 / $5 (max 10) / $10 pts | send $5 / pay $5 | OFF / SNIP / stale | tiny |

## Dropped / dead (with reason)
Uphold $30 BTC (ended 2026-07-15, OFF); Binance.US Rare Evo (ended 2026-08-07, SNIP); Sporttrade (not live in CA);
PrizePicks CAPicks (DFS pick'em; no official CA-legality source); Stash (no official live offer found, SEC conflict
$5-$100); Crypto.com own sign-up (referral-only; ~$5 at $100 volume); TopCashback Kraken ($1,000 trade); Chime, Current,
Alliant, Valley, JPM SDI (payroll DD or >= $1,000); SoFi Invest/Crypto/Plus (seed-runner conflict); Coinbase Learning
Rewards (status conflicting); Mercury/Grasshopper (business); Xe ($300), Remitly (intl recipient).

## Step-by-step (proposals; the USER does every sign-up/KYC/trade; ask first: do you already have Kraken, eToro, Coinbase,
## Acorns, Rakuten? existing customers are ineligible)
1. Kraken (do first; elevated rate can end any day): join/log in to Rakuten -> set payout (PayPal/check, or Amex MR if
   linked) -> disable ad-blocker -> click "Shop Now" on rakuten.com/shop/kraken -> sign up for a NEW Kraken account in that
   same browser session, enter NO referral/promo code -> verify to Intermediate -> ACH ~$205 (free) -> ONE buy of BTC or
   ETH >= $200 excluding fees (not USDC/USDG) -> sell it back within the hour (Kraken Pro limit/maker order) -> leave USD
   until any ACH hold clears, withdraw -> screenshot the trade + note the Kraken Public Account ID (IIBAN) for any
   Rakuten claim -> cash back posts "within a few months".
2. eToro: Rakuten -> rakuten.com/shop/etoro "Shop" -> open eToro US account in the same session -> deposit ~$101 by ACH ->
   buy $100+ of a T-bill ETF (e.g. a 0-3 month Treasury ETF; ETF commission 0) -> hold a few days until Rakuten shows
   the $50 pending -> sell -> withdraw (assume $5 fee).
3. Coinbase (only if never a Coinbase customer): Rakuten -> rakuten.com/shop/coinbase -> sign up, no other code -> ONE
   single BTC buy of ~$51 (> $50; never split) -> claim the in-app reward -> sell after the reward posts.
4. MoneyGram: Rakuten -> moneygram first ONLINE transfer of $50 to a real recipient (partner) paid by bank account; check
   the $0 first-transfer fee on the quote screen before sending; recipient returns it.
5. Acorns: sign up with code ACORNS20 -> open Invest -> recurring $5 -> wait 30-45 days for $20 Reward Shares -> sell all
   -> withdraw -> close the account the same day to stop the $4/mo fee.
6. Western Union: only if the quote shows a fee <= $3 on a $100 first transfer.
7. Gemini referral: after the user's own Gemini account exists (PROMO_TRACKER #5), the partner may sign up via the
   user's referral link on their own device -> partner trades >= $100 crypto within 30 days -> both credited day 32.
   Partner then is NOT eligible for Gemini's code-based new-user crypto offer (referred users excluded).
8. Polymarket US referral: user signs up (PROMO_TRACKER #1) -> share link with the partner -> partner deposits >= $10.
9. DraftKings Predictions: partner's own DK welcome (first $5 cash trade by 11-29 11:59 pm ET, LEARNINGS) via the
   user's in-app referral link -> read the in-app amount first.
10. Kalshi / Fanatics referral: read the in-app referral terms; Fanatics' posted window ended 11-15.

## Open / uncertain
- Whether the user already holds Kraken, eToro, Coinbase, Acorns or Rakuten: not recorded locally -> ask before acting.
- Kraken $150 end date not posted; "was $50" suggests a short elevated window. Whether a Kraken PRO spot trade counts as
  Rakuten's "one qualifying trade" is not stated (safer: the $200+ BUY on the Kraken app, sell on Pro).
- Whether the Kraken official BTC draw stacks with Rakuten (Rakuten lists it as a deal, but its terms void cash back if
  "a separate offer ... (Referral Code, Discount, or Coupon)" is applied).
- eToro US withdrawal fee: $5 (global fees page) vs $0 (US help snippet). Coinbase fees: not readable (Cloudflare).
- Gemini referral exact amount (page shows $50 hero string; reward value is a template variable).
- MoneyGram domestic US->US bank deposit availability.

## Learnings
### Cash-back portals pay more for crypto/broker sign-ups than the venues' own bonuses
Rakuten pays $150 cash back (was $50) for a NEW Kraken account plus one $200+ non-stablecoin trade within 30 days, while
Kraken's own official new-customer draw averages $15.10 ($5-$100 in BTC). eToro via Rakuten pays $50 for a $100 trade.
Source: rakuten.com/shop/kraken and /shop/etoro read 2026-11-24 (results/promo_hunt_2/html/rakuten/*.txt);
kraken.com/legal/promotions. Why it matters: check the portal page before the venue's promo page.
### Kraken's official US new-customer reward is a random BTC draw with EV $15.10 and a 30-day withdrawal hold
Odds $5 33% / $10 32% / $15 15% / $25 12% / $50 6% / $100 2%; needs a partner link, Intermediate KYC, $100 fiat
deposit and $100 of non-stablecoin trades within 30 days; US excludes ME, NY, WA, FL (kraken.com/legal/promotions,
read 2026-11-24; results/promo_hunt_2/rank_out.json).
### Rakuten finance offers exclude stablecoin buys and void on other codes
Kraken via Rakuten excludes "cash-to-stablecoin conversions" and "Applying a separate offer with Kraken (Referral Code,
Discount, or Coupon) will negate your Cash Back"; Coinbase via Rakuten warns that non-Rakuten codes may void it. So buy
BTC/ETH and sell it back quickly; never add a referral code (rakuten.com/shop/kraken, /shop/coinbase, 2026-11-24).
### Fanatics Markets referral runs in dated weekly windows and pays only the referrer
fanaticsmarkets.com/legal/raf (read 2026-11-24): $25 FanCash to the referrer per friend trade >= $25, posted window
Nov 9-15 2026 (ended); "Referees do not earn FanCash" but keep the new-customer offer. Check the in-app window first.
### gemini.com robots.txt sets Crawl-delay 600
Read robots.txt BEFORE the first page request: this run fetched two gemini.com pages back-to-back before reading it
(2026-11-25 02:44Z). Gemini legal pages are client-rendered, so curl gets only the string bundle; use snippets.
### Almost no small brokerage bonus survives at brokers the user lacks; portals are the exception
Uphold $30 BTC ended 2026-07-15, Binance.US welcome ended 2026-08-07, no official Stash offer found; Acorns $20 needs a
$4/month subscription (net ~$12). The live small-capital value is in portal cash back (Kraken $150, eToro $50)
(notes/promo_hunt_2.md Dropped list).
