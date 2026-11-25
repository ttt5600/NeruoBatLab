# promos_W1 -- EXISTING-customer promos on DraftKings Predictions, Fanatics Markets, OG.com (CA user)

Helper W1 for boost_hedge. Read-only. All pages read 2026-11-24 PT (18:35-19:05 PT = 11-25 02:35-03:05Z).
Tags: OFF = venue's own page; FILING = venue/exchange CFTC Rule 40.6 PDF; SEC = affiliate (unverified); SNIP = search
snippet only. Raw copies: `sports/results/boost_hedge/raw_W1/` (Fanatics legal HTML + extracted `.rsc.txt`/`.rsc.raw`;
CFTC PDFs + `.txt` under `raw_W1/cftc/`). New-customer welcome offers are NOT re-studied (see notes/promo_verify_1124.md).

Bottom line: no public Thanksgiving-week (11-26) or weekend (11-29) boost was found at any of the three venues.
The only live, publicly documented existing-customer earn is Fanatics' Always-On FanCash rate, and its literal terms
probably exclude accounts verified after 2026-08-26. OG.com has three new existing-user programs (weekly trading streak,
weekly basketball combo boost, leaderboards) filed 11-12 and 11-19; as of 11-24 they are still in CFTC 10-day review, so
the earliest they can start is about 11-27. DraftKings Predictions publishes nothing official about existing-customer
promos that is readable without logging in.

---------------------------------------------------------------------------------------------------------------------
## 1. Fanatics Markets (Morton St. Trading Investments LLC FCM -> CDNA)

Official legal index (sitemap-static.xml, OFF) lists: fancash, raf, status-match, sweepstakes, socialgiveawayterms,
survey, ticketgiveaway, tradeandget (new-customer), suo. I downloaded all of them with plain curl (robots.txt `Allow: /`).
The accordion bodies exist only in the Next.js RSC payload, which I extracted to `*.rsc.txt`.

### 1a. Always-On FanCash Earn Rate Promotion -- LIVE, but eligibility is doubtful for a new account
- Source: OFF https://fanaticsmarkets.com/legal/fancash (read 2026-11-24; raw_W1/fanatics_legal_fancash.rsc.*).
- Dates: OFF "The Promotion Period begins at 12:00 a.m. Eastern Time ("ET") on August 26, 2026, and ends at 11:59 p.m. ET
  on December 31, 2026".
- CA: OFF explicit: "Void outside of AL, AK, CA, DE, FL, GA, HI, ID, ME, MN, MS, NE, NH, NM, ND, OK, OR, RI, SC, SD, TX,
  UT, U.S. Virgin Islands, Puerto Rico, Guam and American Samoa".
- **ELIGIBILITY PROBLEM (OFF, verbatim, in this promo's own Eligibility list, RSC rows 73-74 referenced from the
  Aug 26-Dec 31 section):** "be an existing Fanatics Markets customer (for the avoidance of doubt, an "existing customer"
  is an individual who, as of the start of the Promotion Period, holds an active and verified Account)." Read literally,
  an account verified after 2026-08-26 does not qualify. The site's own marketing in the same page payload says "For
  everyone" / "Get up to 10% FanCash back" / "On every trade. Win or lose." (OFF, web config `rewards.everyoneTitle`).
  The two conflict. Ask support or check the in-app rewards screen before counting on it.
- Mechanics (OFF): automatic, no opt-in. "Qualifying Trade" = "a buy or sell order for an event contract ... that (i) is
  filled and settles during or following the Promotion Period and (ii) is funded with cash from the customer's Fanatics
  Markets cash balance (i.e., not funded, in whole or in part, with FanCash, promotional credits, or Awards from any
  other Fanatics offer)". "FanCash earned on each Qualifying Trade equals the trade volume (i.e., contract price
  multiplied by number of contracts filled, in U.S. dollars) multiplied by the applicable earn rate". "Trading fees are
  not eligible to earn Rewards."
- Earn Rate Table (OFF, % of volume by fill price): 0.01 10.00% | 0.02 7.60% | 0.03 6.60% | 0.04 5.80% | 0.05 5.05% |
  0.06 4.40% | 0.07 3.85% | 0.08 3.35% | 0.09 2.95% | 0.10 2.60% | 0.11 2.25% | 0.12 2.00% | 0.13 1.75% | 0.14 1.55% |
  0.15 1.35% | 0.16 1.20% | 0.17 1.05% | 0.18 0.95% | 0.19 0.85% | 0.20 0.75% | 0.21 0.65% | 0.22 0.60% | 0.23 0.55% |
  0.24 0.50% | 0.25 0.45% | 0.26 0.40% | 0.27 0.35% | 0.28 0.35% | 0.29 0.30% | 0.30 0.30% | 0.31 0.30% |
  0.32-0.34 0.25% | 0.35-0.42 0.20% | 0.43-0.99 0.15%. So "up to 10%" applies only to 1-cent contracts. At game-winner
  prices (0.43-0.99) the rate is $0.15 per $100 of volume.
- Pays: FanCash, not cash. OFF: "Rewards are non-withdrawable and are not redeemable for cash or cash equivalent."
  Credited "within seventy-two (72) hours after the underlying Qualifying Trade settles". Expiry: "expires three hundred
  sixty-five (365) days after your last earning activity with the Fanatics family of companies".
- Stake returned when FanCash is used? The wording implies yes, but it is not stated outright. OFF: "Payouts from trades
  placed using FanCash will be transferred to the customer's cash balance, at which time they may be withdrawn." The same
  wording at DK paid the full contract value ($52.00 on a $49.92 credit trade, inbox 11-24). FanCash can also pay fees.
  Using it needs a manual choice. OFF: "To place a trade using FanCash, the customer must manually select that option in
  the trade slip; otherwise, the trade will be placed with cash."
- Anti-abuse (OFF verbatim): "Limit one (1) Account per customer; only the original, authorized Account holder may earn
  Rewards. Any attempt to participate through multiple Accounts, or to manipulate the earn rate (including by entering
  and immediately exiting positions, wash trading, or coordinated trading) will result in disqualification and forfeiture
  of any Rewards." FanCash Program Terms (OFF): "If we believe (in our sole discretion) that you are or have abused,
  misused or gamed the FanCash Program ... we may: Suspend ... Revoke any unredeemed FanCash ... Cancel or claw back any
  transactions involving FanCash". **No hedging, opposite-side, or other-platform clause.**
- Sponsor may change rates (OFF): "Sponsor reserves the right, in its sole discretion, to amend, suspend, or terminate
  the earn rates set forth in this table prior to the end of the Promotion Period".
- Fees (OFF /legal/fcm-fee-disclosure "Effective September 2, 2026"): "Total Trading Fee of between $0.00034 and $0.02
  per contract" (prior read). New clause read today: "The Total Trading Fee may be reduced or waived, including to $0.00,
  in connection with promotions or special offers on disclosed terms." No standing fee holiday was found.

### 1b. FanCash Multiplier Promotion (Selected Professional Football Games) -- EXPIRED, no Thanksgiving version posted
- OFF /legal/fancash: the only multiplier text posted covers "Thursday, September 17, 2026, ... and ends at 11:59 p.m.
  ET on Monday, September 21, 2026" (DET-BUF Thursday game, NY-LA Monday game).
- Tier-gated (OFF): "hold Fanatics ONE Pro, Fanatics ONE Gold, Fanatics ONE Platinum, or Fanatics ONE Black membership
  status". The multiplier (% added on top of the base rate) by price band (<=0.29 / 0.30-0.70 / 0.71-0.99) and minimum
  trade size: Pro $100+ 5/25/10%; Gold $250+ 10/50/20%; Platinum $1,000+ 15/75/35%; Black $1,500+ 25/100/50%.
  Multiplier FanCash "expires seven (7) days after issuance". A base Fanatics ONE member gets no multiplier.
- Live check: the public page payload's feature flags (OFF, raw_W1/fanatics_legal_fcm-fee-disclosure.rsc.raw, read
  11-24) show `"fan-cash-earn-rate-multiplier":[]` and `"web-fan-cash-boost-config":{"enabled":false}`. No multiplier or
  FanCash boost was configured on web as of 11-24. A Thanksgiving multiplier could still appear in-app or be posted
  later, but none is posted.

### 1c. Refer-a-Friend (Base RAF) -- ENDED 11-15 (only posted window)
- OFF /legal/raf, "Effective as of November 9, 2026": "The Promotion Period begins at 12:00 a.m. Eastern Time ("ET") on
  November 9, 2026, and ends at 11:59 p.m. ET on November 15, 2026". The referrer got "$25 in FanCash for each Referee
  who completes a Qualifying Trade of $25 or more". FanCash "expire[s] seven (7) days after issuance". CA listed. The
  referee gets nothing from RAF but keeps the new-customer offer. "uses multiple or false accounts" -> withhold/reverse.
  Not live on 11-24 unless re-posted. Self-referral is not allowed.

### 1d. Fanatics ONE Status Match Challenge -- LIVE (applications to 12-07), not reachable at this user's scale
- OFF /legal/status-match, "Effective Date: September 7, 2026". Applications "beginning at 12:00 a.m. ... September 7,
  2026, and ending at 11:59 p.m. ET on December 7, 2026". CA listed. "Eligible Competitor Platforms are Kalshi and
  Robinhood". Annualized volume needed: "$350,000 ONEgold | $1.5 million ONEplatinum | $3 million ONEblack". Grants a
  90-day tier trial. Value: tier perks, plus multipliers only when a multiplier promo runs (none live). Not actionable at
  the fund's ~$100-500 scale.

### 1e. Other Fanatics legal promos -- all expired
sweepstakes (Jan 5-31 2026), socialgiveawayterms (Jun 8-Jul 19 2026), survey (Feb 5-Jun 30 2026), ticketgiveaway
(Jun 12-22 2026) (OFF, dates read from each page). The Thanksgiving slate preview (/learn/2026-nfl-week-12-thanksgiving-
slate-preview, dated 11-24, OFF via WebFetch) contains no offer.

---------------------------------------------------------------------------------------------------------------------
## 2. OG.com (NADEX d/b/a OG.com Markets / CDNA)

Background (FILING): CFTC staff asked OG to file its retail promotions starting October 2026, so OG's existing-user
programs now appear as 40.6 filings. FILING rules10092632811 says: "OG.com Markets has traditionally offered retail
promotions ... however CFTC Staff has requested that OG.com Markets file promotions going forward". All of these
programs are delivered in the app (Promo Center). OFF help.og.com/en/articles/15478533-promo-center says rewards and
boosts appear under "Promos" / the gift icon and are applied from the trade slip.

CA (OFF): "OG.com is currently available in all U.S. states" (help 12752911, per promo_verify_1124). Reward types by
jurisdiction (OFF help 15478570, updated 2026-09-22): Bonus "All jurisdictions except the State of New York"; Rebate
"The State of New York"; Boost "All jurisdictions". In CA, rewards arrive as Bonus or Boost.

Reward mechanics (OFF help 15478570 + FILING rules11122652237):
- Bonus: "Lowers the upfront cost of your trade ... Use a $20 bonus on a $100 trade. You only pay $80." The filing adds:
  "it offsets the trade amount and fees, any excess is forfeited, and it may not be cashed out". You own the full
  contracts, so the payout returns the full $1 per contract: **stake returned (SR-equivalent)**.
- Boost: "Amplifies your profits when your prediction is correct". Boosts don't cover fees ("only your profits are
  amplified"). It pays nothing if the trade loses. Whether boosted profit arrives as cash is NOT stated for profit
  boosts. The combo program (2c) says the boost is "credited automatically with the payout".
- Bonuses/rebates apply only to contracts with "a calculated probability below 80%". "One reward or boost per open
  order"; "Up to $1,000 USD of coupons can be applied per user per day" (OFF help 15478570).
- Fees (OFF help 12958828, dated 2026-09-15): taker 0.07*C*P*(1-P), maker 0.0175*C*P*(1-P) on sports/politics; fees
  waived on positions that settle in the money. No fee holiday, rebate or interest is mentioned.

### 2a. Weekly Trading Streak Program -- FILED 11-12, NOT YET LIVE (earliest ~11-27); recurring weekly
- FILING https://www.cftc.gov/filings/orgrules/rules11122652237.pdf (CFTC detail 67736, receipt 2026-11-12, status
  "10 Day Review" on the CFTC index read 11-24). OFF help pages do not mention it yet.
- Dates (FILING): "The Program began on September 6, 2026 and was paused effective September 30, 2026 ... The Program
  will resume upon the certification of the Program and will be available as a standing framework until October 31,
  2027". Ten business days after the 11-12 receipt, excluding Thanksgiving, is 11-27 at the earliest (my count, not
  stated). Weeks run Sunday 00:00 ET to Saturday 23:59 ET.
- Qualifier (FILING verbatim): "two single trades of $10 or more each in eligible contracts priced below $0.80, with both
  trades placed and settled before the week ends at Saturday 23:59 ET. Trades using other promotional boosts do not count
  toward the $10 minimum. Only bona fide settled single trades count".
- Reward ladder (FILING): "Week 1, a 50% profit boost up to $10; Week 2, a 50% profit boost up to $20; Week 3, a $10
  bonus; Week 4, a 25% profit boost up to $25; Week 5, a 50% profit boost up to $25; Week 6, a $10 bonus. A missed week
  resets the streak to Week 1, users may restart unlimited times, and one reward is available per user per week, subject
  to a maximum reward of $25 per user per week." Reward usable "only on contracts below 80% probability". Expiry:
  "Rewards expire 7 days from disbursement at 23:59 UTC." Bonus rewards (weeks 3 and 6) are SR-equivalent; boost rewards
  are profit-only.
- CA: segment = "All eligible users meeting disclosed jurisdiction and account requirements" (FILING). Bonus is
  available outside NY (OFF).
- Anti-abuse (FILING verbatim): "The Exchange will exclude wash trades, circular trading, self-dealing, linked-account
  activity, artificial volume, and other transactions intended to create qualifying activity without bona fide market
  risk." "may disqualify a user, forfeit a reward, limit an account, withhold, reverse, or recover a reward". **No
  explicit cross-venue hedge clause. But "without bona fide market risk" could be read to cover qualifiers hedged
  elsewhere. Treat a hedged qualifier as a forfeiture risk.**

### 2b. Reactivation and Retention Program ("Trade $x Get $y") -- LIVE framework since 10-26; targeted, in-app only
- FILING rules10092632811 (framework, 2026-10-09) + campaign certifications 10-30: Campaigns 1, 2, 4, 5 = "retail user
  retention incentive", reward "provided as a bonus" (rules10302640951, -40965, -41321, -41000); Campaign 3 =
  "reactivation", reward "provided as a trading coupon" (rules10302640974); Campaign 6 = reactivation, bonus
  (rules10302641323). All have status "Certified".
- Dates (FILING): "Going forward this Program start date is October 26, 2026 ... standing framework until October 31,
  2027". Each campaign: "start date is upon the certification of the Campaign".
- Mechanics (FILING): amounts are REDACTED ("Exhibit B ... REDACTED"; "public disclosure of these parameters could enable
  users to manipulate their trading activity or account status"). Framework: "a user must be included in the applicable
  user segment ... and must complete a minimum qualifying trade that settles within a certain time period after the
  campaign is communicated to the user. The qualifying trade must be in an eligible contract priced under a certain
  amount, typically $0.80." Reward: "$[y] bonus, or rebate for New York users; maximum $[ ] per user per campaign".
  Reward expires "after a set time". Terms are communicated to eligible users before the qualifying period. **Whether
  this user is in a segment shows up only in the app or by email/push.**
- Anti-abuse (FILING verbatim): "The Exchange will exclude wash trades, self-dealing, linked-account activity, artificial
  volume, and other transactions intended to create qualifying activity without bona fide market risk."

### 2c. Basketball Combo Program ("Combo Run Campaign") -- FILED 11-12, NOT YET LIVE (earliest ~11-27); weekly
- FILING https://www.cftc.gov/filings/orgrules/rules11122652222.pdf (detail 67733, "10 Day Review" on 11-24).
- Mechanics (FILING verbatim): "Each week, a participating user receives a 25% boost that may be applied to a trade in a
  featured basketball combo; if the combo trade settles in the user's favor, the user receives an additional amount
  equal to 25% of the combo payout, capped at $25 per week." Example: "a featured combo that settles in user's favor with
  an $80 payout earns a $20 boost, for a total of $100." NOTE: the boost is 25% of the PAYOUT (contracts x $1), not of
  the profit. Issued "Monday at 12:00am ET, expires the following Sunday at 11:59pm ET ... one per user per week".
  "Featured combos rotate throughout the week."
- Pays: "The reward will be provided as a boost credited with the payout." Nothing extra is paid on a loss.
- Stacking (FILING): "The boost may not be combined with any other boost, coupon, bonus, rebate, or promotional reward".
- Anti-abuse: same "without bona fide market risk" exclusion as 2b. No cross-venue clause.
- Starts "upon certification" (earliest ~11-27 by my count), so the first full week would be about Mon 11-30.

### 2d. Basketball leaderboards (Tipoff Top-100, filed 11-12; Top-50, filed 11-19) -- NOT YET LIVE; opt-in, cash
- FILING rules11122652227 (Tipoff) and rules11192656043 (Top 50, detail 68212). Both "10 Day Review" on 11-24. They will
  not run at the same time.
- Mechanics (FILING): rank by "bona fide opening and closing volume in covered contracts" (basketball, priced at $0.80 or
  lower; "opening a $100 contract counts as $100 of volume; closing it adds $100; and a contract held to settlement
  counts as $100"). Opt-in is required, and only volume after opt-in counts.
  - Top 50: pool "the greater of $5,000 or 1% of total platform-wide eligible basketball trading volume"; Rank 1 16% ...
    "Ranks 26-50: 0.8% each" (= $40 at the minimum pool).
  - Tipoff: "$5,000 initial reward pool and scales at 1%"; top 100 paid; "Ranks 76-100: 0.40%" (= $20 at the minimum pool).
- Pays: "Cash reward paid within 14 days after the campaign ends ... no reward expiry." Coupon-linked trades are excluded.
- Anti-abuse (FILING verbatim): "The Exchange will exclude trades linked to a coupon, wash trading, round-tripping,
  self-matching, self-dealing, linked-account or multiple-account activity, collusion, artificial volume, and other
  transactions intended to create qualifying volume without bona fide market risk."
- Volume needed to place is unknown (it depends on other traders). This matters only at scale.

### 2e. Refer-a-Friend -- LIVE (amount only)
- OFF help 13333913 ("Updated this week"): referee must "trade and settle at least $10" within 30 days of KYC; both get
  a "$10 bonus" (rebate in NY). Limit: "up to 40 rewards over their account's lifetime". Rewards "expire at 11:59 pm ET on
  the seventh day after issuance". FILING rules10092632815 (Refer a Friend Program, 10-09). Self-referral is banned.

### 2f. VIP / Macro volume rebates -- not applicable
- OFF og.com/vip: "Invite-only. Eligibility and benefits may vary." No amounts are given.
- FILING rules09162626774 (Macro Markets Volume Incentive, effective 09-30-2026 to 10-31-2027): weekly-volume fee
  rebates, but only on crypto/commodities/FX/equity-index markets. Sports are explicitly excluded: "Event Contracts and
  Derivatives that settle based on sports, politics, cultural events, economics or climate ... are excluded". The
  thresholds are redacted.
- Side note on a new-customer item only: FILING rules11122652232 (11-12) certifies "New User Rewards Program (Trade $25,
  Get $50)" as "an alternative to the existing $10/$10 New User Rewards campaign". This resolves the "up to $10 vs
  $25+$25" conflict in promo_verify_1124 section 6: both are filed, and the app shows which one is active.

---------------------------------------------------------------------------------------------------------------------
## 3. DraftKings Predictions (GUS III LLC d/b/a DraftKings Predictions FCM -> Railbird Exchange "DKeX")

- Official pages: predictions.draftkings.com -> 403; sportsbook.draftkings.com/golfterms. (the promo-terms page that names
  GUS III LLC, per SNIP) -> 403; railbirdexchange.com/notices -> 403; support.draftkings.com/dk/en-us -> JS-only
  ("Loading..."). **No official existing-customer promo text was readable without logging in.** Per the brief, I did
  not retry blocked pages.
- CFTC (FILING): DKeX consumer-facing filings are only the Market Maker Program (rules09042624071, rules09232629210,
  rules10192634689; MMs only) and the fee schedule (rules10192634652, effective ~11-02). REX filings dated 11-23
  (68408/68409) cover minimum deposits and dispute claims, not promos. DK consumer promos run at the FCM (GUS III) and
  are not filed with the exchange, so this route is a dead end.
- Fees (FILING rules10192634652): DKeX FCM taker fee $0.0100 per contract for $0.030-$0.979 contracts (lower at the
  extremes); maker fee "$0.0025 per contract" (non-combo); combo maker fee round-up(0.07*C*P*(1-P)); total "rounded up to
  the nearest $0.01"; "DKeX does not charge a Settlement Fee." Any extra DK FCM fee is unknown.
- Boosts / no-sweat / reloads for existing users: NONE found in any source (SEC/SNIP results cover only Sportsbook
  boosts, e.g. the sportsbook "NFL Boost Pack", not Predictions). Affiliates say existing-customer offers are
  "personalized" and appear in the in-app Promotions tab (SEC bleachernation/next.io). **Visible only in the app.**
- Refer-a-friend (SEC only, conflicting): $25 Predictions Dollars per referee, cap 5 (most affiliates; thespike "maximum
  reward of $125"); next.io (updated 11-20) says "$10 in Predictions Dollars", cap 5, friend must "deposit at least $5
  and place their first trade". UNVERIFIED. Self-referral is banned.
- Crowns / Dynasty Rewards on Predictions trades: SEC sources list Sportsbook/Casino/Pick6/DFS as the earning products,
  not Predictions. UNVERIFIED, probably none.
- Stake treatment of Predictions Dollars: next.io (SEC) says winnings "return only profit". That contradicts the user's
  confirmed DK experience: $49.92 of Predictions Dollars paid $52.00 (inbox 11-24 03:06Z), i.e. stake returned. Trust
  the observed payout.

---------------------------------------------------------------------------------------------------------------------
## 4. Thanksgiving week (11-26) / weekend (11-29) specific
- Fanatics: none posted. The multiplier list is empty in the live web config (1b). The Thanksgiving preview page has no
  offer.
- OG.com: none posted. The weekly streak and combo programs can start no earlier than ~11-27 (2a, 2c). Basketball only
  for the combo program; the streak covers any market priced < $0.80.
- DraftKings Predictions: unknown (in-app only; official pages blocked).

## 5. Summary table
| venue | promo | live? | CA? | payout type | stake returned? | hedge/offset clause | source |
|---|---|---|---|---|---|---|---|
| Fanatics | Always-On FanCash earn rate (0.15% at >=0.43 ... 10% at 0.01) | yes 8/26-12/31, but literal terms require an account verified by 8/26 | yes, explicit | FanCash (365-day expiry) | when used: payouts to cash (SR implied, DK-observed) | no hedge clause; bans "entering and immediately exiting positions, wash trading, or coordinated trading" (1a) | OFF /legal/fancash |
| Fanatics | FanCash Multiplier (NFL games) | no (only 9/17-9/21 posted; web flag empty) | yes | FanCash, 7d | as above | wash/self-match only (1b) | OFF /legal/fancash + page config |
| Fanatics | Refer-a-Friend $25 FanCash | no (11/9-11/15 only) | yes | FanCash, 7d | as above | "multiple or false accounts" (1c) | OFF /legal/raf |
| Fanatics | ONE Status Match | yes to 12/7 | yes | tier trial | n/a | n/a ($350k Kalshi volume min) | OFF /legal/status-match |
| OG.com | Weekly Trading Streak (2 x $10 trades < $0.80; boosts/bonus up to $25/wk) | filed 11-12, earliest ~11-27 | yes ("all U.S. states"; Bonus outside NY) | 50%/25% profit boosts; $10 bonus wks 3 and 6 | bonus yes (offsets cost); boost profit-only | "without bona fide market risk" exclusion (2a) | FILING rules11122652237 |
| OG.com | Reactivation & Retention "Trade $x Get $y" (6 campaigns) | yes since 10-26 for targeted users; amounts redacted | yes (bonus outside NY) | bonus (Campaign 3: trading coupon) | yes (bonus) | same exclusion (2b) | FILING rules10092632811, rules103026409xx/41xxx |
| OG.com | Basketball Combo Run (25% of combo payout, $25/wk) | filed 11-12, earliest ~11-27 | yes | boost paid with payout | stake paid normally; +25% of payout | same exclusion; no stacking (2c) | FILING rules11122652222 |
| OG.com | Basketball leaderboards (Top 50 / Tipoff Top 100) | filed 11-12 and 11-19, not yet live | yes unless restricted | cash, no expiry | n/a | wash/round-trip/multi-account (2d) | FILING rules11192656043, rules11122652227 |
| OG.com | Refer-a-Friend $10/$10 | yes | yes | bonus, 7d | yes | none found | OFF help 13333913 |
| DK Predictions | Existing-customer boosts/credits | unknown (in-app only) | unknown | unknown | observed SR on PD credits | unknown | official pages 403 |
| DK Predictions | Refer-a-Friend | SEC only | unknown | Predictions Dollars ($10 or $25, cap 5) | SR observed | unknown | SEC next.io 11-20 / others |
