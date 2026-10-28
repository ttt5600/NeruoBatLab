# Kalshi Liquidity Incentive Program (LIP): is it income for us? (2026-10-27/28, read-only research)

## VERDICT (read this first)
**Yes, it's worth a 10-day paper test, but only as a paper-quote logger. We place no orders.** Weather is in the program.
Every daily-temp bracket (KXHIGH*/KXLOWT*, 48 series) carries **$20 per ~1.6-day period**. Rain markets carry $20-$170.
Our numbers come from one book snapshot at 03:44Z on 10-28. At that moment, **75% of daily-temp brackets were paying
nobody**, because the YES side of the book was short of the 1,000-contract Target Size. Kalshi only pays when both sides
reach Target. A small maker can complete that side with cheap 1-2c YES bids and collect most of the YES-half of the pool.
The catch is that these bids get filled: the tape shows a 1c YES bid on tail brackets fills on 97% of nights. The second
use is our E1/E4 NO bids. LIP pays them while they rest, even when unfilled, but only at about 100 contracts per bracket,
because the payout minimum is $1 per program.

Headline ranges (gross, one snapshot, not yet measured over time):
- Joining the best bid with 100 contracts on a valid daily-temp bracket pays a median **$1.4-2.3 per period per side**.
  That is about **1.8-5%/day on the capital tied up**, before fills.
- If every valid bracket is quoted on its cheaper side: **~$157/day gross on ~$3.5k capital** (152 brackets). The
  $3.5k is above what the user has. For a **$500 slice**, scale it to roughly $20/day gross, before adverse selection.
- Two costs are unmeasured, so **net is unknown**:
  - **Fill losses.** The tape gives -$0.5 to -$1.3 per bracket-night for a 1,000-lot at 1c. It is much worse at 2c
    (-$3 to -$7).
  - **Time-of-day validity.** One snapshot can't show it.

**Runway is short.** The program ends **2027-01-01**, so there are about 9 weeks left. Kalshi can end it at any time.
After the 10-28 amendment, periods are 1-7 days and payment is due within 1 day. So a 10-day paper test followed by
about 6 weeks live is the whole opportunity. Expected value: tens of dollars per day at best, at our size.

## 1. The exact rules (primary sources)
Sources:
- CFTC self-certification "Amendment to August 2025 Liquidity Incentive Program", filed 2026-10-14, effective
  2026-10-28: cftc.gov/filings/orgrules/rules10142633813.pdf (Appendix B, clean terms).
- help.kalshi.com/en/articles/13823851 and help.kalshi.com/en/articles/16076644.

- **Sampling.** One snapshot of the book every second, at a random uniform moment within the second.
- **Snapshot exclusions.** A snapshot is **excluded** if the market is closed, or if **either side** has less than
  **Target Size** contracts resting. The "two-sided liquidity" requirement is about the whole book, not about you. You
  do NOT need to quote both sides yourself.
- **Per side.** The filing treats a YES ask as a NO bid. The algorithm on each side:
  - Walk down from the best bid, adding whole price levels, until cumulative size >= Target. Those are the
    **qualifying bids**. Bids below that level score nothing.
  - **Reference price** = the first level where cumulative size >= Target/5.
  - Score(bid) = size × 0.5^max(ref − price in ticks, 0). The **Discount Factor is fixed at 0.5** by the amendment.
    Bids at or above the reference get full weight.
  - Normalise each side so its scores sum to 1. Your snapshot score = your YES share + your NO share, so the total
    across all users is 2.
- **Payout.** payout = (your total score / everyone's total score) × period reward × (non-excluded snapshots / all
  snapshots).
  - Rounded down to the cent. **Programs worth less than $1.00 to you pay nothing.**
  - Pro rata, and it cannot run out early.
  - **Excluded snapshots are lost money.** Nobody gets them.
- **Variables after the amendment.**
  - Target Size: between 100 and 10,000 contracts.
  - Reward: $10 to $10,000 per business day per market. Confidential Appendix D sets a lower cap for less volatile
    markets.
  - Time Period: 1 to 7 days.
  - Each contract is posted at least 24 h ahead.
  - **Paid no later than 1 day after the period ends.**
  - Credited as USD to the account.
- **Eligibility.**
  - Kalshi members are eligible, except: Kalshi affiliates; IBs and FCMs and their customers; anyone under a Kalshi
    market-maker agreement in that market; non-US users.
  - An SSN is needed for reward credits above the IRS threshold.
- **Integrity.** Rule 5.17 still applies (spoofing, layering, wash trading). Orders under inquiry earn nothing.
  - Our orders would be genuine, willing-to-fill bids, so this is not a blocker.
  - It is a reason **not to cancel and repost to game snapshots**.
- **Interaction with fees.** Weather series are `fee_type: quadratic`, so makers pay 0 (notes/kalshi_temp_maker.md).
  The net-negative-fee clawback in §6 applies to per-trade incentives, not to LIP.
- **Context.** Kalshi ended the separate *Volume* Incentive Program (no earlier than Oct 13) amid wash-trading
  allegations about perps (The Block, 2026-09-30). LIP was amended and continues.

## 2. What is incentivised right now (API, pulled 2026-10-28 ~03:40Z)
Public, keyless endpoint: `GET https://api.elections.kalshi.com/trade-api/v2/incentive_programs?status=active&type=liquidity`
(docs.kalshi.com/api-reference/incentive-programs). It returns, per market:
- `period_reward`, in **centi-cents** (÷10,000 = $)
- `target_size_fp`
- `discount_factor_bps`
- start and end dates
- `paid_out`

It also returns `max_reward_per_account`, which is null on every active program.

- There are 9,972 programs: 7,601 active liquidity, 2,369 upcoming, and 2 volume.
  - 6,672 active programs last at least 1 day. They pay about **$138k/day** combined.
  - 929 are 15-minute crypto/FX programs ($50 each). Those predate the amendment and should disappear under the
    1-day minimum.
- Target is 1,000 on 99%. Discount factor is 0.5 on all but 3.
- Biggest series by $/day:

| Series | Markets | $/day | Per market |
|---|---|---|---|
| KXRT (Rotten Tomatoes) | 135 | 13,500 | $100/day |
| KXTRUMPAPPROVE | 9 | 5,400 | $600 |
| KXAAAGASD* (state gas prices) | ~17-27 each | ~2.5-5k per state | $150-190 |
| KXGGWINNER | 263 | 5,000 | $19 |
| KXYTVIEWSW | 248 | 3,100 | $13 |

- **Weather: 840 markets, ~$12.5k/day.**
  - **KXHIGH* (24 series, 288 brackets) and KXLOWT* (24 series, 288 brackets).** Each bracket pays **$20 per period**.
    A period runs from 14:00Z on D-1 to ~05:00-06:00Z on D+1 (1.6-1.75 days, about $12/day). Target 1,000, DF 0.5,
    `incentive_description: series_lip`, so the program recurs daily.
  - **KXRAIN* (246 markets).** Daily rain pays $20 per ~1.9 days. New monthly-rain events pay $100-170 per 6 days.
    These are the L39 monthly ladders.
- Every E1/E2 city appears, including NY, CHI, MIA, AUS, DEN, LAX, PHIL and the KXHIGHT* expansion cities.

## 3. What a small maker could earn (one snapshot, filed formula applied exactly)
Scripts are in scratch, `/Users/jonathanwang/.claude/jobs/588c8b26/tmp/lip/`:
- `books.py` takes the snapshot: 1,200 live markets, all 840 weather plus 360 random others.
- `yield.py`, `complete.py` and `yes1c.py` do the scoring.

Formula for your $ from one side over a period:

`$ = R × s × v / 2`

- R = period reward.
- s = your share of that side's weighted qualifying size, after inserting your order. It is recomputed with your order
  included, because your order can move the reference price and truncate the qualifying set.
- v = the fraction of valid snapshots. Assumption: v = 1 if the book is valid now and 0 if not. **This is the weakest
  assumption.** One snapshot can't give v.

Fills are ignored here and handled in §4.

Daily-temp brackets, valid now. KXHIGH 76/288, KXLOWT 76/306. These are medians:

| Order (join best bid, N=0) | $/period | Capital | Gross %/day on capital | P(clear $1 min) |
|---|---|---|---|---|
| YES ×100 | $1.4-2.3 | $26-28 | 2.6-5.3% | 61-72% |
| NO ×100 | $1.7-2.2 | $60-70 | 1.8-2.2% | 79-95% |
| NO ×100 on E1-type tail (NO bid >= 90c, n=24) | $1.51 (p25 1.14, p75 2.16) | ~$94 | 0.9% | most |
| YES or NO ×500 | $7.8-9.7 | $130-350 | 1.4-3.9% | ~100% |
| ×10, our current E1 size | $0.12-0.19 | — | — | **0-3%** (min-payout kills it) |

Other context from the same snapshot:
- **Competing depth is thin.** Weighted qualifying size W has a median of only ~520 (KXHIGH) or ~1,130 (KXLOWT)
  contracts per side. So 100 contracts buys a 10-20% side share.
- Rain markets are valid 79% of the time, with W ≈ 2,500-3,300. Other categories are 89% valid, with W ≈ 4,400-5,000.
  Our share there is smaller, but pools are bigger: KXRT is $100/day, KXTRUMPAPPROVE is $600/day.

**The "completer" observation.**
- 119 of 594 daily-temp brackets were invalid **only** because YES bids totalled less than 1,000. The median YES depth
  was 71.
- Adding the shortfall at the best YES bid (usually **1c**, about $10 capital) would make every snapshot valid. Our
  share of the YES side would then be at least shortfall/1,000, which means **up to ~$10 of the $20 period**.
- Taken literally, that is an implausibly good ~100% return on capital per period. It needs the §4 fill check, and a
  check that the YES side is not already filled during US daytime.

## 4. Adverse selection: what resting costs
- **NO side (the E1/E4 picks).** Measured, sealed 2025-07..2026-10, 4,308 picks:
  - Every losing pick fills the whole order (unfilled loss rate 0%). Filled loss rate is 3.07% vs 2.46% overall.
  - Even so, maker ROI was **+2.22% per filled $** (E4, notes/EDGES_AND_AGENTS.md).
  - So on E1 picks, resting NO is already positive-EV, and **LIP adds to it**. LIP pays while unfilled, so it also pays
    on winners that never fill, which fund_economics.md calls the capacity limit.
  - At ×100 and NO 0.94, the ~$1.5/period is **~1.6% of stake per pick**, on top of +2.2%. But size raises loss
    exposure: a loser at ×100 costs ~$94.
  - With the planning ROI of 1.5% and a 3% loss rate, the size decision is a bankroll question. notes/risk.md caps
    still bind.
- **Non-pick brackets.** We don't know our own adverse selection there. The aggregate prior is Becker's 72M trades:
  weather makers earn +1.29% pre-fee (notes/social_edges_2026-10-22.md, L24). That means roughly break-even before
  LIP. Not measured for our own quotes.
- **YES 1-2c completer bids on tail brackets.** Tape check (`yes1c.py`): public `/historical/trades` on the 4,308
  sealed E1 tickers, from 22:00 to close. A fill is a taker buying NO at >= 1−y. phi is our share of the volume that
  trades through our price.

| Bid | Size | phi | Nights with a fill | Mean filled | Mean P&L per bracket-night | Capital |
|---|---|---|---|---|---|---|
| 1c | 1,000 | 1.0 | 97% | 521 | **−$1.34** | $10 |
| 1c | 1,000 | 0.25 | 97% | 173 | −$0.48 | $10 |
| 2c | 1,000 | 0.25 | 98% | 251 | −$2.97 | $20 |
| 2c | 1,000 | 1.0 | 98% | 627 | −$7.43 | $20 |

  - Read: **1c is cheap insurance against an expensive problem.** E1 takers buying NO at 99c fill our 1c YES. Each
    fill costs 1c, and once filled the side may fall below Target again, so the order has to be **replenished**. A
    replenish-to-1,000 policy costs roughly the filled volume × 1c.
  - **2c is clearly worse**, because YES at 2c is overpriced on these brackets: YES hits ~0.69× the implied rate,
    from notes/kalshi_sizing.md.
  - Caveat: this window only covers 22:00 to close. The LIP period starts at 14:00Z on D-1, so the D-1 daytime fills
    are not in this tape.

## 5. Can we read our own accruals? (point 4)
**No API endpoint for accruals exists today.**
- `GET /incentive_programs` is public and per-market only. It has no per-user field.
- docs.kalshi.com lists `GET /portfolio/credit_history`. It has a free-form `type`, and a `reason` that can include a
  ticker. But **on 2026-10-28 it returns HTTP 404 on both external-api.kalshi.com and api.elections.kalshi.com, signed
  with the user's key.** For comparison, `/portfolio/balance` returns 401 unsigned, so that route exists. So
  credit_history is documented but not deployed.
- No `/portfolio/{credits,rewards,incentives,...}` route exists either (all 404).
- The only per-user views are in the web UI:
  - A popover with an "estimate as of" figure.
  - Blue and gray qualification dots on your resting orders.
  - Paid credits under Rewards and under Account → Activity → Credits.
- How we would measure live: compute our own score with the filed formula from our orders plus public book snapshots,
  then reconcile against balance jumps not explained by fills. (`/portfolio/balance` and `/portfolio/fills` both work.)
  Re-probe credit_history weekly.
- Payout timing in practice: 10,000 programs are `closed` but not `paid_out`, with end dates going back to 10-21.
  That's 6+ days unpaid under the pre-amendment terms. The amendment promises payment within 1 day.

## 6. Pre-registered paper-quote plan (read-only; 10 days, 2026-10-29 00:00Z → 2026-11-08 00:00Z)
**Logger.** Poll `/incentive_programs?status=active&type=liquidity` hourly, and the books of every live KXHIGH*/KXLOWT*
LIP market **every 5 min** (~600 markets, ~2 req/s, public).
- At each poll, insert the hypothetical orders below into the real book and score them with the filed algorithm.
- Fills come from the public trade tape: prints at or through our price after the order's start. The queue share phi
  is set as below.
- Replenish rule: after a fill, the order returns to full size at the next poll. The fill loss is booked.
- Settle each bracket on the market result. Our orders never touch the book.

**Policies.** The size and price of each is frozen here.
- **P1 "E1 piggyback".** On every E1/E2 pick at decision time (22:00 local), rest NO ×100 at the E4 M3 price (mid
  floored, <= ask−1c) until close.
  - P&L = fills (phi 0.25) + LIP.
  - Baseline = the same pick at ×100 with no LIP.
- **P2 "YES completer at 1c".** On every daily-temp bracket whose best YES bid is <= 2c, rest YES ×1,000 at **1c**
  from period start to close. Replenish after fills.
  - P&L = −0.01 × filled on a NO outcome, or +0.99 × filled on a YES outcome, + LIP.
  - phi 1.0 (conservative: we absorb all through-volume).
- **P3 "join cheaper side ×100".** On every valid daily-temp bracket, rest ×100 at the best bid on the cheaper side.
  Fills use phi 0.25.

**Primary metric.** Net $ per bracket-period per policy, and net $ per $-day of capital.
- LIP $ = R × our summed side share / (2 × N polls) × (valid polls / N polls). This is the payout formula with 5-min
  polls standing in for 1-s snapshots.
- Programs worth less than $1 count as $0.

**Success**, judged per policy:
- 10-day net > 0, with the **day-clustered bootstrap 95% lower bound > 0**.
- Median per-program LIP >= $1.
- For P2: valid fraction with us minus without us >= 0.3, across all hours, not just night.
- Bar is Bonferroni over 3 policies: one-sided p < 0.0167.

**Kill.**
- Net <= 0, or the lower bound <= 0.
- Or competing W grows so our ×100 share is below 5%.
- Or Kalshi changes, pauses or removes the weather LIP. Check the API daily.

**If one passes.** Ask the user before any live test. Then run the smallest live version (P1 at ×20-50, or P2 at ×300)
for 1 week and reconcile against the UI estimate and paid credits.

**Caveats the paper test cannot fix:**
- Real snapshots are per second, and orders that rest only between our 5-min polls are invisible to us.
- Our real order would change others' quotes.
- Queue position is unknown.
- Confidential Appendix D caps and the market-selection method are not public. Kalshi chooses which markets get
  rewards.
