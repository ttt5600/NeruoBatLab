# Fund economics: can the edges pay for ~$200-400/month of AI subscriptions? (2026-10-27)

Repo evidence only. Nothing deployed, no orders. Scripts (scratch, not committed):
`/Users/jonathanwang/.claude/jobs/588c8b26/tmp/{wx_econ.py, wx_sim2.py, l39.py}`. They re-derive the E4 sealed
maker result from `data/kalshi_temp_maker_{picks,trades}_sealed.parquet` (reproduced: fill 80.2%, maker ROI +2.22%
per filled $, loss rate 3.07%, exactly the registry numbers) and use live paper depth from
`~/.claude/quant-kalshi/data/kalshi_paper.jsonl` (131 picks, 12 nights).

## Verdict (blunt)

1. **No proven edge can pay $200/month at a bankroll under ~$10k, and at the conservative planning ROI not under
   ~$65k.** The weather edge is real but small (about 1-2 cents per dollar) and capacity-limited.
2. **Maker weather (the best per-$ edge) tops out at roughly $150/month at ANY bankroll** under the planning ROI, because
   size makes adverse selection worse: losers always fill your whole order, winners only fill up to what trades through
   your bid. Above about $75 per pick, extra size loses money.
3. **Taker weather scales further (books ~$195 deep within 1c at the median), but earns less per $.** At the planning
   0.6% it reaches $200/month at about a $66k bankroll, with a 39% chance of a losing month.
4. **L32 NBA totals and L39 monthly rain are not edges yet.** L32 on Kalshi is about break-even as a taker (the 1.75c fee
   is about 3.4% of a 50c stake, which eats the shrunk edge). L39 is capped at single-digit dollars a month by thin books.
5. **Learning time does not depend on bankroll.** It depends on the number of bets. L32 needs about 1,500 bets
   (2.6 seasons) to confirm even a true +6.4%, and about 7,000 bets (12 seasons) for a true +3%.
6. **Today's live caps ($30/night) limit weather to about $900/month turnover, about $8-14/month expected.**

## Headline table: expected monthly profit +- SD, P(losing month)

Planning ROI (shrunk): weather maker +1.5%/filled $, weather taker +0.6%, L32 +2% (maker, IF fee-free; unverified),
L39 +1%. Weather rows come from a simulation that resamples sealed nights (2026-03..10, both cohorts live). L32 and L39
rows are analytic (formulas below). L32 counts in-season months only (Oct-Apr). Annual average = about 0.48x.

| Bankroll | Weather MAKER (cap $75/pick) | Weather TAKER (cap $2k/pick, 0.5x depth) | L32 NBA totals (1% B, cap $250 ASSUMED) | L39 rain (3% B, cap $30) |
|---|---|---|---|---|
| $50 | $1.7 +- 1.1, P(loss) 8% | $0.9 +- 1.1, 21% | $1 +- 5, 42% | $0.4 +- 1.1, ~37% |
| $500 | $17 +- 11, 8% | $9 +- 11, 22% | $10 +- 50, 42% | $3.8 +- 11, ~37% |
| $2,000 | $57 +- 46, 11% | $29 +- 43, 24% | $40 +- 200, 42% | $7.5 +- 23, ~37% |
| $10,000 | $122 +- 207, 27% | $88 +- 196, 31% | $200 +- 1,000, 42% | $7.5 +- 23, ~37% |
| $50,000 | $147 +- 263, 28% | $159 +- 732, 39% | $500 +- 2,500, 42% | $7.5 +- 23, ~37% |

Pessimistic weather (maker +0.5%): $5 +- 11 at $500, $14 +- 46 at $2k, and **negative at $10k+** (-$39/month), because
the adverse-selection cost of size exceeds the edge. "Sealed as-is" maker (+2.2%): $25 at $500, $88 at $2k, and about
$310/month at most.

## Break-even bankroll

| Monthly cost | Weather maker | Weather taker @0.6% plan | Weather taker @1.07% (sealed as-is) | L32 (if fee-free maker, +2%, capacity >= $100-200/game) | L39 |
|---|---|---|---|---|---|
| $200 | **never** (ceiling ~$144; ~$310 at sealed 2.2%) | ~$66k (P loss-month 39%) | ~$10.5k (17%) | ~$10k, in-season only; annual average ~$95/month | never |
| $400 | never | ~$98k (32%) | ~$33k (22%) | ~$20k, in-season only | never |

These are pre-tax trading profits. The subscription is paid from after-tax money, so the real hurdle is higher.
Running maker and taker together (rest $75 at the mid, take more at the ask) would add at most roughly the two columns.
The gain over taker alone is about $100-150/month.

## Per-edge inputs

### Weather E1+E2 (passed sealed), executed as E4 maker or as taker
- **(a) Bets/month:** sealed 2026-03..10, all picks 453/30 days (166-615 filled per month); about 8 filled per night.
  Winter is uncertain: sealed E1 had only 23 picks/month in Nov-Dec 2025, but 99-155 in Nov-Dec 2024.
- **(b) ROI per $:**
  - Maker, sealed: +2.22%. The night-cluster bootstrap 95% CI is [+1.57%, +2.84%]. The within-night design effect
    is 1.00, so picks on the same night are not measurably correlated.
  - Taker, sealed: +1.07%.
  - Live-mix shift: 65% of picks are now at 99c, up from 25-41% in 2025. By NO-ask bucket, the sealed maker ROI is
    99c +1.7%, 98c +2.3%, 96-97c +2.7%, 91-95c +1.1%. Taker at 99c is only +0.65%.
  - Reweighted to the live mix: maker +1.95%, taker +0.86%.
  - **Planning: maker 1.5%, taker 0.6%**, about 2/3 of sealed. The reasons are public forecast bots (registry
    warning), the price mix drifting to 99c, July 2026 at -0.9%, and live maker not yet run.
- **(c) Capacity:**
  - Taker: the executor takes 0.5x the depth within 1c of the ask. Live median depth is 199 contracts, about $195
    (IQR 52-761).
  - Maker: for WINNING picks, contracts printed at or below our bid after 22:00 are p25 148, median 550, p75 1,840.
    The fill model assumes we get 25% of that (phi). Our place in the queue is unknown.
  - **Every loser fills the whole order** (observed: the unfilled loss rate was 0%).
  - So profit = sum over winners of min(o, phi\*V\*q)\*r_win - sum over losers of o. It peaks at a per-pick cap of
    about $75 (phi .25), $30 (phi .1) or $150 (phi .5). The maximum is $61 / $144 / $288 per month at plan.
- **Monthly SD:** taken from resampling nights. Roughly SD = sigma\*sqrt(N)\*s, with sigma = 0.195 per filled $
  (maker) or 0.17 (taker).

### L32 NBA opening totals (candidate; sealed FAIL p .054)
- **(a) Bets/month:** 575 bets in 2025-26, about 100 per in-season month. None in Jul-Sep.
- **(b) ROI:**
  - Evidence at the no-vig opener: dev +8.7% (1,190 bets), sealed +6.4% [-1.6, +14.4] (575 bets). The registry's own
    prior is a true edge of about +3-4%.
  - Graded vs the close it wins 48.2%. The edge is the stale opener.
  - **Kalshi taker:** cost = 0.51 ask + 0.0175 fee = 0.5275, against a shrunk win rate of about 52.5-53%. That gives
    ROI = p/c - 1 ≈ -0.5% to +0.5%. **Planning 0%.**
  - **Maker:** about +2% only IF KXNBATOTAL charges no maker fee and Kalshi's line is still near the opener. Both are
    unverified. Adverse selection was 1.5-2 points of loss rate in weather.
  - sigma ≈ 1.0 per $.
- **(c) Capacity:** **not measured.** The logger records `kal_ask` but no depth. The $250/game in the table is a
  placeholder assumption, not data. NBA orders also need the separate Kalshi shard funded; until then they are
  phone-only.
- **(d) Losing-month probability** = Phi(-mu\*sqrt(N)/sigma) = Phi(-0.02\*10/1) = 42% at any bankroll below the cap.

### L39 monthly rain/snow unreachable-NO (candidate; dev only, p .016 vs bar .0083)
- **(a) Bets/month:** about 20-25 (dev 9-27 per month; 22 qualify in the 10-27 preview). All of them fall in the last
  7 days of the month.
- **(b) ROI:**
  - Dev maker +3.6% on 146 bets with 1 loss. The break-even loss rate is 4.1%, against about 0.7% observed.
  - **Planning +1%**: it was chosen among 6 trials, and one loss costs about 40 wins.
  - sigma: 0.09 observed in dev, but 0.15 is used, which is what a true 2% loss rate would give.
- **(c) Capacity:** books are thin. Live best-bid sizes were 2-205 contracts, and 4 of 6 markets had volume under 310.
  A cap of about $30/pick is generous, so the maximum turnover is about $750/month and **the maximum profit is about
  $7.5/month**. The capacity cap binds, not the edge.

## Formulas

- Per-$ ROI of a NO bought at all-in cost c, with true P(NO) = p: r = p/c - 1.
  - Weather model: p = 1 - theta\*yes_bid, theta ≈ 0.69 (dev) / 0.665 (sealed).
  - Taker c = no_ask + 0.07\*a\*(1-a). Maker c = q, the resting bid (fee 0 on `quadratic` series).
- Per-$ SD of a binary NO bet with loss rate l: sigma ≈ sqrt(l\*(1-l))/c.
  - At l = 3% and c = 0.95 this gives 0.18. The empirical values are 0.195 (maker) and 0.17 (taker).
- Monthly profit at stake s per bet, N bets: mean = N\*s\*mu, SD = sigma\*s\*sqrt(N\*deff) (deff = 1.00 measured for
  weather). P(losing month) ≈ Phi(-mu\*sqrt(N)/sigma). **It does not depend on s**, so a bigger bankroll does not make
  losing months rarer. Only more bets or a bigger edge does.
- Weather sizing: o = min(0.10\*B/n_night, 0.044\*B, C). The 0.10 is the executor's NIGHT_FRAC and 0.044 is the
  quarter-Kelly per-pick fraction from `kalshi_sizing.md`. Monthly turnover is about 3B until the caps C and depth bind.
- Bets needed to show mu > 0 (one-sided alpha .05, power .8): **n = ((1.645 + 0.842)\*sigma/mu)^2**.

## How long until we KNOW (same at every bankroll)

| Question | mu, sigma | Bets needed | Time |
|---|---|---|---|
| Weather maker live is really > 0 | 1.5%, 0.195 | ~1,050 filled picks | **~3 months** (if every pick is taken) |
| ...if the true edge is only 0.5% | 0.5%, 0.195 | ~9,400 | ~26 months |
| Weather taker > 0 | 0.6%, 0.17 | ~5,000 | ~11 months |
| L32 on Kalshi, true +6.4% | 6.4%, 1.0 | ~1,500 | ~15 in-season months ≈ 2.6 seasons |
| L32, true +3% | 3%, 1.0 | ~6,900 | ~69 in-season months ≈ 12 seasons |
| L32, true +1% | 1%, 1.0 | ~62,000 | never, practically |
| L39 rain, true +1% | 1%, 0.09-0.15 | ~500-1,400 | ~25-70 months |
| L39 rain, true +3.6% (dev) | 3.6%, 0.09-0.15 | ~40-110 | ~2-5 months |

Bankroll changes what learning COSTS, not how long it takes. If L32's true edge is 0, a taker pays about 3.4% of
turnover in fees: about $34/month at $10/bet x 100 bets, and $340/month at $100/bet. CLV (L32 sealed +1.2 pts, p .005)
resolves in a few hundred bets. But it shows that the model leads the opener, not that Kalshi pays.

## What this implies

- At realistic bankrolls (<= $10k), the proven edge yields about **$15-120/month expected at plan** (maker). That is
  below a $200 subscription, with a 10-27% chance of a losing month.
- To cover $200/month from proven edges: about $10k if the sealed taker ROI holds, or about $66k at the planning ROI.
  It is a coin flip in roughly a third of months either way. For $400/month: $33k-98k.
- The cheapest lever is the one already passed: **turn on maker** and raise the per-pick cap toward about $75. In the
  same per-pick bucket, maker is about 2x taker per $. Above about $75 per pick, adverse selection eats the gain.
- L32 can only matter if (1) KXNBATOTAL maker fees are 0, (2) Kalshi's first ladder sits near the sportsbook opener, and
  (3) depth is at least $100-200 per game. All three are unmeasured. Log depth and first-ladder time in
  `nba_open_totals.py` before putting money behind it.

## Caveats
- phi (our share of through-volume) is a guess. Results are shown at .1, .25 and .5.
- Taker depth comes from 12 nights of live paper (heavy tail, max 10k contracts).
- The sealed per-pick outcomes are reused with a uniform per-$ haircut, which slightly flatters the 99c picks.
- Bankroll is held fixed within a month (no compounding). OpenAI agent costs are unknown. Substitute any cost K and
  read the break-even off the table.
