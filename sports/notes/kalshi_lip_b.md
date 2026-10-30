# L40b: Kalshi LIP on NON-weather markets: paper test PRE-REGISTRATION (filed 2026-10-29 ~23:10Z)

Written before any L40b snapshot is logged and before the sample is drawn. Paper only: no orders, public GET
endpoints only, no API key. Code: `scripts/lip_paper_b.py` (reuses `scripts/lip_paper.py` scoring, book parsing, tape
and market pulls). Data: `data/lip_b_{sample.json,state.json,ticks.jsonl,events.jsonl}`.

## Why
L40 (notes/kalshi_lip.md section 6) is dead by its own kill rule: Kalshi ended every weather LIP program on
2026-10-29 08:00Z (`program_change` events with n_active=0 in data/lip_paper_events.jsonl). Kalshi still lists
thousands of active liquidity programs in other series (probe 2026-10-29 ~23:00Z: 5,872 active + 1,093 upcoming,
569 series; 6,839 at target 1,000 and 126 at 2,500; DF 0.5 on all but 6 at 0.4; period lengths 1 to 14 days).

## Question
Does resting liquidity on NON-weather LIP markets earn LIP payouts that exceed the fill (adverse-selection) losses,
net, per program?

## Universe and sample (frozen once, by `lip_paper_b.py sample`, before T0)
- Source: `GET /incentive_programs?status={active,upcoming}&type=liquidity`, all pages.
- Eligible program: overlaps the window (start < T1 and end > T0 + 1 h); period length >= 0.9 day (drops pre-amendment
  remnants such as sub-day KXMLBSERIESHR); its series' category is not "Climate and Weather" (from `GET /series`);
  its market is open at draw time (`status` active/initialized) and `close_time` > T0.
- Unit = market ticker (a ticker can carry several programs over the window via `series_lip` renewals; every
  program on a sampled ticker that overlaps the window is scored, including renewals that appear later).
- Two strata by the market's `close_time` read at draw time:
  - **PRIMARY** = close_time <= T0 + 30 days (2026-11-29 00:00Z). Up to 240 tickers. **Only these are scored
    against the bars.**
  - **SECONDARY (long-dated)** = close_time > T0 + 30 days. Up to 60 tickers. Logged identically; LIP is computed;
    fills are marked to the book mid at report time (MTM). Reported descriptively, never used for the bars. Choice:
    score as pending/MTM rather than drop, so the long-dated adverse selection is still seen.
- Draw: `random.Random(4040)` shuffle of eligible tickers (sorted first, so the draw is reproducible); walk the
  shuffle and take a ticker unless its series already holds floor(0.20 x stratum quota) slots. Outcome-free: the
  draw sees only program terms, series category and close_time; no prices, books or results.
- The sample file records the seed, draw time, quotas, eligible counts per stratum and the per-series mix. It is
  never redrawn; `sample` refuses to overwrite it.
- Rate budget: one tick = ceil(300/100) = 3 batched orderbook calls + global trade tape pages (13 pages for 60 s
  measured at 23:05Z 10-29) + occasional markets calls; paced at <= 8 req/s (REQ_GAP 0.125 s, lip_paper.get) under
  the Basic tier's 20 reads/s. Program list refreshed hourly (2-3 calls).

## Window
T0 = 2026-10-30 00:00Z (the next 00:00Z after deployment), T1 = 2026-11-09 00:00Z (10 days). A hypothetical maker
quotes only inside [T0, T1). Ticks before T0 are a shakedown and not scored.

## Policies (sizes scale with each program's own Target Size T; orders per (policy, ticker))
- **P2 "YES completer at 1c"**: YES x T at 1c on every sampled ticker whose best YES bid is <= 2c. Judged once, at
  the first poll in the window where a sampled program is live and the book has at least one YES and one NO bid.
  Replenish to full size at the next poll after a fill. phi 1.0.
- **P3 "join cheaper side"**: x (100 x T / 1000) (i.e. 100 at T=1000, 250 at T=2500) at the best bid of the cheaper
  side (lower price; tie -> NO), placed once at the first poll in the window where a program is live and the book is
  valid without us. Never re-priced; replenished at the same price. phi 0.25.
- P1 (E1 piggyback) is dropped: weather-only.
- If a ticker has several live programs with different T at once, the order size uses the largest T.
- An order is cancelled (stops resting and filling) at the first poll where no sampled program on its ticker is
  live, at market close, or at T1, whichever is first. If a program on that ticker goes live again later in the window
  the policy is judged afresh (new order).

## Scoring (L40's filed algorithm, unchanged: lip_paper.side_score)
- Snapshot excluded if the market is closed or EITHER side has < T resting (whole book, our order inserted).
- Per side: walk down from the best bid adding whole levels until cumulative >= T; reference price = first level with
  cumulative >= T/5; score = size x DF^max(ref - price, 0). Our share = our weighted size / side total.
- **Per-program R, T and DF** are used as Kalshi lists them; period lengths differ by program.
- Each poll stands in for the interval since the previous poll of that program, capped at 120 s, clipped to the
  program window and to [T0, T1). Unobserved time counts as zero for us.
- Program payout (ours) = R x sum(share x dt) / (2 x full program length in seconds), floored to the cent; < $1.00 ->
  $0. Because we only quote inside [T0, T1), this IS what Kalshi would pay us for the whole program (we contribute
  zero outside the window), so programs that end after T1 are scored at T1 without waiting.
- Observed-time-only estimate (denominator = observed seconds) is reported alongside, never used for the bars.

## Fill models (L40's, unchanged)
Public trade tape. YES bid at y is hit by taker_side 'no' at yes_price <= y; NO bid at q by taker_side 'yes' at
no_price <= q. Only prints after placement and at or before the order's end count.
- **phi**: fill += phi x print size per print at or through our price.
- **fifo**: join the back of the queue (queue_ahead = resting size at our price, us excluded); a print through our
  price fills our remainder, a print at our price first eats queue_ahead. Cancellations ahead are not credited.
- Settlement at the market result (YES bid: +(1 - y) per contract if YES else -y; NO bid likewise).
- **Maker fee**: 0 on series fee_type `quadratic`; otherwise ceil-to-cent of 0.0175 x fee_multiplier x n x p x (1-p)
  per fill event (repo convention, edge_math.py).

## Metric and day clustering
Net $ per policy = LIP + fill settlement P&L - maker fees, over PRIMARY programs. Day clusters = UTC days of the
window: a program's floored payout is allocated to days in proportion to its per-day share x dt; each fill's P&L is
allocated to the day the fill happened. Also: net per $-day of capital.

## Success (per policy; Bonferroni over 2 policies -> one-sided p < 0.025)
10-day PRIMARY net > 0 AND day-clustered bootstrap (10,000 resamples of the 10 day nets, seed 0) 2.5th percentile > 0
AND bootstrap one-sided p (share of resampled sums <= 0) < 0.025 AND median per-program LIP >= $1. Must hold under
BOTH fill models. Verdict only after T1 and after every primary market with a fill has settled; programs still
unsettled at report time are shown as pending and excluded from net (the report says how many).

## Kill
- net <= 0 or LB <= 0 (under either fill model) at the final report; or
- Kalshi removes or changes the sampled programs: each hourly refresh logs `program_change` when a sampled program's
  R, T or DF changes, or a sampled program disappears from the active/upcoming lists before its end_date.
  Kill if such events touch >= 20% of sampled PRIMARY tickers; fewer is logged and reported. (Clarified 23:12Z,
  before T0: a start/end move with unchanged R/T/DF is logged as `program_retimed` and does not count.)

## Power note
10 day-clusters give ~9 degrees of freedom. For the 2.5th-percentile LB to clear 0 the mean daily net must be roughly
> 0.7 x its day-to-day SD (t_9 2.26 / sqrt(10)). L40's one-snapshot study (notes/kalshi_lip.md section 3) found other
categories 89% valid with W ~4,400-5,000 per side, so a x100 order buys a ~2% side share: on a $100 program that is
~$1 per program, right at the $1 minimum. P3 is therefore expected to sit near the median-LIP bar; P2 depends on how
many sampled tickers have a sub-2c YES side. Day clustering does not capture within-series correlation across
tickers (all KXRT markets move together), so a pass is a reason for a small live test, not proof.

## Caveats the paper test cannot fix
60-s polls vs 1-s snapshots; our order would change others' quotes; queue position unknown; Kalshi picks which
markets are incentivised and can end programs at any time (it just did for weather).

## AMENDMENT 2026-10-30 ~04:30Z (operational, after T0; no outcome looked at)
Ticks stalled 00:08Z-04:30Z: sustained HTTP 429 on our IP made the trade-tape catch-up fail every tick. Rule added: if the tape
since the last tick cannot be read, every open order is ended with NO fill and NO further LIP (logged `cancel why=tape_gap` +
a `tape_gap` event), and its ticker starts a fresh episode so the frozen policies re-place at the next poll. Rationale: carrying
orders with unknown fills would credit LIP without fill risk (optimistic); unobserved time already earns zero (W_CAP).
Report must list tape_gap events and their total duration; bars are unchanged.
