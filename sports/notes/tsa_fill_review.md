# tsa_fill_review -- why does B4 (KXTSAW weekly maker) show 0 fills? (agent, 2026-11-15)

## Question
B4 scorecard (notes/SCORECARDS.md) says paper posted 17 / tape-filled 0, live 8 resting / 0 filled, and the 11-08 week's
8 live picks were all skip_stale_signal. Diagnose: our bid vs book and tape; dev fill model vs today's books; what price
fills and is it +EV; is skip_stale_signal a bug.

## Pre-registered decision rule (written before pulling the tape)
- "Fill problem" = no trade at/through our price after posting on >= 1 full week of posts -> reprice question.
- Reprice is only proposable if, at the price that would have filled, dev break-even loss rate (1-p) > dev filled
  loss-rate upper bound. Otherwise kill. This is diagnosis on n = 2 weeks, so no edge claim is made either way.
- skip_stale_signal = bug iff the signal was fresh by the design (<= max_age_h) when lead_exec ran and still skipped.

## Plan / data
- Paper posts ~/.claude/quant-sports/data/e1x/tsa_paper.jsonl; live journal ~/.claude/quant-kalshi/data/lead_exec.jsonl.
- Tape/books read-only via scripts/kx_throttle.py -> results/tsa_fill_review/pull.py -> tape.json.

## Results (2026-11-15; tape pulled ~11:45 UTC, results/tsa_fill_review/tape.json, dump tape_dump.txt, analyze.py)

### 0. The "0 fills" premise is stale
- lead_exec.jsonl: KXTSAW-26NOV15-A1.95 / A1.90 / A1.85 YES @ .96 status filled, first_fill 2026-11-13 21:55 / 22:10 / 22:10 UTC.
- tsa_paper.jsonl: 4 paper fills on 26NOV15 (A1.95, A1.90, A1.85, A1.80). So live 3/8 placed filled, paper 4/17 posted (24%).
- The B4 card in notes/SCORECARDS.md was generated before 11-13. Settlement of 26NOV15 is pending (rungs 1.80-1.95M vs
  F 2.22M; A2.20 still bid .84 at pull time), so 0 settled fills is still true.

### 1. Our bid vs book and tape, every posted order
- 26NOV08 (8 posts, paper only; live skipped): 6/8 rungs had ZERO trades all week (A2.80, A2.05-A1.85). A2.10 and A1.80
  traded only as YES-taker buys at .91 on 11-05 13:33 (a resting YES ask at .91 across A1.80-A2.20). The paper rule counts
  only NO-taker prints, so it scores 0; a real .96 YES bid would have been crossed by that .91 ask and filled at .96. The
  paper model under-counts here (conservative direction).
- 26NOV15 low side (A1.80-A1.95, YES @ .96, yes_bid .95 / ask 1.00 at post): one seller swept the books 11-13 21:50-22:06
  UTC. The 1-ct prints at .96 on A1.95 / A1.90 / A1.85 are OUR live orders (we were alone at .96, nothing queued ahead;
  next level .95 x ~200). On A1.85 the same sweep then hit .95 x200, .60, .21 x39.7 (a liquidation, not information-looking;
  settlement will tell). A1.80 (skipped live by skip_period) printed a NO-taker at .93 x26.8 on 11-14 06:18.
- 26NOV15 high side (A2.60-A2.80, NO @ .96): ZERO trades on every rung >= 2.45M all week (market volume 0). Book at pull:
  NO .97 x270 on A2.70-A2.80 sits AHEAD of us; on A2.60/A2.65 we share .96 with ~250 ct. These 5 orders ($4.80) earn nothing
  and use up the $8/week cap, which is why A1.80 got skip_period -- and A1.80 is the rung paper says filled.

### 2. Dev fill model vs today's books
- Quote at post is the same as dev: low rungs yes_bid .95 / ask 1.00 both weeks; dev/sealed low-side fills mostly had yb
  .94-.97 (tsa_bets_T2_{dev,sealed}.csv). Forward fill rate 4/17 vs dev 13% / sealed 7% -- consistent at n=17, not optimistic.
- High-side NO fills by quarter (dev+sealed CSVs): 0 since 2026Q1 (7 in 2026Q1, 0 in Q2-Q3). The high leg is now dead weight.
- Capacity: sealed 2026Q2-Q3 had 8 fills in ~6 months. Fills come in sweeps (one seller per week at most).

### 3. What price fills, and is it +EV there
- The fills we get are at q = yes_bid + .01 = .96. Going down to .95 would have queued behind ~200 ct and missed A1.90/A1.95
  (the sweep took only 1 ct there). Repricing down loses the fills; repricing up (.97) raises break-even loss to 3%.
- At .96 break-even loss rate = 4.0%. Dev+sealed fills at q > .955: 42 + 17 = 59 fills, 0 losses, ROI +3.0-3.3% per bucket;
  q > .94: 81 fills, 1 loss (TSAW-24OCT20-A2.60 NO @ .95). Upper 95% bound on the loss rate ~5-6.7% > 4% break-even, so
  +EV at .96 is plausible, not established. The headline +8.4% DEV came from cheap fills (q < .80) in chaos weeks
  (24DEC01, 25NOV16), which are also where all 5 other dev losses sit. Expect ~+3% per fill in calm weeks, not +8%.
- Adverse selection forward: cannot judge until 26NOV15 settles (~11-17).

### 4. skip_stale_signal = not a bug
- The very first row of lead_exec.jsonl (all strategies) is 2026-11-05T17:25:10 -- lead_exec was deployed then. The 26NOV08
  posts were made 11-02 17:04, src_age 72.35 h > TSA max_age_h 24 -> skipped by design (guards against posting a Monday
  quote on Thursday). On 26NOV15 the same path posted 14 min after the paper post (src_age 0.24 h). TSA data timing is not
  involved: the logger's forecast uses data through the Saturday before T (forecast()), as pre-registered.

## Recommendation (PROPOSAL, lead/user decide; nothing changed)
KEEP B4, with two live-mirror changes (paper logger stays exactly as pre-registered):
1. In lead_exec, skip TSA side=no (high-rung) posts, or at least exclude them from the $8 weekly cap. 0 trades on them this
   week, 0 high-side fills in the backtest since 2026Q1, and other makers sit at .97 ahead of us. They blocked A1.80.
2. Do not reprice: the rule's yes_bid+.01 is what actually gets hit in sweeps.
Also: refresh the B4 scorecard (3 live fills); note that paper tape-fills now include our own live prints (not independent
evidence), and that the paper rule misses fills from resting asks that would have crossed our bid (NOV08 .91 asks).
Kill rule unchanged (2 losses in first 30 fills). Expected pace ~0.5-2 fills/week -> 30 fills takes months.

## Learnings
### A 1-ct maker bid alone at the inside is first in line for sweeps -- the paper tape fill then prints our own order
26NOV15 fills: the 1-ct .96 prints on A1.85/A1.90/A1.95 are our live orders (lead_exec.jsonl vs results/tsa_fill_review/
tape_dump.txt). Once live mirrors paper, the paper fill rule is no longer independent evidence of fill rates.
### Trade-print fill models miss fills from incoming resting orders that would have crossed our bid
KXTSAW-26NOV08: a YES ask rested at .91 (bought by YES takers) while our paper bid was .96; the paper rule (NO-taker prints
only) scored no fill, a real order would have filled. Conservative bias, but it understates fill rate on thin ladders.
### Legs that never fill can starve the legs that do through a shared budget cap
TSA high-rung NO posts (0 trades all week) used $4.80 of the $8 weekly cap; the A1.80 YES post was then skip_period and that
rung later printed a qualifying fill (tsa_paper.jsonl). Check cap allocation per leg before deploying a multi-leg maker.
### "0 fills" on a scorecard can be a stale snapshot -- read the journal before diagnosing
B4 card said 0 fills; lead_exec.jsonl had 3 fills from 11-13. First step of any fill review: last status per order id.
### stale-signal skips in the first week after a deploy are expected
lead_exec first journal row 2026-11-05 17:25; signals from 11-02 were 72 h old vs max_age 24 h.
