# nba_lockin_fill -- can a 99c maker bid on NBA locks get filled earlier than the horn? (agent nba_lockin_fill, 2026-11-11)

Parent: lock-in T2 (edgelab trial lock_t2_nba_maker99, needs_audit; sealed NOT opened and stays closed).
Logger data: ~/.claude/quant-sports/data/live/lock_queue.jsonl (11-09, 11-10; read-only). Historical: data/nba_late_states.parquet
(ESPN Q4+ states 2021-10..2026-06, espn_wp, home_won), data/lock_in/markets.parquet + Kalshi public historical trades/candles.
Scratch: results/nba_lockin_fill/.

## Questions
1. When is each lock detectable vs when the 99c queue builds; is the queue shorter earlier?
2. Fill probability given queue ahead and 99c sell volume after joining.
3. Which locks are reachable (the 21% of KXNBAGAME with queue <= 5,000)?
4. Risk of 'practical' moneyline locks (lead L, t seconds left); EV/contract after fee; capacity $/night.
5. More logging needed? Nights to a decision? Pre-register a NEW forward trial.

## Pre-registration (written 2026-11-12 ~01:45 UTC, before computing any outcome below)
Fee: KXNBAGAME and KXNBATOTAL fee_type = quadratic_with_maker_fees (GET /series, 2026-11-12) -> maker 0.0175*C*p*(1-p),
rounded up to the cent per order (repo convention, scripts/edge_math.py:21-23).
Practical-lock grid (fixed now, 16 cells): leader by >= L in {5, 7, 9, 12} points with <= t in {120, 60, 30, 15} s left in
regulation Q4 (period 4 only; OT excluded from cells).
- Comeback (loss) rate per cell estimated on ESPN states with date < 2025-10-21 (seasons 2021-24, no Kalshi) and checked on
  the Kalshi dev window 2025-10-21..2026-01-31 with Kalshi `result` as the outcome (not ESPN last play). Rows dated >= 2026-02-01
  (the sealed window) are NOT read for outcomes.
- Cell is RISK-OK iff the Wilson 95% upper bound of the loss rate < 0.5% (half the 1c margin, leaving room for fee + adverse
  selection).
- Fill channel (dev Kalshi tape + 1-min candles, winner-side ticker): at the first state entering the cell (+60 s ESPN latency),
  read the YES bid/ask. If best YES bid < 99 and ask > 99, a 99c bid is first in queue -> fill = 99c sells (taker_side no at 99)
  after T. If ask <= 99 the bid would cross (taker at the ask). If bid == 99 the queue is unknown (counted as NOT fillable).
- Cell is FILL-OK iff >= 30% of dev games entering the cell are first-in-queue AND get >= 100 contracts of 99c sell volume after T.
- SUCCESS (forward trial worth registering as a maker trial): >= 1 cell both RISK-OK and FILL-OK.
- KILL (lock-in 99c maker dead for moneylines): no cell is both; then the horn-only version is judged from the logger alone
  (share of 99c volume reaching a joiner after the queue: 0.000 on both nights per the SCORECARDS LOCK card).
- Adverse selection check: loss rate among dev 'filled' entries vs unfilled (E5 lesson). Report, not a gate (too few losses).

## Log
- **Horn reachability (Q3)** (`scripts/nba_lockin_fill.py horn`, logger 11-09/10): the "21% of KXNBAGAME <= 5,000" is
  3/14 books that were ALREADY CLOSED (finalized 2, determined 1; queue 0 because the book was empty, lag 3.6-595 s).
  Active books with 99c queue <= 5,000: **0 of 188** (KXNBAGAME 0/10, min 109,590, median 267,042; KXNBATOTAL 0/178,
  min 6,628, median 24,595). Logger's own share of later 99c volume reaching a joiner: 0.000 at every delay. So nothing
  is reachable at the horn or at the total-rung lock; the 21%/4% were closed markets, not short queues.
- **Comeback grid (Q4)** (`... grid` -> results/nba_lockin_fill/comeback_grid.csv; ESPN home_won, pre = 2021-10..2025-10-20,
  ~5,300 games): loss rate at first state in cell, Wilson 95% UB:
  L5: t120 1.99% (UB 2.43), t60 0.67% (0.95), t30 0.23% (0.42), t15 0.07% (0.21);
  L7: t120 0.71% (1.02), t60 0.13% (0.30), t30 1/3,778 (0.15), t15 0/3,677 (0.10);
  L9: t120 0.23% (0.45), t60 2/3,349 (0.22), t30/t15 0 (0.12); L12: 0 in every t (UB 0.14-0.16).
  RISK-OK (UB < 0.5%): L5 t<=30, L7 t<=60, L9 all, L12 all. Dev (Oct25-Jan26, ~730 games) agrees: L7t30 and up 0 losses,
  but n ~400-550 so UB 0.75-1.15% (dev alone cannot certify <0.5%). Caveat: home_won matches the last ESPN score in
  99.18% of games and only 86.6% of games end at clock 0 (truncated feeds), so ESPN-label losses may be off by a few;
  dev re-checked against Kalshi `result` in the fills step.
- **Fills at practical lock (Q1/Q2)** (`... fills` -> results/nba_lockin_fill/fills_dev.parquet, fills_grid.csv; dev 729
  joined games, 7,386 cell entries, 604 leader tickers): at the first state in EVERY cell (+60 s), the leader's best YES bid
  is ALREADY 99c (median bid .99, ask 1.00 in all 16 cells). First-in-queue (bid < 99, ask > 99): 0.0-3.1% of entries;
  FILL-OK (first-in-queue + >= 100 ct of 99c sells after): 0.0-1.9% per cell (bar was 30%). Cross (ask <= 99): 0% at
  L>=9 t<=30, up to 37% at L5 t120 (= taker at 99 in the riskiest cell). Median 99c sell volume after entry is large
  (L5t120 17,688; L9t60 3,748; L12t15 457 ct) but it goes to the queue already standing at 99c.
  Kalshi-result losses: L5t120 8/602, L5t60 4/573, L7t60 2/507, L7t120 3/527, L5t30/t15 1; L>=9 and L7t<=30 0.
  ESPN and Kalshi loss labels agree in every cell. Losses among FILL-OK entries 0/67 vs 19/7,319 otherwise (n too small).
  **PRE-REGISTERED VERDICT: KILL** -- no cell is both RISK-OK and FILL-OK; posting at practical lock is NOT earlier than
  the queue: the 99c bid (and its queue) exists before any of these game states.
- **When does the 99c bid appear? (Q1, post-kill descriptive)** (`... onset` -> onset_dev.parquet, onset.log; dev, 1,410
  ML tickers, 706 ever showed a 99c bid on a 1-min candle close): game state at onset median 126.5 s left (p25 18 s,
  p75 368 s, p90 564 s), lead median 9 (p75 15, p90 20), ESPN wp median .986 (p25 .942); onset is median 1,589 s before
  market close. 99c sells AFTER onset: median 65,173 ct (44,613 within 5 min; p10 13,312). So the queue starts ~2 min
  before the horn at lead ~9 and the horn queue (median 267k live) is what accumulates after. Being FIRST at 99c = posting
  at or before the onset state: losers among onset tickers 3/706 = 0.42% (Wilson ~[0.14, 1.24]%): PHX 25-11-16 (lead 21
  at 8:21 by ESPN state, lost 122-124), BKN 26-01-23 (OT, lead 3 7.9 s, lost in 2OT), HOU 25-12-18 (onset in Q3/OT
  mismatch, 40 ct). The two big losers had 60,104 and 7,367 ct of 99c sells after onset -> a front-of-queue bid fills on
  the comebacks too (adverse selection is real).
  Note: grid rows with clock 0 now excluded (clock_s > 0) for the early grid; the first grid ran without it.
- **Earlier/larger-lead comeback grid** (`grid 12,15,18,20 600,420,300,180` -> comeback_grid_early.csv, chosen AFTER the kill;
  pre seasons): Wilson UB < 0.5%: L12 t180 (0.30%), L15 t300 (0.48%)/t180 (0.17%), L18 t<=300 (0.20-0.22%), L20 t<=420
  (0.24-0.34%). e.g. L15 t300 5/2,459 = 0.20%; L18 t420 4/2,016 = 0.20% (UB .51). Dev (n 200-430) 0 losses in those cells.
- **Early-cell fills** (`fills 12,15,18,20 600,420,300,180` -> fills_dev_early.parquet, fills_grid_early.csv, fills_early.log;
  dev 4,873 entries / 466 tickers): first-in-queue at 99c is still 0.0-3.4% in every cell. Mechanism: the book runs a
  1c spread, so while the bid is < 99 the ask is <= 99 and a 99c bid CROSSES (cross share 37-69% at t 420-600) -- you are a
  taker, not a maker. The instant the bid reaches 99 the ask goes to 100 and the 99c queue is already large.
- **Tonight's live probe** (results/nba_lockin_fill/prelock_probe.jsonl, scripts/nba_lockin_prelock_probe.py, read-only,
  11-11, descriptive only): WAS@IND leader IND 98/99 at lead 14 6:21 left; 40 s later (5:50, lead 16) bid 99 with 80,265
  contracts already resting, no NO bids (ask 100). Queue 150k at 2:47, 226k at 0:59, 312k at 0:18. The 99c queue
  appears in one block (market makers), not gradually.
- **Taker side (NOT pre-registered, descriptive)**: when the ask <= 99 at a cell entry, buying at the ask: dev distinct
  tickers 224 (first grid) EV +2.74 c/ct [+0.05, +4.92] (driven by L5 t120 asks ~0.93, 8 losses) and 333 (early grid)
  +1.00 c/ct [-0.98, +2.82] (not distinguishable). This is a late-game win-probability bet (nba_late_wp / bigleads
  territory), not a lock; any test belongs to that family, forward-only.

## Answers
1. Detectable vs queue: over rungs lock when sum >= K+3.5 (all 191 logged in period 4); the queue is already there
   (median 24,595, min 6,628). Moneylines: the 99c bid appears ~2 min before the horn (median 126 s, lead 9, ESPN wp
   .986) and arrives as a block (80k in < 40 s tonight). Posting earlier than that = crossing a 98-99c ask.
2. Fill model: FIFO, fill = max(0, 99c sells after joining - queue ahead). Horn: 0.000 share of later 99c volume reached
   a joiner at every delay (logger report). Front-of-queue (dev): 99c sells after onset median 65,173 ct -> a front bid
   would fill, but front is reachable in 0-3.4% of practical-lock entries.
3. Reachable: none. The 21%/4% "<= 5,000" were finalized/determined/inactive (empty) books; active books 0/188.
4. Risk: practical-lock loss rates (pre 2021-25) e.g. L9 t<=60 2/3,349, L12 any t<=120 0/~2,500, L5 t120 1.99%. First at
   the 99c onset: 3/706 = 0.42% [0.14, 1.24] -> EV = 1 - 99*p - fee = +0.58c [-0.24, +0.86] per contract, not
   distinguishable from 0; losers drew big 99c sells (adverse selection). Fee: qwmf maker 0.0175 p(1-p) = 0.0173c/ct,
   rounded up per order -> a 1-contract fill pays 1c = the whole margin; need >= ~58 ct/order for fee < 0.02c/ct.
   Capacity if front: ~65k ct/game of 99c sells (~$650/game at 1c); realized: ~$0 (never front).
5. Logging: the horn logger has answered its question (min active ML queue 109,590 vs 117,063 total later 99c volume
   over 10 locks). No more nights needed. A front-of-queue maker trial would qualify ~2-3% of entries (~0.2/night) ->
   100 fills ~= 500+ nights. Decide now.

## Verdict + proposal
- lock_t2_nba_maker99 (horn, N=5,000): DEAD on live depth (0/188 reachable). Propose `edgelab retire lock_t2_nba_maker99`
  (lead runs it; sealed stays closed) and uninstalling launchd quant-sports-lock-queue (lead/user; nothing touched).
- NO new maker trial registered: the pre-registered rule killed it and no N/timing from pre-11-11 data gives a fillable
  maker (front-of-queue <= 3.4% in all 32 cells). Registering one would only raise the lock_in family bar. Draft if the
  lead wants it on record: lead >= 15, <= 300 s left, Q4, leader KXNBAGAME, post 99c only if bid < 99 AND ask > 99 (queue
  ahead 0), fill = forward 99c sells after post, kill losses_in_first n 100 max 1; ~0.15 entries/night (dev 1.9%).

## Learnings
### NBA 99c lock queues are unreachable at the horn
Active books at the lock moment had 99c YES queues of 109,590-1,745,605 (moneyline, n 10) and 6,628-67,852 (over rungs,
n 178); 0/188 were <= 5,000 and 0% of later 99c volume reached a joiner (lock_queue.jsonl 11-09/10; nba_lockin_fill.py horn).
### "Short queue" rows were closed markets
The 21% of moneylines with queue <= 5,000 were finalized/determined/inactive books with nothing resting. Filter
market_status == active before reading depth.
### The 99c bid on NBA moneylines appears ~2 min before the horn, as a block
Dev 706 tickers: onset median 126 s left, lead 9, ESPN wp .986 (results/nba_lockin_fill/onset.log); live 11-11: 80,265
contracts at 99c within 40 s of the bid reaching 99 (prelock_probe.jsonl). Posting earlier means crossing a 98-99c ask.
### With a 1c spread there is no front-of-queue at 99c
While the bid is < 99 the ask is <= 99, so a 99c bid is a taker. First-in-queue at 99c in 0-3.4% of 7,386 + 4,873
practical-lock entries (fills_grid.csv, fills_grid_early.csv).
### Kalshi per-order fee rounding kills 1-cent edges on small orders
qwmf maker fee 0.0175 p(1-p) at 99c = 0.017c/ct but rounds up to 1c per order: a 1-contract fill pays the whole margin.
### Being first at 99c on NBA leads loses ~0.4%, and losers fill
3/706 onset tickers lost (0.42% [0.14, 1.24]); EV +0.58c [-0.24, +0.86]/ct, not distinguishable; the two big losers
drew 60,104 and 7,367 contracts of 99c sells (onset_dev.parquet).
