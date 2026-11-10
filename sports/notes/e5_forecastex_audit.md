# e5_forecastex_audit -- E5 ForecastEx weather consensus-reject: live record + executable-price check (2026-11-09)

## Question
Does E5 (sealed PASS-UNDERPOWERED 10-28 at LAST-TRADE prices: 219 bets, 10 losses, ROI +6.9%, p .0004) survive at
prices we can actually execute (ask = taker; best bid = maker, only if filled)? What is the full live record?

## Pre-registered rules (written before computing the forward/live numbers below)
- Primary executable test = FX1-fwd paper at the real ASK (+$0.01 fee), all settled picks with an ask present.
- Verdict map:
  - confirmed: ask-ROI > 0 with 95% bootstrap CI excluding 0 (station-day clusters).
  - holds: ask-ROI point > 0, CI crosses 0.
  - weakened: ask-ROI point <= 0 but CI includes the sealed +6.9%; OR ask present on < 25% of picks (not executable at scale).
  - dead: ask-ROI CI upper bound < +6.9%... no -- dead = CI upper bound < 0, OR maker fills show adverse selection
    (filled maker picks lose more than unfilled at the same price, difference CI excludes 0) AND ask-ROI <= 0.
- Maker: fill rate = filled contracts / placed; adverse selection = loss rate filled vs unfilled picks.
- Q008: pre-set kill (QUESTIONS.md): at 60 settled, ROI <= 0 or qualifying-ask rate < 25%; early >= 3 losses in first 30.

## Data (read-only)
~/.claude/quant-kalshi/data/forecastex_{exec,orders,live,paper,q008_obs}.jsonl, bets.jsonl;
`forecastex_paper.py report`, `forecastex_exec.py status` (deployed copy ~/.claude/quant-kalshi).
Scratch: sports/results/e5_forecastex_audit/

## Log
### 1. Live record (forecastex_orders.jsonl 222 intents; forecastex_exec.jsonl; forecastex_live.jsonl; bets.jsonl 20 fx rows)
- 10-29 old script (FX1L-, CP gateway): 26 orders (25 MK @ bid, 1 taker NO LGA_102926_71 @ .98), 1 contract each, $24.27
  committed. NO fill/cancel/settle rows anywhere (GAP: never reconciled, absent from bets.jsonl and the +$5.40).
  All 26 tickers WON in paper -> P&L bounded 0..+$1.73 (if all filled).
- 10-30..11-04: 176 E5X intents returned IBKR "Inactive" (emailed-token verification block, FUND_NOTES 10-31/11-01) = 0 risk.
- 11-03..11-10: 20 orders accepted (exec 'order' rows) -> 1 filled (917325502 UHATL_110626_75 NO 20 @ .72, win +$5.40 net
  of $0.20 fee), 12 cancelled at cutoff, 3 expired_not_found (11-04 LAS/LAX, lost in the gateway outage; all 3 tickers won,
  gateway positions now 0), 4 resting (open risk $59.04). Settled 16: +$5.40 on $14.60 risked (n=1 fill; uninformative).
- Live maker fill rate: 1/16 settled orders (6%), 20/245 contracts.
- Throughput choke: 373 decisions REFUSED by bet.py daily loss stop on "open B+C risk ~2.9u" (non-FX positions), incl. every
  day 11-05..11-10 after the 11-07 phantom fix; 185 skipped by daily cap; 38 not listed; 34 TWS symbol/contract errors.
### 2. Executable prices (paper_audit.py -> settled_picks.csv; 693 settled FX1-fwd picks 10-29..11-08, 230 station-days)
- Forecast signal is real: 12 losses / 693 (1.7%, Wilson [1.0%, 3.0%]) vs b/e ~7% at bid. Losses = 6 station-days.
- But: ask present on only 6.9% of picks; bid 97.5%. last_d1 present 0% (sealed last-trade method cannot be replicated fwd).
- A at ask: n36 (28 clusters) 5 losses ROI -8.5% CI [-25.1%, +6.1%].  MID: n48 -4.1% [-20.2%, +9.4%].
- MK proxy (fill iff day-D low < bid): n53 12 losses ROI -15.1% CI [-30.9%, +2.3%].
- ADVERSE SELECTION: ALL 12 losses are in the filled-by-proxy set: filled loss rate 12/53 = 22.6% [13.5%, 35.5%] vs
  unfilled 0/623 [0, 0.6%]; Fisher p 1.5e-14. 'All at bid' +5.7% [+3.9%, +7.2%] is NOT executable (fills only come on
  losing days). Proxy validated on live: 16/16 settled live orders filled exactly where the proxy said (only fill had
  low_d .25 < bid .72; ATL_110826_72 low .93 > bid .91 not filled).
- Mechanism: bids rest D-1 evening -> cutoff on event day; the only sellers are day-D traders who already see the temp going
  against us. Sealed +6.9% at last trade likely carried the same selection (last trades happen on moving days).
### 3. Q008: 0 picks, 65 obs rows; 56 Kalshi-favoured contract-days, FX ask present 68%, qualifying-ask rate 0% (< 25%
  kill); median FX ask - Kalshi mid +0.005. Never reaches 60 settled; kill condition on qualifying rate met -> KILLED (STATE
  already lists Q008 as dropped).
### 4. Verdict (pre-registered map): DEAD as an executable edge -- adverse-selection CI excludes 0 AND ask-ROI <= 0;
  also ask present 6.9% < 25%. The forecast is good; no execution channel captures it.
PROPOSAL (lead/user decides): STOP E5 live (unload quant-forecastex-exec after the 4 resting orders resolve or cancel them);
  keep forecastex_paper logging free. Only revival worth a prereg: D-1-only maker (cancel bids before local midnight of
  event day) -- but D-1 prints are ~absent (last_d1 0%), so capacity ~0. Reconcile 10-29 FX1L orders from the IBKR
  statement (<= +$1.73).
