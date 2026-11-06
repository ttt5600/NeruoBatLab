# Cross-venue in-game lead-lag: Polymarket (intl CLOB) vs Kalshi, same game winner (2026-11-06)

Script: `scripts/xvenue_ingame.py` (list / match / minute / ticks / lag). Data: `data/xvenue_ingame/` (gitignored).

## Pre-registration (written 2026-11-06 BEFORE any price series was pulled)
Question: after in-game price moves, does one venue lag the other by enough to buy the stale side after spread + fee?

DATE split (fixed now, before looking; game start date, UTC):
- dev: 2025-01-01 .. 2026-02-28 -- the ONLY window the descriptive analysis below reads.
- sealed: 2026-03-01 .. 2026-11-05 -- not pulled / not read unless a registered trial passes dev.
- forward: 2026-11-07 ..

Descriptive measures (dev only):
1. Matched n per league (same game, same team side, unique match on date + team codes/names).
2. Minute grid: Kalshi 1-min candle close yes bid/ask (mid); Polymarket clob prices-history fidelity=1 (last value
   per minute). In-game window = game start .. start + league duration, while both mids in [0.03, 0.97].
   Lead-lag = pooled cross-correlation of 1-min mid changes at lags -5..+5; error-correction regression
   dK(t->t+1) ~ (P_t - K_t) and dP(t->t+1) ~ (K_t - P_t), game-clustered bootstrap CIs.
3. Tick sample (<= 40 dev games per league for NBA/NFL/NHL/CBB): trade tapes both venues; 5-s last-trade grid,
   return cross-correlation at -60..+60 s; event study on jumps (>= 4c within 10 s on one venue): seconds until the
   other venue covers half the jump; median + game-bootstrap CI.
4. Gap/convergence: gap = P_mid - K_mid. Among minutes with |gap| >= g (g = 2, 3, 5c): share of the gap closed by
   each venue over the next 1 and 5 min.

DESCRIPTIVE GATE (decides whether any money trial is registered): signal = Poly mid - (Kalshi ask + Kalshi taker
fee) >= X (X in {3c, 5c}; mirror for NO). Gate passes only if, for some X, mean CLV_5min = Kalshi mid(t+5 min) -
ask(t) - fee(ask(t)) > 0 with game-clustered 95% CI excluding 0, on >= 200 signals in >= 50 games, with the ask
taken at the candle AFTER the signal minute (one-minute execution delay) -- because the Poly minute value is a
last-in-minute print, same-minute execution would assume we saw Poly before Kalshi's close. Same-minute version
reported as an upper bound only. If Kalshi leads Poly instead, the mirrored test (buy Poly at its ask) needs a Poly
ask, which only the tick sample has; it cannot be traded by a US user on the intl book anyway (see Execution).
If the gate fails: no trial is registered; the result is a descriptive DEAD.

## Results (2026-11-06, dev only; sealed 2026-03-01+ never pulled)
Matched (same game + side, unique): 12,176 games (dev 7,325: cbb 2,734, mlb 2,112, nba 926, nhl 762, cfb 461, nfl 330).
Minute study = random 400 dev games/league (seed 0) -> 2,285 usable; tick study = 30/league NBA/NFL/NHL/CBB -> 112.
Polymarket prices-history `p` = MIDPOINT (sits 0.5c from the last trade toward the other side, 120 games), sampled
~every 60 s at ~:07 past the minute.

**Alignment bug found and fixed:** `lag` (as pre-registered) paired Kalshi candle close (:00) with the Poly point
53 s earlier -> fake "Kalshi leads by 1 min" (corr L+1 .63-.90) and gate CLV -2.4c. `lag2` uses the Poly point time
as the grid with Kalshi = last close <= it (7 s older). All numbers below are `lag2` unless marked.

Lead-lag, 1-min aligned grid, corr(dK_i, dP_{i+L}): contemporaneous dominates (L0 .69 cbb/cfb, .78 mlb, .81 nfl,
.87 nhl, .92 nba); spillovers small both ways (L-1 .02-.06, L+1 .02-.07).
Trade level (5-s last trade grid, 112 games): corr at +5 s .204, +10 s .143 vs -5 s .009 -> Kalshi trades first.
Jump study (>= 4c in 10 s, persisting): Kalshi-led 2,435 events, Poly covers half in median 10 s [10,15];
Poly-led 2,303 events, Kalshi median 20 s [15,30] but 34% already moved. CAVEAT: Poly trade timestamps are on-chain
(settled after off-chain match), so a few seconds of the Kalshi lead may be stamping delay.
Gap |Poly mid - Kalshi mid|: median 0.95c, p90 2.0c, p99 9.0c; 5.5% of minutes >= 3c.
Error correction (share of gap closed in 1 step, game-bootstrap): |gap|>=3c Kalshi .114 [.085,.154], Poly .295
[.218,.396]; |gap|>=5c Kalshi .055 [.036,.086], Poly .277 [.191,.419]. Neither venue fully converges in 5 min.

DESCRIPTIVE GATE (pre-registered: buy Kalshi at ask+fee toward Poly) -- FAIL:
- aligned, delayed (ask 1 candle later): X=3c n 2,578 / 968 games CLV_5min -2.58c [-3.06,-2.08] (-6.7%/$);
  X=5c n 1,656 / 680 games -2.55c [-3.08,-2.02]. Same-candle upper bound: -1.21c [-1.68,-0.70]; -1.28c [-1.85,-0.68].
- as originally coded (misaligned): X=3c -2.37c [-2.53,-2.20]; X=5c -2.41c [-2.64,-2.17].
-> No edgelab trial registered (family cross_venue stays at 7 trials, bar 7.1e-3).

Mirror direction (NOT pre-registered as a gate; NOT tradable by a US user on the intl book):
- minute (`xvenue_ingame_mirror_minute.py`): buy Poly at mid + 0.5c + PMUS fee 0.0695p(1-p) when Kalshi mid - Poly
  mid >= X: X=3c h=1 +0.87c [+0.66,+1.07] (n 17,815, overlapping); X=5c h=5 +4.10c [+3.34,+4.91].
- ticks (`xvenue_ingame_mirror_ticks.py`): after a Kalshi jump, Poly last trade moves +4.5c in 60 s; net of 0.5c +
  fee +2.55c [+2.07,+3.05] (2,077 events, 107 games).
TOO-GOOD WARNING: the Poly "mid" has no ask behind it -- when Kalshi jumps, Poly makers likely pull/widen, so the
real ask can be far above the reported mid; trade stamps are on-chain. Treat as a LEAD only. Right test = Polymarket
US (the book a US user can trade; its gateway price-history gives ASK/BID quotes ~1-min, history from 2026-05) vs Kalshi,
pre-registered on a new dataset with its own date split, plus a live websocket logger to measure real latency.

## Execution facts
- US user: international Polymarket (polymarket.com) blocks US users -- never bypass. Polymarket US
  (polymarket.us, own DCM book, taker fee 0.0695*C*p(1-p), API keys at polymarket.us/developer, automated orders
  allowed) is a DIFFERENT book; nothing here measures it. Kalshi: taker 0.07*p*(1-p).
- Latency needed: the measured lag is ~10-20 s at trade level and gone within ~1 min -> needs websocket feeds on
  both venues and order placement in < ~5 s; a 1-min poller cannot capture it (our gate's 1-candle delay kills it).
