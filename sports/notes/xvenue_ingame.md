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

## Results
(filled in below after the run)
