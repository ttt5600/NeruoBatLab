# Kalshi -> Polymarket US in-game lag at the REAL ask (2026-11-06)

Follow-up to notes/xvenue_ingame.md: there Kalshi led the international Polymarket book by ~10-20 s after jumps, and
buying Polymarket toward Kalshi looked +0.9..+4.1c at the Polymarket MID (no real ask; untradable for a US user).
This note tests the SAME idea on Polymarket US (separate CFTC book, US-tradable) at its book-derived ASK.

## Data available (probed 2026-11-06, metadata + probe day only)
- PM US public gateway (no key): `/v1/events?seriesId=&closed=true` lists closed games back to 2025 (NBA/NHL/CBB/NFL
  from 2025-10/11, MLB 2026-03.., CFB 2026-08..); moneyline long side = first team in the slug.
- `/v1/price-history` custom range, fidelity 1: ONE stored observation per minute exactly ON the minute (probe game
  aec-nba-orl-pho-2026-11-05: 181 obs, gaps all 60 s). longPrice = YES display price (normally best ask), shortPrice =
  NO display price (normally 1 - best bid). So the historical ask is seen only at :00 of each minute.
- PM US trades: not public (report/search-trades needs keys).
- **PM US Markets WebSocket (`wss://api.polymarket.us/v1/ws/markets`) REQUIRES an API key** (docs websocket/overview,
  markets). No key -> the live logger polls the public REST `/v1/markets/{slug}/bbo` instead (~1 s per market).
- Kalshi: public trade tape with sub-second created_time; Kalshi WS also needs a key (we have one in .env, read-only use).

- Live smoke (2026-11-06 10:34Z, 10 pre-game NBA/NHL games, 90 s): Kalshi WS trade+ticker works with our key.
  PM US REST bbo is Cloudflare-cached `max-age=30`: 657/678 views were cache HITs, Age median 14 s (p90 26 s), so a
  keyless poller sees PM US 0-30 s late -- it cannot observe the ask 3 s after a jump except when a refill happens to
  land then. bbo's `lastPriceSample.ts` was 20-60 s old pre-game (sampled on change? on a timer? -- unknown).

## Pre-registration (written BEFORE any PM US price or Kalshi tape of a split game was pulled)
Dataset `pmus_ingame_lag` (vault, UTC date of PM US game start): dev 2026-05-01..2026-08-31; sealed 2026-09-01..
2026-11-04; 2026-11-05 = plumbing probe day (no split); forward 2026-11-06.. = live logger only.
Games: PM US closed `aec-` game moneylines (NBA, NHL, MLB, NFL, CFB, CBB) matched uniquely to a Kalshi game event
(date +-1, both team codes); random 700 games per split (seed 20261106 dev, 20261107 sealed).
Jump (frame): Kalshi trade on the PM-long team's ticker whose YES price differs >= 4c from the last trade at or before
10 s earlier; start <= t <= Kalshi close, 0.05 <= price <= 0.95; at most one per game per 60 s.
Side: Kalshi up -> buy PM US YES at longPrice; down -> buy PM US NO at shortPrice.
Execution: the PM US observation with timestamp in [t+3 s, t+10 s] (decision = that observation; latency 3-10 s).
Jumps with no observation in that window are dropped (the minute grid's phase is unrelated to jump timing, so this is
a ~12% random subsample). Ask in [0.03, 0.97]. Cost = ask + PM US taker fee 0.0695 p(1-p) (exact; docs fees.md,
effective 2026-10-28; banker's rounding to the cent per fill ignored = exact). $1 cost per bet, hold to PM US settlement.
- T1 `pmus_lag_t1`: every such jump.
- T2 `pmus_lag_t2`: only if still stale: Kalshi jump price (side terms) - ask - fee >= 0.02.
CLV mark = PM US side mid at the first observation >= decision + 300 s (also +60 s reported as a diagnostic).
Bar: family cross_venue 7 -> 9 trials, 0.05/9 = 5.6e-3. min_n 150. Pass = edgelab rule 5. Sealed once on dev pass.
Known weakness: is the :00 observation really the book at :00, or a snapshot lagging it? The live logger records the
REST bbo every ~1 s so it can be compared with the stored history for the same minute (validates the history).

## Results (dev only, 2026-11-06; sealed never pulled)
Dev build: 700 sampled games (mlb 614, cfb 36, nhl 20, nba 15, nfl 15) -> 19,892 Kalshi jumps in 698 games; PM US
history 146,745 minute obs in 700 games (history exists from May 2026 as expected).
Descriptive (`describe dev`): mean |Kalshi jump| 5.0c; PM US side ask move vs last obs <= t: +0.90c at 3-10 s
(n 2,370), +0.72-0.75c at 10-20/20-30/30-60/60-120 s -> PM US moves ~0.7-0.9c and no further; Kalshi jump price -
PM US ask is already -0.5c at 3-10 s -> PM US is NOT stale at its ask (most 4c Kalshi trade "jumps" in thin MLB
books look like bid/ask bounce that PM US never follows -- a reading, not tested).
edgelab verbatim:
  pmus_lag_t1  cross_venue (9, 5.6e-03)  dead  dev  2341 -3.60% [-8.80%,+1.62%]   p 9.1e-01 CLV -0.0044  dead (dev fail): 5a p>=bar,5b half<=0,5c conc>30%,5e CLV<0/null
  pmus_lag_t2  cross_venue (9, 5.6e-03)  dead  dev   481 +1.01% [-15.73%,+20.99%] p 4.8e-01 CLV +0.0085  dead (dev fail): 5a p>=bar,5b half<=0,5c conc>30%
Verdict: the "buy Polymarket toward Kalshi" mirror lead does not survive Polymarket US's real ask -> DEAD (dev).
Caveats: dev is 88% MLB; the 3-10 s window relies on the history's :00 observation being the live book at :00
(unverified; the live logger compares). Live logger launched for 2026-11-07 00:05Z-02:05Z (descriptive only;
`python scripts/pmus_lag_logger.py report`).
