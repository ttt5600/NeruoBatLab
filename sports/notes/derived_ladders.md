# derived_ladders -- are Kalshi SECONDARY sports ladders mispriced in shape? (agent, 2026-11-19)

Hypothesis (from KFG1, notes/kalshi_fg_markets.md): main game lines are made off sharp books; derived ladders (team totals,
halves, player yards, corners, shots...) are priced less carefully, with systematic SHAPE errors (KFG1: 3+ FG mid .218 vs
.269 hit; placebo +5%). Do-not-redo: KXNFLFG (KFG1), KXNBAPTS/REB/AST (prop ladders paper), NFL/NBA/NHL main totals/winners.

Plan: (1) census of Kalshi sports series (public read-only GETs) -> derived ones with >= ~80 settled events + real volume;
(2) descriptive calibration of mid AND ask at T = start - 60 min by rung position / price bucket, event-clustered bootstrap,
count cells; (3) prereg best 1-2 BEFORE returns, edgelab register, dev only (sealed latest ~30% unopened).
Scripts/output: sports/results/derived_ladders/.

## Log
- 2026-11-19 census: results/derived_ladders/census.py over 698 candidate game-level series (regex over the 4,316 Sports
  series in series_sports.json, main GAME/SPREAD/TOTAL + futures excluded, KXNFLFG/KXNBAPTS/REB/AST excluded) -> census.json.
  Settled markets from /markets?status=settled + /historical/markets, capped at 8 pages each (flag `capped`).
- Timing: Kalshi occurrence_datetime - 3 h == ESPN start exactly (std 0) on the 67 ESPN-matched KXEPLCORNERS events;
  fetch_ev.py uses ESPN start, else occ-3h when >= 95% of the series' ESPN-matched events agree within 5 min; MLB from ticker HHMM.
- Quotes: event-level candlesticks endpoint (/series/S/events/E/candlesticks, 1 call per event, works for historical too),
  1-min over [T-6h, T], T = start - 60 min, last candle with end <= T. Output results/derived_ladders/ev/q_<S>.jsonl.
