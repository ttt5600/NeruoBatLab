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
- **SPLIT fixed before any outcome is looked at** (calib.py `split`): per series, events sorted by start; DEV = earliest 70%,
  SEALED = latest 30% (cut = start of event at index floor(0.7 n)). calib.py loads DEV only. fetch_ev.py pulls quotes for all
  events (one call per event, cheaper than re-fetching later), but sealed quotes are never joined to outcomes or scored.
- Descriptive pass (calib.py): valid quote = market open_time <= T, bid > 0, ask < 1, ask > bid. Cells = mid buckets (10)
  and, for strike ladders, relative rung (strike rank minus rank of the rung nearest .5, clipped +-4); cell kept if n >= 30
  and >= 15 events. Per cell: hit rate - mid (event-clustered bootstrap 95% CI), taker ROI YES at ask / NO at 1-bid incl.
  quadratic fee x series multiplier (10-lot, cent-rounded). Descriptive only; every cell counted for the multiplicity note.
- 11-19 lead message: prioritise STRATEGY_v2 derived candidates: soccer correct score (7 leagues), KXNFL1HFT, NFL win margin,
  NFL OT/tie, NBA win margin, KXNCAAMB1HWINNER; adopt gate = base-rate-shape placebo + per-rung calibration (no returns) first.
  Soccer 1H/2H/team totals were listed 11-18/19 -> forward paper only (not enough dev history).
- **Data limit found**: markets older than Kalshi's historical cutoff (/historical/cutoff market_settled_ts 2026-09-19) have NO
  pre-game candles unless they traded (e.g. KXNBA1HTOTAL-26FEB10LACHOU-98: first candle 01:52Z, tip 01:00Z), and the event
  candlestick endpoint returns empty for them. So usable pre-game quote history starts ~2026-09-19 for every series:
  MLB derived series (season ended) mostly drop out; NBA from 10-20, NFL/NCAAF/soccer ~2 months. fetch_ev.py now fetches only
  live-src (post-cutoff) events; the 70/30 split is defined over those events only (amended before any outcome was viewed).
- **AMENDMENT (11-19, after viewing only KXEPLCORNERS and KXNBA1HTOTAL full-dev tables, neither has a usable cell):**
  to avoid choosing and testing a rung class on the same rows, DEV is halved by start time: devA (first 50% of dev events)
  = discovery only (calib.py default part); devB (rest of dev) = the dev test of any pre-registered rule; SEALED (latest
  30%) untouched. KXNBA1HTOTAL full-dev looked at: 19 cells, only mid .5-.6 gap CI > 0 (+.083 [.004,.153], spread .095 =
  stale wide quotes); YES-at-ask ROI negative everywhere, NO ~0 on longshots. Treated as burned for discovery.
