# RESEARCH QUEUE -- keep agents running (user 2026-11-15: "you should have been trying to confirm and find more
# edges in the market through continually running agents")

Rule for the lead: keep 3-4 research agents running at all times while the user is active. When one finishes: merge
its inbox rows (AGENT_PROTOCOL.md), ingest LEARNINGS into the vault, then launch the next item below. Re-prioritise by
evidence: anything with CLV/ROI signal jumps the queue; game-day prep comes first on game days. Agents never place
orders; money changes stay with the user.

## Running (2026-11-15)
- news_lag -- NFL inactives (~90 min pre-kick) / NBA late scratches vs Kalshi repricing speed (Sunday logger)
- nhl_opener_model -- de-biased NHL totals model vs Kalshi (lists ~70 h before Pinnacle), goalie-confirmation timing
- sharp_feed_research -- find a real-time (not 15-min cached) sharp odds feed + cost

## Done 11-15: gameday_ticket (shard-0 cash blocker), critic_e4_step2 (keep 24 ct), e4_ops_diag (fixes deployed),
## maker_fairfill (maker adversely selected even with model direction -> opener leads capped by execution cost)

## Next up (priority order)
1. K-NFL/K50-NFL/K1 forward scoring after Sunday's games (judge on CLV vs Kalshi's own strike, not ROI).
2. Information edges (the only lever left given execution cost): late news timing (news_lag), goalie confirmations
   (NHL), weather updates for NFL totals between model runs, official injury designations.
3. Live in-game Kalshi vs a real-time sharp feed (depends on sharp_feed_research cost).
4. Hurricane-landfall NO (LX2 climate +3.0%, 36 events, 0 losses) -- sealed 2025 needs the user's yes.
5. X1 maker forward paper (17 cities; user call to go live).
6. Fix gap_logger team matcher (West Ham/West Brom) + log raw Pinnacle odds/age (KS12 proposal).
7. NFLF re-arm prerequisites: deploy order_guard fix (cec89a0, user gate) + replay test on 11-08/09 fills.
8. Wire scripts/kx_maker_paper.py as a free shadow judged on filled CLV (low priority).

## From series watch (auto, scripts/series_watch.py; read-only census leads, untested)
- 2026-11-15 **KXSILVEREVENT** Silver after economic events -- score 10/10, macro, custom, 371 mkts, vol24 1653 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXGOLDEVENT** Gold after economic events -- score 10/10, macro, custom, 371 mkts, vol24 93 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXCOPPEREVENT** Copper after economic events -- score 10/10, macro, custom, 371 mkts, vol24 12 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXAAAGASCD*** Alabama gas price (+49 siblings) -- score 8/10, transport/energy, daily, 5 mkts, vol24 3088 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXBRENTD** Brent Oil Daily -- score 9/10, markets/crypto, daily, 20 mkts, vol24 1687 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXTTFMON** Dutch TTF gas monthly -- score 9/10, transport/energy, monthly, 40 mkts, vol24 7 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXERCOTHOUM** ERCOT Houston Hub monthly power price -- score 9/10, transport/energy, monthly, 7 mkts, vol24 0 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXERCOTNORTHM** ERCOT North Hub monthly power price -- score 9/10, transport/energy, monthly, 7 mkts, vol24 0 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXGOLDD** Gold Daily -- score 8/10, markets/crypto, daily, 40 mkts, vol24 22764 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXBTCEVENT** Bitcoin after economic events -- score 8/10, macro, custom, 136 mkts, vol24 10027 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXAUDUSDEVENT** AUD/USD after economic events -- score 8/10, macro, custom, 254 mkts, vol24 4564 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXSOLEVENT** Solana after economic events -- score 8/10, macro, custom, 136 mkts, vol24 3462 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXETHEVENT** Ethereum after economic events -- score 8/10, macro, custom, 136 mkts, vol24 2194 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXUSDCADEVENT** USD/CAD after economic events -- score 8/10, macro, custom, 212 mkts, vol24 1774 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXTEMPNYCHS** Hourly Directional NYC Temperature -- score 8/10, weather, hourly, 10 mkts, vol24 1563 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXNATGASD** Natural Gas Daily -- score 8/10, transport/energy, daily, 110 mkts, vol24 1080 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXARCTICICEMAX** Arctic sea ice max extent -- score 8/10, weather, custom, 11 mkts, vol24 1041 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXGBPUSDEVENT** GBP/USD after economic events -- score 8/10, macro, custom, 233 mkts, vol24 1000 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXUSDMXNEVENT** USD/MXN after economic events -- score 8/10, macro, custom, 233 mkts, vol24 15 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXUSDJPYEVENT** USD/JPY after economic events -- score 8/10, macro, custom, 254 mkts, vol24 14 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXBIGGESTQUAKEMONTH** Biggest earthquake in a given month -- score 8/10, geo/space, monthly, 9 mkts, vol24 5 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXEURUSDEVENT** EUR/USD after economic events -- score 8/10, macro, custom, 191 mkts, vol24 4 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KX10YRRATEEVENT** US 10Y Treasury yield after events -- score 8/10, macro, custom, 651 mkts, vol24 0 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KX2YRRATEEVENT** US 2Y Treasury yield after events -- score 8/10, macro, custom, 651 mkts, vol24 0 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KX30YRRATEEVENT** US 30Y Treasury yield after events -- score 8/10, macro, custom, 651 mkts, vol24 0 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KX5YRRATEEVENT** US 5Y Treasury yield after events -- score 8/10, macro, custom, 549 mkts, vol24 0 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXCOINBASE** Coinbase KPI -- score 8/10, markets/crypto, one_off, 8 mkts, vol24 0 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXERCOTWESTM** ERCOT West Hub monthly power price -- score 8/10, transport/energy, monthly, 7 mkts, vol24 0 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXHYPEEVENT** HYPE after economic events -- score 8/10, macro, custom, 136 mkts, vol24 0 (notes/series_watch/2026-11-15.md)
- 2026-11-15 **KXPERFORMROLE** will person perform role -- score 8/10, charts/media, one_off, 15 mkts, vol24 0 (notes/series_watch/2026-11-15.md)
