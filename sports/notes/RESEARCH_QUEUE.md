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
