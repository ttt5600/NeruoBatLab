# RESEARCH QUEUE -- keep agents running (user 2026-11-15: "you should have been trying to confirm and find more
# edges in the market through continually running agents")

Rule for the lead: keep 3-4 research agents running at all times while the user is active. When one finishes: merge
its inbox rows (AGENT_PROTOCOL.md), ingest LEARNINGS into the vault, then launch the next item below. Re-prioritise by
evidence: anything with CLV/ROI signal jumps the queue; game-day prep comes first on game days. Agents never place
orders; money changes stay with the user.

## Running (2026-11-15)
- gameday_ticket -- NFL Sunday + NBA slate readiness (every firing strategy, pipelines verified, ticket file)
- maker_fairfill -- model-directed maker at Kalshi's 50c strike; trade-through fills, filled vs unfilled (the cost lever)
- news_lag -- NFL inactives (~90 min pre-kick) / NBA late scratches vs Kalshi repricing speed (information timing)
- critic_e4_step2 -- adversarial check of the E4 24-ct sizing evidence (money is at stake)

## Next up (priority order)
1. K-NFL/K50-NFL forward scoring after Sunday's games; L32b-K1 forward check (n vs 100).
2. NHL totals opener model (Kalshi lists NHL ~70 h vs Pinnacle ~32 h; L34 model dead -> new de-biased model).
3. Live in-game Kalshi vs a real-time sharp feed (needs a non-cached source; KS12 showed free Pinnacle is ~15 min stale).
4. Hurricane-landfall NO (LX2 climate +3.0%, 36 events, 0 losses) -- sealed 2025 needs the user's yes.
5. X1 maker forward paper (17 cities; user call to go live).
6. Fix gap_logger team matcher (West Ham/West Brom) + log raw Pinnacle odds/age (KS12 proposal).
7. NFLF re-arm prerequisites: deploy order_guard fix (cec89a0, user gate) + replay test on 11-08/09 fills.

## Done today and earlier: see EDGES_AND_AGENTS.md wave lines and LEARNINGS_2026-11.md.
