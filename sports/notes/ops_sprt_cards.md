# ops_sprt_cards -- descriptive sequential-test panel in scorecards (ops agent, 2026-11-16)

DECISION SUPPORT ONLY. Not an edgelab verdict, not a GOAL.md promotion/kill. edgelab verdict logic and GOAL.md untouched.

## Question
Where does each near-certain weather stream stand on the calibrated forward SPRT of notes/bar_calibration_v2.md?
Streams: E4 orig7 live maker (exchange fills since 2026-10-29), X1 paper (kalshi_paper.jsonl cohort x1, since the
11-04 pause, tape-through fills), E4P-C / E4P-CA (data/kx_forward/e4p.jsonl), E4-0600 (data/kx_forward/e4_0600.jsonl).

## Test (fixed before running; copied from bar_calibration_v2 amendment text / sim2.py)
- Per fill i, cost c_i (fill price + fee per contract); H0 loss prob L0_i = 1 - c_i; H1 loss prob 0.70 x L0_i.
- LLR increment: loss -> log(0.7); win -> log((1 - 0.7 L0_i)/(1 - L0_i)). Sum, then divide by
  deff = 1 + (fills per night observed) x 0.031.
- Boundaries: alpha .005, beta .10 -> A = log(.9/.005) = +5.19 (accept H1), B = log(.1/.995) = -2.30 (reject).
- Fills with c >= .999 carry no H0 loss probability -> excluded and counted (warning).
- Status wording (pre-set): LLR >= A "H1 boundary crossed (ask user; no self-promotion)"; LLR <= B "reject (H0 boundary
  crossed)"; else "continue": fills-to-accept at the CURRENT per-fill LLR rate if > 0, and expected fills-to-accept if H1
  were true (E_H1 increment at the observed mean L0); "consistent with no edge" when losses >= break-even expectation
  (sum L0_i); fills-to-reject at the current rate if the rate is < 0.
- X1 fill model = e4_evidence T2 conservative: losers always fill; winners fill iff tape volume strictly through q
  (no_price < q) >= 10 ct between logged_at and close. Cost = q (maker fee 0). Executor skip rule (ask - q <= .05).
- E4 live: one row per ticker from kalshi_exec.jsonl (fold as results/e4_live_audit/build.py), mode live, exec_style
  maker, cohort orig7, order ts >= 2026-10-29, filled > 0, settled (journal settle rows; 111/111 reconciled with exchange
  settlements in e4_live_audit). cost = avg_no + fee/filled. Loss = result 'yes'.
- E4P / E4-0600: tt_fill settled primary bids, cost = q (same as those loggers' own judge()).

## Log
- Code: scripts/scorecards.py section "SPRT panel" (sprt_nc, e4_live_fills, x1_paper_fills, e4p_fills, e4_0600_fills,
  sprt_panel); new flag --no-tape. Panel sits between "Summary" and "Cards" in notes/SCORECARDS.md; it sets no card
  status. X1 tape volumes cached per ticker in results/scorecards/x1_tape_cache.json (public GETs, settled tickers only).
- Tests: tests/test_scorecards_sprt.py, 7 pass (boundaries, exact increments, deff, c >= .999 exclusion, placebo
  rho 1.0 -> H1 <= 2/60 and reject >= 40/60, planted edge rho .4 -> H1 >= 30/40, status wording, E4 fold filters/cost).
- Wrong turn: first run gave E4 "0 fills" because main()'s `hist` is lead_exec.jsonl, not kalshi_exec.jsonl; the panel
  now loads kalshi_exec.jsonl itself.
- Cross-checks: E4 29 settled orig7 maker fills, 1 loss (MIA .85, 26NOV02) = notes/e4_evidence.md live_fill (29 filled,
  1 loss). X1 94 settled picks since pause, 38 fills, 3 losses = e4_evidence T2 conservative (94 / 38 / 3).

## Panel, run 2026-11-16 (scorecards.py default run; snapshot results/scorecards/sprt_panel_2026-11-16.md)
| Stream | Fills (excl.) | Losses / break-even exp. | BE loss rate | Obs loss rate | LLR (deff) | Status |
|---|---|---|---|---|---|---|
| E4 orig7 live maker | 29 (0) | 1 / 1.18 | .0407 | .0345 | -0.01 (1.07) | continue; flat; if H1 true ~2,444 more fills to accept, if no edge ~966 to reject (1 filled order unsettled) |
| X1 paper (T2 cons.) | 38 (0) | 3 / 2.89 | .0761 | .0789 | +0.28 (1.13) | continue; consistent with no edge (losses >= break-even); if H1 true ~836 more fills to accept, if no edge ~439 to reject |
| E4P-C / E4P-CA / E4-0600 | 0 | - | - | - | 0 | no fills yet (first decisions 11-16) |
A = +5.19, B = -2.30. Both live streams sit at ~0 LLR: the data so far say nothing either way.

## Learnings
### At 29 fills the calibrated E4 SPRT is at LLR -0.01 of a +5.19 / -2.30 range
E4 orig7 live maker 29 fills, 1 loss vs 1.18 break-even expected (notes/SCORECARDS.md SPRT panel, 2026-11-16).
Expected ~2,400 more fills to accept if H1 (rho .7) is true, ~970 to reject if no edge: forward data are a decay guard,
not a promotion route, consistent with bar_calibration_v2.
### Per-fill information at 96-98c is tiny: KL(H1||H0) ~ .002 nats at loss .04
E_H1 LLR increment = .7 L0 ln .7 + (1 - .7 L0) ln((1 - .7 L0)/(1 - L0)) ~ .0021 at L0 .04, so ~2,500 fills to cross
A = 5.19 (Wald). Explains the ~2-year medians in bar_calibration_v2. Source: sprt_nc in scripts/scorecards.py.
### X1 post-pause paper is consistent with no edge but undecided
38 tape-through fills, 3 losses vs 2.89 break-even, LLR +0.28 (results/scorecards/sprt_panel_2026-11-16.md). X1 failed
dev, so under the proposed rule-6 amendment its forward data cannot promote it anyway.
### Scorecards' `hist` is lead_exec.jsonl; E4 lives in kalshi_exec.jsonl
Wiring an E4 panel to main()'s hist gave 0 fills. Any E4 scorecard must read ~/.claude/quant-kalshi/data/kalshi_exec.jsonl.
