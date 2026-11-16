## Sequential-test panel -- DECISION SUPPORT ONLY (not an edgelab verdict; sets no card status)

Calibrated near-certain forward SPRT from `sports/notes/bar_calibration_v2.md` (sim2.py): H0 loss prob = 1 - c (c = fill price + fee), H1 = 0.70 x (1 - c), alpha 0.005, beta 0.10; LLR / (1 + fills-per-night x 0.031); accept A +5.19, reject B -2.30. Under the proposed (not adopted) rule-6 amendment only sealed-passed trials may promote on this test; X1 failed dev, so its row is descriptive only. Fills at c >= .999 are excluded (no information). 'Current rate' = LLR per fill so far (noisy at small n). Notes: `sports/notes/ops_sprt_cards.md`.

| Stream | Source | Fills (excl.) | Losses / break-even expected | Break-even loss rate | Observed loss rate | LLR (deff) | Status |
|---|---|---|---|---|---|---|---|
| E4 orig7 live maker | kalshi_exec.jsonl exchange fills since 2026-10-29, settled | 29 (0) | 1 / 1.18 | 0.0407 | 0.0345 | -0.01 (1.07) | continue: ~4792 fills from reject at current rate; losses 1 below break-even 1.18; if H1 true ~2444 more fills to accept, if no edge ~966 to reject |
| | 1 filled orig7 maker orders not yet settled (excluded until settled) | | | | | | |
| X1 paper | kalshi_paper.jsonl cohort x1 since 11-04 pause, tape-through (T2 conservative) | 38 (0) | 3 / 2.89 | 0.0761 | 0.0789 | +0.28 (1.13) | continue: ~657 fills from accept at current rate; consistent with no edge (losses 3 >= break-even 2.89); if H1 true ~836 more fills to accept, if no edge ~439 to reject |
| | settled x1 picks since pause 94 (3 losers, all counted filled; 91 winners, 35 filled by tape-through) | | | | | | |
| E4P-C | kx_forward/e4p.jsonl sel_c, tt fills | 0 (0) | - | - | - | +0.00 | no fills yet |
| E4P-CA | kx_forward/e4p.jsonl sel_cagree, tt fills | 0 (0) | - | - | - | +0.00 | no fills yet |
| E4-0600 | kx_forward/e4_0600.jsonl cohort filled, tt fills | 0 (0) | - | - | - | +0.00 | no fills yet |
