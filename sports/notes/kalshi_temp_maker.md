# L24: E1/E2 picks executed as a MAKER instead of a taker (DEV only), 2026-10-22

Script: `scripts/kalshi_temp_maker.py` (pre-registration in the docstring, written before any tape was pulled).
Data: `data/kalshi_temp_maker_{picks,trades,quotes_t10}.parquet` (public `/historical/trades` tape, 159,077 trades on
1,700 of 1,713 pick tickers; T+10h hourly quotes on all 1,713). Ledger: 9 trials, dataset `kalshi_temp_maker`, bar p < 0.0056.

**This tests EXECUTION, not the rule.** The picks are exactly the frozen rule's DEV picks: E1 = `kalshi_reject_safe_check`
on kalshi_temp dev quarters (rule fires 2024-03..2025-06, 1,269 picks), X1 = `wx_expand.picks` on highs before 2026-03-01
(2026-01-14..02-28, 444 picks). Sealed periods were not loaded. **Replaying the chosen execution on the sealed periods
(E1 2025Q3+, X1 2026-03+) is a separate one-shot that needs the user's explicit yes.**

## Fee schedule (checked 10-22)
- `GET /series/KXHIGHNY|KXHIGHTATL|KXHIGHAUS` -> `fee_type: quadratic`, `fee_multiplier: 1`. API docs (get-series): `quadratic`
  = General Trading Fees Table; maker fees exist only under `quadratic_with_maker_fees` / `..._combo_...`. So **makers pay 0
  on these series today**. `/series/fee_changes?show_historical=true` lists no KXHIGH change. Whether weather makers paid
  anything in 2024-25 (secondary sources mention a flat maker fee in some periods) is NOT verified -> sensitivity row
  (maker fee 0.0175·q·(1-q)); every passing variant stays positive under it.
- Fee rounding (docs fee_rounding): direct members settle at $0.0001, so the taker baseline uses the unrounded 0.07·p·(1-p).
  A maker fill with fee 0 also removes any cent-rounding risk at 1 contract/order.

## Results (pooled E1+X1 dev picks; d_units = maker minus taker over ALL picks, unfilled maker = 0; bootstrap by date)
Taker baseline: +30.46 units on 1,713 bets, ROI +1.78%, loss rate 4.09%, avg NO ask 0.941.

| variant | fill rate | bets | ROI/$ incl fee | units | d_units [95% CI] | p | loss rate filled (all 4.09%) | verdict |
|---|---|---|---|---|---|---|---|---|
| M1 ask-1c, N=1 | 80.2% | 1374 | +2.60% | +35.7 | +5.3 [+0.8, +9.4] | .0125 | 5.09% | fail (bar) |
| M1 ask-1c, N=10 | 76.2% | 1306 | +2.44% | +31.9 | +1.4 [-3.2, +5.6] | .27 | 5.36% | fail |
| M2 ask-2c, N=1 | 70.9% | 1214 | +3.57% | +43.3 | +12.9 [+7.9, +17.3] | <.0003 | 5.77% | PASS |
| **M2 ask-2c, N=10** | 67.3% | 1152 | +3.35% | +38.6 | +8.1 [+3.2, +12.9] | .0010 | 6.08% | **PASS** |
| M3 mid, N=1 | 76.5% | 1311 | +3.45% | +45.2 | +14.7 [+9.5, +19.5] | <.0003 | 5.34% | PASS |
| **M3 mid, N=10** | 72.3% | 1239 | +3.22% | +39.9 | +9.5 [+4.2, +14.4] | .0003 | 5.65% | **PASS** |
| F1 ask-1c 10h then take | 61.8% | 1266 | +2.54% | +32.1 | +1.7 [-7.3, +10.6] | .37 | 5.20% | fail |
| F2 ask-2c 10h then take | 51.0% | 1161 | +3.21% | +37.3 | +6.8 [-3.2, +17.0] | .091 | 5.95% | fail |
| F3 mid 10h then take | 57.7% | 1221 | +3.21% | +39.2 | +8.7 [-1.3, +19.0] | .047 | 5.26% | fail |

Per cohort (N=10): M2 E1 d +4.99 (p .019), X1 d +3.15 (p .0003); M3 E1 d +6.42 (p .006), X1 d +3.03 (p .0003).
Taker units/month: E1 1.57, X1 3.66. M3_N10 units/month: E1 1.97, X1 5.70 (at 1 unit per pick).
With a 0.0175 maker fee: M2_N10 d +6.25, M3_N10 d +7.53.

## Mechanism (why it works and what it costs)
- **Adverse selection is total on losers:** every losing pick filled (unfilled loss rate 0.00%) -- a bracket that hits
  YES drives NO to 0, through any bid. Filled loss rate rises 4.1% -> 5.7-6.1%. Losers fill fast (median 1.3 h after
  22:00): an overnight forecast update moves the market straight through the bid.
- **The price improvement pays for it:** ~2.0c on a ~0.94 NO = ~2.1% of stake on every filled winner (1,082-1,169 of
  1,643 winners filled), vs ~6% profit forgone on each missed winner. Net positive in both cohorts.
- **ask-1c is not enough** (M1_N10 d +1.4, p .27): 1c of improvement barely covers the extra losers.
- **Hybrid "rest 10 h then take" is WORSE than resting to close** (all F fail): the fallback takes the winners that did
  not dip, at the T+10h ask, and loses the late-day maker fills.
- **Regime warning:** by quarter (M2_N10) 2024Q1 -1.9, Q2 -2.5, Q3 -0.9, Q4 +5.2, 2025Q1 +2.1, Q2 +3.0, 2026Q1 +3.2.
  The first three quarters (thin early markets, few picks) were negative; the last four positive. Not a trend claim
  (7 points), but the gain is not uniform.

## Caveats (not hidden)
- Quotes at T are the hourly candle's close bid/ask, not a live book snapshot; the maker price is defined off them.
- Fill model: a print at no_price <= q in (T, close]. N=10 size-through is the queue proxy; at 1 contract/order we are
  small, but if many others rest at q our place in line is unknown. The tape shows prints, not cancellations.
- Counterfactual: assumes our resting bid would not change anyone else's trading (fine at 1-10 contracts).
- Liquidity-incentive rewards (Kalshi LIP on KXHIGH brackets) are NOT modelled; they could only add.
- Dev only. The variant to deploy was picked among 4 passing variants; M2 and M3 are within noise of each other.

## Recommendation
Switch E1/E2 execution from IOC-at-ask to a **resting post-only NO bid at M3 (NO mid floored to the cent, capped at
NO ask - 1c; equals ask-1c when the spread is 1-2c, ask-2c at 3-4c)**, resting until market close, **no taker
fallback**. M2 (ask-2c) is an equally defensible choice. Before trusting it with more than $1 units:
(1) the user's yes for the one-shot sealed replay (same script, sealed rows, one variant, frozen now = M3_N10), and/or
(2) a forward A/B on the live $1 units (pick-hash parity: even -> maker, odd -> taker) for 4+ weeks.

## Change spec for scripts/kalshi_exec.py (NOT applied; the deployed ~/.claude/quant-kalshi is untouched)
1. `config()`: add `"exec_style": e.get("KALSHI_EXEC_STYLE", "taker")` (values `taker` | `maker`), default taker so
   nothing changes until the user edits .env; optional `KALSHI_AB=1` -> maker iff `int(sha1(ticker)[:8],16) % 2 == 0`.
2. `decide()`: when maker, read the book's NO best bid too (`book_no` currently returns ask + depth; add the best NO bid
   = 1 - best YES ask). Price `q = min(floor_cent(1 - (yes_bid + yes_ask)/2), no_ask - 0.01)`; skip if `q <= 0` or
   `no_ask - q > 0.05`. Size with `n * q` (maker fee 0 on fee_type quadratic; re-check `series.fee_type` each run and
   fall back to taker sizing/fee if it is not `quadratic`). Keep all existing guards (rule_holds, stale pick, slippage,
   caps, min_close). The depth cap (0.5 x depth within 1c of the ask) does not apply to a resting bid.
3. `place()`: body `{"ticker": t, "side": "ask", "count": n, "price": f"{1 - q:.4f}", "time_in_force":
   "good_till_canceled", "post_only": True, "expiration_ts": <market close_time as unix s>,
   "self_trade_prevention_type": "taker_at_cross", "client_order_id": coid}` (selling YES at 1-q == buying NO at q;
   the same shape smoke() already uses at kalshi_exec.py:854-855). A post_only reject (would cross) -> record
   `status: "rejected_post_only"`, no retry, no taker fallback. A 201 with fill_count 0 -> `status: "resting"`,
   `cost = n * q` (committed for the daily/open caps while resting).
4. `reconcile()`: the stale-order canceller (age >= KALSHI_STALE_ORDER_MIN) must SKIP orders whose journal row has
   `exec_style == "maker"`; cancel those only when the market is no longer active or after close. Keep updating
   resting rows via `fills_from_order` (already reads maker_fill_cost_dollars / maker_fees_dollars).
5. Fill-convention check: for maker fills compare `avg_no` to `q` (not the ask); `|avg_no - q| > 0.02` -> KILL as today.
6. `committed()` / caps: a `resting` row commits `n * q`; on cancel/expiry with 0 fills -> `no_fill`, cost 0;
   partial -> filled count only. `settle()` already uses `held()` and `committed()`.
7. Status/report: show fills by style (maker vs taker), fill rate, and filled loss rate vs paper.
8. Tests (tests/test_kalshi_exec.py): maker body shape; resting order not cancelled by the stale sweep; resting cost
   counted in caps; post_only reject path; partial maker fill; A/B parity split.
