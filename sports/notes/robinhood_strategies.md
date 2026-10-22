# Robinhood strategies: public summary with no account data (2026-10-22)

Account-specific numbers, positions and tickets are in the gitignored `notes/robinhood_strategies_private.md`.
Evidence script: `scripts/rh_strategies_bt.py`. DEV = up to 2018 (used for design); OOS = 2019 onward (reported only, not tuned on).
Financing is assumed at T-bill + 2%.

Ranked by risk-adjusted benefit:
1. **Deleverage a single-sector book to about 1.5-1.8x or less.**
   - Lever once and hold SMH: P(hitting 30% maintenance within 1y) is 8% at 1.5x, 22% at 2x and 92% at 3.3x (DEV). OOS: 0%, 21% and 85%.
   - Daily-rebalanced SMH at 2x or more wiped out in DEV.
   - With about 49% book vol, expected log growth turns negative above about 2x even if the book returns 40%/yr.
2. **Choose tax lots deliberately.** Sell the highest-cost lots first and exit loss names completely. Long calls bought within 30 days count as replacement shares, and so do DRIP and IRA purchases. Turn DRIP off and make no IRA rebuys for 31 days.
3. **Index core plus sector satellite.**

   | | SMH | QQQ |
   |---|---|---|
   | CAGR, DEV 2001-18 | -0.1% | 6.4% |
   | Max drawdown, DEV | -86% | -83% |
   | CAGR, OOS 2019-26 | 38.6% | 22.3% |

   The sector bet depends on the regime, so the default core is buy-and-hold index. All 14 ETF timing rules and the pairs trades failed.
4. **Vol-targeted exposure, capped at 1.3x.** It cuts max drawdown about 20-25% at a similar Sharpe (SMH DEV -86% -> -67%, QQQ -83% -> -68%). This is risk control, not alpha.
5. **Use the Roth for the core and for frequent trading.** Hold taxable winners until they turn long-term.
6. **Deep-ITM LEAPS** cap the loss and cannot be margin-called. Untested; use them only after deleveraging.
7. **Earnings straddles.** Buy them before earnings and exit before the print (Gao-Xing-Zhang 2018). Holding through the print loses on average (earnings premium). Untested by us, so keep each one at 1% of equity or less.
8. **Covered calls and cash-secured puts: no.**
   - QYLD vs QQQ CAGR: 5.7% vs 13.8% (DEV 2014-18) and 10.4% vs 22.3% (OOS).
   - XYLD vs SPY: similar.
   - ^PUT vs SPY Sharpe: 0.48 vs 0.37 (DEV) but 0.44 vs 0.78 (OOS).
   - Cash-secured puts on margin are just more leverage.
9. **Thesis tilt.** No backtested edge, so express it through which names you keep, not by adding exposure.
