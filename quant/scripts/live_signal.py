"""Today's target weights and the orders that would reach them. Dry run by default.

This is the seam between research and execution. It prints an order list; it
does not place anything and holds no credentials. Whatever executes those
orders -- a broker REST API, or an agent with a brokerage MCP connection -- is
downstream of this file and deliberately separate from it.

    ../.venv_quant/bin/python scripts/live_signal.py
    ../.venv_quant/bin/python scripts/live_signal.py --strategy inverse_vol --equity 25000
    ../.venv_quant/bin/python scripts/live_signal.py --positions "SPY=12,TLT=40"

Run it after the close. Weights are computed from the most recent completed
daily bar, which is the same convention the backtest used -- signals from the
close of day t are executed into day t+1. Running this intraday and filling
immediately is the close-to-close leak the engine exists to prevent, and it
would make live results diverge from the backtest.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant import strategies as S
from quant.data import load_prices
from quant.execution import Account, RebalancePolicy, plan_rebalance

UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ"]

BUILDERS = {
    "buy_and_hold": lambda px: S.buy_and_hold(px),
    "inverse_vol": lambda px: S.inverse_vol(px, lookback=60),
    "sixty_forty": lambda px: S.sixty_forty(px),
    "tsmom_12m": lambda px: S.time_series_momentum(px, 252, long_only=True),
    "xsmom_12_1": lambda px: S.cross_sectional_momentum(px, 252, 21, n_long=2),
}


def parse_positions(s: str) -> dict[str, float]:
    if not s:
        return {}
    out = {}
    for part in s.split(","):
        sym, _, qty = part.partition("=")
        out[sym.strip().upper()] = float(qty)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", default="inverse_vol", choices=sorted(BUILDERS))
    ap.add_argument("--equity", type=float, default=10_000.0,
                    help="account equity if --positions is empty")
    ap.add_argument("--positions", default="", help='e.g. "SPY=12,TLT=40"')
    ap.add_argument("--cash", type=float, default=None)
    ap.add_argument("--band", type=float, default=0.02)
    ap.add_argument("--min-notional", type=float, default=50.0)
    ap.add_argument("--max-order-notional", type=float, default=10_000.0)
    args = ap.parse_args()

    prices = load_prices(UNIVERSE, "2015-01-01", "2100-01-01").dropna(how="any")
    asof = prices.index[-1]
    last = {s: float(prices[s].iloc[-1]) for s in prices.columns}

    weights = BUILDERS[args.strategy](prices)
    target = {s: float(w) for s, w in weights.iloc[-1].items() if abs(w) > 1e-9}

    positions = parse_positions(args.positions)
    cash = args.cash if args.cash is not None else (
        args.equity - sum(q * last[s] for s, q in positions.items())
    )
    account = Account(cash=cash, positions=positions)
    equity = account.equity(last)

    print(f"strategy   {args.strategy}")
    print(f"as of      {asof.date()}  (most recent completed daily bar)")
    print(f"equity     ${equity:,.2f}   cash ${account.cash:,.2f}")
    print(f"policy     band={args.band:.3f}  min=${args.min_notional:,.0f}  "
          f"max_order=${args.max_order_notional:,.0f}")

    current = account.weights(last) if equity > 0 else {}
    print("\ntarget weights vs current")
    print(f"  {'sym':<6}{'target':>9}{'current':>10}{'drift':>9}")
    for sym in sorted(set(target) | set(current)):
        t, c = target.get(sym, 0.0), current.get(sym, 0.0)
        print(f"  {sym:<6}{t:>9.4f}{c:>10.4f}{t - c:>+9.4f}")

    policy = RebalancePolicy(
        band=args.band,
        min_notional=args.min_notional,
        max_order_notional=args.max_order_notional,
    )
    orders = plan_rebalance(account, target, last, policy)

    print(f"\nPLAN ({len(orders)} order{'s' if len(orders) != 1 else ''}) "
          f"-- DRY RUN, nothing was placed")
    if not orders:
        print("  nothing to do: every position is inside the no-trade band.")
    else:
        for o in orders:
            print(f"  {o!r}   {o.reason}")
        gross = sum(o.est_notional for o in orders)
        print(f"\n  traded notional ${gross:,.2f} "
              f"({gross / equity:.2%} of equity)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
