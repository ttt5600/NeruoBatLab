"""Run and inspect a forward paper-trading journal. No money, real prices.

    python scripts/paper_run.py init   --strategy inverse_vol --cash 25000
    python scripts/paper_run.py backfill --since 2026-03-01
    python scripts/paper_run.py step                      # once per trading day, after 17:00 ET
    python scripts/paper_run.py report

``step`` is idempotent on the bar date, so running it twice in a day, or via a
cron that retries, cannot double-trade.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from quant import strategies as S
from quant.backtest import CostModel
from quant.data import load_prices
from quant.execution import RebalancePolicy
from quant.execution import GuardrailError
from quant.paper import PaperAccount, backfill, catch_up, divergence, step
from quant.stats import max_drawdown, sharpe

UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ"]
STATE = Path(__file__).resolve().parents[1] / "results" / "paper_account.json"
COSTS = CostModel()

BUILDERS = {
    "buy_and_hold": lambda px: S.buy_and_hold(px),
    "inverse_vol": lambda px: S.inverse_vol(px, lookback=60),
    "sixty_forty": lambda px: S.sixty_forty(px),
    "tsmom_12m": lambda px: S.time_series_momentum(px, 252, long_only=True),
}


def _prices() -> pd.DataFrame:
    return load_prices(UNIVERSE, "2015-01-01", "2100-01-01").dropna(how="any")


def cmd_init(args) -> int:
    if STATE.exists() and not args.force:
        print(f"{STATE} already exists. Use --force to overwrite.")
        return 1
    policy = RebalancePolicy(band=args.band, min_notional=args.min_notional,
                             max_order_notional=args.max_order_notional)
    acct = PaperAccount.create(args.strategy, args.cash, policy)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    acct.save(STATE)
    print(f"created {STATE}")
    print(f"  strategy {acct.strategy}   cash ${acct.cash:,.2f}   band {acct.band:.3f}")
    return 0


def cmd_step(args) -> int:
    acct = PaperAccount.load(STATE)
    px = _prices()
    if args.force:
        entries = [step(acct, px, BUILDERS[acct.strategy], COSTS, force=True)]
    else:
        try:
            entries = catch_up(acct, px, BUILDERS[acct.strategy], COSTS)
        except GuardrailError as e:
            # Non-zero so a scheduler surfaces it. Exiting 0 here is how a
            # stale cache went three weeks without anyone noticing.
            print(f"REFUSED: {e}", file=sys.stderr)
            return 2
    if not entries:
        print(f"bar {px.index[-1].date()} already processed; nothing to do.")
        return 0
    acct.save(STATE)
    for entry in entries:
        print(f"bar {entry.bar_date} [{entry.source}]: {entry.n_orders} order(s), "
              f"${entry.traded_notional:,.2f} traded, ${entry.costs:,.2f} cost")
        if entry.basis:
            print("  basis carry: " + ", ".join(f"{k} x{v:.5f}" for k, v in sorted(entry.basis.items())))
        for o in entry.orders:
            print(f"  {o['side'].upper():4s} {o['qty']:>10.4f} {o['symbol']:<5s} @ ~{o['price']:.2f}")
        print(f"  equity ${entry.equity_after:,.2f}")
    print(f"live days so far: {acct.live_days}")
    return 0


def cmd_backfill(args) -> int:
    acct = PaperAccount.load(STATE)
    px = _prices()
    n = backfill(acct, px, BUILDERS[acct.strategy], COSTS, start=args.since)
    acct.save(STATE)
    print(f"replayed {n} bars; journal now holds {len(acct.entries)}")
    return 0


def cmd_report(args) -> int:
    acct = PaperAccount.load(STATE)
    if not acct.entries:
        print("journal is empty; run `step` or `backfill` first.")
        return 1

    eq = acct.equity_curve()
    r = acct.returns()
    total = eq.iloc[-1] / acct.initial_cash - 1
    print(f"strategy   {acct.strategy}")
    print(f"period     {eq.index[0].date()} .. {eq.index[-1].date()}  ({len(eq)} bars)")
    print(f"equity     ${acct.initial_cash:,.2f} -> ${eq.iloc[-1]:,.2f}   ({total:+.2%})")
    if len(r) > 2:
        print(f"sharpe     {sharpe(r):.3f}      max drawdown {max_drawdown(r):.2%}")
    print(f"costs paid ${sum(e.costs for e in acct.entries):,.2f}")
    print(f"days traded {sum(1 for e in acct.entries if e.n_orders):d} of {len(acct.entries)}")

    print("\n-- divergence vs the backtest over the same days --")
    d = divergence(acct, _prices(), BUILDERS[acct.strategy], COSTS)
    if d["status"] != "ok":
        print(f"  {d['status']} ({d['n_days']} days)")
        return 0
    print(f"  paper    {d['paper_total_return']:+.4%}   Sharpe {d['paper_sharpe']:.3f}"
          f"   turnover {d['paper_ann_turnover']:.2f}x")
    print(f"  backtest {d['backtest_total_return']:+.4%}   Sharpe {d['backtest_sharpe']:.3f}"
          f"   turnover {d['backtest_ann_turnover']:.2f}x")
    print(f"  correlation {d['correlation']:.4f}   mean daily gap "
          f"{d['mean_daily_gap_bps']:+.2f} bps   worst {d['max_abs_daily_gap_bps']:.2f} bps")
    if abs(d["correlation"]) < 0.95:
        print("  WARNING: correlation below 0.95. The live path is not tracking the")
        print("  strategy you researched. Investigate before trusting either number.")
    else:
        print("  Tracking. Residual gap is the no-trade band plus rounding, as expected.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("init")
    i.add_argument("--strategy", default="inverse_vol", choices=sorted(BUILDERS))
    i.add_argument("--cash", type=float, default=25_000.0)
    i.add_argument("--band", type=float, default=0.02)
    i.add_argument("--min-notional", type=float, default=50.0)
    i.add_argument("--max-order-notional", type=float, default=10_000.0)
    i.add_argument("--force", action="store_true")
    i.set_defaults(fn=cmd_init)

    s = sub.add_parser("step")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_step)

    b = sub.add_parser("backfill")
    b.add_argument("--since", default=None)
    b.set_defaults(fn=cmd_backfill)

    r = sub.add_parser("report")
    r.set_defaults(fn=cmd_report)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
