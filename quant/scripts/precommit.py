"""Seal a go/no-go rule before the evidence exists, then check against it.

    python scripts/precommit.py seal  --strategy sixty_forty --decision-date 2027-03-15
    python scripts/precommit.py check
    python scripts/precommit.py show

`check` is mechanical: it reports GO or NO-GO from the sealed criteria and does
not weigh anything. That is the point -- the interpretation was fixed in
advance, when nobody had a position to defend.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant import strategies as S
from quant.backtest import CostModel
from quant.data import load_prices
from quant.paper import PaperAccount
from quant.precommit import DecisionRule, evaluate

ROOT = Path(__file__).resolve().parents[1]
RULE = ROOT / "results" / "decision_rule.json"
STATE = ROOT / "results" / "paper_account.json"
UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ"]

BUILDERS = {
    "buy_and_hold": lambda px: S.buy_and_hold(px),
    "inverse_vol": lambda px: S.inverse_vol(px, lookback=60),
    "sixty_forty": lambda px: S.sixty_forty(px),
    "tsmom_12m": lambda px: S.time_series_momentum(px, 252, long_only=True),
}


def cmd_seal(args) -> int:
    if RULE.exists() and not args.force:
        print(f"{RULE} already sealed. Use `supersede` to revise it on the record.")
        return 1
    rule = DecisionRule(
        strategy=args.strategy,
        decision_date=args.decision_date,
        min_trading_days=args.min_days,
        min_paper_sharpe=args.min_sharpe,
        max_drawdown_abort=args.max_dd,
        notes=args.notes,
    ).seal_now()
    RULE.parent.mkdir(parents=True, exist_ok=True)
    rule.save(RULE)
    print(f"sealed {RULE}")
    print(f"  seal {rule.seal}  at {rule.sealed_at}")
    print(f"  strategy {rule.strategy}, decide on/after {rule.decision_date}")
    print(f"  needs >= {rule.min_trading_days} trading days")
    print(f"  aborts if drawdown breaches {rule.max_drawdown_abort:.1%}")
    return 0


def cmd_supersede(args) -> int:
    old = DecisionRule.load(RULE)
    changes = {}
    if args.decision_date:
        changes["decision_date"] = args.decision_date
    if args.max_dd is not None:
        changes["max_drawdown_abort"] = args.max_dd
    if args.min_sharpe is not None:
        changes["min_paper_sharpe"] = args.min_sharpe
    if args.notes:
        changes["notes"] = args.notes
    if not changes:
        print("nothing to change.")
        return 1
    new = old.supersede(**changes)
    new.save(RULE)
    print(f"superseded. {len(new.superseded)} prior rule(s) kept on the record.")
    for k, v in changes.items():
        print(f"  {k}: {getattr(old, k)!r} -> {v!r}")
    print("\nA revision is legitimate; a quiet one is not. This is now visible.")
    return 0


def cmd_show(args) -> int:
    rule = DecisionRule.load(RULE)
    print(f"strategy            {rule.strategy}")
    print(f"decision date       {rule.decision_date}")
    print(f"min trading days    {rule.min_trading_days}")
    print(f"min paper Sharpe    {rule.min_paper_sharpe:.3f}")
    print(f"abort drawdown      {rule.max_drawdown_abort:.2%}")
    print(f"min correlation     {rule.min_divergence_correlation:.2f}")
    print(f"max tracking gap    {rule.max_abs_mean_daily_gap_bps:.2f} bps")
    print(f"sealed at           {rule.sealed_at}")
    print(f"seal                {rule.seal}  ({'INTACT' if rule.intact else 'BROKEN'})")
    if rule.superseded:
        print(f"superseded rules    {len(rule.superseded)}")
    if rule.notes:
        print(f"notes               {rule.notes}")
    return 0


def cmd_check(args) -> int:
    rule = DecisionRule.load(RULE)
    acct = PaperAccount.load(STATE)
    px = load_prices(UNIVERSE, "2015-01-01", "2100-01-01").dropna(how="any")
    out = evaluate(rule, acct, px, BUILDERS[acct.strategy], CostModel(), today=args.today)

    print(f"rule seal {rule.seal}   strategy {rule.strategy}   journal {len(acct.entries)} bars\n")
    for c in out["checks"]:
        print(f"  [{'PASS' if c.passed else 'FAIL'}]  {c.name:<26} {c.detail}")
    print(f"\n  VERDICT: {out['verdict']}")
    if out["failures"]:
        print(f"  blocked by: {', '.join(out['failures'])}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("seal")
    s.add_argument("--strategy", default="sixty_forty", choices=sorted(BUILDERS))
    s.add_argument("--decision-date", required=True)
    s.add_argument("--min-days", type=int, default=126)
    s.add_argument("--min-sharpe", type=float, default=0.30)
    s.add_argument("--max-dd", type=float, default=-0.15)
    s.add_argument("--notes", default="")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_seal)

    sp = sub.add_parser("supersede")
    sp.add_argument("--decision-date")
    sp.add_argument("--max-dd", type=float)
    sp.add_argument("--min-sharpe", type=float)
    sp.add_argument("--notes")
    sp.set_defaults(fn=cmd_supersede)

    sub.add_parser("show").set_defaults(fn=cmd_show)

    c = sub.add_parser("check")
    c.add_argument("--today", default=None, help="override for testing")
    c.set_defaults(fn=cmd_check)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
