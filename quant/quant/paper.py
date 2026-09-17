"""Persistent forward paper trading, and the divergence check that justifies it.

A backtest is a claim about a strategy. Paper trading forward on real prices is
the first evidence for that claim that the strategy's own author did not
construct. It is the only genuinely out-of-sample record available before money
moves, and it costs nothing but time.

The important output is not the paper P&L -- a few months of it says almost
nothing about skill, and reading it as though it does is the same error as
trusting a 22-config sweep. The important output is :func:`divergence`: does the
live path reproduce what the backtest engine says the same strategy earned over
the same days? A gap means the live system and the research system disagree
about something, and every source of that gap is a bug or an unmodelled cost.
Finding it now is free. Finding it after funding the account is not.

State is a single JSON file. Stepping is idempotent on the bar date, so a cron
that fires twice, or a manual re-run, cannot double-trade.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .backtest import CostModel, run_backtest
from .execution import Account, GuardrailError, PaperBroker, RebalancePolicy, plan_rebalance
from .stats import sharpe


@dataclass
class JournalEntry:
    bar_date: str
    recorded_at: str
    equity_before: float
    equity_after: float
    cash: float
    traded_notional: float
    costs: float
    n_orders: int
    orders: list[dict] = field(default_factory=list)
    positions: dict[str, float] = field(default_factory=dict)
    marks: dict[str, float] = field(default_factory=dict)
    target: dict[str, float] = field(default_factory=dict)


@dataclass
class PaperAccount:
    strategy: str
    started: str
    initial_cash: float
    cash: float
    positions: dict[str, float] = field(default_factory=dict)
    band: float = 0.02
    min_notional: float = 50.0
    max_order_notional: float = 10_000.0
    entries: list[JournalEntry] = field(default_factory=list)

    # ---------------- persistence ----------------
    @classmethod
    def create(cls, strategy: str, cash: float, policy: RebalancePolicy) -> "PaperAccount":
        return cls(
            strategy=strategy,
            started=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            initial_cash=float(cash),
            cash=float(cash),
            band=policy.band,
            min_notional=policy.min_notional,
            max_order_notional=policy.max_order_notional,
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        payload = asdict(self)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(path)  # atomic: a crash mid-write cannot corrupt the journal

    @classmethod
    def load(cls, path: Path) -> "PaperAccount":
        raw = json.loads(Path(path).read_text())
        entries = [JournalEntry(**e) for e in raw.pop("entries", [])]
        return cls(entries=entries, **raw)

    # ---------------- derived ----------------
    @property
    def policy(self) -> RebalancePolicy:
        return RebalancePolicy(
            band=self.band,
            min_notional=self.min_notional,
            max_order_notional=self.max_order_notional,
        )

    @property
    def processed_dates(self) -> set[str]:
        return {e.bar_date for e in self.entries}

    def equity_curve(self) -> pd.Series:
        if not self.entries:
            return pd.Series(dtype=float)
        return pd.Series(
            [e.equity_after for e in self.entries],
            index=pd.to_datetime([e.bar_date for e in self.entries]),
            name="paper_equity",
        ).sort_index()

    def returns(self) -> pd.Series:
        eq = self.equity_curve()
        return eq.pct_change().dropna() if len(eq) > 1 else pd.Series(dtype=float)


# --------------------------------------------------------------------------
def step(
    account: PaperAccount,
    prices: pd.DataFrame,
    build_weights,
    costs: CostModel = CostModel(),
    force: bool = False,
) -> JournalEntry | None:
    """Process the most recent bar in ``prices``. Returns None if already done.

    ``build_weights`` maps a price frame to a weight frame; only its last row is
    used. Passing the full history rather than a precomputed vector keeps the
    live path and the research path running the identical strategy code -- if
    they were allowed to diverge, :func:`divergence` would be measuring the
    wrong thing.
    """
    if prices.empty:
        raise GuardrailError("no price data")
    bar = prices.index[-1]
    bar_date = bar.strftime("%Y-%m-%d")
    if bar_date in account.processed_dates and not force:
        return None

    marks = {s: float(prices[s].iloc[-1]) for s in prices.columns}
    for s, p in marks.items():
        if not pd.notna(p) or p <= 0:
            raise GuardrailError(f"{s}: bad mark {p!r} on {bar_date}")

    weights = build_weights(prices)
    target = {s: float(w) for s, w in weights.iloc[-1].items() if abs(float(w)) > 1e-9}

    broker = PaperBroker(cash=account.cash, costs=costs, marks=marks,
                         positions=dict(account.positions))
    before = broker.account().equity(marks)

    orders = plan_rebalance(broker.account(), target, marks, account.policy)
    broker.submit(orders)

    after_acct = broker.account()
    entry = JournalEntry(
        bar_date=bar_date,
        recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        equity_before=before,
        equity_after=after_acct.equity(marks),
        cash=after_acct.cash,
        traded_notional=sum(o.est_notional for o in orders),
        costs=broker.total_costs,
        n_orders=len(orders),
        orders=[{"symbol": o.symbol, "side": o.side, "qty": o.qty,
                 "price": o.est_price, "reason": o.reason} for o in orders],
        positions=dict(after_acct.positions),
        marks=marks,
        target=target,
    )
    account.cash = after_acct.cash
    account.positions = dict(after_acct.positions)
    account.entries.append(entry)
    account.entries.sort(key=lambda e: e.bar_date)
    return entry


def backfill(
    account: PaperAccount,
    prices: pd.DataFrame,
    build_weights,
    costs: CostModel = CostModel(),
    start: str | None = None,
) -> int:
    """Replay history one bar at a time, as if it had been run live each day.

    Each step sees only prices up to that bar, so the strategy cannot use data
    it would not have had. This is a convenience for standing up a journal with
    some history -- it is simulation, not evidence, and :func:`divergence`
    reports on it exactly as it would on live days.
    """
    idx = prices.index
    lo = 0 if start is None else int(idx.searchsorted(pd.Timestamp(start)))
    n = 0
    for i in range(max(lo, 1), len(idx)):
        if step(account, prices.iloc[: i + 1], build_weights, costs) is not None:
            n += 1
    return n


# --------------------------------------------------------------------------
def divergence(
    account: PaperAccount,
    prices: pd.DataFrame,
    strategy_fn,
    costs: CostModel = CostModel(),
) -> dict:
    """Compare the paper path against the backtest over the same dates.

    The two differ for known, enumerable reasons -- chiefly the no-trade band,
    which the live path applies and the backtest does not, plus fractional-share
    rounding and the order-notional cap. A small gap is expected and is worth
    quantifying. A large one means the live system is not running the strategy
    you researched, and the number to trust is neither of them until you know
    which.
    """
    paper_ret = account.returns()
    if len(paper_ret) < 2:
        return {"status": "insufficient history", "n_days": int(len(paper_ret))}

    lo, hi = paper_ret.index[0], paper_ret.index[-1]
    px = prices.loc[: hi]
    bt = run_backtest(px, strategy_fn(px), costs=costs)
    bt_ret = bt.net_returns.loc[lo:hi]

    joined = pd.concat([paper_ret.rename("paper"), bt_ret.rename("backtest")],
                       axis=1).dropna()
    if len(joined) < 2:
        return {"status": "no overlapping days", "n_days": int(len(joined))}

    p, b = joined["paper"], joined["backtest"]
    gap = p - b
    return {
        "status": "ok",
        "n_days": int(len(joined)),
        "from": str(lo.date()),
        "to": str(hi.date()),
        "paper_total_return": float((1 + p).prod() - 1),
        "backtest_total_return": float((1 + b).prod() - 1),
        "total_gap": float((1 + p).prod() - (1 + b).prod()),
        "paper_sharpe": sharpe(p),
        "backtest_sharpe": sharpe(b),
        "correlation": float(p.corr(b)),
        "mean_daily_gap_bps": float(gap.mean() * 1e4),
        "max_abs_daily_gap_bps": float(gap.abs().max() * 1e4),
        "paper_ann_turnover": float(
            sum(e.traded_notional for e in account.entries)
            / max(account.equity_curve().mean(), 1e-9)
            / max(len(account.entries), 1) * 252
        ),
        "backtest_ann_turnover": float(bt.turnover.loc[lo:hi].mean() * 252),
    }
