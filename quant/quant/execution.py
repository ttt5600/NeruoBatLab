"""Turning target weights into orders, with the rails that keep a live account alive.

A backtest produces a weight vector. An account holds share counts. This module
is the reconciliation between them, and it is where the findings from
``scripts/run_demo.py`` become engineering constraints rather than prose:

* **The no-trade band exists because of ``meanrev_5d``.** That strategy earned a
  gross Sharpe of 0.565 and a net Sharpe of -0.021 at 134.9x annual turnover.
  Rebalancing to the exact target every day is how a real edge becomes a losing
  account. ``RebalancePolicy.band`` refuses to trade drift smaller than a
  threshold, which is the single highest-leverage cost control available.

* **Dry run is the default.** ``PaperBroker`` is the reference implementation and
  a live adapter must be passed explicitly. Nothing here places a real order.

* **Guardrails reject rather than clamp.** A NaN weight or a stale price is a bug
  upstream; silently coercing it to something tradeable is how bugs reach the
  market. Every rejection names what failed.

This module deliberately contains no broker credentials, no network calls, and
no live adapter. Authentication belongs to the user, in their own environment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from .backtest import CostModel


class GuardrailError(ValueError):
    """A rebalance was refused. The message names the specific violation."""


@dataclass(frozen=True)
class Order:
    symbol: str
    side: str  # "buy" | "sell"
    qty: float
    est_price: float
    reason: str = ""

    @property
    def est_notional(self) -> float:
        return self.qty * self.est_price

    def __repr__(self) -> str:
        return (f"{self.side.upper():4s} {self.qty:>10.4f} {self.symbol:<6s} "
                f"@ ~{self.est_price:>9.2f}  (${self.est_notional:>11,.2f})")


@dataclass
class Account:
    cash: float
    positions: dict[str, float] = field(default_factory=dict)  # symbol -> shares

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(q * prices[s] for s, q in self.positions.items() if q)

    def weights(self, prices: dict[str, float]) -> dict[str, float]:
        eq = self.equity(prices)
        if eq <= 0:
            raise GuardrailError(f"account equity is {eq:,.2f}; cannot form weights")
        return {s: q * prices[s] / eq for s, q in self.positions.items() if q}


@dataclass
class RebalancePolicy:
    """Constraints applied to every rebalance. Defaults are conservative.

    ``band`` is the important one. At 0.02, a position whose weight has drifted
    less than two percentage points from target is left alone. This trades a
    little tracking error for a large reduction in turnover, and turnover is
    what the cost model shows eating edges alive.
    """

    band: float = 0.02
    min_notional: float = 50.0
    max_order_notional: float = 10_000.0
    max_gross_leverage: float = 1.0
    allow_short: bool = False
    allow_fractional: bool = True


class Broker(Protocol):
    def account(self) -> Account: ...
    def prices(self, symbols: list[str]) -> dict[str, float]: ...
    def submit(self, orders: list[Order]) -> list[Order]: ...


# --------------------------------------------------------------------------
def _validate(target: dict[str, float], prices: dict[str, float],
              policy: RebalancePolicy) -> None:
    for sym, w in target.items():
        if not math.isfinite(w):
            raise GuardrailError(f"{sym}: target weight is {w!r}")
        if w < 0 and not policy.allow_short:
            raise GuardrailError(f"{sym}: short weight {w:.4f} but allow_short=False")
        if sym not in prices:
            raise GuardrailError(f"{sym}: no price available")
        p = prices[sym]
        if not math.isfinite(p) or p <= 0:
            raise GuardrailError(f"{sym}: price is {p!r}")

    gross = sum(abs(w) for w in target.values())
    if gross > policy.max_gross_leverage + 1e-9:
        raise GuardrailError(
            f"gross exposure {gross:.4f} exceeds max_gross_leverage "
            f"{policy.max_gross_leverage:.4f}"
        )


def plan_rebalance(
    account: Account,
    target: dict[str, float],
    prices: dict[str, float],
    policy: RebalancePolicy = RebalancePolicy(),
) -> list[Order]:
    """Orders that move ``account`` toward ``target``, subject to ``policy``.

    Returns an empty list when every position is already inside the band --
    which, for a slow strategy, is most days. That is the point.
    """
    held = {s: q for s, q in account.positions.items() if q}
    for sym in held:
        if sym not in prices:
            raise GuardrailError(f"{sym}: held but no price available")
    _validate(target, prices, policy)

    equity = account.equity(prices)
    if equity <= 0:
        raise GuardrailError(f"account equity is {equity:,.2f}; refusing to trade")

    current = account.weights(prices)
    orders: list[Order] = []

    for sym in sorted(set(target) | set(held)):
        tgt_w = target.get(sym, 0.0)
        cur_w = current.get(sym, 0.0)
        drift = tgt_w - cur_w

        # Exiting to flat is always allowed through the band: a position the
        # strategy no longer wants is a risk decision, not a rebalance.
        exiting = abs(tgt_w) < 1e-12 and abs(cur_w) > 1e-12
        if abs(drift) < policy.band and not exiting:
            continue

        price = prices[sym]
        notional = drift * equity
        if abs(notional) < policy.min_notional and not exiting:
            continue
        notional = math.copysign(min(abs(notional), policy.max_order_notional), notional)

        qty = notional / price
        if not policy.allow_fractional:
            qty = math.trunc(qty)
        if abs(qty) < 1e-9:
            continue

        orders.append(Order(
            symbol=sym,
            side="buy" if qty > 0 else "sell",
            qty=abs(qty),
            est_price=price,
            reason=("exit" if exiting else f"drift {drift:+.4f} vs band {policy.band:.4f}"),
        ))

    return orders


# --------------------------------------------------------------------------
class PaperBroker:
    """In-memory broker that charges the same costs the backtest assumed.

    Using a different cost model here than in ``run_backtest`` would make live
    results diverge from the backtest for reasons that have nothing to do with
    the strategy, so it takes the same :class:`CostModel`.
    """

    def __init__(self, cash: float = 100_000.0, costs: CostModel = CostModel(),
                 marks: dict[str, float] | None = None):
        self._account = Account(cash=float(cash), positions={})
        self.costs = costs
        self.marks = dict(marks or {})
        self.fills: list[Order] = []
        self.total_costs = 0.0

    def account(self) -> Account:
        return Account(cash=self._account.cash, positions=dict(self._account.positions))

    def prices(self, symbols: list[str]) -> dict[str, float]:
        missing = [s for s in symbols if s not in self.marks]
        if missing:
            raise GuardrailError(f"no mark for {missing}")
        return {s: self.marks[s] for s in symbols}

    def set_marks(self, marks: dict[str, float]) -> None:
        self.marks.update(marks)

    def submit(self, orders: list[Order]) -> list[Order]:
        for o in orders:
            px = self.marks[o.symbol]
            signed = o.qty if o.side == "buy" else -o.qty
            # Slippage and spread always work against you, both directions.
            fill_px = px * (1 + math.copysign(self.costs.per_trade_bps / 1e4, signed))
            cost = abs(signed) * px * self.costs.per_trade_bps / 1e4

            self._account.cash -= signed * fill_px
            self._account.positions[o.symbol] = (
                self._account.positions.get(o.symbol, 0.0) + signed
            )
            if abs(self._account.positions[o.symbol]) < 1e-12:
                self._account.positions.pop(o.symbol)
            self.total_costs += cost
            self.fills.append(o)
        return orders


def execute(
    broker: Broker,
    target: dict[str, float],
    policy: RebalancePolicy = RebalancePolicy(),
    dry_run: bool = True,
) -> list[Order]:
    """Plan and (optionally) place a rebalance.

    ``dry_run`` defaults to True. Sending real orders is an explicit, typed act
    at the call site -- never a default, and never a config file's job.
    """
    symbols = sorted(set(target) | set(broker.account().positions))
    prices = broker.prices(symbols)
    orders = plan_rebalance(broker.account(), target, prices, policy)
    if dry_run or not orders:
        return orders
    return broker.submit(orders)
