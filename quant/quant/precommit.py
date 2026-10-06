"""Pre-registered go/no-go criteria, sealed before the evidence exists.

The failure this prevents is not a bad strategy. It is a moving goalpost. Six
months of paper trading produces a number, and without a rule written in
advance, that number gets interpreted by someone who wants to trade: a losing
run becomes "an unlucky regime", a flat run becomes "it held up in a hard
tape", and a winning run becomes proof. Every outcome argues for going live,
which means the experiment had no power to say no.

So the criteria are written first, hashed, and timestamped. At the decision
date the checker reports PASS or FAIL mechanically. Changing a sealed rule is
allowed -- people learn things -- but it breaks the seal visibly and records
the old rule, so a revision is a decision you can see rather than one that
happens quietly inside your own head.

This is the pre-committed stop rule, applied to money instead of to a model.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .paper import PaperAccount, divergence
from .stats import max_drawdown, sharpe


@dataclass
class DecisionRule:
    """Falsifiable conditions, fixed before the data arrives."""

    strategy: str
    decision_date: str
    """No go-live decision may be taken before this date, whatever the numbers."""

    min_trading_days: int = 126
    """~6 months. Fewer days cannot distinguish skill from noise at these Sharpes."""

    min_paper_sharpe: float = 0.30
    """Deliberately low. This is a sanity floor, not evidence of skill -- six
    months cannot establish a Sharpe. Setting it high would invite a lucky run
    to be read as proof."""

    max_drawdown_abort: float = -0.15
    """Breach this at any point and the answer is NO, regardless of end value.
    Drawdown tolerance is the one thing you know honestly in advance and stop
    knowing honestly once you are losing."""

    min_divergence_correlation: float = 0.95
    """The live path must track the backtest. Failure here indicts the SYSTEM,
    not the strategy, and no strategy verdict is valid until it is fixed."""

    max_abs_mean_daily_gap_bps: float = 2.0
    """Systematic drift between live and backtest means an unmodelled cost."""

    must_beat_benchmark: bool = False
    """Left False on purpose. Six months cannot resolve a Sharpe difference of
    the size at stake -- demanding it would guarantee a coin-flip verdict
    dressed as a test. The honest bar here is 'runs correctly and survives its
    drawdown limit', with the edge question left to the statistics, not to a
    short live window."""

    notes: str = ""
    sealed_at: str = field(default="")
    seal: str = field(default="")
    superseded: list[dict] = field(default_factory=list)

    # ---------------- sealing ----------------
    def _payload(self) -> dict:
        d = asdict(self)
        for k in ("sealed_at", "seal", "superseded"):
            d.pop(k, None)
        return d

    def compute_seal(self) -> str:
        blob = json.dumps(self._payload(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def seal_now(self) -> "DecisionRule":
        self.sealed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.seal = self.compute_seal()
        return self

    @property
    def intact(self) -> bool:
        return bool(self.seal) and self.seal == self.compute_seal()

    def save(self, path: Path) -> None:
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "DecisionRule":
        return cls(**json.loads(Path(path).read_text()))

    def supersede(self, **changes) -> "DecisionRule":
        """Revise a sealed rule, keeping the old one on the record."""
        old = asdict(self)
        for k in ("superseded",):
            old.pop(k, None)
        history = list(self.superseded) + [old]
        data = {**self._payload(), **changes}
        new = DecisionRule(**data)
        new.superseded = history
        return new.seal_now()


# --------------------------------------------------------------------------
@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    blocking: bool = True


def evaluate(rule: DecisionRule, account: PaperAccount, prices, build_weights,
             costs=None, today: str | None = None) -> dict:
    """Mechanically apply ``rule`` to ``account``. No judgement calls."""
    from .backtest import CostModel

    costs = costs or CostModel()
    checks: list[CheckResult] = []

    if not rule.intact:
        checks.append(CheckResult(
            "seal", False,
            f"rule was modified after sealing (seal {rule.seal}, "
            f"recomputed {rule.compute_seal()})"))
    else:
        checks.append(CheckResult("seal", True, f"intact ({rule.seal})"))

    if rule.strategy != account.strategy:
        checks.append(CheckResult(
            "strategy", False,
            f"rule covers {rule.strategy!r}, journal runs {account.strategy!r}"))
    else:
        checks.append(CheckResult("strategy", True, rule.strategy))

    # LIVE days only. Backfilled bars replay the period the strategy was chosen
    # on; counting them here would let anyone clear a six-month forward-evidence
    # bar instantly by replaying five years of history.
    live = account.live_days
    total = len(account.entries)
    checks.append(CheckResult(
        "min_live_days", live >= rule.min_trading_days,
        f"{live} live of {rule.min_trading_days} required "
        f"({total - live} backfill/catch-up bars do not count)"))

    now = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    checks.append(CheckResult(
        "decision_date", now >= rule.decision_date,
        f"today {now}, decision date {rule.decision_date}"))

    r = account.returns()
    if len(r) > 2:
        dd = max_drawdown(r)
        checks.append(CheckResult(
            "max_drawdown_abort", dd >= rule.max_drawdown_abort,
            f"{dd:.2%} vs limit {rule.max_drawdown_abort:.2%}"))
        sr = sharpe(r)
        checks.append(CheckResult(
            "min_paper_sharpe", sr >= rule.min_paper_sharpe,
            f"{sr:.3f} vs floor {rule.min_paper_sharpe:.3f}"))
    else:
        checks.append(CheckResult("max_drawdown_abort", False, "insufficient history"))
        checks.append(CheckResult("min_paper_sharpe", False, "insufficient history"))

    d = divergence(account, prices, build_weights, costs)
    if d.get("status") == "ok":
        checks.append(CheckResult(
            "divergence_correlation", d["correlation"] >= rule.min_divergence_correlation,
            f"{d['correlation']:.4f} vs floor {rule.min_divergence_correlation:.2f}"))
        gap = abs(d["mean_daily_gap_bps"])
        checks.append(CheckResult(
            "tracking_gap", gap <= rule.max_abs_mean_daily_gap_bps,
            f"{gap:.2f} bps vs limit {rule.max_abs_mean_daily_gap_bps:.2f}"))
    else:
        checks.append(CheckResult("divergence_correlation", False, d.get("status", "?")))
        checks.append(CheckResult("tracking_gap", False, d.get("status", "?")))

    blocking_failures = [c for c in checks if c.blocking and not c.passed]
    return {
        "verdict": "GO" if not blocking_failures else "NO-GO",
        "checks": checks,
        "failures": [c.name for c in blocking_failures],
        "divergence": d,
    }
