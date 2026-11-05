"""Promotion ladder (GOAL.md rule 6) as a state machine.

    registered -> dev_pass -> sealed_pass -> forward_paper -> live_small -> live
    any state -> dead (a fail / kill / retire; turning OFF is always allowed)

Who may move what:
  * evaluate(): registered->dev_pass|dead, dev_pass->sealed_pass|dead, sealed_pass->forward_paper,
    forward_paper->dead (pre-set kill). Nothing else.
  * the user, through `edgelab approve --user-approved --by <name>`: forward_paper->live_small, live_small->live,
    and size changes. Requires forward eligibility (>= 100 forward bets passing rule 5 on forward data alone),
    except ONE early size step <= EARLY_SIZE_MAX for a forward_paper strategy with a pre-set kill (GOAL.md rule 6
    amendment, user 2026-11-05).
  * anyone: -> dead (retire).
A needs_audit verdict never moves a trial; it is recorded on the evaluation and blocks the next step.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

STATES = ["registered", "dev_pass", "sealed_pass", "forward_paper", "live_small", "live", "dead"]
EVALUATE_EDGES = {("registered", "dev_pass"), ("registered", "dead"), ("dev_pass", "sealed_pass"),
                  ("dev_pass", "dead"), ("sealed_pass", "forward_paper"), ("forward_paper", "dead"),
                  ("registered", "sealed_pass")}  # last edge: forward-only datasets (dev: null), see vault.py
USER_EDGES = {("forward_paper", "live_small"), ("live_small", "live")}
USER_ONLY_TARGETS = {"live_small", "live"}
FORWARD_MIN_N = 100
EARLY_SIZE_MAX = 2.0   # GOAL.md rule 6 amendment 2026-11-05: one early size step, at most 2x


class LadderError(RuntimeError):
    pass


@dataclass(frozen=True)
class UserApproval:
    by: str
    when: str
    note: str = ""

    @staticmethod
    def make(by: str, note: str = "") -> "UserApproval":
        if not by or not by.strip():
            raise LadderError("user approval needs --by <name>")
        return UserApproval(by=by.strip(), when=datetime.now(timezone.utc).isoformat(timespec="seconds"), note=note)


def check(frm: str, to: str, via: str, approval: UserApproval | None = None) -> None:
    if to not in STATES:
        raise LadderError(f"unknown state {to!r}")
    if to == "dead":
        return
    if via == "evaluate":
        if to in USER_ONLY_TARGETS:
            raise LadderError(f"evaluate() can never move a strategy to {to}: user-only (GOAL.md rule 6)")
        if (frm, to) not in EVALUATE_EDGES:
            raise LadderError(f"evaluate() may not move {frm} -> {to}")
        return
    if via == "user":
        if not isinstance(approval, UserApproval):
            raise LadderError("user transitions need an explicit UserApproval (--user-approved --by NAME)")
        if (frm, to) not in USER_EDGES:
            raise LadderError(f"user transition {frm} -> {to} is not on the ladder")
        return
    raise LadderError(f"unknown actor {via!r}")


def transition(reg, trial_id: str, to: str, via: str, approval: UserApproval | None = None, **info) -> dict:
    frm = reg.state(trial_id)
    if frm == to:
        return {}
    if frm == "dead":
        raise LadderError(f"{trial_id} is dead; register a new trial (it counts)")
    check(frm, to, via, approval)
    ev = {"event": "transition", "trial_id": trial_id, "from": frm, "to": to, "via": via, **info}
    if approval:
        ev["approval"] = {"by": approval.by, "when": approval.when, "note": approval.note}
    return reg.append(ev)
