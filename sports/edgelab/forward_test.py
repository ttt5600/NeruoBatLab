"""GOAL.md rule 6 (amendment 2026-11-18, BAR_VERSION 2): the forward sequential test that replaces "100 forward bets
passing rule 5" as the forward_paper -> live_small criterion. Calibration: notes/bar_calibration_v2.md +
results/bar_calibration_v2/sim2.py (nc_task / sp_task are the reference implementation; the LLRs below are the same).

Two tests, chosen by the trial's registered `forward_test.kind` or, if the spec has none, by the forward rows' prices:
  near_certain (mean all-in cost >= .90): Wald SPRT on FILLED LOSSES. H0 loss prob = 1 - c_i (c = price + fee),
    H1 = RHO1 x (1 - c_i). Per bet: loss -> log RHO1, win -> log((1 - RHO1 L0) / (1 - L0)). Looks after each cluster
    (event-day / night). LLR divided by the design effect 1 + f x .031, f = fills per night FIXED AT REGISTRATION
    (spec forward_test.fills_per_night; default 55/30). alpha .005, beta .10. Horizon 36 months from the first
    forward bet; no crossing by then -> 'horizon_user_decides'.
  sports_roi (every bet's price in [.40, .60]; the sim covered 49-53.5c): Wald SPRT on win/loss vs break-even at cost.
    H1 win prob = c (1 + r1), r1 = .03 (or .05 if registered). Looks per date (cluster). LLR divided by
    1 + (b - 1) x .05, b = bets per date fixed at registration (default 7). alpha .05, beta .20. Horizon 24 in-season
    months (calendar months with >= 1 bet). Other price mixes have NO calibrated forward test ('out_of_scope': no
    acceptance and no SPRT kill) until a placebo calibration for that mix is registered.
Status: accept_h1 | reject_h0 | continue | horizon_user_decides | out_of_scope. The first boundary crossing in look
order is final (later looks cannot undo it). Rows whose outcome is not 0/1 (push, scalar settlement) carry no LLR.
accept_h1 -> needs_audit (the user audits before live_small); reject_h0 -> a rule-7 kill IN ADDITION to the pre-set
kills. Only a trial that reached sealed_pass can be promoted by its forward test (`gate_ok`); others are reported only.
Trials already scoring forward bets on ADOPTED use the defaults below, fixed at adoption.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ADOPTED = "2026-11-18"
NC_COST_MIN = 0.90
NC_RHO1, NC_ALPHA, NC_BETA, NC_ICC, NC_F_DEFAULT, NC_HORIZON = 0.70, 0.005, 0.10, 0.031, 55 / 30, 36
SP_PRICE_LO, SP_PRICE_HI = 0.40, 0.60
SP_R1_ALLOWED, SP_R1_DEFAULT, SP_ALPHA, SP_BETA, SP_ICC, SP_B_DEFAULT, SP_HORIZON = (0.03, 0.05), 0.03, 0.05, 0.20, \
    0.05, 7.0, 24
KINDS = ("near_certain", "sports_roi")


class ForwardTestError(ValueError):
    pass


def validate(ft: dict | None) -> dict | None:
    """Spec `forward_test` block -> normalized dict (registration-time check). None = use the adoption defaults."""
    if ft is None:
        return None
    if not isinstance(ft, dict) or ft.get("kind") not in KINDS:
        raise ForwardTestError(f"forward_test.kind must be one of {KINDS}")
    allowed = {"near_certain": {"kind", "fills_per_night"}, "sports_roi": {"kind", "bets_per_date", "h1_roi"}}
    extra = set(ft) - allowed[ft["kind"]]
    if extra:
        raise ForwardTestError(f"forward_test ({ft['kind']}) does not take {sorted(extra)}")
    if ft["kind"] == "near_certain":
        f = float(ft.get("fills_per_night", NC_F_DEFAULT))
        if not 0 < f < 1000:
            raise ForwardTestError("fills_per_night must be > 0")
        return {"kind": "near_certain", "fills_per_night": f}
    b = float(ft.get("bets_per_date", SP_B_DEFAULT))
    r1 = float(ft.get("h1_roi", SP_R1_DEFAULT))
    if b < 1:
        raise ForwardTestError("bets_per_date must be >= 1")
    if r1 not in SP_R1_ALLOWED:
        raise ForwardTestError(f"h1_roi must be one of {SP_R1_ALLOWED} (the calibrated alternatives)")
    return {"kind": "sports_roi", "bets_per_date": b, "h1_roi": r1}


def classify(cost: np.ndarray, price: np.ndarray) -> str:
    if len(cost) and float(np.mean(cost)) >= NC_COST_MIN - 1e-9:
        return "near_certain"
    if len(price) and bool(np.all((price >= SP_PRICE_LO) & (price <= SP_PRICE_HI))):
        return "sports_roi"
    return "out_of_scope"


def boundaries(alpha: float, beta: float) -> tuple[float, float]:
    return float(np.log((1 - beta) / alpha)), float(np.log(beta / (1 - alpha)))


def sprt_path(llr: np.ndarray, look: np.ndarray, deff: float, A: float, B: float, in_horizon: np.ndarray) -> dict:
    """llr per bet, look = 0..K-1 index of the look (cluster in time order) each bet belongs to, in_horizon per look.
    -> first crossing (only looks inside the horizon count)."""
    K = int(look.max()) + 1 if len(look) else 0
    c = np.cumsum(np.bincount(look, weights=llr, minlength=K)) / deff
    hi = np.flatnonzero((c >= A) & in_horizon)
    lo = np.flatnonzero((c <= B) & in_horizon)
    ih = int(hi[0]) if len(hi) else 10 ** 9
    il = int(lo[0]) if len(lo) else 10 ** 9
    if ih == il == 10 ** 9:
        status = "horizon_user_decides" if K and not bool(in_horizon[-1]) else "continue"
        at = None
    else:
        status, at = ("accept_h1", ih) if ih < il else ("reject_h0", il)
    return {"status": status, "decided_at_look": at, "looks": K,
            "llr": float(c[-1]) if K else 0.0, "llr_at_decision": float(c[at]) if at is not None else None,
            "llr_min": float(c.min()) if K else 0.0, "llr_max": float(c.max()) if K else 0.0}


def _looks(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Cluster -> look index in time order (by the cluster's last decided_at, then first event_start)."""
    cl = df["cluster"].astype(str).to_numpy()
    g = pd.DataFrame({"cl": cl, "d": df["decided_at"].to_numpy(), "e": df["event_start"].to_numpy()}) \
        .groupby("cl").agg(d=("d", "max"), e=("e", "min")).sort_values(["d", "e"], kind="stable")
    pos = pd.Series(np.arange(len(g)), index=g.index)
    return pos.loc[cl].to_numpy(), g


def run_test(df: pd.DataFrame, cost: np.ndarray, spec_ft: dict | None, gate_ok: bool) -> dict:
    """df = enforced forward bet rows; cost = all-in cost per row (evaluator.cost_and_pnl)."""
    params = validate(spec_ft)
    source = "registration" if params else "adoption_default"
    price = df["price"].to_numpy(float)
    auto = classify(cost, price)
    kind = params["kind"] if params else auto
    base = {"kind": kind, "params_source": source, "adopted": ADOPTED, "gate_ok": bool(gate_ok),
            "mean_cost": float(np.mean(cost)) if len(cost) else None, "price_range": [float(price.min()), float(price.max())] if len(price) else None}
    if len(df) == 0:
        return {**base, "status": "continue", "n_used": 0, "looks": 0, "promotes": False, "kills": False}
    if kind == "out_of_scope" or (kind == "near_certain" and auto != "near_certain") or \
            (kind == "sports_roi" and auto != "sports_roi"):
        why = (f"mean cost {np.mean(cost):.3f} < {NC_COST_MIN}" if kind == "near_certain" else
               f"prices outside [{SP_PRICE_LO}, {SP_PRICE_HI}] (no placebo calibration for this mix)")
        return {**base, "kind": kind, "status": "out_of_scope", "why": why, "n_used": 0, "looks": 0,
                "promotes": False, "kills": False}
    out = df["outcome"].to_numpy(float)
    binary = (out == 0) | (out == 1)
    d = df.loc[binary]
    c, o = cost[binary], out[binary]
    if len(d) == 0:
        return {**base, "status": "continue", "n_used": 0, "looks": 0, "promotes": False, "kills": False}
    look, g = _looks(d)
    t_first = pd.Timestamp(g["e"].min())
    if kind == "near_certain":
        f = params["fills_per_night"] if params else NC_F_DEFAULT
        L0 = np.clip(1 - c, 1e-9, 1 - 1e-9)
        llr = np.where(o == 0, np.log(NC_RHO1), np.log((1 - NC_RHO1 * L0) / (1 - L0)))
        deff = 1 + f * NC_ICC
        A, B = boundaries(NC_ALPHA, NC_BETA)
        le = pd.to_datetime(g["e"])
        tf = pd.Timestamp(le.min())
        months = (le.dt.year - tf.year) * 12 + (le.dt.month - tf.month) + 1
        in_h = (months.to_numpy() <= NC_HORIZON)
        p = {"rho1": NC_RHO1, "alpha": NC_ALPHA, "beta": NC_BETA, "fills_per_night": f, "icc": NC_ICC,
             "horizon_months": NC_HORIZON}
        span = int(months.max())
    else:
        b = params["bets_per_date"] if params else SP_B_DEFAULT
        r1 = params["h1_roi"] if params else SP_R1_DEFAULT
        p1 = np.clip(c * (1 + r1), 1e-9, 0.999)
        llr = np.where(o == 1, np.log(p1 / c), np.log((1 - p1) / (1 - c)))
        deff = 1 + (b - 1) * SP_ICC
        A, B = boundaries(SP_ALPHA, SP_BETA)
        ym = pd.to_datetime(g["e"]).dt.strftime("%Y-%m").to_numpy()
        seen, cnt = set(), []
        for m in ym:
            seen.add(m)
            cnt.append(len(seen))
        in_h = np.asarray(cnt) <= SP_HORIZON
        p = {"h1_roi": r1, "alpha": SP_ALPHA, "beta": SP_BETA, "bets_per_date": b, "icc": SP_ICC,
             "horizon_in_season_months": SP_HORIZON}
        span = int(cnt[-1])
    r = sprt_path(llr, look, deff, A, B, in_h)
    return {**base, **r, "params": p, "deff": float(deff), "A": A, "B": B, "n_used": int(len(d)),
            "months_elapsed": span, "first_event": str(t_first),
            "promotes": bool(r["status"] == "accept_h1" and gate_ok), "kills": bool(r["status"] == "reject_h0")}
