"""GOAL.md rule 5 -- the one verdict function.

score(rows, bar, min_n, fee) -> dict with the metric (net ROI per $ risked after costs), its event-day cluster
bootstrap interval, the one-sided p vs 0, both time halves, cluster concentration, CLV, the too-good guard, and the
verdict: pass | fail | needs_audit. Reported ROI is the OBSERVED ratio sum(pnl)/sum(stake); the interval is the
2.5/97.5 percentiles of the resampled ratio (never the resample mean as the point estimate).

Pre-set constants (written 2026-11-02 before any strategy was scored through this code):
  B_COARSE 2,000 resamples, refined to B_FINE 40,000 when the coarse p is within 20x of the bar, seed 0.
  p = (1 + #{resampled ROI <= 0}) / (B + 1)  (never exactly 0).
  Halves: bets ordered by decided_at, split at the middle bet; each half's observed ROI must be > 0.
  Concentration: max share of total profit from one cluster, computed at the cluster unit AND by ISO week of the
    event (the stricter of the two); must be <= 0.30; undefined (fail) when total profit <= 0.
  CLV: mean(close_price - price) over rows with a close; needs >= 50% coverage, else CLV is null and (e) fails.
  Too-good guard: ROI > +10% with n > 200, OR annualized Sharpe of cluster P&L > 5 -> needs_audit.
    (Disclosed: before fixing 5 I estimated by hand that E1 taker sits near 3 and E4 maker near 7.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import fees as F

B_COARSE, B_FINE, SEED = 2_000, 40_000, 0
CONCENTRATION_MAX = 0.30
CLV_MIN_COVERAGE = 0.50
TOO_GOOD_ROI, TOO_GOOD_N, SHARPE_MAX = 0.10, 200, 5.0


def cost_and_pnl(df: pd.DataFrame, fee: dict) -> tuple[np.ndarray, np.ndarray]:
    """All-in cost per $1 payout and pnl per row, from price + the venue fee schedule. The rule's own pnl is ignored
    (checked for agreement by the caller)."""
    price = df["price"].to_numpy(float)
    sched_by_src = {**F.DEFAULT_SCHEDULE, **(fee.get("schedules") or {})}
    feec = np.zeros(len(df))
    for src, sched in sched_by_src.items():
        m = (df["price_source"] == src).to_numpy()
        if m.any():
            feec[m] = F.fee_per_contract(sched, price[m], int(fee.get("contracts_per_order", 1)),
                                         fee.get("rounding", "cent"))
    cost = price + feec
    out = df["outcome"].to_numpy(float)
    if "contracts" in df.columns and df["contracts"].notna().all():
        # contract staking: risk = contracts x all-in cost; a win pays contracts x (1 - cost)
        n = df["contracts"].to_numpy(float)
        stake = n * cost
        pnl = np.where(out == 1, n * (1 - cost), np.where(out == 0, -stake, 0.0))
    else:
        # dollar staking: `stake` dollars of cost buy stake/cost contracts
        stake = df["stake"].to_numpy(float)
        pnl = np.where(out == 1, stake * (1 / cost - 1), np.where(out == 0, -stake, 0.0))
    return cost, pnl, stake


def _boot_ratio(cl_pnl: np.ndarray, cl_stake: np.ndarray, B: int, rng) -> np.ndarray:
    K = len(cl_pnl)
    out = np.empty(B)
    step = max(1, 4_000_000 // max(K, 1))
    for i in range(0, B, step):
        b = min(step, B - i)
        idx = rng.integers(0, K, size=(b, K))
        s = cl_stake[idx].sum(1)
        out[i:i + b] = np.where(s > 0, cl_pnl[idx].sum(1) / np.maximum(s, 1e-12), 0.0)
    return out


def bootstrap(cluster: np.ndarray, pnl: np.ndarray, stake: np.ndarray, bar: float | None) -> dict:
    u, inv = np.unique(cluster, return_inverse=True)
    cp = np.bincount(inv, weights=pnl, minlength=len(u))
    cs = np.bincount(inv, weights=stake, minlength=len(u))
    rng = np.random.default_rng(SEED)
    bs = _boot_ratio(cp, cs, B_COARSE, rng)
    p = (1 + np.sum(bs <= 0)) / (len(bs) + 1)
    if bar is not None and p < 20 * bar:
        bs = _boot_ratio(cp, cs, B_FINE, np.random.default_rng(SEED + 1))
        p = (1 + np.sum(bs <= 0)) / (len(bs) + 1)
    return {"p": float(p), "ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
            "B": int(len(bs)), "clusters": int(len(u)), "cluster_pnl": cp}


def concentration(df: pd.DataFrame, pnl: np.ndarray) -> dict:
    total = float(pnl.sum())
    by_cl = pd.Series(pnl).groupby(df["cluster"].astype(str).to_numpy()).sum()
    wk = df["event_start"].dt.tz_convert("UTC").dt.strftime("%G-W%V").to_numpy()
    by_wk = pd.Series(pnl).groupby(wk).sum()
    if total <= 0:
        return {"share": None, "cluster_share": None, "week_share": None, "top_cluster": None}
    cs, ws = float(by_cl.max() / total), float(by_wk.max() / total)
    return {"share": max(cs, ws), "cluster_share": cs, "week_share": ws,
            "top_cluster": str(by_cl.idxmax()), "top_week": str(by_wk.idxmax())}


def clv(df: pd.DataFrame) -> dict:
    c = df["close_price"].to_numpy(float)
    m = ~np.isnan(c)
    cov = float(m.mean()) if len(m) else 0.0
    if cov < CLV_MIN_COVERAGE or not m.any():
        return {"mean": None, "coverage": cov, "n": int(m.sum()), "p_negative": None}
    d = c[m] - df["price"].to_numpy(float)[m]
    # one-sided p that true mean CLV < 0, cluster bootstrap over the bets with a close
    cl = df["cluster"].astype(str).to_numpy()[m]
    u, inv = np.unique(cl, return_inverse=True)
    sd = np.bincount(inv, weights=d, minlength=len(u))
    sn = np.bincount(inv, minlength=len(u)).astype(float)
    bs = _boot_ratio(sd, sn, B_COARSE, np.random.default_rng(SEED + 2))
    return {"mean": float(d.mean()), "coverage": cov, "n": int(m.sum()),
            "p_negative": float((1 + np.sum(bs >= 0)) / (len(bs) + 1))}


def sharpe_annualized(df: pd.DataFrame, cluster_pnl: np.ndarray) -> float | None:
    if len(cluster_pnl) < 10 or cluster_pnl.std(ddof=1) == 0:
        return None
    span_days = max(1.0, (df["event_start"].max() - df["event_start"].min()).total_seconds() / 86400)
    per_year = len(cluster_pnl) / (span_days / 365.25)
    return float(cluster_pnl.mean() / cluster_pnl.std(ddof=1) * np.sqrt(per_year))


def score(df: pd.DataFrame, bar: float | None, min_n: int, fee: dict | None = None) -> dict:
    """df = enforced (normalized, PIT-clean) bet rows. bar = alpha / all-time family trials (None = report only)."""
    fee = fee or {}
    n = int(len(df))
    if n == 0:
        return {"n": 0, "verdict": "fail", "reasons": ["no bets"], "net_roi": None, "ci": [None, None], "p": None,
                "bar": bar, "half1": None, "half2": None, "concentration": None, "clv": None, "min_n": min_n,
                "needs_audit": False, "checks": {}}
    cost, pnl, stake = cost_and_pnl(df, fee)
    if "pnl" in df.columns and df["pnl"].notna().any():
        given = df["pnl"].to_numpy(float)
        ok = np.isnan(given) | np.isclose(given, pnl, atol=2e-3, rtol=1e-3)
        if not ok.all():
            raise ValueError(f"{int((~ok).sum())} rows carry a pnl that disagrees with price + fee schedule + outcome; "
                             "the rule may not set its own costs")
    roi = float(pnl.sum() / stake.sum())
    bt = bootstrap(df["cluster"].astype(str).to_numpy(), pnl, stake, bar)
    order = np.argsort(df["decided_at"].to_numpy(), kind="stable")
    h = n // 2
    i1, i2 = order[:h], order[h:]
    half1 = float(pnl[i1].sum() / stake[i1].sum()) if h else None
    half2 = float(pnl[i2].sum() / stake[i2].sum()) if len(i2) else None
    conc = concentration(df, pnl)
    cv = clv(df)
    sh = sharpe_annualized(df, bt["cluster_pnl"])
    checks = {
        "a_p_below_bar": bool(bar is not None and bt["p"] < bar),
        "b_both_halves_positive": bool(half1 is not None and half2 is not None and half1 > 0 and half2 > 0),
        "c_concentration_ok": bool(conc["share"] is not None and conc["share"] <= CONCENTRATION_MAX),
        "d_min_n": bool(n >= min_n),
        "e_clv_nonnegative": bool(cv["mean"] is not None and cv["mean"] >= 0),
    }
    too_good = bool((roi > TOO_GOOD_ROI and n > TOO_GOOD_N) or (sh is not None and sh > SHARPE_MAX))
    if all(checks.values()):
        verdict = "needs_audit" if too_good else "pass"
    else:
        verdict = "fail"
    return {"n": n, "staked": float(stake.sum()), "pnl": float(pnl.sum()), "net_roi": roi, "ci": bt["ci"],
            "p": bt["p"], "B": bt["B"], "clusters": bt["clusters"], "bar": bar, "min_n": min_n,
            "half1": half1, "half2": half2, "concentration": conc, "clv": cv, "sharpe_ann": sh,
            "mean_cost": float(cost.mean()), "losses": int((df["outcome"] == 0).sum()),
            "needs_audit": too_good, "checks": checks, "verdict": verdict,
            "reasons": [k for k, v in checks.items() if not v] + (["too_good_guard"] if too_good else []),
            "fee": fee}
