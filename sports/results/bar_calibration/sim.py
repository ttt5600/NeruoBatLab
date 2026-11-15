"""bar_calibration -- experiment A of notes/bar_review.md. Seeded placebo + planted-edge calibration of
CURRENT edgelab rule 5 (edgelab.evaluator.score, imported unchanged) vs the PROPOSED rules of bar_review s2.

Run from sports/:  ../.venv_quant/bin/python results/bar_calibration/sim.py [--quick]
Outputs (results/bar_calibration/): params.json, fixed_n.csv, sprt.csv, summary.json, run.log

Nothing in edgelab/ or GOAL.md is modified; no trial is registered (instrument calibration, not an edge trial).
Family bars use registry.family_bar's formula alpha / n_family with counts read read-only from the real registry.
Parameters are derived at start-up from real dev frames (lab/edgelab_runs/*_dev.parquet, outcomes only used for the
clustering estimate, all already seen) and notes; every one is written to params.json with its source.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import betainc

SPORTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SPORTS))
from edgelab import ALPHA, evaluator  # noqa: E402
from edgelab import fees as F  # noqa: E402

OUT = Path(__file__).resolve().parent
RUNS = SPORTS / "lab" / "edgelab_runs"
MASTER_SEED = 20261115

# ----------------------------------------------------------------------------------------------- parameters
def derive_params() -> dict:
    from edgelab.registry import Registry
    reg = Registry()                                   # read-only: family_counts / events only
    fc = reg.family_counts()
    sealed = {}
    for e in reg.events():
        if e.get("event") == "evaluate" and e.get("split") == "sealed":
            sealed[e.get("family")] = sealed.get(e.get("family"), 0) + 1
    x1 = pd.read_parquet(RUNS / "e4_maker_x1-e02827f2_dev.parquet")
    day0 = x1["event_start"].min().normalize()
    off = ((x1["event_start"].dt.normalize() - day0) // pd.Timedelta(days=1)).to_numpy()
    cnt = np.bincount(off)
    p = {
        "fam_wx": fc["kalshi_weather_temp"], "sealed_wx": sealed.get("kalshi_weather_temp", 0),
        "fam_sp": fc["nba_totals"], "sealed_sp": sealed.get("nba_totals", 0),
        "min_n_wx": 200, "min_n_sp": 100,                       # specs e4_* / l32b_k1 (registry)
        "e4_prices": [0.96, 0.97, 0.98], "e4_w": [3 / 25, 7 / 25, 15 / 25],  # live orig7 fills 96-98c (live_maker_orders.csv)
        "e4_fills_per_night": 54.9 / 30.44,                     # e4_maker_e1 dev 871 fills / 15.9 months
        "e4_per_month": 55.0, "x1_per_month": 250.0, "sp_per_month": 60.0,
        "icc_night": 0.031,                                     # dev dispersion e4_maker_e1 / e1_taker (notes)
        "close_sd_nc": 0.07,                                    # sd(close - price) e4 dev .0756, x1 dev .0611
        "x1_prices": x1["price"].round(2).tolist(), "x1_night_counts": cnt.tolist(), "x1_day0": str(day0.date()),
        "sp_bets_per_date": 7.0,                                # k50 primary 6.4 (nba) / 8.9 (cfb) per date
        "sp_price_lo": 0.49, "sp_price_hi": 0.535,              # ~50c rung; cost ~.53 with taker fee (k50 clv_value)
        "sp_clv_sd": 0.058,                                     # 2.69 pts x 2.14 c/pt (k50 rows_nba / clv_value)
        "sp_clv_sd_stress": 0.076,                              # 3.3 pts (cbb_opener_model) x 2.3 c/pt
        "sp_icc_date": 0.05,                                    # ASSUMPTION (no estimate): shared same-date line move
        "sprt_r1": 0.02, "sprt_alpha": 0.05, "sprt_beta": 0.20, "horizon_months": 36,
    }
    p["bar_wx"] = ALPHA / p["fam_wx"]
    p["bar_sp"] = ALPHA / p["fam_sp"]
    p["sealed_alpha_wx"] = ALPHA / (p["sealed_wx"] + 1)
    p["sealed_alpha_sp"] = ALPHA / (p["sealed_sp"] + 1)
    p["icc_cm_stress"] = 0.01                                   # ASSUMPTION: stress only; dev shows none (ratio <1)
    return p


P: dict = {}


def _init(params):
    global P
    P = params


# ----------------------------------------------------------------------------------------------- generators
def gen_nc(rng, n, r, kind, stress):
    """Near-certain NO fills. kind 'e4' (forward-like, ~1.8 fills/night) or 'x1' (n packed into the X1 dev 7-week
    window with its ramp). Loss prob L_i = 1 - c_i (1 + r) (floored at 0), times a gamma night multiplier (mean 1),
    and under stress x3 night variance plus a city-month multiplier the rules do not model."""
    if kind == "e4":
        k = rng.poisson(P["e4_fills_per_night"], size=int(n / P["e4_fills_per_night"] * 1.5) + 50)
        cs = np.cumsum(k)
        last = int(np.searchsorted(cs, n))
        k = k[:last + 1].copy()
        k[-1] -= cs[last] - n
        night = np.repeat(np.arange(len(k)), k)
        price = rng.choice(P["e4_prices"], n, p=P["e4_w"])
        day0 = pd.Timestamp("2026-11-02", tz="UTC")
    else:
        w = np.asarray(P["x1_night_counts"], float)
        night = np.sort(rng.choice(len(w), n, p=w / w.sum()))
        price = rng.choice(P["x1_prices"], n)
        day0 = pd.Timestamp(P["x1_day0"], tz="UTC")
    cost = price                                                  # maker, quadratic series: fee 0
    L = np.clip(1 - cost * (1 + r), 0, 1)
    # shared shocks with a ~constant pairwise loss ICC whatever the price: factor_i = max(0, 1 + a_i (G - 1)) / c(a_i),
    # G ~ Exp(1) per cluster, a_i = sqrt(icc (1 - L_i) / L_i), c(a) = E[max(0, 1 + a (G - 1))] = a e^-(1 - 1/a) for a > 1
    # (keeps E[p_i] = L_i exactly, so placebos stay zero-edge).
    Ls = np.maximum(L, 1e-9)

    def factor(icc, cl):
        aa = np.sqrt(icc * (1 - Ls) / Ls)
        G = rng.exponential(1.0, cl.max() + 1)[cl] - 1
        c = np.where(aa > 1, aa * np.exp(-(1 - 1 / np.maximum(aa, 1e-12))), 1.0)
        return np.maximum(0, 1 + aa * G) / c

    M = factor(P["icc_night"] * (3 if stress else 1), night)
    if stress:                                                    # x city-month shock the rules do not model
        city = rng.integers(0, 7, n)
        cm = np.unique(city * 100 + night // 30, return_inverse=True)[1]
        M = M * factor(P["icc_cm_stress"], cm)
    loss = rng.random(n) < np.clip(L * M, 0, 1)
    mu = 1 - L                                                    # fair value; close is a martingale around it
    s = P["close_sd_nc"]
    kap = np.maximum(mu * (1 - mu) / s ** 2 - 1, 0.5)
    close = rng.beta(np.maximum(mu * kap, 1e-3), np.maximum((1 - mu) * kap, 1e-3))
    return {"price": price, "cost": cost, "loss": loss, "night": night, "close": close, "day0": day0,
            "true_roi": float(np.mean((1 - L) / cost) - 1), "src": "maker_tape"}


def gen_sp(rng, n, r, bias=0.0, sd=None):
    """50c sports bets at ask + taker fee. Kalshi close q = true win prob (calibrated close) = cost(1+r) + date shock +
    noise. bias > 0: the reported close sits `bias` above the true prob (close NOT calibrated; experiment B fails)."""
    sd = sd or P["sp_clv_sd"]
    k = rng.poisson(P["sp_bets_per_date"], size=int(n / P["sp_bets_per_date"] * 1.5) + 50)
    k = np.maximum(k, 1)
    cs = np.cumsum(k)
    last = int(np.searchsorted(cs, n))
    k = k[:last + 1].copy()
    k[-1] -= cs[last] - n
    night = np.repeat(np.arange(len(k)), k)
    price = rng.uniform(P["sp_price_lo"], P["sp_price_hi"], n)
    cost = price + F.kalshi_fee_per_contract(price, 1, rounding="exact")
    icc = P["sp_icc_date"]
    z = rng.normal(0, sd * np.sqrt(icc), night.max() + 1)[night]
    ptrue = np.clip(cost * (1 + r) + z + rng.normal(0, sd * np.sqrt(1 - icc), n), 0.02, 0.98)
    loss = rng.random(n) >= ptrue
    close = np.clip(ptrue + bias, 0.01, 0.99)
    return {"price": price, "cost": cost, "loss": loss, "night": night, "close": close,
            "day0": pd.Timestamp("2026-11-02", tz="UTC"), "true_roi": float(np.mean(ptrue / cost) - 1), "src": "taker"}


def to_frame(g) -> pd.DataFrame:
    ev = g["day0"] + pd.to_timedelta(g["night"], unit="D") + pd.Timedelta(hours=5)
    return pd.DataFrame({"cluster": pd.Series(ev).dt.strftime("%Y-%m-%d").to_numpy(), "event_start": ev,
                         "decided_at": ev - pd.Timedelta(hours=2), "side": "no", "stake": 1.0,
                         "price": g["price"], "price_source": g["src"], "close_price": g["close"],
                         "outcome": (~g["loss"]).astype(float)})


# ----------------------------------------------------------------------------------------------- rules
def pnl_of(g):
    return np.where(g["loss"], -1.0, 1 / g["cost"] - 1)      # $1 staked per bet (edgelab dollar staking)


def deff_of(night, icc):
    m = np.bincount(night)
    m = m[m > 0]
    mw = float((m ** 2).sum() / m.sum())
    return 1 + (mw - 1) * icc


def dispersion(g):
    """Pearson dispersion of loss counts per night around the OBSERVED loss level (E0 scaled to the observed total),
    i.e. a quasi-binomial design effect estimated from the data being tested. 1.0 when there are no losses."""
    L0 = 1 - g["cost"]
    k = float(g["loss"].sum())
    if k == 0:
        return 1.0
    s = k / L0.sum()
    Ls = np.clip(L0 * s, 0, 1)
    u, inv = np.unique(g["night"], return_inverse=True)
    obs = np.bincount(inv, weights=g["loss"].astype(float))
    ex = np.bincount(inv, weights=Ls)
    v = np.bincount(inv, weights=Ls * (1 - Ls))
    K = len(u)
    return float(((obs - ex) ** 2).sum() / max(v.sum(), 1e-12) * K / max(K - 1, 1))


def p_exact_clustered(g, data_deff=True):
    """Exact binomial on the loss count vs break-even L0 = mean(1 - c_i) with a Kish design effect: max(the dev-ICC
    prior, the data's own Pearson dispersion) when data_deff, else the prior only.
    (Binomial at the mean is wider than the Poisson-binomial -> conservative on that account.)"""
    n = len(g["cost"])
    k = float(g["loss"].sum())
    L0 = float(np.mean(1 - g["cost"]))
    d = deff_of(g["night"], P["icc_night"])
    if data_deff:
        d = max(d, dispersion(g))
    ne, ke = n / d, k / d
    return float(betainc(ne - ke, ke + 1, 1 - L0))


def iso_weeks(g):
    ev = g["day0"] + pd.to_timedelta(g["night"], unit="D")
    return pd.factorize(pd.Series(ev).dt.strftime("%G-W%V"))[0]


def loo_ok_iso(g, stat):
    """stat per bet (pnl for near-certain ROI, net CLV for sports): sum > 0 after dropping the top ISO week AND,
    separately, after dropping the top cluster (night/date)."""
    for key in (g["night"], iso_weeks(g)):
        u, inv = np.unique(key, return_inverse=True)
        tot = np.bincount(inv, weights=stat)
        keep = inv != np.argmax(tot)
        if keep.sum() == 0 or stat[keep].sum() <= 0:
            return False
    return True


def halves_ok(stat_num, stat_den):
    h = len(stat_num) // 2
    return bool(stat_num[:h].sum() / stat_den[:h].sum() > 0 and stat_num[h:].sum() / stat_den[h:].sum() > 0)


def cluster_mean_t(x, night):
    """One-sided p for mean(x) > 0 with a date-cluster-robust SE (t, K-1 df)."""
    u, inv = np.unique(night, return_inverse=True)
    K, n = len(u), len(x)
    mu = x.mean()
    s = np.bincount(inv, weights=x - mu)
    se = np.sqrt(K / (K - 1) * (s ** 2).sum()) / n
    return float(stats.t.sf(mu / se, K - 1)) if se > 0 else (0.0 if mu > 0 else 1.0), mu, se


def roi_ci(pnl, night):
    u, inv = np.unique(night, return_inverse=True)
    K, n = len(u), len(pnl)
    mu = pnl.mean()
    s = np.bincount(inv, weights=pnl - mu)
    se = np.sqrt(K / (K - 1) * (s ** 2).sum()) / n
    return mu - 1.96 * se, mu + 1.96 * se


def proposed_fixed(g, kind):
    """Returns {stat_p, loo, halves, consistency} for the proposed fixed-n rule (stage alpha applied by the caller)."""
    pnl = pnl_of(g)
    if kind == "sp":
        net = g["close"] - g["cost"]
        p, mu, _ = cluster_mean_t(net, g["night"])
        lo, hi = roi_ci(pnl, g["night"])
        implied = mu / g["cost"].mean()
        return {"p": p, "loo": loo_ok_iso(g, net), "halves": halves_ok(net, np.ones_like(net)),
                "consistent": bool(lo <= implied <= hi), "p_prior": p}
    return {"p": p_exact_clustered(g), "p_prior": p_exact_clustered(g, False), "loo": loo_ok_iso(g, pnl),
            "halves": halves_ok(pnl, np.ones_like(pnl)), "consistent": True}


# ----------------------------------------------------------------------------------------------- workers
def cell_fixed(args):
    """One (scenario, variant, r, n) chunk: proposed rules on `reps` fresh datasets; current rule on the first
    `reps_cur` of them."""
    scen, var, r, n, reps, reps_cur, seed = args
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(reps):
        if scen == "sp":
            g = gen_sp(rng, n, r, bias={"base": 0, "bias05": .005, "bias1": .01, "sd76": 0}[var],
                       sd=P["sp_clv_sd_stress"] if var == "sd76" else None)
        else:
            g = gen_nc(rng, n, r, scen, stress=(var == "stress"))
        pr = proposed_fixed(g, "sp" if scen == "sp" else "nc")
        row = {"true_roi": g["true_roi"], "prop_p": pr["p"], "prop_p_prior": pr["p_prior"], "loo": pr["loo"], "halves": pr["halves"],
               "consistent": pr["consistent"], "n": n}
        if i < reps_cur:
            df = to_frame(g)
            bar = P["bar_sp"] if scen == "sp" else P["bar_wx"]
            min_n = P["min_n_sp"] if scen == "sp" else P["min_n_wx"]
            res = evaluator.score(df, bar, min_n, {"rounding": "exact"})
            row.update({"cur_pass": res["verdict"] == "pass", "cur_audit": res["verdict"] == "needs_audit",
                        "cur_p": res["p"], **{f"cur_{k}": v for k, v in res["checks"].items()}})
        rows.append(row)
    return scen, var, r, n, rows


def cell_sprt(args):
    """Sequential tests on a stream of fills (forward stage). Near-certain: Wald SPRT on losses, H0 L = 1 - c,
    H1 L = 1 - c(1 + r1); sports: SPRT on net CLV, normal with the pre-set sd, H1 mean = r1 x mean cost.
    LLR divided by the design-time design effect; looks after every night/date; truncated at the horizon."""
    scen, var, r, reps, seed = args
    rng = np.random.default_rng(seed)
    A = np.log((1 - P["sprt_beta"]) / P["sprt_alpha"])
    B = np.log(P["sprt_beta"] / (1 - P["sprt_alpha"]))
    per_month = {"e4": P["e4_per_month"], "x1": P["x1_per_month"], "sp": P["sp_per_month"]}[scen]
    N = int(per_month * P["horizon_months"])
    out = []
    for _ in range(reps):
        if scen == "sp":
            g = gen_sp(rng, N, r, bias={"base": 0, "bias05": .005, "bias1": .01, "sd76": 0}[var],
                       sd=P["sp_clv_sd_stress"] if var == "sd76" else None)
            x = g["close"] - g["cost"]
            d1 = P["sprt_r1"] * float(np.mean(g["cost"]))
            s = P["sp_clv_sd"]
            llr = d1 / s ** 2 * (x - d1 / 2)
            deff = 1 + (1 + P["sp_bets_per_date"] - 1) * P["sp_icc_date"]
        else:
            g = gen_nc(rng, N, r, "e4", stress=(var == "stress")) if scen == "e4" else None
            L0 = 1 - g["cost"]
            L1 = np.clip(1 - g["cost"] * (1 + P["sprt_r1"]), 1e-6, 1)
            llr = np.where(g["loss"], np.log(L1 / L0), np.log((1 - L1) / (1 - L0)))
            lam = P["e4_fills_per_night"]
            deff = 1 + lam * P["icc_night"]                       # E[m^2]/E[m] - 1 = lam for Poisson nights
        cum = np.cumsum(llr) / deff
        ends = np.r_[np.flatnonzero(np.diff(g["night"])), len(cum) - 1]   # last fill of each night
        c = cum[ends]
        hi = np.flatnonzero(c >= A)
        lo = np.flatnonzero(c <= B)
        ih = hi[0] if len(hi) else 10 ** 9
        il = lo[0] if len(lo) else 10 ** 9
        if ih == il == 10 ** 9:
            out.append(("none", N))
        elif ih < il:
            out.append(("H1", int(ends[ih]) + 1))
        else:
            out.append(("H0", int(ends[il]) + 1))
    return scen, var, r, out


# ----------------------------------------------------------------------------------------------- summary
def cp(k, n, a=0.05):
    lo = stats.beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def rate(mask):
    mask = np.asarray(mask, bool)
    k, n = int(mask.sum()), int(len(mask))
    lo, hi = cp(k, n)
    return {"rate": k / n if n else None, "k": k, "reps": n, "lo": lo, "hi": hi}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    params = derive_params()
    (OUT / "params.json").write_text(json.dumps({k: v for k, v in params.items()
                                                 if k not in ("x1_prices",)}, indent=1))
    _init(params)
    ss = np.random.SeedSequence(MASTER_SEED)
    edges = [0.0, 0.01, 0.02, 0.04]
    grid = {"e4": [100, 200, 300, 500, 1000, 2000], "x1": [100, 200, 300, 500, 1000], "sp": [100, 200, 300, 500, 1000, 2000]}
    cur_grid = {"e4": [100, 300, 1000, 2000], "x1": [100, 300, 1000], "sp": [100, 300, 1000, 2000]}
    variants = {"e4": ["base", "stress"], "x1": ["base", "stress"], "sp": ["base", "sd76", "bias05", "bias1"]}
    scale = 0.1 if a.quick else 1.0
    tasks = []
    for scen in ("e4", "x1", "sp"):
        for var in variants[scen]:
            for r in edges:
                if var != "base" and r not in (0.0, 0.02):
                    continue
                for n in grid[scen]:
                    reps = int((20000 if r == 0 else 4000) * scale)
                    reps_cur = int((2000 if r == 0 else 1000) * scale) if (n in cur_grid[scen] and var == "base") else 0
                    if var != "base":
                        reps_cur = int(1000 * scale) if (n in cur_grid[scen] and r == 0) else 0
                    chunk = max(reps // 40, 50)
                    for j in range(0, reps, chunk):
                        rc = max(0, min(chunk, reps_cur - j))
                        tasks.append(("fixed", (scen, var, r, n, min(chunk, reps - j), rc,
                                                ss.spawn(1)[0].generate_state(1)[0])))
    for scen in ("e4", "sp"):
        for var in variants[scen]:
            for r in edges:
                if var != "base" and r not in (0.0, 0.02):
                    continue
                reps = int((10000 if r == 0 else 4000) * scale)
                chunk = max(reps // 20, 50)
                for j in range(0, reps, chunk):
                    tasks.append(("sprt", (scen, var, r, min(chunk, reps - j), ss.spawn(1)[0].generate_state(1)[0])))
    t0 = time.time()
    fixed, seq = {}, {}
    log = open(OUT / "run.log", "w")
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1), initializer=_init,
                             initargs=(params,)) as ex:
        futs = [ex.submit(cell_fixed if kind == "fixed" else cell_sprt, t) for kind, t in tasks]
        for i, f in enumerate(futs):
            res = f.result()
            if len(res) == 5:
                scen, var, r, n, rows = res
                fixed.setdefault((scen, var, r, n), []).extend(rows)
            else:
                scen, var, r, out = res
                seq.setdefault((scen, var, r), []).extend(out)
            if i % 50 == 0:
                print(f"{i}/{len(futs)} {time.time() - t0:.0f}s", file=log, flush=True)
    print(f"done {len(futs)} tasks {time.time() - t0:.0f}s", file=log, flush=True)

    # ---- fixed-n table
    recs = []
    for (scen, var, r, n), rows in sorted(fixed.items()):
        d = pd.DataFrame(rows)
        if scen == "sp":
            stage_alpha = {"dev": params["bar_sp"], "sealed": params["sealed_alpha_sp"], "fwd_fixed": ALPHA}
            min_n = params["min_n_sp"]
        else:
            stage_alpha = {"dev": params["bar_wx"], "sealed": params["sealed_alpha_wx"], "fwd_fixed": ALPHA}
            min_n = params["min_n_wx"]
        base = {"scenario": scen, "variant": var, "planted_roi": r, "n": n, "true_roi_mean": d.true_roi.mean()}
        for st, al in stage_alpha.items():
            mn = min_n if st != "fwd_fixed" else 0
            stat = (d.prop_p < al).to_numpy() & (n >= mn)
            prop = stat & d.loo.to_numpy() & d.consistent.to_numpy()
            stat_prior = (d.prop_p_prior < al).to_numpy() & (n >= mn)
            for rule, m in (("proposed_stat_only", stat), ("proposed_stat_prior_deff_only", stat_prior), ("proposed", prop),
                            ("proposed+halves", prop & d.halves.to_numpy())):
                recs.append({**base, "stage": st, "rule": rule, "nominal": al, **rate(m)})
        if "cur_pass" in d:
            c = d.dropna(subset=["cur_pass"])
            bar = params["bar_sp"] if scen == "sp" else params["bar_wx"]
            recs.append({**base, "stage": "all (same bar)", "rule": "current_pass", "nominal": bar,
                         **rate(c.cur_pass.astype(bool))})
            recs.append({**base, "stage": "all (same bar)", "rule": "current_pass_or_audit", "nominal": bar,
                         **rate(c.cur_pass.astype(bool) | c.cur_audit.astype(bool))})
            for ck in [k for k in c.columns if k.startswith("cur_") and k[4:6] in ("a_", "b_", "c_", "d_", "e_")]:
                recs.append({**base, "stage": "check", "rule": ck, "nominal": None, **rate(c[ck].astype(bool))})
            recs.append({**base, "stage": "check", "rule": "cur_bootstrap_p_lt_0.05", "nominal": None,
                         **rate((c.cur_p < 0.05).to_numpy())})
    fx = pd.DataFrame(recs)
    fx.to_csv(OUT / "fixed_n.csv", index=False)

    # ---- sequential table
    srec = []
    for (scen, var, r), out in sorted(seq.items()):
        dec = np.array([o[0] for o in out])
        nn = np.array([o[1] for o in out])
        pm = {"e4": params["e4_per_month"], "sp": params["sp_per_month"]}[scen]
        rec = {"scenario": scen, "variant": var, "planted_roi": r, "reps": len(out), "per_month": pm,
               "horizon_months": params["horizon_months"]}
        for lab in ("H1", "H0", "none"):
            rec.update({f"p_{lab}": rate(dec == lab)["rate"], f"p_{lab}_lo": rate(dec == lab)["lo"],
                        f"p_{lab}_hi": rate(dec == lab)["hi"]})
        for nb in (100, 300, 1000):
            rec[f"p_H1_by_{nb}"] = rate((dec == "H1") & (nn <= nb))["rate"]
            rec[f"p_H1_by_{nb}_hi"] = rate((dec == "H1") & (nn <= nb))["hi"]
        for mo in (6, 12, 24):
            rec[f"p_H1_by_{mo}mo"] = float(np.mean((dec == "H1") & (nn <= mo * pm)))
        d = dec != "none"
        rec["mean_fills_to_decision"] = float(nn[d].mean()) if d.any() else None
        rec["median_fills_to_decision"] = float(np.median(np.where(d, nn, np.inf)))
        rec["mean_months_to_decision"] = rec["mean_fills_to_decision"] / pm if d.any() else None
        rec["median_months_to_decision"] = rec["median_fills_to_decision"] / pm
        srec.append(rec)
    sq = pd.DataFrame(srec)
    sq.to_csv(OUT / "sprt.csv", index=False)

    # ---- n for 80% power at +2% (fixed rules, base) -> months at real volumes
    vol = {"e4": params["e4_per_month"], "x1": params["x1_per_month"], "sp": params["sp_per_month"]}
    need = []
    for (scen, st, rule), g in fx[(fx.variant == "base") & (fx.planted_roi == 0.02)].groupby(["scenario", "stage", "rule"]):
        if rule.startswith("cur_") or st == "check":
            continue
        g = g.sort_values("n")
        ok = g[g.rate >= 0.8]
        n80 = int(ok.n.iloc[0]) if len(ok) else None
        need.append({"scenario": scen, "stage": st, "rule": rule, "n80_on_grid": n80,
                     "max_n": int(g.n.max()), "power_at_max_n": float(g.rate.iloc[-1]),
                     "months_at_volume": (n80 / vol[scen]) if n80 else None, "volume_per_month": vol[scen]})
    pd.DataFrame(need).to_csv(OUT / "n80.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({"seed": MASTER_SEED, "quick": a.quick, "tasks": len(tasks),
                                                  "seconds": time.time() - t0}, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
