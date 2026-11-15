"""bar_calibration_v2 -- forward sequential tests revised per critic_bar_calibration (notes/bar_calibration_v2.md).

Run from sports/:  ../.venv_quant/bin/python results/bar_calibration_v2/sim2.py [--quick]
Outputs here: params.json, nc_tune.csv, nc_grid.csv, nc_valid.csv, sp_grid.csv, sp_valid.csv, family.json, run.log

Near-certain: loss-ratio H1 (L = rho1 x (1 - c)), real price mixes (live orig7 fills, sealed E1), seasonal volume, night
ICC + month shock (base / 3x stress), forward start month uniform. Sports: win/loss SPRT on ROI vs break-even at cost.
No edgelab code is modified; nothing registered.
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

SPORTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SPORTS))
from edgelab import fees as F  # noqa: E402

OUT = Path(__file__).resolve().parent
RUNS = SPORTS / "lab" / "edgelab_runs"
SEED = 20261116
HORIZON_NC = 36          # months
HORIZON_SP = 120         # in-season months
ALPHAS = [0.05, 0.04, 0.03, 0.025, 0.02, 0.015, 0.01, 0.005, 0.0025, 0.001]
BETAS = [0.1, 0.2]
RHO1 = [0.5, 0.6, 0.7]
MINM = [0, 6, 12]       # H1 may be accepted only after this many forward months (amendment 1)
SP_R1 = [0.02, 0.03, 0.05]


def month_v(frames):
    """Month-level extra-Poisson variance v (Var k = mu + v mu^2) of loss counts around a common loss ratio,
    moment estimate pooled over frames; also the Pearson dispersion."""
    num = den = chi = dfree = 0.0
    for d, cost in frames:
        loss = 1 - d["outcome"].to_numpy()
        L0 = 1 - cost
        m = d["event_start"].dt.strftime("%Y-%m").to_numpy()
        g = pd.DataFrame({"m": m, "k": loss, "e": L0}).groupby("m").sum()
        rho = g.k.sum() / g.e.sum()
        mu = rho * g.e
        num += float(((g.k - mu) ** 2 - mu).sum())
        den += float((mu ** 2).sum())
        chi += float(((g.k - mu) ** 2 / mu).sum())
        dfree += len(g) - 1
    return max(num / den, 0.0), chi / dfree


def derive():
    live = pd.read_parquet(SPORTS / "results/e4_live_audit/live_rows.parquet")
    o = live[(live.cohort == "orig7") & (live.filled > 0)]
    live_c = (o.avg_no + o.fee / o.filled).to_numpy(float)
    e1 = pd.read_parquet(RUNS / "e1_taker-40389097_sealed.parquet")
    e1_c = e1.price.to_numpy() + F.kalshi_fee_per_contract(e1.price.to_numpy(), 1, rounding="exact")
    e4s = pd.read_parquet(RUNS / "e4_maker_e1-f3f00f4d_sealed.parquet")
    e4s_c = e4s.price.to_numpy(float)
    e1d = pd.read_parquet(RUNS / "e1_taker-40389097_dev.parquet")
    e1d_c = e1d.price.to_numpy() + F.kalshi_fee_per_contract(e1d.price.to_numpy(), 1, rounding="exact")
    e4d = pd.read_parquet(RUNS / "e4_maker_e1-f3f00f4d_dev.parquet")
    v_seal, disp_seal = month_v([(e1, e1_c), (e4s, e4s_c)])
    v_dev, disp_dev = month_v([(e1d, e1d_c), (e4d, e4d.price.to_numpy(float))])
    # seasonal volume profile by calendar month from the E4 maker sealed frame (2025-07..2026-10; Jul-Oct appear twice)
    cm = e4s.event_start.dt.month.value_counts().reindex(range(1, 13)).fillna(0)
    yrs = e4s.event_start.dt.to_period("M").astype(str).str[5:7].astype(int).value_counts().reindex(range(1, 13))
    nmo = e4s.event_start.dt.strftime("%Y-%m").drop_duplicates().str[5:7].astype(int).value_counts().reindex(range(1, 13))
    prof = (cm / nmo).to_numpy(float)
    prof = prof / prof.mean()
    p = {
        "live_costs": live_c.round(4).tolist(), "e1_costs_n": len(e1_c),
        "e1_sealed_L": float(1 - e1.outcome.mean()), "e1_sealed_L0": float((1 - e1_c).mean()),
        "e1_sealed_roi": float((e1.outcome / e1_c).mean() - 1),
        "e4s_ratio": float((1 - e4s.outcome.mean()) / (1 - e4s_c).mean()),
        "month_v_sealed": v_seal, "month_disp_sealed": disp_seal, "month_v_dev": v_dev, "month_disp_dev": disp_dev,
        "season_profile": prof.round(3).tolist(), "nc_per_month": 55.0,
        "icc_night": 0.031, "sp_per_month": 60.0, "sp_bets_per_date": 7.0, "sp_icc_date": 0.05,
        "sp_price_lo": 0.49, "sp_price_hi": 0.535,
    }
    p["e1_ratio"] = p["e1_sealed_L"] / p["e1_sealed_L0"]
    p["month_v_base"] = max(v_seal, v_dev)
    p["month_v_stress"] = max(3 * p["month_v_base"], 0.15)
    return p, {"live": live_c, "e1": e1_c}


P: dict = {}
MIX: dict = {}


def _init(p, mix):
    global P, MIX
    P, MIX = p, mix


def shock(rng, L, icc, cl):
    """Shared Exp(1) shock per cluster with ~constant pairwise loss ICC, mean-preserving (as v1 sim.py)."""
    Ls = np.maximum(L, 1e-9)
    aa = np.sqrt(icc * (1 - Ls) / Ls)
    G = rng.exponential(1.0, cl.max() + 1)[cl] - 1
    c = np.where(aa > 1, aa * np.exp(-(1 - 1 / np.maximum(aa, 1e-12))), 1.0)
    return np.maximum(0, 1 + aa * G) / c


def stream_nc(rng, mix, rho, stress, seasonal_edge):
    """One forward stream of fills over HORIZON_NC months from a uniform start month. Returns cost, loss, night id,
    month index (0..)."""
    start = rng.integers(0, 12)
    prof = np.asarray(P["season_profile"])
    nights_per_month = 30
    lam = []
    mon = []
    for mi in range(HORIZON_NC):
        cal = (start + mi) % 12
        lam.append(np.full(nights_per_month, P["nc_per_month"] * prof[cal] / nights_per_month))
        mon.append(np.full(nights_per_month, mi))
    lam = np.concatenate(lam)
    mon_n = np.concatenate(mon)
    k = rng.poisson(lam)
    night = np.repeat(np.arange(len(k)), k)
    month = mon_n[night]
    cal = (start + month) % 12
    n = len(night)
    cost = rng.choice(MIX[mix], n)
    L0 = 1 - cost
    if seasonal_edge and rho != 1.0:
        # edge 1.6x stronger Dec-Mar (cal 11,0,1,2), weaker elsewhere; fill-weighted mean edge (1-rho) preserved
        s = np.where(np.isin(cal, [11, 0, 1, 2]), 1.6, 1.0)
        s = s / np.average(s, weights=L0)
        r = 1 - (1 - rho) * s
    else:
        r = np.full(n, rho)
    L = np.clip(r * L0, 0, 1)
    icc = P["icc_night"] * (3 if stress else 1)
    M = shock(rng, L, icc, night)
    v = P["month_v_stress"] if stress else P["month_v_base"]
    if v > 0:
        gm = rng.gamma(1 / v, v, HORIZON_NC)[month]
        M = M * gm
    loss = rng.random(n) < np.clip(L * M, 0, 1)
    return cost, loss, night, month


def first_cross(cum_at_ends, A, B, ok_h1=None):
    """First boundary crossing at a look; H1 acceptance only at looks where ok_h1 (min-information rule)."""
    hi = np.flatnonzero((cum_at_ends >= A) & (True if ok_h1 is None else ok_h1))
    lo = np.flatnonzero(cum_at_ends <= B)
    ih = hi[0] if len(hi) else 10 ** 9
    il = lo[0] if len(lo) else 10 ** 9
    if ih == il == 10 ** 9:
        return 0, -1
    return (1, ih) if ih < il else (-1, il)


def nc_task(args):
    mix, rho, stress, seasonal, reps, seed = args
    rng = np.random.default_rng(seed)
    deff = 1 + (P["nc_per_month"] / 30) * P["icc_night"]          # design-time: prior night ICC, mean volume
    out = []
    for _ in range(reps):
        cost, loss, night, month = stream_nc(rng, mix, rho, stress, seasonal)
        L0 = 1 - cost
        ends = np.r_[np.flatnonzero(np.diff(night)), len(night) - 1]
        mo_end = month[ends]
        for r1 in RHO1:
            llr = np.where(loss, np.log(r1), np.log((1 - r1 * L0) / (1 - L0)))
            c = np.cumsum(llr)[ends] / deff
            for mm in MINM:
                okh = mo_end >= mm
                for b in BETAS:
                    for a in ALPHAS:
                        d, i = first_cross(c, np.log((1 - b) / a), np.log(b / (1 - a)), okh)
                        out.append((r1, mm, b, a, d, int(mo_end[i]) + 1 if d else HORIZON_NC,
                                    int(ends[i]) + 1 if d else len(night)))
    return mix, rho, stress, seasonal, out


def sp_task(args):
    r, icc, per_month, reps, seed = args
    rng = np.random.default_rng(seed)
    N = int(per_month * HORIZON_SP)
    deff = 1 + (P["sp_bets_per_date"] - 1) * P["sp_icc_date"]
    out = []
    for _ in range(reps):
        k = np.maximum(rng.poisson(P["sp_bets_per_date"], int(N / P["sp_bets_per_date"] * 1.5) + 50), 1)
        cs = np.cumsum(k)
        last = int(np.searchsorted(cs, N))
        k = k[:last + 1].copy()
        k[-1] -= cs[last] - N
        date = np.repeat(np.arange(len(k)), k)
        price = rng.uniform(P["sp_price_lo"], P["sp_price_hi"], N)
        cost = price + F.kalshi_fee_per_contract(price, 1, rounding="exact")
        sd = 0.058                                                   # per-bet true-prob dispersion (v1 CLV sd)
        z = rng.normal(0, sd * np.sqrt(icc), len(k))[date]
        ptrue = np.clip(cost * (1 + r) + z + rng.normal(0, sd * np.sqrt(max(1 - icc, 0)), N), 0.02, 0.98)
        # shared date shock on outcomes too: correlated results (latent-variable, ICC on outcomes ~ icc)
        win = rng.random(N) < ptrue
        ends = np.r_[np.flatnonzero(np.diff(date)), N - 1]
        dates_per_month = len(k) / HORIZON_SP
        for r1 in SP_R1:
            p1 = np.clip(cost * (1 + r1), 0, 0.999)
            llr = np.where(win, np.log(p1 / cost), np.log((1 - p1) / (1 - cost)))
            c = np.cumsum(llr)[ends] / deff
            for b in BETAS:
                for a in ALPHAS:
                    d, i = first_cross(c, np.log((1 - b) / a), np.log(b / (1 - a)))
                    out.append((r1, b, a, d, (i + 1) / dates_per_month if d else HORIZON_SP, int(ends[i]) + 1 if d else N))
    return r, icc, per_month, out


def cp(k, n):
    lo = stats.beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def summarise(df, keys):
    rows = []
    for key, g in df.groupby(keys):
        n = len(g)
        k1, k0 = int((g.d == 1).sum()), int((g.d == -1).sum())
        lo1, hi1 = cp(k1, n)
        lo0, hi0 = cp(k0, n)
        dec = g[g.d != 0]
        rows.append(dict(zip(keys, key), reps=n, p_H1=k1 / n, H1_lo=lo1, H1_hi=hi1, p_kill=k0 / n, kill_lo=lo0,
                         kill_hi=hi0, p_none=1 - (k1 + k0) / n,
                         med_months_decided=float(dec.months.median()) if len(dec) else np.nan,
                         p_H1_24m=float(((g.d == 1) & (g.months <= 24)).mean()),
                         p_H1_12m=float(((g.d == 1) & (g.months <= 12)).mean()),
                         med_months_H1=float(g[g.d == 1].months.median()) if k1 else np.nan,
                         med_n_decided=float(dec.n.median()) if len(dec) else np.nan))
    return pd.DataFrame(rows)


def run(tasks, fn, p, mix, log):
    res = []
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1), initializer=_init, initargs=(p, mix)) as ex:
        for i, r in enumerate(ex.map(fn, tasks)):
            res.append(r)
            if i % 20 == 0:
                print(f"{fn.__name__} {i}/{len(tasks)}", file=log, flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    sc = 0.1 if a.quick else 1.0
    p, mix = derive()
    mmean = {k: float(np.mean((1 - v) / v)) for k, v in mix.items()}
    p["mix_mean_odds"] = mmean
    # rho for +1% / +1.5% ROI per mix (ROI = (1 - rho) mean((1-c)/c))
    rhos = {m: sorted({1.25, 1.0, 0.9, 0.8, round(p["e1_ratio"], 3), round(p["e4s_ratio"], 3), 0.5,
                       round(1 - 0.01 / mmean[m], 3), round(1 - 0.015 / mmean[m], 3)}, reverse=True) for m in mix}
    p["rhos"] = rhos
    (OUT / "params.json").write_text(json.dumps(p, indent=1))
    _init(p, mix)
    log = open(OUT / "run.log", "w")
    t0 = time.time()
    ss = np.random.SeedSequence(SEED)
    # ---- phase 1: near-certain grid (tuning placebos + edges), base/stress, flat/seasonal edge
    tasks = []
    for m in mix:
        for stress in (False, True):
            for rho in rhos[m]:
                for seasonal in ((False, True) if rho < 1 else (False,)):
                    reps = int((20000 if rho == 1.0 else 3000) * sc)
                    ch = max(reps // 40, 25)
                    for j in range(0, reps, ch):
                        tasks.append((m, rho, stress, seasonal, min(ch, reps - j), ss.spawn(1)[0].generate_state(1)[0]))
    res = run(tasks, nc_task, p, mix, log)
    rows = []
    for m, rho, stress, seasonal, out in res:
        for r1, mm, b, al, d, months, n in out:
            rows.append((m, rho, stress, seasonal, r1, mm, b, al, d, months, n))
    df = pd.DataFrame(rows, columns=["mix", "rho", "stress", "seasonal", "rho1", "minm", "beta", "alpha", "d", "months", "n"])
    grid = summarise(df, ["mix", "stress", "seasonal", "rho", "rho1", "minm", "beta", "alpha"])
    grid["roi"] = [(1 - r) * mmean[m] for m, r in zip(grid.mix, grid.rho)]
    grid.to_csv(OUT / "nc_grid.csv", index=False)
    print(f"nc grid done {time.time() - t0:.0f}s", file=log, flush=True)
    # tuning: largest alpha with stress placebo CP upper <= .05, per (mix, rho1, beta)
    tune = []
    for (m, r1, mm, b), g in grid[(grid.rho == 1.0) & grid.stress].groupby(["mix", "rho1", "minm", "beta"]):
        ok = g[g.H1_hi <= 0.05].sort_values("alpha", ascending=False)
        if len(ok):
            al = float(ok.alpha.iloc[0])
            e1 = grid[(grid.mix == m) & grid.stress & ~grid.seasonal & (grid.rho == round(p["e1_ratio"], 3)) &
                      (grid.rho1 == r1) & (grid.minm == mm) & (grid.beta == b) & (grid.alpha == al)].iloc[0]
            tune.append(dict(mix=m, rho1=r1, minm=mm, beta=b, alpha=al, stress_placebo=float(ok.p_H1.iloc[0]),
                             stress_placebo_hi=float(ok.H1_hi.iloc[0]), e1_kill=e1.p_kill, e1_H1=e1.p_H1,
                             e1_H1_24m=e1.p_H1_24m, e1_med_months=e1.med_months_decided))
    tune = pd.DataFrame(tune)
    tune.to_csv(OUT / "nc_tune.csv", index=False)
    # ---- phase 2: validate every tuned config on fresh-seed placebos (base + stress) and E1 edge
    ss2 = np.random.SeedSequence(SEED + 1)
    vt = []
    for m in mix:
        for stress in (False, True):
            for rho in (1.0, round(p["e1_ratio"], 3)):
                reps = int((20000 if rho == 1.0 else 4000) * sc)
                ch = max(reps // 40, 25)
                for j in range(0, reps, ch):
                    vt.append((m, rho, stress, False, min(ch, reps - j), ss2.spawn(1)[0].generate_state(1)[0]))
    res = run(vt, nc_task, p, mix, log)
    rows = [(m, rho, s, se, r1, mm, b, al, d, mo, n) for m, rho, s, se, out in res for r1, mm, b, al, d, mo, n in out]
    dv = pd.DataFrame(rows, columns=["mix", "rho", "stress", "seasonal", "rho1", "minm", "beta", "alpha", "d", "months", "n"])
    dv = dv.merge(tune[["mix", "rho1", "minm", "beta", "alpha"]], on=["mix", "rho1", "minm", "beta", "alpha"])
    summarise(dv, ["mix", "stress", "rho", "rho1", "minm", "beta", "alpha"]).to_csv(OUT / "nc_valid.csv", index=False)
    print(f"nc valid done {time.time() - t0:.0f}s", file=log, flush=True)
    # ---- sports ROI SPRT
    ss3 = np.random.SeedSequence(SEED + 2)
    st = []
    for icc in (P["sp_icc_date"], 3 * P["sp_icc_date"]):
        for per in (60.0, 200.0):
            for r in (0.0, -0.02, 0.01, 0.015, 0.02, 0.03, 0.05):
                if per == 200.0 and r in (-0.02, 0.015):
                    continue
                reps = int((4000 if r == 0 else 1000) * sc)
                ch = max(reps // 20, 25)
                for j in range(0, reps, ch):
                    st.append((r, icc, per, min(ch, reps - j), ss3.spawn(1)[0].generate_state(1)[0]))
    res = run(st, sp_task, p, mix, log)
    rows = [(r, icc, per, r1, b, al, d, mo, n) for r, icc, per, out in res for r1, b, al, d, mo, n in out]
    ds = pd.DataFrame(rows, columns=["roi", "icc", "per_month", "r1", "beta", "alpha", "d", "months", "n"])
    summarise(ds, ["per_month", "icc", "roi", "r1", "beta", "alpha"]).to_csv(OUT / "sp_grid.csv", index=False)
    print(f"all done {time.time() - t0:.0f}s", file=log, flush=True)


if __name__ == "__main__":
    main()
