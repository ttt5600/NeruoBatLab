"""N0 nowcast calibration (prereg notes/home_values.md). Zillow data only, no Kalshi prices.

(a) panel c_h on the current (revised) City SFR smoothed-SA file, cities SizeRank <= 3000, origins 2015-01..(end-h);
(b) real-time check with Wayback vintages: forecast from the 2023-04-12 vintage, target = first-published value in the
    2023-05-15 vintage; revision of first-published months vs the current file.
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

H = Path(__file__).resolve().parent
PHI = 0.85


def load(f, max_rank=3000):
    d = pd.read_csv(f)
    d = d[d.SizeRank <= max_rank] if max_rank else d
    cols = [c for c in d.columns if c[:2] in ("19", "20")]
    X = np.log(d[cols].to_numpy(float))
    return d.RegionID.to_numpy(), pd.to_datetime(cols), X


def forecast(X, L, h):
    """X[:, :L+1] known. returns mu (log), s_city."""
    d = np.diff(X[:, : L + 1], axis=1)
    dL = d[:, -1]
    mu = X[:, L] + dL * sum(PHI ** k for k in range(1, h + 1))
    dd = np.diff(d[:, -37:], axis=1)
    s = np.nanstd(dd, axis=1)
    return mu, s


def panel(ids, dates, X, hs=(1, 2, 3), start="2015-01-31"):
    out = []
    i0 = int(np.searchsorted(dates, pd.Timestamp(start)))
    for h in hs:
        for L in range(i0, len(dates) - h):
            mu, s = forecast(X, L, h)
            e = X[:, L + h] - mu
            ok = np.isfinite(e) & np.isfinite(s) & (s > 0)
            out.append(pd.DataFrame({"h": h, "origin": dates[L], "rid": ids[ok], "err": e[ok], "s": s[ok]}))
    p = pd.concat(out, ignore_index=True)
    p["z"] = p.err / p.s
    return p


if __name__ == "__main__":
    ids, dates, X = load(H / "raw" / "City_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month.csv")
    p = panel(ids, dates, X)
    p.to_parquet(H / "calib_panel.parquet")
    res = {"n_cities": int(len(ids)), "last_month": str(dates[-1].date()), "by_h": {}}
    for h, g in p.groupby("h"):
        az = g.z.abs()
        q95, q99 = az.quantile(.95), az.quantile(.99)
        c = q95 / 1.96
        # tail check: share of |z| > 1.645*c*1.5 / 1.96*c*1.5 (what the prereg sigma would call 10% / 5% two-sided)
        res["by_h"][int(h)] = dict(n=int(len(g)), q50=float(az.median()), q95=float(q95), q99=float(q99), c_h=float(c),
                                   frac_beyond_1p96_c=float((az > 1.96 * c).mean()),
                                   frac_beyond_1p96_c15=float((az > 1.96 * c * 1.5).mean()),
                                   frac_beyond_2p58_c=float((az > 2.576 * c).mean()),
                                   abs_err_pct_median=float(g.err.abs().median() * 100),
                                   abs_err_pct_q95=float(g.err.abs().quantile(.95) * 100))
        # by year: does tail width drift (2020-22 boom/turn)?
        yy = g.assign(y=g.origin.dt.year).groupby("y").z.apply(lambda v: float(v.abs().quantile(.95)))
        res["by_h"][int(h)]["q95_by_year"] = {int(k): round(v, 2) for k, v in yy.items()}
    # (b) Wayback real-time check
    W = H / "raw" / "wb"
    iA, dA, XA = load(W / "v_20230412124124.csv")
    iB, dB, XB = load(W / "v_20230515120042.csv", max_rank=None)
    res["wb"] = {"A_last": str(dA[-1].date()), "B_last": str(dB[-1].date())}
    LA = len(dA) - 1
    mu, s = forecast(XA, LA, 1)
    B = pd.DataFrame(XB[:, -1], index=iB, columns=["tgt"])  # first-published for B_last
    A = pd.DataFrame({"mu": mu, "s": s, "lastA": XA[:, LA]}, index=iA)
    j = A.join(B, how="inner")
    # also B's restatement of A's last month
    j = j.join(pd.DataFrame(XB[:, -2], index=iB, columns=["lastA_inB"]), how="inner")
    j = j[np.isfinite(j.tgt) & np.isfinite(j.mu) & (j.s > 0)]
    j["z_rt"] = (j.tgt - j.mu) / j.s
    j["rev1"] = (j.lastA_inB - j.lastA)  # one-release restatement of the latest month (log)
    # same origin/horizon on the revised current file, same cities
    cur = p[(p.h == 1) & (p.origin == pd.Timestamp(dA[-1]))].set_index("rid")
    jj = j.join(cur[["z"]].rename(columns={"z": "z_revised"}), how="inner")
    res["wb"].update(n=int(len(j)), q50_rt=float(j.z_rt.abs().median()), q95_rt=float(j.z_rt.abs().quantile(.95)),
                     q99_rt=float(j.z_rt.abs().quantile(.99)), mean_z_rt=float(j.z_rt.mean()),
                     q95_revised_same=float(jj.z_revised.abs().quantile(.95)) if len(jj) else None,
                     n_same=int(len(jj)),
                     rev1_pct_median_abs=float(j.rev1.abs().median() * 100),
                     rev1_pct_q95_abs=float(j.rev1.abs().quantile(.95) * 100),
                     rev1_over_s_q95=float((j.rev1.abs() / j.s).quantile(.95)))
    # first-published (vintages) vs current file for the same month: long-run restatement size
    ic, dc, Xc = load(H / "raw" / "City_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month.csv", max_rank=None)
    C = pd.DataFrame(Xc, index=ic, columns=dc)
    for nm, (iv, dv, Xv) in {"2022-10-27": load(W / "v_20221027175129.csv"), "2023-05-15": (iB, dB, XB)}.items():
        m = dv[-1]
        v = pd.Series(Xv[:, -1], index=iv)
        cc = C[m].reindex(v.index)
        r = (cc - v).dropna()
        res["wb"][f"restate_{nm}_month_{m.date()}"] = dict(n=int(len(r)), med_abs_pct=float(r.abs().median() * 100),
                                                          q95_abs_pct=float(r.abs().quantile(.95) * 100))
    json.dump(res, open(H / "calib.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
