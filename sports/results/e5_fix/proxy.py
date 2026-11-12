"""E5 fix trials on the Kalshi KXHIGH proxy (pre-registered in sports/notes/e5_fix.md).

    ../.venv_quant/bin/python results/e5_fix/proxy.py [dev|sealed]

Builds per-pick: q0 = NO best bid at T, side (above/below consensus), losing region, fill times (N=1/N=10) for every
price offset, obs-trigger cancel times from IEM ASOS running max; then scores B0, F1 x5, F2 x3, F3 x5 (+ descriptive
combos). Writes results/e5_fix/proxy_<split>.json and proxy_<split>_picks.parquet.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.path.expanduser("~/agent-harness/lib"))
from traps import check_epoch_seconds, checked_merge  # noqa: E402

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "dev"
OUTD = Path(__file__).parent
FEE = 0.01  # FX commission per contract
MODELS = ["gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless", "ncep_nbm_conus"]
TZ = {"KNYC": "America/New_York", "KMDW": "America/Chicago", "KMIA": "America/New_York", "KAUS": "America/Chicago",
      "KDEN": "America/Denver", "KLAX": "America/Los_Angeles", "KPHL": "America/New_York",
      "KATL": "America/New_York", "KBOS": "America/New_York", "KDFW": "America/Chicago", "KDCA": "America/New_York",
      "KEWR": "America/New_York", "KHOU": "America/Chicago", "KLAS": "America/Los_Angeles",
      "KMSP": "America/Chicago", "KMSY": "America/Chicago", "KOKC": "America/Chicago", "KPHX": "America/Phoenix",
      "KSAN": "America/Los_Angeles", "KSAT": "America/Chicago", "KSDF": "America/New_York",
      "KSEA": "America/Los_Angeles", "KSFO": "America/Los_Angeles", "KTTN": "America/New_York"}
OBS_LAG = 15 * 60
F1_HOURS = [0, 6, 9, 12, 15]
F2_X = [2, 4, 6]
F3_K = [2, 5, 10, 15, 20]
NBAR = 13
ALPHA = 0.05 / NBAR


def load_picks():
    sfx = "" if SPLIT == "dev" else "_sealed"
    p = pd.read_parquet(ROOT / f"data/kalshi_temp_maker_picks{sfx}.parquet")
    # strikes + consensus forecast at decision: E1 cities from wx_<CITY>.parquet, X1 from wx_expand.parquet
    parts = []
    for f in ["AUS", "CHI", "DEN", "LAX", "MIA", "NY", "PHIL"]:
        d = pd.read_parquet(ROOT / f"data/wx_{f}.parquet")
        parts.append(d[["ticker", "strike_type", "floor_strike", "cap_strike", "station"]
                       + [f"fcsafe_{m}" for m in MODELS]])
    d = pd.read_parquet(ROOT / "data/wx_expand.parquet")
    parts.append(d[["ticker", "strike_type", "floor_strike", "cap_strike", "station"] + [f"fcsafe_{m}" for m in MODELS]])
    s = pd.concat(parts).drop_duplicates("ticker")
    p = checked_merge(p, s, on="ticker")
    p["cons"] = p[[f"fcsafe_{m}" for m in MODELS]].mean(axis=1)
    p["tz"] = p.station.map(TZ)
    assert p.tz.notna().all(), p[p.tz.isna()].station.unique()
    check_epoch_seconds(p["T"], "T")
    check_epoch_seconds(p["close_ts"], "close_ts")
    # side + losing region (integer degF): greater -> YES iff v > floor ; less -> YES iff v < cap ; between [floor, cap]
    st = p.strike_type
    p["side"] = np.where(st == "greater", "above", np.where(st == "less", "below",
                         np.where(p.floor_strike > p.cons, "above", "below")))
    # above-side: NO loses once max >= L.  below-side: NO safe once max >= C (cleared the losing region)
    p["L"] = np.where(st == "greater", p.floor_strike + 1, p.floor_strike)
    p["C"] = np.where(st == "less", p.cap_strike, p.cap_strike + 1)
    p["q0"] = (1 - p.yes_ask).round(2)
    # local midnight of event day D, in epoch seconds
    mid = [pd.Timestamp(d.date(), tz=z) for d, z in zip(p.date, p.tz)]
    p["mid_D"] = [(m - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1) for m in mid]
    check_epoch_seconds(p["mid_D"], "mid_D")
    assert ((p["T"] < p.mid_D) & (p.mid_D < p.close_ts)).mean() > 0.95, "T / local midnight / close ordering off"
    return p


def obs_triggers(p):
    o = pd.read_parquet(OUTD / "asos_obs.parquet")
    o["stn"] = "K" + o.stn.astype(str).str[-3:]
    o["ts"] = (o.valid - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)
    check_epoch_seconds(o["ts"], "obs ts")
    g = {k: v[["ts", "tmpf"]].to_numpy() for k, v in o.groupby("stn")}
    trig = {x: np.full(len(p), np.inf) for x in F2_X}
    clear_t = np.full(len(p), np.inf)
    nobs = np.zeros(len(p), int)
    for i, r in enumerate(p.itertuples()):
        a = g.get(r.station)
        if a is None:
            continue
        w = a[(a[:, 0] >= r.mid_D) & (a[:, 0] <= r.close_ts)]
        nobs[i] = len(w)
        if not len(w):
            continue
        rm = np.maximum.accumulate(w[:, 1])
        avail = w[:, 0] + OBS_LAG
        if r.side == "above":
            for x in F2_X:
                h = np.nonzero(rm >= r.L - x)[0]
                if len(h):
                    trig[x][i] = avail[h[0]]
        else:
            h = np.nonzero(rm >= r.C)[0]
            if len(h):
                clear_t[i] = avail[h[0]]
    for x in F2_X:
        p[f"trig{x}"] = trig[x]
    p["clear_t"] = clear_t
    p["nobs"] = nobs
    return p


def fill_times(p, tr, q, n):
    """first time the cumulative printed count at no_price <= q in (T, close] reaches n; inf if never."""
    out = np.full(len(p), np.inf)
    by = {k: v.sort_values("ts")[["ts", "no_price", "count"]].to_numpy() for k, v in tr.groupby("ticker")}
    for i, (tk, qq, T, c) in enumerate(zip(p.ticker, q, p["T"], p.close_ts)):
        a = by.get(tk)
        if a is None:
            continue
        a = a[(a[:, 0] > T) & (a[:, 0] <= c) & (a[:, 1] <= qq + 1e-9)]
        if not len(a):
            continue
        cs = np.cumsum(a[:, 2])
        h = np.nonzero(cs >= n)[0]
        if len(h):
            out[i] = a[h[0], 0]
    return out


def boot(dates, pnl, bet, rng, n=4000):
    ud, inv = np.unique(dates, return_inverse=True)
    sp = np.bincount(inv, weights=np.where(bet, pnl, 0), minlength=len(ud))
    sb = np.bincount(inv, weights=bet.astype(float), minlength=len(ud))
    idx = rng.integers(0, len(ud), (n, len(ud)))
    num, den = sp[idx].sum(1), sb[idx].sum(1)
    roi = np.where(den > 0, num / np.maximum(den, 1), 0)
    return np.percentile(roi, [2.5, 97.5]), float((roi <= 0).mean())


def score(p, name, q, filled, rng):
    won = p.won.to_numpy() == 1
    cost = q + FEE
    pnl = np.where(won, 1 / cost - 1, -1.0)
    n = int(filled.sum())
    roi = float(pnl[filled].mean()) if n else float("nan")
    ci, pv = boot(p.date.to_numpy(), pnl, filled, rng) if n else ((np.nan, np.nan), 1.0)
    ndays = p.date.nunique()
    return {"variant": name, "n_picks": len(p), "fills": n, "fill_rate": round(n / len(p), 3),
            "losses_filled": int((filled & ~won).sum()), "losses_all": int((~won).sum()),
            "loss_rate_filled": round(float((~won[filled]).mean()), 4) if n else None,
            "roi": round(roi, 4), "ci": [round(float(ci[0]), 4), round(float(ci[1]), 4)], "p": round(pv, 4),
            "units": round(float(pnl[filled].sum()), 2), "fills_per_day": round(n / ndays, 2),
            "mean_cost_filled": round(float(cost[filled].mean()), 3) if n else None,
            "pass": bool(n and roi > 0 and ci[0] > 0 and pv < ALPHA)}


def main():
    p = load_picks()
    p = obs_triggers(p)
    sfx = "" if SPLIT == "dev" else "_sealed"
    tr = pd.read_parquet(ROOT / f"data/kalshi_temp_maker_trades{sfx}.parquet")
    tr = tr[tr.ticker.isin(set(p.ticker))]
    check_epoch_seconds(tr["ts"], "trade ts")
    q0 = p.q0.to_numpy()
    ft = {}
    for n in (1, 10):
        for k in [0] + F3_K:
            ft[(n, k)] = fill_times(p, tr, q0 - k / 100, n)
            p[f"ft_n{n}_k{k}"] = ft[(n, k)]
    rng = np.random.default_rng(0)
    res = {}
    for n in (10, 1):
        R = []
        f0 = ft[(n, 0)]
        R.append(score(p, "B0 rest to close", q0, f0 < np.inf, rng))
        for h in F1_HOURS:
            cut = p.mid_D.to_numpy() + h * 3600
            R.append(score(p, f"F1 cancel {h:02d}:00 local D", q0, f0 <= cut, rng))
        for x in F2_X:
            cancel = np.where(p.side == "above", p[f"trig{x}"],
                              np.where(p.clear_t < p.mid_D + 12 * 3600, np.inf, p.mid_D + 12 * 3600))
            R.append(score(p, f"F2 obs X={x}F", q0, f0 < cancel, rng))
        for k in F3_K:
            R.append(score(p, f"F3 bid q0-{k}c", q0 - k / 100, ft[(n, k)] < np.inf, rng))
        # descriptive combos
        for k in (5, 10):
            for h in (9, 12):
                cut = p.mid_D.to_numpy() + h * 3600
                R.append(score(p, f"[desc] F1 {h:02d}:00 x F3 -{k}c", q0 - k / 100, ft[(n, k)] <= cut, rng))
            cancel = np.where(p.side == "above", p["trig4"],
                              np.where(p.clear_t < p.mid_D + 12 * 3600, np.inf, p.mid_D + 12 * 3600))
            R.append(score(p, f"[desc] F2 X=4 x F3 -{k}c", q0 - k / 100, ft[(n, k)] < cancel, rng))
        res[f"N{n}"] = R
    # timing profile of fills (B0, N10): hours after local midnight D, by outcome
    f0 = ft[(10, 0)]
    m = f0 < np.inf
    hrs = (f0[m] - p.mid_D.to_numpy()[m]) / 3600
    won = p.won.to_numpy()[m] == 1
    bins = [-6, 0, 6, 9, 12, 15, 18, 30]
    res["timing_N10"] = {"bins_h_after_local_midnight": bins,
                         "winners": np.histogram(hrs[won], bins)[0].tolist(),
                         "losers": np.histogram(hrs[~won], bins)[0].tolist()}
    res["meta"] = {"split": SPLIT, "n_picks": len(p), "dates": [str(p.date.min().date()), str(p.date.max().date())],
                   "side_counts": p.side.value_counts().to_dict(), "picks_no_obs": int((p.nobs == 0).sum()),
                   "bar_p": ALPHA, "fee": FEE}
    json.dump(res, open(OUTD / f"proxy_{SPLIT}.json", "w"), indent=1)
    p.to_parquet(OUTD / f"proxy_{SPLIT}_picks.parquet")
    for key in ("N10", "N1"):
        print(key)
        print(pd.DataFrame(res[key]).drop(columns=["n_picks"]).to_string(index=False))
    print(json.dumps(res["timing_N10"]), json.dumps(res["meta"]))


if __name__ == "__main__":
    main()
