"""E4 better-data study (agent e4_data_v2). DEV ONLY; no sealed row is loaded. Pre-registration: notes/e4_data_v2.md.

    python scripts/e4_data_v2.py headroom    # step 1: C22 oracle rows on dev (taker units + maker where tapes exist)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from scipy import optimize, stats as sst

from sports import wx
from kalshi_paper import PAPER_CITIES
import wx_timing as WT

OUT = ROOT / "results" / "e4_data_v2"
OUT.mkdir(parents=True, exist_ok=True)
FEE = 0.07
MODELS = list(WT.MODELS)


# ------------------------------------------------------------------------------------------------ shared frame
def dev_frame() -> pd.DataFrame:
    """Dev universe + 22:00 quote + decision ts + E4 flag. Sealed rows are dropped inside WT.universe()."""
    u = WT.universe()
    px = pd.concat([pd.read_parquet(ROOT / "data" / f"wx_{s.replace('KXHIGH', '')}.parquet")[["ticker", "yes_bid", "yes_ask", "price_ts"]]
                    for s in wx.CITIES] + [pd.read_parquet(ROOT / "data" / "wx_expand.parquet")[["ticker", "yes_bid", "yes_ask", "price_ts"]]]
                   ).drop_duplicates("ticker")
    n0 = len(u)
    u = u.merge(px, on="ticker", how="left", validate="one_to_one")
    assert len(u) == n0
    u["station"] = u.series.map(lambda s: PAPER_CITIES[s][0])
    u["tz"] = u.series.map(lambda s: PAPER_CITIES[s][1])
    u["T"] = [WT.t_local(d, tz, 22) for d, tz in zip(u.date, u.tz)]
    u["close_ts"] = (pd.to_datetime(u.close_time, utc=True) - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
    assert (u.price_ts <= u["T"]).all(), "quote after decision"
    b, a = u.yes_bid, u.yes_ask
    u["tradeable"] = ((b > 0) & (a < 1) & ((a - b) <= 0.10) & (a > 0.01)).to_numpy()
    for m in MODELS:
        u[f"fc22_{m}"] = u[f"fcsafe_{m}"]
    u["e4"] = WT.rejected(u, 22, 3.0)
    u["won"] = (u.result == "no").astype(int)
    u["y"] = u.expiration_value.astype(float)
    ask = 1 - u.yes_bid
    u["ask"] = ask
    u["tcost"] = ask + FEE * ask * (1 - ask)
    u["q"] = np.round(np.minimum(np.floor(np.round((1 - (u.yes_bid + u.yes_ask) / 2) * 100, 6)) / 100, ask - 0.01), 4)
    return u


def reject_at(f, c, st, hi, lo, M):
    return np.where(st == "greater", hi + M < f + 1, np.where(st == "less", lo - M > c - 1, (hi + M < f) | (lo - M > c)))


def tapes() -> dict:
    fr, seen = [], set()
    for f in [ROOT / "data" / "kalshi_temp_maker_trades.parquet", ROOT / "results" / "e4_timing" / "trades.parquet",
              ROOT / "results" / "e4_probabilistic" / "trades.parquet", OUT / "trades_extra.parquet"]:
        if not f.exists():
            continue
        t = pd.read_parquet(f)
        t = t[~t.ticker.isin(seen)]
        seen |= set(t.ticker)
        fr.append(t)
    tr = pd.concat(fr, ignore_index=True)
    return {k: v.sort_values("ts") for k, v in tr.groupby("ticker")}


def maker_cols(s: pd.DataFrame, g: dict) -> pd.DataFrame:
    """has_tape, filled (trade-through: any print NO < q in (T, close]), v_lt (contracts printed NO < q in window)."""
    has, fil, vlt = [], [], []
    for tk, t0, t1, q in zip(s.ticker, s["T"], s.close_ts, s.q):
        d = g.get(tk)
        if d is None:
            has.append(False); fil.append(False); vlt.append(0.0)
            continue
        w = (d.ts.to_numpy() > t0) & (d.ts.to_numpy() <= t1) & (d.no_price.to_numpy() < q - 1e-9)
        has.append(True); fil.append(bool(w.any())); vlt.append(float(d["count"].to_numpy()[w].sum()))
    return s.assign(has_tape=has, filled=fil, v_lt=vlt)


def pnl(won, cost):
    return np.where(won == 1, 1 / cost - 1, -1.0)


def boot_units(df, days, col="pnl", n=4000, seed=0):
    d = df.groupby(df.date.dt.strftime("%Y-%m-%d"))[col].sum().reindex(days, fill_value=0.0).to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(days), size=(n, len(days)))
    bs = d[idx].sum(1)
    return [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], float(np.mean(bs <= 0))


def taker_row(s, days, name):
    s = s.assign(pnl=pnl(s.won.to_numpy(), s.tcost.to_numpy()))
    ci, p = boot_units(s, days)
    return {"row": name, "bets": int(len(s)), "losses": int((s.won == 0).sum()), "units": round(float(s.pnl.sum()), 2),
            "units_ci": [round(x, 2) for x in ci], "roi": round(float(s.pnl.mean()), 4) if len(s) else None,
            "per_night": round(len(s) / len(days), 2), "mean_ask": round(float(s.ask.mean()), 4) if len(s) else None}


def maker_row(s, days, name):
    k = s[s.has_tape]
    f = k[k.filled].assign(pnl=lambda x: pnl(x.won.to_numpy(), x.q.to_numpy()))
    ci, p = boot_units(f, days) if len(f) else ([0, 0], 1.0)
    return {"row": name, "picks": int(len(s)), "tape_cov": round(float(s.has_tape.mean()), 3), "fills": int(len(f)),
            "losses": int((f.won == 0).sum()), "unfilled_losses": int(((~k.filled) & (k.won == 0)).sum()),
            "units": round(float(f.pnl.sum()), 2), "units_ci": [round(x, 2) for x in ci],
            "roi": round(float(f.pnl.mean()), 4) if len(f) else None}


# ------------------------------------------------------------------------------------------------ step 1
def emos_insample(u: pd.DataFrame) -> np.ndarray:
    """R5: optimistic in-sample t-EMOS on ALL dev station-days (e4p form + station intercepts). Returns P(hit) per row."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import e4_probabilistic as EP
    nb = EP.nbs_daily()
    v = u.merge(nb, on=["station", "date"], how="left", validate="many_to_one")
    cols = [f"fcsafe_{m}" for m in MODELS]
    v["n_mod"] = v[cols].notna().sum(axis=1)
    v["m"], v["s2"] = v[cols].mean(axis=1), v[cols].var(axis=1, ddof=0)
    stns = sorted(v.station.unique())
    S = {s: i for i, s in enumerate(stns)}
    v["si"] = v.station.map(S)
    ok = v.n_mod >= 3
    sd = v[ok].drop_duplicates("event_ticker")
    out = np.full(len(v), np.nan)
    for has in (True, False):
        X = sd[sd.txn.notna()] if has else sd
        y = X.y.to_numpy(float)
        si = X.si.to_numpy()

        def musig(th, X, si):
            a = th[:len(stns)]
            b1, b2, c0, c1, c2, d = th[len(stns):]
            mu = a[si] + b1 * X.m.to_numpy(float) + (b2 * X.txn.fillna(0).to_numpy(float) if has else 0)
            sig = np.sqrt(np.exp(c0) + (np.exp(c1) * X.xnd.fillna(0).to_numpy(float) ** 2 if has else 0) + np.exp(c2) * X.s2.to_numpy(float))
            return mu, sig, 2 + np.exp(d)

        def nll(th):
            mu, sig, df = musig(th, X, si)
            return -sst.t.logpdf(y, df, mu, sig).sum()
        th0 = np.r_[np.zeros(len(stns)), .5 if has else 1.0, .5 if has else 0.0, np.log(2), 0.0, 0.0, np.log(5)]
        th = optimize.minimize(nll, th0, method="L-BFGS-B").x
        k = (ok & (v.txn.notna() if has else v.txn.isna())).to_numpy()
        R = v[k]
        mu, sig, df = musig(th, R, R.si.to_numpy())
        F = lambda x: sst.t.cdf((x - mu) / sig, df)
        f, c, st = R.floor_strike.to_numpy(float), R.cap_strike.to_numpy(float), R.strike_type.to_numpy()
        out[k] = np.where(st == "between", F(c + .5) - F(f - .5), np.where(st == "greater", 1 - F(f + .5), F(c - .5)))
    return out


def cmd_headroom() -> int:
    u = dev_frame()
    t = u[u.tradeable].copy()
    days = pd.Index(sorted(t.date.dt.strftime("%Y-%m-%d").unique()))
    g = tapes()
    t = maker_cols(t, g)
    f, c, st, y = t.floor_strike.to_numpy(float), t.cap_strike.to_numpy(float), t.strike_type.to_numpy(), t.y.to_numpy()
    rows, mrows = [], []
    R0 = t[t.e4]
    rows.append(taker_row(R0, days, "R0 E4 frozen"))
    mrows.append(maker_row(R0, days, "R0 E4 frozen (maker TT)"))
    rows.append(taker_row(R0[R0.won == 1], days, "R1 E4 minus losers (loss veto)"))
    mrows.append(maker_row(R0[R0.won == 1], days, "R1 E4 minus losers (maker TT)"))
    for M in (3.0, 2.0, 1.0, 0.0):
        k = reject_at(f, c, st, y, y, M)
        rows.append(taker_row(t[k], days, f"R2/R3 perfect forecast margin {M:g}"))
        mrows.append(maker_row(t[k], days, f"R2/R3 perfect forecast margin {M:g} (maker TT)"))
    chk = int((reject_at(f, c, st, y, y, 0.0) != (t.won == 1).to_numpy()).sum())
    # skill curve
    cols = [f"fcsafe_{m}" for m in MODELS]
    sd = u.drop_duplicates("event_ticker")
    sd = sd[sd[cols].notna().sum(axis=1) >= 3]

    def tail(s_, M):
        mx = sd.y + s_ * (sd[cols].max(axis=1) - sd.y)
        mn = sd.y + s_ * (sd[cols].min(axis=1) - sd.y)
        return float(((sd.y > mx + M) | (sd.y < mn - M)).mean())
    target = tail(1.0, 3.0)
    curve = []
    for s_ in (1.0, .9, .8, .7, .5, .25, 0.0):
        sh = t.y.to_numpy()[:, None] + s_ * (t[cols].to_numpy(float) - t.y.to_numpy()[:, None])
        n = np.isfinite(sh).sum(1)
        hi, lo = np.nanmax(np.where(np.isfinite(sh), sh, -np.inf), 1), np.nanmin(np.where(np.isfinite(sh), sh, np.inf), 1)
        Ms = next(M for M in np.arange(0, 10.01, .5) if tail(s_, M) <= target)
        for M, lab in ((3.0, "fixed M3"), (Ms, f"retuned M{Ms:g}")):
            k = reject_at(f, c, st, hi, lo, M) & (n >= 3)
            r = taker_row(t[k], days, f"R4 s={s_:g} {lab}")
            r["maker"] = maker_row(t[k], days, "")
            curve.append(r)
    # R5 in-sample EMOS
    t["phit_is"] = emos_insample(t)
    r5 = []
    for ps in (.005, .01, .02, .03, .05):
        k = (t.phit_is <= ps).to_numpy()
        r = taker_row(t[k], days, f"R5 in-sample EMOS p*={ps}")
        r["new"] = taker_row(t[k & ~t.e4.to_numpy()], days, "new only")
        r["union_E4"] = taker_row(t[k | t.e4.to_numpy()], days, "union with E4")
        r5.append(r)
    # R6 capacity on E4 dev picks
    k = R0[R0.has_tape]
    cap = []
    for S in (1, 6, 24, 96, 10 ** 9):
        fill = np.minimum(S, k.v_lt.to_numpy())
        usd = np.where(k.won == 1, fill * (1 - k.q), -fill * k.q)
        cap.append({"S": "tape" if S > 1e6 else S, "usd_per_night": round(float(usd.sum() / len(days)), 3),
                    "staked_per_night": round(float((fill * k.q).sum() / len(days)), 2),
                    "contracts_filled": int(fill.sum())})
    # per-station headroom table (C22: weight x headroom)
    per = []
    for sname, x in t.groupby("station"):
        e = x[x.e4]
        pf = x[reject_at(x.floor_strike.to_numpy(float), x.cap_strike.to_numpy(float), x.strike_type.to_numpy(), x.y.to_numpy(), x.y.to_numpy(), 3.0)]
        pe, pp = pnl(e.won.to_numpy(), e.tcost.to_numpy()), pnl(pf.won.to_numpy(), pf.tcost.to_numpy())
        per.append({"station": sname, "days": int(x.date.nunique()), "e4_bets": int(len(e)), "e4_losses": int((e.won == 0).sum()),
                    "e4_units": round(float(pe.sum()), 2), "pf3_bets": int(len(pf)), "pf3_units": round(float(pp.sum()), 2),
                    "headroom": round(float(pp.sum() - pe.sum()), 2)})
    res = {"days": len(days), "tradeable": int(len(t)), "R3_equals_result_check_mismatches": chk, "tail_target": target,
           "taker": rows, "maker": mrows, "skill_curve": curve, "R5": r5, "capacity_R6": cap,
           "per_station": sorted(per, key=lambda r: -r["headroom"])}
    (OUT / "headroom.json").write_text(json.dumps(res, indent=1, default=float))
    for r in rows + mrows:
        print(r)
    print("tail target", target)
    for r in curve:
        print({k: v for k, v in r.items() if k != "maker"}, "| maker", {k: r["maker"][k] for k in ("fills", "losses", "units", "tape_cov")})
    for r in r5:
        print(r["row"], r["bets"], r["losses"], r["units"], r["units_ci"], "| new", r["new"]["bets"], r["new"]["losses"], r["new"]["units"],
              r["new"]["units_ci"], "| union", r["union_E4"]["bets"], r["union_E4"]["units"])
    for r in cap:
        print(r)
    for r in res["per_station"]:
        print(r)
    return 0




# ------------------------------------------------------------------------------------------------ step 2: DGP
import re
from datetime import datetime as _dt
from zoneinfo import ZoneInfo as _Z

_RMK = re.compile(r"\bRMK\b(.*)$")
_G24 = re.compile(r"(?:^|\s)4([01])(\d{3})([01])(\d{3})(?=\s|$)")
_G6X = re.compile(r"(?:^|\s)1([01])(\d{3})(?=\s|$)")


def c2f(sign, ttt):
    c = (-1 if sign == "1" else 1) * int(ttt) / 10.0
    return c * 9 / 5 + 32


def rhu(x):  # round half up (NWS convention for F)
    return np.floor(np.asarray(x, float) + 0.5)


def cmd_dgp() -> int:
    o = pd.read_parquet(ROOT / "data" / "wx_lockin_obs.parquet")
    days = pd.read_parquet(ROOT / "data" / "wx_lockin_days.parquet")
    o["vts"] = (o.valid - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
    rm = o.metar.astype(str).str.extract(_RMK)[0].fillna("")
    g24 = rm.str.extract(_G24)
    o["max24"] = [c2f(a, b) if isinstance(a, str) else np.nan for a, b in zip(g24[0], g24[1])]
    g6 = rm.str.extract(_G6X)
    o["max6"] = [c2f(a, b) if isinstance(a, str) else np.nan for a, b in zip(g6[0], g6[1])]
    o["flag"] = o.metar.astype(str).str.strip().str.endswith("$")
    og = {k: v.sort_values("vts") for k, v in o.groupby("station")}
    rows = []
    for r in days.itertuples(index=False):
        z = _Z(r.tz)
        std = z.utcoffset(_dt(2024, 1, 15))
        d0 = pd.Timestamp(r.date)
        s_lst = int((_dt(d0.year, d0.month, d0.day) - std).replace(tzinfo=_Z("UTC")).timestamp())
        e_lst = s_lst + 86400
        s_loc = int(_dt(d0.year, d0.month, d0.day, tzinfo=z).timestamp())
        x = og.get(r.station)
        if x is None:
            continue
        w = x[(x.vts >= s_lst) & (x.vts < e_lst)]
        if len(w) < 12:
            continue
        tf = w.tf.to_numpy(float)
        i = int(np.nanargmax(tf))
        tmax_ts = int(w.vts.iloc[i])
        loc18 = x[(x.vts >= s_loc) & (x.vts <= s_loc + 18 * 3600 + 1800)]
        g6w = w[(w.vts >= s_lst + 6 * 3600 - 900) & w.max6.notna()]
        g24w = x[(x.vts >= e_lst - 5400) & (x.vts < e_lst + 1800) & x.max24.notna()]
        rows.append({"station": r.station, "series": r.series, "cohort": r.cohort, "date": d0, "event_ticker": r.event_ticker,
                     "y": r.expiration_value, "n_obs": len(w), "hmax": float(np.nanmax(tf)),
                     "tmax_lst_h": (tmax_ts - s_lst) / 3600.0, "tmax_loc_h": (tmax_ts - s_loc) / 3600.0,
                     "hmax_loc18": float(loc18.tf.max()) if len(loc18) else np.nan,
                     "max6": float(rhu(g6w.max6.max())) if len(g6w) else np.nan,
                     "max24": float(rhu(g24w.max24.iloc[-1])) if len(g24w) else np.nan,
                     "max24_raw": float(g24w.max24.iloc[-1]) if len(g24w) else np.nan,
                     "flag_any": bool(w.flag.any()), "dst": bool(z.utcoffset(_dt(d0.year, d0.month, d0.day, 12)) != std)})
    D = pd.DataFrame(rows)
    D.to_parquet(OUT / "dgp_days.parquet", index=False)
    res = {"station_days": len(D)}
    k = D.max24.notna()
    g = (D.y - D.max24)[k]
    res["cli_minus_24hgroup"] = {"n": int(k.sum()), "eq": float((g == 0).mean()), "dist": g.value_counts().sort_index().to_dict()}
    gb = (D.y[k] - np.ceil(D.max24_raw[k] - 0.5))  # round half down, same rows (critic fix: earlier version mis-aligned NaNs)
    res["cli_minus_24hgroup_roundhalfdown_eq"] = float((gb == 0).mean())
    res["n_exact_half_F"] = int((np.abs(D.max24_raw[k] - np.floor(D.max24_raw[k]) - 0.5) < 1e-9).sum())
    g = D.y - D.hmax
    res["cli_minus_hourly_metar_max"] = {"mean": float(g.mean()), "dist": g.clip(-3, 6).value_counts().sort_index().to_dict()}
    k6 = D.max6.notna()
    g = (D.y - np.maximum(D.max6, D.hmax))[k6]
    res["cli_minus_max(6h groups, obs)"] = {"n": int(k6.sum()), "eq": float((g == 0).mean()), "dist": g.clip(-3, 6).value_counts().sort_index().to_dict()}
    g = D.y - D.hmax_loc18
    res["cli_minus_obs_max_local_0_1830"] = {"mean": float(g.mean()), "p_ge3": float((g >= 3).mean()),
                                             "dist": g.clip(-3, 8).value_counts().sort_index().to_dict()}
    D["late"] = D.tmax_loc_h > 18.5
    D["early_lst"] = D.tmax_lst_h < 1.0
    res["timing"] = {"max_after_1830_local": float(D.late.mean()), "max_in_first_LST_hour": float(D.early_lst.mean()),
                     "by_month_late": D.groupby(D.date.dt.month).late.mean().round(3).to_dict()}
    (OUT / "dgp.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    return 0



# ------------------------------------------------------------------------------------------------ step 3: long point-in-time MOS + EMOS-L
E1_W2_START, E1_END = pd.Timestamp("2021-07-01"), pd.Timestamp("2025-06-30")
X1_START, X1_END = pd.Timestamp("2026-01-14"), pd.Timestamp("2026-02-28")
TAIL_TARGET = 0.07087148185825703   # results/e4_timing/margin.json target_tail_22_m3


def _mos_daily(model, run_h, lead_h, col):
    m = pd.read_parquet(OUT / f"mos_{model}.parquet")
    m["runtime"], m["ftime"] = pd.to_datetime(m.runtime), pd.to_datetime(m.ftime)
    m = m[(m.runtime.dt.hour == run_h) & (m.ftime.dt.hour == 0) & (m.ftime - m.runtime == pd.Timedelta(hours=lead_h))].copy()
    m["date"] = m.ftime.dt.normalize() - pd.Timedelta(days=1)
    keep = ["station", "date", col] + (["xnd"] if model == "NBS" else [])
    m = m[keep].drop_duplicates(["station", "date"])
    return m.rename(columns={col: {"NBS": "nbs", "GFS": "mav", "NAM": "met"}[model]})


def station_days() -> pd.DataFrame:
    a = pd.read_parquet(OUT / "acis_maxt.parquet")[["station", "date", "acis_val"]].rename(columns={"acis_val": "y_acis"})
    a = a[(a.date >= "2018-01-01") & ((a.date <= E1_END) | ((a.date >= X1_START) & (a.date <= X1_END)))]
    n0 = len(a)
    for mdl, rh, lh, c in (("NBS", 19, 29, "txn"), ("GFS", 12, 36, "n_x"), ("NAM", 12, 36, "n_x")):
        f = OUT / f"mos_{mdl}.parquet"
        if not f.exists():
            continue
        d = _mos_daily(mdl, rh, lh, c)
        a = a.merge(d, on=["station", "date"], how="left", validate="one_to_one")
        sh = d.copy()
        sh["date"] = sh.date + pd.Timedelta(days=7)          # placebo: members issued for D-7 (wrong day, same season)
        sh = sh.rename(columns={k: k + "7" for k in sh.columns if k not in ("station", "date")})
        a = a.merge(sh, on=["station", "date"], how="left", validate="one_to_one")
    assert len(a) == n0
    for c in ("nbs", "mav", "met", "xnd", "nbs7", "mav7", "met7", "xnd7"):
        if c not in a.columns:
            a[c] = np.nan
    a["doy"] = a.date.dt.dayofyear
    return a


def _feat(X, tier):
    w = 2 * np.pi * X["doy"].to_numpy(float) / 365.25
    sn, cs = np.sin(w), np.cos(w)
    mav = X["mav"].to_numpy(float)
    met = np.where(np.isfinite(X["met"].to_numpy(float)), X["met"].to_numpy(float), mav)
    if tier == "A":
        nbs, xnd = X["nbs"].to_numpy(float), X["xnd"].to_numpy(float)
        M = np.c_[np.ones(len(X)), sn, cs, nbs, mav, met]
        v = np.var(np.c_[nbs, mav, met], axis=1)
    else:
        xnd = np.zeros(len(X))
        M = np.c_[np.ones(len(X)), sn, cs, mav, met]
        v = np.var(np.c_[mav, met], axis=1)
    return M, sn, cs, xnd, v


def _musig(th, M, sn, cs, xnd, v, tier):
    k = M.shape[1]
    beta = th[:k]
    c0, c1, c2, k1, k2, d = th[k:]
    mu = M @ beta
    s2 = np.exp(c0 + c1 * sn + c2 * cs) + (np.exp(k1) * xnd ** 2 if tier == "A" else 0) + np.exp(k2) * v
    return mu, np.sqrt(s2), 2 + np.exp(d)


def fit_emos(tr: pd.DataFrame, tier: str):
    M, sn, cs, xnd, v = _feat(tr, tier)
    y = tr["y_acis"].to_numpy(float)
    beta = np.linalg.lstsq(M, y, rcond=None)[0]
    r = y - M @ beta
    th0 = np.r_[beta, np.log(max(r.var() * .6, .5)), 0, 0, np.log(.3), np.log(.3), np.log(5)]

    def nll(th):
        mu, sig, df = _musig(th, M, sn, cs, xnd, v, tier)
        return -sst.t.logpdf(y, df, mu, sig).sum()
    res = optimize.minimize(nll, th0, method="L-BFGS-B")
    return res.x


def _need(X, tier):
    ok = X["mav"].notna() & X["y_acis"].notna() if "y_acis" in X else X["mav"].notna()
    if tier == "A":
        ok &= X["nbs"].notna() & X["xnd"].notna()
    return ok.to_numpy()


def emos_walk(sd: pd.DataFrame, fits_name: str = "emos_fits.json") -> pd.DataFrame:
    """Per station, per scoring month: refit on that station's days strictly before the month (expanding from 2018).
    Tier A (NBS+MAV+MET) once >= 330 tier-A training days exist; else tier B (MAV+MET). X1 dev months use data <= 2025-06-30."""
    sd = sd.copy()
    sd["ym"] = sd.date.dt.to_period("M")
    months = [p for p in pd.period_range("2021-07", "2025-06", freq="M")] + [pd.Period("2026-01"), pd.Period("2026-02")]
    for c in ("mu", "sig", "df", "tier", "pit"):
        sd[c] = np.nan if c != "tier" else ""
    fits = {}
    orig7 = [wx.CITIES[k][0] for k in wx.CITIES]
    for st, g in sd.groupby("station"):
        for ym in (months[:-2] if st in orig7 else months[-2:]):
            k = (g.ym == ym).to_numpy()
            if not k.any():
                continue
            cut = min(ym.start_time, E1_END + pd.Timedelta(days=1))
            tr = g[(g.date < cut) & g.y_acis.notna()]
            th = {}
            if _need(tr, "A").sum() >= 330:
                th["A"] = fit_emos(tr[_need(tr, "A")], "A")
            if _need(tr, "B").sum() >= 330:
                th["B"] = fit_emos(tr[_need(tr, "B")], "B")
            fits[f"{st}|{ym}"] = {t: np.round(v, 4).tolist() for t, v in th.items()}
            X = g[k]
            hasA = (X.nbs.notna() & X.xnd.notna() & X.mav.notna()).to_numpy()
            hasB = X.mav.notna().to_numpy()
            for tier, mask in (("A", hasA & ("A" in th)), ("B", hasB & ~(hasA & ("A" in th)) & ("B" in th))):
                if not mask.any():
                    continue
                R = X[mask]
                mu, sig, df = _musig(th[tier], *_feat(R, tier), tier)
                idx = R.index
                sd.loc[idx, "mu"], sd.loc[idx, "sig"], sd.loc[idx, "df"], sd.loc[idx, "tier"] = mu, sig, df, tier
                y = R.y_acis.to_numpy(float)
                lo, hi = sst.t.cdf((y - .5 - mu) / sig, df), sst.t.cdf((y + .5 - mu) / sig, df)
                sd.loc[idx, "pit"] = lo + np.random.default_rng(1).random(len(R)) * (hi - lo)
        print(st, "done", flush=True)
    (OUT / fits_name).write_text(json.dumps(fits))
    return sd.drop(columns=["ym"])


def margin_wf(sd: pd.DataFrame) -> dict:
    """T2 margin per scoring month: smallest 0.5F step with tail(M) <= TAIL_TARGET on ALL stations' training days before the
    month (members nbs, mav, met-if-present; nbs and mav required; ACIS truth)."""
    ok = sd.nbs.notna() & sd.mav.notna() & sd.y_acis.notna()
    x = sd[ok]
    mem = x[["nbs", "mav", "met"]].to_numpy(float)
    hi = np.nanmax(mem, 1)
    lo = np.nanmin(mem, 1)
    y = x.y_acis.to_numpy(float)
    out = {}
    for ym in [p for p in pd.period_range("2021-07", "2025-06", freq="M")] + [pd.Period("2026-01"), pd.Period("2026-02")]:
        cut = min(ym.start_time, E1_END + pd.Timedelta(days=1))
        k = (x.date < cut).to_numpy()
        if k.sum() < 500:
            out[str(ym)] = None
            continue
        for M in np.arange(0, 15.01, .5):
            t = float(((y[k] > hi[k] + M) | (y[k] < lo[k] - M)).mean())
            if t <= TAIL_TARGET:
                out[str(ym)] = {"M": float(M), "tail": t, "n": int(k.sum())}
                break
    return out


def cmd_emos() -> int:
    sd = station_days()
    print("station-days", len(sd), "nbs", int(sd.nbs.notna().sum()), "mav", int(sd.mav.notna().sum()), "met", int(sd.met.notna().sum()))
    orig7 = [wx.CITIES[k][0] for k in wx.CITIES]
    part = OUT / "emos_orig7_partial.parquet"     # orig-7 walk already run (per-station fits are independent of X1 data)
    if part.exists():
        w7 = pd.read_parquet(part)
        chk = sd[sd.station.isin(orig7)]
        assert len(w7) == len(chk) and np.allclose(w7.mav.fillna(-99).to_numpy(), chk.mav.fillna(-99).to_numpy()), "orig7 inputs changed"
    else:
        w7 = emos_walk(sd[sd.station.isin(orig7)], "emos_fits_orig7.json")
    wx1 = emos_walk(sd[~sd.station.isin(orig7)], "emos_fits_x1.json")
    w = pd.concat([w7, wx1], ignore_index=True)
    w.to_parquet(OUT / "emos_station_days.parquet", index=False)
    mg = margin_wf(sd)
    (OUT / "margin_wf.json").write_text(json.dumps(mg, indent=1))
    # temperature-only diagnostics on scored months
    s = w[w.mu.notna()].copy()
    s["win"] = np.where(s.date >= X1_START, "X1dev", np.where(s.date >= "2024-03-01", "W1", "W2"))
    orig7 = [wx.CITIES[k][0] for k in wx.CITIES]
    s = s[(s.win == "X1dev") & ~s.station.isin(orig7) | (s.win != "X1dev") & s.station.isin(orig7)]
    diag = {}
    for (win, tier), g in s.groupby(["win", "tier"]):
        p, y = g.pit.to_numpy(), g.y_acis.to_numpy()
        diag[f"{win}|{tier}"] = {"n": int(len(g)), "mae_mu": float(np.abs(y - g.mu).mean()),
                                 "mae_nbs": float(np.abs(y - g.nbs).mean()) if g.nbs.notna().any() else None,
                                 "mae_mav": float(np.abs(y - g.mav).mean()), "bias_mu": float((y - g.mu).mean()),
                                 "bias_nbs": float((y - g.nbs).mean()) if g.nbs.notna().any() else None,
                                 "bias_mav": float((y - g.mav).mean()),
                                 "bias_met": float((y - g.met).mean()) if g.met.notna().any() else None,
                                 "tail_lo": {q: float((p < q).mean()) for q in (.01, .025)},
                                 "tail_hi": {q: float((p > 1 - q).mean()) for q in (.01, .025)},
                                 "cover90": float(((p > .05) & (p < .95)).mean()), "cover98": float(((p > .01) & (p < .99)).mean())}
    res = {"diag": diag, "margin_wf": mg}
    (OUT / "emos_diag.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(diag, indent=1, default=float))
    print({k: (v["M"] if v else None) for k, v in mg.items()})
    return 0



# ------------------------------------------------------------------------------------------------ step 4: frame, tapes, scoring
def universe_all() -> pd.DataFrame:
    """E1 orig-7 2021-07-01..2025-06-30 (W2 + W1) + X1 2026-01-14..02-28 (W1). Same filters as WT.universe; sealed dropped on load."""
    keep = ["ticker", "event_ticker", "strike_type", "floor_strike", "cap_strike", "expiration_value", "result",
            "close_time", "date", "result_mismatch", "yes_bid", "yes_ask", "price_ts"] + [f"fcsafe_{m}" for m in MODELS]
    fr = []
    for s in wx.CITIES:
        d = pd.read_parquet(ROOT / "data" / f"wx_{s.replace('KXHIGH', '')}.parquet")
        d["date"] = pd.to_datetime(d["date"])
        d = d[(d.date >= E1_W2_START) & (d.date <= E1_END)]
        fr.append(d[keep].assign(series=s, cohort="E1"))
    x = pd.read_parquet(ROOT / "data" / "wx_expand.parquet")
    x["date"] = pd.to_datetime(x["date"])
    x = x[(x.kind == "high") & (x.date >= X1_START) & (x.date <= X1_END)]
    fr.append(x[keep + ["series"]].assign(cohort="X1"))
    u = pd.concat(fr, ignore_index=True)
    u = u[~u.result_mismatch.astype(bool)].drop_duplicates("ticker").reset_index(drop=True)
    u["station"] = u.series.map(lambda s: PAPER_CITIES[s][0])
    u["tz"] = u.series.map(lambda s: PAPER_CITIES[s][1])
    u["T"] = [WT.t_local(d, tz, 22) for d, tz in zip(u.date, u.tz)]
    u["close_ts"] = (pd.to_datetime(u.close_time, utc=True) - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
    assert (u.price_ts <= u["T"]).all(), "quote after decision"
    b, a = u.yes_bid, u.yes_ask
    u = u[((b > 0) & (a < 1) & ((a - b) <= 0.10) & (a > 0.01)).to_numpy()].copy()
    u["won"] = (u.result == "no").astype(int)
    u["win"] = np.where(u.date >= "2024-03-01", "W1", "W2")
    ask = 1 - u.yes_bid
    u["ask"] = ask
    u["tcost"] = ask + FEE * ask * (1 - ask)
    u["q"] = np.round(np.minimum(np.floor(np.round((1 - (u.yes_bid + u.yes_ask) / 2) * 100, 6)) / 100, ask - 0.01), 4)
    return u.reset_index(drop=True)


def p_hit_t(mu, sig, df, X):
    F = lambda v: sst.t.cdf((v - mu) / sig, df)
    f, c, st = X.floor_strike.to_numpy(float), X.cap_strike.to_numpy(float), X.strike_type.to_numpy()
    return np.where(st == "between", F(c + .5) - F(f - .5), np.where(st == "greater", 1 - F(f + .5), F(c - .5)))


def cmd_frame() -> int:
    import importlib
    R = importlib.import_module("edgelab.rules.kalshi_wx_mos")
    u = universe_all()
    w = pd.read_parquet(OUT / "emos_station_days.parquet")
    cols = ["station", "date", "nbs", "mav", "met", "xnd", "nbs7", "mav7", "met7", "mu", "sig", "df", "tier"]
    n0 = len(u)
    u = u.merge(w[cols], on=["station", "date"], how="left", validate="many_to_one")
    assert len(u) == n0
    mg = json.loads((OUT / "margin_wf.json").read_text())
    u["m_wf"] = u.date.dt.to_period("M").astype(str).map(lambda k: (mg.get(k) or {}).get("M", np.nan))
    ok = u.mu.notna().to_numpy()
    u["phit_l"] = np.nan
    u.loc[ok, "phit_l"] = p_hit_t(u.mu[ok].to_numpy(), u.sig[ok].to_numpy(), u.df[ok].to_numpy(), u[ok])
    u["e4"] = R.sel_e4(u)
    u["t1"] = R.sel_t1(u)
    u["t2"] = R.sel_t2(u)
    u["t3"] = R.sel_t3(u)
    pl = u.copy()
    pl[["nbs", "mav", "met"]] = pl[["nbs7", "mav7", "met7"]].to_numpy()
    u["t1_pl7"] = R.sel_t1(pl)
    u["t2_pl7"] = R.sel_t2(pl)
    u.to_parquet(OUT / "frame_all.parquet", index=False)
    e = u[u.cohort == "E1"]
    keep = ["ticker", "event_ticker", "series", "date", "T", "close_ts", "yes_bid", "yes_ask", "strike_type", "floor_strike",
            "cap_strike"] + [f"fcsafe_{m}" for m in MODELS] + ["nbs", "mav", "met", "m_wf", "phit_l", "won"]
    e[keep].to_parquet(OUT / "edgelab_frame.parquet", index=False)
    print(u.groupby(["cohort", "win"])[["e4", "t1", "t2", "t3", "t1_pl7", "t2_pl7"]].sum())
    print("member coverage", u.groupby("win")[["nbs", "mav", "met", "mu"]].apply(lambda g: g.notna().mean()).round(3))
    return 0


def cmd_tapes() -> int:
    """Read-only public tapes (kx_throttle via e4_timing._tape) for every pick / placebo-pool bracket lacking one."""
    import e4_timing as ET
    u = pd.read_parquet(OUT / "frame_all.parquet")
    have = set()
    for f in [ROOT / "data" / "kalshi_temp_maker_trades.parquet", ROOT / "results" / "e4_timing" / "trades.parquet",
              ROOT / "results" / "e4_probabilistic" / "trades.parquet", OUT / "trades_extra.parquet"]:
        if f.exists():
            have |= set(pd.read_parquet(f, columns=["ticker"]).ticker)
    sel = u.e4 | u.t1 | u.t2 | u.t3 | u.t1_pl7          # t2_pl7 placebo is scored taker-only (8k tickers; disclosed)
    # T3 random-selection placebo pool: tradeable non-E4 brackets on T3 pick dates, in NO-ask buckets where T3 has picks
    u["bkt"] = pd.cut(u.ask, [0, .90, .95, .97, .98, 1.0], labels=False)
    t3k = set(zip(u[u.t3].date, u[u.t3].bkt))
    pool = (~u.e4) & np.array([(d, b) in t3k for d, b in zip(u.date, u.bkt)])
    need = u[(sel | pool) & ~u.ticker.isin(have)]
    print(f"picks/pool {int((sel | pool).sum())}, have tape {int(((sel | pool) & u.ticker.isin(have)).sum())}, pulling {len(need)}", flush=True)
    f = OUT / "trades_extra.parquet"
    rows = []
    prev = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["ticker", "ts", "no_price", "count", "taker_side"])
    for i, r in enumerate(need.itertuples(index=False)):
        got = ET._tape(r.ticker, int(r.T) - 3600, int(r.close_ts))
        rows += got if got else [(r.ticker, 0, np.nan, 0.0, "none")]     # marker row: tape fetched, no prints
        if i % 200 == 0 or i == len(need) - 1:
            pd.concat([prev, pd.DataFrame(rows, columns=["ticker", "ts", "no_price", "count", "taker_side"])],
                      ignore_index=True).to_parquet(f, index=False)
            print(f"  {i + 1}/{len(need)}", flush=True)
    return 0



def _mk(u, g):
    """maker pnl per row (trade-through; missing tape: loser counted filled, winner unfilled) + taker pnl."""
    u = maker_cols(u, g)
    fil = np.where(u.has_tape, u.filled, u.won == 0)
    u["mfilled"] = fil
    u["mpnl"] = np.where(fil, pnl(u.won.to_numpy(), u.q.to_numpy()), 0.0)
    u["tpnl"] = pnl(u.won.to_numpy(), u.tcost.to_numpy())
    return u


def _boot(u, days, col, mask, sign=1.0, n=4000, seed=0):
    x = u[mask]
    d = (sign * x.groupby(x.date.dt.strftime("%Y-%m-%d"))[col].sum()).reindex(days, fill_value=0.0).to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(days), size=(n, len(days)))
    bs = d[idx].sum(1)
    return {"sum": round(float(d.sum()), 3), "ci95": [round(float(np.quantile(bs, .025)), 3), round(float(np.quantile(bs, .975)), 3)],
            "p": float(np.mean(bs <= 0))}


def _halves(u, days, col, mask, sign=1.0):
    half = pd.Timestamp(days[len(days) // 2])
    x = u[mask]
    return [round(float(sign * x[x.date < half][col].sum()), 3), round(float(sign * x[x.date >= half][col].sum()), 3)], str(half.date())


def _desc(x, col, fillcol=None):
    n = len(x) if fillcol is None else int(x[fillcol].sum())
    xx = x if fillcol is None else x[x[fillcol].astype(bool)]
    return {"n": int(n), "losses": int((xx.won == 0).sum()), "units": round(float(xx[col].sum()), 3),
            "roi": round(float(xx[col].mean()), 4) if len(xx) else None}


def cmd_score() -> int:
    u = pd.read_parquet(OUT / "frame_all.parquet")
    g = tapes()
    ef = pd.read_parquet(OUT / "edgelab_frame.parquet", columns=["ticker"])
    et = [g[k] for k in ef.ticker if k in g]
    pd.concat(et, ignore_index=True).to_parquet(OUT / "edgelab_tape.parquet", index=False)
    u["bkt"] = pd.cut(u.ask, [0, .90, .95, .97, .98, 1.0], labels=False)
    need = (u.e4 | u.t1 | u.t2 | u.t3 | u.t1_pl7 | u.t2_pl7).to_numpy()
    t3k = set(zip(u[u.t3].date, u[u.t3].bkt))
    pool = ((~u.e4) & np.array([(d, b) in t3k for d, b in zip(u.date, u.bkt)])).to_numpy()
    v = _mk(u[need | pool].copy(), g)
    out = {}
    W1 = v[v.win == "W1"]
    W2 = v[(v.win == "W2") & (v.cohort == "E1")]
    d1 = pd.Index(sorted(u[u.win == "W1"].date.dt.strftime("%Y-%m-%d").unique()))
    d2 = pd.Index(sorted(u[(u.win == "W2")].date.dt.strftime("%Y-%m-%d").unique()))
    cov = lambda x: round(float(x.has_tape.mean()), 4) if len(x) else None
    out["tape_coverage"] = {"W1_e4": cov(W1[W1.e4]), "W1_t2": cov(W1[W1.t2]), "W1_t3": cov(W1[W1.t3]), "W2_t2": cov(W2[W2.t2]),
                            "W2_t3": cov(W2[W2.t3]), "W2_t2_pl7": cov(W2[W2.t2_pl7]), "pool": cov(v[pool[need | pool]])}
    # ---------------- T1: veto delta on W1
    vet = W1.e4 & ~W1.t1
    T1 = {"e4_maker": _desc(W1[W1.e4], "mpnl", "mfilled"), "t1_maker": _desc(W1[W1.t1], "mpnl", "mfilled"),
          "vetoed": int(vet.sum()), "vetoed_losers": int((W1[vet].won == 0).sum()),
          "vetoed_maker": _desc(W1[vet], "mpnl", "mfilled"), "vetoed_taker": _desc(W1[vet], "tpnl")}
    for col in ("mpnl", "tpnl"):
        T1[f"delta_{col}"] = _boot(W1, d1, col, vet, sign=-1.0)
        T1[f"halves_{col}"], T1["half_date"] = _halves(W1, d1, col, vet, sign=-1.0)
    # placebo (i) random veto, same count per NO-ask bucket
    rng = np.random.default_rng(7)
    e4 = W1[W1.e4]
    cnt = W1[vet].bkt.value_counts().to_dict()
    pls = {"mpnl": [], "tpnl": []}
    byb = {b: e4[e4.bkt == b] for b in cnt}
    for _ in range(2000):
        ch = pd.concat([byb[b].sample(n=int(k), random_state=int(rng.integers(1 << 31))) for b, k in cnt.items()])
        for col in pls:
            pls[col].append(-float(ch[col].sum()))
    T1["placebo_random_veto"] = {col: {"p95": round(float(np.quantile(x, .95)), 3), "mean": round(float(np.mean(x)), 3),
                                       "frac_ge_actual": float(np.mean(np.array(x) >= T1[f"delta_{col}"]["sum"]))} for col, x in pls.items()}
    vet7 = W1.e4 & ~W1.t1_pl7
    T1["placebo_wrongday"] = {"vetoed": int(vet7.sum()), "vetoed_losers": int((W1[vet7].won == 0).sum()),
                              "delta_mpnl": _boot(W1, d1, "mpnl", vet7, sign=-1.0), "delta_tpnl": _boot(W1, d1, "tpnl", vet7, sign=-1.0)}
    T1["by_cohort"] = {c: {"vetoed": int((vet & (W1.cohort == c)).sum()), "losers": int((W1[vet & (W1.cohort == c)].won == 0).sum()),
                           "delta_mpnl": round(-float(W1[vet & (W1.cohort == c)].mpnl.sum()), 3)} for c in ("E1", "X1")}
    T1["vetoed_rows"] = W1[vet][["ticker", "date", "station", "strike_type", "floor_strike", "cap_strike", "ask", "q", "won", "mfilled",
                                 "nbs", "mav", "met"] + [f"fcsafe_{m}" for m in MODELS]].assign(date=lambda x: x.date.astype(str)).to_dict("records")
    out["T1"] = T1
    # ---------------- T2: MOS consensus; W2 primary, W1 paired vs E4
    T2 = {"W2_maker": _desc(W2[W2.t2], "mpnl", "mfilled"), "W2_taker": _desc(W2[W2.t2], "tpnl"),
          "W2_units_mpnl": _boot(W2, d2, "mpnl", W2.t2), "W2_units_tpnl": _boot(W2, d2, "tpnl", W2.t2),
          "W2_unfilled_losers": int((W2[W2.t2 & ~W2.mfilled].won == 0).sum())}
    T2["W2_halves_mpnl"], T2["W2_half_date"] = _halves(W2, d2, "mpnl", W2.t2)
    T2["W2_halves_tpnl"], _ = _halves(W2, d2, "tpnl", W2.t2)
    T2["W2_placebo_wrongday_taker"] = _desc(W2[W2.t2_pl7], "tpnl")
    T2["W1_placebo_wrongday_taker"] = _desc(W1[W1.t2_pl7], "tpnl")
    T2["W2_taker_actual_vs_placebo_roi"] = [T2["W2_taker"]["roi"], T2["W2_placebo_wrongday_taker"]["roi"]]
    T2["W1_maker"] = _desc(W1[W1.t2], "mpnl", "mfilled")
    T2["W1_units_mpnl"] = _boot(W1, d1, "mpnl", W1.t2)
    W1d = W1.assign(dm=np.where(W1.t2, W1.mpnl, 0) - np.where(W1.e4, W1.mpnl, 0), dt=np.where(W1.t2, W1.tpnl, 0) - np.where(W1.e4, W1.tpnl, 0))
    T2["W1_delta_vs_e4_mpnl"] = _boot(W1d, d1, "dm", W1d.t2 | W1d.e4)
    T2["W1_delta_vs_e4_tpnl"] = _boot(W1d, d1, "dt", W1d.t2 | W1d.e4)
    T2["W1_overlap_with_e4"] = int((W1.t2 & W1.e4).sum())
    out["T2"] = T2
    # ---------------- T3: EMOS-L NEW picks
    T3 = {"W1_maker": _desc(W1[W1.t3], "mpnl", "mfilled"), "W1_taker": _desc(W1[W1.t3], "tpnl"),
          "W1_units_mpnl": _boot(W1, d1, "mpnl", W1.t3), "W1_units_tpnl": _boot(W1, d1, "tpnl", W1.t3),
          "W2_maker": _desc(W2[W2.t3], "mpnl", "mfilled"), "W2_taker": _desc(W2[W2.t3], "tpnl"),
          "W2_units_mpnl": _boot(W2, d2, "mpnl", W2.t3), "W2_units_tpnl": _boot(W2, d2, "tpnl", W2.t3),
          "W1_unfilled_losers": int((W1[W1.t3 & ~W1.mfilled].won == 0).sum()), "expected_losses_model_W1": float(W1[W1.t3].phit_l.sum()),
          "W1_by_cohort": {c: _desc(W1[W1.t3 & (W1.cohort == c)], "mpnl", "mfilled") for c in ("E1", "X1")}}
    T3["W1_halves_mpnl"], T3["W1_half_date"] = _halves(W1, d1, "mpnl", W1.t3)
    # random-selection placebo: same count per (date, NO-ask bucket) from tradeable non-E4 brackets (pool)
    P = v[pool[need | pool]]
    for wname, Wx in (("W1", W1), ("W2", W2)):
        cells = Wx[Wx.t3].groupby([Wx[Wx.t3].date, Wx[Wx.t3].bkt]).size()
        Pw = P[P.ticker.isin(set(Wx.ticker))]
        grp = {k: x for k, x in Pw.groupby(["date", "bkt"])}
        res = {"mpnl": [], "tpnl": []}
        rng = np.random.default_rng(11)
        for _ in range(2000):
            tot = {"mpnl": 0.0, "tpnl": 0.0}
            for k, nk in cells.items():
                x = grp[k]
                ix = rng.choice(len(x), size=min(int(nk), len(x)), replace=False)
                for col in tot:
                    tot[col] += float(x[col].to_numpy()[ix].sum())
            for col in res:
                res[col].append(tot[col])
        act = {"mpnl": float(Wx[Wx.t3].mpnl.sum()), "tpnl": float(Wx[Wx.t3].tpnl.sum())}
        T3[f"{wname}_placebo_random"] = {col: {"p95": round(float(np.quantile(x, .95)), 3), "mean": round(float(np.mean(x)), 3),
                                               "frac_ge_actual": float(np.mean(np.array(x) >= act[col]))} for col, x in res.items()}
    out["T3"] = T3
    (OUT / "score.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items()}, indent=1, default=float)[:20000])
    return 0



def cmd_compare() -> int:
    """Temperature-only: EMOS-L vs e4p pooled EMOS on the SAME e4p TEST station-days/brackets (no prices, no P&L)."""
    e = pd.read_parquet(ROOT / "results" / "e4_probabilistic" / "phit.parquet")
    e = e[e.phase == "test"]
    f = pd.read_parquet(OUT / "frame_all.parquet") if (OUT / "frame_all.parquet").exists() else None
    w = pd.read_parquet(OUT / "emos_station_days.parquet")
    e["date"] = pd.to_datetime(e.date)
    sd = e.drop_duplicates("event_ticker").merge(w[["station", "date", "mu", "sig", "df", "pit", "tier", "y_acis"]],
                                                 on=["station", "date"], how="left", validate="one_to_one")
    k = sd.mu.notna()
    out = {"station_days": int(len(sd)), "with_emos_l": int(k.sum())}
    pe = sd.pit_x[k].to_numpy() if "pit_x" in sd else sd.pit[k].to_numpy()
    pl = sd.pit_y[k].to_numpy() if "pit_y" in sd else None
    for name, p in (("e4p", pe), ("emos_l", pl)):
        out[name] = {"tail_lo": {q: float((p < q).mean()) for q in (.01, .025)}, "tail_hi": {q: float((p > 1 - q).mean()) for q in (.01, .025)},
                     "cover98": float(((p > .01) & (p < .99)).mean())}
    y = sd.expiration_value[k].to_numpy(float)
    out["mae_emos_l_mu"] = float(np.abs(y - sd.mu[k]).mean())
    out["mae_nbs_txn"] = float(np.abs(y - sd.txn[k]).mean())
    # log score of the realised integer value
    def ls(mu, sig, df):
        return np.log(np.clip(sst.t.cdf((y + .5 - mu) / sig, df) - sst.t.cdf((y - .5 - mu) / sig, df), 1e-12, 1))
    out["logscore_emos_l"] = float(ls(sd.mu[k].to_numpy(), sd.sig[k].to_numpy(), sd.df[k].to_numpy()).mean())
    # bracket-level reliability on the same brackets (all brackets of these events)
    b = e.merge(w[["station", "date", "mu", "sig", "df"]], on=["station", "date"], how="left")
    b = b[b.mu.notna()].copy()
    b["phit_l"] = p_hit_t(b.mu.to_numpy(), b.sig.to_numpy(), b.df.to_numpy(), b)
    b["hit"] = (b.result == "yes").astype(int)
    bins = [0, .005, .01, .02, .05, .1, 1.0001]
    for col in ("phit", "phit_l"):
        r = b.groupby(pd.cut(b[col], bins, right=False), observed=True).agg(n=("hit", "size"), mean_p=(col, "mean"), hit=("hit", "mean"))
        out[f"reliability_{col}"] = {str(kk): vv for kk, vv in r.round(4).to_dict("index").items()}
        out[f"brier_{col}"] = float(((b[col] - b.hit) ** 2).mean())
    (OUT / "compare_e4p.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    return 0



def cmd_histfc() -> int:
    """Open-Meteo historical-forecast API vs previous-runs previous_day1 vs CLI (KNYC JJA 2024; network GET)."""
    import requests
    u = WT.universe()
    u = u[(u.series == "KXHIGHNY") & (u.date >= "2024-06-01") & (u.date <= "2024-08-31")].drop_duplicates("event_ticker")
    out = {}
    for m in ("gfs_seamless", "ecmwf_ifs025"):
        r = requests.get("https://historical-forecast-api.open-meteo.com/v1/forecast",
                         params={"latitude": 40.779, "longitude": -73.969, "models": m, "hourly": "temperature_2m",
                                 "temperature_unit": "fahrenheit", "timezone": "America/New_York",
                                 "start_date": "2024-06-01", "end_date": "2024-08-31"}, timeout=120).json()
        h = pd.DataFrame(r["hourly"])
        h["time"] = pd.to_datetime(h.time)
        gm = h[h.time.dt.hour <= 18].groupby(h.time.dt.normalize()).temperature_2m.max()
        v = u.set_index("date")
        j = pd.DataFrame({"y": v.expiration_value, "hf": gm.reindex(v.index), "p1": v[f"fcsafe_{m}"]}).dropna()
        out[m] = {"n": len(j), "mae_histfc": round(float((j.y - j.hf).abs().mean()), 2), "mae_prev_day1": round(float((j.y - j.p1).abs().mean()), 2),
                  "bias_histfc": round(float((j.y - j.hf).mean()), 2), "bias_prev_day1": round(float((j.y - j.p1).mean()), 2),
                  "corr_err": round(float(np.corrcoef(j.y - j.hf, j.y - j.p1)[0, 1]), 2)}
    (OUT / "openmeteo_histfc_check.json").write_text(json.dumps(out, indent=1))
    print(out)
    return 0


def cmd_descriptive() -> int:
    """Re-generates the descriptive (non-trial) outputs quoted in notes/e4_data_v2.md, so each has a generating command."""
    from scipy.stats import fisher_exact
    cols = [f"fcsafe_{m}" for m in MODELS]
    u = dev_frame()
    t = u[u.tradeable].copy()
    hi, lo = t[cols].max(axis=1), t[cols].min(axis=1)
    # loser anatomy
    D = pd.read_parquet(OUT / "dgp_days.parquet")
    L = t[t.e4 & (t.won == 0)].assign(hi=hi, lo=lo).merge(D[["event_ticker", "hmax", "hmax_loc18", "tmax_loc_h", "tmax_lst_h"]],
                                                         on="event_ticker", how="left", validate="many_to_one")
    L["side"] = np.where(L.y > L.hi, "hot", np.where(L.y < L.lo, "cold", "inside"))

    def hits(r, v):
        return (v >= r.floor_strike + 1) if r.strike_type == "greater" else (v <= r.cap_strike - 1) if r.strike_type == "less" \
            else (v >= r.floor_strike) & (v <= r.cap_strike)
    L["hit_by_hourly"] = [hits(r, r.hmax) for r in L.itertuples()]
    L.to_csv(OUT / "e4_dev_losers.csv", index=False)
    H = L[L.side == "hot"]
    anat = {"losers": int(len(L)), "hot": int(len(H)), "cold": int((L.side == "cold").sum()),
            "hot_late_after_1830": int((H.tmax_loc_h > 18.5).sum()), "hot_first_lst_hour": int((H.tmax_lst_h < 1).sum()),
            "hot_hit_by_hourly_max": int(H.hit_by_hourly.sum()), "hot_cli_minus_hourly": (H.y - H.hmax).value_counts().sort_index().to_dict()}
    # side asymmetry
    e = t[t.e4].copy()
    e["side"] = np.where(e.strike_type == "greater", "above", np.where(e.strike_type == "less", "below",
                                                                      np.where(e.floor_strike > hi[e.index], "above", "below")))
    e = maker_cols(e, tapes())
    e["tpnl"] = pnl(e.won.to_numpy(), e.tcost.to_numpy())
    side = {}
    for nm, x in (("taker_all", e), ("maker_filled", e[e.filled])):
        side[nm] = {sname: {"n": int(len(y)), "losses": int((y.won == 0).sum()), "rate": round(float((y.won == 0).mean()), 4),
                            "mean_ask": round(float(y.ask.mean()), 4), "breakeven": round(float((1 - y.tcost).mean()), 4),
                            "taker_roi": round(float(y.tpnl.mean()), 4), "taker_units": round(float(y.tpnl.sum()), 2)} for sname, y in x.groupby("side")}
        a, b = side[nm]["above"], side[nm]["below"]
        side[nm]["fisher_p"] = float(fisher_exact([[a["losses"], a["n"] - a["losses"]], [b["losses"], b["n"] - b["losses"]]])[1])
    sd = u.drop_duplicates("event_ticker")
    sd = sd[sd[cols].notna().sum(axis=1) >= 3]
    sh, sl, y = sd[cols].max(axis=1), sd[cols].min(axis=1), sd.y
    side["tail_asym"] = {M: {"warm": round(float((y > sh + M).mean()), 4), "cold": round(float((y < sl - M).mean()), 4)} for M in (1, 2, 3, 4, 5)}
    # bias decomposition
    x = sd.assign(m=sd[cols].mean(axis=1)).merge(D[["event_ticker", "hmax_loc18"]], on="event_ticker", how="inner")
    x["season"] = x.date.dt.month.map(lambda mm: "DJF" if mm in (12, 1, 2) else "MAM" if mm in (3, 4, 5) else "JJA" if mm in (6, 7, 8) else "SON")
    x["total"], x["sampling"], x["model2m"] = x.y - x.m, x.y - x.hmax_loc18, x.hmax_loc18 - x.m
    bd = x.groupby(["station", "season"])[["total", "sampling", "model2m"]].mean().round(2)
    bd["n"] = x.groupby(["station", "season"]).size()
    bd.to_csv(OUT / "bias_decomp.csv")
    pd.concat({c.replace("fcsafe_", ""): (sd.y - sd[c]).groupby(sd.station).mean() for c in cols}, axis=1).round(2).to_csv(OUT / "bias_model_station.csv")
    # asymmetric margin grid + market calibration + taker prelim (frame_all)
    f = pd.read_parquet(OUT / "frame_all.parquet")
    f["tpnl"] = pnl(f.won.to_numpy(), f.tcost.to_numpy())
    W1, W2 = f[f.win == "W1"], f[(f.win == "W2") & (f.cohort == "E1")]
    d1 = pd.Index(sorted(W1.date.dt.strftime("%Y-%m-%d").unique()))
    d2 = pd.Index(sorted(W2.date.dt.strftime("%Y-%m-%d").unique()))
    m = W1[cols].to_numpy(float)
    ok = np.isfinite(m)
    h1, l1, n1 = np.where(ok, m, -np.inf).max(1), np.where(ok, m, np.inf).min(1), ok.sum(1)
    fl, cp_, st = W1.floor_strike.to_numpy(float), W1.cap_strike.to_numpy(float), W1.strike_type.to_numpy()
    grid = []
    for Mw in (2, 3, 4, 5):
        for Mc in (1, 2, 3, 4):
            r = np.where(st == "greater", h1 + Mw < fl + 1, np.where(st == "less", l1 - Mc > cp_ - 1, (h1 + Mw < fl) | (l1 - Mc > cp_))) & (n1 >= 3)
            s_ = W1[r]
            grid.append({"Mw": Mw, "Mc": Mc, "n": int(len(s_)), "losses": int((s_.won == 0).sum()), "units": round(float(s_.tpnl.sum()), 2),
                         "ci95": _boot(W1, d1, "tpnl", pd.Series(r, index=W1.index))["ci95"]})
    f["mid"], f["hit"] = (f.yes_bid + f.yes_ask) / 2, 1 - f.won
    mc = {}
    for w, x_ in (("W1", W1.assign(mid=(W1.yes_bid + W1.yes_ask) / 2, hit=1 - W1.won)), ("W2", W2.assign(mid=(W2.yes_bid + W2.yes_ask) / 2, hit=1 - W2.won))):
        r = x_.groupby(pd.cut(x_.mid, [0, .02, .05, .1, .2, .35, .5, 1]), observed=True).agg(n=("hit", "size"), mid=("mid", "mean"), hit=("hit", "mean"))
        mc[w] = {str(k): v for k, v in r.round(4).to_dict("index").items()}
    W1m = W1.assign(mid=(W1.yes_bid + W1.yes_ask) / 2, hit=1 - W1.won)
    for nm, k in (("E4", W1m.e4), ("T1_vetoed", W1m.e4 & ~W1m.t1), ("T1_kept", W1m.t1), ("T3", W1m.t3)):
        xx = W1m[k]
        mc[nm] = {"n": int(len(xx)), "mean_mid": round(float(xx.mid.mean()), 4), "mean_phit_l": round(float(xx.phit_l.mean()), 4),
                  "realised": round(float(xx.hit.mean()), 4)}
    vet = W1.e4 & ~W1.t1
    prelim = {"T1_delta_taker": _boot(W1, d1, "tpnl", vet, sign=-1.0), "T1_vetoed": int(vet.sum()), "T1_vetoed_losers": int((W1[vet].won == 0).sum()),
              "E4_taker": _desc(W1[W1.e4], "tpnl"), "T1_taker": _desc(W1[W1.t1], "tpnl"),
              "T1_pl7_delta_taker": _boot(W1, d1, "tpnl", W1.e4 & ~W1.t1_pl7, sign=-1.0),
              "T2_W2_taker": _desc(W2[W2.t2], "tpnl"), "T2_W2_boot": _boot(W2, d2, "tpnl", W2.t2), "T2_W1_taker": _desc(W1[W1.t2], "tpnl"),
              "T2_W2_pl7_taker": _desc(W2[W2.t2_pl7], "tpnl"), "T3_W1_taker": _desc(W1[W1.t3], "tpnl"), "T3_W1_boot": _boot(W1, d1, "tpnl", W1.t3),
              "T3_W2_taker": _desc(W2[W2.t3], "tpnl"), "T3_W2_boot": _boot(W2, d2, "tpnl", W2.t3)}
    for name, obj in (("loser_anatomy.json", anat), ("side_asym.json", side), ("asym_margin_grid_taker.json", grid),
                      ("market_calibration.json", mc), ("score_taker_prelim.json", prelim)):
        (OUT / name).write_text(json.dumps(obj, indent=1, default=float))
    print(json.dumps({"anatomy": anat, "side_fisher": {k: side[k]["fisher_p"] for k in ("taker_all", "maker_filled")},
                      "T1_delta_taker": prelim["T1_delta_taker"], "T3_W1": prelim["T3_W1_boot"]}, default=float, indent=1))
    return 0


COMMANDS = {"headroom": "cmd_headroom", "dgp": "cmd_dgp", "emos": "cmd_emos", "frame": "cmd_frame", "tapes": "cmd_tapes", "score": "cmd_score", "compare": "cmd_compare", "histfc": "cmd_histfc", "descriptive": "cmd_descriptive"}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit(globals()[COMMANDS[cmd]]() if cmd in COMMANDS else (print(__doc__) or 1))
