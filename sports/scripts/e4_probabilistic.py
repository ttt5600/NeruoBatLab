"""E4 with a calibrated probabilistic forecast (NBM NBS txn/xnd + 5 det models, Student-t EMOS). DEV only.
Pre-registration: notes/e4_probabilistic.md.

    python scripts/e4_probabilistic.py nbs     # pull NBM NBS 19Z runs from the IEM MOS archive (no outcomes)
    python scripts/e4_probabilistic.py calib   # fit + PIT/reliability diagnostics (temperature outcomes only, no prices)
    python scripts/e4_probabilistic.py tapes   # public tapes for the rule's NEW picks lacking one (read-only, kx_throttle)
    python scripts/e4_probabilistic.py dev     # score the 4 trials on TEST
"""
from __future__ import annotations

import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import requests
from scipy import optimize, stats as sst

from sports import wx
from kalshi_paper import PAPER_CITIES
import wx_timing as WT

OUT = ROOT / "results" / "e4_probabilistic"
NBS = OUT / "nbs_19z.parquet"
FEE = 0.07
CAL_END = pd.Timestamp("2024-10-31")
PSTAR = 0.02
IEM = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"


# ------------------------------------------------------------------------------------------------ NBS pull
def cmd_nbs() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    u = WT.universe()
    have = pd.read_parquet(NBS) if NBS.exists() else pd.DataFrame(columns=["station", "runtime"])
    done = set(zip(have.station, pd.to_datetime(have.runtime).dt.strftime("%Y-%m"))) if len(have) else set()
    rows = [have] if len(have) else []
    for s, g in u.groupby("series"):
        st = PAPER_CITIES[s][0]
        months = pd.period_range(g.date.min() - pd.Timedelta(days=1), g.date.max(), freq="M")
        for p in months:
            if (st, str(p)) in done:
                continue
            a = max(p.start_time, g.date.min() - pd.Timedelta(days=1))
            b = min(p.end_time.normalize(), g.date.max())
            prm = {"station": st, "model": "NBS", "sts": f"{a:%Y-%m-%d}T00:00Z", "ets": f"{b:%Y-%m-%d}T23:59Z",
                   "format": "csv"}
            for k in range(5):
                r = requests.get(IEM, params=prm, timeout=120)
                if r.ok and r.text.startswith("runtime"):
                    break
                time.sleep(20 * (k + 1))
            else:
                print(f"FAIL {st} {p}: {r.text[:100]}", flush=True)
                continue
            d = pd.read_csv(io.StringIO(r.text))
            d = d[pd.to_datetime(d.runtime).dt.hour == 19][["runtime", "ftime", "station", "txn", "xnd"]]
            d = d[d.txn.notna()]
            rows.append(d)
            print(f"{st} {p}: {len(d)} txn rows", flush=True)
            pd.concat(rows, ignore_index=True).to_parquet(NBS, index=False)
            time.sleep(4)
    return 0


def nbs_daily() -> pd.DataFrame:
    """Day-D max from the 19Z D-1 run = txn at ftime (D+1) 00Z."""
    n = pd.read_parquet(NBS)
    n["runtime"], n["ftime"] = pd.to_datetime(n.runtime), pd.to_datetime(n.ftime)
    n = n[(n.ftime.dt.hour == 0) & (n.ftime - n.runtime == pd.Timedelta(hours=29))]
    n["date"] = n.ftime.dt.normalize() - pd.Timedelta(days=1)
    return n[["station", "date", "txn", "xnd"]].drop_duplicates(["station", "date"])


# ------------------------------------------------------------------------------------------------ model
def frame() -> pd.DataFrame:
    u = WT.universe()
    u["station"] = u.series.map(lambda s: PAPER_CITIES[s][0])
    u = u.merge(nbs_daily(), on=["station", "date"], how="left")
    cols = [f"fcsafe_{m}" for m in WT.MODELS]
    u["n_mod"] = u[cols].notna().sum(axis=1)
    u["m"] = u[cols].mean(axis=1)
    u["s2"] = u[cols].var(axis=1, ddof=0)
    for m in WT.MODELS:
        u[f"fc22_{m}"] = u[f"fcsafe_{m}"]
    return u


def _unpack(th, nbs):
    if nbs:
        b0, b1, b2, c0, c1, c2, d = th
    else:
        b0, b1, c0, c2, d = th
        b2, c1 = 0.0, -50.0
    return b0, b1, b2, c0, c1, c2, d


def _musig(th, X, nbs):
    b0, b1, b2, c0, c1, c2, d = _unpack(th, nbs)
    mu = b0 + b1 * X["m"] + b2 * (X["txn"].fillna(0) if nbs else 0)
    sig = np.sqrt(np.exp(c0) + np.exp(c1) * (X["xnd"].fillna(0) ** 2 if nbs else 0) + np.exp(c2) * X["s2"])
    return np.asarray(mu, float), np.asarray(sig, float), 2 + np.exp(d)


def fit(sd: pd.DataFrame, nbs: bool):
    X = sd[sd.txn.notna() & sd.xnd.notna()] if nbs else sd
    y = X.expiration_value.to_numpy(float)

    def nll(th):
        mu, sig, df = _musig(th, X, nbs)
        return -sst.t.logpdf(y, df, mu, sig).sum()

    th0 = [0, 1, 0, np.log(4), np.log(1), np.log(1), np.log(5)] if nbs else [0, 1, np.log(4), np.log(1), np.log(5)]
    th0 = np.array(th0, float)
    if nbs:
        th0[1], th0[2] = .5, .5
    r = optimize.minimize(nll, th0, method="L-BFGS-B")
    return r.x


def cdf(th, X, nbs, v):
    mu, sig, df = _musig(th, X, nbs)
    return sst.t.cdf((v - mu) / sig, df)


def p_hit(th, X, nbs):
    st, f, c = X.strike_type.to_numpy(), X.floor_strike.to_numpy(float), X.cap_strike.to_numpy(float)
    F = lambda v: cdf(th, X, nbs, v)
    return np.where(st == "between", F(c + .5) - F(f - .5),
                    np.where(st == "greater", 1 - F(f + .5), F(c - .5)))


def walk_forward(u: pd.DataFrame) -> pd.DataFrame:
    """P(hit) for every bracket: calibration rows by leave-one-month-out, test rows by expanding refit (pre-reg)."""
    u = u[u.n_mod >= 3].copy()
    sd = u.drop_duplicates("event_ticker")
    u["ym"] = u.date.dt.to_period("M")
    sd = sd.assign(ym=sd.date.dt.to_period("M"))
    u["phit"], u["has_nbs"], u["pit"] = np.nan, u.txn.notna() & u.xnd.notna(), np.nan
    fits = {}
    for ym in sorted(u.ym.unique()):
        ts = ym.start_time
        if ts <= CAL_END:   # leave-one-month-out within calibration
            tr = sd[(sd.date <= CAL_END) & (sd.ym != ym)]
            phase = "cal"
        elif u[u.ym == ym].cohort.iloc[0] == "X1":
            tr = sd[sd.cohort == "E1"]
            phase = "test"
        else:
            tr = sd[sd.date < ts]
            phase = "test"
        th = {True: fit(tr, True), False: fit(tr, False)}
        fits[str(ym)] = {"phase": phase, "n_train": int(len(tr)), "nbs": th[True].round(4).tolist(),
                         "nonbs": th[False].round(4).tolist()}
        for nb in (True, False):
            k = (u.ym == ym) & (u.has_nbs == nb)
            if k.any():
                X = u[k]
                u.loc[k, "phit"] = p_hit(th[nb], X, nb)
                y = X.expiration_value.to_numpy(float)
                rng = np.random.default_rng(1)
                lo, hi = cdf(th[nb], X, nb, y - .5), cdf(th[nb], X, nb, y + .5)
                u.loc[k, "pit"] = lo + rng.random(len(X)) * (hi - lo)
        u.loc[u.ym == ym, "phase"] = phase
    (OUT / "fits.json").write_text(json.dumps(fits, indent=1))
    return u


def cmd_calib() -> int:
    u = walk_forward(frame())
    u.drop(columns=["ym"]).to_parquet(OUT / "phit.parquet", index=False)
    sd = u.drop_duplicates("event_ticker")
    out = {}
    for ph, g in sd.groupby("phase"):
        p = g.pit.to_numpy()
        out[ph] = {"station_days": int(len(g)), "with_nbs": int(g.has_nbs.sum()),
                   "pit_deciles": np.histogram(p, bins=10, range=(0, 1))[0].tolist(),
                   "tail_lo": {q: float((p < q).mean()) for q in (.01, .025, .05)},
                   "tail_hi": {q: float((p > 1 - q).mean()) for q in (.01, .025, .05)},
                   "cover90": float(((p > .05) & (p < .95)).mean()), "cover98": float(((p > .01) & (p < .99)).mean())}
    # NBS-only skill check: MAE of txn vs mean det model (temperature only)
    k = sd.txn.notna()
    out["mae"] = {"det_mean": float((sd.m - sd.expiration_value).abs()[k].mean()),
                  "nbs_txn": float((sd.txn - sd.expiration_value).abs()[k].mean()), "n": int(k.sum())}
    (OUT / "calib.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    # bracket-level reliability on CALIBRATION rows only (the test reliability is reported with the trade scoring)
    c = u[u.phase == "cal"]
    hit = (c.result == "yes").astype(int)
    bins = [0, .005, .01, .02, .05, .1, .2, .5, 1.0001]
    r = pd.DataFrame({"p": c.phit, "h": hit}).groupby(pd.cut(c.phit, bins, right=False), observed=True).agg(
        n=("h", "size"), mean_p=("p", "mean"), hit=("h", "mean"))
    print("CAL bracket reliability\n", r)
    return 0


# ------------------------------------------------------------------------------------------------ prices / selection
def priced() -> pd.DataFrame:
    u = pd.read_parquet(OUT / "phit.parquet")
    px = pd.concat([pd.read_parquet(ROOT / "data" / f"wx_{s.replace('KXHIGH', '')}.parquet")[["ticker", "yes_bid", "yes_ask"]]
                    for s in wx.CITIES] + [pd.read_parquet(ROOT / "data" / "wx_expand.parquet")[["ticker", "yes_bid", "yes_ask"]]]
                   ).drop_duplicates("ticker")
    u = u.merge(px, on="ticker", how="left")
    b, a = u.yes_bid, u.yes_ask
    u["tradeable"] = ((b > 0) & (a < 1) & ((a - b) <= 0.10) & (a > 0.01)).to_numpy()
    u["e4"] = WT.rejected(u, 22, 3.0) & u.tradeable.to_numpy()
    u["tz"] = u.series.map(lambda s: PAPER_CITIES[s][1])
    u["T"] = [int(datetime((d - pd.Timedelta(days=1)).year, (d - pd.Timedelta(days=1)).month, (d - pd.Timedelta(days=1)).day,
                           22, tzinfo=ZoneInfo(tz)).timestamp()) for d, tz in zip(u.date, u.tz)]
    u["close_ts"] = pd.to_datetime(u.close_time, utc=True).map(lambda x: int(x.timestamp()))
    ask = 1 - u.yes_bid
    u["ask"] = ask
    u["tcost"] = ask + FEE * ask * (1 - ask)
    u["q"] = np.round(np.minimum(np.floor(np.round((1 - (u.yes_bid + u.yes_ask) / 2) * 100, 6)) / 100, ask - 0.01), 4)
    u["sel_c"] = u.phit <= PSTAR
    u["sel_pa_t"] = u.phit <= 0.5 * (1 - u.tcost)
    u["sel_pa_m"] = u.phit <= 0.5 * (1 - u.q)
    return u


def cmd_tapes() -> int:
    import e4_timing as ET  # read-only tape fetch via kx_throttle
    u = priced()
    need = u[u.tradeable & (u.phase == "test") & (u.sel_c | u.sel_pa_t | u.sel_pa_m | u.e4)]
    have = pd.read_parquet(ROOT / "data" / "kalshi_temp_maker_trades.parquet")
    fr = [have[have.ticker.isin(need.ticker)]]
    f2 = ROOT / "results" / "e4_timing" / "trades.parquet"
    if f2.exists():
        t2 = pd.read_parquet(f2)
        fr.append(t2[t2.ticker.isin(need.ticker) & ~t2.ticker.isin(fr[0].ticker)])
    got = set(pd.concat(fr).ticker)
    todo = need[~need.ticker.isin(got)]
    print(f"need {len(need)} tickers, cached {len(got)}, pulling {len(todo)}", flush=True)
    rows = []
    for i, r in enumerate(todo.itertuples(index=False)):
        rows += ET._tape(r.ticker, int(r.T) - 3600, int(r.close_ts))
        if i % 200 == 0:
            print(f"  {i}/{len(todo)}", flush=True)
    fr.append(pd.DataFrame(rows, columns=["ticker", "ts", "no_price", "count", "taker_side"]))
    tr = pd.concat(fr, ignore_index=True)
    tr.to_parquet(OUT / "trades.parquet", index=False)
    print(f"trades {len(tr)} on {tr.ticker.nunique()} tickers; tickers without any print: {len(set(need.ticker) - set(tr.ticker))}")
    return 0


# ------------------------------------------------------------------------------------------------ scoring
def cmd_dev() -> int:
    import e4_timing as ET
    u = priced()
    u["won"] = (u.result == "no").astype(int)
    t = u[(u.phase == "test") & u.tradeable].copy()
    tr = pd.read_parquet(OUT / "trades.parquet")
    g = {k: v.sort_values("ts") for k, v in tr.groupby("ticker")}
    t = t.assign(Tts=t["T"])
    t["fill_tt"], t["fill_n10"] = ET.fills(t, g)
    # through-q depth after T (capacity proxy): contracts printed at no_price < q in (T, close]
    dep = []
    for tk, t0, t1, q in zip(t.ticker, t["T"], t.close_ts, t.q):
        d = g.get(tk)
        dep.append(0.0 if d is None else float(d["count"][(d.ts > t0) & (d.ts <= t1) & (d.no_price < q - 1e-9)].sum()))
    t["depth_tt"] = dep
    days = pd.Index(sorted(t.date.dt.strftime("%Y-%m-%d").unique()))
    half = pd.Timestamp(days[len(days) // 2])
    months = sum((g_.date.max() - g_.date.min()).days / 30.4 for _, g_ in t.groupby("cohort"))
    # E1 frame for edgelab: test, tradeable, NOT E4
    e = t[(t.cohort == "E1") & ~t.e4]
    pd.DataFrame({"ticker": e.ticker, "event_ticker": e.event_ticker, "series": e.series, "date": e.date,
                  "T": e["T"].astype("int64"), "yes_bid": e.yes_bid, "yes_ask": e.yes_ask, "phit": e.phit,
                  "close_ts": e.close_ts.astype("int64"), "won": e.won}).to_parquet(OUT / "edgelab_new.parquet", index=False)
    out = {"test_dates": len(days), "half": str(half.date()), "months": months, "pstar": PSTAR}

    def blk(sel, ex, new_only):
        s = t[sel & (~t.e4 if new_only else True)].copy()
        if ex == "taker":
            s["cost"] = s.tcost
            s["pnl"] = ET.pnl(s.won, s.cost)
            r = ET.stats(s, days, half, months)
        else:
            s["cost"] = s.q
            s["pnl"] = ET.pnl(s.won, s.cost)
            fl = s.fill_tt >= 0
            uf = s[~fl]
            r = ET.stats(s[fl], days, half, months)
            r.update({"picks": int(len(s)), "fill_rate": float(fl.mean()) if len(s) else None, "unfilled": int(len(uf)),
                      "losses_unfilled": int((uf.won == 0).sum()),
                      "loss_rate_unfilled": float((uf.won == 0).mean()) if len(uf) else None,
                      "loss_cp95_unfilled": ET.cp(int((uf.won == 0).sum()), len(uf)) if len(uf) else None,
                      "depth_tt_per_night": float(s[fl].depth_tt.sum() / len(days)),
                      "usd_tt_per_night": float((s[fl].depth_tt * s[fl].q).sum() / len(days))})
            s = s[fl]
        r["by_cohort"] = {c: {"bets": int(len(x)), "losses": int((x.won == 0).sum()), "units": float(x.pnl.sum())}
                          for c, x in s.groupby("cohort")}
        r["mean_phit"] = float(s.phit.mean()) if len(s) else None
        r["exp_losses_model"] = float(s.phit.sum())
        return r

    out["E4_ref"] = {"taker": blk(t.e4, "taker", False), "maker": blk(t.e4, "maker", False)}
    out["C"] = {"taker_new": blk(t.sel_c, "taker", True), "maker_new": blk(t.sel_c, "maker", True),
                "e4_dropped": int((t.e4 & ~t.sel_c).sum())}
    out["PA"] = {"taker_new": blk(t.sel_pa_t, "taker", True), "maker_new": blk(t.sel_pa_m, "maker", True),
                 "e4_dropped_taker": int((t.e4 & ~t.sel_pa_t).sum()), "e4_dropped_maker": int((t.e4 & ~t.sel_pa_m).sum())}
    # test-period bracket reliability (all tradeable test brackets)
    bins = [0, .005, .01, .02, .05, .1, .2, .5, 1.0001]
    rel = t.groupby(pd.cut(t.phit, bins, right=False), observed=True).agg(n=("won", "size"), mean_p=("phit", "mean"),
                                                                           hit=("won", lambda w: 1 - w.mean()))
    out["reliability_test"] = {str(k): v for k, v in rel.round(4).to_dict("index").items()}
    (OUT / "results.json").write_text(json.dumps(out, indent=1, default=float))
    t.to_parquet(OUT / "test_frame.parquet", index=False)
    print(rel)
    for k in ("E4_ref", "C", "PA"):
        for kk, x in out[k].items():
            if not isinstance(x, dict):
                print(f"  {k} {kk} {x}")
                continue
            print(f"{k:6s} {kk:10s} n {x['bets']:4d} L {x['losses']:3d} ({(x['loss_rate'] or 0):.4f} CP {x['loss_cp95']}) "
                  f"b/e {(x['breakeven'] or 0):.4f} ROI {(x['roi'] or 0):+.4f} units {x['units']:+.2f} CI {x.get('units_ci')} "
                  f"p {x.get('p')} h1/h2 {x['units_h1']:+.2f}/{x['units_h2']:+.2f} /night {x['per_night']:.2f} "
                  f"phit {x['mean_phit']} expL {x['exp_losses_model']:.1f}" +
                  (f" fill {x['fill_rate']} unfilledL {x['losses_unfilled']}/{x['unfilled']} $tt/night {x['usd_tt_per_night']:.1f}"
                   if 'fill_rate' in x else ""), x["by_cohort"])
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"nbs": cmd_nbs, "calib": cmd_calib, "tapes": cmd_tapes, "dev": cmd_dev}.get(cmd, lambda: print(__doc__) or 1)())
