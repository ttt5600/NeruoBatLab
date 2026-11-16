"""E4 consensus-reject at other decision times, ADDITIVE capacity framing (DEV only). Pre-registration: notes/e4_timing.md.

    python scripts/e4_timing.py forecasts       # T10 forecasts + error-climatology margin (no prices, no P&L)
    python scripts/e4_timing.py prices   # hourly candles T10-4h..T06D + public tapes (read-only, kx_throttle)
    python scripts/e4_timing.py dev      # score
"""
from __future__ import annotations

import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from scipy.stats import beta as _beta

from sports import wx
from kalshi_paper import PAPER_CITIES
import wx_timing as WT
from kx_throttle import kget

OUT = ROOT / "results" / "e4_timing"
FC = OUT / "fc.parquet"
PX = OUT / "quotes.parquet"
TR = OUT / "trades.parquet"
MARG = OUT / "margin.json"
FEE = 0.07


def t_at(date: pd.Timestamp, tz: str, hour: int, day_off: int) -> int:
    d = date + pd.Timedelta(days=day_off)
    return int(datetime(d.year, d.month, d.day, hour, tzinfo=ZoneInfo(tz)).timestamp())


# ------------------------------------------------------------------------------------------------ forecasts + margin
def cmd_fc() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    u = WT.universe()
    rows = []
    for s, g in u.groupby("series"):
        st, tz = PAPER_CITIES[s]
        lat, lon = wx.station_coords(st)
        days = pd.DataFrame(index=sorted(g.date.unique()))
        for m, start in WT.MODELS.items():
            lo = max(pd.Timestamp(start), g.date.min())
            if lo > g.date.max():
                continue
            h = WT.model_hourly(m, lat, lon, tz, lo.date(), g.date.max().date())
            if h.empty:
                continue
            for T in (10, 22):
                days = days.join(WT.fc_at(h, T).rename(f"fc{T}_{m}"))
        days["series"] = s
        rows.append(days.rename_axis("date").reset_index())
        print(f"{s}: {len(days)} days", flush=True)
    fc = pd.concat(rows, ignore_index=True)
    fc.to_parquet(FC, index=False)
    j = u.merge(fc, on=["series", "date"], how="left")
    for m in WT.MODELS:  # reproduction check: fc22 == frozen fcsafe
        a, b = j[f"fcsafe_{m}"], j[f"fc22_{m}"]
        both = a.notna() & b.notna()
        print(f"  repro {m}: rows {int(both.sum())} |diff|>0.05: {int(((a - b).abs() > 0.05)[both].sum())}")
    # error climatology per station-day (temperature outcome only; no prices)
    sd = j.drop_duplicates("event_ticker")
    res = {}
    for T in (10, 22):
        cols = [f"fc{T}_{m}" for m in WT.MODELS]
        ok = sd[cols].notna().sum(axis=1) >= 3
        x = sd[ok]
        e_up = x.expiration_value - x[cols].max(axis=1)
        e_dn = x[cols].min(axis=1) - x.expiration_value
        grid = np.arange(0, 12.01, 0.5)
        res[T] = {"n": int(len(x)), "tail": {f"{M:.1f}": float((e_up > M).mean() + (e_dn > M).mean()) for M in grid},
                  "e_up_q": [float(q) for q in np.quantile(e_up, [.5, .9, .95, .99])],
                  "e_dn_q": [float(q) for q in np.quantile(e_dn, [.5, .9, .95, .99])]}
    target = res[22]["tail"]["3.0"]
    m10 = next(float(k) for k, v in res[10]["tail"].items() if v <= target)
    out = {"target_tail_22_m3": target, "M10": m10, "M06D": 3.0, "M22": 3.0, "clim": res}
    MARG.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "clim"}),
          {T: {"n": res[T]["n"], "tail": {k: round(v, 4) for k, v in res[T]["tail"].items() if float(k) <= 7}} for T in res})
    return 0


def frame() -> pd.DataFrame:
    u = WT.universe().merge(pd.read_parquet(FC), on=["series", "date"], how="left")
    u["tz"] = u.series.map(lambda s: PAPER_CITIES[s][1])
    u["T10"] = [t_at(d, tz, 10, -1) for d, tz in zip(u.date, u.tz)]
    u["T22"] = [t_at(d, tz, 22, -1) for d, tz in zip(u.date, u.tz)]
    u["T06D"] = [t_at(d, tz, 6, 0) for d, tz in zip(u.date, u.tz)]
    u["close_ts"] = pd.to_datetime(u.close_time, utc=True).map(lambda x: int(x.timestamp()))
    return u


def cands(u: pd.DataFrame, mg: dict) -> pd.DataFrame:
    m = WT.rejected(u, 10, mg["M10"]) | WT.rejected(u, 22, 3.0)
    return u[m].copy()


# ------------------------------------------------------------------------------------------------ prices + tapes
def _candles(tk, t0, t1):
    s = tk.split("-")[0]
    for p in (f"/historical/markets/{tk}/candlesticks", f"/series/{s}/markets/{tk}/candlesticks"):
        d = kget(p, {"start_ts": t0, "end_ts": t1, "period_interval": 60})
        if d and d.get("candlesticks"):
            return d["candlesticks"]
    return []


def _tape(tk, t0, t1):
    out = []
    for p in ("/historical/trades", "/markets/trades"):
        cur = None
        while True:
            prm = {"ticker": tk, "min_ts": t0, "max_ts": t1, "limit": 1000}
            if cur:
                prm["cursor"] = cur
            d = kget(p, prm)
            if not d:
                break
            out += d.get("trades", [])
            cur = d.get("cursor")
            if not cur or not d.get("trades"):
                break
        if out:
            break
    rows = []
    for r in out:
        npx = r.get("no_price_dollars")
        npx = float(npx) if npx is not None else float(r.get("no_price")) / 100
        cnt = r.get("count_fp")
        cnt = float(cnt) if cnt is not None else float(r.get("count", 0))
        rows.append((tk, int(pd.Timestamp(r["created_time"]).timestamp()), npx, cnt, r.get("taker_side")))
    return rows


def cmd_prices() -> int:
    mg = json.loads(MARG.read_text())
    c = cands(frame(), mg)
    print(f"{len(c)} candidate tickers (M10={mg['M10']})", flush=True)

    def one(r):
        cs = _candles(r.ticker, r.T10 - 6 * 3600, r.T06D)
        rec = {"ticker": r.ticker}
        for T in ("T10", "T22", "T06D"):
            x = wx._row(r.ticker, r.event_ticker, cs, getattr(r, T))
            if x:
                rec.update({f"yes_bid_{T}": x["yes_bid"], f"yes_ask_{T}": x["yes_ask"], f"pts_{T}": x["price_ts"]})
        return rec

    with ThreadPoolExecutor(max_workers=3) as ex:
        out = []
        for i, rec in enumerate(ex.map(one, c.itertuples(index=False))):
            out.append(rec)
            if i % 500 == 0:
                print(f"  candles {i}/{len(c)}", flush=True)
    px = pd.DataFrame(out)
    px.to_parquet(PX, index=False)
    print({T: int(px[f"yes_bid_{T}"].notna().sum()) for T in ("T10", "T22", "T06D") if f"yes_bid_{T}" in px}, flush=True)
    # tapes only for tickers tradeable at some T (rule-selected); window T10 .. close
    j = c.merge(px, on="ticker", how="left")
    need = np.zeros(len(j), bool)
    for T in ("T10", "T22", "T06D"):
        b, a = j[f"yes_bid_{T}"], j[f"yes_ask_{T}"]
        need |= ((b > 0) & (a < 1) & ((a - b) <= 0.10) & (a > 0.01)).to_numpy()
    j = j[need]
    print(f"tapes for {len(j)} tickers", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        for i, r in enumerate(ex.map(lambda a: _tape(*a), zip(j.ticker, j.T10, j.close_ts))):
            rows += r
            if i % 500 == 0:
                print(f"  tapes {i}/{len(j)}", flush=True)
                pd.DataFrame(rows, columns=["ticker", "ts", "no_price", "count", "taker_side"]).to_parquet(TR, index=False)
    tr = pd.DataFrame(rows, columns=["ticker", "ts", "no_price", "count", "taker_side"])
    tr.to_parquet(TR, index=False)
    print(f"trades {len(tr)} on {tr.ticker.nunique()} tickers")
    return 0


# ------------------------------------------------------------------------------------------------ scoring
def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else _beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else _beta.ppf(1 - a / 2, k + 1, n - k)
    return [float(lo), float(hi)]


def _cent_floor(x):
    return np.floor(np.round(x * 100, 6)) / 100


def picks(u, T, M, fcT) -> pd.DataFrame:
    b, a = u[f"yes_bid_{T}"], u[f"yes_ask_{T}"]
    ok = WT.rejected(u, fcT, M) & (b > 0).to_numpy() & (a < 1).to_numpy() & ((a - b) <= 0.10).to_numpy() & (a > 0.01).to_numpy()
    s = u[ok].copy()
    s["Tts"] = s[T]
    s["ask"] = 1 - s[f"yes_bid_{T}"]
    s["q"] = np.round(np.minimum(_cent_floor(1 - (s[f"yes_bid_{T}"] + s[f"yes_ask_{T}"]) / 2), s["ask"] - 0.01), 4)
    return s


def fills(s, tr_g, t_from=None):
    """trade-through fill time (first print < q) and N10 fill time (cum >= 10 at <= q), in (T, close]."""
    tt, n10 = np.full(len(s), -1, "int64"), np.full(len(s), -1, "int64")
    t0s = s.Tts.to_numpy() if t_from is None else t_from
    for i, (tk, t0, t1, q) in enumerate(zip(s.ticker, t0s, s.close_ts, s.q)):
        d = tr_g.get(tk)
        if d is None:
            continue
        ts, px, ct = d.ts.to_numpy(), d.no_price.to_numpy(), d["count"].to_numpy()
        w = (ts > t0) & (ts <= t1)
        m = w & (px < q - 1e-9)
        if m.any():
            tt[i] = ts[m].min()
        m = w & (px <= q + 1e-9)
        if m.any():
            cum = np.cumsum(ct[m])
            h = np.nonzero(cum >= 10)[0]
            if len(h):
                n10[i] = ts[m][h[0]]
    return tt, n10


def pnl(won, cost):
    return np.where(won == 1, 1 / cost - 1, -1.0)


def stats(df, days, half, months, boot=True):
    n, k = len(df), int((df.won == 0).sum())
    r = {"bets": n, "losses": k, "loss_rate": k / n if n else None, "loss_cp95": cp(k, n) if n else None,
         "breakeven": float((1 - df.cost).mean()) if n else None, "mean_cost": float(df.cost.mean()) if n else None,
         "units": float(df.pnl.sum()), "roi": float(df.pnl.mean()) if n else None,
         "per_month": n / months, "per_night": n / len(days),
         "units_h1": float(df[df.date < half].pnl.sum()), "units_h2": float(df[df.date >= half].pnl.sum())}
    if boot and n:
        d = df.groupby(df.date.dt.strftime("%Y-%m-%d")).pnl.sum().reindex(days, fill_value=0.0).to_numpy()
        idx = np.random.default_rng(0).integers(0, len(days), size=(4000, len(days)))
        bs = d[idx].sum(1)
        r.update({"units_ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "p": float(np.mean(bs <= 0))})
    return r


def cmd_dev() -> int:
    mg = json.loads(MARG.read_text())
    u = frame().merge(pd.read_parquet(PX), on="ticker", how="left")
    u["won"] = (u.result == "no").astype(int)
    tr = pd.read_parquet(TR)
    tr_g = {k: v.sort_values("ts") for k, v in tr.groupby("ticker")}
    days = pd.Index(sorted(u.date.dt.strftime("%Y-%m-%d").unique()))
    half = pd.Timestamp(days[len(days) // 2])
    months = (u[u.cohort == "E1"].date.max() - u[u.cohort == "E1"].date.min()).days / 30.4 + \
             (u[u.cohort == "X1"].date.max() - u[u.cohort == "X1"].date.min()).days / 30.4
    print(f"DEV {u.series.nunique()} series, {u.event_ticker.nunique()} station-days, {len(days)} dates, "
          f"half {half.date()}, months(sum cohorts) {months:.1f}; M10={mg['M10']}")
    P = {"T22": picks(u, "T22", 3.0, 22), "T10": picks(u, "T10", mg["M10"], 10), "T06D": picks(u, "T06D", 3.0, 22)}
    e = P["T10"][P["T10"].cohort == "E1"]
    pd.DataFrame({"ticker": e.ticker, "event_ticker": e.event_ticker, "series": e.series, "date": e.date,
                  "T": e.T10.astype("int64"), "yes_bid": e.yes_bid_T10, "yes_ask": e.yes_ask_T10,
                  "close_ts": e.close_ts.astype("int64"), "won": e.won}).to_parquet(OUT / "edgelab_T10.parquet", index=False)
    F = {}
    for T, s in P.items():
        F[T] = fills(s, tr_g)
        s["fill_tt"], s["fill_n10"] = F[T]
        miss = int(sum(tk not in tr_g for tk in s.ticker))
        print(f"{T}: picks {len(s)}, no-tape tickers {miss}")
    ref = P["T22"].set_index("ticker")
    out = {"margin": mg, "dates": len(days), "months": months}
    for T in ("T22", "T10", "T06D"):
        s = P[T].copy()
        s["in22"] = s.ticker.isin(ref.index)
        if T == "T10":  # crowding: our T10 resting bid still unfilled at 22:00 on a ticker the 22:00 rule also picks
            s["crowd"] = s.in22 & ((s.fill_tt < 0) | (s.fill_tt > s.T22))
        elif T == "T06D":  # crowding: our 22:00 maker order (trade-through) still resting at 06:00 D
            f22 = ref.reindex(s.ticker).fill_tt.to_numpy()
            s["crowd"] = s.in22.to_numpy() & ((f22 < 0) | (f22 > s.T06D.to_numpy()))
        else:
            s["crowd"] = False
        res = {"picks": len(s), "overlap_22": int(s.in22.sum()), "crowded": int(s.crowd.sum()),
               "picks_per_night": len(s) / len(days)}
        tk = s.copy()
        tk["cost"] = tk.ask + FEE * tk.ask * (1 - tk.ask)
        tk["pnl"] = pnl(tk.won, tk.cost)
        res["taker"] = stats(tk, days, half, months)
        res["taker_new"] = stats(tk[~tk.in22], days, half, months) if T != "T22" else None
        res["taker_by_cohort"] = {c: stats(g, days, half, months, boot=False) for c, g in tk.groupby("cohort")}
        for fm in ("tt", "n10"):
            mk = s.copy()
            mk["cost"] = mk.q
            mk["pnl"] = pnl(mk.won, mk.cost)
            fl = mk[f"fill_{fm}"] >= 0
            uf = mk[~fl]
            r = stats(mk[fl], days, half, months)
            r["fill_rate"] = float(fl.mean())
            r["loss_rate_unfilled"] = float((uf.won == 0).mean()) if len(uf) else None
            r["losses_unfilled"] = int((uf.won == 0).sum())
            r["unfilled"] = int(len(uf))
            res[f"maker_{fm}"] = r
            if T != "T22":
                res[f"maker_{fm}_new"] = stats(mk[fl & ~mk.crowd & (~mk.in22 if T == "T10" else True)], days, half, months)
                res[f"maker_{fm}_by_cohort"] = {c: stats(g, days, half, months, boot=False) for c, g in mk[fl].groupby("cohort")}
        out[T] = res
        s.to_parquet(OUT / f"picks_{T}.parquet", index=False)
    (OUT / "results.json").write_text(json.dumps(out, indent=1, default=float))
    for T in ("T22", "T10", "T06D"):
        r = out[T]
        print(f"\n== {T}: picks {r['picks']} ({r['picks_per_night']:.2f}/night), overlap22 {r['overlap_22']}, crowded {r['crowded']}")
        for k in ("taker", "taker_new", "maker_tt", "maker_tt_new", "maker_n10", "maker_n10_new"):
            x = r.get(k)
            if not x:
                continue
            print(f"  {k:14s} n {x['bets']:5d} L {x['losses']:3d} ({(x['loss_rate'] or 0):.4f} CP {x['loss_cp95']}) "
                  f"b/e {(x['breakeven'] or 0):.4f} ROI {(x['roi'] or 0):+.4f} units {x['units']:+.2f} "
                  f"CI {x.get('units_ci')} p {x.get('p')} h1/h2 {x['units_h1']:+.2f}/{x['units_h2']:+.2f} "
                  f"/mo {x['per_month']:.1f}" + (f" fill {x['fill_rate']:.3f} unfilled L {x['losses_unfilled']}/{x['unfilled']}"
                                                 if 'fill_rate' in x else ""))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"forecasts": cmd_fc, "prices": cmd_prices, "dev": cmd_dev}.get(cmd, lambda: print(__doc__) or 1)())
