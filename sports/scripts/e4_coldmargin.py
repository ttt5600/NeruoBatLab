"""E4 cold-margin 3F -> 1F, scored once on a fresh unsealed window F (agent e4_coldmargin). Prereg: notes/e4_coldmargin.md.

    python scripts/e4_coldmargin.py markets   # window-F markets, one read-only GET per event (no sealed date requested)
    python scripts/e4_coldmargin.py prices    # last hourly candle ending at/before 22:00 local D-1 (wx._row)
    python scripts/e4_coldmargin.py fc        # fcsafe members (Open-Meteo previous-runs previous_day1) for D and D-7
    python scripts/e4_coldmargin.py live      # as-issued LIVE forecasts archived by kalshi_paper.jsonl + e4p.jsonl
    python scripts/e4_coldmargin.py frame     # selections only (no outcome is read)
    python scripts/e4_coldmargin.py tapes     # public trade tapes (kx_throttle) for every selected / placebo-pool bracket
    python scripts/e4_coldmargin.py score     # the one pre-registered scoring -> results/e4_coldmargin/score.json

Window F: orig-7 event dates 2026-10-07..2026-11-22, X1 2026-10-19..2026-11-22 (after each cohort's sealed window).
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, os.path.expanduser("~/agent-harness/lib"))

import numpy as np
import pandas as pd

from sports import wx
from kalshi_paper import PAPER_CITIES, X1_CITIES
import wx_timing as WT
import traps

OUT = ROOT / "results" / "e4_coldmargin"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = list(WT.MODELS)
WIN = {"orig7": ("2026-10-07", "2026-11-22"), "x1": ("2026-10-19", "2026-11-22")}
SEALED = {"orig7": ("2025-07-01", "2026-10-06"), "x1": ("2026-03-01", "2026-10-18")}
FEE = 0.07
BUCKETS = [0, .90, .95, .97, .98, 1.0001]
QK = Path.home() / ".claude" / "quant-kalshi" / "data"
QS = Path.home() / ".claude" / "quant-sports" / "data" / "kx_forward"


def cohort(s: str) -> str:
    return "x1" if s in X1_CITIES else "orig7"


def win_dates(c: str) -> pd.DatetimeIndex:
    d = pd.date_range(*WIN[c], freq="D")
    lo, hi = map(pd.Timestamp, SEALED[c])
    assert not ((d >= lo) & (d <= hi)).any(), "window F touches a sealed date"
    return d


def ev_ticker(s: str, d: pd.Timestamp) -> str:
    return f"{s}-{d.strftime('%y%b%d').upper()}"


def kg(path: str, params: dict | None = None):
    """kx_throttle.kget plus a 0.7 s pause: ~1 req/s, so the shared throttle slots stay free for the live-window loggers
    (pulls run partly inside 03-07Z, the E4 / e4p decision window)."""
    import time
    from kx_throttle import kget
    r = kget(path, params)
    time.sleep(float(os.environ.get("E4C_PAUSE", "0.7")))
    return r


# ------------------------------------------------------------------------------------------------ selection (pure)
def members(d: pd.DataFrame, prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cols = [f"{prefix}{m}" for m in MODELS]
    v = d[cols].to_numpy(float)
    ok = np.isfinite(v)
    hi = np.where(ok, v, -np.inf).max(1)
    lo = np.where(ok, v, np.inf).min(1)
    return hi, lo, ok.sum(1)


def reject(st, f, c, hi, lo, n, mw: float, mc: float) -> np.ndarray:
    """Frozen E4 formula with separate warm (mw) and cold (mc) margins; mw = mc = 3 is kalshi_paper.rejected()."""
    st, f, c = np.asarray(st), np.asarray(f, float), np.asarray(c, float)
    warm = np.where(st == "greater", hi + mw < f + 1, np.where(st == "between", hi + mw < f, False))
    cold = np.where(st == "less", lo - mc > c - 1, np.where(st == "between", lo - mc > c, False))
    return (warm | cold) & (n >= 3)


def tradeable(yb, ya) -> np.ndarray:
    yb, ya = np.asarray(yb, float), np.asarray(ya, float)
    return (yb > 0) & (ya < 1) & ((ya - yb) <= 0.10) & (ya > 0.01)


def maker_q(yb, ya) -> np.ndarray:
    yb, ya = np.asarray(yb, float), np.asarray(ya, float)
    ask = 1 - yb
    return np.round(np.minimum(np.floor(np.round((1 - (yb + ya) / 2) * 100, 6)) / 100, ask - 0.01), 4)


def select(d: pd.DataFrame, prefix: str) -> dict:
    hi, lo, n = members(d, prefix)
    st, f, c = d.strike_type.to_numpy(), d.floor_strike.to_numpy(float), d.cap_strike.to_numpy(float)
    tr = d["tradeable"].to_numpy(bool)
    e4 = reject(st, f, c, hi, lo, n, 3.0, 3.0) & tr
    mc1 = reject(st, f, c, hi, lo, n, 3.0, 1.0) & tr
    mw1 = reject(st, f, c, hi, lo, n, 1.0, 3.0) & tr
    return {"e4": e4, "mc1": mc1, "extra": mc1 & ~e4, "warm_band": mw1 & ~e4, "nmod": n}


# ------------------------------------------------------------------------------------------------ data pulls
def cmd_markets() -> int:
    rows, miss = [], []
    for s in PAPER_CITIES:
        for d in win_dates(cohort(s)):
            ev = ev_ticker(s, d)
            r = kg("/markets", {"event_ticker": ev, "limit": 100}) or {}
            ms = r.get("markets", [])
            if not ms:
                miss.append(ev)
            for m in ms:
                rows.append({k: m.get(k) for k in ("ticker", "event_ticker", "strike_type", "floor_strike", "cap_strike",
                                                   "expiration_value", "result", "status", "open_time", "close_time")}
                            | {"series": s, "date": d, "cohort": cohort(s)})
        print(f"{s}: {sum(r['series'] == s for r in rows)} markets", flush=True)
    m = pd.DataFrame(rows)
    m["expiration_value"] = pd.to_numeric(m["expiration_value"], errors="coerce")
    m["expiration_value"] = m.groupby("event_ticker")["expiration_value"].transform("max")
    m.to_parquet(OUT / "markets.parquet", index=False)
    (OUT / "markets_missing.json").write_text(json.dumps(miss))
    print(f"{len(m)} markets on {m.event_ticker.nunique()} events; {len(miss)} events with no markets; statuses "
          f"{m.status.value_counts().to_dict()}")
    return 0


def _T(date: pd.Timestamp, tz: str) -> int:
    return WT.t_local(pd.Timestamp(date), tz, 22)


def cmd_prices() -> int:
    m = pd.read_parquet(OUT / "markets.parquet")
    out = []
    for ev, g in m.groupby("event_ticker"):
        s = g.series.iloc[0]
        T = _T(g.date.iloc[0], PAPER_CITIES[s][1])
        d = kg(f"/series/{s}/events/{ev}/candlesticks", {"start_ts": T - 3 * 3600, "end_ts": T, "period_interval": 60}) or {}
        got = set()
        for tk, cs in zip(d.get("market_tickers", []), d.get("market_candlesticks", [])):
            r = wx._row(tk, ev, cs, T)
            if r:
                out.append(r | {"T": T})
                got.add(tk)
        for tk in sorted(set(g.ticker) - got):  # per-market fallback (live, then historical)
            for p in (f"/series/{s}/markets/{tk}/candlesticks", f"/historical/markets/{tk}/candlesticks"):
                dd = kg(p, {"start_ts": T - 3 * 3600, "end_ts": T, "period_interval": 60}) or {}
                r = wx._row(tk, ev, dd.get("candlesticks", []), T)
                if r:
                    out.append(r | {"T": T})
                    break
    px = pd.DataFrame(out).drop_duplicates("ticker")
    assert (px.price_ts <= px["T"]).all(), "quote after decision"
    traps.check_epoch_seconds(px["T"].to_numpy(float), "T")
    traps.check_epoch_seconds(px["price_ts"].to_numpy(float), "price_ts")
    px.to_parquet(OUT / "quotes.parquet", index=False)
    print(f"quotes {len(px)} of {len(m)} markets; decision-candle age (s): {(px['T'] - px.price_ts).describe().round(0).to_dict()}")
    return 0


def cmd_fc() -> int:
    rows = []
    for s in PAPER_CITIES:
        st, tz = PAPER_CITIES[s]
        lat, lon = wx.station_coords(st)
        lo, hi = pd.Timestamp(WIN[cohort(s)][0]) - pd.Timedelta(days=7), pd.Timestamp(WIN[cohort(s)][1])
        days = pd.DataFrame(index=pd.date_range(lo, hi, freq="D"))
        for mdl in MODELS:
            h = WT.model_hourly(mdl, lat, lon, tz, lo.date(), hi.date())
            if h.empty:
                continue
            days = days.join(WT.fc_at(h, 22).rename(f"fc_{mdl}"))
        days["series"] = s
        rows.append(days.rename_axis("date").reset_index())
        print(f"{s}: {len(days)} days, member coverage {days[[c for c in days if c.startswith('fc_')]].notna().mean().round(2).to_dict()}", flush=True)
    fc = pd.concat(rows, ignore_index=True)
    fc.to_parquet(OUT / "fc.parquet", index=False)
    return 0


def cmd_live() -> int:
    """As-issued LIVE member values (the live /v1/forecast response the 22:00 logger used). kalshi_paper.jsonl carries them
    only on pick rows (city-nights with >= 1 live E4 pick); e4p.jsonl city_day rows carry them for every city-night from
    event day 2026-11-17. Forecast fields only are read here (no result field)."""
    rows = []
    for line in (QK / "kalshi_paper.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows.append({"event_ticker": r["event"], "series": r["city"], "src": "kalshi_paper", "at": r["logged_at"],
                     **{f"live_{k}": v for k, v in (r.get("forecasts") or {}).items()}})
    for line in (QS / "e4p.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("kind") == "city_day" and r.get("status") == "decided":
            rows.append({"event_ticker": r["event"], "series": r["series"], "src": "e4p", "at": r["decided_at"],
                         **{f"live_{k}": v for k, v in (r.get("forecasts") or {}).items()}})
    d = pd.DataFrame(rows)
    cols = [f"live_{m}" for m in MODELS]
    # one forecast set per (event, source); kalshi_paper rows of one city-night must agree
    g = d.groupby(["event_ticker", "src"])[cols].nunique(dropna=True).max(axis=1)
    bad = g[g > 1]
    print(f"city-night/source pairs with >1 distinct forecast set: {len(bad)}")
    d = d.drop_duplicates(["event_ticker", "src"])
    piv = d.pivot(index="event_ticker", columns="src", values=cols)
    both = piv.dropna(how="any")
    agree = {m: float((both[(f'live_{m}', 'kalshi_paper')] - both[(f'live_{m}', 'e4p')]).abs().max()) for m in MODELS} if len(both) else {}
    print(f"events with both sources: {len(both)}; max |diff| by model {agree}")
    keep = d.sort_values("src", key=lambda x: x.map({"kalshi_paper": 0, "e4p": 1})).drop_duplicates("event_ticker")
    keep.to_parquet(OUT / "live_fc.parquet", index=False)
    picks = sorted({json.loads(l)["ticker"] for l in (QK / "kalshi_paper.jsonl").read_text().splitlines()})
    (OUT / "live_e4_tickers.json").write_text(json.dumps(picks))
    print(f"live forecast city-nights {len(keep)} ({keep.src.value_counts().to_dict()}); live E4 pick tickers {len(picks)}")
    return 0


# ------------------------------------------------------------------------------------------------ frame (no outcomes)
def cmd_frame() -> int:
    m = pd.read_parquet(OUT / "markets.parquet")
    px = pd.read_parquet(OUT / "quotes.parquet").drop(columns=["event_ticker"])
    fc = pd.read_parquet(OUT / "fc.parquet")
    n0 = len(m)
    u = traps.checked_merge(m, px, on="ticker", how="left", validate="one_to_one")
    u = traps.checked_merge(u, fc, on=["series", "date"], how="left", validate="many_to_one")
    pl = fc.assign(date=fc.date + pd.Timedelta(days=7)).rename(columns={f"fc_{k}": f"pl7_{k}" for k in MODELS})
    u = traps.checked_merge(u, pl, on=["series", "date"], how="left", validate="many_to_one")
    lv = pd.read_parquet(OUT / "live_fc.parquet")[["event_ticker", "src", "at"] + [f"live_{k}" for k in MODELS]]
    u = traps.checked_merge(u, lv.rename(columns={"src": "live_src", "at": "live_at"}), on="event_ticker", how="left",
                            validate="many_to_one")
    assert len(u) == n0
    u["tz"] = u.series.map(lambda s: PAPER_CITIES[s][1])
    u["T"] = [_T(d, tz) for d, tz in zip(u.date, u.tz)]
    u["close_ts"] = traps.epoch_seconds(u.close_time).astype("int64")
    traps.check_epoch_seconds(u["close_ts"].to_numpy(float), "close_ts")
    assert ((u.close_ts - u["T"]) > 0).all() and ((u.close_ts - u["T"]) < 40 * 3600).all(), "close vs decision"
    u["settled"] = u.result.isin(["yes", "no"])
    own = np.select([u.strike_type == "greater", u.strike_type == "less", u.strike_type == "between"],
                    [u.expiration_value > u.floor_strike, u.expiration_value < u.cap_strike,
                     (u.expiration_value >= u.floor_strike) & (u.expiration_value <= u.cap_strike)], default=np.nan).astype(float)
    u["result_mismatch"] = u.settled & u.expiration_value.notna() & (own != (u.result == "yes").astype(float))  # critic fix 11-24: NaN value contradicts nothing
    u["has_quote"] = u.yes_bid.notna()
    u["tradeable"] = tradeable(u.yes_bid, u.yes_ask) & u.has_quote.to_numpy()
    u["ask"] = 1 - u.yes_bid
    u["q"] = maker_q(u.yes_bid, u.yes_ask)
    u["bkt"] = pd.cut(u.ask, BUCKETS, labels=False, right=False)
    s = select(u, "fc_")
    for k in ("e4", "mc1", "extra", "warm_band"):
        u[k] = s[k]
    u["nmod"] = s["nmod"]
    p = select(u, "pl7_")
    u["p2"] = p["mc1"] & ~p["e4"] & ~u.e4.to_numpy()     # wrong-day (D-7) cold band, minus the real E4 picks
    lv_ok = u.live_src.notna().to_numpy()
    L = select(u, "live_")
    u["in_live"] = lv_ok
    u["live_e4"] = L["e4"] & lv_ok
    u["live_extra"] = L["extra"] & lv_ok
    # P1 pool: tradeable, not E4, not EXTRA, on a (date, bucket) cell where EXTRA has picks
    cells = set(zip(u[u.extra].date, u[u.extra].bkt))
    u["p1_pool"] = u.tradeable & ~u.e4 & ~u.extra & np.array([(d, b) in cells for d, b in zip(u.date, u.bkt)])
    keep = u.settled & ~u.result_mismatch
    print(f"markets {n0}: settled {int(u.settled.sum())}, result_mismatch {int(u.result_mismatch.sum())}, quote {int(u.has_quote.sum())}, "
          f"tradeable {int(u.tradeable.sum())}; member counts (>=3) {int((u.nmod >= 3).sum())}")
    print(u[keep].groupby("cohort")[["e4", "mc1", "extra", "warm_band", "p2", "p1_pool", "live_e4", "live_extra"]].sum())
    print(f"city-nights: {u.groupby(['series', 'date']).ngroups}; with live forecasts {u[lv_ok].event_ticker.nunique()}")
    u.to_parquet(OUT / "frame.parquet", index=False)
    return 0


def _tape(tk: str, t0: int, t1: int) -> list[tuple]:
    """Public trades in [t0, t1] (live endpoint first: window-F markets are recent; historical as fallback)."""
    out = []
    for path in ("/markets/trades", "/historical/trades"):
        cur = None
        while True:
            prm = {"ticker": tk, "min_ts": t0, "max_ts": t1, "limit": 1000}
            if cur:
                prm["cursor"] = cur
            d = kg(path, prm)
            if not d:
                break
            out += d.get("trades", [])
            cur = d.get("cursor")
            if not cur or not d.get("trades"):
                break
        if out:
            break
    if not out:
        return []
    ts = traps.epoch_seconds(pd.Series([r["created_time"] for r in out])).round().astype("int64").to_numpy()
    traps.check_epoch_seconds(ts.astype(float), "trade ts")
    rows = []
    for r, t in zip(out, ts):
        npx = r.get("no_price_dollars")
        npx = float(npx) if npx is not None else float(r.get("no_price")) / 100
        cnt = r.get("count_fp")
        cnt = float(cnt) if cnt is not None else float(r.get("count", 0))
        rows.append((tk, int(t), npx, cnt, r.get("taker_side")))
    return rows


def cmd_tapes() -> int:
    """Priority order: EXTRA (primary), E4 reference, P2, P1 pool, U_live, warm band. Resumable."""
    u = pd.read_parquet(OUT / "frame.parquet")
    u = u[u.settled & ~u.result_mismatch]
    f = OUT / "trades.parquet"
    cols = ["ticker", "ts", "no_price", "count", "taker_side"]
    prev = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=cols)
    have = set(prev.ticker)
    order = []
    for k in ("extra", "e4", "p2", "p1_pool", "live_extra", "live_e4", "warm_band"):
        order += [t for t in u[u[k]].ticker if t not in have and t not in order]
    tix = u.set_index("ticker")
    print(f"pulling {len(order)} tapes ({len(have)} already)", flush=True)
    rows = []
    for i, tk in enumerate(order):
        r = tix.loc[tk]
        got = _tape(tk, int(r["T"]) - 3600, int(r["close_ts"]))
        rows += got if got else [(tk, 0, np.nan, 0.0, "none")]   # marker: fetched, no prints
        if i % 100 == 0 or i == len(order) - 1:
            pd.concat([prev, pd.DataFrame(rows, columns=cols)], ignore_index=True).to_parquet(f, index=False)
            print(f"  {i + 1}/{len(order)}", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ scoring
def pnl(won, cost):
    return np.where(np.asarray(won) == 1, 1 / np.asarray(cost, float) - 1, -1.0)


def maker_cols(u: pd.DataFrame, tapes: dict) -> pd.DataFrame:
    tt, n10, has = [], [], []
    for tk, t0, t1, q in zip(u.ticker, u["T"], u.close_ts, u.q):
        d = tapes.get(tk)
        if d is None:
            tt.append(False); n10.append(False); has.append(False)
            continue
        ts, px, ct = d.ts.to_numpy(), d.no_price.to_numpy(float), d["count"].to_numpy(float)
        w = (ts > t0) & (ts <= t1)
        tt.append(bool((w & (px < q - 1e-9)).any()))
        n10.append(bool(ct[w & (px <= q + 1e-9)].sum() >= 10))
        has.append(True)
    u = u.assign(has_tape=has, tt=tt, n10=n10)
    won = u.won.to_numpy()
    skip = ((u.ask - u.q) > 0.05 + 1e-9).to_numpy() | (u.q <= 0).to_numpy()
    for k in ("tt", "n10"):
        fil = np.where(u.has_tape, u[k], won == 0) & ~skip
        u[f"{k}_filled"] = fil
        u[f"{k}_pnl"] = np.where(fil, pnl(won, u.q.to_numpy()), 0.0)
    u["maker_skip"] = skip
    u["tcost"] = u.ask + FEE * u.ask * (1 - u.ask)
    u["tk_pnl"] = pnl(won, u.tcost.to_numpy())
    return u


def day_vec(u: pd.DataFrame, mask, col: str, days: list[str]) -> np.ndarray:
    x = u[np.asarray(mask, bool)]
    return x.groupby(x.date.dt.strftime("%Y-%m-%d"))[col].sum().reindex(days, fill_value=0.0).to_numpy()


def boot(v: np.ndarray, n=4000, seed=0) -> dict:
    idx = np.random.default_rng(seed).integers(0, len(v), size=(n, len(v)))
    bs = v[idx].sum(1)
    return {"sum": round(float(v.sum()), 3), "ci95": [round(float(np.quantile(bs, .025)), 3), round(float(np.quantile(bs, .975)), 3)],
            "p_le0": float(np.mean(bs <= 0))}


def ratio_ci(num: np.ndarray, den: np.ndarray, n=4000, seed=1) -> list:
    idx = np.random.default_rng(seed).integers(0, len(num), size=(n, len(num)))
    a, b = num[idx].sum(1), den[idx].sum(1)
    r = a[b > 0] / b[b > 0]
    return [round(float(np.quantile(r, .025)), 4), round(float(np.quantile(r, .975)), 4)]


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def summary(u, mask, days, half) -> dict:
    x = u[np.asarray(mask, bool)]
    out = {"picks": int(len(x)), "tape_cov": round(float(x.has_tape.mean()), 4) if len(x) else None}
    for k, lab in (("tt", "maker_tt"), ("n10", "maker_n10")):
        f = x[x[f"{k}_filled"]]
        v = day_vec(u, np.asarray(mask, bool), f"{k}_pnl", days)
        out[lab] = {"fills": int(len(f)), "losses": int((f.won == 0).sum()), "loss_rate": round(float((f.won == 0).mean()), 4) if len(f) else None,
                    "loss_wilson": wilson(int((f.won == 0).sum()), len(f)), "breakeven_loss": round(float((1 - f.q).mean()), 4) if len(f) else None,
                    "mean_q": round(float(f.q.mean()), 4) if len(f) else None, **boot(v),
                    "halves": [round(float(x[x.date < half][f"{k}_pnl"].sum()), 3), round(float(x[x.date >= half][f"{k}_pnl"].sum()), 3)],
                    "unfilled_losses": int(((~x[f"{k}_filled"]) & (x.won == 0) & ~x.maker_skip).sum())}
        nf = day_vec(u.assign(one=u[f"{k}_filled"].astype(float)), np.asarray(mask, bool), "one", days)
        out[lab]["roi_per_fill"] = round(float(f[f"{k}_pnl"].mean()), 4) if len(f) else None
        out[lab]["roi_per_fill_ci"] = ratio_ci(v, nf) if len(f) else None
    v = day_vec(u, np.asarray(mask, bool), "tk_pnl", days)
    out["taker"] = {"bets": int(len(x)), "losses": int((x.won == 0).sum()), "loss_wilson": wilson(int((x.won == 0).sum()), len(x)),
                    "mean_ask": round(float(x.ask.mean()), 4) if len(x) else None, **boot(v),
                    "halves": [round(float(x[x.date < half].tk_pnl.sum()), 3), round(float(x[x.date >= half].tk_pnl.sum()), 3)],
                    "roi": round(float(x.tk_pnl.mean()), 4) if len(x) else None}
    return out


def placebo_random(u, days, col, n=2000, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    ex = u[u.extra]
    need = ex.groupby(["date", "bkt"]).size()
    pool = {k: g[col].to_numpy() for k, g in u[u.p1_pool].groupby(["date", "bkt"])}
    short = 0
    sums = np.zeros(n)
    for k, cnt in need.items():
        pv = pool.get(k, np.array([]))
        take = min(cnt, len(pv))
        short += cnt - take
        if take == 0:
            continue
        for i in range(n):
            sums[i] += pv[rng.choice(len(pv), take, replace=False)].sum()
    act = float(ex[col].sum())
    return {"col": col, "mean": round(float(sums.mean()), 3), "p90": round(float(np.quantile(sums, .9)), 3),
            "p95": round(float(np.quantile(sums, .95)), 3), "actual": round(act, 3),
            "actual_pctile": round(float(np.mean(sums < act)), 4), "shortfall_picks": int(short), "cells": int(len(need))}


def cmd_score() -> int:
    u = pd.read_parquet(OUT / "frame.parquet")
    n_all = len(u)
    u = u[u.settled & ~u.result_mismatch].copy()
    u["won"] = (u.result == "no").astype(int)
    tr_all = pd.read_parquet(OUT / "trades.parquet")
    fetched = set(tr_all.ticker)                      # fetched with no prints (marker row) = has tape, never filled
    tr = tr_all[tr_all.taker_side != "none"]
    tapes = {**{k: tr_all.iloc[0:0] for k in fetched}, **{k: v.sort_values("ts") for k, v in tr.groupby("ticker")}}
    u = maker_cols(u, tapes)
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range(min(w[0] for w in WIN.values()), max(w[1] for w in WIN.values()))]
    half = pd.Timestamp(days[len(days) // 2])
    res = {"n_markets": n_all, "n_scored_rows": int(len(u)), "days": len(days), "half_split": str(half.date())}
    tape_cov = float(u[u.extra].has_tape.mean())
    res["primary_exec"] = "maker_tt" if tape_cov >= 0.97 else "taker"
    res["EXTRA"] = summary(u, u.extra, days, half)
    res["E4_ref"] = summary(u, u.e4, days, half)
    res["MC1_union"] = summary(u, u.mc1, days, half)
    res["P2_wrongday"] = summary(u, u.p2, days, half)
    res["P3_warm_band"] = summary(u, u.warm_band, days, half)
    pool_cov = float(u[u.p1_pool].has_tape.mean())
    res["P1_random"] = placebo_random(u, days, "tt_pnl" if pool_cov >= 0.97 else "tk_pnl")
    res["P1_random"]["pool_tape_cov"] = round(pool_cov, 4)
    res["P1_random_taker"] = placebo_random(u, days, "tk_pnl", seed=1)
    for c in ("orig7", "x1"):
        res[f"EXTRA_{c}"] = summary(u, u.extra & (u.cohort == c), days, half)
    res["U_live"] = {"city_nights": int(u[u.in_live].event_ticker.nunique()), "EXTRA_live": summary(u, u.live_extra, days, half),
                     "E4_live": summary(u, u.live_e4, days, half)}
    live_tk = set(json.loads((OUT / "live_e4_tickers.json").read_text()))
    win_live = u[u.ticker.isin(live_tk)]
    res["recon"] = {"live_e4_picks_in_window": int(len(win_live)), "fcsafe_e4_picks": int(u.e4.sum()),
                    "overlap": int((u.e4 & u.ticker.isin(live_tk)).sum()),
                    "live_picks_in_fcsafe_extra": int((u.extra & u.ticker.isin(live_tk)).sum()),
                    "fcsafe_minus_live_by_model": {m: round(float((u.drop_duplicates('event_ticker')[f'fc_{m}'] - u.drop_duplicates('event_ticker')[f'live_{m}']).mean()), 3)
                                                   for m in MODELS},
                    "U_live_extra_vs_fcsafe_extra_overlap": int((u.live_extra & u.extra).sum())}
    e = res["EXTRA"]["maker_tt" if res["primary_exec"] == "maker_tt" else "taker"]
    S, p, h = e["sum"], e["p_le0"], e["halves"]
    p2 = res["P2_wrongday"]["maker_tt" if res["primary_exec"] == "maker_tt" else "taker"]["sum"]
    p1 = res["P1_random"]["p90"]
    conds = {"S>0": S > 0, "p<.05": p < .05, "halves>0": h[0] > 0 and h[1] > 0, "S>P2": S > p2, "S>P1_p90": S > p1}
    res["conditions"] = conds
    res["verdict"] = "KILL" if S <= 0 else ("PASS" if all(conds.values()) else "LEAD")
    ins = 0.040
    ci = res["EXTRA"]["maker_tt"]["roi_per_fill_ci"]
    res["insample_roi_per_fill_inside_ci"] = bool(ci and ci[0] <= ins <= ci[1])
    (OUT / "score.json").write_text(json.dumps(res, indent=1, default=str))
    cols = ["ticker", "cohort", "date", "strike_type", "floor_strike", "cap_strike", "expiration_value", "yes_bid", "yes_ask", "q",
            "tt_filled"] + [f"fc_{m}" for m in MODELS]
    los = u[u.extra & (u.won == 0)][cols].copy()
    los["coldest"] = u.loc[los.index, [f"fc_{m}" for m in MODELS]].min(axis=1)
    los.to_csv(OUT / "extra_losers.csv", index=False)
    u[u.extra | u.e4 | u.p2 | u.warm_band | u.live_extra].to_parquet(OUT / "scored_rows.parquet", index=False)
    print(json.dumps({k: res[k] for k in ("primary_exec", "verdict", "conditions", "insample_roi_per_fill_inside_ci")}, indent=1))
    for k in ("EXTRA", "EXTRA_orig7", "EXTRA_x1", "E4_ref", "P2_wrongday", "P3_warm_band"):
        r = res[k]
        print(f"{k:13s} picks {r['picks']:4d} cov {r['tape_cov']} | TT fills {r['maker_tt']['fills']} L {r['maker_tt']['losses']} "
              f"{r['maker_tt']['sum']:+.2f} {r['maker_tt']['ci95']} p {r['maker_tt']['p_le0']:.3f} halves {r['maker_tt']['halves']} | "
              f"N10 {r['maker_n10']['fills']} {r['maker_n10']['sum']:+.2f} | taker {r['taker']['bets']} L {r['taker']['losses']} "
              f"{r['taker']['sum']:+.2f} {r['taker']['ci95']}")
    print("P1", res["P1_random"], "\nP1 taker", res["P1_random_taker"])
    print("U_live", {k: (v if not isinstance(v, dict) else {kk: v[kk] for kk in ("picks",)} | {"tt": v["maker_tt"]["sum"], "tt_ci": v["maker_tt"]["ci95"], "fills": v["maker_tt"]["fills"], "L": v["maker_tt"]["losses"], "taker": v["taker"]["sum"]}) for k, v in res["U_live"].items()})
    print("recon", res["recon"])
    print(los.to_string())
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "score"
    sys.exit({"markets": cmd_markets, "prices": cmd_prices, "fc": cmd_fc, "live": cmd_live, "frame": cmd_frame,
              "tapes": cmd_tapes, "score": cmd_score}[cmd]())
