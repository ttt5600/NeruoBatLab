"""Polymarket: expected return by TIME-TO-RESOLUTION x price x category (DESCRIPTIVE, dev window only).

    python scripts/pm_timing.py enum      # 92 cached dev days (smallmarket seed) -> all binary markets with outcome
    python scripts/pm_timing.py prices    # CLOB prices-history: window A [end-8d, end+1h] + window B [S-36d, S-29d]
    python scripts/pm_timing.py panel     # -> data/pm_timing/panel.parquet (one row per market x horizon)
    python scripts/pm_timing.py describe  # tables (cluster bootstrap over events) -> data/pm_timing/tables.json

User idea 2026-11-06: "EV within the timeframe of the Polymarket system". Question: does realized-minus-implied
depend on the time remaining to the SCHEDULED resolution (early longshots overpriced / favourites underpriced as a
capital-lockup discount, or the reverse)?

Design (fixed before any outcome was tabulated by time-to-resolution):
  Universe : the 92 random DEV days of smallmarket_calibration (2025-05-01..2026-04-30, seed 20261027); every closed
             binary market whose endDate falls on one of those days. DEV ONLY -- nothing from 2026-05-01 on is read.
  Sample   : (a) the exact smallmarket pm_select sample (26.6k mkts, prices cached) + (b) long-dated augmentation:
             markets with S - startDate >= 30 d not already in (a), <= 2,000 per category, <= 10 legs per event.
  Anchor S : min(endDate, gameStartTime) as stored by gamma (the scheduled end; caveat: gamma stores the LAST
             endDate, so a postponed market carries its revised date -- flagged `extended` if closedTime > end+2d).
  Horizons : h in {30d, 7d, 3d, 1d, 6h, 1h}; t = S - h. A row exists only if the market was still open at t
             (t < closedTime) and a display price exists within 6 h before t (last point at or before t).
  Price    : CLOB display price of outcome-0 (mid, or last trade if spread > 10c). Folded to the LONGSHOT side
             ls = min(p, 1-p); fav = 1 - ls. Outcome = whether that side won (50/50 resolutions dropped).
  Returns  : fav_ret = fav_won / fav - 1 (buy favourite at mid, no costs); ls_ret likewise. Lockup = closedTime - t
             (realized; descriptive only). Annualized = mean ret / mean lockup-years per cell (simple, not compounded).
  CIs      : 95% percentile bootstrap, 1000 draws, resampling EVENTS (all legs + horizons of an event together).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smallmarket_calibration as sm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "pm_timing"
OUT.mkdir(parents=True, exist_ok=True)
H = {"30d": 720, "7d": 168, "3d": 72, "1d": 24, "6h": 6, "1h": 1}
LONG = ("30d",)


def dev_days():
    rng = random.Random(20261027)
    days = list(pd.date_range(sm.DEV0, sm.DEV1 - pd.Timedelta(days=1), freq="D"))
    return sorted(rng.sample(days, 92))


def cmd_enum() -> int:
    allrows = []
    for evs in sm.pmap(sm.pm_day, dev_days(), workers=3):
        allrows += sm.pm_flatten(evs, keep_outcome=True)
    df = pd.DataFrame(allrows).drop_duplicates("market_id")
    df["end"] = pd.to_datetime(df["end"], utc=True, errors="coerce", format="ISO8601")
    df = df[(df.end >= sm.DEV0) & (df.end < sm.DEV1)]
    assert df.end.max() < sm.DEV1, "dev window breach"
    df.to_parquet(OUT / "markets_all.parquet", index=False)
    print(len(df), "markets;", df.event_id.nunique(), "events")
    return 0


def anchor(df):
    gs = pd.to_datetime(df.game_start, utc=True, errors="coerce", format="mixed")
    S = df.end.copy()
    m = gs.notna() & (gs < S)
    S[m] = gs[m]
    return S


def select(df):
    df = df[df.y0.notna()].copy()
    base = sm.pm_select(df)
    base = base.assign(src="base")
    rest = df[~df.market_id.isin(base.market_id)].copy()
    st = pd.to_datetime(rest.start, utc=True, errors="coerce", format="ISO8601")
    rest = rest[(anchor(rest) - st) >= pd.Timedelta(days=30)]
    rng = np.random.default_rng(30)
    keep = []
    for cat, g in rest.groupby("cat"):
        g = g.sample(frac=1, random_state=int(rng.integers(1e9)))
        g = g.groupby("event_id", sort=False).head(10)
        keep.append(g.head(2000))
    aug = pd.concat(keep).assign(src="aug")
    return pd.concat([base, aug], ignore_index=True)


def hist(token, a, b):
    d = sm.get(f"{sm.CLOB}/prices-history", {"market": token, "startTs": a, "endTs": b, "fidelity": 60})
    if not d or not d.get("history"):
        return np.array([]), np.array([])
    h = d["history"]
    return np.array([x["t"] for x in h]), np.array([x["p"] for x in h], dtype=float)


def price_row(r):
    end = int(r.end.timestamp())
    S = int(r.S.timestamp())
    closed = pd.to_datetime(r.closed_time, utc=True, errors="coerce")
    cts = closed.timestamp() if pd.notna(closed) else np.inf
    st = pd.to_datetime(r.start, utc=True, errors="coerce", format="ISO8601")
    out = {"market_id": r.market_id}
    tA, pA = hist(r.token0, end - 8 * 86400, end + 3600)  # identical URL to smallmarket pm_hist (cache)
    need_b = pd.notna(st) and st.timestamp() <= S - 30 * 86400
    tB, pB = hist(r.token0, S - 36 * 86400, S - 29 * 86400) if need_b else (np.array([]), np.array([]))
    for k, hh in H.items():
        t = S - hh * 3600
        if t >= cts:
            continue
        ts, ps = (tB, pB) if k in LONG else (tA, pA)
        if k not in LONG and t < end - 8 * 86400:
            continue
        if len(ts) == 0:
            continue
        j = np.searchsorted(ts, t, side="right") - 1
        if j >= 0 and t - ts[j] <= 6 * 3600:
            out[f"p_{k}"] = ps[j]
            out[f"pt_{k}"] = int(ts[j])
    return out


def cmd_prices() -> int:
    df = pd.read_parquet(OUT / "markets_all.parquet")
    sel = select(df)
    sel["S"] = anchor(sel)
    print(f"selected {len(sel)} ({(sel.src == 'aug').sum()} aug) / {sel.event_id.nunique()} events", flush=True)
    rows, rs = [], list(sel.itertuples())
    for i in range(0, len(rs), 1000):
        rows += sm.pmap(price_row, rs[i:i + 1000], workers=6)
        print(f"  {min(i + 1000, len(rs))}/{len(rs)}", flush=True)
    sel.merge(pd.DataFrame(rows), on="market_id", how="left").to_parquet(OUT / "priced.parquet", index=False)
    return 0


def cmd_panel() -> int:
    d = pd.read_parquet(OUT / "priced.parquet")
    d["closed"] = pd.to_datetime(d.closed_time, utc=True, errors="coerce", format="mixed")
    d["extended"] = d.closed > d.end + pd.Timedelta(days=2)
    rows = []
    for k, hh in H.items():
        g = d[d[f"p_{k}"].notna()].copy()
        g["h"], g["hours"] = k, hh
        g["p0"] = g[f"p_{k}"]
        g["price_ts"] = pd.to_datetime(g[f"pt_{k}"], unit="s", utc=True)
        g["t"] = g.S - pd.Timedelta(hours=hh)
        rows.append(g)
    p = pd.concat(rows, ignore_index=True)
    # point-in-time checks (traps): the price used must be at/before t; t must precede the close
    assert (p.price_ts <= p.t).all(), "price after decision time"
    assert ((p.closed.isna()) | (p.t < p.closed)).all(), "decision after close"
    ls0 = p.p0 <= 0.5
    p["ls"] = np.where(ls0, p.p0, 1 - p.p0)
    p["ls_won"] = np.where(ls0, p.y0, 1 - p.y0)
    p["fav"] = 1 - p.ls
    p["fav_won"] = 1 - p.ls_won
    p["lock_days"] = ((p.closed.fillna(p.S) - p.t) / pd.Timedelta(days=1)).clip(lower=1 / 24)
    keep = ["market_id", "event_id", "cat", "src", "question", "volume", "neg_risk", "n_mkts_event", "end", "S",
            "closed", "extended", "h", "hours", "t", "price_ts", "p0", "y0", "ls", "ls_won", "fav", "fav_won",
            "lock_days", "fees_enabled"]
    p[keep].to_parquet(OUT / "panel.parquet", index=False)
    print(p.groupby(["h"]).agg(n=("market_id", "size"), ev=("event_id", "nunique")).to_string())
    return 0


PB = [0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.501]
PL = ["<2c", "2-5c", "5-10c", "10-20c", "20-35c", "35-50c"]


def boot(g, stat, B=1000, seed=0):
    """Cluster bootstrap over events; stat(df)->float."""
    rng = np.random.default_rng(seed)
    evs = g.event_id.unique()
    idx = {e: ix for e, ix in g.groupby("event_id").indices.items()}
    vals = []
    for _ in range(B):
        pick = rng.choice(evs, len(evs), replace=True)
        ii = np.concatenate([idx[e] for e in pick])
        vals.append(stat(g.iloc[ii]))
    return np.nanpercentile(vals, [2.5, 97.5])


def cell(g, B=1000):
    gap = lambda x: (x.ls_won - x.ls).mean()  # noqa: E731  (<0 = longshots overpriced)
    fr = lambda x: (x.fav_won / x.fav - 1).mean()  # noqa: E731
    lo, hi = boot(g, gap, B)
    flo, fhi = boot(g, fr, B, seed=1)
    ret = fr(g)
    return {"n": int(len(g)), "events": int(g.event_id.nunique()), "ls_mean": round(g.ls.mean(), 4),
            "gap": round(gap(g), 4), "gap_ci": [round(lo, 4), round(hi, 4)],
            "fav_ret": round(ret, 4), "fav_ci": [round(flo, 4), round(fhi, 4)],
            "lock_days": round(g.lock_days.mean(), 2), "fav_ann": round(ret / (g.lock_days.mean() / 365), 3)}


def cmd_describe() -> int:
    p = pd.read_parquet(OUT / "panel.parquet")
    p = p[p.ls >= 0.005]  # drop effectively-settled 0/1 prints
    # p0 == 0.5 exactly = default display of an untraded book; the folded side is then arbitrary (token0 = "Yes",
    # which loses ~56%) -> fake longshot gap at 35-50c. Dropped (found 2026-11-06 on the first describe pass).
    p = p[p.p0 != 0.5]
    p["pb"] = pd.cut(p.ls, PB, labels=PL, right=False)
    out = {"by_h": {}, "by_h_pb": {}, "by_h_cat": {}, "by_h_pb_cat": {}}
    order = list(H)
    for k in order:
        out["by_h"][k] = cell(p[p.h == k])
    for (k, b), g in p.groupby(["h", "pb"], observed=True):
        if len(g) >= 30:
            out["by_h_pb"][f"{k}|{b}"] = cell(g, 500)
    for (k, c), g in p.groupby(["h", "cat"], observed=True):
        if len(g) >= 50:
            out["by_h_cat"][f"{k}|{c}"] = cell(g, 500)
    p["pb2"] = np.where(p.ls < 0.10, "ls<10c", np.where(p.ls < 0.30, "10-30c", "30-50c"))
    for (k, c, b), g in p.groupby(["h", "cat", "pb2"], observed=True):
        if len(g) >= 80:
            out["by_h_pb_cat"][f"{k}|{c}|{b}"] = cell(g, 300)
    # YES-side view (Yes/No markets): y - p for the YES token, by horizon x YES price
    yn = p[p.question.notna()].copy()
    yn["yb"] = pd.cut(yn.p0, [0, .05, .15, .35, .65, .85, .95, 1.0], right=False)
    out["yes_by_h_pb"] = {}
    for (k, b), g in yn.groupby(["h", "yb"], observed=True):
        if len(g) >= 50:
            gg = g.assign(d=g.y0 - g.p0)
            lo, hi = boot(gg, lambda x: x.d.mean(), 300)
            out["yes_by_h_pb"][f"{k}|{b}"] = {"n": int(len(g)), "y_minus_p": round(gg.d.mean(), 4),
                                              "ci": [round(lo, 4), round(hi, 4)]}
    (OUT / "tables.json").write_text(json.dumps(out, indent=1))
    for sec in ("by_h", "by_h_pb", "by_h_cat"):
        print("==", sec)
        for kk, v in out[sec].items():
            print(f"{kk:16s} n={v['n']:6d} ev={v['events']:5d} ls={v['ls_mean']:.3f} gap={v['gap']:+.4f} "
                  f"{v['gap_ci']} fav={v['fav_ret']:+.4f} {v['fav_ci']} lock={v['lock_days']}d ann={v['fav_ann']:+.3f}")
    return 0


# ----------------------------------------------------------------------------------------- trial frame (edgelab)
# Trials pm_t1_fav7d_maker / pm_t2_fav30d_maker were registered 2026-11-06 BEFORE anything below ran.
SEAL0, SEAL1 = pd.Timestamp("2026-05-01", tz="UTC"), pd.Timestamp("2026-11-01", tz="UTC")
EXCL = ("mentions", "culture_tech")
DATA_API = "https://data-api.polymarket.com/v2/trades"
sm.RATE["data-api.polymarket.com"] = 0.15
TH = {"7d": 168, "30d": 720}


def sealed_days():
    rng = random.Random(20261106)
    days = list(pd.date_range(SEAL0, SEAL1 - pd.Timedelta(days=1), freq="D"))
    return sorted(rng.sample(days, 46))


def meta_rows(evs):
    out = []
    for e in evs:
        for m in e.get("markets") or []:
            try:
                toks = json.loads(m.get("clobTokenIds") or "[]")
            except Exception:
                continue
            if len(toks) == 2:
                out.append({"market_id": m["id"], "cond": m.get("conditionId"), "token1": toks[1],
                            "closed_flag": bool(m.get("closed"))})
    return out


def cmd_sealed_enum() -> int:
    """Sealed universe: 46 sampled days, every binary market except mentions/culture_tech, whose startDate is at least
    7 d before S (ex-ante: known at listing). Outcome kept in the file but never printed here."""
    allrows, meta = [], []
    for evs in sm.pmap(sm.pm_day, sealed_days(), workers=3):
        allrows += sm.pm_flatten(evs, keep_outcome=True)
        meta += meta_rows(evs)
    df = pd.DataFrame(allrows).drop_duplicates("market_id").merge(
        pd.DataFrame(meta).drop_duplicates("market_id"), on="market_id", how="left")
    df["end"] = pd.to_datetime(df["end"], utc=True, errors="coerce", format="ISO8601")
    df["S"] = anchor(df)
    df = df[(df.S >= SEAL0) & (df.S < SEAL1) & ~df.cat.isin(EXCL) & df.y0.notna() & df.closed_flag]
    st = pd.to_datetime(df.start, utc=True, errors="coerce", format="ISO8601")
    df = df[(df.S - st) >= pd.Timedelta(days=7)]
    df.to_parquet(OUT / "sealed_markets.parquet", index=False)
    print("sealed markets", len(df), "events", df.event_id.nunique(), df.cat.value_counts().to_dict())
    return 0


def cmd_sealed_prices() -> int:
    sel = pd.read_parquet(OUT / "sealed_markets.parquet")
    # cost cap (ex-ante, outcome-blind): events with <= 40 legs (as dev), <= 10 random legs per event, <= 4,000
    # markets per category (events in random order) -- mirrors the dev pm_select caps
    rng = np.random.default_rng(1106)
    sel = sel[sel.n_mkts_event <= 40]
    keep = []
    for cat, g in sel.groupby("cat"):
        g = g.sample(frac=1, random_state=int(rng.integers(1e9)))
        g = g.groupby("event_id", sort=False).head(10)
        keep.append(g.head(4000))
    sel = pd.concat(keep)
    print("sealed priced sample", len(sel), sel.cat.value_counts().to_dict(), flush=True)
    rows, rs = [], list(sel.itertuples())
    for i in range(0, len(rs), 1000):
        rows += sm.pmap(price_row, rs[i:i + 1000], workers=6)
        print(f"  {min(i + 1000, len(rs))}/{len(rs)}", flush=True)
    sel.merge(pd.DataFrame(rows), on="market_id", how="left").to_parquet(OUT / "sealed_priced.parquet", index=False)
    return 0


def _frame_rows(d, window):
    d = d.copy()
    d["closed"] = pd.to_datetime(d.closed_time, utc=True, errors="coerce", format="mixed")
    rows = []
    for k, hh in TH.items():
        g = d[d[f"p_{k}"].notna()].copy()
        g["h"] = k
        g["t"] = g.S - pd.Timedelta(hours=hh)
        g["p0"] = g[f"p_{k}"]
        g["p0_time"] = pd.to_datetime(g[f"pt_{k}"], unit="s", utc=True)
        g["p_close"] = g["p_1h"] if "p_1h" in g else np.nan
        rows.append(g)
    f = pd.concat(rows, ignore_index=True)
    f["ticker"] = f.market_id.astype(str)
    f["closed_time"] = f.closed
    f["result"] = np.where(f.y0 == 1, "tok0", "tok1")
    f["date"] = f.S.dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    f["window"] = window
    return f[["ticker", "event_id", "cat", "question", "cond", "token0", "token1", "S", "h", "t", "p0", "p0_time",
              "p_close", "closed_time", "volume", "date", "window", "result"]]


def cmd_frame_pre() -> int:
    """Dev frame rows (from the descriptive priced sample) + sealed rows -> frame_pre.parquet (no tape yet)."""
    dev = pd.read_parquet(OUT / "priced.parquet")
    meta = []
    for evs in sm.pmap(sm.pm_day, dev_days(), workers=3):
        meta += meta_rows(evs)
    dev = dev.merge(pd.DataFrame(meta).drop_duplicates("market_id"), on="market_id", how="left")
    parts = [_frame_rows(dev, "dev")]
    if (OUT / "sealed_priced.parquet").exists():
        parts.append(_frame_rows(pd.read_parquet(OUT / "sealed_priced.parquet"), "sealed"))
    f = pd.concat(parts, ignore_index=True)
    f = f[f.closed_time.notna()]
    assert (f.p0_time <= f.t).all(), "p0 after decision"
    assert (f.t < f.closed_time).all(), "decision after close"
    f.to_parquet(OUT / "frame_pre.parquet", index=False)
    print(f.groupby(["window", "h"]).size().to_string())
    return 0


INCOMPLETE: list = []


def trades_after(r):
    t = int(r.t.timestamp())
    out, cursor = [], None
    for _ in range(120):
        params = {"condition": r.cond, "limit": 1000} if cursor is None else {"condition": r.cond, "cursor": cursor}
        d = sm.get(DATA_API, params, cache=True)
        if not d or not d.get("data"):
            if cursor is not None:
                INCOMPLETE.append(r.ticker)
            break
        for x in d["data"]:
            if int(x["timestamp"]) > t:
                tid = str(x.get("token_id") or x.get("asset"))
                out.append({"ticker": r.ticker, "timestamp": int(x["timestamp"]), "price": float(x["price"]),
                            "size": float(x["size"]), "tok0": tid == str(r.token0),
                            "token_ok": tid in (str(r.token0), str(r.token1))})
        oldest = min(int(x["timestamp"]) for x in d["data"])
        cursor = (d.get("pagination") or {}).get("next_cursor")
        if oldest <= t or not cursor:
            break
    else:
        INCOMPLETE.append(r.ticker)
    return out


def cmd_trades() -> int:
    """Tape for every rule candidate (superset: fav 0.64-0.96, cats except mentions/culture_tech), earliest t per
    market (30d row when it exists, so one pull covers both horizons). Outcome is not read here."""
    f = pd.read_parquet(OUT / "frame_pre.parquet")
    fav = np.maximum(f.p0, 1 - f.p0)
    c = f[fav.between(0.64, 0.96) & ~f.cat.isin(EXCL) & f.cond.notna() & (f.p0 != 0.5)]
    c = c.sort_values("t").drop_duplicates("ticker")
    print(f"tape markets: {len(c)} ({c.window.value_counts().to_dict()})", flush=True)
    rows, rs = [], list(c.itertuples())
    for i in range(0, len(rs), 300):
        for x in sm.pmap(trades_after, rs[i:i + 300], workers=4):
            rows += x
        print(f"  {min(i + 300, len(rs))}/{len(rs)} prints {len(rows)} incomplete {len(set(INCOMPLETE))}", flush=True)
    tr = pd.DataFrame(rows, columns=["ticker", "timestamp", "price", "size", "tok0", "token_ok"])
    print(f"prints {len(tr)}; token mismatch {int((~tr.token_ok).sum())}; truncated tapes {len(set(INCOMPLETE))}")
    assert tr.timestamp.between(1.6e9, 1.9e9).all(), "trade timestamps not epoch seconds"
    tr[tr.token_ok].drop(columns=["token_ok"]).to_parquet(OUT / "trades.parquet", index=False)
    pd.Series(sorted(set(INCOMPLETE)), name="ticker").to_frame().to_parquet(OUT / "tape_incomplete.parquet")
    return 0


def cmd_frame() -> int:
    f = pd.read_parquet(OUT / "frame_pre.parquet")
    f.drop(columns=["window"]).to_parquet(OUT / "frame.parquet", index=False)
    print("frame rows", len(f))
    return 0


if __name__ == "__main__":
    cmds = {"enum": cmd_enum, "prices": cmd_prices, "panel": cmd_panel, "describe": cmd_describe,
            "sealed_enum": cmd_sealed_enum, "sealed_prices": cmd_sealed_prices, "frame_pre": cmd_frame_pre,
            "trades": cmd_trades, "frame": cmd_frame}
    sys.exit(cmds[sys.argv[1]]())
