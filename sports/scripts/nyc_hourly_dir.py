"""nyc_hourly_dir: KXTEMPNYCHS nowcast vs market at T = H - 20 min (prereg notes/nyc_hourly_dir.md).

  python scripts/nyc_hourly_dir.py index      # station 1-min -> reconstructed index R(t) (results/nyc_hourly_dir/R.parquet)
  python scripts/nyc_hourly_dir.py model      # fit beta + residuals on 08-20..09-09 (station data only)
  python scripts/nyc_hourly_dir.py candles    # dev-window event candlesticks + tapes (read-only GETs)
  python scripts/nyc_hourly_dir.py frame      # inputs-only frame -> data/nyc_hourly_dir/frame.parquet
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "nyc_hourly_dir"
DATA = ROOT / "data" / "nyc_hourly_dir"
OFF = {"LGA": 1.0, "EWR": 1.0, "TEB": 1.0, "CDW": 0.0, "HPN": -1.0, "FRG": 0.0, "SMQ": -1.0, "ISP": 0.0}
BC = 0.125
LEAD_PUB = 6      # minutes: index public with <= 5 min lag
T_BEFORE = 20     # decision T = H - 20 min
SLOPE_WIN = 30
FIT = ("2026-08-20", "2026-09-10")   # [start, end) UTC, station data only
DEV = ("2026-09-10", "2026-09-30")   # event local dates inclusive
RECON_SD = 0.3


def index():
    zs = {}
    for st, b in OFF.items():
        d = pd.read_csv(OUT / "raw" / f"{st}.csv")
        d = d[pd.to_numeric(d["tmpf"], errors="coerce").notna()]
        t = pd.to_datetime(d["valid(UTC)"], utc=True)
        c = (d["tmpf"].astype(float).to_numpy() - 32) / 1.8
        s = pd.Series(c - b + BC, index=t).groupby(level=0).last()
        zs[st] = s
    idx = pd.date_range("2026-08-19", "2026-10-01 06:00", freq="1min", tz="UTC")
    Z = pd.DataFrame({k: v.reindex(idx).ffill(limit=10) for k, v in zs.items()})
    n = Z.notna().sum(1)
    R = Z.mean(1) * 1.8 + 32
    R[n < 5] = np.nan
    df = pd.DataFrame({"R": R, "n": n})
    df.to_parquet(OUT / "R.parquet")
    print(df.describe(), "\nshare minutes with R:", df.R.notna().mean())


def _bin(local_hour):
    h = np.asarray(local_hour)
    return np.where((h >= 8) & (h <= 12), "morning", np.where((h >= 13) & (h <= 18), "afternoon", "night"))


def nowcast_inputs(R: pd.Series, H: pd.DatetimeIndex):
    t0 = H - pd.Timedelta(minutes=T_BEFORE + LEAD_PUB)
    r0 = R.reindex(t0).to_numpy()
    rb = R.reindex(t0 - pd.Timedelta(minutes=SLOPE_WIN)).to_numpy()
    L = T_BEFORE + LEAD_PUB
    slope = (r0 - rb) / SLOPE_WIN
    return t0, r0, slope, L


def model():
    R = pd.read_parquet(OUT / "R.parquet")["R"]
    H = pd.date_range(FIT[0], FIT[1], freq="1h", tz="UTC", inclusive="left")
    t0, r0, slope, L = nowcast_inputs(R, H)
    y = R.reindex(H).to_numpy()
    lh = H.tz_convert("America/New_York").hour
    df = pd.DataFrame({"H": H, "r0": r0, "sl": slope * L, "y": y, "bin": _bin(lh)}).dropna()
    x, yy = df.sl.to_numpy(), (df.y - df.r0).to_numpy()
    beta = float((x * yy).sum() / (x * x).sum())
    df["e"] = yy - beta * x
    out = {"beta": beta, "n": len(df), "L": L, "resid": {b: sorted(g.e.round(4).tolist()) for b, g in df.groupby("bin")}}
    for b, g in df.groupby("bin"):
        print(b, len(g), "sd", round(g.e.std(), 3), "q01/q99", np.round(g.e.quantile([.01, .99]).to_numpy(), 2))
    print("beta", round(beta, 3), "n", len(df), "persist-only sd", round(yy.std(), 3))
    (OUT / "model.json").write_text(json.dumps(out))


def p_yes(xhat, K, bins, mdl, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    res = {b: np.asarray(v) for b, v in mdl["resid"].items()}
    noise = {b: (np.repeat(v, 20) + rng.normal(0, RECON_SD, len(v) * 20)) for b, v in res.items()}
    out = np.full(len(xhat), np.nan)
    for i, (x, k, b) in enumerate(zip(xhat, K, bins)):
        if np.isfinite(x):
            out[i] = (x + noise[b] > k).mean()
    return out


def candles():
    from kx_throttle import kget
    m = pd.read_parquet(ROOT / "data" / "kalshi_hourly_temp" / "markets.parquet")
    m = m[(m.series == "KXTEMPNYCHS") & (m.date >= DEV[0]) & (m.date <= DEV[1])]
    evs = m.groupby("event_ticker").agg(open_ts=("open_ts", "min"), H=("H", "max")).reset_index()
    rows = []
    for i, r in enumerate(evs.itertuples()):
        d = None
        for path in (f"/series/KXTEMPNYCHS/events/{r.event_ticker}/candlesticks",):
            d = kget(path, {"start_ts": int(r.open_ts) - 60, "end_ts": int(r.H), "period_interval": 1})
        if not d:
            continue
        for tk, cs in zip(d.get("market_tickers", []), d.get("market_candlesticks", [])):
            for c in cs or []:
                yb, ya = c.get("yes_bid", {}), c.get("yes_ask", {})
                rows.append({"ticker": tk, "end_ts": c["end_period_ts"],
                             "yb": yb.get("close_dollars", yb.get("close")), "ya": ya.get("close_dollars", ya.get("close")),
                             "vol": c.get("volume_fp", c.get("volume"))})
        if i % 50 == 0:
            print("candles", i, len(evs), len(rows), flush=True)
    c = pd.DataFrame(rows)
    c.to_parquet(OUT / "candles.parquet")
    print(len(c), c.head())


def tapes(tickers, t0s, t1s):
    from kx_throttle import kget
    rows = []
    for tk, a, b in zip(tickers, t0s, t1s):
        for path in ("/historical/trades", "/markets/trades"):
            cur, got = None, []
            while True:
                p = {"ticker": tk, "min_ts": int(a), "max_ts": int(b), "limit": 1000}
                if cur:
                    p["cursor"] = cur
                d = kget(path, p)
                if not d:
                    break
                got += d.get("trades", [])
                cur = d.get("cursor")
                if not cur or not d.get("trades"):
                    break
            if got:
                break
        for x in got:
            rows.append({"ticker": tk, "ts": int(pd.Timestamp(x["created_time"]).timestamp()),
                         "yes_price": float(x.get("yes_price_dollars", x.get("yes_price", np.nan))),
                         "no_price": float(x.get("no_price_dollars", x.get("no_price", np.nan))),
                         "count": float(x.get("count_fp", x.get("count", 0)))})
    return pd.DataFrame(rows, columns=["ticker", "ts", "yes_price", "no_price", "count"])


def frame():
    mdl = json.loads((OUT / "model.json").read_text())
    m = pd.read_parquet(ROOT / "data" / "kalshi_hourly_temp" / "markets.parquet")
    m = m[(m.series == "KXTEMPNYCHS") & (m.date >= DEV[0]) & (m.date <= DEV[1])].copy()
    R = pd.read_parquet(OUT / "R.parquet")["R"]
    H = pd.to_datetime(m["H"].to_numpy("int64"), unit="s", utc=True)
    t0, r0, slope, L = nowcast_inputs(R, H)
    m["T"] = m["H"].astype("int64") - T_BEFORE * 60
    m["t0"] = (t0 - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
    assert (m["t0"] < m["T"]).all() and (m["T"] < m["H"]).all()
    m["xhat"] = r0 + mdl["beta"] * slope * L
    m["r0"] = r0
    m["bin"] = _bin(H.tz_convert("America/New_York").hour)
    m["p_yes"] = p_yes(m.xhat.to_numpy(), m.K.to_numpy(float), m["bin"].to_numpy(), mdl)
    c = pd.read_parquet(OUT / "candles.parquet").sort_values("end_ts")
    c["yb"], c["ya"] = pd.to_numeric(c.yb), pd.to_numeric(c.ya)
    gs = dict(tuple(c.groupby("ticker")))
    q = []
    for tk, T, Hh in zip(m.ticker, m["T"], m["H"]):
        g = gs.get(tk, c.iloc[:0])
        a = g[g.end_ts <= T].tail(1)
        b = g[g.end_ts <= Hh - 300].tail(1)
        q.append((a.yb.iloc[0] if len(a) else np.nan, a.ya.iloc[0] if len(a) else np.nan,
                  b.yb.iloc[0] if len(b) else np.nan, b.ya.iloc[0] if len(b) else np.nan))
    m[["yes_bid", "yes_ask", "mark_yes_bid", "mark_yes_ask"]] = np.array(q, float)
    DATA.mkdir(parents=True, exist_ok=True)
    keep = ["ticker", "series", "event_ticker", "strike_type", "K", "open_ts", "H", "T", "t0", "date", "r0", "xhat", "bin",
            "p_yes", "yes_bid", "yes_ask", "mark_yes_bid", "mark_yes_ask", "result", "expiration_value", "volume", "settle_ts"]
    m[keep].to_parquet(DATA / "frame.parquet")
    print(len(m), "markets", m.event_ticker.nunique(), "events; p_yes finite", np.isfinite(m.p_yes).mean().round(3),
          "quote finite", np.isfinite(m.yes_ask).mean().round(3))


def tape():
    sys.path.insert(0, str(ROOT))
    from edgelab.rules.kalshi_nyc_dir import select
    f = pd.read_parquet(DATA / "frame.parquet")
    s = select(f, maker=True)
    print("maker-selected", len(s))
    t = tapes(s.ticker, s["T"], s["H"])
    t.to_parquet(DATA / "tape.parquet")
    print(len(t), "prints")


if __name__ == "__main__":
    {"index": index, "model": model, "candles": candles, "frame": frame, "tape": tape}[sys.argv[1]]()
