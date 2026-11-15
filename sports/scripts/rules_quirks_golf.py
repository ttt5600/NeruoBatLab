"""rules_quirks golf top-N 'including ties' study (read-only Kalshi GETs via kx_throttle).

    ../.venv_quant/bin/python scripts/rules_quirks_golf.py markets   # settled markets -> results/rules_quirks/golf_markets.parquet
    ../.venv_quant/bin/python scripts/rules_quirks_golf.py candles   # hourly candles around first-round start
    ../.venv_quant/bin/python scripts/rules_quirks_golf.py score     # prereg addendum G (notes/rules_quirks.md)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from kx_throttle import kget

D = Path(__file__).resolve().parents[1] / "results" / "rules_quirks"
D.mkdir(parents=True, exist_ok=True)
SERIES = {"KXPGATOP5": 5, "KXPGATOP10": 10, "KXPGATOP20": 20}
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def secs(ts) -> int:
    return int((pd.Timestamp(ts) - EPOCH) // pd.Timedelta(seconds=1))


def cmd_markets() -> int:
    rows = []
    for s in SERIES:
        for path, extra in (("/historical/markets", {}), ("/markets", {"status": "settled"})):
            cur = None
            while True:
                p = {"series_ticker": s, "limit": 1000, **extra}
                if cur:
                    p["cursor"] = cur
                d = kget(path, p) or {}
                for m in d.get("markets", []):
                    m["series"] = s
                    m["src"] = path
                    rows.append(m)
                cur = d.get("cursor")
                if not cur or not d.get("markets"):
                    break
        print(s, len(rows), flush=True)
    m = pd.DataFrame(rows).drop_duplicates("ticker")
    keep = [c for c in ["ticker", "event_ticker", "series", "src", "result", "open_time", "close_time",
                        "expected_expiration_time", "yes_sub_title", "title", "volume_fp", "volume",
                        "settlement_value_dollars", "settlement_value", "rules_primary", "rules_secondary"] if c in m]
    m = m[keep]
    m.to_parquet(D / "golf_markets.parquet", index=False)
    print(m.groupby("series").agg(events=("event_ticker", "nunique"), mk=("ticker", "size")))
    print(m.result.value_counts())
    return 0


def first_round_day(g: pd.DataFrame) -> pd.Timestamp:
    """Thursday on or before the LAST market close (= tournament end, Sun/Mon ET). expected_expiration_time is ~2 weeks
    after the end, and min(close_time) is the Friday cut (missed-cut players settle early) -- both wrong anchors."""
    end = pd.to_datetime(g.close_time, utc=True, format="ISO8601").max().tz_convert("America/New_York")
    d = end.normalize().tz_localize(None)
    while d.dayofweek != 3:
        d -= pd.Timedelta(days=1)
    return d.tz_localize("UTC")


def cmd_candles() -> int:
    m = pd.read_parquet(D / "golf_markets.parquet")
    m = m[m.result.isin(["yes", "no"])]
    m = m[m.groupby("event_ticker").ticker.transform("size") >= 30]
    cache_f = D / "golf_candles.json"
    out = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    ev_t = {ev: first_round_day(g) for ev, g in m.groupby("event_ticker")}
    # rate limit (~0.3 req/s shared): DEV events only, TOP10 first, then TOP20, TOP5
    m = m[m.event_ticker.map(ev_t) < pd.Timestamp("2026-07-01", tz="UTC")].copy()
    m["_o"] = m["series"].map({"KXPGATOP10": 0, "KXPGATOP20": 1, "KXPGATOP5": 2})
    m = m.sort_values(["_o", "event_ticker", "ticker"])
    print("dev markets to fetch", len(m), flush=True)
    for i, r in enumerate(m.itertuples()):
        if r.ticker in out:
            continue
        t0 = ev_t[r.event_ticker] + pd.Timedelta(hours=11)
        p = {"start_ts": secs(t0) - 36 * 3600, "end_ts": secs(t0) + 3600, "period_interval": 60}
        cs = []
        paths = ([f"/historical/markets/{r.ticker}/candlesticks", f"/series/{r.series}/markets/{r.ticker}/candlesticks"]
                 if r.src == "/historical/markets" else
                 [f"/series/{r.series}/markets/{r.ticker}/candlesticks", f"/historical/markets/{r.ticker}/candlesticks"])
        for path in paths:
            cs = (kget(path, p) or {}).get("candlesticks", [])
            if cs:
                break
        out[r.ticker] = cs
        if i % 100 == 0:
            cache_f.write_text(json.dumps(out))
            print("candles", i, len(m), flush=True)
    cache_f.write_text(json.dumps(out))
    print("done", len(out))
    return 0


def px(c, k, f="close"):
    d = c.get(k) or {}
    v = d.get(f + "_dollars", d.get(f))
    if v is None:
        return np.nan
    v = float(v)
    return v / 100 if v > 1.0 else v   # historical endpoint may return cents


def fee(p):
    return np.ceil(0.07 * p * (1 - p) * 100 - 1e-9) / 100


def boot(df, col_num, col_den, B=2000, seed=0):
    g = df.groupby("event_ticker")[[col_num, col_den]].sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(B, len(g)))
    num, den = g[col_num].to_numpy(), g[col_den].to_numpy()
    r = num[idx].sum(1) / den[idx].sum(1)
    return num.sum() / den.sum(), np.percentile(r, [2.5, 97.5]), float((r <= 0).mean())


def cmd_frame() -> int:
    """Inputs + hidden result for the edgelab loader (no outcome summary printed)."""
    m = build_frame()
    m["date"] = m.R1.dt.tz_localize(None)
    m = m[["ticker", "event_ticker", "series", "N", "R1", "date", "bid", "ask", "mid", "result"]]
    m.to_parquet(D / "golf_edgelab_frame.parquet", index=False)
    print("rows", len(m), "with ask", int(m.ask.notna().sum()))
    return 0


def cmd_score() -> int:
    m = build_frame()
    _score(m)
    return 0


def build_frame() -> pd.DataFrame:
    m = pd.read_parquet(D / "golf_markets.parquet")
    m = m[m.result.isin(["yes", "no"])].copy()
    m = m[m.groupby("event_ticker").ticker.transform("size") >= 30]
    cs = json.loads((D / "golf_candles.json").read_text())
    ev_t = {ev: first_round_day(g) for ev, g in m.groupby("event_ticker")}
    m["R1"] = m.event_ticker.map(ev_t)
    m["N"] = m["series"].map(SERIES)
    m["y"] = (m.result == "yes").astype(int)
    bid, ask = [], []
    for r in m.itertuples():
        tref = secs(r.R1 + pd.Timedelta(hours=10))
        c = [x for x in cs.get(r.ticker, []) if int(x["end_period_ts"]) <= tref]
        if c:
            x = max(c, key=lambda z: int(z["end_period_ts"]))
            bid.append(px(x, "yes_bid")); ask.append(px(x, "yes_ask"))
        else:
            bid.append(np.nan); ask.append(np.nan)
    m["bid"], m["ask"] = bid, ask
    m.loc[(m.ask <= 0) | (m.ask >= 1), "ask"] = np.nan
    m["bid"] = m.bid.fillna(0.0)
    m["mid"] = (m.bid + m.ask) / 2
    m["split"] = np.where(m.R1 < pd.Timestamp("2026-07-01", tz="UTC"), "dev", "holdout")
    m.to_parquet(D / "golf_frame.parquet", index=False)
    return m


def _score(m: pd.DataFrame) -> None:
    # G-a definition gap (all settled events, both splits: settlement counts are not a price outcome)
    ev = m.groupby(["event_ticker", "series"]).agg(N=("N", "first"), R=("y", "sum"), n=("y", "size"),
                                                    q=("ask", lambda s: s.notna().mean()), S=("mid", "sum"),
                                                    split=("split", "first"), R1=("R1", "first")).reset_index()
    ev.to_csv(D / "golf_events.csv", index=False)
    print("== G-a: realized YES count per event (R) vs N")
    for s, g in ev.groupby("series"):
        rr = g.R / g.N
        bs = np.random.default_rng(1).choice(rr.to_numpy(), size=(2000, len(rr))).mean(1)
        print(f"{s}: events {len(g)}  listed mk/event med {g.n.median():.0f}  mean R {g.R.mean():.2f}  R/N {rr.mean():.3f} "
              f"[{np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f}]  share events R>N {(g.R>g.N).mean():.2f}")
    print("== G-b (DEV only): sum of mids at T_ref vs R, events with >=90% quoted")
    for s, g in ev[(ev.split == "dev") & (ev.q >= 0.9)].groupby("series"):
        d = (g.S - g.R).to_numpy()
        bs = np.random.default_rng(2).choice(d, size=(2000, len(d))).mean(1)
        print(f"{s}: events {len(g)}  N {g.N.iloc[0]}  mean S_mid {g.S.mean():.2f}  mean R {g.R.mean():.2f}  "
              f"S-R {d.mean():+.2f} [{np.percentile(bs,2.5):+.2f},{np.percentile(bs,97.5):+.2f}]")
    print("== trades (DEV only)")
    for name, lo, hi in (("GT1", 0.03, 0.95), ("GT2", 0.10, 0.60)):
        t = m[(m.split == "dev") & m.ask.between(lo, hi)].copy()
        t["cost"] = t.ask + fee(t.ask)
        t["pnl"] = t.y - t.cost
        roi, ci, p = boot(t, "pnl", "cost")
        print(f"{name}: n {len(t)} events {t.event_ticker.nunique()} wins {t.y.sum()} ROI {roi:+.4f} "
              f"[{ci[0]:+.4f},{ci[1]:+.4f}] p(boot<=0) {p:.4f}")
        for s, g in t.groupby("series"):
            r2, c2, _ = boot(g, "pnl", "cost")
            print(f"   {s}: n {len(g)} ROI {r2:+.4f} [{c2[0]:+.4f},{c2[1]:+.4f}]")
        t.to_csv(D / f"golf_{name}_dev_bets.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit({"markets": cmd_markets, "candles": cmd_candles, "frame": cmd_frame, "score": cmd_score}[sys.argv[1]]())
