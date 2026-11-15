"""rules_quirks addendum C: cricket NR pricing (read-only). candles | score
Start time parsed from the event ticker (e.g. KXT20MATCH-26SEP151330SRIENG = 2026-09-15 13:30 ET) and localised as
America/New_York (the earlier UTC reading when the rules say 'EST' during DST -> T_ref cannot fall after the start).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from kx_throttle import kget

D = Path(__file__).resolve().parents[1] / "results" / "rules_quirks"
SER = ["KXODIMATCH", "KXT20MATCH", "KXWT20MATCH", "KXIPLGAME", "KXCRICKETT20IMATCH"]
C2 = ["KXT20MATCH", "KXODIMATCH"]
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
MON = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def start(ev: str):
    x = re.match(r"^[A-Z0-9]+-(\d{2})([A-Z]{3})(\d{2})(\d{4})", ev)
    if not x:
        return pd.NaT
    y, mo, d, hm = x.groups()
    t = pd.Timestamp(year=2000 + int(y), month=MON[mo], day=int(d), hour=int(hm[:2]), minute=int(hm[2:]))
    return t.tz_localize("America/New_York", ambiguous=True, nonexistent="shift_forward").tz_convert("UTC")


def frame() -> pd.DataFrame:
    m = pd.read_parquet(D / "cricket_markets.parquet")
    m = m[m.series.isin(SER)].copy()
    m["start"] = m.event_ticker.map(start)
    m = m[m.start.notna()]
    n = m.groupby("event_ticker").ticker.transform("size")
    m = m[n == 2]
    m["split"] = np.where(m.start < pd.Timestamp("2026-07-01", tz="UTC"), "dev", "holdout")
    return m


def secs(t) -> int:
    return int((t - EPOCH) // pd.Timedelta(seconds=1))


def cmd_candles() -> int:
    m = frame()
    m = m[m.split == "dev"]
    print("dev markets", len(m), "events", m.event_ticker.nunique(), flush=True)
    f = D / "cricket_candles.json"
    out = json.loads(f.read_text()) if f.exists() else {}
    for i, r in enumerate(m.itertuples()):
        if r.ticker in out:
            continue
        tref = secs(r.start) - 3600
        p = {"start_ts": tref - 24 * 3600, "end_ts": tref, "period_interval": 60}
        cs = []
        for path in (f"/historical/markets/{r.ticker}/candlesticks", f"/series/{r.series}/markets/{r.ticker}/candlesticks"):
            cs = (kget(path, p) or {}).get("candlesticks", [])
            if cs:
                break
        out[r.ticker] = cs
        if i % 200 == 0:
            f.write_text(json.dumps(out))
            print("candles", i, flush=True)
    f.write_text(json.dumps(out))
    return 0


def px(c, k):
    d = c.get(k) or {}
    v = d.get("close_dollars", d.get("close"))
    if v is None:
        return np.nan
    v = float(v)
    return v / 100 if v > 1.0 else v


def fee(p):
    return np.ceil(0.07 * p * (1 - p) * 100 - 1e-9) / 100


def cmd_score() -> int:
    m = frame()
    m = m[m.split == "dev"].copy()
    cs = json.loads((D / "cricket_candles.json").read_text())
    bid, ask = [], []
    for r in m.itertuples():
        tref = secs(r.start) - 3600
        c = [x for x in cs.get(r.ticker, []) if int(x["end_period_ts"]) <= tref]
        x = max(c, key=lambda z: int(z["end_period_ts"])) if c else {}
        bid.append(px(x, "yes_bid")); ask.append(px(x, "yes_ask"))
    m["bid"], m["ask"] = bid, ask
    m.loc[(m.ask <= 0) | (m.ask >= 1), "ask"] = np.nan
    m["mid"] = (m.bid.fillna(0) + m.ask) / 2
    m["sv"] = pd.to_numeric(m.sv, errors="coerce")
    m["pay"] = np.where(m.result == "yes", 1.0, np.where(m.result == "no", 0.0, m.sv))
    m["nr"] = (m.result == "scalar") | m.sv.between(0.4, 0.6)
    m.to_parquet(D / "cricket_frame.parquet", index=False)
    q = m[m.ask.notna()]
    ok = q.groupby("event_ticker").ticker.transform("size") == 2
    q = q[ok]
    print(f"dev events with both quotes at T_ref: {q.event_ticker.nunique()} (of {m.event_ticker.nunique()}); NR share {q.groupby('event_ticker').nr.first().mean():.3f}")
    fav = q.loc[q.groupby("event_ticker").mid.idxmax()]
    dog = q.loc[q.groupby("event_ticker").mid.idxmin()]
    print("== C0 favourite calibration (payout incl. NR=0.5)")
    fav = fav.assign(bin=pd.cut(fav.mid, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
    print(fav.groupby("bin", observed=True).agg(n=("pay", "size"), mid=("mid", "mean"), pay=("pay", "mean"), nr=("nr", "mean"),
                                                win_ex_nr=("result", lambda s: (s == "yes").sum())).to_string())
    for name, sers in (("C1", SER), ("C2", C2)):
        t = dog[dog.ask.between(0.10, 0.45) & dog.series.isin(sers)].copy()
        t["cost"] = t.ask + fee(t.ask)
        t["pnl"] = t.pay - t.cost
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(t), size=(2000, len(t)))
        roi_b = t.pnl.to_numpy()[idx].sum(1) / t.cost.to_numpy()[idx].sum(1)
        roi = t.pnl.sum() / t.cost.sum()
        print(f"{name}: n {len(t)} wins {(t.result=='yes').sum()} NR {t.nr.sum()} ROI {roi:+.4f} "
              f"[{np.percentile(roi_b,2.5):+.4f},{np.percentile(roi_b,97.5):+.4f}] p(boot<=0) {(roi_b<=0).mean():.4f}")
        print("   NR contribution to pnl:", round(float(t.loc[t.nr, "pnl"].sum()), 2), "of total", round(float(t.pnl.sum()), 2))
        print(t.groupby("series").agg(n=("pnl", "size"), pnl=("pnl", "sum"), cost=("cost", "sum")).assign(roi=lambda x: x.pnl / x.cost).to_string())
        t.to_csv(D / f"cricket_{name}_dev_bets.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit({"candles": cmd_candles, "score": cmd_score}[sys.argv[1]]())
