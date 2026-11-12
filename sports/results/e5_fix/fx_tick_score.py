"""FX-native validation of F1/F2/F3 on timestamped IBKR ticks (events 11-05..11-08, 11-10..11-11).
Fill = first print strictly below our bid (audit rule) at time f; F1 keeps it iff f < local cutoff on D;
F2 keeps it iff f < obs trigger (above-side: running max >= X+1-x, +15 min latency; below-side: noon unless cleared).
Outcome: settlement for 11-05..11-08 (fx_picks_d1.csv), METAR max for 11-10/11 (agrees 693/693 historically)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fx_f2 import TZ

D = Path(__file__).parent
FEE = 0.01
a = pd.read_csv(D / "fx_tick_timing_old.csv")
b = pd.read_csv(D / "fx_tick_timing.csv")
t = pd.concat([pd.read_parquet(D / "fx_ticks_old.parquet"), pd.read_parquet(D / "fx_ticks_recent.parquet")])
t = t[t.status == "ok"]
p = pd.concat([pd.read_csv(D / "ib_in_old.csv"), pd.read_csv(D / "ib_in_recent.csv").pipe(
    lambda x: x[x.event_date >= "2026-11-10"])]).drop_duplicates(["ticker", "side"])
p["code"] = p.ticker.str[2:5]
p["X"] = p.ticker.str.split("_").str[-1].astype(float)
won_metar = pd.concat([a, b]).drop_duplicates(["ticker", "side"]).set_index(["ticker", "side"]).won_by_metar
o = pd.read_parquet(D / "asos_obs.parquet")
o["stn"] = o.stn.astype(str).str[-3:]
rows = []
for r in p.itertuples():
    x = t[(t.ticker == r.ticker) & (t.side == r.side)].sort_values("ts")
    thr = x[x.price < r.side_bid - 1e-9]
    z = TZ[r.code]
    mid = pd.Timestamp(r.event_date, tz=z)
    won = bool(r.won) if isinstance(r.won, (bool, np.bool_)) or r.won in (0, 1, True, False) else None
    if won is None or pd.isna(r.won):
        won = bool(won_metar.get((r.ticker, r.side), np.nan))
    day = o[(o.stn == r.code) & (o.valid >= mid) & (o.valid < mid + pd.Timedelta(days=1))].sort_values("valid")
    avail, rmax = day.valid + pd.Timedelta(minutes=15), day.tmpf.cummax()
    trig = {}
    for xx in (2, 4, 6):
        if r.side == "NO":
            h = avail[rmax >= r.X + 1 - xx]
            trig[xx] = h.iloc[0] if len(h) else pd.Timestamp.max.tz_localize("UTC")
        else:
            h = avail[rmax >= r.X + 1]
            noon = mid + pd.Timedelta(hours=12)
            trig[xx] = pd.Timestamp.max.tz_localize("UTC") if (len(h) and h.iloc[0] < noon) else noon
    f = thr.ts.iloc[0] if len(thr) else None
    rows.append({"ticker": r.ticker, "side": r.side, "bid": r.side_bid, "won": won, "fill_ts": f,
                 "fill_h": (f - mid) / pd.Timedelta(hours=1) if f is not None else np.nan,
                 **{f"trig{xx}": trig[xx] for xx in trig}, "event_date": r.event_date})
d = pd.DataFrame(rows)
d.to_csv(D / "fx_tick_score_rows.csv", index=False)
q = d.bid.to_numpy()
pnl = np.where(d.won, 1 / (q + FEE) - 1, -1.0)
filled = d.fill_ts.notna().to_numpy()
out = {"n_traded_picks": len(d), "event_days": sorted(d.event_date.unique().tolist()), "variants": []}


def add(name, keep):
    k = filled & keep
    out["variants"].append({"variant": name, "fills": int(k.sum()), "losses": int((k & ~d.won.to_numpy()).sum()),
                            "roi": round(float(pnl[k].mean()), 4) if k.any() else None,
                            "units": round(float(pnl[k].sum()), 2)})


add("B0 rest to close", np.ones(len(d), bool))
for h in (0, 6, 9, 12, 15):
    add(f"F1 cancel {h:02d}:00", (d.fill_h < h).to_numpy())
for xx in (2, 4, 6):
    add(f"F2 X={xx}", np.array([f is not None and f < tr for f, tr in zip(d.fill_ts, d[f"trig{xx}"])]))
out["loser_fill_hours"] = sorted(round(float(h), 2) for h in d[filled & ~d.won.to_numpy()].fill_h)
out["winner_fill_hours"] = sorted(round(float(h), 2) for h in d[filled & d.won.to_numpy()].fill_h)
json.dump(out, open(D / "fx_tick_score.json", "w"), indent=1, default=str)
print(pd.DataFrame(out["variants"]).to_string(index=False))
print("losers h:", out["loser_fill_hours"])
print("winners h:", out["winner_fill_hours"])
print(len(d), out["event_days"])
