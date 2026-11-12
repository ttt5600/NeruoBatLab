"""Summarize tonight_logger ticks (forward rows, descriptive): per league x market, events/games FIRST seen after the
logger's first tick (uncensored). Kalshi: first-sighting anchor spread, implied, hours before kick; Pinnacle: first
sighting line. Joined Kalshi<->Pinnacle by kickoff (Kalshi occurrence - 3 h, +-20 min... checked per league below) and
the Pinnacle line being within 4 pts (totals) of the Kalshi implied at Pinnacle's first sighting.
Run after the logger exits (08:00Z): q3_tonight.py -> q3_tonight_rows.csv, q3_tonight_summary.json"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
d = pd.read_json(OUT / "ticks.jsonl", lines=True)
d["t"] = pd.to_datetime(d.t, utc=True)
t0 = d.t.min()
k = d[d.src.isna()].copy()
p = d[d.src == "pin"].copy()
res = {"ticks": int(d.t.nunique()), "first_tick": str(t0), "last_tick": str(d.t.max())}
kf = k.sort_values("t").groupby("event").first().reset_index()
kf["censored"] = kf.t <= t0
kf["kick_est"] = pd.to_datetime(kf.occ, utc=True) - pd.Timedelta(hours=3)
kf["h_before_kick_est"] = (kf.kick_est - kf.t).dt.total_seconds() / 3600
kf["first_spread"] = kf.ask - kf.bid
# first tight per event
k["spread"] = k.ask - k.bid
ft = k[k.spread <= 0.04].sort_values("t").groupby("event").first()[["t", "implied"]].rename(columns={"t": "ft_t", "implied": "ft_implied"})
kf = kf.merge(ft, left_on="event", right_index=True, how="left")
pf = p.sort_values("t").groupby(["lg", "pin_id", "mk"]).first().reset_index()
pf["censored"] = pf.t <= t0
pf["h_before_kick"] = (pd.to_datetime(pf.start, utc=True) - pf.t).dt.total_seconds() / 3600
kf.to_csv(OUT / "q3_tonight_kalshi_rows.csv", index=False)
pf.to_csv(OUT / "q3_tonight_pin_rows.csv", index=False)
for (lg, mk), x in kf.groupby(["lg", "mk"]):
    n = x[~x.censored]
    res[f"kal_{lg}_{mk}"] = dict(events=len(x), new=len(n), new_h_before_kick_med=float(n.h_before_kick_est.median()) if len(n) else None,
                                 new_first_spread_med=float(n.first_spread.median()) if len(n) else None,
                                 new_tight=int(n.ft_t.notna().sum()))
for (lg, mk), x in pf.groupby(["lg", "mk"]):
    n = x[~x.censored]
    res[f"pin_{lg}_{mk}"] = dict(games=len(x), new=len(n), new_h_before_kick_med=float(n.h_before_kick.median()) if len(n) else None)
print(json.dumps(res, indent=1))
json.dump(res, open(OUT / "q3_tonight_summary.json", "w"), indent=1)
