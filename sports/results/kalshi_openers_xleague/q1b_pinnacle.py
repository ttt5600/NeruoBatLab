"""Q1 reference B: logger (open_lines_<lg>) Pinnacle first sighting vs Kalshi open_time / first tight (15-min logger).
Games: picked games (picks file has open_t) + games in current state. Pre-existing-at-first-run games excluded."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

D = Path.home() / ".claude/quant-kalshi/data"
OUT = Path(__file__).resolve().parent
out = {}
for lg in ("nfl", "nhl"):
    st = json.load(open(D / f"open_lines_{lg}_state.json"))
    T = json.load(open(D / f"open_lines_{lg}_tight.json"))["markets"]
    first_run = pd.Timestamp(st["first_run_at"])
    pin = {}
    for l in open(D / f"open_lines_{lg}_picks.jsonl"):
        r = json.loads(l)
        if r.get("type") == "pick":
            pin[(r["key"], r["market"])] = (pd.Timestamp(r["open_t"]), pd.Timestamp(r["start"]))
    for k, g in st["games"].items():
        for mk, v in (g.get("pin") or {}).items():
            if v.get("open"):
                pin.setdefault((k, mk), (pd.Timestamp(v["open"]["t"]), pd.Timestamp(g["start"])))
    tk = {}
    for ev, m in T.items():
        tk[(m["date"], frozenset(m["teams"]), m["mk"])] = m
    rows = []
    for (k, mk), (pt, start) in pin.items():
        if pt <= first_run + pd.Timedelta(minutes=20):
            continue
        date, teams = k.split("|")
        a, h = teams.split("@")
        m = tk.get((date, frozenset([a, h]), mk))
        if not m:
            continue
        ot = pd.Timestamp(m["open_time"]) if m.get("open_time") else None
        ft = m.get("first_tight") or {}
        rows.append(dict(lg=lg, mk=mk, key=k, pin_first=pt, start=start, h_pin_before=(start - pt).total_seconds() / 3600,
                         kal_open=ot, h_kal_minus_pin=(ot - pt).total_seconds() / 3600 if ot is not None else np.nan,
                         censored=m.get("listing_censored"),
                         h_ft_minus_pin=(pd.Timestamp(ft["t"]) - pt).total_seconds() / 3600 if ft and not ft.get("censored") else np.nan,
                         ft_spread=ft.get("spread"), ft_usd_1c=(ft.get("depth_yes_ask") or {}).get("usd_1c"),
                         ft_noask_usd_1c=(ft.get("depth_no_ask") or {}).get("usd_1c")))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / f"q1b_{lg}_rows.csv", index=False)
    for mk, x in d.groupby("mk"):
        x = x[x.censored == False]  # noqa: E712
        s = dict(n=len(x), h_pin_before_med=x.h_pin_before.median(), h_kal_minus_pin_med=x.h_kal_minus_pin.median(),
                 h_kal_minus_pin_p10=x.h_kal_minus_pin.quantile(.1), h_kal_minus_pin_p90=x.h_kal_minus_pin.quantile(.9),
                 share_kal_first=(x.h_kal_minus_pin < 0).mean(), n_ft=int(x.h_ft_minus_pin.notna().sum()),
                 h_ft_minus_pin_med=x.h_ft_minus_pin.median(), share_ft_by_pin=(x.h_ft_minus_pin <= 0.25).mean(),
                 ft_usd_1c_med=x.ft_usd_1c.median(), ft_noask_usd_1c_med=x.ft_noask_usd_1c.median())
        out[f"{lg}_{mk}"] = {k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in s.items()}
        print(lg, mk, out[f"{lg}_{mk}"])
json.dump(out, open(OUT / "q1b_summary.json", "w"), indent=1)
