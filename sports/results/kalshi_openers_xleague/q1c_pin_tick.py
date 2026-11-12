"""Q1 reference B (descriptive, as nba_l32b_exec Q1): Kalshi implied line at our first Pinnacle sighting (hourly candle,
carry-forward) vs the Pinnacle open and Pinnacle close. Games = logger picks with a settle row (close known).
Usage: q1c_pin_tick.py nfl|nhl"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from kx import implied, side_x  # noqa: E402
from q1_timing import nhl_book_median, boot  # noqa: E402

D = Path.home() / ".claude/quant-kalshi/data"
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
lg = sys.argv[1]
rows = [json.loads(l) for l in open(D / f"open_lines_{lg}_picks.jsonl")]
picks = {(r["key"], r["market"]): r for r in rows if r["type"] == "pick"}
settle = {(r["key"], r["market"]): r for r in rows if r["type"] == "settle"}
T = json.load(open(D / f"open_lines_{lg}_tight.json"))["markets"]
C = pd.read_parquet(OUT / f"{lg}_candles.parquet")
out = []
for (k, mk), p in picks.items():
    s = settle.get((k, mk))
    if not s or mk not in ("total", "spread"):
        continue
    date, teams = k.split("|")
    a, h = teams.split("@")
    ev = next((e for e, m in T.items() if m["mk"] == mk and m["date"] == date and set(m["teams"]) == {a, h}), None)
    if ev is None:
        continue
    t = int((pd.Timestamp(p["open_t"]) - EPOCH) // pd.Timedelta(seconds=1))
    c = C[(C.event == ev) & (C.end <= t + 3600) & (C.end >= t - 6 * 3600) & C.bid.notna() & C.ask.notna()]
    # the hourly candle that CONTAINS the sighting closes up to 1 h after it; use candles ending <= t+1h only if
    # none end <= t (flagged) -- the close of the containing hour can include up to 59 min of post-sighting quotes
    c0 = c[c.end <= t]
    flag = c0.empty
    c = c if flag else c0
    q = c.sort_values("end").groupby("ticker").tail(1)
    q = q[(q.bid > 0) & (q.ask < 1)]
    pts = []
    for r in q.itertuples():
        x, sg = side_x(ev, r.ticker, r.strike, mk)
        mid = (r.bid + r.ask) / 2
        pts.append((x, mid if sg > 0 else 1 - mid))
    k_imp = implied(pts) if pts else np.nan
    if lg == "nhl":
        po = nhl_book_median(p["open_line"], p["open_p1"], p["open_p2"])
        pc = nhl_book_median(s["close_line"], s["close_p1"], s["close_p2"])
    elif mk == "total":
        po, pc = p["open_line"], s["close_line"]
    else:
        po, pc = -p["open_line"], -s["close_line"]
    out.append(dict(key=k, mk=mk, event=ev, pin_open_t=p["open_t"], used_next_hour=flag, k_at_pin_open=k_imp,
                    pin_open=po, pin_close=pc))
d = pd.DataFrame(out)
d["a_open"] = (d.k_at_pin_open - d.pin_open).abs()
d["a_close"] = (d.k_at_pin_open - d.pin_close).abs()
d["D"] = d.a_open - d.a_close
d.to_csv(OUT / f"q1c_{lg}_rows.csv", index=False)
res = {}
for mk, x in d.groupby("mk"):
    x = x[x.D.notna()]
    mv = x[(x.pin_open - x.pin_close).abs() > 1e-9]
    m, lo, hi = boot(mv.D)
    res[mk] = dict(n=len(x), n_moved=len(mv), a_open=round(mv.a_open.mean(), 3), a_close=round(mv.a_close.mean(), 3),
                   D=[round(m, 3), round(lo, 3), round(hi, 3)], closer_open=round((mv.a_open < mv.a_close).mean(), 3),
                   k_minus_open=round((x.k_at_pin_open - x.pin_open).mean(), 3), used_next_hour=int(x.used_next_hour.sum()),
                   open_close_abs=round((mv.pin_open - mv.pin_close).abs().mean(), 3))
print(lg, json.dumps(res))
json.dump(res, open(OUT / f"q1c_{lg}_summary.json", "w"), indent=1)
