"""wx_capacity Q3: per-city maker depth for E4 picks (orig7 + X1). Descriptive, pre-registered in notes/wx_capacity.md.
Run from sports/: ../.venv_quant/bin/python results/wx_capacity/capacity.py
Read-only public GETs (tape + market close time). No orders."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import kalshi_temp_maker as K
from sports import wx
from sports.http import fetch

OUT = "results/wx_capacity/"
rows = [json.loads(l) for l in open("/Users/jonathanwang/.claude/quant-kalshi/data/kalshi_paper.jsonl")]
p = pd.DataFrame(rows)
p["t0"] = pd.to_datetime(p.logged_at, utc=True)
p = p[(p.t0 >= "2026-10-22") & (p.t0 < "2026-11-11") & p.result.isin(["yes", "no"])].copy()
p["cohort"] = p.cohort.fillna("orig7")
p = p.drop_duplicates("ticker")
yb, ya = p.yes_bid.astype(float), p.yes_ask.astype(float)
ask = 1 - yb
q = np.minimum(np.floor(np.round((1 - (yb + ya) / 2) * 100, 6)) / 100, ask - 0.01)
p["q"] = np.round(q, 4)
p["ask"] = ask
p = p[(p.q > 0) & (ask - p.q <= 0.05)]  # executor skip rule (kalshi_temp_maker change spec 2)


def one(r):
    m = fetch(f"{wx.API}/markets/{r.ticker}", {}, cache=True).get("market", {})
    close = int(pd.Timestamp(m.get("close_time")).timestamp()) if m.get("close_time") else int(r.t0.timestamp()) + 36 * 3600
    t0 = int(r.t0.timestamp())
    tp = pd.DataFrame(K._tape(r.ticker, t0, close))
    if len(tp):
        tp = tp[(tp.ts > t0) & (tp.ts <= close)]
    le = float(tp[tp.no_price <= r.q + 1e-9]["count"].sum()) if len(tp) else 0.0
    lt = float(tp[tp.no_price < r.q - 1e-9]["count"].sum()) if len(tp) else 0.0
    tot = float(tp["count"].sum()) if len(tp) else 0.0
    return {"vol_le_q": le, "vol_lt_q": lt, "vol_total": tot, "close_ts": close}


with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(one, p.itertuples()))
p = pd.concat([p.reset_index(drop=True), pd.DataFrame(res)], axis=1)
p["won"] = (p.result == "no").astype(int)
p["night"] = p.t0.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")  # all picks ~22:00 local -> same ET evening
p[["ticker", "city", "cohort", "night", "q", "ask", "won", "vol_le_q", "vol_lt_q", "vol_total", "depth_1c"]].to_csv(OUT + "picks_depth.csv", index=False)
print(p.groupby("cohort").agg(n=("q", "size"), losses=("won", lambda x: int((x == 0).sum())), med_lt=("vol_lt_q", "median"),
                              med_le=("vol_le_q", "median"), q=("q", "mean")))
