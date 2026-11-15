"""T2: X1 forward paper record since the pause (and full forward for context), tape-through fill model.
Read-only public GETs (tape + market close). Run from sports/: ../.venv_quant/bin/python results/e4_evidence/t2_x1_forward.py"""
import json, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import kalshi_temp_maker as K
from sports import wx
from sports.http import fetch

OUT = "results/e4_evidence/"
PAUSE = pd.Timestamp("2026-11-04T03:00:00Z")  # first skip_cohort_paused order in kalshi_exec.jsonl (2026-11-04T03:03:12Z)
p = pd.DataFrame([json.loads(l) for l in open("/Users/jonathanwang/.claude/quant-kalshi/data/kalshi_paper.jsonl")])
p["t0"] = pd.to_datetime(p.logged_at, utc=True)
p["cohort"] = p.cohort.fillna("orig7")
p = p[(p.cohort == "x1") & p.result.isin(["yes", "no"]) & (p.t0 >= "2026-10-20")].drop_duplicates("ticker").copy()
yb, ya = p.yes_bid.astype(float), p.yes_ask.astype(float)
ask = 1 - yb
p["q"] = np.round(np.minimum(np.floor(np.round((1 - (yb + ya) / 2) * 100, 6)) / 100, ask - 0.01), 4)
p["ask"] = ask
p = p[(p.q > 0) & (ask - p.q <= 0.05)]  # executor skip rule


def one(r):
    m = fetch(f"{wx.API}/markets/{r.ticker}", {}, cache=True).get("market", {})
    close = int(pd.Timestamp(m.get("close_time")).timestamp()) if m.get("close_time") else int(r.t0.timestamp()) + 36 * 3600
    t0 = int(r.t0.timestamp())  # epoch seconds from tz-aware Timestamp (not the datetime64[us]//1e9 trap)
    tp = pd.DataFrame(K._tape(r.ticker, t0, close))
    if len(tp):
        tp = tp[(tp.ts > t0) & (tp.ts <= close)]
    le = float(tp[tp.no_price <= r.q + 1e-9]["count"].sum()) if len(tp) else 0.0
    lt = float(tp[tp.no_price < r.q - 1e-9]["count"].sum()) if len(tp) else 0.0
    return {"vol_le_q": le, "vol_lt_q": lt, "close_ts": close, "n_prints": len(tp)}


with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(one, p.itertuples()))
p = pd.concat([p.reset_index(drop=True), pd.DataFrame(res)], axis=1)
p["won"] = (p.result == "no").astype(int)
p["night"] = p.t0.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
p["post_pause"] = p.t0 >= PAUSE
p[["ticker", "city", "night", "post_pause", "q", "ask", "won", "vol_le_q", "vol_lt_q", "n_prints"]].to_csv(OUT + "x1_forward_rows.csv", index=False)

rng = np.random.default_rng(0)


def score(d, rule):
    lt, le, won, q = d.vol_lt_q.to_numpy(), d.vol_le_q.to_numpy(), d.won.to_numpy(), d.q.to_numpy()
    f = (won == 0) | ((lt if rule == "cons" else le) >= 10)  # losers always fill (wx_capacity: 6/6 traded through)
    pn = np.where(won == 1, 1 / q - 1, -1.0) * f  # 1 unit of cost per bet, maker fee 0 (as lockbox)
    nights = d.night.to_numpy(); u, inv = np.unique(nights, return_inverse=True)
    sx = np.bincount(inv, weights=pn, minlength=len(u)); sd = np.bincount(inv, weights=f.astype(float), minlength=len(u))
    idx = rng.integers(0, len(u), size=(4000, len(u)))
    den = sd[idx].sum(1)
    r = np.where(den > 0, sx[idx].sum(1) / np.maximum(den, 1e-12), 0.0)
    return {"picks": int(len(d)), "nights": int(len(u)), "fills": int(f.sum()), "fill_rate": round(float(f.mean()), 3),
            "losses": int((won == 0).sum()), "units": round(float(pn.sum()), 3),
            "roi": round(float(pn.sum() / max(f.sum(), 1)), 5),
            "ci90": [round(float(np.quantile(r, a)), 5) for a in (.05, .95)],
            "ci95": [round(float(np.quantile(r, a)), 5) for a in (.025, .975)]}


out = {}
for lab, d in (("post_pause", p[p.post_pause]), ("all_forward", p)):
    for rule in ("cons", "opt"):
        out[f"{lab}_{rule}"] = score(d, rule)
out["post_pause_window"] = [str(p[p.post_pause].night.min()), str(p[p.post_pause].night.max())]
out["losers_post_pause"] = p[p.post_pause & (p.won == 0)][["ticker", "q", "vol_lt_q"]].to_dict("records")
json.dump(out, open(OUT + "t2_x1_forward.json", "w"), indent=1)
print(json.dumps(out, indent=1))
