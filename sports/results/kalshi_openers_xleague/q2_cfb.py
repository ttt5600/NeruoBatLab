"""Q2 K-CFB (pre-registered, notes/kalshi_openers_xleague.md): frozen L36 totals model vs the Kalshi KXNCAAFTOTAL ladder
at Kalshi's FIRST TIGHT hour (>= 30 min pre-kick). S = strike nearest the DK open (tie -> favorable to our side);
bet if |pred - S| >= 3.5; price = that hour's close ask (YES over / 1 - bid under) + exact taker fee; graded on final
vs S; CLV = DK close - S signed. Descriptive: de-biased variant (c = trailing mean pred - open over prior games).
Needs q1_cfb_rows.csv, cfb_candles.parquet, cfb_games.parquet, cfb_preds_2026.parquet."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
import os  # noqa: E402
TAG = os.environ.get("KX_TAG", "")


def fee(p, n=100):
    return math.ceil(round(n * 0.07 * p * (1 - p) * 100, 6)) / 100 / n


def boot(v, seed=0):
    v = np.asarray(v, float)
    if len(v) < 2:
        return [float(v.mean()) if len(v) else None, None, None]
    bs = np.random.default_rng(seed).choice(v, (4000, len(v))).mean(1)
    return [round(v.mean(), 4), round(np.percentile(bs, 2.5), 4), round(np.percentile(bs, 97.5), 4)]


R = pd.read_csv(OUT / f"q1_cfb{TAG}_rows.csv")
R = R[(R.mk == "total") & R.ft_t.notna()]
C = pd.read_parquet(OUT / f"cfb{TAG}_candles.parquet")
G = pd.read_parquet(OUT / f"cfb{TAG}_games.parquet")
K = pd.read_parquet(OUT / f"cfb{TAG}_kmarkets.parquet")
P = pd.read_parquet(OUT / "cfb_preds_2026.parquet")
P["event_id"] = P.event_id.astype(str)
G["event_id"] = G.event_id.astype(str)
G = G.merge(P[["event_id", "pred_total", "fbs_game"]], on="event_id", how="left")
ev2key = K.drop_duplicates("event").set_index("event").key
gk = G.drop_duplicates("key").set_index("key")
# de-bias c: trailing mean(pred - DK open) over games that kicked off before this one (all 2026 FBS games with both)
H = G[G.pred_total.notna() & G.total_open.notna()].copy()
H["start"] = pd.to_datetime(H.start_utc, utc=True)
H = H.sort_values("start")
out = []
for r in R.itertuples():
    g = gk.loc[ev2key[r.event]]
    if pd.isna(g.pred_total) or pd.isna(g.total_open) or not bool(g.fbs_game):
        out.append(dict(event=r.event, status="no pred/open/non-FBS")); continue
    T = pd.Timestamp(r.ft_t)
    occ = pd.Timestamp(r.occ)
    if (occ - T) < pd.Timedelta(minutes=30):
        out.append(dict(event=r.event, status="ft < 30 min pre-kick")); continue
    t = int((T - EPOCH) // pd.Timedelta(seconds=1))
    c = C[(C.event == r.event) & (C.end <= t) & (C.end >= t - 6 * 3600) & C.bid.notna() & C.ask.notna()]
    q = c.sort_values("end").groupby("strike").tail(1)
    q = q[(q.bid > 0) & (q.ask < 1)]
    if q.empty:
        out.append(dict(event=r.event, status="no quote")); continue
    prev = H[H.start < T - pd.Timedelta(hours=0)]
    cbias = float((prev.pred_total - prev.total_open).tail(300).mean()) if len(prev) >= 50 else np.nan
    for variant, pred in (("raw", g.pred_total), ("debiased", g.pred_total - cbias if np.isfinite(cbias) else np.nan)):
        if not np.isfinite(pred):
            continue
        ks = q.strike.to_numpy()
        dist = np.abs(ks - g.total_open)
        cand = ks[dist == dist.min()]
        up0 = pred > g.total_open
        S = float(cand.min() if up0 else cand.max())
        gap = pred - S
        if abs(gap) < 3.5:
            out.append(dict(event=r.event, variant=variant, status="skip gap", S=S, gap=gap)); continue
        up = gap > 0
        z = q[q.strike == S].iloc[0]
        ask = z.ask if up else 1 - z.bid
        cost = ask + fee(ask)
        pts = g.home_score + g.away_score
        won = pts > S if up else pts < S
        clv = (g.total_close - S) if up else (S - g.total_close)
        out.append(dict(event=r.event, variant=variant, status="bet", T=T, pick="over" if up else "under", pred=pred,
                        c=cbias, open=g.total_open, close=g.total_close, S=S, gap=gap, ask=ask, cost=cost,
                        spread_c=round((z.ask - z.bid) * 100), final=pts, won=won,
                        roi=(1 - cost) / cost if won else -1.0, clv=clv))
d = pd.DataFrame(out)
d.to_csv(OUT / f"q2_cfb_rows_{TAG or 'dev'}.csv", index=False)
S = {"status": d.drop_duplicates(["event", "status"]).status.value_counts().to_dict()}
for v, x in d[d.status == "bet"].groupby("variant"):
    S[v] = dict(n=len(x), W=int((x.won == True).sum()), roi=boot(x.roi), clv=boot(x.clv), mean_cost=round(x.cost.mean(), 4),
                overs=int((x.pick == "over").sum()), mean_gap=round(x.gap.mean(), 2), mean_c=round(x.c.mean(), 2))
    for side, y in x.groupby("pick"):
        S[v][side] = dict(n=len(y), roi=round(y.roi.mean(), 4), clv=round(y.clv.mean(), 2))
print(json.dumps(S, indent=1, default=str))
json.dump(S, open(OUT / f"q2_cfb_summary_{TAG or 'dev'}.json", "w"), indent=1, default=str)
