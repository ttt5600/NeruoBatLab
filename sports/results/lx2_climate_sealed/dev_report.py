"""Dev-split report for lx2c_a_t90_climate_tail_no (sealed blocked). Run from sports/."""
import json

import pandas as pd
from scipy.stats import beta

TID = "lx2c_a_t90_climate_tail_no-a7c423d2"
ev = [json.loads(l) for l in open("lab/edgelab_registry.jsonl") if TID in l and '"event": "evaluate"' in l][-1]
print({k: ev.get(k) for k in ["n", "net_roi", "ci", "p", "bar", "losses", "mean_cost", "clusters", "checks"]})
b = pd.read_csv("results/lx2_tail_no_carry/bets_dev.csv")
b = b[(b.trial == "A") & (b.category == "Climate and Weather")]
n, L = len(b), int((b.won == 0).sum())
lo = 0.0 if L == 0 else beta.ppf(.025, L, n - L + 1)
hi = beta.ppf(.975, L + 1, n - L)
print("n", n, "events", b.event_ticker.nunique(), "losses", L, "CP95 loss rate", [round(lo, 4), round(hi, 4)],
      "break-even loss", round(1 - b.cost.mean(), 4))
# events as the unit: a season's markets move together
ne = b.event_ticker.nunique()
print("event-level CP95 upper", round(beta.ppf(.975, 1, ne), 4))
print("roi noint", b.pnl.sum() / b.cost.sum(), "roi +APY", (b.pnl.sum() + b.interest.sum()) / b.cost.sum(),
      "mean days", b.days.mean())
b["season"] = pd.to_datetime(b.settlement_ts).dt.year
g = b.groupby("season").agg(bets=("pnl", "size"), events=("event_ticker", "nunique"), pnl=("pnl", "sum"))
g["share"] = g.pnl / b.pnl.sum()
print(g)
s = b.groupby("series").agg(bets=("pnl", "size"), pnl=("pnl", "sum"))
s["share"] = s.pnl / b.pnl.sum()
print(s.sort_values("pnl", ascending=False).head(8))
print("series n", b.series.nunique())
