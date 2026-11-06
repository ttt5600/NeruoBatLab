"""mirror direction at minute resolution (descriptive; intl book not tradable for US): when Kalshi mid - Poly mid >= X
(Kalshi state ~7 s older than the Poly point), buy Poly at mid + 0.5c + PMUS fee; CLV = Poly mid 1 and 5 steps later."""
import numpy as np
import pandas as pd

D = "data/xvenue_ingame/"
DUR = {"nba": 3.0, "nfl": 4.0, "nhl": 3.0, "mlb": 4.0, "cbb": 2.5, "cfb": 4.5}
m = pd.read_parquet(D + "match.parquet")
m = m[m.split == "dev"]
k = pd.read_parquet(D + "kalshi_minute_dev.parquet").sort_values("t")
p = pd.read_parquet(D + "poly_points_dev.parquet").merge(m[["cond", "lg", "start"]], on="cond", validate="many_to_one")
p["t0"] = (p.start - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
x = pd.merge_asof(p.sort_values("pt"), k[["cond", "t", "k_bid", "k_ask"]], left_on="pt", right_on="t", by="cond",
                  direction="backward", tolerance=59).sort_values(["cond", "pt"]).reset_index(drop=True)
x["k_mid"] = (x.k_bid + x.k_ask) / 2
x = x[(x.pt >= x.t0) & (x.pt <= x.t0 + x.lg.map(DUR) * 3600) & x.k_mid.between(.03, .97) & x.p_px.between(.03, .97)
      & ((x.k_ask - x.k_bid) <= .10)].copy()
for h in (1, 5):
    x[f"p{h}"] = x.groupby("cond").p_px.shift(-h)
rng = np.random.default_rng(0)
for X in (.03, .05):
    d = x.k_mid - x.p_px
    s = x[d.abs() >= X].copy()
    sg = np.sign(d[d.abs() >= X])
    cost = 0.005 + 0.0695 * s.p_px * (1 - s.p_px)
    for h in (1, 5):
        s["net"] = sg * (s[f"p{h}"] - s.p_px) - cost
        y = s.dropna(subset=["net"])
        g = y.groupby("cond").net.agg(["sum", "count"]).values
        v = g[rng.integers(0, len(g), (2000, len(g)))].sum(1)
        print(f"X={X:.2f} h={h} n={len(y)} games={len(g)} mean net {g[:, 0].sum() / g[:, 1].sum():+.4f} "
              f"[{np.percentile(v[:, 0] / v[:, 1], 2.5):+.4f},{np.percentile(v[:, 0] / v[:, 1], 97.5):+.4f}]")
