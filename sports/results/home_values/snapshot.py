"""DESCRIPTIVE (not scored): N0 nowcast vs current Kalshi quotes for every *HOME market.
usage: snapshot.py [--rev_pct R]   R = extra 1-sigma level-restatement noise in % (added in quadrature)."""
import argparse, json
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import norm

H = Path(__file__).resolve().parent
PHI = 0.85
ap = argparse.ArgumentParser(); ap.add_argument("--rev_pct", type=float, default=0.0); a = ap.parse_args()
cal = json.load(open(H / "calib.json"))
C = {int(k): v["c_h"] for k, v in cal["by_h"].items()}
m = pd.DataFrame(json.load(open(H / "markets.json")))
m["rid"] = m.custom_strike.map(lambda c: int(c["region_id"]))
m["ht"] = m.custom_strike.map(lambda c: c["home_type"])
m["period"] = pd.to_datetime(m.event_ticker.str.split("-").str[1], format="%d%b%y")
files = {("single-family homes", "city"): "City_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month.csv",
         ("single-family homes", "nb"): "Neighborhood_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month.csv",
         ("condos/co-ops", "city"): "City_zhvi_uc_condo_tier_0.33_0.67_sm_sa_month.csv"}
ser = {}
for (ht, _), f in files.items():
    d = pd.read_csv(H / "raw" / f)
    cols = [c for c in d.columns if c[:2] in ("19", "20")]
    for rid in set(m.rid[m.ht == ht]) & set(d.RegionID):
        ser[(ht, rid)] = pd.Series(np.log(d.loc[d.RegionID == rid, cols].to_numpy(float)[0]), index=pd.to_datetime(cols))
rows = []
for _, x in m.iterrows():
    s = ser[(x.ht, x.rid)].dropna()
    L = s.index[-1]
    h = (x.period.year - L.year) * 12 + x.period.month - L.month
    d = s.diff().dropna()
    mu = s.iloc[-1] + d.iloc[-1] * sum(PHI ** k for k in range(1, h + 1))
    sc = np.std(np.diff(d.iloc[-37:].to_numpy()))
    c = C.get(h, C[3] * (h / 3) ** 1.5)
    sig = np.sqrt((c * sc * 1.5) ** 2 + (a.rev_pct / 100) ** 2)
    K = float(x.floor_strike) if x.strike_type == "greater" else float(x.cap_strike)
    z = (np.log(K) - mu) / sig
    p_yes = 1 - norm.cdf(z) if x.strike_type == "greater" else norm.cdf(z)
    rows.append(dict(ticker=x.ticker, h=h, last=round(np.exp(s.iloc[-1])), nowcast=round(np.exp(mu)), K=K,
                     sig_pct=round(sig * 100, 2), p_yes=round(p_yes, 3), yb=float(x.yes_bid_dollars), ya=float(x.yes_ask_dollars),
                     vol=float(x.volume_fp)))
r = pd.DataFrame(rows)
r["mid"] = (r.yb + r.ya) / 2
r["edge_yes"] = r.p_yes - r.ya
r["edge_no"] = (1 - r.p_yes) - (1 - r.yb)
pd.set_option("display.width", 250)
print(r.to_string(index=False))
r.to_csv(H / f"snapshot_rev{a.rev_pct}.csv", index=False)
