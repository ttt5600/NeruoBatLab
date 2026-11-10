"""Q1 live maker ROI with date-cluster bootstrap; Q2 filled vs unfilled; maker-vs-taker counterfactual on same picks."""
import json
import numpy as np, pandas as pd
d = pd.read_parquet("results/e4_live_audit/live_rows.parquet")
FEE = 0.07
m = d[(d.exec_style == "maker") & d.status.isin(["filled", "no_fill", "partial"])].copy()
m["filled"] = m.filled.astype(float)
m["paper_result"] = m.paper_result.where(m.paper_result.isin(["yes", "no"]))
s = m[m.paper_result.notna()].copy()
f = s[s.filled > 0].copy()
assert (f.result == f.paper_result).all(), "exchange settle result != paper result"
rng = np.random.default_rng(0)
B = 20000


def boot_ratio(df, num, den, B=B):
    dates = df.edate.unique()
    g = df.groupby("edate")[[num, den]].sum().reindex(dates)
    idx = rng.integers(0, len(dates), (B, len(dates)))
    nn, dd = g[num].values[idx].sum(1), g[den].values[idx].sum(1)
    return np.percentile(nn / dd, [2.5, 97.5])

out = {}
# Q1a: dollar-weighted ROI on settled maker fills
roi = f.pnl.sum() / f.cost.sum()
out["maker_fills_settled"] = int(len(f)); out["maker_record"] = f"{(f.result=='no').sum()}-{(f.result=='yes').sum()}"
out["maker_pnl"] = round(f.pnl.sum(), 2); out["maker_cost"] = round(f.cost.sum(), 2)
out["roi_dollar"] = round(roi, 4); out["roi_dollar_ci"] = [round(x, 4) for x in boot_ratio(f, "pnl", "cost")]
# Q1b: per-fill equal-weight ROI (each fill = 1 unit), what the sealed test measures
f["r1"] = f.pnl / f.cost; f["one"] = 1.0
out["roi_equal"] = round(f.r1.mean(), 4); out["roi_equal_ci"] = [round(x, 4) for x in boot_ratio(f, "r1", "one")]
out["n_dates"] = int(f.edate.nunique())
# Q2: fill rate, outcomes
out["placed_settled"] = int(len(s)); out["fill_rate"] = round((s.filled > 0).mean(), 3)
out["filled_loss"] = f"{(f.result=='yes').sum()}/{len(f)}"
u = s[s.filled == 0]
out["unfilled_loss"] = f"{(u.paper_result=='yes').sum()}/{len(u)}"
# counterfactual taker on the same placed picks at the pick price + taker fee, same count
s["q_t"] = s.pick_no_price.astype(float)
s["n"] = s["count"].astype(float)
s["taker_pnl"] = np.where(s.paper_result == "no", s.n * (1 - s.q_t), -s.n * s.q_t) - s.n * FEE * s.q_t * (1 - s.q_t)
s["taker_cost"] = s.n * s.q_t
s["maker_pnl"] = s.pnl.fillna(0.0)
s["d"] = s.maker_pnl - s.taker_pnl
out["taker_cf_pnl"] = round(s.taker_pnl.sum(), 2); out["maker_minus_taker"] = round(s.d.sum(), 2)
dates = s.edate.unique(); g = s.groupby("edate").d.sum().reindex(dates).values
bs = g[rng.integers(0, len(dates), (B, len(dates)))].sum(1)
out["maker_minus_taker_ci"] = [round(x, 2) for x in np.percentile(bs, [2.5, 97.5])]
# Power: per-fill r1 SD from live and from a 5% loss-rate model
sd_live = f.r1.std()
p, q = 0.0565, 0.955  # dev M3_N10 filled loss rate; typical q
r_win, r_loss = (1 - q) / q, -1.0
sd_model = np.sqrt(p * (1 - p)) * (r_win - r_loss)
for name, sd in [("sd_live", sd_live), ("sd_model", sd_model)]:
    out[name] = round(sd, 3)
    out[f"n_needed_{name}"] = int(np.ceil(((1.96 + 0.84) * sd / 0.0222) ** 2))
    out[f"halfwidth_now_{name}"] = round(1.96 * sd / np.sqrt(len(f)), 4)
print(json.dumps(out, indent=1))
json.dump(out, open("results/e4_live_audit/q1_q2.json", "w"), indent=1)
