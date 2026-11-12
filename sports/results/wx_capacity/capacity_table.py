"""wx_capacity Q3 table from picks_depth.csv (pre-registered in notes/wx_capacity.md).
Winner fill at size S = min(S, vol_lt_q) (conservative, level cleared) ; loser fill = S.
Realised: forward outcomes. Model: per-cohort filled-loss rate L (lockbox 10-23 pooled filled 3.07%) on every pick."""
import json
import numpy as np, pandas as pd

d = pd.read_csv("results/wx_capacity/picks_depth.csv")
nights = d.night.nunique()
SIZES = [6, 12, 24, 48, 96, 192]
L = 0.0307
# losers: check they traded through
lo = d[d.won == 0]
print("losers:", len(lo), "vol_lt_q>=24 on", int((lo.vol_lt_q >= 24).sum()), lo[["ticker", "q", "vol_lt_q"]].to_string())
out = {"nights": int(nights), "picks": int(len(d)), "window": [d.night.min(), d.night.max()]}
rows = []
for (coh, city), g in list(d.groupby(["cohort", "city"])) + [((c, "ALL"), g) for c, g in d.groupby("cohort")] + [(("all", "ALL"), d)]:
    r = {"cohort": coh, "city": city, "picks": len(g), "picks_per_night": round(len(g) / nights, 2),
         "losses": int((g.won == 0).sum()), "mean_q": round(g.q.mean(), 3),
         "med_vol_lt_q": float(g.vol_lt_q.median()), "med_vol_le_q": float(g.vol_le_q.median())}
    for S in SIZES:
        fw = np.minimum(S, g.vol_lt_q)
        # model E[$]: winners fill min(S, V) with prob (1-L), losers S
        model = ((1 - L) * fw * (1 - g.q) - L * S * g.q).sum() / nights
        real = (np.where(g.won == 1, fw * (1 - g.q), -S * g.q)).sum() / nights
        cap = (np.where(g.won == 1, fw, S) * g.q).sum() / nights
        r[f"model_$n_S{S}"] = round(model, 2)
        r[f"real_$n_S{S}"] = round(real, 2)
        r[f"capital_$n_S{S}"] = round(cap, 1)
    rows.append(r)
t = pd.DataFrame(rows)
t.to_csv("results/wx_capacity/capacity_table.csv", index=False)
pd.set_option("display.width", 250)
cols = ["cohort", "city", "picks_per_night", "losses", "mean_q", "med_vol_lt_q"] + [f"model_$n_S{S}" for S in SIZES]
print(t[cols].sort_values(["cohort", "model_$n_S24"], ascending=[True, False]).to_string(index=False))
print(t[t.city == "ALL"][["cohort"] + [f"real_$n_S{S}" for S in SIZES] + [f"capital_$n_S{S}" for S in SIZES]].to_string(index=False))
json.dump({**out, "table": rows}, open("results/wx_capacity/capacity_table.json", "w"), indent=1, default=float)

# ---- AMENDMENT 1 (written after seeing the table above; disclosed in notes): the pre-registered flat L = 3.07% is the
# FILLED-loss rate, wrongly applied to every pick incl. 99c picks that rarely lose and rarely fill. Model 2 uses the
# per-pick (all picks) loss rate by NO-ask bucket from DEV picks (data/kalshi_temp_maker_picks.parquet).
dev = pd.read_parquet("data/kalshi_temp_maker_picks.parquet")
EDGES = [0, .90, .94, .96, .97, .98, 1.0]
devL = dev.groupby(pd.cut(1 - dev.yes_bid, EDGES)).won.apply(lambda x: 1 - x.mean())
d["L2"] = pd.cut(d.ask, EDGES).map(devL).astype(float)
d["askb"] = pd.cut(d.ask, EDGES).astype(str)
print("dev loss by ask bucket", devL.round(4).to_dict())
rows2 = []
grp = list(d.groupby(["cohort", "city"])) + [((c, "ALL"), g) for c, g in d.groupby("cohort")] + [(("all", "ALL"), d)] \
    + [(("bucket", b), g) for b, g in d.groupby("askb")]
for (coh, city), g in grp:
    r = {"cohort": coh, "city": city, "picks_per_night": round(len(g) / nights, 2), "losses": int((g.won == 0).sum()),
         "share_lt_ge24": round(float((g.vol_lt_q >= 24).mean()), 2), "med_vol_lt_q": float(g.vol_lt_q.median())}
    for S in SIZES:
        fw = np.minimum(S, g.vol_lt_q)
        r[f"m2_S{S}"] = round((((1 - g.L2) * fw * (1 - g.q)) - g.L2 * S * g.q).sum() / nights, 2)
    rows2.append(r)
t2 = pd.DataFrame(rows2)
t2.to_csv("results/wx_capacity/capacity_table_m2.csv", index=False)
print(t2.sort_values(["cohort", "m2_S24"], ascending=[True, False]).to_string(index=False))
# executor-relevant: drop picks with NO ask >= .99 (q = .98 can only fill by trading through to .97)
k = d[d.ask < 0.985]
print("picks with ask<.985:", len(k), "of", len(d))
for coh, g in list(k.groupby("cohort")) + [("all", k)]:
    print(coh, {S: round((((1 - g.L2) * np.minimum(S, g.vol_lt_q) * (1 - g.q)) - g.L2 * S * g.q).sum() / nights, 2) for S in SIZES},
          "real", {S: round(np.where(g.won == 1, np.minimum(S, g.vol_lt_q) * (1 - g.q), -S * g.q).sum() / nights, 2) for S in SIZES})
