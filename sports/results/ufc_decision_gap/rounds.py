"""UFCStats round-level stats -> per-fight round-difference table for decisions (outcome-only; no Kalshi data).
Output rounds.parquet: one row per (fid, round) with A-minus-B diffs, A = f1 as listed by UFCStats (orientation handled later)."""
import re, sys
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / "ufc_tsa"))
from ufc_hist import nkey
s = pd.read_csv(OUT / "dl/ufc_fight_stats.csv")
r = pd.read_csv(OUT / "dl/ufc_fight_results.csv"); e = pd.read_csv(OUT / "dl/ufc_event_details.csv")
for x in (s, r, e): x["EVENT"] = x.EVENT.str.strip()
s["BOUT"] = s.BOUT.str.strip().str.replace(r"\s+", " ", regex=True); r["BOUT"] = r.BOUT.str.strip().str.replace(r"\s+", " ", regex=True)
r = r.merge(e[["EVENT", "DATE"]], on="EVENT", how="inner"); r["date"] = pd.to_datetime(r.DATE, format="%B %d, %Y")
ab = r.BOUT.str.split(" vs. ", n=1, expand=True); r["f1"] = ab[0].str.strip(); r["f2"] = ab[1].str.strip()
o = r.OUTCOME.str.strip(); r["y"] = np.select([o == "W/L", o == "L/W"], [1.0, 0.0], np.nan); r["draw"] = o == "D/D"
r["dec"] = r.METHOD.str.strip().str.startswith("Decision"); r["fid"] = r.URL
r["sched5"] = r["TIME FORMAT"].str.startswith("5 Rnd"); r["nround"] = pd.to_numeric(r.ROUND, errors="coerce")
dup = r.duplicated(["EVENT", "BOUT"], keep=False); print("dup EVENT+BOUT in results", dup.sum())
r = r[~dup]
s = s[s.ROUND.astype(str).str.startswith("Round")].copy()
s["rnd"] = s.ROUND.str.extract(r"(\d+)").astype(int)
def of(col):
    x = s[col].astype(str).str.extract(r"(\d+) of (\d+)")
    return pd.to_numeric(x[0], errors="coerce"), pd.to_numeric(x[1], errors="coerce")
s["sig_l"], s["sig_a"] = of("SIG.STR."); s["tot_l"], s["tot_a"] = of("TOTAL STR."); s["td_l"], s["td_a"] = of("TD")
s["head_l"], _ = of("HEAD"); s["body_l"], _ = of("BODY"); s["leg_l"], _ = of("LEG")
s["dist_l"], _ = of("DISTANCE"); s["clin_l"], _ = of("CLINCH"); s["grd_l"], _ = of("GROUND")
c = s.CTRL.astype(str).str.extract(r"(\d+):(\d+)"); s["ctrl"] = pd.to_numeric(c[0], errors="coerce") * 60 + pd.to_numeric(c[1], errors="coerce")
for col in ("KD", "SUB.ATT", "REV."): s[col] = pd.to_numeric(s[col], errors="coerce")
s = s.rename(columns={"KD": "kd", "SUB.ATT": "subatt", "REV.": "rev"})
s = s.merge(r[["EVENT", "BOUT", "f1", "f2", "fid", "date", "y", "draw", "dec", "sched5", "nround", "METHOD"]], on=["EVENT", "BOUT"], how="inner")
s["fk"] = s.FIGHTER.map(nkey); s["is1"] = s.fk == s.f1.map(nkey); s["is2"] = s.fk == s.f2.map(nkey)
print("rows", len(s), "fighter name resolves to f1/f2:", (s.is1 | s.is2).mean().round(4), "both?", (s.is1 & s.is2).sum())
s = s[s.is1 ^ s.is2]
F = ["kd", "sig_l", "sig_a", "tot_l", "tot_a", "td_l", "td_a", "ctrl", "subatt", "rev", "head_l", "body_l", "leg_l", "dist_l", "clin_l", "grd_l"]
a = s[s.is1].set_index(["fid", "rnd"])[F]; b = s[s.is2].set_index(["fid", "rnd"])[F]
dif = (a - b).dropna().add_prefix("d_")
meta = s[s.is1].drop_duplicates(["fid", "rnd"]).set_index(["fid", "rnd"])[["date", "y", "draw", "dec", "sched5", "nround", "METHOD", "f1", "f2", "EVENT", "BOUT"]]
out = meta.join(dif, how="inner").join(a.add_prefix("a_"), how="left").join(b.add_prefix("b_"), how="left").reset_index()
# complete-round check for decisions: all scheduled rounds present
nr = out.groupby("fid").rnd.agg(["max", "size"])
out = out.merge(nr.rename(columns={"max": "rmax", "size": "rn"}), left_on="fid", right_index=True)
d = out[out.dec].drop_duplicates("fid")
print("decision fights with round stats", len(d), "complete rounds", ((d.rn == d.rmax) & (d.rmax == np.where(d.sched5, 5, 3))).mean().round(4))
print("decisions by year", d.groupby(d.date.dt.year).size().tail(12).to_dict())
out.to_parquet(OUT / "rounds.parquet")
