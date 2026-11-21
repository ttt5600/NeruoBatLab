"""UFC live JSON -> per-fight timing table (horn, results, unofficial winner timestamps) + outcome; join to fights.parquet.
No Kalshi prices here."""
import json, glob, sys
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / "ufc_tsa"))
from ufc_hist import nkey, norm
rows = []
for f in sorted(glob.glob(str(OUT / "ufclive/ev_*.json"))):
    d = json.load(open(f))["LiveEventDetail"]
    for fc in d.get("FightCard") or []:
        fs = fc.get("Fighters") or []
        if len(fs) != 2: continue
        tr = fc.get("FightNightTracking") or []
        t = {}
        for a in tr:
            k = a["Type"]
            if k in ("fight_open", "fight_over", "results", "fight_complete") or k.startswith("unofficial_winner"):
                t.setdefault(k, a["Timestamp"])
                if k.startswith("unofficial_winner"): t["uw_type"] = k; t["uw_fighter"] = a.get("FighterId")
        rs = [a for a in tr if a["Type"] == "round_start"]; re_ = [a for a in tr if a["Type"] == "round_end"]
        npause = sum(a["Type"] == "round_pause" for a in tr)
        res = fc.get("Result") or {}
        name = lambda x: f"{x['Name']['FirstName'] or ''} {x['Name']['LastName'] or ''}".strip()
        rows.append({"ev_id": d["EventId"], "event": d["Name"], "ev_start": d["StartTime"], "fight_fmid": fc["FightId"],
                     "n1": name(fs[0]), "n2": name(fs[1]), "id1": fs[0]["FighterId"], "id2": fs[1]["FighterId"],
                     "out1": (fs[0].get("Outcome") or {}).get("Outcome"), "out2": (fs[1].get("Outcome") or {}).get("Outcome"),
                     "method": res.get("Method"), "end_round": res.get("EndingRound"), "end_time": res.get("EndingTime"),
                     "poss_rounds": (fc.get("RuleSet") or {}).get("PossibleRounds"),
                     "t_open": t.get("fight_open"), "t_r1": rs[0]["Timestamp"] if rs else None,
                     "t_last_round_end": re_[-1]["Timestamp"] if re_ else None, "t_horn": t.get("fight_over"),
                     "t_results": t.get("results"), "t_uw": next((t[k] for k in t if k.startswith("unofficial_winner") and k != "uw_type" and k != "uw_fighter"), None),
                     "uw_type": t.get("uw_type"), "uw_fighter": t.get("uw_fighter"), "t_complete": t.get("fight_complete"),
                     "n_rounds_tracked": len(rs), "n_pause": npause, "has_scores": bool(res.get("FightScores"))})
u = pd.DataFrame(rows).drop_duplicates("fight_fmid")
for c in ["ev_start", "t_open", "t_r1", "t_last_round_end", "t_horn", "t_results", "t_uw", "t_complete"]:
    u[c] = pd.to_datetime(u[c], utc=True, format="ISO8601")
u.to_parquet(OUT / "ufclive_fights.parquet")
print("ufc live fights", len(u), "events", u.ev_id.nunique(), "with horn", u.t_horn.notna().sum(), "with uw", u.t_uw.notna().sum())
j = pd.read_parquet(OUT / "fights.parquet")
j = j[j.fid.notna()].copy()
def match(r):
    c = u[(u.ev_start.dt.tz_convert(None).dt.normalize() - r.card).abs() <= pd.Timedelta(days=1)]
    A, B = set(norm(r.ufA)), set(norm(r.ufB))
    def sc(x):
        s1, s2 = set(norm(x.n1)), set(norm(x.n2))
        return max(len(A & s1) + len(B & s2), len(A & s2) + len(B & s1))
    if not len(c): return pd.Series({"fight_fmid": None})
    s = c.apply(sc, axis=1); best = s.max()
    if best < 2 or (s == best).sum() != 1: return pd.Series({"fight_fmid": None})
    x = c.loc[s.idxmax()]
    a_is_1 = len(A & set(norm(x.n1))) >= len(A & set(norm(x.n2)))
    return pd.Series({"fight_fmid": x.fight_fmid, "a_is_1": a_is_1})
m = pd.concat([j.reset_index(drop=True), j.reset_index(drop=True).apply(match, axis=1)], axis=1)
m = m.merge(u, on="fight_fmid", how="left")
print("Kalshi-matched fights with UFC live row", m.fight_fmid.notna().sum(), "of", len(m), "| decisions", (m.dec & m.fight_fmid.notna()).sum(), "of", m.dec.sum())
dd = m[m.dec & m.t_horn.notna()]
dd = dd.assign(horn_to_results=(dd.t_results - dd.t_horn).dt.total_seconds(), horn_to_uw=(dd.t_uw - dd.t_horn).dt.total_seconds(),
               kalshi_close_after_uw=(dd.ct - dd.t_uw).dt.total_seconds())
print(dd[["horn_to_results", "horn_to_uw", "kalshi_close_after_uw"]].describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_string())
print("uw_type on decisions", dd.uw_type.value_counts(dropna=False).to_dict())
m.to_parquet(OUT / "fights_timed.parquet")
