"""Join Kalshi MOV/VICROUND/FIGHT events (all splits; no prices) to UFCStats bouts by card date +-1 d and fighter names.
Output fight-level table with fighter A/B full names, Kalshi codes, weight class, scheduled rounds."""
import re, sys
from pathlib import Path
import pandas as pd
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from ufc_hist import fights, nkey, norm
m = pd.read_parquet(OUT / "ufc_markets.parquet")
mov = m[m.series == "KXUFCMOV"].copy()
mov["suffix"] = mov.ticker.str.split("-").str[2]
mov = mov[~mov.suffix.str.startswith("DRAW")]
mov["code"] = mov.suffix.str.replace(r"(KOTKODQ|SUB|DEC)$", "", regex=True)
mov["fname"] = mov.yes_sub_title.str.replace(r" by (Submission|KO/TKO/DQ|Decision)$", "", regex=True)
ev = mov.groupby(["event_ticker", "fight", "card", "split"]).apply(lambda g: sorted(set(zip(g["code"], g["fname"]))), include_groups=False).rename("fs").reset_index()
ev = ev[ev.fs.map(len) == 2]
fr = fights()
def match(r):
    c = fr[(fr.date - r.card).abs() <= pd.Timedelta(days=1)]
    (ca, na), (cb, nb) = r.fs
    ka, kb = nkey(na), nkey(nb)
    hit = c[((c.f1.map(nkey) == ka) & (c.f2.map(nkey) == kb)) | ((c.f1.map(nkey) == kb) & (c.f2.map(nkey) == ka))]
    how = "full"
    if len(hit) != 1:   # fall back: last names
        la, lb = norm(na)[-1], norm(nb)[-1]
        hit = c[c.apply(lambda x: {norm(x.f1)[-1], norm(x.f2)[-1]} == {la, lb}, axis=1)]
        how = "last"
    if len(hit) != 1:   # fall back: shared tokens (len >= 3) for both fighters, unique bout
        ta = {t for t in norm(na) if len(t) >= 3}; tb = {t for t in norm(nb) if len(t) >= 3}
        def ok(x):
            s1 = set(norm(x.f1)); s2 = set(norm(x.f2))
            return (bool(ta & s1) and bool(tb & s2)) or (bool(ta & s2) and bool(tb & s1))
        hit = c[c.apply(ok, axis=1)] if len(c) else c
        how = "token"
    if len(hit) != 1: return pd.Series({"fid": None, "how": f"n{len(hit)}"})
    h = hit.iloc[0]
    # map Kalshi names onto UFCStats names (by last name)
    ta = set(norm(na)); ufa = h.f1 if len(ta & set(norm(h.f1))) >= len(ta & set(norm(h.f2))) else h.f2
    ufb = h.f2 if ufa == h.f1 else h.f1
    return pd.Series({"fid": h.fid, "how": how, "codeA": ca, "nameA": na, "ufA": ufa, "codeB": cb, "nameB": nb, "ufB": ufb,
                      "wclass": h.wclass, "sched5": h.sched5, "women": h.women, "date": h.date,
                      "mclass": h.mclass, "winner": h.winner, "round": h["round"], "draw": h.draw})
j = pd.concat([ev, ev.apply(match, axis=1)], axis=1)
print(j.how.value_counts().to_dict(), "by split", j.groupby("split").fid.apply(lambda x: x.notna().mean()).round(3).to_dict())
print("unmatched sample", j[j.fid.isna()][["event_ticker", "fs"]].head(8).to_string())
j.drop(columns=["fs"]).to_parquet(OUT / "ufc_fights.parquet")
