"""KXUFCFIGHT finalized events -> UFCStats bouts (card date +-1 d, fighter names). No prices, no Kalshi outcomes used.
Output: fights.parquet (one row per Kalshi event, both market tickers, UFCStats fid, method, scheduled rounds)."""
import json, sys
from pathlib import Path
import pandas as pd
OUT = Path(__file__).resolve().parent
UT = OUT.parent / "ufc_tsa"
sys.path.insert(0, str(UT))
from ufc_hist import fights, nkey, norm
d = json.load(open(UT / "census_raw.json"))["KXUFCFIGHT"]
m = pd.DataFrame(d["markets"] + d["hist_markets"]).drop_duplicates("ticker")
hist = {x["ticker"] for x in d["hist_markets"]}
m = m[m.status == "finalized"].copy()
m["fight"] = m.event_ticker.str.split("-").str[1]
m["card"] = pd.to_datetime(m.fight.str[:7], format="%y%b%d")
m["ct"] = pd.to_datetime(m.close_time, format="ISO8601"); m["ot"] = pd.to_datetime(m.open_time, format="ISO8601")
m["st"] = pd.to_datetime(m.settlement_ts, format="ISO8601")
m["vol"] = m.volume_fp.astype(float)
m["src_hist"] = m.ticker.isin(hist)
ev = m.sort_values("ticker").groupby("event_ticker").agg(fight=("fight", "first"), card=("card", "first"), n=("ticker", "size"),
      tA=("ticker", "first"), tB=("ticker", "last"), nameA=("yes_sub_title", "first"), nameB=("yes_sub_title", "last"),
      resA=("result", "first"), resB=("result", "last"), ct=("ct", "max"), ctmin=("ct", "min"), st=("st", "max"),
      ot=("ot", "min"), vol=("vol", "sum"), src_hist=("src_hist", "max"),
      svA=("settlement_value_dollars", "first")).reset_index()
print("events", len(ev), "n markets/event", ev.n.value_counts().to_dict())
ev = ev[ev.n == 2]
fr = fights()
def match(r):
    c = fr[(fr.date - r.card).abs() <= pd.Timedelta(days=1)]
    na, nb = r.nameA, r.nameB
    ka, kb = nkey(na), nkey(nb)
    hit = c[((c.f1.map(nkey) == ka) & (c.f2.map(nkey) == kb)) | ((c.f1.map(nkey) == kb) & (c.f2.map(nkey) == ka))]
    how = "full"
    if len(hit) != 1:
        la, lb = norm(na)[-1], norm(nb)[-1]
        hit = c[c.apply(lambda x: {norm(x.f1)[-1], norm(x.f2)[-1]} == {la, lb}, axis=1)] if len(c) else c
        how = "last"
    if len(hit) != 1:
        ta = {t for t in norm(na) if len(t) >= 3}; tb = {t for t in norm(nb) if len(t) >= 3}
        def ok(x):
            s1 = set(norm(x.f1)); s2 = set(norm(x.f2))
            return (bool(ta & s1) and bool(tb & s2)) or (bool(ta & s2) and bool(tb & s1))
        hit = c[c.apply(ok, axis=1)] if len(c) else c
        how = "token"
    if len(hit) != 1: return pd.Series({"fid": None, "how": f"n{len(hit)}"})
    h = hit.iloc[0]
    ta = set(norm(na)); ufa = h.f1 if len(ta & set(norm(h.f1))) >= len(ta & set(norm(h.f2))) else h.f2
    ufb = h.f2 if ufa == h.f1 else h.f1
    return pd.Series({"fid": h.fid, "how": how, "ufA": ufa, "ufB": ufb, "EVENT": h.EVENT, "BOUT": h.BOUT,
                      "wclass": h.wclass, "sched5": h.sched5, "women": h.women, "date": h.date, "METHOD": h.METHOD.strip(),
                      "mclass": h.mclass, "winner": h.winner, "round": h["round"], "TIME": h.TIME, "draw": h.draw,
                      "DETAILS": h.DETAILS, "fmt": h["TIME FORMAT"]})
j = pd.concat([ev.reset_index(drop=True), ev.reset_index(drop=True).apply(match, axis=1)], axis=1)
print(j.how.value_counts().to_dict())
print("unmatched sample", j[j.fid.isna()][["event_ticker", "nameA", "nameB", "card"]].head(10).to_string())
j["dec"] = j.mclass == "DEC"
j.to_parquet(OUT / "fights.parquet")
dec = j[j.dec]
print("decisions", len(dec), "by month", dec.groupby(dec.card.dt.to_period("M")).size().to_dict())
print(dec.METHOD.value_counts().to_dict())
