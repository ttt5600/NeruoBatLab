"""source_shift detectors (frozen in notes/source_shift.md pre-registration) over census_markets.jsonl.
(a) late far strikes, (b) mid-event rules-template change + series template changes, (c) holds / >=85c favourite flips / odd settlements.
Writes det_a.csv, det_b_mid.csv, det_b_series.csv, det_c.csv, events.parquet. Usage: python analyze.py"""
import json, statistics
from pathlib import Path
import numpy as np, pandas as pd

H = Path(__file__).resolve().parent
ser = json.load(open(H.parent / "count_nowcast" / "series_live.json")); ser = ser["series"] if isinstance(ser, dict) else ser
SMETA = {x["ticker"]: (x["category"], x["title"], ";".join(s.get("name", "") for s in x.get("settlement_sources") or [])) for x in ser}
rows = []
for line in open(H / "census_markets.jsonl"):
    d = json.loads(line)
    for m in d["markets"]:
        m["series"] = d["series"]; rows.append(m)
df = pd.DataFrame(rows).drop_duplicates("t")
for c in ("cr", "op", "cl", "st", "ee", "le"):
    df[c] = pd.to_datetime(df[c], utc=True, errors="coerce", format="ISO8601")
for c in ("last", "prev", "vol", "sv"):
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["k"] = df.fs.where(df.fs.notna(), df.cs)
df["cat"] = df.series.map(lambda s: SMETA.get(s, ("?",))[0])
df["weather"] = df.series.str.match(r"^KX(HIGH|LOW|TEMP|RAIN|SNOW)")
print("markets", len(df), "events", df.e.nunique(), "series with markets", df.series.nunique())

# ---------------------------------------------------------------- (a) late far strikes
A = []
for e, g in df[df.k.notna()].groupby("e"):
    t0 = g.cr.min()
    ini, late = g[g.cr <= t0 + pd.Timedelta(hours=6)], g[g.cr > t0 + pd.Timedelta(hours=6)]
    if late.empty or len(ini) < 2:
        continue
    ks = sorted(set(ini.fs.dropna()) | set(ini.cs.dropna()))
    if len(ks) < 2:
        continue
    step = statistics.median(np.diff(ks)); span = ks[-1] - ks[0]
    if step <= 0:
        continue
    for _, m in late.iterrows():
        dist = max(ks[0] - m.k, m.k - ks[-1], 0)
        if dist > 2 * step:
            A.append(dict(series=g.series.iloc[0], event=e, ticker=m.t, strike=m.k, ini_lo=ks[0], ini_hi=ks[-1], step=step,
                          dist_spans=dist / span if span else np.nan, side="below" if m.k < ks[0] else "above",
                          created=m.cr, first_created=t0, close=m.cl, hours_before_close=(m.cl - m.cr).total_seconds() / 3600,
                          res=m.res, last=m.last, vol=m.vol, ev=g.ev.iloc[0]))
A = pd.DataFrame(A)
if len(A):
    A["cat"] = A.series.map(lambda s: SMETA.get(s, ("?",))[0])
    A.to_csv(H / "det_a.csv", index=False)
    ev_a = A.groupby(["series", "event"]).agg(n_far=("ticker", "size"), max_dist=("dist_spans", "max"), side=("side", "first"),
                                               created=("created", "min"), cat=("cat", "first"), ev=("ev", "first")).reset_index()
    print("\n(a) events with late far strikes:", len(ev_a), "in", ev_a.series.nunique(), "series")
    print(ev_a.groupby("cat").size().sort_values(ascending=False).to_string())

# ---------------------------------------------------------------- (b) rules template changes
B = []
for e, g in df.groupby("e"):
    t0 = g.cr.min()
    ini, late = g[g.cr <= t0 + pd.Timedelta(hours=6)], g[g.cr > t0 + pd.Timedelta(hours=6)]
    if late.empty:
        continue
    a, b = set(zip(ini.hp, ini.hs)), set(zip(late.hp, late.hs))
    if b - a:
        B.append(dict(series=g.series.iloc[0], event=e, ini_tpl=len(a), late_new=len(b - a), sec_changed=bool(set(late.hs) - set(ini.hs)),
                      prim_changed=bool(set(late.hp) - set(ini.hp)), late_created=late.cr.min(), close=g.cl.max(), n=len(g)))
B = pd.DataFrame(B)
if len(B):
    B["cat"] = B.series.map(lambda s: SMETA.get(s, ("?",))[0]); B.to_csv(H / "det_b_mid.csv", index=False)
    print("\n(b) events whose later-listed markets carry a new rules template:", len(B), "| secondary changed:", int(B.sec_changed.sum()))
# series-level: modal rules_secondary template per event, in event order
S = []
for s, g in df.groupby("series"):
    ev = g.groupby("e").agg(t0=("cr", "min"), hs=("hs", lambda x: x.mode().iloc[0])).sort_values("t0")
    if len(ev) < 4:
        continue
    prev = None
    for e, r in ev.iterrows():
        if prev is not None and r.hs != prev[1]:
            S.append(dict(series=s, from_event=prev[0], to_event=e, date=r.t0, old=prev[1], new=r.hs))
        prev = (e, r.hs)
S = pd.DataFrame(S)
if len(S):
    S["cat"] = S.series.map(lambda s: SMETA.get(s, ("?",))[0]); S.to_csv(H / "det_b_series.csv", index=False)
    print("(b) series-level rules_secondary template changes:", len(S), "in", S.series.nunique(), "series")

# ---------------------------------------------------------------- (c) holds / flips / odd settlements
E = df.groupby("e").agg(series=("series", "first"), cat=("cat", "first"), weather=("weather", "first"), close=("cl", "max"),
                        settle=("st", "max"), n=("t", "size"), vol=("vol", "sum"), ev=("ev", "first")).reset_index()
E["delay_h"] = (E.settle - E.close).dt.total_seconds() / 3600
med = E.groupby("series").delay_h.median()
E["series_med_h"] = E.series.map(med)
E["hold"] = E.delay_h > np.maximum(48, E.series_med_h + 24)
liq = df[df.vol >= 100].copy()
fav = liq.sort_values("last", ascending=False).groupby("e").head(1).set_index("e")
E = E.join(fav[["t", "last", "res"]].rename(columns={"t": "fav", "last": "fav_last", "res": "fav_res"}), on="e")
E["flip_fav"] = (E.fav_last >= 0.85) & (E.fav_res == "no")
cheap_win = liq[(liq["last"] <= 0.15) & (liq.res == "yes")].groupby("e").agg(cheap_winner=("t", "first"), cheap_last=("last", "first"))
E = E.join(cheap_win, on="e")
odd = df[(~df.res.isin(["yes", "no"])) | (df.sv.notna() & ~df.sv.isin([0.0, 1.0]))].groupby("e").agg(
    odd_n=("t", "size"), odd_res=("res", lambda x: ",".join(sorted(set(map(str, x))))), odd_sv=("sv", lambda x: ",".join(sorted(set(map(str, x))))))
E = E.join(odd, on="e")
E["odd"] = E.odd_n.notna()
E.to_parquet(H / "events.parquet")
C = E[E.hold | E.flip_fav | E.cheap_winner.notna() | E.odd].copy()
C.to_csv(H / "det_c.csv", index=False)
print(f"\n(c) events {len(E)} | holds {int(E.hold.sum())} | fav>=.85 lost {int(E.flip_fav.sum())} (of {int((E.fav_last >= .85).sum())} with fav>=.85)"
      f" | <=.15 winner {int(E.cheap_winner.notna().sum())} | odd settlement {int(E.odd.sum())}")
nw = E[~E.weather]
print(f"    non-weather: events {len(nw)} | holds {int(nw.hold.sum())} | fav lost {int(nw.flip_fav.sum())} of {int((nw.fav_last >= .85).sum())}"
      f" | cheap winner {int(nw.cheap_winner.notna().sum())} | odd {int(nw.odd.sum())}")
print("    by category (non-weather):")
print(nw.groupby("cat").agg(events=("e", "size"), holds=("hold", "sum"), fav85=("fav_last", lambda x: (x >= .85).sum()),
                            fav_lost=("flip_fav", "sum"), odd=("odd", "sum")).sort_values("events", ascending=False).to_string())
