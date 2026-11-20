"""Descriptive calibration of Kalshi derived ladders at T = start - 60 min. DEV ONLY (earliest 70% of events by start per
series; latest 30% = sealed, never loaded here). usage: calib.py S1 S2 ... -> calib_cells.csv (+ printed tables)."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path.home() / "agent-harness" / "lib"))
import traps
D = Path(__file__).resolve().parent; EV = D / "ev"
SER = json.load(open(D / "series_sports.json")); MULT = {x["ticker"]: float(x.get("fee_multiplier") or 1) for x in SER}
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
def fee(p, mult):  # 10-lot quadratic taker, cent-rounded per order, per contract
    return np.ceil(np.round(100 * 0.07 * mult * 10 * p * (1 - p), 9)) / 1000
def fill_start(st):  # same rule as fetch_ev.py: occ-3h where ESPN unmatched, if validated on >= 95% of matched events
    agree = (st.off_min.dropna().abs() <= 5).mean() if st.off_min.notna().sum() >= 20 else 0.0
    if agree >= 0.95:
        f = st.start_utc.isna() & st.occ_m3.notna(); st.loc[f, "start_utc"] = st.loc[f, "occ_m3"]
    return st
def split(st):
    st = fill_start(st.copy())
    st = st[st.start_utc.notna()].sort_values("start_utc").reset_index(drop=True)
    cut = st.start_utc.iloc[int(np.floor(0.7 * len(st)))]
    return st, cut
def load(s, part="dev"):
    m = pd.read_parquet(EV / f"markets_{s}.parquet"); st = pd.read_parquet(EV / f"starts_{s}.parquet")
    st = st[st.event_ticker.isin(set(m[m.src == "live"].event_ticker))]   # quote-bearing (post-cutoff) events only
    st, cut = split(st)
    dv = st[st.start_utc < cut]
    cutA = dv.start_utc.iloc[int(np.floor(0.5 * len(dv)))] if len(dv) else cut
    keep = {"dev": dv, "devA": dv[dv.start_utc < cutA], "devB": dv[dv.start_utc >= cutA]}[part]
    rows = []
    for l in (EV / f"q_{s}.jsonl").open():
        j = json.loads(l)
        if j["event"] not in set(keep.event_ticker): continue
        for tk, q in j["q"].items():
            if q: rows.append(dict(ticker=tk, event_ticker=j["event"], T=j["T"], **q))
    q = pd.DataFrame(rows)
    if q.empty: return q, cut
    traps.check_epoch_seconds(q["T"], "T")
    d = m.merge(q, on=["ticker", "event_ticker"], how="inner", validate="one_to_one")
    d = d.merge(keep[["event_ticker", "start_utc"]], on="event_ticker", how="left", validate="many_to_one")
    for c in ["ask", "bid", "oi", "volume_fp", "floor_strike"]: d[c] = pd.to_numeric(d[c], errors="coerce")
    assert (d.end <= d["T"]).all()
    assert ((d["T"] - (pd.to_datetime(d.start_utc, utc=True) - EPOCH) // pd.Timedelta(seconds=1)) == -3600).all()
    ot = (pd.to_datetime(d.open_time, utc=True, format="mixed") - EPOCH) // pd.Timedelta(seconds=1)
    d["pre_open"] = ot <= d["T"]
    d["y"] = (d.result == "yes").astype(int); d["series"] = s; d["mult"] = MULT.get(s, 1.0)
    d["valid"] = d.pre_open & (d.bid > 0) & (d.ask < 1) & (d.ask > d.bid)
    d["stale_min"] = (d["T"] - d.end) / 60
    return d, cut
def boot_mean(x, g, n=2000, seed=0):
    df = pd.DataFrame({"x": x, "g": g}).groupby("g").x.agg(["sum", "size"])
    S, N = df["sum"].to_numpy(), df["size"].to_numpy(); rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(df), (n, len(df))); r = S[idx].sum(1) / N[idx].sum(1)
    return np.percentile(r, [2.5, 97.5])
def cells(d, by):
    out = []
    for k, g in d.groupby(by, observed=True):
        if len(g) < 30 or g.event_ticker.nunique() < 15: continue
        fy = fee(g.ask, g.mult); fn = fee(1 - g.bid, g.mult)
        ryes = (g.y - g.ask - fy) / (g.ask + fy); rno = ((1 - g.y) - (1 - g.bid) - fn) / (1 - g.bid + fn)
        lo, hi = boot_mean(g.y - (g.ask + g.bid) / 2, g.event_ticker)
        ly, hy = boot_mean(ryes, g.event_ticker, seed=1); ln, hn = boot_mean(rno, g.event_ticker, seed=2)
        out.append(dict(series=g.series.iloc[0], by=by, cell=str(k), n=len(g), n_ev=g.event_ticker.nunique(), y=g.y.mean(),
                        mid=((g.ask + g.bid) / 2).mean(), ask=g.ask.mean(), bid=g.bid.mean(), spread=(g.ask - g.bid).median(),
                        gap=(g.y - (g.ask + g.bid) / 2).mean(), gap_lo=lo, gap_hi=hi,
                        roi_yes=ryes.mean(), ry_lo=ly, ry_hi=hy, roi_no=rno.mean(), rn_lo=ln, rn_hi=hn))
    return out
if __name__ == "__main__":
    allc = []
    part = "devA"
    for s in [a for a in sys.argv[1:] if not a.startswith("--")]:
        if s.startswith("POOL:"):   # pooled across series (e.g. soccer leagues); label = the arg
            fr = [load(x, part)[0] for x in s[5:].split("+")]
            d = pd.concat([f for f in fr if len(f)], ignore_index=True); d["series"] = s[5:].split("+")[0] + "_POOL"; cut = "per-series"
        else:
            d, cut = load(s, part)
        if d.empty: print(s, "no data"); continue
        v = d[d.valid].copy()
        v["mid"] = (v.ask + v.bid) / 2
        v["mb"] = pd.cut(v.mid, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
        # relative rung: strike rank within (event, side-group) minus rank of the rung whose mid is nearest .5
        grp = v.event_ticker + "|" + v.title.str.extract(r"^([^:]*?)(?: over| \d|:)", expand=False).fillna("")
        v["grp"] = grp
        v["rk"] = v.groupby("grp").floor_strike.rank(method="first")
        c50 = v.assign(dd=(v.mid - .5).abs()).sort_values("dd").drop_duplicates("grp").set_index("grp").rk
        v["rel"] = (v.rk - v.grp.map(c50)).clip(-4, 4)
        print(f"\n=== {s} [{part}]: events {d.event_ticker.nunique()} (cut {cut}), markets {len(d)}, valid {len(v)} "
              f"(pre-open {d.pre_open.mean():.2f}, bid>0 {(d.bid>0).mean():.2f}); stale median {v.stale_min.median():.0f} min; "
              f"spread median {(v.ask-v.bid).median():.3f}; vol6h>0 {(v.vol6h>0).mean():.2f}")
        c = cells(v, "mb") + (cells(v, "rel") if v.floor_strike.notna().mean() > .9 else [])
        allc += c
        t = pd.DataFrame(c)
        if len(t): print(t[["by", "cell", "n", "n_ev", "y", "mid", "spread", "gap", "gap_lo", "gap_hi", "roi_yes", "ry_lo", "ry_hi", "roi_no", "rn_lo", "rn_hi"]].round(3).to_string(index=False))
    out = pd.DataFrame(allc)
    f = D / "calib_cells.csv"
    if f.exists() and f.stat().st_size > 10:
        old = pd.read_csv(f); out = pd.concat([old[~old.series.isin(out.series.unique())], out]) if len(out) else old
    out.to_csv(f, index=False); print("\ncells total", len(out))
