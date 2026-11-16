"""tropical_nhc: Kalshi tropical/hurricane series vs NHC official products (notes/tropical_nhc.md). Read-only GETs.

    python scripts/tropical_nhc.py census     # market metadata for every hurricane/tropical series (result column split off)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from kx_throttle import kget  # noqa: E402

OUT = ROOT / "results" / "tropical_nhc"
OUT.mkdir(parents=True, exist_ok=True)
PAT = re.compile(r"hurric|tropic|landfall|cyclone|namedstorm|HURC|TROPSTORM", re.I)
EXCL = re.compile(r"EMERGENCY|GSTORM|RAINS|SNOWSTORM|GRAMBTLA")
KEEP = ["ticker", "event_ticker", "market_type", "open_time", "close_time", "expected_expiration_time",
        "settlement_ts", "status", "title", "yes_sub_title", "subtitle", "rules_primary", "volume", "volume_fp",
        "open_interest", "result"]


def series():
    ser = json.load(open(ROOT / "results" / "lx2_tail_no_carry" / "series.json"))
    return [s for s in ser if PAT.search(s["ticker"] + " " + (s.get("title") or "")) and not EXCL.search(s["ticker"])
            and s.get("category") == "Climate and Weather"]


def page(path, st):
    rows, cur = [], None
    while True:
        d = kget(path, {"series_ticker": st, "limit": 1000, **({"cursor": cur} if cur else {})})
        if not d:
            return rows
        ms = d.get("markets", []) or []
        rows += [{k: m.get(k) for k in KEEP} | {"series": st, "src": path} for m in ms]
        cur = d.get("cursor")
        if not cur or not ms:
            return rows


def cmd_census():
    ser = series()
    json.dump(ser, open(OUT / "series.json", "w"), indent=1)
    rows = []
    for s in ser:
        r = page("/historical/markets", s["ticker"]) + page("/markets", s["ticker"])
        print(s["ticker"], len(r), flush=True)
        rows += r
    df = pd.DataFrame(rows).drop_duplicates("ticker")
    # outcomes kept in a separate file, not looked at until the prereg is written
    df[["ticker", "result"]].to_parquet(OUT / "results_SEPARATE.parquet", index=False)
    df.drop(columns=["result"]).to_parquet(OUT / "markets.parquet", index=False)
    print("markets", len(df))


DATA = ROOT / "data" / "tropical_nhc"
HKT = {"T1": 64, "T2": 83, "T3": 96, "T4": 113, "T5": 137}
H = 3600


def _sec(x):
    t = pd.to_datetime(x, utc=True, format="ISO8601")
    return int((t - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1))  # unit-safe (traps)


def dev_markets():
    d = pd.read_parquet(OUT / "markets.parquet")
    d = d[d.series.isin(["KXHURCAT", "HURCAT"])].copy()
    d["thr"] = d.ticker.str.split("-").str[-1]
    d = d[d.thr.isin(list(HKT))]
    d["ot"] = pd.to_datetime(d.open_time, utc=True, format="ISO8601")
    first = d.groupby("event_ticker").ot.transform("min")
    d = d[first.dt.year <= 2024].copy()  # DEV only; 2025 excluded, 2026 sealed -- never built
    d["name"] = d.title.str.extract(r"Will (?:Storm )?([A-Za-z]+) become", expand=False).str.upper()
    d["year"] = first[d.index].dt.year
    d["storm"] = d.name + "-" + d.year.astype(str)
    d["date"] = first[d.index].dt.tz_localize(None).dt.normalize()
    d["H_kt"] = d.thr.map(HKT)
    return d


def storm_ids():
    rows = []
    for ln in open(DATA / "storm.table"):
        p = [x.strip() for x in ln.split(",")]
        if len(p) > 20 and p[8].isdigit():
            rows.append({"name": p[0], "basin": p[1], "year": int(p[8]), "id": p[20]})
    return pd.DataFrame(rows)


def ofcl(atcf_id):
    import gzip
    import requests
    bb, nn, yy = atcf_id[:2].lower(), atcf_id[2:4], atcf_id[4:]
    p = DATA / "atcf" / f"a{bb}{nn}{yy}.dat.gz"
    if not p.exists():
        r = requests.get(f"https://ftp.nhc.noaa.gov/atcf/archive/{yy}/a{bb}{nn}{yy}.dat.gz", timeout=120)
        r.raise_for_status()
        p.write_bytes(r.content)
    rows = []
    for ln in gzip.open(p, "rt", errors="replace"):
        q = [x.strip() for x in ln.split(",")]
        if len(q) > 8 and q[4] == "OFCL" and q[8].lstrip("-").isdigit():
            rows.append((q[2], int(q[5]), int(q[8])))
    o = pd.DataFrame(rows, columns=["t0", "tau", "v"])
    o = o[o.v > 0]
    o["t0s"] = [_sec(pd.Timestamp(f"{x[:4]}-{x[4:6]}-{x[6:8]} {x[8:10]}:00", tz="UTC")) for x in o.t0]
    g = o.groupby("t0s").agg(vmax=("v", "max"), vnow=("v", lambda s: s[o.loc[s.index, "tau"] <= 3].max()))
    return g.reset_index()


def candles(tk, a, b):
    out, s = [], a
    while s < b:
        e = min(b, s + 4000 * H)
        d = kget(f"/historical/markets/{tk}/candlesticks", {"start_ts": s, "end_ts": e, "period_interval": 60}) or {}
        for c in d.get("candlesticks", []) or []:
            out.append((c["end_period_ts"], float(c["yes_bid"]["close"] or "nan"), float(c["yes_ask"]["close"] or "nan"),
                        float(c.get("volume") or 0)))
        s = e
    return pd.DataFrame(out, columns=["ts", "yb", "ya", "vol"]).drop_duplicates("ts").sort_values("ts")


def cmd_build():
    import numpy as np
    d = dev_markets()
    ids = storm_ids()
    ofs, miss = {}, []
    for st, g in d.groupby("storm"):
        m = ids[(ids.name == g.name.iloc[0]) & (ids.year == g.year.iloc[0])].sort_values("basin")  # AL < CP < EP
        if m.empty:
            miss.append(st)
            continue
        ofs[st] = (m.id.iloc[0], ofcl(m.id.iloc[0]))
    print("storms", d.storm.nunique(), "matched", len(ofs), "no ATCF match (never formed / unnamed):", miss, flush=True)
    rows, cs = [], []
    for _, m in d[d.storm.isin(ofs)].iterrows():
        aid, o = ofs[m.storm]
        a, b = _sec(m.open_time), _sec(m.close_time)
        c = candles(m.ticker, a - H, b + H)
        c["ticker"] = m.ticker
        cs.append(c)
        o = o.sort_values("t0s").reset_index(drop=True)
        o["vnow_cum"] = o.vnow.cummax()
        dec = [("HC1", t0, t0 + 3 * H, t0 + 5 * H, m.H_kt - vm) for t0, vm in zip(o.t0s, o.vmax)]
        t = o.t0s.to_numpy()
        gaps = np.append(np.diff(t), np.inf)
        i = int(np.argmax(gaps > 30 * H))  # first advisory followed by >= 30 h of silence (observable at t0 + 27 h)
        r = t[i] + 27 * H
        dec.append(("HC2", t[i], r, r + 2 * H, m.H_kt - o.vnow_cum.iloc[i]))
        for tr, t0, r, dd, gap in dec:
            if not (a <= dd < b):
                continue
            q = c[(c.ts <= dd) & (c.ts >= dd - 7 * 24 * H)]
            qb, qa, qe = (q.yb.iloc[-1], q.ya.iloc[-1], q.ts.iloc[-1]) if len(q) else (np.nan, np.nan, -1)
            rows.append({"ticker": m.ticker, "event_ticker": m.storm, "ev_orig": m.event_ticker, "atcf": aid,
                         "thr": m.thr, "H_kt": m.H_kt, "trial": tr, "t0": t0, "r": r, "d": dd, "gap": gap,
                         "yes_bid": qb, "yes_ask": qa, "q_end": qe, "close_time": m.close_time,
                         "date": m.date})
    f = pd.DataFrame(rows)
    res = pd.read_parquet(OUT / "results_SEPARATE.parquet")
    f = f.merge(res, on="ticker", how="left")  # hidden by the vault; the rule never sees it
    DATA.mkdir(parents=True, exist_ok=True)
    f.to_parquet(DATA / "frame.parquet", index=False)
    pd.concat(cs).to_parquet(DATA / "candles.parquet", index=False)
    print("frame rows", len(f), "tickers", f.ticker.nunique(), "-> data/tropical_nhc/frame.parquet")


def cmd_gate():
    d = dev_markets()
    ids = storm_ids()
    res = pd.read_parquet(OUT / "results_SEPARATE.parquet").set_index("ticker").result
    rows = []
    for st, g in d.groupby("storm"):
        m = ids[(ids.name == g.name.iloc[0]) & (ids.year == g.year.iloc[0])].sort_values("basin")
        vpk = ofcl(m.id.iloc[0]).vnow.max() if len(m) else 0
        for _, x in g.iterrows():
            rows.append({"ticker": x.ticker, "storm": st, "H_kt": x.H_kt, "vpk": vpk, "pred": "yes" if vpk >= x.H_kt else "no",
                         "result": res.get(x.ticker)})
    g = pd.DataFrame(rows)
    g["agree"] = g.pred == g.result
    g.to_csv(OUT / "gate.csv", index=False)
    print("gate agreement", round(g.agree.mean(), 4), "n", len(g), "results:", g.result.value_counts().to_dict())
    print(g[~g.agree].to_string())


def _score(b, f):
    import numpy as np
    from scipy.stats import beta
    x = b.assign(ticker=b.bet_id.str.split("|").str[0]).merge(
        f.drop_duplicates("ticker")[["ticker", "event_ticker", "thr", "result"]], on="ticker")
    p = x.price.to_numpy(float)
    x["fee"] = 0.07 * p * (1 - p)
    x["fee_1ct"] = np.ceil(np.round(x.fee * 100, 6)) / 100
    x["cost"] = p + x.fee
    x["won"] = (x.result == "no").astype(float)
    x["pnl"] = x.won - x.cost
    x["pnl_1ct"] = x.won - p - x.fee_1ct
    g = x.groupby("event_ticker").agg(P=("pnl", "sum"), P1=("pnl_1ct", "sum"), C=("cost", "sum"))
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(g), (10000, len(g)))
    out = {"n": int(len(x)), "storms": int(len(g)), "losses": int((x.won == 0).sum()),
           "mean_price": round(float(p.mean()), 4), "roi": round(float(g.P.sum() / g.C.sum()), 5)}
    for k in ("P", "P1"):
        r = g[k].to_numpy()[idx].sum(1) / g.C.to_numpy()[idx].sum(1)
        out[f"ci_{k}"] = [round(float(np.percentile(r, 2.5)), 5), round(float(np.percentile(r, 97.5)), 5)]
    out["roi_1ct"] = round(float(g.P1.sum() / g.C.sum()), 5)
    L, n = out["losses"], out["n"]
    out["loss_rate_cp95"] = [round(float(beta.ppf(.025, L, n - L + 1)) if L else 0.0, 4),
                             round(float(beta.ppf(.975, L + 1, n - L)), 4)]
    out["breakeven_loss_rate"] = round(float(1 - x.cost.mean()), 4)
    tot = g.P.sum()
    out["top_storm_share"] = round(float(g.P.max() / tot), 3) if tot > 0 else None
    out["by_thr"] = x.groupby("thr").agg(n=("pnl", "size"), pnl=("pnl", "sum"), px=("price", "mean")).round(3).to_dict("index")
    return out, x


def cmd_dev():
    import importlib
    f = pd.read_parquet(DATA / "frame.parquet")
    f = f[pd.to_datetime(f.date) <= "2024-12-31"]
    R = importlib.import_module("edgelab.rules.hurcat_nhc")
    rule_in = f.drop(columns=["result"])
    summ, allb = {}, []
    for name, b in [("HC1_m20", R.hc1(rule_in, {"margin": 20})), ("HC2", R.hc2(rule_in)),
                    ("HC1_m10_sens", R.hc1(rule_in, {"margin": 10})), ("HC1_m30_sens", R.hc1(rule_in, {"margin": 30}))]:
        if not len(b):
            summ[name] = {"n": 0}
            continue
        s, x = _score(b, f)
        summ[name] = s
        allb.append(x.assign(trial=name))
    json.dump(summ, open(OUT / "dev_summary.json", "w"), indent=1, default=str)
    pd.concat(allb).to_csv(OUT / "bets_dev.csv", index=False)
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    {"census": cmd_census, "build": cmd_build, "gate": cmd_gate, "dev": cmd_dev}[sys.argv[1]]()
