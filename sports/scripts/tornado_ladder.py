"""Builder + gate for TC1 (tc1_tornado_maker-6dcaab12; rule edgelab/rules/tornado_count.py). Read-only public APIs.

    python scripts/tornado_ladder.py spc     # SPC NAT_<MON>.json 2005..now -> data/phys_ladders/spc_months.json + cap table
    python scripts/tornado_ladder.py gate    # PRE-REGISTERED: SPC month total today vs Kalshi expiration_value, DEV months
    python scripts/tornado_ladder.py frame   # only after the gate passes: candles, tape -> tornado_frame/_tape.parquet
"""
from __future__ import annotations

import calendar
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import phys_ladders as P  # noqa: E402
from edgelab.rules import tornado_count as TC  # noqa: E402

OUT = ROOT / "data" / "phys_ladders"
SPC = OUT / "spc_months.json"
GATE = ROOT / "results" / "tornado_gate.json"
MONS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
ET = ZoneInfo("America/New_York")
CLIM = range(2005, 2023)
DEV = ("2023-07-01", "2025-09-30")
MATCH_SHARE = 0.90


def _one(ym):
    y, m = ym
    d = P._get(f"https://www.spc.noaa.gov/climo/summary/{y}/ruf/NAT/NAT_{MONS[m - 1]}.json")
    if not d:
        return None
    return {"y": y, "m": m, "torn": d["torn"], "daily": {k: v["torn"] for k, v in d["daily"].items()}}


def spc():
    OUT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    yms = [(y, m) for y in range(2005, now.year + 1) for m in range(1, 13) if (y, m) <= (now.year, now.month)]
    with ThreadPoolExecutor(8) as ex:
        rows = [r for r in ex.map(_one, yms) if r]
    for r in rows:
        if r["torn"] != sum(r["daily"].values()):
            raise SystemExit(f"SPC month total != sum of days: {r['y']}-{r['m']}")
    json.dump(rows, open(SPC, "w"))
    have = {(r["y"], r["m"]) for r in rows}
    print(len(rows), "of", len(yms), "months; missing", sorted(set(yms) - have)[:20])


def load_spc():
    return {(r["y"], r["m"]): r for r in json.load(open(SPC))}


def caps(months):
    """cap[(m, L)] = max sum over any L consecutive report days of calendar month m, over 2005..2022."""
    out = {}
    for m in range(1, 13):
        ser = []
        for y in CLIM:
            r = months.get((y, m))
            if r:
                ser.append([r["daily"][k] for k in sorted(r["daily"])])
        if not ser:
            raise SystemExit(f"no climatology for month {m}")
        for L in range(1, 32):
            best = 0
            for s in ser:
                for i in range(len(s)):
                    best = max(best, sum(s[i:i + L]))
            out[(m, L)] = best
    return out


def markets():
    mk = P.markets("KXTORNADO")
    mm = mk["event_ticker"].str.extract(r"-(\d\d)([A-Z]{3})$")
    mk = mk[mm[1].isin(MONS)].copy()
    mk["y"], mk["m"] = 2000 + mm.loc[mk.index, 0].astype(int), mm.loc[mk.index, 1].map(lambda s: MONS.index(s) + 1)
    mk["date"] = [f"{y}-{m:02d}-01" for y, m in zip(mk["y"], mk["m"])]
    return mk


def gate():
    """Dev months only. Prints the match, never rung-level results."""
    months = load_spc()
    mk = markets()
    dev = mk[(mk["date"] >= DEV[0]) & (mk["date"] <= DEV[1])]
    per = dev.groupby("event_ticker").agg(y=("y", "first"), m=("m", "first"), ev=("expiration_value", "first"),
                                          nv=("expiration_value", "nunique"))
    if (per["nv"] > 1).any():
        print("WARNING: events with several expiration values:", list(per.index[per["nv"] > 1]))
    rows = []
    for ev, r in per.iterrows():
        sp = months.get((int(r["y"]), int(r["m"])))
        if sp is None or not np.isfinite(r["ev"]):
            rows.append({"event": ev, "spc": None, "kalshi": None if not np.isfinite(r["ev"]) else float(r["ev"]),
                         "match": False, "why": "missing"})
            continue
        tol = max(3.0, 0.03 * float(r["ev"]))
        rows.append({"event": ev, "spc": sp["torn"], "kalshi": float(r["ev"]), "diff": sp["torn"] - float(r["ev"]),
                     "tol": tol, "match": abs(sp["torn"] - float(r["ev"])) <= tol})
    df = pd.DataFrame(rows).sort_values("event")
    share = float(df["match"].mean())
    res = {"trial": "tc1_tornado_maker-6dcaab12", "n_months": int(len(df)), "n_match": int(df["match"].sum()),
           "share": share, "bar": MATCH_SHARE, "passed": bool(share >= MATCH_SHARE and len(df) > 0),
           "median_diff": float(pd.to_numeric(df.get("diff")).median()),
           "rows": df.to_dict("records"),
           "_sources": [{"path": str(SPC)}], "_generated_by": "scripts/tornado_ladder.py gate"}
    GATE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(GATE, "w"), indent=1, default=str)
    print(df.to_string(index=False))
    print(f"GATE {'PASS' if res['passed'] else 'FAIL'}: {res['n_match']}/{res['n_months']} = {share:.3f} (bar {MATCH_SHARE})")
    return res["passed"]


def frame():
    if not json.load(open(GATE)).get("passed"):
        raise SystemExit("gate not passed; frame refused")
    months = load_spc()
    C = caps(months)
    mk = markets()
    rows, cand = [], {}
    for ev, g in mk.groupby("event_ticker"):
        y, m = int(g["y"].iloc[0]), int(g["m"].iloc[0])
        sp = months.get((y, m))
        if sp is None:
            print("no SPC", ev)
            continue
        last = calendar.monthrange(y, m)[1]
        me = P._month_bounds(y, m)[1]
        t_open = int(g["open_ts"].min())
        cs = P.candles("KXTORNADO", ev, list(g["ticker"]), t_open - 3600, me + 3600)
        keyed = {int(k[-2:]): v for k, v in sp["daily"].items()}
        for d in range(1, last + 1):
            day = datetime(y, m, d, 16, tzinfo=timezone.utc)
            T = int(day.timestamp())
            if T < t_open:
                continue
            c = sum(v for dd, v in keyed.items() if dd <= d - 1)
            L = last - d + 1
            cap = C[(m, L)]
            its = int(datetime(y, m, d, 12, tzinfo=timezone.utc).timestamp())
            for r in g.itertuples():
                if T >= r.close_ts or not np.isfinite(r.K):
                    continue
                yb, ya = P.quote(cs.get(r.ticker, []), T)
                mb, ma = P.quote(cs.get(r.ticker, []), me - 3600)
                rows.append({"ticker": r.ticker, "event_ticker": ev, "date": f"{y}-{m:02d}-01", "K": r.K,
                             "strike_type": r.strike_type, "close_ts": r.close_ts, "month_end_ts": me, "T": T,
                             "c": c, "L": L, "cap": cap, "inputs_ts": its, "yes_bid": yb, "yes_ask": ya,
                             "mark_yes_bid": mb, "mark_yes_ask": ma, "result": r.result,
                             "expiration_value": r.expiration_value, "volume": r.volume})
                if TC.input_true(np.array([r.K], float), c, np.array([cap], float), np.array([r.strike_type]))[0]:
                    cand[r.ticker] = min(cand.get(r.ticker, T), T)
        print(ev, len(g), "rungs", flush=True)
    fr = pd.DataFrame(rows)
    if len(fr):
        bad = fr[(fr["yes_ask"] > 1) | (fr["yes_bid"] > 1)]
        if len(bad):
            raise SystemExit(f"quote scale check failed: {len(bad)} rows above $1")
    tp, end = [], fr.groupby("ticker")["month_end_ts"].first().to_dict()
    for tk, t0 in cand.items():
        tp += P.tape(tk, t0, end[tk] + 3600)
    tape = pd.DataFrame(tp, columns=["ticker", "ts", "yes_price", "no_price", "count", "trade_id"]).drop_duplicates("trade_id")
    fr.to_parquet(OUT / "tornado_frame.parquet")
    tape.to_parquet(OUT / "tornado_tape.parquet")
    print(len(fr), "rows;", fr["ticker"].nunique(), "tickers;", len(cand), "with input true;", len(tape), "trades")


if __name__ == "__main__":
    {"spc": spc, "gate": gate, "frame": frame}[sys.argv[1]]()
