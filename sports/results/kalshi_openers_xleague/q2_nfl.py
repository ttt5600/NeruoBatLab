"""Q2 K-NFL (pre-registered, notes/kalshi_openers_xleague.md): L33 logger picks re-priced on Kalshi at the decision tick.
S = Kalshi strike (home-margin / total axis) nearest the Pinnacle open; re-check L33 gap at S (spread 2.0, total 2.5);
price = last 1-min candle ending <= tick (<= 60 min old), ask + exact taker fee; graded on final vs S; CLV = Pinnacle
close vs S, signed. Usage: q2_nfl.py dev|forward  (dev = decision tick < 2026-11-12)."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from kx import get, side_x  # noqa: E402

D = Path.home() / ".claude/quant-kalshi/data"
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
CUT = pd.Timestamp("2026-11-12", tz="UTC")
SER = {"total": "KXNFLTOTAL", "spread": "KXNFLSPREAD"}
TH = {"total": 2.5, "spread": 2.0}
CODE = {"WAS": "WAS", "LAR": "LA", "JAX": "JAC"}  # logger code -> Kalshi ticker code (checked against tickers)


def fee(p, n=100):
    return math.ceil(round(n * 0.07 * p * (1 - p) * 100, 6)) / 100 / n


def ts(x):
    return int((pd.Timestamp(x) - EPOCH) // pd.Timedelta(seconds=1))


def find_event(mk, date, a, h, tight):
    for ev, m in tight.items():
        if m["mk"] == mk and m["date"] == date and set(m["teams"]) == {a, h}:
            return ev
    return None


def main(split):
    rows = [json.loads(l) for l in open(D / "open_lines_nfl_picks.jsonl")]
    picks = [r for r in rows if r["type"] == "pick"]
    settle = {(r["key"], r["market"]): r for r in rows if r["type"] == "settle"}
    tight = json.load(open(D / "open_lines_nfl_tight.json"))["markets"]
    cache = OUT / "cache"
    res = []
    for p in picks:
        T = pd.Timestamp(p["t"])
        if (split == "dev") != (T < CUT):
            continue
        mk, k = p["market"], p["key"]
        date, teams = k.split("|")
        a, h = teams.split("@")
        ev = find_event(mk, date, a, h, tight)
        base = dict(key=k, mk=mk, pick=p["pick"], T=T, open_line=p["open_line"], model=p["model"])
        if not ev:
            res.append({**base, "status": "no event"}); continue
        cf = cache / f"q2nfl_{ev}_{ts(T)}.json"
        if cf.exists():
            q = json.loads(cf.read_text())
        else:
            ms = get("/markets", {"event_ticker": ev, "limit": 200}).get("markets", [])
            x_open = p["open_line"] if mk == "total" else -p["open_line"]
            q = []
            for m in ms:
                if m.get("floor_strike") is None:
                    continue
                x, sg = side_x(ev, m["ticker"], m["floor_strike"], mk)
                if abs(x - x_open) > 3.01:
                    continue
                j = get(f"/series/{SER[mk]}/markets/{m['ticker']}/candlesticks",
                        {"start_ts": ts(T) - 3600, "end_ts": ts(T), "period_interval": 1})
                cs = [c for c in j.get("candlesticks", []) if c["end_period_ts"] <= ts(T)]
                if not cs:
                    continue
                c = cs[-1]
                yb, ya = c.get("yes_bid") or {}, c.get("yes_ask") or {}
                b = yb.get("close_dollars", yb.get("close")); aa = ya.get("close_dollars", ya.get("close"))
                if b is None or aa is None:
                    continue
                b, aa = float(b), float(aa)
                b, aa = (b / 100 if b > 1 else b), (aa / 100 if aa > 1 else aa)
                q.append(dict(ticker=m["ticker"], x=x, sg=sg, bid=b, ask=aa, end=c["end_period_ts"], result=m.get("result")))
            cf.write_text(json.dumps(q))
        q = [z for z in q if z["bid"] > 0 and z["ask"] < 1]
        if not q:
            res.append({**base, "status": "no quote"}); continue
        x_open = p["open_line"] if mk == "total" else -p["open_line"]
        dmin = min(abs(z["x"] - x_open) for z in q)
        cand = [z for z in q if abs(abs(z["x"] - x_open) - dmin) < 1e-9]
        up = p["pick"] in ("over", "home")
        z = min(cand, key=lambda z: z["x"]) if up else max(cand, key=lambda z: z["x"])  # favorable strike for our side
        gap = p["model"] - z["x"]
        ok = gap >= TH[mk] if up else gap <= -TH[mk]
        # want P(axis > x) if up: YES if sg>0 else NO ; down: NO if sg>0 else YES
        buy_yes = (up and z["sg"] > 0) or ((not up) and z["sg"] < 0)
        ask = z["ask"] if buy_yes else 1 - z["bid"]
        cost = ask + fee(ask)
        s = settle.get((k, mk))
        r = {**base, "status": "bet" if ok else "skip gap", "S": z["x"], "gap_S": gap, "ticker": z["ticker"],
             "buy_yes": buy_yes, "ask": ask, "cost": cost, "quote_age_min": (ts(T) - z["end"]) / 60,
             "spread_c": round((z["ask"] - z["bid"]) * 100)}
        if s:
            hs, as_ = s["home_score"], s["away_score"]
            v = hs + as_ if mk == "total" else hs - as_
            won = v > z["x"] if up else v < z["x"]
            xc = s["close_line"] if mk == "total" else -s["close_line"]
            r.update(final=v, won=won, roi=(1 - cost) / cost if won else -1.0, clv=(xc - z["x"]) if up else (z["x"] - xc))
        res.append(r)
    d = pd.DataFrame(res)
    d.to_csv(OUT / f"q2_nfl_rows_{split}.csv", index=False)
    b = d[(d.status == "bet") & d.roi.notna()] if "roi" in d else d.iloc[:0]
    print(d.status.value_counts().to_dict())
    out = {"n_bet_settled": len(b)}
    if len(b):
        rng = np.random.default_rng(0)
        for col in ("roi", "clv"):
            v = b[col].to_numpy(float)
            bs = rng.choice(v, (4000, len(v))).mean(1)
            out[col] = [round(v.mean(), 4), round(np.percentile(bs, 2.5), 4), round(np.percentile(bs, 97.5), 4)]
        out["W"] = int((b.won == True).sum()); out["mean_cost"] = round(b.cost.mean(), 4)
        out["by_mk"] = {m: dict(n=len(x), W=int((x.won == True).sum()), roi=round(x.roi.mean(), 4), clv=round(x.clv.mean(), 3))
                        for m, x in b.groupby("mk")}
    print(json.dumps(out))
    json.dump(out, open(OUT / f"q2_nfl_summary_{split}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
