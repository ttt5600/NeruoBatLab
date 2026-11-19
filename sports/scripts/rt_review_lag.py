"""rt_review_lag: KXRT rules census + Kalshi-vs-RT lead-lag on Q013 logs (read-only; notes/rt_review_lag.md).

    python scripts/rt_review_lag.py census     # Kalshi KXRT events/markets -> results/rt_review_lag/kxrt_events.json
    python scripts/rt_review_lag.py leadlag    # M1 on Q013 dev window (<= 2026-11-11 23:59Z)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(__file__).resolve().parents[1] / "results" / "rt_review_lag"
OUT.mkdir(parents=True, exist_ok=True)
Q013 = Path.home() / ".claude" / "quant-kalshi" / "data" / "q013"
DEV_END = 1794441599  # 2026-11-11 23:59:59Z


def census():
    from kx_throttle import kget
    print(json.dumps(kget("/series/KXRT"), indent=0)[:2500])
    evs = []
    for st in ["settled", "open", "closed"]:
        cur = None
        while True:
            p = {"series_ticker": "KXRT", "status": st, "limit": 200, "with_nested_markets": "true"}
            if cur:
                p["cursor"] = cur
            r = kget("/events", p)
            evs += r.get("events", [])
            cur = r.get("cursor")
            if not cur or not r.get("events"):
                break
    json.dump(evs, open(OUT / "kxrt_events.json", "w"))
    print("events", len(evs))
    m = evs[0]["markets"][0]
    print(evs[0]["title"])
    print(m.get("rules_primary"))
    print(m.get("rules_secondary"))
    print({k: m.get(k) for k in ["open_time", "close_time", "expiration_time", "settlement_timer_seconds", "result",
                                 "early_close_condition", "can_close_early"]})


def _read(name, tmax):
    out = []
    with open(Q013 / name) as f:
        for line in f:
            r = json.loads(line)
            t = r.get("ts", r.get("first_seen"))
            if t is not None and t <= tmax:
                out.append(r)
    return out


def leadlag(max_gap=20 * 60, tag="pre"):
    import bisect
    import numpy as np
    tmax = DEV_END + 3 * 3600  # +60 min post window for events up to DEV_END (events themselves <= DEV_END)
    sc = [dict(r, score=r["score"] if r["score"] is not None else r.get("score_calc")) for r in _read("scores.jsonl", tmax) if r["source"] == "rt"]  # RT dropped criticsScore.score on 11-03; score_calc = round-half-up(liked/n)
    snaps = [r for r in _read("snapshots.jsonl", tmax) if r["series"] == "KXRT"]
    by_tk = {}
    for s in snaps:
        if s["yes_bid"] is None or s["yes_ask"] is None or s["yes_ask"] <= 0:
            continue
        by_tk.setdefault(s["ticker"], []).append((s["ts"], (s["yes_bid"] + s["yes_ask"]) / 2, s["floor"],
                                                   s["strike_type"], s["event"]))
    for v in by_tk.values():
        v.sort()
    ev_tk = {}
    for tk, v in by_tk.items():
        ev_tk.setdefault(v[0][4], []).append((tk, v[0][2], v[0][3]))

    def mid(tk, t):
        v = by_tk[tk]
        i = bisect.bisect_right([x[0] for x in v], t) - 1
        if i < 0 or t - v[i][0] > 30 * 60:
            return None
        return v[i][1]

    per_ev = {}
    for r in sc:
        per_ev.setdefault(r["event"], []).append(r)
    rows, dropped = [], {"gap": 0, "no_mid": 0, "no_strike": 0}
    for ev, rs in per_ev.items():
        rs.sort(key=lambda r: r["ts"])
        for a, b in zip(rs, rs[1:]):
            if b["ts"] > DEV_END or a["score"] is None or b["score"] is None:
                continue
            ds, dn = b["score"] - a["score"], (b["n"] or 0) - (a["n"] or 0)
            if abs(ds) < 2 and dn < 5:
                continue
            if ds == 0:
                continue  # direction undefined (n moved, score did not)
            if b["ts"] - a["ts"] > max_gap:
                dropped["gap"] += 1
                continue
            cands = [x for x in ev_tk.get(ev, []) if x[2] == "greater" and x[1] is not None]
            if not cands:
                dropped["no_strike"] += 1
                continue
            tk, fl, _ = min(cands, key=lambda x: abs(x[1] - b["score"]))
            s = 1 if ds > 0 else -1
            m0, m1, m2, m3 = mid(tk, a["ts"] - 3600), mid(tk, a["ts"]), mid(tk, b["ts"]), mid(tk, b["ts"] + 3600)
            if None in (m0, m1, m2, m3):
                dropped["no_mid"] += 1
                continue
            rows.append({"event": ev, "t_prev": a["ts"], "t": b["ts"], "ds": ds, "dn": dn, "n_prev": a["n"],
                         "score": b["score"], "ticker": tk, "floor": fl, "m0": m0, "m1": m1, "m2": m2, "m3": m3,
                         "pre": s * (m1 - m0), "conc": s * (m2 - m1), "post": s * (m3 - m2), "tot": s * (m3 - m0)})
    json.dump(rows, open(OUT / f"leadlag_{tag}.json", "w"), indent=0)
    print("events", len(rows), "films", len({r["event"] for r in rows}), "dropped", dropped)
    if not rows:
        return
    films = sorted({r["event"] for r in rows})
    F = {f: np.array([[r["pre"], r["conc"], r["post"], r["tot"]] for r in rows if r["event"] == f]) for f in films}
    for f in films:
        x = F[f]
        print(f, len(x), "sum pre/conc/post/tot", np.round(x.sum(0), 3))
    tot = np.vstack(list(F.values())).sum(0)
    print("POOLED pre/conc/post/tot", np.round(tot, 3), "pre-share", round(tot[0] / tot[3], 3) if tot[3] else None,
          "post-share", round(tot[2] / tot[3], 3) if tot[3] else None)
    rng = np.random.default_rng(4)
    bs = []
    for _ in range(10000):
        pick = rng.integers(0, len(films), len(films))
        s_ = sum(F[films[i]].sum(0) for i in pick)
        bs.append((s_[0] / s_[3] if s_[3] > 0 else np.nan, s_[2] / s_[3] if s_[3] > 0 else np.nan, s_[3]))
    bs = np.array(bs)
    ok = bs[:, 2] > 0
    print("bootstrap valid", ok.mean(), "pre-share CI", np.nanpercentile(bs[ok, 0], [2.5, 97.5]).round(3),
          "post-share CI", np.nanpercentile(bs[ok, 1], [2.5, 97.5]).round(3))
    fs = [F[f].sum(0) for f in films]
    print("film median pre-share (tot>0)", np.median([x[0] / x[3] for x in fs if x[3] > 0]).round(3),
          "films tot>0", sum(x[3] > 0 for x in fs))


def trades(event, t0, t1):
    """Public trade tape for every market of `event` in [t0, t1] -> results/rt_review_lag/trades_<event>.json"""
    from kx_throttle import kget
    E = {e["event_ticker"]: e for e in json.load(open(OUT / "kxrt_events.json"))}
    out = []
    for m in E[event]["markets"]:
        cur = None
        while True:
            p = {"ticker": m["ticker"], "min_ts": t0, "max_ts": t1, "limit": 1000}
            if cur:
                p["cursor"] = cur
            r = kget("/markets/trades", p)
            out += r.get("trades", [])
            cur = r.get("cursor")
            if not cur or not r.get("trades"):
                break
    json.dump(out, open(OUT / f"trades_{event}.json", "w"))
    print(event, len(out))


if __name__ == "__main__":
    a = sys.argv[1]
    if a == "trades":
        trades(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    if a == "census":
        census()
    elif a == "leadlag":
        leadlag()
    elif a == "leadlag70":
        leadlag(max_gap=70 * 60, tag="gap70_posthoc")
