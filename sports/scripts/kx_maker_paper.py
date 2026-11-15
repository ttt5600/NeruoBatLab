"""Forward PAPER maker shadow for the Kalshi-execution trials (READ-ONLY public GETs; no keys, no orders).

    python scripts/kx_maker_paper.py run       # shadow every settled-window bet in kx_forward/{k1,k50nfl,kcfb}.jsonl
    python scripts/kx_maker_paper.py status

Rule = maker_fairfill (notes/maker_fairfill.md; edgelab mf_<lg>_x{0,1,2}): for each taker `pick` row with status bet,
post (on paper) a bid on the SAME side of the SAME ticker at b_x = floor_cent(side_mid - x), x in {0,1,2}c, capped at
side ask - 1c, where side_mid comes from the decision quote already logged (yes_bid / yes_ask), at decided_at; rest
until event_start - 5 min; FILL iff the public trade tape prints strictly through b on our side in that window
(YES bid: yes_price < b; NO bid: no_price < b). Cost b + maker fee 0.0175 p(1-p) (ceil per 100-ct order).
The tape is historical and immutable and the order is fixed by pre-decision quotes, so shadowing after the window
closes is lookahead-free -- no live process is needed. Grading reuses the taker trial's `settle` row.

Output (append-only): ~/.claude/quant-sports/data/kx_forward/maker_<trial>.jsonl, one `shadow` row per bet (written
once its window has closed) with fills for all x, thru/touch contract counts, and won/roi once the settle row exists
(a later `grade` row is appended when the settle arrives after the shadow).
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kx_throttle import kget  # noqa: E402

D = Path(os.environ.get("KX_FWD_DIR", str(Path.home() / ".claude" / "quant-sports" / "data" / "kx_forward")))
TRIALS = ("k1", "k50nfl", "kcfb")
XS = (0, 1, 2)
CANCEL_MIN = 5
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def ts(x) -> int:
    return int((pd.Timestamp(x) - EPOCH) // pd.Timedelta(seconds=1))


def mfee(p, n=100):
    return math.ceil(round(n * 0.0175 * p * (1 - p) * 100, 6)) / 100 / n


def rows(f):
    return [json.loads(l) for l in open(f) if l.strip()] if f.exists() else []


def tape(ticker, t0, t1):
    p, out = {"ticker": ticker, "limit": 1000, "min_ts": t0, "max_ts": t1}, []
    while True:
        d = kget("/markets/trades", p) or {}
        out += d.get("trades", [])
        if not d.get("cursor") or not d.get("trades"):
            return out
        p["cursor"] = d["cursor"]


def settle_of(R, b):
    k = b.get("bet_key") or b.get("key") or b.get("event")
    for r in R:
        if r.get("kind") == "settle" and (r.get("bet_key") or r.get("key") or r.get("event")) == k:
            return r
    return None


def won_of(s):
    if s is None:
        return None
    for k in ("won", "win"):
        if k in s and s[k] is not None:
            return bool(s[k])
    return None


def run():
    now = pd.Timestamp.now(tz="UTC")
    for tr in TRIALS:
        R = rows(D / f"{tr}.jsonl")
        out_f = D / f"maker_{tr}.jsonl"
        done = {(r["ticker"], r["decided_at"]): r for r in rows(out_f) if r["kind"] == "shadow"}
        graded = {(r["ticker"], r["decided_at"]) for r in rows(out_f) if r["kind"] == "grade"}
        new = []
        for b in R:
            if b.get("kind") != "pick" or b.get("status") != "bet":
                continue
            key = (b["ticker"], b["decided_at"])
            s = settle_of(R, b)
            if key in done:
                if done[key].get("won") is None and key not in graded and won_of(s) is not None:
                    new.append(dict(kind="grade", trial=tr, ticker=b["ticker"], decided_at=b["decided_at"],
                                    won=won_of(s), clv=s.get("clv_S", s.get("clv")), roi={x: (1 - c) / c if won_of(s) else -1.0
                                                        for x, c in done[key]["cost"].items()}))
                continue
            end = pd.Timestamp(b["event_start"]) - pd.Timedelta(minutes=CANCEL_MIN)
            if end > now:
                continue
            yes = b["side"] == "yes"
            q2 = "yes_bid" in b and "yes_ask" in b
            ask = (b["yes_ask"] if yes else round(1 - b["yes_bid"], 2)) if q2 else b["ask"]
            spread = (b["yes_ask"] - b["yes_bid"]) if q2 else b["spread_c"] / 100
            mid = ask - spread / 2
            T = pd.Timestamp(b["decided_at"])
            t = tape(b["ticker"], ts(T), ts(end))
            pr = [(float(z["yes_price_dollars"] if yes else z["no_price_dollars"]), float(z["count_fp"]),
                   pd.Timestamp(z["created_time"])) for z in t]
            pr = [(p, n) for p, n, c in pr if T < c <= end]
            r = dict(kind="shadow", trial=tr, ticker=b["ticker"], decided_at=b["decided_at"], event_start=b["event_start"],
                     side=b["side"], side_ask=ask, spread=round(spread, 4), n_trades=len(pr), bid={}, cost={}, fill={},
                     thru_ct={}, touch_ct={}, taker_cost=round(ask + b.get("fee", 0), 4), logged_at=now.isoformat())
            for x in XS:
                bx = min(math.floor(round((mid - x / 100) * 100, 6)) / 100, round(ask - 0.01, 2))
                thru = sum(n for p, n in pr if p < bx - 1e-9)
                r["bid"][x], r["cost"][x] = bx, round(bx + mfee(bx), 4)
                r["thru_ct"][x], r["touch_ct"][x] = thru, sum(n for p, n in pr if p <= bx + 1e-9)
                r["fill"][x] = thru > 0
            w = won_of(s)
            r["won"] = w
            r["clv"] = (s or {}).get("clv_S", (s or {}).get("clv"))
            if w is not None:
                r["roi"] = {x: (1 - c) / c if w else -1.0 for x, c in r["cost"].items()}
            new.append(r)
        if new:
            with open(out_f, "a") as f:
                for r in new:
                    f.write(json.dumps(r) + "\n")
        print(tr, "appended", len(new))


def status():
    for tr in TRIALS:
        S = [r for r in rows(D / f"maker_{tr}.jsonl") if r["kind"] == "shadow"]
        G = {(r["ticker"], r["decided_at"]): r["won"] for r in rows(D / f"maker_{tr}.jsonl") if r["kind"] == "grade"}
        for x in map(str, XS):
            f = [r for r in S if r["fill"][x]]
            w = [r.get("won") if r.get("won") is not None else G.get((r["ticker"], r["decided_at"])) for r in f]
            w = [v for v in w if v is not None]
            print(f"{tr} x{x}: shadows {len(S)} filled {len(f)} graded {len(w)} W {sum(w)}")


if __name__ == "__main__":
    {"run": run, "status": status}[sys.argv[1] if len(sys.argv) > 1 else "status"]()
