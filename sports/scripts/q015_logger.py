"""Q015 forward paper logger: Kalshi approval-average markets vs the published RCP value + newly released polls.

    python scripts/q015_logger.py tick      # one snapshot (run every ~15-30 min; NOT installed in launchd)
    python scripts/q015_logger.py report    # join settled results -> paper ROI + poll-pressure check

Why a logger: the historical RCP poll tables (which polls were in the average, when each appeared) are not reachable
(Cloudflare 403; Wayback unreachable), so the poll-level recompute can only be tested forward, with OUR clock as the
point-in-time stamp. Each tick records:
  snapshots.jsonl  open markets of KXAPRPOTUS / KXTRUMPAPPROVE / KXTRUMPVH with yes_bid/yes_ask (top of book)
  polls.jsonl      every VoteHub Trump-approval poll the first time we see it (first_seen = our UTC clock)
  picks.jsonl      frozen paper rule (notes/q015_approval_avg.md trials T1/T3/T4): one pick per (trial, market) at the
                   first tick inside [tau, tau + 60 min]; info = latest KXTRUMPAPPROVE 13:00 ET settlement value
                   (the RCP history JSON stopped updating on 2026-02-25). Edge >= 5c at ask + 0.07 p(1-p).
No orders are ever sent; only public endpoints are read.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from sports.http import fetch
import q015_approval_avg as Q

OUT = ROOT / "data" / "q015" / "logger"
VH = "https://api.votehub.com/polls"
SERIES = ("KXAPRPOTUS", "KXTRUMPAPPROVE", "KXTRUMPVH")
LIVE_TRIALS = {"T1_wk24": ("KXAPRPOTUS", "close-24h", 2), "T3_day24": ("KXTRUMPAPPROVE", "close-24h+6m", 1),
               "T4_wk2": ("KXAPRPOTUS", "close-2h", 1)}
WINDOW = 3600


def _append(name: str, rows: list[dict]) -> None:
    if rows:
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / name).open("a") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")


def _read(name: str) -> pd.DataFrame:
    p = OUT / name
    return pd.read_json(p, lines=True) if p.exists() and p.stat().st_size else pd.DataFrame()


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def latest_rcp() -> tuple[float, int] | None:
    d = fetch(f"{Q.API}/markets", {"series_ticker": "KXTRUMPAPPROVE", "status": "settled", "limit": 200}, cache=False)
    best = None
    for m in d.get("markets", []):
        v = _num(m.get("expiration_value"))
        if v == v:
            t = int(pd.Timestamp(m["close_time"]).timestamp()) + 60
            if best is None or t > best[1]:
                best = (v, t)
    return best


def tick(now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    ts = int(now.timestamp())
    snaps = []
    for s in SERIES:
        d = fetch(f"{Q.API}/markets", {"series_ticker": s, "status": "open", "limit": 1000}, cache=False)
        for m in d.get("markets", []):
            snaps.append({"ts": ts, "series": s, "ticker": m["ticker"], "event_ticker": m["event_ticker"],
                          "close_time": m["close_time"], "yes_sub_title": m.get("yes_sub_title"),
                          "strike_type": m.get("strike_type"), "floor_strike": m.get("floor_strike"),
                          "cap_strike": m.get("cap_strike"), "yes_bid": _num(m.get("yes_bid_dollars")),
                          "yes_ask": _num(m.get("yes_ask_dollars")), "volume": _num(m.get("volume_fp"))})
    _append("snapshots.jsonl", snaps)

    # VoteHub polls: record first_seen on our clock (the point-in-time stamp the archive lacks)
    seen = set(_read("polls.jsonl").get("id", pd.Series(dtype=str)).tolist())
    try:
        polls = fetch(VH, {"poll_type": "approval", "subject": "Trump"}, cache=False)
    except Exception as e:  # noqa: BLE001
        polls, _ = [], print(f"votehub fetch failed: {e}")
    new = []
    for p in polls or []:
        if p.get("id") in seen:
            continue
        a = {x["choice"]: x["pct"] for x in p.get("answers", [])}
        new.append({"id": p["id"], "first_seen": ts, "created_at": p.get("created_at"), "end_date": p.get("end_date"),
                    "pollster": p.get("pollster"), "population": p.get("population"), "approve": a.get("Approve"),
                    "disapprove": a.get("Disapprove"), "backfill": not seen})
    _append("polls.jsonl", new)

    info = latest_rcp()
    picks = []
    if info:
        iv, iv_ts = info
        done = _read("picks.jsonl")
        done_keys = set(zip(done.trial, done.ticker)) if len(done) else set()
        h = pd.read_parquet(Q.D / "rcp_history.parquet")
        pms = {k: Q.pmf(h, k) for k in (1, 2)}
        for tid, (series, rule, k) in LIVE_TRIALS.items():
            for sn in snaps:
                if sn["series"] != series or (tid, sn["ticker"]) in done_keys:
                    continue
                close_ts = int(pd.Timestamp(sn["close_time"]).timestamp())
                tau = Q.decision_ts(close_ts, rule)
                if not (tau <= ts < tau + WINDOW) or iv_ts > ts:
                    continue
                y = Q.yes_set(sn)
                b, a = sn["yes_bid"], sn["yes_ask"]
                base = {"ts": ts, "trial": tid, "ticker": sn["ticker"], "event_ticker": sn["event_ticker"],
                        "close_ts": close_ts, "tau": tau, "info": iv, "info_ts": iv_ts, "yes_bid": b, "yes_ask": a}
                if y is None or not (b == b and a == a and 0 < b < a < 1):
                    picks.append({**base, "side": None, "reason": "no quote/strike"})
                    continue
                p = Q.p_yes(iv, y[0], y[1], pms[k])
                ey, en = p - a - Q.fee(a), (1 - p) - (1 - b) - Q.fee(1 - b)
                side = "yes" if ey >= Q.EDGE and ey >= en else ("no" if en >= Q.EDGE else None)
                price = a if side == "yes" else (1 - b if side == "no" else np.nan)
                picks.append({**base, "p_yes": p, "side": side, "price": price,
                              "cost": price + Q.fee(price) if side else np.nan, "edge": max(ey, en)})
    _append("picks.jsonl", picks)
    out = {"ts": now.isoformat(timespec="seconds"), "snapshots": len(snaps), "new_polls": len(new),
           "info": info, "picks_logged": len(picks), "bets": sum(1 for p in picks if p.get("side"))}
    print(json.dumps(out))
    return out


def report() -> int:
    pk = _read("picks.jsonl")
    if pk.empty:
        print("no picks yet")
        return 0
    res = {}
    for s in ("KXAPRPOTUS", "KXTRUMPAPPROVE"):
        d = fetch(f"{Q.API}/markets", {"series_ticker": s, "status": "settled", "limit": 1000}, cache=False)
        res.update({m["ticker"]: (m.get("result"), _num(m.get("expiration_value"))) for m in d.get("markets", [])})
    pk["result"] = pk.ticker.map(lambda t: res.get(t, (None, np.nan))[0])
    pk["settle_val"] = pk.ticker.map(lambda t: res.get(t, (None, np.nan))[1])
    bt = pk[pk.side.notna() & pk.result.isin(["yes", "no"])].copy()
    print(f"picks logged {len(pk)}, bets {int(pk.side.notna().sum())}, settled bets {len(bt)}")
    if len(bt):
        bt["pnl"] = np.where(bt.side == bt.result, 1.0, 0.0) - bt.cost
        for tid, g in bt.groupby("trial"):
            roi = g.pnl.sum() / g.cost.sum()
            lo = hi = np.nan
            if g.event_ticker.nunique() >= 5:
                lo, hi, p, _ = Q.boot(g.rename(columns={}), n=5000)
            print(f"  {tid}: bets {len(g)} events {g.event_ticker.nunique()} ROI {roi:+.4f} [{lo:+.4f},{hi:+.4f}]")
    # poll-pressure check: do VoteHub polls first seen between the info time and close predict (settle - info)?
    po = _read("polls.jsonl")
    if len(po) and len(pk):
        po = po[~po.backfill.astype(bool)]
        ev = pk.drop_duplicates(["trial", "event_ticker"]).dropna(subset=["settle_val"])
        rows = []
        for r in ev.itertuples():
            w = po[(po.first_seen > r.info_ts) & (po.first_seen <= r.tau)]
            rows.append({"event": r.event_ticker, "trial": r.trial, "resid": r.settle_val - r.info,
                         "n_new": len(w), "pressure": (w.approve.mean() - r.info) if len(w) else 0.0})
        pr = pd.DataFrame(rows)
        if len(pr) >= 5:
            print("poll pressure vs residual: corr", round(pr[["resid", "pressure"]].corr().iloc[0, 1], 3), "n", len(pr))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "tick"
    if cmd == "tick":
        tick()
    elif cmd == "report":
        report()
    else:
        print(__doc__)
