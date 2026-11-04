"""NYT House needle vs Kalshi (user 2026-11-03, election night). Tier C judgment, routed through bet.py.

    nyt_needle_house.py [--live] [--once] [--edge 0.25] [--kelly 0.25] [--interval 60]

Each loop: NYT results-house.json -> per-race party win_probability (the needle). Kalshi HOUSE<ST><N> series,
markets -<suffix> D/R. Fires when needle p - ask >= edge on YES or NO side. Size = kelly * full-Kelly fraction
of bankroll (Kalshi cash + stake already placed by this bot), clipped to bet.py's tier-C cap ($12.50). IOC taker
at the observed ask (limit), one fill per market. bet.py enforces KILL file + all loss caps. Default is DRY.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kalshi_exec as K  # noqa: E402

NYT = "https://static01.nyt.com/elections-assets/pages/data/2026-11-03/results-house.json"
KB = "https://api.elections.kalshi.com/trade-api/v2"
SER = re.compile(r"^(?:KX)?HOUSE(?:PARTY-)?([A-Z]{2})-?(\d{1,2}|AL)$")
STATE = K.DEPLOY / "data" / "nyt_needle_house_state.json"
LOG = K.DEPLOY / "data" / "nyt_needle_house_log.jsonl"
TIER_C_CAP = 12.50
PARTY = {"DEM": "D", "GOP": "R", "REP": "R"}


def get(url: str, tries: int = 5):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            return json.load(urllib.request.urlopen(req, timeout=30))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(1.5 * (i + 1))
                continue
            raise
    raise RuntimeError(f"429 x{tries}: {url}")


def needles() -> dict:
    """(ST, district) -> {'D': p, 'R': p, 'counted': n, 'updated': iso}"""
    out = {}
    for r in get(NYT)["races"]:
        t = r.get("top_reporting_unit") or {}
        st, seat = t.get("state_postal"), r.get("seat_number")
        if not st or (r.get("model_settings") or {}).get("model_status") != "active":
            continue
        est = t.get("nyt_model_estimates") or {}
        ps = {}
        for c in t.get("candidates") or []:
            e = c.get("nyt_model_estimates") or {}
            pa = PARTY.get(e.get("party_caucus"))
            if pa and e.get("win_probability") is not None:
                ps[pa] = ps.get(pa, 0.0) + float(e["win_probability"])
        if not ps:
            continue
        d = "AL" if seat in (None, "", "0", "AL") else str(int(seat))
        out[(st, d)] = dict(ps, counted=est.get("votes_counted") or 0, updated=est.get("nyt_updated_at"),
                            call=r.get("race_call_status"))
    return out


_series: list[str] = []


def kalshi_markets() -> list[dict]:
    global _series
    if not _series:
        _series = [x["ticker"] for x in get(KB + "/series?category=Elections")["series"] if SER.match(x["ticker"])]
    ms = []
    for s in _series:
        st, d = SER.match(s).groups()
        d = d if d == "AL" else str(int(d))
        for m in get(f"{KB}/markets?series_ticker={s}&status=open&limit=200").get("markets", []):
            suf = m["ticker"].rsplit("-", 1)[-1]
            if suf in ("D", "R"):
                m["_key"], m["_party"] = (st, d), suf
                ms.append(m)
        time.sleep(0.15)
    return ms


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def kelly(p: float, q: float) -> float:
    return max(0.0, (p - q) / (1 - q))


def cash() -> float:
    import bet as B
    c = B.client()
    code, bal = c.request("GET", "/portfolio/balance")
    return (bal.get("balance") or 0) / 100.0 if code == 200 else 0.0


def held(ticker: str) -> bool:
    """IOC can be accepted with zero fill; only a real position closes the race."""
    import bet as B
    code, p = B.client().request("GET", f"/portfolio/positions?ticker={ticker}")
    return code == 200 and any(abs(float(x.get("position_fp") or x.get("position") or 0)) > 0
                               for x in p.get("market_positions", []) if x.get("ticker") == ticker)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--edge", type=float, default=0.25)
    ap.add_argument("--kelly", type=float, default=0.25)
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--max-age-min", type=float, default=20)
    ap.add_argument("--min-counted", type=int, default=1)
    a = ap.parse_args()
    state = json.loads(STATE.read_text()) if STATE.exists() else {"done": {}, "staked": 0.0}
    while True:
        now = datetime.now(timezone.utc)
        try:
            N, M = needles(), kalshi_markets()
        except Exception as e:  # keep running through transient fetch errors
            print(now.isoformat(timespec="seconds"), "fetch error:", e, flush=True)
            time.sleep(a.interval)
            continue
        bank = None
        sigs = []
        for m in M:
            n = N.get(m["_key"])
            if not n or m["_party"] not in n:
                continue
            age = (now - datetime.fromisoformat(n["updated"])).total_seconds() / 60 if n.get("updated") else 1e9
            if age > a.max_age_min or n["counted"] < a.min_counted:
                continue
            p = n[m["_party"]]
            ya, yb = f(m.get("yes_ask_dollars")), f(m.get("yes_bid_dollars"))
            for side, prob, ask in (("yes", p, ya), ("no", 1 - p, None if yb is None else round(1 - yb, 4))):
                if ask is None or not 0.01 <= ask <= 0.97:
                    continue
                e = prob - ask
                if e >= a.edge:
                    sigs.append((e, m, side, prob, ask, n, age))
        sigs.sort(key=lambda s: -s[0])
        best = {}
        for s_ in sigs:  # one leg per race: the YES-R and NO-D legs are the same bet
            best.setdefault(s_[1]["_key"], s_)
        sigs = list(best.values())
        print(f"{now.isoformat(timespec='seconds')} needles={len(N)} kalshi={len(M)} signals={len(sigs)}", flush=True)
        for e, m, side, prob, ask, n, age in sigs:
            key = "-".join(m["_key"])
            row = dict(ts=now.isoformat(timespec="seconds"), ticker=m["ticker"], side=side, needle=round(prob, 4),
                       ask=ask, edge=round(e, 4), counted=n["counted"], call=n.get("call"), age_min=round(age, 1))
            if key in state["done"]:
                continue
            if bank is None:
                bank = cash() + state["staked"]
            stake = min(TIER_C_CAP, a.kelly * kelly(prob, ask) * bank)
            row["stake"] = round(stake, 2)
            print(f"  SIGNAL {m['ticker']} {side.upper()} needle {prob:.3f} ask {ask:.2f} edge {e:+.2f} "
                  f"counted {n['counted']} age {age:.0f}m -> ${stake:.2f}", flush=True)
            if stake < 1.0:
                row["action"] = "too_small"
            else:
                cmd = [sys.executable, str(HERE / "bet.py"), "place", m["ticker"], side, "--units", f"{stake / 50:.4f}",
                       "--price", f"{ask:.2f}", "--taker", "--tier", "C", "--strategy", "nyt_needle_house"]
                if not a.live:
                    cmd.append("--dry")
                r = subprocess.run(cmd, capture_output=True, text=True)
                row["action"], row["bet_out"] = ("live" if a.live else "dry"), (r.stdout + r.stderr).strip()[-400:]
                print("   ", row["bet_out"], flush=True)
                if a.live and "placed" in r.stdout and held(m["ticker"]):
                    state["done"][key] = row
                    state["staked"] += stake
                    STATE.write_text(json.dumps(state, indent=1))
            with LOG.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
        if a.once:
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
