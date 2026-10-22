"""Polymarket "Will X beat quarterly earnings?" -- does the price already carry the beat base rate?

Data (public, read-only; the international Polymarket data feed -- these markets are flagged restricted=true there, so this
is used as DATA ONLY; tradability for us depends on Polymarket US listing them, unverified):
  gamma-api /events?tag_slug=earnings (closed + open)  -> strike, endDate (= expected release time), resolution
  clob /prices-history?market=<YES token>&fidelity=60  -> hourly price path

    python scripts/earnings_poly.py pull          # cache events + price histories -> data/earnings/poly_*.json
    python scripts/earnings_poly.py describe      # counts per month, NO outcomes looked at
    python scripts/earnings_poly.py dev --ledger  # score E1-E3 on DEV only
    python scripts/earnings_poly.py sealed --confirm   # ONE shot, only after the user's explicit yes

PRE-REGISTERED 2026-10-21 (written before any outcome or price was looked at):
  Universe : closed events tagged 'earnings' whose market question contains "beat" (EPS-beat markets: YES iff reported
             EPS > the strike = consensus at market creation; in-line = NO), resolved cleanly to 1/0.
  Decision : T = endDate - 24 h. Price p = last price-history point at or before T, required within the 72 h before T.
             Skip if p < 0.03 or p > 0.97 (already-known outcome / early report leak guard).
  Cost     : YES at p + 0.02, NO at (1 - p) + 0.02  (0.02 = ~half the observed 0.03 spread + a 1c fee; conservative).
  Trials   : E1 all_yes   -- buy YES on every market (is the base rate under-priced?)
             E2 cheap_yes -- buy YES when p <= 0.60 (market well under the ~70-80% market-wide beat rate)
             E3 rich_no   -- buy NO  when p >= 0.85 (fade the most confident YES prices)
  Split    : DEV = endDate < 2026-07-01.  SEALED = endDate >= 2026-07-01 (Q2-2026 + Q3-2026 seasons), unscored.
  Stats    : $1 per bet; ROI = units / staked; one-sided bootstrap over report DATES (clusters same-day macro shocks),
             3,000 resamples. Dev pass = ROI > 0 and p < 0.05/3 (Bonferroni over E1-E3). Pass -> ask user for sealed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import requests

G = "https://gamma-api.polymarket.com/events"
C = "https://clob.polymarket.com/prices-history"
D = Path(__file__).resolve().parents[1] / "data" / "earnings"
LEDGER = Path(__file__).resolve().parents[1] / "lab" / "ledger.jsonl"
SEAL = "2026-07-01"
COST = 0.02


def get(url: str, **params):
    for i in range(5):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(2 + i)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(1 + i)
    return None


def pull() -> None:
    ev = []
    for closed in ("true", "false"):
        off = 0
        while True:
            j = get(G, tag_slug="earnings", closed=closed, limit=100, offset=off)
            if not j:
                break
            ev += j
            off += 100
            time.sleep(0.15)
    json.dump(ev, open(D / "poly_events.json", "w"))
    print("events", len(ev))
    hist_fn = D / "poly_hist.json"
    hist = json.load(open(hist_fn)) if hist_fn.exists() else {}
    for e in ev:
        if not e.get("closed"):
            continue
        for m in e.get("markets", []):
            try:
                tok = json.loads(m["clobTokenIds"])[0]
            except Exception:
                continue
            if tok in hist:
                continue
            end = dt.datetime.fromisoformat(e["endDate"].replace("Z", "+00:00"))
            j = get(C, market=tok, startTs=int((end - dt.timedelta(days=10)).timestamp()),
                    endTs=int((end + dt.timedelta(days=2)).timestamp()), fidelity=60)
            hist[tok] = (j or {}).get("history", [])
            time.sleep(0.1)
    json.dump(hist, open(hist_fn, "w"))
    print("histories", len(hist))


def rows() -> list[dict]:
    ev = json.load(open(D / "poly_events.json"))
    hist = json.load(open(D / "poly_hist.json"))
    out, seen = [], set()
    for e in ev:
        if not e.get("closed") or e["id"] in seen:
            continue
        seen.add(e["id"])
        for m in e.get("markets", []):
            if "beat" not in (m.get("question") or "").lower():
                continue
            try:
                tok = json.loads(m["clobTokenIds"])[0]
                op = [float(x) for x in json.loads(m["outcomePrices"])]
            except Exception:
                continue
            if sorted(op) != [0.0, 1.0]:
                continue
            end = dt.datetime.fromisoformat(e["endDate"].replace("Z", "+00:00"))
            T = (end - dt.timedelta(hours=24)).timestamp()
            pts = [h for h in hist.get(tok, []) if T - 72 * 3600 <= h["t"] <= T]
            if not pts:
                continue
            p = float(pts[-1]["p"])
            if not (0.03 <= p <= 0.97):
                continue
            out.append({"slug": e.get("slug"), "end": e["endDate"][:10], "p": p, "y": int(op[0] == 1.0),
                        "vol": float(m.get("volumeNum") or 0)})
    return out


def score(rs: list[dict], rule: str) -> dict:
    bets = []
    for r in rs:
        if rule == "E1_all_yes" or (rule == "E2_cheap_yes" and r["p"] <= 0.60):
            c = r["p"] + COST
            bets.append((r["end"], (r["y"] / c) - 1))
        elif rule == "E3_rich_no" and r["p"] >= 0.85:
            c = 1 - r["p"] + COST
            bets.append((r["end"], ((1 - r["y"]) / c) - 1))
    if not bets:
        return {"bets": 0}
    days = sorted({d for d, _ in bets})
    by = {d: [u for dd, u in bets if dd == d] for d in days}
    units = sum(u for _, u in bets)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(3000):
        s = rng.choice(len(days), len(days))
        uu = [u for i in s for u in by[days[i]]]
        boot.append(sum(uu) / len(uu))
    boot = np.array(boot)
    return {"bets": len(bets), "days": len(days), "units": round(units, 3), "roi": round(units / len(bets), 4),
            "ci_lo": round(float(np.quantile(boot, .025)), 4), "ci_hi": round(float(np.quantile(boot, .975)), 4),
            "p": round(float((boot <= 0).mean()), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pull", "describe", "dev", "sealed"])
    ap.add_argument("--ledger", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()
    if a.cmd == "pull":
        pull()
        return 0
    rs = rows()
    dev = [r for r in rs if r["end"] < SEAL]
    if a.cmd == "describe":
        import collections
        print("usable", len(rs), "dev", len(dev), "sealed", len(rs) - len(dev))
        print(sorted(collections.Counter(r["end"][:7] for r in rs).items()))
        return 0
    if a.cmd == "sealed":
        if not a.confirm:
            print("sealed needs --confirm AND the user's explicit yes")
            return 1
        # SEALED E2 (user yes 2026-10-22): E2_cheap_yes only, frozen rule/cost/bootstrap above, on the cached pull of
        # 2026-10-21 (sha256 of both cache files recorded). PASS = ROI > 0 and one-sided p < 0.05. Scored once.
        import hashlib
        lb = LEDGER.parent / "lockbox.jsonl"
        if any(json.loads(l).get("strategy") == "poly_earnings_E2_cheap_yes" for l in lb.read_text().splitlines() if l.strip()):
            print("E2 already scored -- never re-score")
            return 1
        sealed = [r for r in rs if r["end"] >= SEAL]
        s = score(sealed, "E2_cheap_yes")
        verdict = "PASS" if s.get("bets") and s["roi"] > 0 and s["p"] < 0.05 else "FAIL"
        shas = {f: hashlib.sha256((D / f).read_bytes()).hexdigest()[:12] for f in ("poly_events.json", "poly_hist.json")}
        print(f"SEALED n={len(sealed)} markets (endDate >= {SEAL}); E2 {s} -> {verdict}; cache {shas}")
        rec = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "strategy": "poly_earnings_E2_cheap_yes", "sha": shas["poly_hist.json"], "dataset": "poly_earnings_beat",
               "promotion_number": 1, **s, "staked": float(s.get("bets", 0)), "cache_sha": shas, "verdict": verdict}
        with open(lb, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return 0
    print(f"DEV n={len(dev)}  mean price {np.mean([r['p'] for r in dev]):.3f}  realised YES {np.mean([r['y'] for r in dev]):.3f}")
    for lo, hi in [(0, .4), (.4, .6), (.6, .75), (.75, .85), (.85, 1.01)]:
        b = [r for r in dev if lo <= r["p"] < hi]
        if b:
            print(f"  p in [{lo:.2f},{hi:.2f}): n {len(b):4d}  mean p {np.mean([r['p'] for r in b]):.3f}  YES rate {np.mean([r['y'] for r in b]):.3f}")
    for i, rule in enumerate(["E1_all_yes", "E2_cheap_yes", "E3_rich_no"], 1):
        s = score(dev, rule)
        print(rule, s)
        if a.ledger and s.get("bets"):
            rec = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "strategy": f"poly_earnings_{rule}",
                   "sha": "", "dataset": "poly_earnings_beat", "hypothesis": "pre-registered in scripts/earnings_poly.py",
                   "source": "earnings beat base-rate question (user 2026-10-21)", "status": "ok", "trial_number": i,
                   "bets": s["bets"], "staked": float(s["bets"]), "units": s["units"], "roi": s["roi"],
                   "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"], "p": s["p"]}
            with open(LEDGER, "a") as f:
                f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
