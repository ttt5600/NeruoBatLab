"""READ-ONLY diagnostic for the 2026-11-09 NFLF runaway exit (sports/STATE.md, incident (a)).

Question: why did ingame_exec.find_order(client_order_id) return no fill for IOC exit sells that executed?
Every call here is a GET. No orders, no cancels.

  ../.venv_quant/bin/python scripts/diag_find_order.py [--journal PATH] [--ticker T ...] [--dump DIR]

For each exit client_order_id in the journal it prints what each lookup path returns:
  A  find_order path : GET /portfolio/orders?ticker=T (paged)        -> is the cid listed?
  B  status filters  : same with status=executed / canceled / resting
  C  by order id     : GET /portfolio/orders/{order_id} for ids seen in fills
  D  fills           : GET /portfolio/fills?ticker=T                  -> fills whose order_id maps to the cid
plus the field names the Order objects actually carry (fill_count vs fill_count_fp etc.).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kalshi_exec as kx  # noqa: E402

JOURNAL = kx.DEPLOY / "data" / "lead_exec.jsonl"


def incident_cids(journal: Path, tickers: list[str]) -> dict[str, list[dict]]:
    """ticker -> [{cid, ts, key, attempt, limit, kind}] for entry and exit client_order_ids in the journal."""
    out: dict[str, list[dict]] = {}
    for line in journal.read_text().splitlines():
        r = json.loads(line)
        t = r.get("ticker", "")
        if tickers and not any(t.startswith(x) for x in tickers):
            continue
        if r.get("exit_coid"):
            out.setdefault(t, []).append({"cid": r["exit_coid"], "ts": r["ts"], "key": r.get("key"),
                                          "attempt": r.get("exit_attempts"), "limit": r.get("exit_limit"),
                                          "kind": "exit"})
        elif r.get("event") == "order" and r.get("client_order_id"):
            out.setdefault(t, []).append({"cid": r["client_order_id"], "ts": r["ts"], "key": r.get("key"),
                                          "attempt": 0, "limit": r.get("limit_price"), "kind": "entry"})
    return out


def get(c: kx.Kalshi, path: str, params: dict | None = None):
    code, out = c.request("GET", path, params=params or {})
    return code, out


def pages(c: kx.Kalshi, path: str, key: str, params: dict) -> tuple[list[dict], list[int]]:
    rows, sizes, cursor = [], [], None
    for _ in range(50):
        q = dict(params, limit=200, **({"cursor": cursor} if cursor else {}))
        code, out = get(c, path, q)
        if code != 200:
            print(f"   GET {path} {q} -> HTTP {code} {str(out)[:200]}")
            break
        got = out.get(key, [])
        rows += got
        sizes.append(len(got))
        cursor = out.get("cursor")
        if not cursor:
            break
    return rows, sizes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", type=Path, default=JOURNAL)
    ap.add_argument("--ticker", nargs="*", default=["KXNFLGAME-26NOV09BUFMIN"])
    ap.add_argument("--dump", type=Path, default=None, help="write raw API rows here (json)")
    a = ap.parse_args()
    kx.set_root(kx.DEPLOY)          # read the deployed .env (key); nothing here writes
    c = kx.Kalshi(kx.env(), kx.config(kx.env())["env"])
    if not c.has_key():
        print("no key")
        return 2
    want = incident_cids(a.journal, a.ticker)
    dump = {}
    for t, cids in sorted(want.items()):
        print(f"\n=== {t}: {len(cids)} journal cids ({sum(x['kind'] == 'exit' for x in cids)} exit)")
        # A: exactly what find_order does
        listed = c.orders(ticker=t)
        by_cid = {x.get("client_order_id"): x for x in listed}
        print(f"A  client.orders(ticker) -> {len(listed)} orders; statuses {Counter(x.get('status') for x in listed)}")
        print(f"   field names on an order: {sorted(listed[0].keys()) if listed else None}")
        # A2: raw paging sizes, no status filter
        raw, sizes = pages(c, "/portfolio/orders", "orders", {"ticker": t})
        print(f"A2 raw pages (limit=200): sizes {sizes}, total {len(raw)}")
        # B: status filters
        for s in ("resting", "canceled", "executed"):
            rs, _ = pages(c, "/portfolio/orders", "orders", {"ticker": t, "status": s})
            print(f"B  status={s:9s} -> {len(rs)} orders; journal cids among them: "
                  f"{sum(1 for x in rs if x.get('client_order_id') in {y['cid'] for y in cids})}")
        # D: fills
        fills, fsizes = pages(c, "/portfolio/fills", "fills", {"ticker": t})
        print(f"D  fills -> {len(fills)} rows (pages {fsizes}); fill fields {sorted(fills[0].keys()) if fills else None}")
        fill_by_oid: dict[str, list[dict]] = {}
        for f in fills:
            fill_by_oid.setdefault(f.get("order_id"), []).append(f)
        oid_of_cid = {x.get("client_order_id"): x.get("order_id") for x in raw}
        # per-cid table
        print(f"   {'kind':5s} {'att':>3s} {'ts':25s} {'cid':10s} listed status   fill_count_fp fill_count "
              f"taker$   find_order->fills_from_order  fills(order_id)")
        for w in cids:
            x = by_cid.get(w["cid"])
            ff = fills_from_order_safe(x)
            oid = (x or {}).get("order_id") or oid_of_cid.get(w["cid"])
            fsum = sum(float(f.get("count_fp") or f.get("count") or 0) for f in fill_by_oid.get(oid, []))
            print(f"   {w['kind']:5s} {w['attempt'] or 0:>3} {w['ts']:25s} {w['cid'][:8]:10s} "
                  f"{'Y' if x else 'N':6s} {str((x or {}).get('status')):8s} "
                  f"{str((x or {}).get('fill_count_fp')):13s} {str((x or {}).get('fill_count')):10s} "
                  f"{str((x or {}).get('taker_fill_cost_dollars')):8s} {ff:29s} {fsum:g}")
        # C: by order id for fills whose order_id is NOT among listed orders
        listed_oids = {x.get("order_id") for x in raw}
        orphan = [o for o in fill_by_oid if o not in listed_oids]
        print(f"C  fills whose order_id is not in the orders list: {len(orphan)}")
        for o in orphan[:5]:
            code, out = get(c, f"/portfolio/orders/{o}")
            od = out.get("order", out) if isinstance(out, dict) else out
            print(f"   GET /portfolio/orders/{o[:8]} -> HTTP {code} cid={str(od.get('client_order_id'))[:8]} "
                  f"status={od.get('status')} fill_count_fp={od.get('fill_count_fp')}")
        # positions now
        pos = c.positions(ticker=t)
        print(f"   position now: {[(p.get('ticker'), p.get('position_fp')) for p in pos]}")
        dump[t] = {"orders": raw, "fills": fills, "positions": pos}
    if a.dump:
        a.dump.mkdir(parents=True, exist_ok=True)
        (a.dump / "raw.json").write_text(json.dumps(dump, indent=1, default=str))
        print(f"\nraw rows -> {a.dump / 'raw.json'}")
    return 0


def fills_from_order_safe(x: dict | None) -> str:
    if x is None:
        return "None (cid not listed)"
    u = kx.fills_from_order(x)
    return f"filled={u['filled']:g} st={u['status']}"


if __name__ == "__main__":
    sys.exit(main())
