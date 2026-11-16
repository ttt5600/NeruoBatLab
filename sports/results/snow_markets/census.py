"""Read-only census of KXSNOWTHANKSGIVING and KXSNOWDAYS (series, events, markets, rules, books)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from kx_throttle import kget

OUT = Path(__file__).resolve().parent
res = {}
for s in ["KXSNOWTHANKSGIVING", "KXSNOWDAYS"]:
    ser = kget("/series/" + s)
    evs = []
    for st in [None, "settled", "closed"]:
        p = {"series_ticker": s, "limit": 200, "with_nested_markets": "true"}
        if st:
            p["status"] = st
        r = kget("/events", p) or {}
        evs += r.get("events", [])
    seen = {}
    for e in evs:
        seen[e["event_ticker"]] = e
    res[s] = {"series": ser, "events": list(seen.values())}
(OUT / "census.json").write_text(json.dumps(res, indent=1))
for s, d in res.items():
    print("=" * 20, s)
    print(json.dumps(d["series"], indent=1)[:1500])
    for e in d["events"]:
        ms = e.get("markets", [])
        print(e["event_ticker"], "|", e.get("title"), "|", e.get("sub_title"), "| n=", len(ms))
        for m in ms:
            print("   ", m["ticker"], m.get("status"), "yb", m.get("yes_bid"), "ya", m.get("yes_ask"),
                  "last", m.get("last_price"), "vol", m.get("volume"), "oi", m.get("open_interest"),
                  "close", m.get("close_time"), "|", m.get("yes_sub_title") or m.get("subtitle"), "| res", m.get("result"))
