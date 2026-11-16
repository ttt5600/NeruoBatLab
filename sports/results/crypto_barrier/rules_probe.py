"""Outcome-free probe: series metadata + open events + rules text for Kalshi long-dated crypto max/min ladders."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
from kx_throttle import kget  # noqa: E402

SER = ["KXBTCMAXMON", "KXBTCMINMON", "KXETHMAXMON", "KXETHMINMON", "KXSOLMAXMON", "KXSOLMINMON",
       "KXBTCMAXY", "KXBTCMINY", "KXETHMAXY", "KXETHMINY", "KXBTCMAXM"]
out = {}
for s in SER:
    se = (kget("/series/" + s) or {}).get("series", {})
    print("==", s, se.get("title"), se.get("frequency"), se.get("fee_type"), se.get("fee_multiplier"))
    print("   sources:", se.get("settlement_sources"), se.get("contract_url"), se.get("contract_terms_url"))
    e = kget("/events", {"series_ticker": s, "status": "open", "with_nested_markets": "true"}) or {}
    out[s] = {"series": se, "events": []}
    for ev in e.get("events", []):
        ms = ev.get("markets", [])
        print("  ", ev["event_ticker"], ev.get("title"), "markets", len(ms))
        keep = [{k: m.get(k) for k in ("ticker", "subtitle", "yes_sub_title", "strike_type", "floor_strike", "cap_strike",
                                        "open_time", "close_time", "expiration_time", "yes_bid", "yes_ask", "volume",
                                        "rules_primary", "rules_secondary", "status")} for m in ms]
        out[s]["events"].append({"event_ticker": ev["event_ticker"], "title": ev.get("title"), "markets": keep})
        if ms:
            m = ms[0]
            print("    ", m["ticker"], m.get("open_time"), m.get("close_time"), m.get("expiration_time"))
            print("     R1:", m.get("rules_primary"))
            print("     R2:", m.get("rules_secondary"))
(HERE / "rules_probe.json").write_text(json.dumps(out, indent=1))
