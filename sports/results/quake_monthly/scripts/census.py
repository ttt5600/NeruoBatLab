"""Read-only census of KXBIGGESTQUAKEMONTH (series, events, markets, rules). No orders."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from kx_throttle import kget

OUT = Path(__file__).resolve().parents[1]
S = "KXBIGGESTQUAKEMONTH"
ser = kget(f"/series/{S}")
evs = kget("/events", {"series_ticker": S, "limit": 200})
mk = {}
for st in ["open", "closed", "settled"]:
    m = kget("/markets", {"series_ticker": S, "limit": 1000, "status": st}) or {"markets": []}
    mk[st] = m["markets"]
json.dump({"series": ser, "events": evs, "markets": mk}, open(OUT / "census_raw.json", "w"), indent=1)
print(json.dumps(ser, indent=1)[:2500])
for e in evs["events"]:
    print("EV", e["event_ticker"], e.get("title"), e.get("sub_title"), e.get("strike_date"))
for st, ms in mk.items():
    print("==", st, len(ms))
    for x in ms:
        print(x["ticker"], "bid", x.get("yes_bid_dollars"), "ask", x.get("yes_ask_dollars"), "vol", x.get("volume_fp"),
              "oi", x.get("open_interest_fp"), "res", x.get("result"), x.get("open_time"), x.get("close_time"),
              "exp", x.get("expiration_value"), "fee", x.get("fee_type") if "fee_type" in x else "")
if mk["open"]:
    x = mk["open"][0]
    print("RULES1:", x.get("rules_primary"))
    print("RULES2:", x.get("rules_secondary"))
    print({k: v for k, v in x.items() if "rules" not in k})
