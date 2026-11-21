"""Pull all Kalshi series (public /series), keyword-filter for free public count/chart settlement sources."""
import json, sys, re
sys.path.insert(0, "scripts")
from kx_throttle import kget
out = "results/count_nowcast/series_live.json"
r = kget("/series", {})
ser = r.get("series", []) if r else []
json.dump(ser, open(out, "w"))
print("series", len(ser))
KW = re.compile(r"spotify|kworb|netflix|box ?office|the-numbers|mojo|app ?store|apple\.com/app|youtube|wikipedia|wikimedia|trends\.google|flightaware|opensky|\bfaa\b|caiso|ercot|\beia\b|gasbuddy|pjm|miso|nyiso|isone|steam|twitch|tiktok|instagram|openrouter|tsa\.gov|bts\.gov|subscriber|views|downloads|streams|pageview|passeng|flights?|load|demand|electric|grid|airport|chart|rank", re.I)
rows = []
for s in ser:
    src = " ".join((x.get("name","")+" "+x.get("url","")) for x in (s.get("settlement_sources") or []))
    txt = f"{s.get('ticker')} {s.get('title')} {src}"
    if KW.search(txt):
        rows.append((s.get("ticker"), s.get("frequency"), s.get("category"), (s.get("title") or "")[:60], src[:110]))
for r_ in sorted(rows): print(" | ".join(str(x) for x in r_))
print("matched", len(rows))
