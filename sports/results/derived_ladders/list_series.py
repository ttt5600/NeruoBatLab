import sys, json
sys.path.insert(0, "scripts")
from kx_throttle import kget
j = kget("/series", {"category": "Sports"}) or {}
s = j.get("series") or []
print(len(s), list(j.keys()))
json.dump(s, open("results/derived_ladders/series_sports.json", "w"))
