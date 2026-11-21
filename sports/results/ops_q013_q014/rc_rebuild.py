"""Page the NEW Factbase endpoint (rollcall.com/api/factbase/social, the URL the public archive page itself calls)
newest-first back to a cutoff and save every post. Honest UA, 2 s between pages."""
import json, sys, time
from datetime import datetime
import requests
URL = "https://rollcall.com/api/factbase/social"
UA = "quant-research-logger/1.0 (personal read-only research; contact jonathanwang)"
cut = int(datetime.fromisoformat(sys.argv[2]).timestamp())
out = open(sys.argv[1], "w")
n = 0
for page in range(1, 200):
    r = requests.get(URL, params={"sort": "date", "sort_order": "desc", "page": page, "size": 50, "platform": "Truth Social"},
                     headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    d = r.json()
    data = d.get("data") or []
    for x in data:
        s = x.get("social") or {}
        out.write(json.dumps({"id": str(x["id"]), "date": x["date"], "post_ts": int(datetime.fromisoformat(x["date"]).timestamp()),
                              "deleted": bool(x.get("deleted_flag")), "deleted_date": s.get("deleted_date"),
                              "repost": bool(s.get("repost_flag")), "quote": bool(s.get("quote_flag")),
                              "reply_to": s.get("in_reply_to_screen_name"), "platform": x.get("platform"), "page": page}) + "\n")
        n += 1
    if not data or min(int(datetime.fromisoformat(x["date"]).timestamp()) for x in data) < cut:
        break
    time.sleep(2)
print("pages", page, "posts", n, "total_hits", d["meta"].get("total_hits"))
