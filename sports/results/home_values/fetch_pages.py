"""Read-only: fetch the Zillow Home Values page (server-rendered, all-homes ZHVI) for each *HOME region and extract
zhviLatest + zhviRange. The page is the governing source (HOMEVALUE terms); it showed October 2026 on 2026-11-16 while
the Research CSV still ended at September. Output: results/home_values/pages/<UTC stamp>.json"""
import json, re, time
from datetime import datetime, timezone
from pathlib import Path
import requests

H = Path(__file__).resolve().parent
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
ser = json.load(open(H / "series.json"))
out = {}
for s in ser:
    url = [x["url"] for x in s["settlement_sources"] if "/home-values/" in x["url"]][0]
    rid = int(url.split("/home-values/")[1].split("/")[0])
    try:
        r = requests.get(url, headers={"User-Agent": UA, "Accept-Language": "en-US"}, timeout=30)
        t = r.text
        i = t.find('"zhviLatest"')
        if r.status_code != 200 or i < 0:
            out[s["ticker"]] = {"rid": rid, "error": f"http {r.status_code} zhviLatest {i}"}
        else:
            latest = json.loads(re.match(r'"zhviLatest":(\{.*?\})', t[i:]).group(1))
            j = t.find('"zhviRange":', i)
            k = t.find("]", j)
            rng = json.loads(t[j + len('"zhviRange":'):k + 1])
            out[s["ticker"]] = {"rid": rid, "latest": latest, "range": [(x["timePeriodEnd"], x["dataValue"]) for x in rng]}
    except Exception as e:  # noqa: BLE001
        out[s["ticker"]] = {"rid": rid, "error": repr(e)}
    time.sleep(4)
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
(H / "pages").mkdir(exist_ok=True)
json.dump({"fetched_utc": stamp, "pages": out}, open(H / "pages" / f"{stamp}.json", "w"))
print(stamp, {k: (v.get("latest", {}).get("timePeriodEnd"), round(v["latest"]["dataValue"]) if "latest" in v else v.get("error"))
              for k, v in out.items()})
