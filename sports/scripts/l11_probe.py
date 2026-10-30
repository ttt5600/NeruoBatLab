"""L11 step 1 helper: run ad-hoc SQL against DoltHub post-no-preference/options (coverage probes only)."""
import json
import sys
import time

import requests

API = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"


def q(s):
    t = time.time()
    r = requests.get(API, params={"q": s}, timeout=150)
    d = json.loads(r.text)
    return d.get("query_execution_status"), d.get("query_execution_message", "")[:200], d.get("rows", [])[:60], round(time.time() - t, 1)


if __name__ == "__main__":
    for s in sys.argv[1:]:
        print(s[:140], q(s), flush=True)
        time.sleep(0.6)
