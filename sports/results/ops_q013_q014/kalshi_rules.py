import json, time, requests
API = "https://api.elections.kalshi.com/trade-api/v2"; UA = "quant-research-logger/1.0 (personal read-only research)"
ev = json.load(open("results/ops_q013_q014/kalshi_settled.json")); out = {}
for e in sorted(ev):
    r = requests.get(f"{API}/markets", params={"event_ticker": e, "limit": 5}, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status(); out[e] = r.json()["markets"][0]["rules_secondary"]; time.sleep(0.3)
json.dump(out, open("results/ops_q013_q014/kalshi_rules.json", "w"), indent=0); print(len(out))
