"""Fold kalshi_exec.jsonl (live) into one row per ticker, join paper results. Read-only."""
import json, sys
import pandas as pd
D = "/Users/jonathanwang/.claude/quant-kalshi/data/"
hist = [json.loads(l) for l in open(D + "kalshi_exec.jsonl")]
picks = {}
for l in open(D + "kalshi_paper.jsonl"):
    p = json.loads(l.replace("NaN", "null"))
    picks[p["ticker"]] = p
out = {}
for r in hist:
    if r.get("mode") != "live" or "ticker" not in r:
        continue
    ev = r.get("event", "order"); t = r["ticker"]
    if ev == "order" and t not in out:
        out[t] = dict(r); out[t]["order_ts"] = r["ts"]; out[t]["order_status"] = r.get("status")
    elif ev == "order":
        out[t]["dup_orders"] = out[t].get("dup_orders", 0) + 1
    elif ev == "update" and t in out:
        out[t].update({k: v for k, v in r.items() if k not in ("event", "ts", "mode", "ticker", "response")})
        out[t]["last_update_ts"] = r["ts"]; out[t]["nupd"] = out[t].get("nupd", 0) + 1
    elif ev == "settle" and t in out:
        out[t]["nsettle"] = out[t].get("nsettle", 0) + 1
        out[t].update(settled=True, result=r["result"], pnl=r["pnl"], held_settle=r["held"], settled_ts=r["ts"])
    elif ev == "settle":
        print("settle without order:", t)
df = pd.DataFrame(out.values())
df["exec_style"] = df["exec_style"].fillna("taker")
df["paper_result"] = df["ticker"].map(lambda t: (picks.get(t) or {}).get("result"))
df["city"] = df["ticker"].str.split("-").str[0]
df["edate"] = df["ticker"].str.split("-").str[1]
df.to_parquet(sys.argv[1])
print(df.groupby(["exec_style", "status"], dropna=False).size())
