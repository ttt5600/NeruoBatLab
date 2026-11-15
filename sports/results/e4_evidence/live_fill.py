"""Live maker fill + filled/unfilled loss behaviour from kalshi_exec.jsonl + kalshi_paper.jsonl (read-only).
Run from sports/: ../.venv_quant/bin/python results/e4_evidence/live_fill.py"""
import json
import pandas as pd
r = [json.loads(l) for l in open("/Users/jonathanwang/.claude/quant-kalshi/data/kalshi_exec.jsonl")]
d = pd.DataFrame(r)
o = d[(d.event == "order") & (d["mode"] == "live") & (d.status == "submitting")].copy()
o = o[o.exec_style == "maker"]
u = d[(d.event == "update") & (d["mode"] == "live")].copy()
# final non-resting state per ticker (one maker order per ticker)
fin = u[u.status != "resting"].sort_values("ts").groupby("ticker").last()
last = u.sort_values("ts").groupby("ticker").last()
o["final"] = o.ticker.map(fin.status).fillna(o.ticker.map(last.status))
o["filled_ct"] = o.ticker.map(fin.filled)
o["err"] = o.ticker.map(fin.get("error")) if "error" in fin else None
pap = pd.DataFrame([json.loads(l) for l in open("/Users/jonathanwang/.claude/quant-kalshi/data/kalshi_paper.jsonl")]).drop_duplicates("ticker")
o["result"] = o.ticker.map(pap.set_index("ticker").result)
o["x1"] = o.ticker.str.contains("KXHIGHT")
o["day"] = o.ts.str[:10]
print(o.groupby(["x1", "final"]).size())
print(o.groupby(["x1", "final", "result"], dropna=False).size())
print(o[~o.x1][["day", "ticker", "count", "maker_q", "final", "filled_ct", "result"]].to_string())
o.to_csv("results/e4_evidence/live_maker_orders.csv", index=False)
