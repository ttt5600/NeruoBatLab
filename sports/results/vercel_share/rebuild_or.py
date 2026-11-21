"""Rebuild settled OpenRouter share series (KXANTHSHARE/GOOG/DEEP/OPEN; author share of weekly tokens, 1 dp) and KXTOPUSAGEAI
(author of the single top model by weekly tokens) from the free endpoints, BYOK-default vs byok=all."""
import json, glob, re
import pandas as pd
from datetime import datetime
def wk(f):
    b=json.load(open(sorted(glob.glob(f))[-1]))["body"]; b=b.get("data",b); b=b.get("data",b) if isinstance(b,dict) else b
    return {r["x"]:r["ys"] for r in b}
A={"default":wk("or/authreq_text_2*.json"),"byokall":wk("or/authreq_text_byokall_*.json")}
M={"default":wk("or/models_default_*.json"),"byokall":wk("or/models_byokall_*.json")}
AUTH={"KXANTHSHARE":"anthropic","KXGOOGSHARE":"google","KXDEEPSHARE":"deepseek","KXOPENSHARE":"openai"}
rows=[]
for s,a in AUTH.items():
    mk=json.load(open(f"mk/{s}.json")); ev={}
    for m in mk:
        if m["status"]=="finalized": ev.setdefault(m["event_ticker"],m)
    for e,m in sorted(ev.items()):
        d=re.search(r"week of (\w+ \d+, \d{4})",m["rules_primary"]).group(1)
        w=datetime.strptime(d,"%b %d, %Y").strftime("%Y-%m-%d")
        try: kv=float(m["expiration_value"].replace("%",""))
        except ValueError: kv=float("nan")
        r=dict(series=s,event=e,week=w,kalshi=kv)
        for v in ("default","byokall"):
            ys=A[v].get(w)
            if ys: r[v]=round(100*ys.get(a,0)/sum(ys.values()),3)
        rows.append(r)
df=pd.DataFrame(rows)
for v in ("default","byokall"): df[f"ok_{v}"]=(df[v].round(1)-df.kalshi).abs()<0.05
print(df.to_string()); print(df[["ok_default","ok_byokall"]].sum(), len(df))
# TOPUSAGEAI
mk=json.load(open("mk/KXTOPUSAGEAI.json")); ev={}
for m in mk:
    if m["status"]=="finalized": ev.setdefault(m["event_ticker"],[]).append(m)
for e,ms in sorted(ev.items()):
    win=[m["yes_sub_title"] for m in ms if m["result"]=="yes"]
    d=re.search(r"week of (\w+ \d+, \d{4})",ms[0]["rules_primary"]).group(1); w=datetime.strptime(d,"%b %d, %Y").strftime("%Y-%m-%d")
    out=[]
    for v in ("default","byokall"):
        ys=M[v].get(w,{}); top=max(((k,x) for k,x in ys.items() if k!="Others"),key=lambda t:t[1],default=None)
        out.append((v,top[0] if top else None))
    print(e,w,"kalshi yes:",win,out)
df.to_csv("rebuild_or.csv",index=False)
