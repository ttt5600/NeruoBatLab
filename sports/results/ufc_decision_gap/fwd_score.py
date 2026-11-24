import json
from datetime import datetime
P=lambda s: datetime.fromisoformat(s.replace("Z","+00:00"))
rows=[json.loads(l) for l in open("fwd/1348.jsonl")]
errs=[P(r["ts"]) for r in rows if r["kind"]=="error"]
bb=lambda r: max((float(p) for p,q in (r["ob"].get("orderbook_fp",{}).get("yes_dollars") or [])), default=0)
res=[]
for d in [r for r in rows if r["kind"]=="action" and r["type"]=="unofficial_winner_decision"]:
    f=d["fight"]; A={r["type"]:r for r in rows if r["kind"]=="action" and r["fight"]==f}
    uw=P(A["unofficial_winner_decision"]["action_ts"]); over=P(A["fight_over"]["action_ts"])
    rs=min(P(r["ts"]) for r in rows if r["kind"]=="result_seen" and r["fight"]==f)
    obs=[r for r in rows if r["kind"]=="orderbook" and r["fight"]==f]
    tk=sorted({r["ticker"] for r in obs})
    last={t:bb([r for r in obs if r["ticker"]==t][-1]) for t in tk}
    w=max(last,key=last.get)
    wo=[r for r in obs if r["ticker"]==w]
    at_over=[bb(r) for r in wo if P(r["ts"])<=over]; p_over=at_over[-1] if at_over else None
    j=next((P(r["ts"]) for r in wo if P(r["ts"])>over and bb(r)>=0.95),None)
    # JSON health: errors between uw-30s and rs
    e=sum(1 for x in errs if (uw-x).total_seconds()<=30 and x<=rs)
    res.append((f,w[-3:],p_over,round((j-uw).total_seconds(),1) if j else None,round((rs-uw).total_seconds(),1),e))
print("fight winner bid@horn jump95-uw(s) json_result-uw(s) json_errors_in_window")
for r in res: print(r)
