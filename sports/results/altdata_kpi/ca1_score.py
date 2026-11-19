"""CA1 dev scoring per prereg (notes/altdata_kpi.md). Gate first, then ROI."""
import json,numpy as np,pandas as pd,sys
sys.path.insert(0,".")
from edgelab.fees import kalshi_fee_per_contract
F=pd.read_csv("results/altdata_kpi/ca1_far_dev.csv"); C=json.load(open("results/altdata_kpi/ca1_candles.json"))
rows=[]
for _,f in F.iterrows():
    c=C[f.ticker]; cs=[x for x in c["c"] if x["end_period_ts"]<=c["T"]]
    ya=yb=None
    if cs:
        l=cs[-1]; ya=l.get("yes_ask",{}).get("close_dollars"); yb=l.get("yes_bid",{}).get("close_dollars")
    ya=float(ya) if ya not in (None,"") else np.nan; yb=float(yb) if yb not in (None,"") else np.nan
    two=(ya>0)&(ya<1)&(yb>0)&(yb<1) if not(np.isnan(ya) or np.isnan(yb)) else False
    ask=ya if f.side=="yes" else 1-yb
    rows.append(dict(f, ya=ya, yb=yb, two=two, ask=ask if two else np.nan))
R=pd.DataFrame(rows)
print("far rungs",len(R),"two-sided at T",int(R.two.sum()))
print("side ask distribution (two-sided):",R[R.two].ask.describe().round(3).to_dict())
print("frac side ask > .97:",round((R[R.two].ask>0.97).mean(),3))
B=R[R.two&(R.ask>=0.50)&(R.ask<=0.97)].copy()
B["win"]=np.where(B.side=="yes",B.result=="yes",B.result=="no")
B["fee"]=kalshi_fee_per_contract(B.ask.values,1,rounding="exact")
B["pnl"]=B.win.astype(float)-B.ask-B.fee; B["roi"]=B.pnl/(B.ask+B.fee)
R.to_csv("results/altdata_kpi/ca1_dev_rows.csv",index=False); B.to_csv("results/altdata_kpi/ca1_dev_bets.csv",index=False)
print("bets",len(B),"events",B.event.nunique(),"losses",int((~B.win).sum()))
if len(B):
    roi=B.pnl.sum()/(B.ask+B.fee).sum()
    rng=np.random.default_rng(0); ev=B.event.unique(); g={e:B[B.event==e] for e in ev}; bs=[]
    for _ in range(5000):
        s=pd.concat([g[e] for e in rng.choice(ev,len(ev))]); bs.append(s.pnl.sum()/(s.ask+s.fee).sum())
    print("ROI",round(roi,4),"CI95",np.round(np.percentile(bs,[2.5,97.5]),4))
    B["Tord"]=B.ticker.map(lambda t:C[t]["T"]); first=B.sort_values("Tord").head(20)
    print("losses in first 20",int((~first.win).sum()))
    print(B[["ticker","side","K","v_prev","z","ask","result"]].sort_values("ask").to_string())
