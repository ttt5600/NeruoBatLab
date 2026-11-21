"""R-TOK1 dev backtest (prereg notes/ai_nowcast.md s3). Primary = prereg (naive pace, s=LOWO sd). Exploratory = bias-corrected."""
import json,math,sys,numpy as np,pandas as pd
from scipy.stats import norm
sys.path.insert(0,'../../'); from edgelab.fees import kalshi_fee_per_contract
P=pd.read_csv('nowcast_panel.csv',parse_dates=['cap','cachedAt','wkstart'])
fin={r['x']:sum(r['ys'].values())/1e12 for r in json.load(open('final_byok_all_20261121.json'))['data']['data']}
ms=[m for m in json.load(open('kx_tokenuse_markets_20261121.json'))['markets'] if m['status'] in('finalized','settled')]
C=json.load(open('candles_kxtokenuse.json'))
def bucket(d): return 2 if d<3 else 3 if d<4 else 4 if d<5 else 5
P['b']=P.d.map(bucket)
mode=sys.argv[1] if len(sys.argv)>1 else 'prereg'
rows=[]
for m in ms:
    ev=m['event_ticker']; wend=pd.Timestamp(ev.split('-')[1].replace('26','2026-',1)) if False else None
    # event date code e.g. 26NOV16 -> week start = that Monday - 7d
    code=ev.split('-')[1]; dt=pd.to_datetime(code,format='%y%b%d'); wk=(dt-pd.Timedelta(days=7)).strftime('%Y-%m-%d')
    close=pd.Timestamp(m['close_time']).tz_localize(None); opn=pd.Timestamp(m['open_time']).tz_localize(None)
    K=float(m['floor_strike']); assert m['strike_type']=='greater'
    y=1 if m['result']=='yes' else 0
    caps=P[(P.wk==wk)&(P.d>=2.0)&(P.cap<close)&(P.cap>opn)].sort_values('cap')
    cs=sorted(C.get(m['ticker'],[]),key=lambda c:c['end_period_ts'])
    for _,c in caps.iterrows():
        T=c.cap; Tu=int((T-pd.Timestamp('1970-01-01'))//pd.Timedelta(seconds=1))
        prior=[x for x in cs if x['end_period_ts']<=Tu]
        if not prior: continue
        q=prior[-1]
        if Tu-q['end_period_ts']>7200: continue
        bid=float(q['yes_bid'].get('close_dollars') or 0); ask=float(q['yes_ask'].get('close_dollars') or 1)
        if bid<=0 or ask>=0.99 or ask-bid>0.10: continue
        oth=P[(P.wk!=wk)&(P.b==c.b)]
        s=oth.err.std(); mu=oth.err.mean() if mode=='biascorr' else 0.0
        if mode=='rmse': s=math.sqrt((oth.err**2).mean())
        N=c.N; N0=fin[(pd.Timestamp(wk)-pd.Timedelta(days=7)).strftime('%Y-%m-%d')]
        for arm,NN in (('rule',N),('P0',N0)):
            p=norm.cdf((math.log(NN)+mu-math.log(K))/s)
            fy=float(kalshi_fee_per_contract(ask,10)); noask=1-bid; fn=float(kalshi_fee_per_contract(noask,10))
            for side,pm,px,fee in (('yes',p,ask,fy),('no',1-p,noask,fn)):
                if pm>=0.80 and pm-px-fee>=0.05:
                    win=y if side=='yes' else 1-y
                    rows.append(dict(arm=arm,wk=wk,ticker=m['ticker'],K=K,side=side,T=T,d=c.d,N=NN,s=s,pm=pm,px=px,fee=fee,bid=bid,ask=ask,y=y,
                                     pnl=win-px-fee,stake=px+fee,mid=(bid+ask)/2,pmodel_yes=p))
R=pd.DataFrame(rows)
R=R.sort_values('T').drop_duplicates(['arm','ticker','side'])   # first qualifying T
R.to_csv(f'tok1_trades_{mode}.csv',index=False)
def boot(g,B=10000,seed=0):
    rng=np.random.default_rng(seed); wks=g.wk.unique(); agg=g.groupby('wk').agg(p=('pnl','sum'),s=('stake','sum'))
    pt=agg.p.sum()/agg.s.sum(); res=[]
    for _ in range(B):
        idx=rng.choice(len(agg),len(agg)); a=agg.iloc[idx]; res.append(a.p.sum()/a.s.sum())
    return pt,np.percentile(res,2.5),np.percentile(res,97.5)
for arm in ['rule','P0']:
    for split,f in (('dev',lambda x:x.wk<='2026-10-12'),('hold',lambda x:x.wk>'2026-10-12'),('all',lambda x:x.wk>'0')):
        g=R[(R.arm==arm)&f(R)]
        if len(g)==0: print(arm,split,'n=0'); continue
        pt,lo,hi=boot(g)
        print(f"{mode} {arm:4s} {split:4s} n={len(g):3d} weeks={g.wk.nunique():2d} wins={int(((g.side=='yes')&(g.y==1)|(g.side=='no')&(g.y==0)).sum())} ROI={pt:+.4f} [{lo:+.4f},{hi:+.4f}] meanpx={g.px.mean():.3f}")
