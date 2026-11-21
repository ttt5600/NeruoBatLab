import json,math,numpy as np,pandas as pd
from scipy.stats import norm
P=pd.read_csv('nowcast_panel.csv',parse_dates=['cap'])
ms=[m for m in json.load(open('kx_tokenuse_markets_20261121.json'))['markets'] if m['status'] in('finalized','settled')]
C=json.load(open('candles_kxtokenuse.json'))
b=lambda d:2 if d<3 else 3 if d<4 else 4 if d<5 else 5
P['b']=P.d.map(b); rows=[]
for m in ms:
    code=m['event_ticker'].split('-')[1]; dt=pd.to_datetime(code,format='%y%b%d'); wk=(dt-pd.Timedelta(days=7)).strftime('%Y-%m-%d')
    close=pd.Timestamp(m['close_time']).tz_localize(None); K=float(m['floor_strike']); y=1 if m['result']=='yes' else 0
    cs=sorted(C[m['ticker']],key=lambda c:c['end_period_ts'])
    for _,c in P[(P.wk==wk)&(P.d>=2)&(P.cap<close)].iterrows():
        Tu=int((c.cap-pd.Timestamp('1970-01-01'))//pd.Timedelta(seconds=1)); pr=[x for x in cs if x['end_period_ts']<=Tu and Tu-x['end_period_ts']<=7200]
        if not pr: continue
        bid=float(pr[-1]['yes_bid'].get('close_dollars') or 0); ask=float(pr[-1]['yes_ask'].get('close_dollars') or 1)
        if bid<=0 or ask>=0.99 or ask-bid>0.10: continue
        o=P[(P.wk!=wk)&(P.b==c.b)]
        p0=norm.cdf((math.log(c.N)-math.log(K))/o.err.std()); p1=norm.cdf((math.log(c.N)+o.err.mean()-math.log(K))/o.err.std())
        rows.append(dict(wk=wk,y=y,mid=(bid+ask)/2,p_pre=p0,p_bc=p1))
R=pd.DataFrame(rows); print('rows',len(R),'weeks',R.wk.nunique())
for c in ['mid','p_pre','p_bc']: print(c,'Brier %.4f'%((R[c]-R.y)**2).mean())
rng=np.random.default_rng(0); wks=R.wk.unique()
for c in ['p_pre','p_bc']:
    d=[];
    for _ in range(5000):
        s=pd.concat([R[R.wk==w] for w in rng.choice(wks,len(wks))]); d.append(((s.mid-s.y)**2).mean()-((s[c]-s.y)**2).mean())
    print(c,'Brier(mid)-Brier(model) %+.4f [%+.4f,%+.4f]'%(((R.mid-R.y)**2).mean()-((R[c]-R.y)**2).mean(),np.percentile(d,2.5),np.percentile(d,97.5)))
