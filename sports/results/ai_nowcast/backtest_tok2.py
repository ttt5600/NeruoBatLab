import json,sys,numpy as np,pandas as pd
sys.path.insert(0,'../../'); from edgelab.fees import kalshi_fee_per_contract
fin={r['x']:sum(r['ys'].values())/1e12 for r in json.load(open('final_byok_all_20261121.json'))['data']['data']}
ms=[m for m in json.load(open('kx_tokenuse_markets_20261121.json'))['markets'] if m['status'] in('finalized','settled')]
C=json.load(open('candles_kxtokenuse.json'))
rows=[];desc=[]
for m in ms:
    code=m['event_ticker'].split('-')[1]; dt=pd.to_datetime(code,format='%y%b%d'); wk=(dt-pd.Timedelta(days=7)).strftime('%Y-%m-%d')
    V=fin[wk]; V0=fin[(dt-pd.Timedelta(days=14)).strftime('%Y-%m-%d')]
    wend=int((dt-pd.Timestamp('1970-01-01'))//pd.Timedelta(seconds=1))   # Mon 00:00Z
    close=int((pd.Timestamp(m['close_time']).tz_localize(None)-pd.Timestamp('1970-01-01'))//pd.Timedelta(seconds=1))
    K=float(m['floor_strike']); y=1 if m['result']=='yes' else 0
    cs=sorted(C[m['ticker']],key=lambda c:c['end_period_ts'])
    for c in cs:
        t=c['end_period_ts']
        if t<wend or t>close+60: continue
        bid=float(c['yes_bid'].get('close_dollars') or 0); ask=float(c['yes_ask'].get('close_dollars') or 1)
        desc.append(dict(wk=wk,ticker=m['ticker'],K=K,V=V,hr=(t-wend)/3600,bid=bid,ask=ask,y=y,vol=float(c.get('volume_fp') or 0)))
        if t<wend+3600: continue
        if bid<=0 or ask>=0.99: continue
        for arm,VV in (('rule',V),('P0',V0)):
            for side,cond,px in (('yes',K<=VV*0.995,ask),('no',K>=VV*1.005,1-bid)):
                fee=float(kalshi_fee_per_contract(px,10))
                if cond and px+fee<=0.97:
                    win=y if side=='yes' else 1-y
                    rows.append(dict(arm=arm,wk=wk,ticker=m['ticker'],K=K,V=VV,side=side,hr=(t-wend)/3600,px=px,fee=fee,y=y,pnl=win-px-fee,stake=px+fee))
D=pd.DataFrame(desc); D.to_csv('tok2_boundary_quotes.csv',index=False)
R=pd.DataFrame(rows)
pd.set_option('display.width',220)
print('boundary candles',len(D),'rungs',D.ticker.nunique(),'two-sided',int(((D.bid>0)&(D.ask<0.99)).sum()))
if len(R):
    R=R.sort_values('hr').drop_duplicates(['arm','ticker','side']); R.to_csv('tok2_trades.csv',index=False)
    print(R.round(3).to_string())
    for arm in ['rule','P0']:
        g=R[R.arm==arm]
        if not len(g): print(arm,'n=0'); continue
        agg=g.groupby('wk').agg(p=('pnl','sum'),s=('stake','sum')); rng=np.random.default_rng(0)
        bs=[agg.iloc[rng.choice(len(agg),len(agg))].pipe(lambda a:a.p.sum()/a.s.sum()) for _ in range(10000)]
        print(arm,'n',len(g),'weeks',g.wk.nunique(),'ROI %+.4f [%+.4f,%+.4f]'%(agg.p.sum()/agg.s.sum(),np.percentile(bs,2.5),np.percentile(bs,97.5)))
else: print('no qualifying trades')
# descriptive: rungs within 3% of V at hr 0..1
D['dist']=D.K/D.V-1
print(D[(D.hr<=1.01)].sort_values(['wk','K'])[['wk','K','V','dist','hr','bid','ask','y']].round(3).to_string())
