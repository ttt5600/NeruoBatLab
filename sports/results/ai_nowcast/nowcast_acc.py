import json,pandas as pd,numpy as np
p=pd.read_csv('wayback_panel_raw.csv',parse_dates=['cap','cachedAt'])
fin={r['x']:sum(r['ys'].values())/1e12 for r in json.load(open('final_byok_all_20261121.json'))['data']['data']}
p['wkstart']=pd.to_datetime(p.wk)
LAG=pd.Timedelta(hours=1)
p['d']=((p.cachedAt-LAG)-p.wkstart)/pd.Timedelta(days=1)   # traps: Timedelta division, no int64//1e9
p['N']=p.partial*7/p.d
p['final']=p.wk.map(fin)
# exclude current (unsettled) week 2026-11-16
p=p[p.wk<'2026-11-16'].copy()
p['err']=np.log(p.final/p.N)
p['bucket']=pd.cut(p.d,[0,1,2,3,4,5,6,7],right=False)
pd.set_option('display.width',200)
print(p[['cap','wk','d','partial','N','final','err']].round(3).to_string())
g=p[p.d>=0.5].groupby('bucket',observed=True).err.agg(['count','mean','std',lambda x: np.mean(np.abs(x))])
print(g.round(4))
p.to_csv('nowcast_panel.csv',index=False)
