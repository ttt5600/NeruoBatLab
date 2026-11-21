import json,glob,gzip,pandas as pd,sys
def load(f):
    b=open(f,'rb').read()
    if b[:2]==b'\x1f\x8b': b=gzip.decompress(b)
    return json.loads(b)
rows=[]
for f in sorted(glob.glob('model-rankings-chart_*.json')):
    ts=f.split('_')[1][:14]
    j=load(f); d=j['data']; ca=None; byok=None
    if isinstance(d,dict): ca=d.get('cachedAt'); byok=d.get('significantByokSeries'); d=d['data']
    tot={r['x']:sum(r['ys'].values())/1e12 for r in d}
    last=d[-1]
    rows.append(dict(cap=pd.Timestamp(ts),cachedAt=pd.Timestamp(ca,unit='ms') if ca else pd.NaT,wk=last['x'],partial=tot[last['x']],
       prevwk=d[-2]['x'],prev=tot[d[-2]['x']],nbars=len(d),hasbyok=byok is not None,alltot=json.dumps(tot)))
df=pd.DataFrame(rows); df.to_csv('../wayback_panel_raw.csv',index=False)
pd.set_option('display.width',220)
print(df.drop(columns='alltot').to_string())
