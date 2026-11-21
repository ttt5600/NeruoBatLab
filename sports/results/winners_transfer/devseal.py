import csv, random, statistics as st, json
rows=list(csv.DictReader(open('devseal_pairs.csv')))
for r in rows:
    r['k']=float(r['sealed_roi'])/float(r['dev_roi'])
    r['pass']=r['sealed_verdict'].startswith('pass')
    r['now']=r['cls']=='NOWCAST'
out={}
def boot_med(ks,B=2000,seed=0):
    rnd=random.Random(seed); m=[st.median([rnd.choice(ks) for _ in ks]) for _ in range(B)]
    m.sort(); return st.median(ks),m[int(.05*B)],m[int(.95*B)]
groups={'NOWCAST':[r for r in rows if r['now']],'OTHER_nonNOWCAST_nonXV':[r for r in rows if r['cls'] not in('NOWCAST','CROSSVENUE')],
        'NOWCAST_passes_only':[r for r in rows if r['now'] and r['pass']]}
for g,rs in groups.items():
    ks=[r['k'] for r in rs]; med,lo,hi=boot_med(ks)
    out[g]={'n':len(ks),'k_median':round(med,3),'k_90ci':[round(lo,3),round(hi,3)],'k_each':{r['id']:round(r['k'],3) for r in rs},
            'k_pooled_n_weighted':None}
# LOO rules
def cls(r): return 'NOWCAST' if r['now'] else ('CROSSVENUE' if r['cls']=='CROSSVENUE' else 'OTHER')
res={'A':0,'B':0,'C':0}; detail=[]
for i,r in enumerate(rows):
    others=[x for j,x in enumerate(rows) if j!=i and cls(x)==cls(r)]
    kc=st.median([x['k'] for x in others]) if others else 1.0
    predB=kc*float(r['dev_roi'])>0
    predA=True; predC=r['now'] or r['cls']=='CROSSVENUE'
    res['A']+=predA==r['pass']; res['B']+=predB==r['pass']; res['C']+=predC==r['pass']
    detail.append((r['id'],cls(r),round(kc,3),round(kc*float(r['dev_roi']),4),r['pass'],predB,predC))
out['LOO_accuracy']={k:f"{v}/{len(rows)}" for k,v in res.items()}
out['LOO_detail(id,cls,k_class_minus_i,pred_sealed_roi,actual_pass,predB,predC)']=detail
print(json.dumps(out,indent=1))
json.dump(out,open('devseal_result.json','w'),indent=1)
