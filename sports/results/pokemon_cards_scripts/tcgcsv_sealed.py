import json, sys, re, glob, os, csv
# usage: tcgcsv_sealed.py DIR GROUPS_JSON OUT_CSV ; prints sealed SKUs (ETB, booster bundle, booster box/display, single pack) with market/low prices
d, gpath, out = sys.argv[1], sys.argv[2], sys.argv[3]
groups = {g['groupId']: g for g in json.load(open(gpath))['results']}
pat = re.compile(r'(Elite Trainer Box|Booster Bundle|Booster Box|Booster Display|Booster Pack$|Sleeved Booster)', re.I)
rows = []
for pf in sorted(glob.glob(os.path.join(d, '*_products.json'))):
    gid = int(os.path.basename(pf).split('_')[0])
    pr = os.path.join(d, f'{gid}_prices.json')
    if not os.path.exists(pr) or os.path.getsize(pr) < 100:
        print('missing prices', gid); continue
    prods = {p['productId']: p for p in json.load(open(pf))['results']}
    for q in json.load(open(pr))['results']:
        p = prods.get(q['productId'])
        if not p or not pat.search(p['name']) or 'Case' in p['name']: continue
        rows.append(dict(group=groups[gid]['name'], released=groups[gid]['publishedOn'][:10], product=p['name'],
                         subType=q.get('subTypeName'), market=q.get('marketPrice'), low=q.get('lowPrice'), mid=q.get('midPrice')))
rows.sort(key=lambda r: (r['released'], r['product']))
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for r in rows: print(r['released'], '|', r['group'], '|', r['product'], '|', r['market'], '| low', r['low'])
