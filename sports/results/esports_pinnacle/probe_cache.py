"""Probe: does a cache-busting query param on the Pinnacle guest API return fresher odds than the CF-cached URL?"""
import random
import requests

K = 'CmX2KcMrXuFmNg6YFbmTxE0y9CIrOi0R'
H = {"User-Agent": "Mozilla/5.0", "X-API-Key": K}
B = "https://guest.api.arcadia.pinnacle.com/0.1/sports/12/markets/straight"
rc = requests.get(B, headers=H, timeout=30)
rb = requests.get(B, params={"_": random.randint(1, 10**9)}, headers=H, timeout=30)
for n, r in (("cached", rc), ("busted", rb)):
    print(n, {k: r.headers.get(k) for k in ("Date", "Age", "last-modified", "cf-cache-status", "Cache-Control")})
c = {x['key'] + str(x['matchupId']): x for x in rc.json()}
b = {x['key'] + str(x['matchupId']): x for x in rb.json()}
both = set(c) & set(b)
d = [k for k in both if c[k]['version'] != b[k]['version']]
print(len(c), len(b), len(both), "version differs", len(d), "busted newer", sum(b[k]['version'] > c[k]['version'] for k in d))
for k in d[:5]:
    print(k, c[k]['version'], b[k]['version'], [p['price'] for p in c[k]['prices']][:3], [p['price'] for p in b[k]['prices']][:3])
