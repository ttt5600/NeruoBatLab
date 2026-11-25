import sys, re, json, datetime as dt
# usage: pc_chart.py FILE [KEY...]; prints monthly series (USD) from PriceCharting chart_data (values in cents, ts in ms)
path = sys.argv[1]; keys = sys.argv[2:]
html = open(path, encoding='utf-8', errors='replace').read()
m = re.search(r'chart_data\s*=\s*(\{.*?\});', html, re.S)
d = json.loads(m.group(1))
print('keys:', list(d.keys()))
title = re.search(r'<title>(.*?)</title>', html, re.S)
print('title:', title.group(1).strip() if title else None)
for k in (keys or list(d.keys())):
    ser = d.get(k, [])
    out = []
    for ts, v in ser:
        t = dt.datetime.fromtimestamp(ts/1000, dt.timezone.utc)  # ms -> s explicitly (traps: unit check)
        if t.year >= 2025:
            out.append(f"{t:%Y-%m}:{v/100:.0f}")
    print(k, ' '.join(out))
