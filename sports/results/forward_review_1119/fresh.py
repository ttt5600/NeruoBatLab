import gzip, json, os, sys, time, glob, re
from collections import Counter
paths = sys.argv[1:]
for p in paths:
    for f in sorted(glob.glob(p)):
        if os.path.isdir(f): continue
        op = gzip.open if f.endswith('.gz') else open
        try:
            lines = op(f,'rt').read().splitlines()
        except Exception as e:
            print('ERR', f, e); continue
        days = Counter()
        for l in lines:
            m = re.search(r'"(?:ts|as_of|decided_at|logged_at|time|run_ts|asof|t)"\s*:\s*"(20\d\d-\d\d-\d\d)', l)
            if m: days[m.group(1)] += 1
        recent = {d: days[d] for d in sorted(days)[-6:]}
        print(time.strftime('%m-%d %H:%M', time.localtime(os.path.getmtime(f))), len(lines), f.replace(os.path.expanduser('~'),'~'), recent)
