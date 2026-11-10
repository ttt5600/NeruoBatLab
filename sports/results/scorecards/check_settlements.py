"""READ-ONLY cross-check: Kalshi /portfolio/settlements vs scorecards.ticker_pnl on the cached fills (NFLF 11-08)."""
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.expanduser('~/.claude/quant-kalshi/scripts'))
sys.path.append(str(Path(__file__).resolve().parents[2] / 'scripts'))
import kalshi_exec as kx
import scorecards as sc

cache = json.loads(sc.CACHE.read_text())['tickers']
e = kx.env()
c = kx.Kalshi(e, kx.config(e)['env'])
st = c._pages('/portfolio/settlements', 'settlements', {'min_ts': 1794100000})
by = {s['ticker']: s for s in st}
for tk, v in sorted(cache.items()):
    if 'NFL' not in tk or tk not in by:
        continue
    s = by[tk]
    mine, ok = sc.ticker_pnl(v['fills'], v['market'])
    keep = {k: s.get(k) for k in s if any(w in k for w in ('count', 'cost', 'revenue', 'fee', 'result', 'value'))}
    print(tk, f'mine {mine:+.2f}', json.dumps(keep))
