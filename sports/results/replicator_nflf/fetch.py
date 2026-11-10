"""READ-ONLY GETs: fills + settlements for KXNFLGAME since 2026-11-07. Dumps raw JSON."""
import json, os, sys
from datetime import datetime, timezone
sys.path.insert(0, os.path.expanduser('~/.claude/quant-kalshi/scripts'))
import kalshi_exec as kx

e = kx.env()
c = kx.Kalshi(e, kx.config(e)['env'])
min_ts = int(datetime(2026, 11, 7, tzinfo=timezone.utc).timestamp())
fills = c._pages('/portfolio/fills', 'fills', {'min_ts': min_ts})
sets = c._pages('/portfolio/settlements', 'settlements', {'min_ts': min_ts})
fills = [f for f in fills if f.get('ticker', '').startswith('KXNFLGAME')]
sets = [s for s in sets if s.get('ticker', '').startswith('KXNFLGAME')]
tks = sorted({f['ticker'] for f in fills})
pos = {}
mk = {}
for tk in tks:
    code, out = c.request('GET', '/portfolio/positions', params={'ticker': tk})
    pos[tk] = out.get('market_positions')
    mk[tk] = c.market(tk)
out = os.path.join(os.path.dirname(__file__), 'raw.json')
json.dump({'fills': fills, 'settlements': sets, 'positions': pos, 'markets': mk}, open(out, 'w'), indent=1)
print(len(fills), 'fills', len(sets), 'settlements', len(tks), 'tickers')
