"""READ-ONLY: GET own fills/positions for every NFLF + NHL ticker in lead_exec.jsonl. No POST/DELETE."""
import json, os, sys, collections
sys.path.insert(0, os.path.expanduser('~/.claude/quant-kalshi/scripts'))
import kalshi_exec as kx

rows = [json.loads(l) for l in open(os.path.expanduser('~/.claude/quant-kalshi/data/lead_exec.jsonl'))]
tks = sorted({r['ticker'] for r in rows if r.get('key', '').startswith(('NFLF', 'NHL')) and r['event'] == 'settle'
              or (r.get('strategy') in ('NFLF', 'NHL') and r.get('status') == 'submitting')} |
             {r['ticker'] for r in rows if r.get('key', '').startswith(('NFLF', 'NHL')) and r.get('status') == 'filled'})
e = kx.env()
c = kx.Kalshi(e, kx.config(e)['env'])
for tk in tks[:1] + [t for t in tks if "NYJ" in t]:
    fills = c._pages('/portfolio/fills', 'fills', {'ticker': tk})
    print(json.dumps(fills[:2]))
    code, out = c.request('GET', '/portfolio/positions', params={'ticker': tk})
    pos = (out.get('market_positions') or [{}])
    pos = pos[0] if pos else {}
    m = c.market(tk) or {}
    agg = collections.Counter((f['action'], f['side']) for f in fills)
    n = collections.defaultdict(float)
    for f in fills:
        n[(f['action'], f['side'])] += float(f['count_fp'])
    print(tk, 'res', m.get('result'), m.get('status'), 'fills', dict(agg), {k: round(v, 2) for k, v in n.items()},
          'pos', pos.get('position_fp'), 'rpnl', pos.get('realized_pnl_dollars'), 'fees', pos.get('fees_paid_dollars'))
