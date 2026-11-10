import json, collections, os
rows = [json.loads(l) for l in open(os.path.expanduser('~/.claude/quant-kalshi/data/lead_exec.jsonl'))]
nf = [r for r in rows if r.get('strategy') == 'NFLF' or r.get('key', '').startswith('NFLF')]
print(collections.Counter((r['event'], r.get('status'), r.get('leg') or r.get('action')) for r in nf))
for k, v in collections.Counter(tuple(sorted(r.keys())) for r in nf).items():
    print(v, k)
for r in nf:
    if r['event'] in ('order', 'settle') or 'exit' in json.dumps(r)[:2000] and r['event'] != 'update':
        print(json.dumps({k: v for k, v in r.items() if k not in ('response', 'source_row')})[:360])
