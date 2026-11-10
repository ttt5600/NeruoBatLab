import json, collections, os
rows = [json.loads(l) for l in open(os.path.expanduser('~/.claude/quant-kalshi/data/lead_exec.jsonl'))]
nf = [r for r in rows if r.get('key', '').startswith('NFLF')]
att = collections.defaultdict(int)
for r in nf:
    if 'exit_attempts' in r:
        att[r['key']] = max(att[r['key']], r['exit_attempts'])
print('max exit attempts per key', dict(att))
for r in nf:
    if any(k in r for k in ('exit_filled', 'exit_gave_up')) or r['event'] == 'settle':
        print(json.dumps({k: v for k, v in r.items() if k not in ('exit_raw',)})[:300])
print('--- fills')
for r in nf:
    if r.get('status') == 'filled' or (r['event'] == 'update' and r.get('filled')):
        print(r['ts'], r['key'][-14:], r['ticker'][-12:], r.get('filled'), r.get('avg_price'), r.get('cost'), r.get('status'))
print('--- submitting orders')
for r in nf:
    if r['event'] == 'order' and r['status'] == 'submitting':
        print(r['ts'], r['key'], r['ticker'], r.get('count'), r.get('limit_price'), r.get('exit_at'))
