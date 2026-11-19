"""Q014 kill-rule precondition: RC count at window end vs Kalshi expiration_value, per settled event (>10% disagree -> fix first)."""
import json, urllib.request, time, os
D = os.path.expanduser('~/.claude/quant-kalshi/data/q014')
counts = [json.loads(l) for l in open(f'{D}/counts.jsonl') if l.strip()]
picks = [json.loads(l) for l in open(f'{D}/picks.jsonl') if l.strip()]
events = sorted({p['event'] for p in picks if p.get('side')})
last = {}
for c in counts:
    if c['ts'] <= c['we'] + 300:
        last[c['event']] = c
rows = []
for ev in events:
    u = f'https://api.elections.kalshi.com/trade-api/v2/markets?event_ticker={ev}&limit=100'
    js = json.load(urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'fr1119'}), timeout=20))
    ms = js.get('markets', [])
    st = {m.get('status') for m in ms}
    ev_val = {m.get('expiration_value') for m in ms if m.get('expiration_value')}
    c = last.get(ev)
    rows.append((ev, sorted(st), sorted(ev_val), c and c['count'], c and c['count_incl_deleted'], c and time.strftime('%m-%d %H:%M', time.gmtime(c['ts'])), c and time.strftime('%m-%d %H:%M', time.gmtime(c['we']))))
    time.sleep(0.3)
n = d = 0
for r in rows:
    print(r)
    if r[2] and r[3] is not None:
        n += 1
        try:
            if abs(float(r[2][0]) - r[3]) > 0.5: d += 1
        except ValueError:
            pass
print(f'settled events compared {n}, disagree {d}')
