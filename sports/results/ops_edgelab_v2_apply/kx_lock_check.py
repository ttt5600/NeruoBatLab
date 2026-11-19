"""Which kx_fwd forward trials already have a scored forward evaluation (adapter lock engaged)? Run from sports/."""
import json
from collections import Counter

ev = [json.loads(l) for l in open('lab/edgelab_registry.jsonl')]
reg = {e['trial_id']: e['spec'] for e in ev if e['event'] == 'register'}
c = Counter()
for e in ev:
    if e['event'] == 'evaluate' and e.get('split') == 'forward':
        s = reg.get(e['trial_id'], {})
        if 'kx_fwd' in json.dumps(s):
            locked = bool(e.get('forward_adapter')) and (e.get('n') or 0) > 0
            c[(s.get('id'), locked)] += 1
for k, v in sorted(c.items(), key=str):
    print(k, v)
