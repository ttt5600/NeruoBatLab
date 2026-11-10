import json, collections, os, glob
H = os.path.expanduser
files = ['~/.claude/quant-sports/data/l32b_log.jsonl', '~/.claude/quant-kalshi/data/open_lines_nhl_picks.jsonl',
         '~/.claude/quant-kalshi/data/open_lines_nfl_picks.jsonl', '~/.claude/quant-kalshi/data/open_totals_picks.jsonl',
         '~/.claude/quant-sports/data/nba_early_form_log.jsonl', '~/.claude/quant-kalshi/data/bets.jsonl',
         '~/.claude/quant-kalshi/data/manual_bets.jsonl', '~/.claude/quant-kalshi/data/kalshi_wx_monthly_picks.jsonl',
         '~/.claude/quant-kalshi/data/kalshi_wx_monthly_evals.jsonl'] + glob.glob(H('~/.claude/quant-sports/data/e1x/*'))
for f in files:
    f = H(f)
    print('=====', f)
    if os.path.isdir(f):
        print(os.listdir(f)); continue
    if not f.endswith('.jsonl'):
        print('size', os.path.getsize(f)); continue
    rows = [json.loads(l) for l in open(f) if l.strip()]
    kc = collections.Counter(r.get('kind') or r.get('event') or r.get('type') or r.get('strategy') for r in rows)
    print(len(rows), dict(kc.most_common(12)))
    shown = set()
    for r in rows:
        k = r.get('kind') or r.get('event') or r.get('type') or r.get('strategy')
        if k in shown: continue
        shown.add(k)
        print('  ', json.dumps(r)[:700])
