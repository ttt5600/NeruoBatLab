"""READ-ONLY: GET market results + own fills/positions for the BUFMIN incident tickers. No POST/DELETE."""
import json, os, sys
sys.path.insert(0, os.path.expanduser('~/.claude/quant-kalshi/scripts'))
import kalshi_exec as kx

e = kx.env()
c = kx.Kalshi(e, kx.config(e)['env'])
for tk in ['KXNFLGAME-26NOV09BUFMIN-MIN', 'KXNFLGAME-26NOV09BUFMIN-BUF']:
    m = c.market(tk)
    print(tk, {k: m.get(k) for k in ('status', 'result', 'close_time', 'last_price_dollars', 'yes_bid_dollars')})
    code, out = c.request('GET', '/portfolio/positions', params={'ticker': tk})
    print(' positions', code, json.dumps(out.get('market_positions'))[:600])
    fills = c._pages('/portfolio/fills', 'fills', {'ticker': tk})
    print(' fills', len(fills))
    for f in sorted(fills, key=lambda f: f.get('created_time', '')):
        print('  ', {k: f.get(k) for k in ('created_time', 'action', 'side', 'count', 'count_fp', 'yes_price_dollars', 'no_price_dollars', 'is_taker', 'client_order_id', 'order_id')})
