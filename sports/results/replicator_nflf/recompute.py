"""Independent recompute of NFLF per-ticker fills P&L from raw.json + lead_exec.jsonl journal."""
import json, os, collections
H = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(H, 'raw.json')))
J = [json.loads(l) for l in open(os.path.expanduser('~/.claude/quant-kalshi/data/lead_exec.jsonl'))]
J = [r for r in J if r.get('ticker', '').startswith('KXNFLGAME')]

fills = sorted(d['fills'], key=lambda f: (f['ts'], f['created_time']))
by = collections.defaultdict(list)
for f in fills:
    by[f['ticker']].append(f)

def res(tk):
    m = d['markets'][tk] or {}
    r = m.get('result')
    if r in ('yes', 'no'):
        return (1.0 if r == 'yes' else 0.0), 'settled'
    return float(m.get('last_price_dollars') or 0), 'marked_last'

# journal: filled entries and exit attempts per ticker
jent = collections.Counter(); jatt = collections.Counter(); jpnl = collections.defaultdict(float)
filled_keys = set()
for r in J:
    if r['event'] == 'update' and r.get('status') == 'filled':
        filled_keys.add((r['ticker'], r['key']))
    if r['event'] == 'update' and 'exit_attempts' in r:
        jatt[r['ticker']] = max(jatt[r['ticker']], r['exit_attempts'])
    if r['event'] == 'settle':
        jpnl[r['ticker']] += r['pnl']
for tk, k in filled_keys:
    jent[tk] += 1

tot_all = tot_clean = stake_clean = 0.0; w = l = 0
date_sells = collections.Counter(); date_intended = collections.Counter()
print(f"{'ticker':28s} bought sold net  sellOrd jAtt jEnt  allPnL  cleanPnL  set/mark  incident")
for tk, fs in by.items():
    v, how = res(tk)
    bought = sum(float(f['count_fp']) for f in fs if f['action'] == 'buy')
    sold = sum(float(f['count_fp']) for f in fs if f['action'] == 'sell')
    net = bought - sold
    cash = 0.0
    for f in fs:
        n, p, fee = float(f['count_fp']), float(f['yes_price_dollars']), float(f['fee_cost'])
        cash += (-n * p if f['action'] == 'buy' else n * p) - fee
    allp = cash + net * v
    # clean: each buy order + the first sell order after it, capped at the bought count
    orders = []  # (kind, order_id, [fills]) in time order
    for f in fs:
        if orders and orders[-1][1] == f['order_id']:
            orders[-1][2].append(f)
        else:
            orders.append((f['action'], f['order_id'], [f]))
    sell_orders = len({o[1] for o in orders if o[0] == 'sell'})
    day = fs[0]['created_time'][:10]
    clean = 0.0
    i = 0
    while i < len(orders):
        if orders[i][0] == 'buy':
            ent = orders[i][2]
            nb = sum(float(f['count_fp']) for f in ent)
            c = -sum(float(f['count_fp']) * float(f['yes_price_dollars']) + float(f['fee_cost']) for f in ent)
            stake_clean += -c
            nxt = next((o for o in orders[i + 1:] if o[0] == 'sell'), None)
            # does another buy come before that sell?
            nb_left = nb
            if nxt is not None:
                for f in nxt[2]:
                    n = min(float(f['count_fp']), nb_left)
                    if n <= 0: break
                    fee = float(f['fee_cost']) * n / float(f['count_fp'])
                    c += n * float(f['yes_price_dollars']) - fee
                    nb_left -= n
            c += nb_left * v
            clean += c
            if c > 0: w += 1
            else: l += 1
        i += 1
    tot_all += allp; tot_clean += clean
    date_sells[day] += sell_orders
    date_intended[day] += jent[tk]
    print(f"{tk[10:]:28s} {bought:5.0f} {sold:6.2f} {net:6.2f} {sell_orders:4d} {jatt[tk]:4d} {jent[tk]:4d} {allp:8.2f} {clean:8.2f}  {how}:{v} {allp-clean:8.2f}")
print(f"ALL-fills P&L {tot_all:.2f}  clean {w}-{l} {tot_clean:.2f} on stake {stake_clean:.2f}  incident {tot_all-tot_clean:.2f}")
print('sell orders by date', dict(date_sells), 'intended exits (filled entries) by date', dict(date_intended))
print('-- Kalshi settlements check')
for s in d['settlements']:
    tk = s['ticker']
    y, n = float(s['yes_count_fp']), float(s['no_count_fp'])
    print(tk[10:], 'yes', y, 'no', n, 'rev', s['revenue'] / 100, 'ycost', s['yes_total_cost_dollars'], 'ncost', s['no_total_cost_dollars'], 'fee', s['fee_cost'], 'result', s['market_result'])
