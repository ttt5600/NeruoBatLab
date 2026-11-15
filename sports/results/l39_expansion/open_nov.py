"""Open Nov-2026 / Dec-2026 monthly rain+snow ladders: which series, how many brackets, quotes, volume. Read-only."""
import sys, json, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from kx_throttle import kget

ser = json.load(open(Path(__file__).resolve().parents[1] / 'wx_capacity' / 'series_wx_2026-11-12.json'))
names = sorted(x['ticker'] for x in ser if re.match(r'^KXRAIN[A-Z]+M$', x['ticker']) or re.search(r'SNOWM?B?$', x['ticker']) and x['ticker'].startswith('KX'))
out = []
for s in names:
    d = kget('/markets', {'series_ticker': s, 'status': 'open', 'limit': 200}) or {}
    for m in d.get('markets', []):
        out.append({k: m.get(k) for k in ('ticker', 'event_ticker', 'strike_type', 'floor_strike', 'yes_bid_dollars',
                                           'yes_ask_dollars', 'volume_fp', 'volume_24h_fp', 'open_interest_fp',
                                           'close_time', 'rules_primary')} | {'series': s})
    print(s, len(d.get('markets', [])), flush=True)
json.dump(out, open(sys.argv[1], 'w'), indent=1)
