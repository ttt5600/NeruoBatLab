"""Non-decisional SD1 sensitivity: FV by season set vs market mid (first 22 SD1 rows of snow_log.jsonl)."""
import json
import statistics as st
import sys

sys.path.insert(0, 'scripts')
import snow_markets as S  # noqa: E402

rows = [json.loads(l) for l in open('results/snow_markets/snow_log.jsonl') if '"SD1"' in l][:22]
print('city thr lo-hi | n30 meanN30 | FV30 FV15 FV10 FVall(91-25) | mid | last10 seasons N (2016-17..2025-26)')
for r in rows:
    stn, thr, lo, hi = r['stn'], r['thr'], r['floor'], r['cap']
    out = {}
    for name, yrs in [('30', range(1996, 2026)), ('15', range(2011, 2026)), ('10', range(2016, 2026)), ('all', range(1991, 2026))]:
        S.CLIM_SEASONS = list(yrs)
        c = S.season_counts(stn, thr)
        out[name] = (sum(S.payout(n, lo, hi) for n in c) / len(c) if c else None, len(c), st.mean(c) if c else None)
    S.CLIM_SEASONS = list(range(2016, 2026))
    last = S.season_counts(stn, thr)
    f = lambda x: None if x is None else round(x, 3)
    print(r['ticker'].split('-')[-1], thr, f'{lo:.0f}-{hi:.0f}', '|', out['30'][1], f(out['30'][2]), '|',
          *[f(out[k][0]) for k in ('30', '15', '10', 'all')], '|', r['mid'], '|', last)
