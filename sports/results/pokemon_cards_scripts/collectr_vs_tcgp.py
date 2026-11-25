import sys, os, json, py7zr, tempfile
# usage: collectr_vs_tcgp.py ARCHIVE_7Z OUTDIR ; extracts category 3 (Pokemon) price files for listed groups and
# compares Kalshi KXPOKEMON expiration_value (Collectr) with TCGplayer marketPrice on the archive date.
arc, outdir = sys.argv[1], sys.argv[2]
pairs = {  # ticker: (groupId, productId, expiration_value)
 '2026-09-30': [('KXPOKEMON-26SEP151ULTCO-932.18', 23237, 502005, 879.17), ('KXPOKEMON-26SEPASCHEREL-169.14', 24541, 668496, 151.62),
                ('KXPOKEMON-26SEPASCHERPO-407.86', 24541, 668497, 356.52), ('KXPOKEMON-26SEPPERORDBO-174.14', 24587, 672394, 171.63),
                ('KXPOKEMON-26SEPPERORDPO-121.39', 24587, 672404, 123.3)],
 '2026-10-01': None,
 '2026-10-31': [('KXPOKEMON-26OCT30TCELPO-313.44', 24722, 704144, 277.53)],
}
date = [d for d in pairs if d in os.path.basename(arc)][0]
if pairs[date] is None: pairs[date] = pairs['2026-09-30']  # Sep markets settle on first value AFTER Sep 30
want = {f'{date}/3/{g}/prices' for _, g, _, _ in pairs[date]}
with py7zr.SevenZipFile(arc, 'r') as z:
    names = z.getnames()
    hit = [n for n in names if n in want]
    if not hit:
        print('names sample', [n for n in names if '/3/' in n][:5]); sys.exit(1)
    z.extract(path=outdir, targets=hit)
for t, g, p, ev in pairs[date]:
    fp = os.path.join(outdir, date, '3', str(g), 'prices')
    res = json.load(open(fp))['results']
    r = [x for x in res if x['productId'] == p]
    mk = r[0]['marketPrice'] if r else None
    print(f'{date} {t} collectr_exp={ev} tcgplayer_market={mk} low={r[0]["lowPrice"] if r else None} ratio={ev/mk if mk else float("nan"):.4f}')
