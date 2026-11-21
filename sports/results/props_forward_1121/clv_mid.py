"""CLV vs the last pre-tip MID (1-min Kalshi candle at or before ESPN tip) for settled picks; descriptive."""
import numpy as np, pandas as pd
OUT = 'results/props_forward_1121/'
S = pd.read_parquet(OUT + 'settled_picks.parquet'); C = pd.read_parquet(OUT + 'candles_close.parquet')
X = S.merge(C, on='ticker', how='left', validate='one_to_one')
print(f"settled picks {len(X)}; with a pre-tip candle quote {int(X.close_mid.notna().sum())}; candle_end <= tip violations "
      f"{int((X.candle_end > X.tip_s).sum())}; median minutes before tip of last quote {((X.tip_s - X.candle_end) / 60).median():.1f}")
sg = X.pick.map({'YES': 1, 'NO': -1})
X['entry_mid'] = (X.ask + X.bid) / 2
X['clv_mid_vs_px'] = np.where(X.pick == 'YES', X.close_mid - X.ask, X.bid - X.close_mid)    # prereg sign, mid close
X['drift'] = (X.close_mid - X.entry_mid) * sg                                                # mid-to-mid toward the pick
X['hs'] = (X.ask - X.bid) / 2
X['sim_gap_closed'] = np.where(X.pick == 'YES', (X.close_mid - X.entry_mid) / (X.sim - X.entry_mid), (X.close_mid - X.entry_mid) / (X.sim - X.entry_mid))
def cboot(df, num, n=10000, seed=0):
    df = df.dropna(subset=[num]); g = df.groupby('event_id').agg(s=(num, 'sum'), c=(num, 'size'))
    s, c = g.s.to_numpy(float), g.c.to_numpy(float)
    k = np.random.default_rng(seed).integers(0, len(g), (n, len(g))); bs = s[k].sum(1) / c[k].sum(1)
    return s.sum() / c.sum(), np.quantile(bs, .025), np.quantile(bs, .975), len(df), (bs >= 0).mean()
fl = X.batch <= pd.Timestamp('2026-11-05 08:00', tz='US/Pacific')
for tag, x in (('FULL', X), ('FIRST LOOK', X[fl])):
    print(tag)
    for c in ('clv_re', 'clv_mid_vs_px', 'drift', 'hs'):
        e, lo, hi, n, p = cboot(x, c)
        print(f"  {c:<14} mean {e:+.4f} [{lo:+.4f}, {hi:+.4f}] n {n}  p(>=0) {p:.4f}")
for sd in ('YES', 'NO'):
    e, lo, hi, n, p = cboot(X[X.pick == sd], 'drift'); print(f"  drift {sd}: {e:+.4f} [{lo:+.4f}, {hi:+.4f}] n {n}")
print('  share of sim-mid gap closed by tip (median):', round(float(X.sim_gap_closed.median()), 4))
X.to_parquet(OUT + 'picks_with_close_mid.parquet')
