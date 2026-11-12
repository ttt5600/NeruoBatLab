"""Q1 timing (pre-registered in notes/kalshi_openers_xleague.md). Per league x market: Kalshi listing / first-tight
hours before tip, Kalshi implied line at first tight vs book (DK) open and close, D = |K-open| - |K-close| with a game
bootstrap. Usage: q1_timing.py LEAGUE.. -> q1_<lg>_rows.csv, q1_summary.json (merged)."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from kx import implied, side_x  # noqa: E402

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
import os  # noqa: E402
TAG = os.environ.get("KX_TAG", "")


def am(a):
    return 1 / (1 + (a / 100 if a > 0 else 100 / -a))


def nv_over(o, u):
    if pd.isna(o) or pd.isna(u):
        return np.nan
    p, q = am(o), am(u)
    return p / (p + q)


def pois_sf(k, mu):  # P(T > k) for integer k
    return 1 - sum(math.exp(-mu) * mu ** i / math.factorial(i) for i in range(int(k) + 1))


def mu_from(line, p_over):
    lo, hi = 2.0, 10.0
    for _ in range(60):
        m = (lo + hi) / 2
        po = pois_sf(math.floor(line), m) if line % 1 else pois_sf(line, m) / (pois_sf(line, m) + 1 - pois_sf(line - 1, m))
        lo, hi = (m, hi) if po < p_over else (lo, m)
    return (lo + hi) / 2


def nhl_book_median(line, o, u):
    p = nv_over(o, u)
    if pd.isna(line) or pd.isna(p):
        return np.nan
    mu = mu_from(line, p)
    return implied([(k + .5, pois_sf(k, mu)) for k in range(2, 11)])


def ladder(cd, mk):
    """cd: candles of one event at one hour -> implied, anchor (strike, bid, ask, spread)."""
    cd = cd[(cd.bid > 0) & (cd.ask < 1) & cd.bid.notna() & cd.ask.notna()]
    if cd.empty:
        return np.nan, None
    pts = []
    for r in cd.itertuples():
        x, sg = side_x(r.event, r.ticker, r.strike, mk)
        mid = (r.bid + r.ask) / 2
        pts.append((x, mid if sg > 0 else 1 - mid))
    a = cd.assign(d=((cd.bid + cd.ask) / 2 - .5).abs()).sort_values("d").iloc[0]
    return implied(pts), a


def boot(d, n=4000, seed=0):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = rng.choice(d, (n, len(d))).mean(1)
    return d.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def run(lg):
    G = pd.read_parquet(OUT / f"{lg}{TAG}_games.parquet").set_index("key")
    K = pd.read_parquet(OUT / f"{lg}{TAG}_kmarkets.parquet")
    C = pd.read_parquet(OUT / f"{lg}{TAG}_candles.parquet")
    rows = []
    for ev, kk in K.groupby("event"):
        mk = kk.mk.iloc[0]
        g = G.loc[kk.key.iloc[0]]
        if isinstance(g, pd.DataFrame):
            g = g.iloc[0]
        occ = pd.Timestamp(g.start_utc)  # Kalshi occurrence_datetime = kickoff + ~3 h (checked: BUF@NYJ 21:00Z vs 18:00Z kick)
        if occ.tzinfo is None:
            occ = occ.tz_localize("UTC")
        ot = pd.to_datetime(kk.open_time).min()
        if mk == "total":
            if lg == "nhl":
                bo = nhl_book_median(g.get("total_open"), g.get("over_open"), g.get("under_open"))
                bc = nhl_book_median(g.get("total_close"), g.get("over_close"), g.get("under_close"))
            else:
                bo, bc = g.get("total_open"), g.get("total_close")
        else:
            bo, bc = -g.get("home_spread_open", np.nan), -g.get("home_spread_close", np.nan)
        c = C[C.event == ev]
        r = dict(lg=lg, mk=mk, event=ev, occ=occ, k_occ=pd.to_datetime(kk.occ).max(), open_time=ot, h_list_before=(occ - ot).total_seconds() / 3600,
                 book_open=bo, book_close=bc, book=g.get("book_tot" if mk == "total" else "book_spr"))
        ft = f2 = None
        last = (np.nan, None)
        latest = {}
        for t, cd in c.sort_values("end").groupby("end"):
            T = EPOCH + pd.Timedelta(seconds=int(t))
            if T > occ:
                break
            for rr in cd.itertuples():  # carry each strike's last quote forward (<= 6 h old)
                if pd.notna(rr.bid) and pd.notna(rr.ask):
                    latest[rr.ticker] = rr
            cur = pd.DataFrame([x._asdict() for x in latest.values() if t - x.end <= 6 * 3600])
            imp, a = ladder(cur, mk) if len(cur) else (np.nan, None)
            if a is None:
                continue
            sp = a.ask - a.bid
            if f2 is None:
                f2 = (T, sp, imp)
            if ft is None and sp <= 0.04 and np.isfinite(imp):
                ft = (T, sp, imp, a.vol)
            last = (imp, T)
        if f2:
            r.update(first2_t=f2[0], first2_spread=f2[1], h_first2_before=(occ - f2[0]).total_seconds() / 3600)
        if ft:
            r.update(ft_t=ft[0], ft_spread=ft[1], k_ft=ft[2], h_ft_before=(occ - ft[0]).total_seconds() / 3600,
                     ft_hour_vol=ft[3])
        r["k_last"] = last[0]
        rows.append(r)
    d = pd.DataFrame(rows)
    d["a_open"] = (d.k_ft - d.book_open).abs()
    d["a_close"] = (d.k_ft - d.book_close).abs()
    d["D"] = d.a_open - d.a_close
    d["moved"] = (d.book_open - d.book_close).abs() > 1e-9
    d.to_csv(OUT / f"q1_{lg}{TAG}_rows.csv", index=False)
    S = {}
    for mk, x in d.groupby("mk"):
        m = x[x.k_ft.notna() & x.book_open.notna() & x.book_close.notna()]
        mv = m[m.moved]
        mean, lo, hi = boot(mv.D)
        s = dict(n_events=len(x), n_ft=int(x.k_ft.notna().sum()), n_cmp=len(m), n_moved=len(mv),
                 h_list_before_med=float(x.h_list_before.median()), h_ft_before_med=float(x.h_ft_before.median()),
                 h_ft_before_p10=float(x.h_ft_before.quantile(.1)), first2_spread_med=float(x.first2_spread.median()),
                 ft_spread_med=float(x.ft_spread.median()),
                 a_open_mean=float(mv.a_open.mean()), a_close_mean=float(mv.a_close.mean()),
                 D=mean, D_lo=lo, D_hi=hi, share_closer_open=float((mv.a_open < mv.a_close).mean()),
                 share_tie=float((mv.a_open == mv.a_close).mean()),
                 k_minus_open_mean=float((m.k_ft - m.book_open).mean()),
                 open_close_abs_mean=float((mv.book_open - mv.book_close).abs().mean()),
                 k_last_minus_close_abs=float((m.k_last - m.book_close).abs().mean()),
                 books=m.book.value_counts().to_dict())
        s["GATE"] = bool(len(mv) >= 15 and np.isfinite(hi) and hi < 0 and s["h_ft_before_med"] >= 12)
        S[f"{lg}_{mk}"] = s
        print(lg, mk, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}))
    return S


if __name__ == "__main__":
    f = OUT / f"q1{TAG}_summary.json"
    allS = json.loads(f.read_text()) if f.exists() else {}
    for lg in sys.argv[1:]:
        allS.update(run(lg))
    f.write_text(json.dumps(allS, indent=1, default=str))
