"""Kalshi test (pre-registered in notes/nhl_opener_model.md): trials A (first tight, projected goalie) and B (tip-60,
actual starter) at Kalshi's own 50c rung, ask + taker fee. Plus the descriptive goalie-absorption curve."""
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
OUT = Path(__file__).parent
KXD = OUT.parent / "kalshi_openers_xleague"
sys.path.insert(0, str(OUT)); sys.path.insert(0, str(KXD))
import model as M  # noqa: E402
from kx import implied  # noqa: E402
from q1_timing import nhl_book_median  # noqa: E402

TAGS = ["", "_nhlom"]
Gm = pd.concat([pd.read_parquet(KXD / f"nhl{t}_games.parquet") for t in TAGS]).drop_duplicates("key")
K = pd.concat([pd.read_parquet(KXD / f"nhl{t}_kmarkets.parquet") for t in TAGS]).drop_duplicates("ticker")
C = pd.concat([pd.read_parquet(KXD / f"nhl{t}_candles.parquet") for t in TAGS]).drop_duplicates(["ticker", "end"])
ST = pd.read_parquet(OUT / "starters_2027.parquet")
Gm = Gm[Gm.season_type == "regular-season"]
Gm["tip"] = pd.to_datetime(Gm.start_utc, utc=True)
Gm["date"] = Gm.tip.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
sec = lambda t: int((t - EPOCH) // pd.Timedelta(seconds=1))  # noqa: E731  (traps: no //1e9 on datetime64)
toET = lambda e: (EPOCH + pd.Timedelta(seconds=int(e))).tz_convert("America/New_York").tz_localize(None).normalize()  # noqa: E731
RES = dict(zip(K.ticker, K.result))


def ladder(cd, h):
    x = cd[(cd.end <= h) & (cd.end > h - 6 * 3600)].sort_values("end").groupby("ticker").tail(1)
    x = x[(x.bid > 0) & (x.ask < 1) & x.bid.notna() & x.ask.notna()]
    if x.empty:
        return np.nan, None, x
    x = x.assign(mid=(x.bid + x.ask) / 2, spr=x.ask - x.bid)
    a = x.assign(d=(x.mid - .5).abs()).sort_values(["d", "spr"]).iloc[0]
    return implied(list(zip(x.strike, x.mid))), a, x


def fee(p):
    return math.ceil(round(700 * p * (1 - p), 6)) / 10000


rows = []
for r in Gm.sort_values("tip").itertuples():
    ev = K[K.key == r.key].event.unique()
    if len(ev) == 0:
        continue
    cd = C[C.event == ev[0]]
    tip = sec(r.tip)
    hrs = sorted(cd.end[cd.end <= tip].unique())
    if not hrs:
        continue
    h, a = M.ESPN2MP.get(r.home, r.home), M.ESPN2MP.get(r.away, r.away)
    s = ST[(ST.date == r.date.strftime("%Y-%m-%d")) & (ST.home == h) & (ST.away == a)]
    if s.empty:
        print("no starter row", r.key); continue
    gh, ga = {int(s.home_starter_id.iloc[0]): 1.0}, {int(s.away_starter_id.iloc[0]): 1.0}
    tA = next((x for x in hrs if x <= tip - 7200 and (lambda L: L[1] is not None and L[1].spr <= 0.04)(ladder(cd, x))), None)
    tB = max([x for x in hrs if x <= tip - 3600], default=None)
    tC = hrs[-1]
    LC, _, xC = ladder(cd, tC)
    base = {"key": r.key, "event": ev[0], "date": r.date, "tip": tip, "home": h, "away": a,
            "final": r.home_score + r.away_score, "L_close": LC,
            "dk_close_med": nhl_book_median(r.total_close, r.over_close, r.under_close),
            "dk_open_med": nhl_book_median(r.total_open, r.over_open, r.under_open)}
    for tr, t, D, ghh, gaa in (("A", tA, None, None, None), ("B", tB, r.date, gh, ga)):
        if t is None:
            continue
        if tr == "A":
            D = toET(t)
            ghh, gaa = M.projected(h, D), M.projected(a, D)
        try:
            lh, la, m, bh, ba = M.predict(h, a, D, ghh, gaa, b2b_date=r.date)
        except KeyError as e:
            print("model miss", r.key, e); continue
        L, an, x = ladder(cd, t)
        row = dict(base, trial=tr, Tdec=t, h_before=(tip - t) / 3600, D=D, m=m, lh=lh, la=la, b2b_h=bh, b2b_a=ba, L_K=L,
                   S50=an.strike if an is not None else np.nan, s_ticker=an.ticker if an is not None else None,
                   s_bid=an.bid if an is not None else np.nan, s_ask=an.ask if an is not None else np.nan)
        if tr == "A":  # goalie delta at the same decision date: actual starters vs projected
            row["m_act_sameD"] = M.predict(h, a, D, gh, ga, b2b_date=r.date)[2]
            row["L_by_off"] = {off: ladder(cd, max([y for y in hrs if y <= tip - off * 3600], default=hrs[0]))[0]
                               for off in (24, 12, 8, 6, 4, 3, 2, 1, 0)}
        if an is not None:  # close mid of the same rung (last two-sided candle <= tip)
            q = cd[(cd.ticker == an.ticker) & (cd.end <= tip) & (cd.bid > 0) & (cd.ask < 1)].sort_values("end")
            row["s_close_mid"] = (q.bid.iloc[-1] + q.ask.iloc[-1]) / 2 if len(q) else np.nan
            row["s_result"] = RES.get(an.ticker)
        rows.append(row)
d = pd.DataFrame(rows)
out = []
for tr in ("A", "B"):
    x = d[d.trial == tr].sort_values("Tdec").reset_index(drop=True)
    diff = x.m - x.L_K
    x["c"] = [diff[(x.tip < x.Tdec[i])].dropna().tail(100).mean() if diff[(x.tip < x.Tdec[i])].dropna().size >= 30 else np.nan
              for i in range(len(x))]
    x["g"] = x.m - x.c - x.L_K
    out.append(x)
d = pd.concat(out, ignore_index=True)
d.to_pickle(OUT / "kalshi_rows.pkl")


def boot(v, cl, n=4000, seed=0):
    v = np.asarray(v, float); cl = np.asarray(cl); ok = np.isfinite(v); v, cl = v[ok], cl[ok]
    if len(v) < 2:
        return [float(v.mean()) if len(v) else None, None, None, None]
    u = np.unique(cl); idx = {c: np.where(cl == c)[0] for c in u}
    rng = np.random.default_rng(seed); bs = np.empty(n)
    for i in range(n):
        bs[i] = v[np.concatenate([idx[c] for c in rng.choice(u, len(u))])].mean()
    return [float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float((bs <= 0).mean())]


def score(x, k):
    b = x[(x.g.abs() >= k) & x.s_result.isin(["yes", "no"])].copy()
    over = b.g > 0
    p_entry = np.where(over, b.s_ask, 1 - b.s_bid)
    b["cost"] = [p + fee(p) for p in p_entry]
    b["win"] = np.where(over, b.s_result == "yes", b.s_result == "no")
    b["roi"] = np.where(b.win, (1 - b.cost) / b.cost, -1.0)
    b["pnl"] = np.where(b.win, 1 - b.cost, -b.cost)
    mid = (b.s_bid + b.s_ask) / 2
    sg = np.where(over, 1, -1)
    b["clv_c"] = sg * (b.s_close_mid - mid)
    b["clv_g"] = sg * (b.L_close - b.L_K)
    b["clv_dk"] = sg * (b.dk_close_med - b.L_K)
    cl = b.date.astype(str).values
    return b, {"n": int(len(b)), "W": int(b.win.sum()), "overs": int(over.sum()), "mean_cost": float(b.cost.mean()) if len(b) else None,
               "mean_entry_mid": float(mid.mean()) if len(b) else None,
               "roi": boot(b.roi, cl), "pnl_per_ct": boot(b.pnl, cl), "clv_cents": boot(b.clv_c, cl),
               "clv_goals_kalshi": boot(b.clv_g, cl), "clv_goals_dk_close": boot(b.clv_dk, cl),
               "n_dates": int(len(set(cl)))}


summ = {}
for tr in ("A", "B"):
    x = d[d.trial == tr]
    summ[tr] = {"events": int(len(x)), "with_c": int(x.c.notna().sum()), "c_range": [float(x.c.min()), float(x.c.max())],
                "median_h_before": float(x.h_before.median()), "mean_abs_S50_minus_LK": float((x.S50 - x.L_K).abs().mean()),
                "sd_g": float(x.g.std())}
    for k in (0.4, 0.25, 0.5, 0.6):
        b, s = score(x, k)
        summ[tr][f"k{k}"] = s
        if k == 0.4:
            b.to_csv(OUT / f"bets_{tr}.csv", index=False)
# goalie absorption (descriptive): L_K(tip-off) - L_K(T_A) on goalie delta (actual - projected starters, same D)
A = d[d.trial == "A"].copy()
A["gd"] = A.m_act_sameD - A.m
ab = {}
for off in (24, 12, 8, 6, 4, 3, 2, 1, 0):
    y = A.L_by_off.map(lambda z: z[off]) - A.L_K
    ok = y.notna() & A.gd.notna() & (A.h_before > off)
    if ok.sum() > 20:
        ab[off] = {"n": int(ok.sum()), "slope": float(np.polyfit(A.gd[ok], y[ok], 1)[0])}
okf = A.gd.notna()
ab["final_on_gd_slope"] = float(np.polyfit(A.gd[okf], (A.final - A.L_K)[okf], 1)[0])
ab["sd_gd"] = float(A.gd.std()); ab["share_gd_gt_0.1"] = float((A.gd.abs() > 0.1).mean())
summ["goalie_absorption"] = ab
json.dump(summ, open(OUT / "kalshi.json", "w"), indent=1, default=str)
print(json.dumps(summ, indent=1, default=str))
