"""CBB totals walk-forward model (pre-registered in notes/cbb_opener_model.md).

log(pts of t vs u) = mu + o_t + d_u + h*site; weighted ridge (half-life 45 d) toward 0.6 x last season's final o/d
(mu/h toward last season's value), lambda 6. Fit per ET date on this season's finals with date < that date.
D-I set for season s = teams with >= 8 appearances in season s-1 (2023: its own season); others pooled as NONDI.

    python results/cbb_opener_model/model.py        -> preds_<s>.parquet for 2023..2027, final_<s>.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

OUT = Path(__file__).resolve().parent
HL, LAM, SHRINK, MIN_APP = 45.0, 6.0, 0.6, 8


def di_set(g):
    c = pd.concat([g.home_id, g.away_id]).value_counts()
    return set(c[c >= MIN_APP].index)


def fit(train, teams, prior, asof):
    """train: games; teams: index map; prior: vector of length 2+2n (mu, h, o..., d...). Returns params."""
    n = len(teams)
    P = 2 + 2 * n
    if len(train) == 0:
        return prior.copy()
    w = 0.5 ** ((asof - train.date_et).dt.days.values / HL)
    hi = train.home_id.map(teams).values
    ai = train.away_id.map(teams).values
    site = np.where(train.neutral.values, 0.0, 1.0)
    m = len(train)
    rows = np.repeat(np.arange(2 * m), 4)
    # row j (home scoring): mu, h*site, o_home, d_away ; row m+j (away scoring): mu, -h*site, o_away, d_home
    cols = np.concatenate([np.stack([np.zeros(m, int), np.ones(m, int), 2 + hi, 2 + n + ai], 1),
                           np.stack([np.zeros(m, int), np.ones(m, int), 2 + ai, 2 + n + hi], 1)]).ravel()
    vals = np.concatenate([np.stack([np.ones(m), site, np.ones(m), np.ones(m)], 1),
                           np.stack([np.ones(m), -site, np.ones(m), np.ones(m)], 1)]).ravel()
    X = csr_matrix((vals, (rows, cols)), shape=(2 * m, P))
    y = np.log(np.concatenate([train.home_score.values, train.away_score.values]).clip(1))
    sw = np.sqrt(np.concatenate([w, w]))
    Xw = X.multiply(sw[:, None]).tocsr()
    r = (y - X @ prior) * sw
    delta = lsqr(Xw, r, damp=np.sqrt(LAM), atol=1e-10, btol=1e-10, iter_lim=5000)[0]
    return prior + delta


def season(s, prev_final, di):
    g = pd.read_parquet(OUT / f"games_{s}.parquet")
    g["date_et"] = pd.to_datetime(g.date_et)
    for c in ("home_id", "away_id"):
        g[c] = g[c].where(g[c].isin(di), "NONDI")
    ids = sorted(set(g.home_id) | set(g.away_id) | {"NONDI"})
    teams = {t: i for i, t in enumerate(ids)}
    n = len(ids)
    prior = np.zeros(2 + 2 * n)
    if prev_final:
        prior[0], prior[1] = prev_final["mu"], prev_final["h"]
        for t, i in teams.items():
            prior[2 + i] = SHRINK * prev_final["o"].get(t, 0.0)
            prior[2 + n + i] = SHRINK * prev_final["d"].get(t, 0.0)
    else:
        prior[0], prior[1] = np.log(70.0), 0.015
    out = []
    for d in sorted(g.date_et.unique()):
        d = pd.Timestamp(d)
        b = fit(g[g.date_et < d], teams, prior, d)
        x = g[g.date_et == d]
        hi, ai = x.home_id.map(teams).values, x.away_id.map(teams).values
        site = np.where(x.neutral.values, 0.0, 1.0)
        ph = np.exp(b[0] + b[2 + hi] + b[2 + n + ai] + b[1] * site)
        pa = np.exp(b[0] + b[2 + ai] + b[2 + n + hi] - b[1] * site)
        out.append(pd.DataFrame({"eid": x.eid.values, "date_et": d, "pred_home": ph, "pred_away": pa,
                                 "model": ph + pa, "n_train": (g.date_et < d).sum()}))
    P = pd.concat(out).merge(g[["eid", "home_score", "away_score", "home_id", "away_id", "neutral", "start", "n_periods"]], on="eid")
    P["final"] = P.home_score + P.away_score
    P["season"] = s
    P.to_parquet(OUT / f"preds_{s}.parquet")
    bf = fit(g, teams, prior, g.date_et.max() + pd.Timedelta(days=1))
    final = {"mu": float(bf[0]), "h": float(bf[1]), "o": {t: float(bf[2 + i]) for t, i in teams.items()},
             "d": {t: float(bf[2 + n + i]) for t, i in teams.items()}}
    json.dump(final, open(OUT / f"final_{s}.json", "w"))
    return P, final, di_set(pd.read_parquet(OUT / f"games_{s}.parquet"))


if __name__ == "__main__":
    seasons = [int(a) for a in sys.argv[1:]] or [2023, 2024, 2025, 2026, 2027]
    prev, di = None, di_set(pd.read_parquet(OUT / "games_2023.parquet"))
    for s in seasons:
        if s != 2023 and prev is None:
            prev = json.load(open(OUT / f"final_{s-1}.json"))
            di = di_set(pd.read_parquet(OUT / f"games_{s-1}.parquet"))
        P, prev, di = season(s, prev, di)
        e = P.model - P.final
        print(s, len(P), "MAE %.2f bias %+.2f" % (e.abs().mean(), e.mean()),
              "| first 2 wks MAE %.2f" % e[P.date_et < P.date_et.min() + pd.Timedelta(days=14)].abs().mean())
