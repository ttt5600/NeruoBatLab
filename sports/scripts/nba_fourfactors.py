"""Four-factor matchup model for one game (descriptive; NOT a backtested rule).

    python scripts/nba_fourfactors.py --home OKC --away DEN \
        --core OKC="Shai Gilgeous-Alexander,Jalen Williams,Chet Holmgren" \
        --core DEN="Nikola Jokic,Jamal Murray,Cameron Johnson"

Written 2026-10-22 for DEN @ OKC. Decision rule committed in chat BEFORE running: bet DEN YES only if P(DEN) >= .314
(27c ask + fee + 3 pts), OKC YES only if P(OKC) >= .784, else no bet; $1.
Stage 1: for each of Dean Oliver's four factors on each side of the ball (eFG%, TOV per poss, ORB%, FT/FGA),
  factor(team-game) = mu + off[unit] - def[opp] + h*home, ridge over all 2025-26 regular+post games; DEN/OKC split into
  _H (all core players logged minutes) / _N.
Stage 2: pts per 100 = OLS on the four factors + home, all team-games.
Matchup: each side's expected factors = mu + off[side] - def[other] (+/- home) -> stage-2 pts/100 -> margin x pace.
Win prob uses the residual SD of game margins predicted the same way for every game (in-sample, so optimistic).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from nba_matchup import team_games

F = ["efg", "tov", "orb", "ftr"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--home", required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument("--core", action="append", default=[])
    ap.add_argument("--alpha", type=float, default=2.0)
    a = ap.parse_args()

    g, t, b = team_games(a.season)
    ab = {r.home_team_id: r.home for r in g.itertuples()} | {r.away_team_id: r.away for r in g.itertuples()}
    t["team"] = t.team_id.map(ab)
    core = {k: [s.strip() for s in v.split(",")] for k, v in (c.split("=", 1) for c in a.core)}
    played = b[(b.stat == "minutes") & (b.value > 0)].groupby("event_id").name.apply(set)
    t["unit"] = [r.team + ("_H" if all(p in played.get(r.event_id, set()) for p in core[r.team]) else "_N")
                 if r.team in core else r.team for r in t.itertuples()]
    gm = g.set_index("event_id")
    rows = []
    for eid, x in t.groupby("event_id"):
        if len(x) != 2 or eid not in gm.index:
            continue
        poss = x.poss_raw.mean()
        hid, neu = gm.loc[eid, "home_team_id"], bool(gm.loc[eid, "neutral"])
        for i in range(2):
            me, op = x.iloc[i], x.iloc[1 - i]
            rows.append({"event_id": eid, "unit": me.unit, "opp": op.unit, "poss": poss,
                         "home": 0.0 if neu else (1.0 if me.team_id == hid else -1.0),
                         "pts100": 100 * me.points / poss, "margin": me.points - op.points,
                         "efg": (me.fieldGoalsMade + 0.5 * me.threePointFieldGoalsMade) / me.fieldGoalsAttempted,
                         "tov": me.turnovers / poss,
                         "orb": me.offensiveRebounds / (me.offensiveRebounds + op.defensiveRebounds),
                         "ftr": me.freeThrowsMade / me.fieldGoalsAttempted})
    df = pd.DataFrame(rows)
    units = sorted(set(df.unit) | set(df.opp))
    ix = {u: i for i, u in enumerate(units)}
    k, n = len(units), len(df)
    X = np.zeros((n, 2 * k + 1))
    X[np.arange(n), df.unit.map(ix)] = 1
    X[np.arange(n), k + df.opp.map(ix)] = -1
    X[:, -1] = df.home / 2
    R = a.alpha * np.eye(2 * k + 1)
    R[-1, -1] = 0
    adj = {}
    for f in F:
        y = df[f].to_numpy()
        # ridge in units of each factor's SD so one alpha fits all four
        s = y.std()
        beta = np.linalg.solve(X.T @ X + R, X.T @ ((y - y.mean()) / s))
        adj[f] = {"mu": y.mean(), "s": s, "off": dict(zip(units, beta[:k])), "def": dict(zip(units, beta[k:2 * k])),
                  "h": beta[-1]}
    Z = np.column_stack([np.ones(n)] + [df[f] for f in F] + [df.home])
    w = np.linalg.lstsq(Z, df.pts100.to_numpy(), rcond=None)[0]
    r2 = 1 - np.var(df.pts100 - Z @ w) / np.var(df.pts100)
    print(f"stage 2: pts/100 = {w[0]:.1f} + {w[1]:.1f}*eFG + {w[2]:.1f}*TOV + {w[3]:.1f}*ORB + {w[4]:.1f}*FTr "
          f"+ {w[5]:.2f}*home   (R^2 {r2:.3f})")

    def exp_factors(me, op, home):
        return {f: adj[f]["mu"] + adj[f]["s"] * (adj[f]["off"][me] - adj[f]["def"][op] + adj[f]["h"] * home / 2)
                for f in F}

    def pts(fs, home):
        return w[0] + sum(w[i + 1] * fs[f] for i, f in enumerate(F)) + w[5] * home

    # residual SD over every game, predicted the same way
    one = df[df.home >= 0].drop_duplicates("event_id")
    err = []
    for r in one.itertuples():
        ph = pts(exp_factors(r.unit, r.opp, r.home), r.home)
        pa = pts(exp_factors(r.opp, r.unit, -r.home), -r.home)
        err.append(r.margin - (ph - pa) * r.poss / 100)
    sd = float(np.std(err))

    h, aw = f"{a.home}_H", f"{a.away}_H"
    fh, fa = exp_factors(h, aw, 1.0), exp_factors(aw, h, -1.0)
    print(f"\nexpected four factors in this matchup ({h} home):")
    print(f"  {'':8s} {'eFG%':>6s} {'TOV/poss':>8s} {'ORB%':>6s} {'FT/FGA':>7s}")
    for u, fs in [(h, fh), (aw, fa)]:
        print(f"  {u:8s} {fs['efg']:6.3f} {fs['tov']:8.3f} {fs['orb']:6.3f} {fs['ftr']:7.3f}")
    print("  league   " + " ".join(f"{adj[f]['mu']:.3f}" for f in F))
    for u in [h, aw]:
        print(f"  {u} games: {df[df.unit == u].event_id.nunique()}")
    pace = df[df.unit.isin([h, aw])].poss.mean()
    ph, pa = pts(fh, 1.0), pts(fa, -1.0)
    spread = (ph - pa) * pace / 100
    rng = np.random.default_rng(0)
    p_away = float(np.mean(rng.normal(spread, sd, 400_000) < 0))
    print(f"\n{h} {ph:.1f} vs {aw} {pa:.1f} per 100, pace {pace:.1f} -> {a.home} by {spread:+.1f}; margin SD {sd:.1f}")
    print(f"P({a.away} wins) = {p_away:.3f}   P({a.home} wins) = {1 - p_away:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
