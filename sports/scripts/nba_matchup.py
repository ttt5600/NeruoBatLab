"""One-game matchup estimate from last season's opponent-adjusted offensive/defensive ratings, with "healthy core" splits.

    python scripts/nba_matchup.py --season 2026 --home OKC --away DEN \
        --core OKC="Shai Gilgeous-Alexander,Jalen Williams,Chet Holmgren" \
        --core DEN="Nikola Jokic,Jamal Murray,Aaron Gordon,Christian Braun,Cameron Johnson"

Descriptive tool, NOT a backtested rule (nothing is scored). User ask 2026-10-22 (Nuggets @ Thunder).
Model: per team-game points per 100 possessions = mu + off[team] - def[opp] + hca*home, ridge (alpha) over all regular
season + playoff games; the two named teams are split into TEAM_H (every core player logged minutes) and TEAM_N.
Possessions = mean of both teams' FGA - OREB + TOV + 0.44*FTA. Spread = (net_H home - net_H away)*pace/100 + HCA.
Win prob = P(N(spread, sd) > 0), sd = residual SD of game margins in the same fit. CI = bootstrap over games.
NOTE 2026 = the season sealed for test O1; reading it here does not score O1 (its rule is frozen in notes/player_sim.md).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

D = Path(__file__).resolve().parents[1] / "data" / "espn" / "nba"


def team_games(season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = pd.read_parquet(D / f"games_{season}.parquet")
    g = g[g.season_type.isin(["regular-season", "post-season", "play-in-season"]) & (g.status == "STATUS_FINAL")]
    b = pd.read_parquet(D / f"box_{season}.parquet")
    b = b[b.event_id.isin(set(g.event_id))]
    t = b.pivot_table(index=["event_id", "team_id"], columns="stat", values="value", aggfunc="sum").reset_index()
    t["poss_raw"] = t.fieldGoalsAttempted - t.offensiveRebounds + t.turnovers + 0.44 * t.freeThrowsAttempted
    return g, t, b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--home", required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument("--core", action="append", default=[], help='TEAM="Player A,Player B"')
    ap.add_argument("--window", action="append", default=[], help="TEAM=YYYY-MM-DD:YYYY-MM-DD (H only inside it)")
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()

    g, t, b = team_games(a.season)
    ab = {r.home_team_id: r.home for r in g.itertuples()} | {r.away_team_id: r.away for r in g.itertuples()}
    t["team"] = t.team_id.map(ab)
    core = {}
    for c in a.core:
        k, v = c.split("=", 1)
        core[k] = [s.strip() for s in v.split(",")]
    played = b[(b.stat == "minutes") & (b.value > 0)].groupby("event_id").name.apply(set)
    win = {}
    for w in a.window:
        k, v = w.split("=", 1)
        lo, hi = v.split(":")
        win[k] = (pd.Timestamp(lo, tz="UTC"), pd.Timestamp(hi, tz="UTC"))
    start = g.set_index("event_id").start_utc

    def label(row):
        if row.team in core or row.team in win:
            ok = all(p in played.get(row.event_id, set()) for p in core.get(row.team, []))
            if row.team in win:
                lo, hi = win[row.team]
                ok = ok and lo <= start[row.event_id] <= hi
            return row.team + ("_H" if ok else "_N")
        return row.team

    t["unit"] = t.apply(label, axis=1)
    gm = g.set_index("event_id")
    rows = []
    for eid, x in t.groupby("event_id"):
        if len(x) != 2 or eid not in gm.index:
            continue
        poss = x.poss_raw.mean()
        home_id, neutral = gm.loc[eid, "home_team_id"], bool(gm.loc[eid, "neutral"])
        for i in range(2):
            me, op = x.iloc[i], x.iloc[1 - i]
            rows.append({"event_id": eid, "unit": me.unit, "opp": op.unit, "pts100": 100 * me.points / poss, "poss": poss,
                         "home": 0.0 if neutral else (1.0 if me.team_id == home_id else -1.0), "margin": me.points - op.points})
    df = pd.DataFrame(rows)
    units = sorted(set(df.unit) | set(df.opp))
    for u in [f"{a.home}_H", f"{a.away}_H"]:
        print(f"{u}: {df[df.unit == u].event_id.nunique()} games")

    def fit(d: pd.DataFrame):
        ix = {u: i for i, u in enumerate(units)}
        n, k = len(d), len(units)
        X = np.zeros((n, 2 * k + 1))
        X[np.arange(n), d.unit.map(ix).to_numpy()] = 1
        X[np.arange(n), k + d.opp.map(ix).to_numpy()] = -1
        X[:, -1] = d.home.to_numpy() / 2  # home-court split across the two rows of a game
        y = d.pts100.to_numpy() - d.pts100.mean()
        R = a.alpha * np.eye(2 * k + 1)
        R[-1, -1] = 0
        beta = np.linalg.solve(X.T @ X + R, X.T @ y)
        off = dict(zip(units, beta[:k]))
        dfn = dict(zip(units, beta[k:2 * k]))
        return off, dfn, beta[-1], d.pts100.mean()

    def spread(d: pd.DataFrame):
        off, dfn, hca, mu = fit(d)
        h, w = f"{a.home}_H", f"{a.away}_H"
        pace = d[d.unit.isin([h, w])].poss.mean()
        per100 = (off[h] - dfn[w]) - (off[w] - dfn[h]) + hca
        return per100 * pace / 100, off, dfn, hca, mu, pace

    sp, off, dfn, hca, mu, pace = spread(df)
    one = df.drop_duplicates("event_id")
    # residual SD of game margins: margin vs model prediction for every game
    pred = []
    for r in one.itertuples():
        p100 = (off[r.unit] - dfn[r.opp]) - (off[r.opp] - dfn[r.unit]) + hca * r.home
        pred.append(p100 * r.poss / 100)
    sd = float(np.std(one.margin.to_numpy() - np.array(pred)))
    print(f"\nleague mean {mu:.1f} pts/100, home court {hca:+.2f}/100, residual margin SD {sd:.1f}")
    for u in [f"{a.home}_H", f"{a.home}_N", f"{a.away}_H", f"{a.away}_N"]:
        if u in off:
            print(f"  {u:8s} ORtg {mu + off[u]:6.1f}  DRtg {mu - dfn[u]:6.1f}  net {off[u] + dfn[u]:+5.1f}  "
                  f"(games {df[df.unit == u].event_id.nunique()})")
    rng = np.random.default_rng(0)
    eids = one.event_id.to_numpy()
    bs = []
    for _ in range(a.boot):
        pick = rng.choice(eids, len(eids))
        d = pd.concat([df[df.event_id == e] for e in pick[:0]] or [df.set_index("event_id").loc[pick].reset_index()])
        try:
            bs.append(spread(d)[0])
        except KeyError:
            continue
    lo, hi = np.quantile(bs, [.05, .95])
    sims = rng.normal(sp, sd, 200_000)
    print(f"\npace {pace:.1f}; {a.home} by {sp:+.1f} (90% CI {lo:+.1f}..{hi:+.1f})")
    print(f"P({a.away} wins) = {np.mean(sims < 0):.3f}  (at CI ends: {norm.cdf(-hi / sd):.3f}..{norm.cdf(-lo / sd):.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
