"""NBA matchup graph (offensive player x defender), pre-registered in notes/nba_matchup_graph.md.

    ../.venv_quant/bin/python scripts/nba_matchup_graph.py parse          # raw json -> data/nba_matchups/pairs.parquet
    ../.venv_quant/bin/python scripts/nba_matchup_graph.py reliability [--ledger]
    ../.venv_quant/bin/python scripts/nba_matchup_graph.py example AWAY HOME   # descriptive: who guards whom

Model per cell: y_ij = 100*pts/poss = mu + a_i + d_j + e_ij, WLS weights poss, ridge lambda=300 on a and d.
Interaction r_ij = residual; EB-shrunk m_ij = r * tau2/(tau2 + sigma2/n).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "nba_matchups"
OUT = ROOT / "results" / "nba_matchup_graph"
LEDGER = ROOT / "lab" / "ledger.jsonl"
LAM = 300.0
X_PAIR = 50.0
X_DEF = 500.0
DEV = ["2022-23", "2023-24", "2024-25"]
STATS = {"pts": "playerPoints", "fga": "matchupFieldGoalsAttempted", "fgm": "matchupFieldGoalsMade",
         "ast": "matchupAssists", "tov": "matchupTurnovers"}


def parse() -> pd.DataFrame:
    rows = []
    for gf in sorted(DATA.glob("games_*.json")):
        season = gf.stem.split("_")[1]
        for gid, date in json.loads(gf.read_text()):
            f = DATA / "raw" / f"{gid}.json.gz"
            if not f.exists():
                continue
            b = json.load(gzip.open(f, "rt"))["boxScoreMatchups"]
            pos = {}
            for side in ("homeTeam", "awayTeam"):
                for p in b[side]["players"]:
                    pos[p["personId"]] = p.get("position") or ""
            for side, oside in (("homeTeam", "awayTeam"), ("awayTeam", "homeTeam")):
                t, ot = b[side], b[oside]
                for p in t["players"]:
                    for m in p.get("matchups", []):
                        s = m["statistics"]
                        r = {"season": season, "game_id": gid, "date": date, "off_id": p["personId"],
                             "off_name": f"{p['firstName']} {p['familyName']}", "off_team": t["teamTricode"],
                             "off_pos": p.get("position") or "", "def_id": m["personId"],
                             "def_name": f"{m['firstName']} {m['familyName']}", "def_team": ot["teamTricode"],
                             "def_pos": pos.get(m["personId"], ""), "poss": float(s["partialPossessions"]),
                             "secs": float(s["matchupMinutesSort"])}
                        for k, v in STATS.items():
                            r[k] = float(s[v])
                        rows.append(r)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df.date)
    df.to_parquet(DATA / "pairs.parquet")
    print(len(df), "pair-game rows;", df.groupby("season").game_id.nunique().to_dict())
    return df


def season_totals(season: str) -> pd.DataFrame:
    rs = json.loads((DATA / f"season_{season}.json").read_text())["resultSets"][0]
    t = pd.DataFrame(rs["rowSet"], columns=rs["headers"])
    return pd.DataFrame({"off_id": t.OFF_PLAYER_ID, "def_id": t.DEF_PLAYER_ID, "off_name": t.OFF_PLAYER_NAME,
                         "def_name": t.DEF_PLAYER_NAME, "gp": t.GP, "poss": t.PARTIAL_POSS.astype(float),
                         "pts": t.PLAYER_PTS.astype(float), "fga": t.MATCHUP_FGA.astype(float),
                         "ast": t.MATCHUP_AST.astype(float), "tov": t.MATCHUP_TOV.astype(float)})


def sigma2_from_games(g: pd.DataFrame, stat: str) -> float:
    gg = g[g.poss > 0].copy()
    gg["y"] = 100 * gg[stat] / gg.poss
    k = gg.groupby(["off_id", "def_id"])
    gg["ybar"] = k[stat].transform("sum") * 100 / k.poss.transform("sum")
    gg["G"] = k.poss.transform("size")
    m2 = gg[gg.G >= 2]
    dof = float((m2.groupby(["off_id", "def_id"]).G.first() - 1).sum())
    return float(np.sum(m2.poss * (m2.y - m2.ybar) ** 2) / dof)


def fit_cell(g: pd.DataFrame, stat: str, sigma2: float | None = None) -> dict:
    """g: pair rows of one cell (game-level or season totals; summed per pair). sigma2 from game rows if None."""
    agg = g.groupby(["off_id", "def_id"], as_index=False)[["poss", stat]].sum()
    agg = agg[agg.poss > 0]
    n = agg.poss.to_numpy()
    y = 100 * agg[stat].to_numpy() / n
    mu = np.sum(n * y) / np.sum(n)
    oi, ou = pd.factorize(agg.off_id)
    di, du = pd.factorize(agg.def_id)
    no, nd = len(ou), len(du)
    rr = np.arange(len(agg))
    X = sp.hstack([sp.csr_matrix((np.ones(len(agg)), (rr, oi)), shape=(len(agg), no)),
                   sp.csr_matrix((np.ones(len(agg)), (rr, di)), shape=(len(agg), nd))]).tocsr()
    W = sp.diags(n)
    A = (X.T @ W @ X + LAM * sp.identity(no + nd)).tocsc()
    beta = spsolve(A, X.T @ (n * (y - mu)))
    a, d = beta[:no], beta[no:]
    r = y - mu - a[oi] - d[di]
    if sigma2 is None:
        sigma2 = sigma2_from_games(g, stat)
    big = n >= 20
    tau2 = float(max(np.mean(r[big] ** 2 - sigma2 / n[big]), 0.0))
    shrink = tau2 / (tau2 + sigma2 / n) if tau2 > 0 else np.zeros_like(n)
    agg = agg.assign(y=y, r=r, m=r * shrink, a=a[oi], d=d[di])
    dpos = g.groupby("def_id").poss.sum()
    dd = pd.DataFrame({"def_id": du, "d": d}).assign(dposs=lambda x: x.def_id.map(dpos))
    # ridge shrink factor for d_j approx n/(n+lam)
    dd["shrink"] = dd.dposs / (dd.dposs + LAM)
    return {"pairs": agg, "def": dd, "mu": mu, "sigma2": sigma2, "tau2": tau2}


def boot_corr(x, y, cl, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    x, y, cl = np.asarray(x), np.asarray(y), np.asarray(cl)
    u, inv = np.unique(cl, return_inverse=True)
    idx = [np.flatnonzero(inv == k) for k in range(len(u))]
    out = []
    for _ in range(B):
        s = np.concatenate([idx[k] for k in rng.integers(0, len(u), len(u))])
        if np.std(x[s]) > 0 and np.std(y[s]) > 0:
            out.append(np.corrcoef(x[s], y[s])[0, 1])
    return np.percentile(out, [2.5, 97.5]).tolist()


def fisher(r, n):
    z = np.arctanh(r); se = 1 / np.sqrt(n - 3)
    return [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]


def compare(c1: dict, c2: dict) -> dict:
    p = c1["pairs"].merge(c2["pairs"], on=["off_id", "def_id"], suffixes=("1", "2"))
    p = p[(p.poss1 >= X_PAIR) & (p.poss2 >= X_PAIR)]
    d = c1["def"].merge(c2["def"], on="def_id", suffixes=("1", "2"))
    d = d[(d.dposs1 >= X_DEF) & (d.dposs2 >= X_DEF)]
    return {"pairs": p, "defs": d, "tau2": (c1["tau2"], c2["tau2"]), "sigma2": (c1["sigma2"], c2["sigma2"])}


def summarize(comps: list[dict], starter_k: float) -> dict:
    p = pd.concat([c["pairs"] for c in comps], ignore_index=True)
    d = pd.concat([c["defs"].assign(cell=i) for i, c in enumerate(comps)], ignore_index=True)
    tau_zero = any(min(c["tau2"]) == 0 for c in comps)
    res = {}
    # m_ij
    if len(p) > 3:
        rm = float(np.corrcoef(p.m1, p.m2)[0, 1]) if p.m1.std() > 0 and p.m2.std() > 0 else float("nan")
        rraw = float(np.corrcoef(p.r1, p.r2)[0, 1])
        cov = float(np.cov(p.r1, p.r2)[0, 1])
        sd100 = float(np.sqrt(max(cov, 0)))
        res["m"] = {"n_pairs": int(len(p)), "n_off": int(p.off_id.nunique()), "r_shrunk": rm,
                    "ci_boot": boot_corr(p.m1, p.m2, p.off_id) if np.isfinite(rm) else None,
                    "r_raw": rraw, "ci_boot_raw": boot_corr(p.r1, p.r2, p.off_id), "ci_fisher_raw": fisher(rraw, len(p)),
                    "cov_raw": cov, "sd_persist_per100": sd100, "sd_persist_game_starter": sd100 * starter_k / 100,
                    "sd_persist_game_def_total": sd100 * 0.70, "tau2_zero_in_a_cell": tau_zero,
                    "tau2": [c["tau2"] for c in comps], "sigma2": [c["sigma2"] for c in comps]}
    if len(d) > 3:
        rd = float(np.corrcoef(d.d1, d.d2)[0, 1])
        cov = float(np.cov(d.d1, d.d2)[0, 1])
        sd100 = float(np.sqrt(max(cov, 0)))
        sh = float(np.sqrt((d.shrink1 * d.shrink2).mean()))
        res["d"] = {"n_def": int(len(d)), "r": rd, "ci_boot": boot_corr(d.d1, d.d2, d.def_id), "ci_fisher": fisher(rd, len(d)),
                    "cov": cov, "sd_persist_per100": sd100, "sd_persist_game_starter": sd100 * starter_k / 100,
                    "sd_persist_game_def_total": sd100 * 0.70, "mean_shrink": sh,
                    "sd_persist_game_starter_deshrunk": sd100 / sh * starter_k / 100}
    return res


def gate(c: dict, sdkey="sd_persist_game_starter") -> str:
    r = c.get("r_shrunk", c.get("r"))
    ci = c.get("ci_boot")
    ok = r is not None and np.isfinite(r) and r >= 0.15 and ci is not None and ci[0] > 0 and c[sdkey] >= 1.0
    if c.get("tau2_zero_in_a_cell"):
        ok = False
    return "PASS" if ok else f"DEAD (~{c[sdkey]:.2f} pts persistent)"


def reliability(ledger: bool) -> dict:
    df = pd.read_parquet(DATA / "pairs.parquet")
    df = df[df.season == "2023-24"]  # A1: only per-game data in hand (599 games)
    tot = {s: season_totals(s) for s in DEV}
    OUT.mkdir(parents=True, exist_ok=True)
    s24 = df[df.season == "2023-24"]
    pg = s24.groupby(["game_id", "off_id"]).poss.agg(tot="sum", sq=lambda v: np.sqrt(np.sum(v ** 2)))
    starter_k = float(pg[pg.tot >= 55].sq.median())
    print(f"typical starter sqrt(sum n^2) = {starter_k:.1f} (n player-games {int((pg.tot >= 55).sum())}); "
          f"median total poss {pg[pg.tot >= 55].tot.median():.1f}")
    # halves
    gorder = df[["season", "date", "game_id"]].drop_duplicates().sort_values(["season", "date", "game_id"])
    gorder["half"] = gorder.groupby("season").cumcount() % 2
    df = df.merge(gorder[["game_id", "half"]], on="game_id")
    res = {"starter_k": starter_k, "lambda": LAM, "X_pair": X_PAIR, "X_def": X_DEF, "seasons": DEV,
           "rows_pergame": int(len(df)), "games_pergame": int(df.game_id.nunique()), "amendment": "A1"}
    for stat in ["pts", "fga", "ast", "tov"]:
        sig = sigma2_from_games(df, stat)
        cells = {s: fit_cell(tot[s], stat, sig) for s in DEV}
        s2s = [compare(cells[DEV[i]], cells[DEV[i + 1]]) for i in range(len(DEV) - 1)]
        halves = [compare(fit_cell(df[df.half == 0], stat, sig), fit_cell(df[df.half == 1], stat, sig))]
        res[stat] = {"season_to_season": summarize(s2s, starter_k), "split_half": summarize(halves, starter_k),
                     "mu": {s: cells[s]["mu"] for s in DEV}, "sigma2": sig,
                     "pairs_per_season": {s: int(len(cells[s]["pairs"])) for s in DEV}}
        if stat == "pts":
            for s in DEV:
                cells[s]["pairs"].to_parquet(OUT / f"pairs_fit_{s}.parquet")
                cells[s]["def"].to_parquet(OUT / f"def_fit_{s}.parquet")
    pts = res["pts"]["season_to_season"]
    res["gate"] = {"m_ij": gate(pts["m"]), "d_j": gate(pts["d"])}
    (OUT / "reliability.json").write_text(json.dumps(res, indent=1, default=float))
    for stat in ["pts", "fga", "ast", "tov"]:
        for sp_ in ["season_to_season", "split_half"]:
            c = res[stat][sp_]
            m, d = c.get("m", {}), c.get("d", {})
            print(f"{stat:4s} {sp_:16s} m_ij: n={m.get('n_pairs')} r_shr={m.get('r_shrunk', float('nan')):.3f} "
                  f"ci={np.round(m.get('ci_boot') or [np.nan, np.nan], 3)} r_raw={m.get('r_raw', float('nan')):.3f} "
                  f"ci_raw={np.round(m.get('ci_boot_raw', [np.nan, np.nan]), 3)} sd/100={m.get('sd_persist_per100', float('nan')):.2f} "
                  f"sd_game={m.get('sd_persist_game_starter', float('nan')):.2f} tau0={m.get('tau2_zero_in_a_cell')} | "
                  f"d_j: n={d.get('n_def')} r={d.get('r', float('nan')):.3f} ci={np.round(d.get('ci_boot', [np.nan, np.nan]), 3)} "
                  f"sd/100={d.get('sd_persist_per100', float('nan')):.2f} sd_game={d.get('sd_persist_game_starter', float('nan')):.2f} "
                  f"(deshrunk {d.get('sd_persist_game_starter_deshrunk', float('nan')):.2f}) def_total={d.get('sd_persist_game_def_total', float('nan')):.2f}")
    print("GATE:", res["gate"])
    if ledger:
        sha = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:12]
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with LEDGER.open("a") as f:
            for comp, key in (("m_ij", "m"), ("d_j", "d")):
                c = pts[key]
                rec = {"ts": ts, "strategy": f"nba_matchup_graph_reliability_{comp}", "sha": sha,
                       "dataset": "nba_matchup_graph", "split": "DEV",
                       "hypothesis": "pre-registered in notes/nba_matchup_graph.md (commits f61346d + A1 5ddb89d)",
                       "source": "stats.nba.com leagueseasonmatchups totals 2022-23..2024-25 + boxscorematchupsv3 599 games 2023-24", "status": "gate",
                       "trial": f"reliability_{comp}", "verdict": res["gate"][comp],
                       **{k: v for k, v in c.items() if not isinstance(v, (list, tuple)) or k.startswith("ci")}}
                f.write(json.dumps(rec, default=float) + "\n")
    return res


def example(away: str, home: str) -> None:
    """Descriptive: point-in-time (all games before today) defender shares for each team's top players vs the
    opponent's defenders, from 2025-26 + 2026-27 data, with d_j from a fit on those games."""
    df = pd.read_parquet(DATA / "pairs.parquet")
    df = df[df.season.isin(["2025-26", "2026-27"])]
    cell = fit_cell(df, "pts")
    dmap = cell["def"].set_index("def_id").d
    amap = cell["pairs"].groupby("off_id").a.first()
    mp = cell["pairs"].set_index(["off_id", "def_id"]).m
    cur = df[df.season == "2026-27"]
    roster = {t: set(cur[cur.def_team == t].def_id) | set(cur[cur.off_team == t].off_id) for t in (away, home)}
    names = dict(zip(df.def_id, df.def_name)) | dict(zip(df.off_id, df.off_name))
    for off, de in ((away, home), (home, away)):
        o = cur[cur.off_team == off].groupby("off_id").poss.sum().sort_values(ascending=False).head(4)
        for oid in o.index:
            h = df[(df.off_id == oid) & (df.def_team == de) & df.def_id.isin(roster[de])]
            if h.empty:
                print(f"{names[oid]} ({off}): no prior possessions vs current {de} defenders"); continue
            sh = h.groupby("def_id").poss.sum(); sh = sh / sh.sum()
            dl = sum(sh[j] * dmap.get(j, 0.0) for j in sh.index)
            ml = sum(sh[j] * mp.get((oid, j), 0.0) for j in sh.index)
            avg_d = float(df[df.off_id == oid].assign(dd=lambda x: x.def_id.map(dmap).fillna(0) * x.poss).dd.sum()
                          / df[df.off_id == oid].poss.sum())
            ngames = h.game_id.nunique()
            tot = h.poss.sum()
            top = ", ".join(f"{names[j]} {sh[j]:.0%}" for j in sh.sort_values(ascending=False).index[:3])
            pp = cur[cur.off_id == oid].groupby("game_id").poss.sum().mean()
            print(f"{names[oid]} ({off}) vs {de}: {ngames} prior games, {tot:.0f} poss; guarded by {top}; "
                  f"share-weighted d = {dl:+.1f}/100 vs his usual {avg_d:+.1f}/100 -> {(dl - avg_d) * pp / 100:+.2f} pts "
                  f"over ~{pp:.0f} poss; m_ij-weighted {ml * pp / 100:+.2f} pts (a_i {amap.get(oid, 0):+.1f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["parse", "reliability", "example"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--ledger", action="store_true")
    a = ap.parse_args()
    if a.cmd == "parse":
        parse()
    elif a.cmd == "reliability":
        reliability(a.ledger)
    else:
        example(*a.args)


if __name__ == "__main__":
    main()
