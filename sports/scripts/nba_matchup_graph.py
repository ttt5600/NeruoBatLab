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
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
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


# ------------------------------------------------------------------------------- Step 2 (prior-season state)
SEASON_OF = {2023: ("2022-23", "2021-22"), 2024: ("2023-24", "2022-23"), 2025: ("2024-25", "2023-24")}
G_BAR = 0.05 / 3


def norm(n: str) -> str:
    import re
    import unicodedata
    import nba_injury_teammate_props as itp
    t = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode().lower()
    t = re.sub(r"[^a-z ]", "", t)
    t = "".join(w for w in t.split() if w not in {"jr", "sr", "ii", "iii", "iv"})
    al = {re.sub(r"(jr|iii)$", "", k): re.sub(r"(jr|iii)$", "", v) for k, v in itp.NAME_ALIAS.items()}
    return al.get(t, t)


def names_of(t: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([t[["off_id", "off_name"]].rename(columns={"off_id": "id", "off_name": "name"}),
                      t[["def_id", "def_name"]].rename(columns={"def_id": "id", "def_name": "name"})]).drop_duplicates("id")


def prior_state(prev: str) -> dict:
    t = season_totals(prev)
    c = fit_cell(t, "pts", sigma2=1.0)  # sigma2 irrelevant for d_j / a_i
    dmap = c["def"].set_index("def_id").d
    t = t[t.poss > 0].copy()
    t["d"] = t.def_id.map(dmap).fillna(0.0)
    t["dw"] = t.d * t.poss
    g = t.groupby("off_id")
    usual = g.dw.sum() / g.poss.sum()
    P = g.poss.sum() / g.gp.max()
    return {"pairs": t.set_index(["off_id", "def_id"]).poss, "d": dmap, "usual": usual, "P": P, "names": names_of(t)}


def signals() -> pd.DataFrame:
    import nba_player_opp_venue as pov
    out, audit = [], {}
    for es, (cur, prev) in SEASON_OF.items():
        st = prior_state(prev)
        box = pov.load_box([es]).sort_values("start_utc")
        nb = pd.concat([st["names"], names_of(season_totals(cur))]).drop_duplicates("id")
        nb["k"] = nb.name.map(norm)
        k2id = nb[~nb.k.duplicated(keep=False)].set_index("k").id
        eb = box.drop_duplicates("athlete_id")[["athlete_id", "name"]].copy()
        eb["k"] = eb.name.map(norm)
        eb["nba_id"] = eb.k.map(k2id)
        eb.loc[eb.k.duplicated(keep=False), "nba_id"] = np.nan
        box["nba_id"] = box.athlete_id.map(eb.set_index("athlete_id").nba_id)
        audit[cur] = {"espn_players": int(len(eb)), "matched": int(eb.nba_id.notna().sum()),
                      "matched_minutes_share": float(box[box.nba_id.notna()].minutes.sum() / box.minutes.sum())}
        tg = box[["event_id", "team_id", "start_utc"]].drop_duplicates(["event_id", "team_id"]).sort_values("start_utc")
        tg["prev_event"] = tg.groupby("team_id").event_id.shift(1)
        prev_ev = tg.set_index(["event_id", "team_id"]).prev_event.to_dict()
        roster = box.groupby(["event_id", "team_id"]).nba_id.apply(lambda x: [int(v) for v in x.dropna()]).to_dict()
        # A2 (bug fix before outcomes): GP in totals counts pair meetings, so P_i = prior-season poss / prior-season
        # ESPN games played (minutes > 0), names matched the same way
        pb = pov.load_box([es - 1])
        gpk = pb.groupby(pb.name.map(norm)).event_id.nunique()
        nm = st["names"].assign(k=lambda x: x.name.map(norm)).drop_duplicates("k", keep=False).set_index("id").k
        gp_prev = pd.Series(st["P"].index.map(nm).map(gpk), index=st["P"].index)
        P = (st["pairs"].groupby(level=0).sum() / gp_prev).dropna()
        audit[cur]["P_matched_off_players"] = int(len(P))
        dmap, usual = st["d"], st["usual"]
        pair_by_off = {k: v.droplevel(0) for k, v in st["pairs"].groupby(level=0)}
        for r in box[box.nba_id.notna()].itertuples():
            pe = prev_ev.get((r.event_id, r.opp_id))
            if pe is None or pd.isna(pe):
                continue
            oid = int(r.nba_id)
            pv = pair_by_off.get(oid)
            if pv is None or oid not in P.index:
                continue
            n = pv.reindex(roster.get((pe, r.opp_id), [])).dropna()
            if n.sum() < 30:
                continue
            sh = n / n.sum()
            gd = (float((sh * dmap.reindex(sh.index).fillna(0.0)).sum()) - float(usual[oid])) * float(P[oid]) / 100
            out.append({"event_id": r.event_id, "athlete_id": r.athlete_id, "nba_id": oid, "season_cur": cur,
                        "graph_delta": gd, "n_prior_vs_DO": float(n.sum()), "P": float(P[oid]),
                        "top_def": int(sh.idxmax()), "top_share": float(sh.max())})
    sg = pd.DataFrame(out)
    OUT.mkdir(parents=True, exist_ok=True)
    sg.to_parquet(OUT / "signals.parquet")
    print("match audit", audit)
    print(sg.groupby("season_cur").graph_delta.describe().round(2).to_string())
    print("share |delta|>=1.5:", round(float((sg.graph_delta.abs() >= 1.5).mean()), 3))
    return sg


def trials(ledger: bool) -> dict:
    import nba_injury_teammate_props as itp
    import nba_player_opp_venue as pov
    itp.BAR = G_BAR
    sg = pd.read_parquet(OUT / "signals.parquet")
    q = itp.load_quotes()
    q = q[q.stat == "points"]
    m = q.merge(sg, on=["event_id", "athlete_id"], how="inner")
    m["resid"] = m.actual - m.L
    res = {"quotes_points": int(len(q)), "quotes_with_signal": int(len(m)),
           "G1_over_ge_1.5": itp.roi_trial(m[m.graph_delta >= 1.5], "over"),
           "G2_under_le_-1.5": itp.roi_trial(m[m.graph_delta <= -1.5], "under"),
           "G3_slope": itp.slope_trial(m, "graph_delta", "resid")}
    for k in ("G1_over_ge_1.5", "G2_under_le_-1.5"):
        r = res[k]
        if r.get("bets", 0) > 200 and r["roi"] > 0.10:
            r["TOO_GOOD_inspect"] = True
    res["by_season"] = {s: {"n": int(len(x)), "G1": itp.roi_trial(x[x.graph_delta >= 1.5], "over").get("roi"),
                            "G2": itp.roi_trial(x[x.graph_delta <= -1.5], "under").get("roi")}
                        for s, x in m.groupby("season_cur")}
    box = pov.load_box([2023, 2024, 2025]).sort_values(["athlete_id", "start_utc"])
    box["base10"] = box.groupby(["athlete_id", "season"]).points.transform(
        lambda x: x.shift(1).rolling(10, min_periods=5).mean())
    m2 = m.merge(box[["event_id", "athlete_id", "base10"]], on=["event_id", "athlete_id"]).dropna(subset=["base10"])
    res["desc_line_minus_base10_on_delta"] = itp.slope_ci(m2.assign(y=m2.L - m2.base10), "graph_delta", "y")
    res["desc_actual_minus_base10_on_delta"] = itp.slope_ci(m2.assign(y=m2.actual - m2.base10), "graph_delta", "y")
    m[m.graph_delta >= 1.5].to_csv(OUT / "bets_G1.csv", index=False)
    m[m.graph_delta <= -1.5].to_csv(OUT / "bets_G2.csv", index=False)
    (OUT / "trials.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    if ledger:
        sha = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:12]
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with LEDGER.open("a") as f:
            for n, k in enumerate(["G1_over_ge_1.5", "G2_under_le_-1.5", "G3_slope"], 1):
                rec = {"ts": ts, "strategy": f"nba_matchup_graph_{k}_dev", "sha": sha, "dataset": "nba_matchup_graph",
                       "hypothesis": "pre-registered in notes/nba_matchup_graph.md (commit 373ffad)",
                       "source": "stats.nba.com leagueseasonmatchups prior-season totals + ESPN graded props",
                       "status": "ok", "split": "DEV", "trial_number": n, "study_trials": 3, "bonferroni_bar": G_BAR,
                       "trial": k, "price": "ESPN open", **res[k]}
                f.write(json.dumps(rec, default=str) + "\n")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["parse", "reliability", "example", "signals", "trials"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--ledger", action="store_true")
    a = ap.parse_args()
    if a.cmd == "parse":
        parse()
    elif a.cmd == "reliability":
        reliability(a.ledger)
    elif a.cmd == "signals":
        signals()
    elif a.cmd == "trials":
        trials(a.ledger)
    else:
        example(*a.args)


if __name__ == "__main__":
    main()
