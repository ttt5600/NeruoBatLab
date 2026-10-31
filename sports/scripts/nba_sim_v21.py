"""nba_sim v2.1 tuning + validation harness (shrunk BPM-lite, recency kernel, player x arena). DEV ONLY, no orders.

    ../.venv_quant/bin/python scripts/nba_sim_v21.py tune            # coordinate search on dev 2023-24 only
    ../.venv_quant/bin/python scripts/nba_sim_v21.py confirm         # v2 vs each component vs v2.1 on 2024-25 + 2026-27

Pre-registered rule: notes/nba_sim.md "v2.1 PRE-REGISTRATION". Metric for tuning = spread MAE of the stack MEAN vs the
close (ACT set) on 2023-24 with the team stack fit on 2022-23; player points MAE vs actual for KROLE / recency / arena.
2024-25: stack fit on 2022-23..2023-24. 2026-27: frozen dev stack (2022-23..2024-25), point-in-time state that reads
2025-26 box rows as STATE only (never scored). Never fits or scores 2025-26.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from scipy.stats import norm

import nba_sim as ns
import nba_simple_model as nsm
from sports.market import devig_two_way

OUT = ROOT / "results" / "nba_sim"
BASE = {"A": 0.5, "K": 1500.0, "G": 0.2, "lam": None, "u": 1.0, "c": 0.0, "hs": 14.0, "KA": 0.0, "KM": 0.0,
        "HP": 180.0, "KSHRINK": 500.0, "KROLE": 150.0}


def lines(S: int) -> pd.DataFrame:
    L = pd.read_parquet(nsm.ESPN / f"lines_{S}.parquet")
    L = L[~L.provider.str.lower().isin(nsm.NOT_BOOKS)]
    L = L[L.provider == L.provider.value_counts().index[0]].drop_duplicates("event_id")
    return L[["event_id", "home_spread_close", "home_ml_close", "away_ml_close"]]


def props_open(S: int) -> pd.DataFrame:
    p = nsm.ESPN / f"graded_{S}.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["event_id", "athlete_id", "line_open"])
    x = pd.read_parquet(p, columns=["event_id", "provider", "market", "athlete_id", "side", "line_open"])
    x = x[(x.market == "Total Points") & x.athlete_id.notna() & (x.side == "over")]
    prov = x.provider.value_counts().index[0]
    x = x[x.provider == prov].drop_duplicates(["event_id", "athlete_id"])
    x["event_id"] = x.event_id.astype(str)
    x["athlete_id"] = x.athlete_id.astype(str)
    return x[["event_id", "athlete_id", "line_open"]].assign(prop_book=prov)


def set_cfg(cfg: dict, mode="v21") -> None:
    c = dict(BASE, **cfg)
    ns.set_impact(mode, c["A"], c["K"], c["G"], **{k: c[k] for k in ns.EXTRA0})


def _score(F, PL, p, S, only_home=True) -> tuple[pd.DataFrame, pd.DataFrame]:
    te = F[F.scored & (F.season == S)] if only_home else F[F.season == S]
    te = te.merge(lines(S), on="event_id").dropna(subset=["mu0", "home_spread_close", "dAbs_act", "dPfull"])
    cm = p["margin"]
    mu = (cm["const"] + cm["mu0"] * te.mu0 + cm["dPfull"] * te.dPfull + cm["dAbs"] * te.dAbs_act).to_numpy(float)
    close = -te.home_spread_close.to_numpy(float)
    ph, _, ov = devig_two_way(te.home_ml_close.to_numpy(float), te.away_ml_close.to_numpy(float))
    G = pd.DataFrame({"event_id": te.event_id.astype(str).to_numpy(), "mu": mu, "close": close,
                      "margin": te.margin.to_numpy(float), "p_sim": norm.cdf(mu / p["sigma_m"]),
                      "p_close": np.where((ov >= 0) & (ov <= .10), ph, np.nan)})
    G["y"] = (G.margin > 0).astype(float)
    G["ae_close"] = (G.mu - G.close).abs()
    G["brier"] = (G.p_sim - G.y) ** 2
    rows = []
    evs = set(F.loc[F.season == S, "event_id"])
    for (ev, side), x in PL.items():
        if ev not in evs:
            continue
        for a, mp in x["m"].items():
            if mp >= 15 and a in x["act"]:
                rows.append((str(ev), str(a), mp, mp * x["R"][a]["pts"], x["act"][a][1]))
    P = pd.DataFrame(rows, columns=["event_id", "athlete_id", "min_proj", "pts_proj", "pts"])
    P = P.merge(props_open(S), on=["event_id", "athlete_id"], how="left")
    P["ae"] = (P.pts_proj - P.pts).abs()
    return G, P


def minutes_rows(cfg: dict, S: int, mode="v21") -> pd.DataFrame:
    """Projected (ACT set, before jitter) vs actual minutes for players who played, with the team's game number in
    the season (1 = opener) and the player's previous-season mpg tier (starter proxy: ESPN box has no starter flag)."""
    set_cfg(cfg, mode)
    F, PL = (ns.features(list(range(2021, S + 1)), keep_players=(S,)) if S < 2027 else
             ns.features([2024, 2025, 2026, 2027], keep_players=(2027,)))
    gi = F.set_index("event_id")
    _, box = ns.load_hist([S - 1])
    pm = box.groupby("athlete_id").minutes.agg(["mean", "count"])
    pm = pm[pm["count"] >= 10]["mean"].to_dict()
    rows = []
    for (ev, side), x in PL.items():
        if ev not in gi.index or int(gi.at[ev, "season"]) != S:
            continue
        gn = int(gi.at[ev, f"n_{side}"]) + 1
        for a, mp in x["m"].items():
            if a in x["act"]:
                rows.append((str(ev), str(a), gn, x["R"][a]["role"], pm.get(a, np.nan), mp, x["act"][a][0]))
    M = pd.DataFrame(rows, columns=["event_id", "athlete_id", "gn", "role", "prev_mpg", "min_proj", "min_act"])
    M["tier"] = pd.cut(M.prev_mpg, [0, 15, 25, 32, 60], labels=["<15", "15-25", "25-32", "32+"]).astype(str)
    M.loc[M.prev_mpg.isna(), "tier"] = "none"
    M["ae"] = (M.min_proj - M.min_act).abs()
    M["err"] = M.min_proj - M.min_act
    return M


def eval_dev(cfg: dict, S: int, mode="v21") -> tuple[pd.DataFrame, pd.DataFrame]:
    """S = ESPN season (2024 = 2023-24, 2025 = 2024-25). Stack fit on 2023..S-1."""
    set_cfg(cfg, mode)
    F, PL = ns.features(list(range(2021, S + 1)), keep_players=(S,))
    p = ns.fit_team(F, list(range(2023, S)))
    return _score(F, PL, p, S)


def eval_27(cfg: dict, mode="v21") -> tuple[pd.DataFrame, pd.DataFrame]:
    set_cfg(cfg, mode)
    Fd, _ = ns.features(list(range(2021, 2026)), keep_players=(2025,))   # same cache as fit_all
    p = ns.fit_team(Fd, [2023, 2024, 2025])
    F, PL = ns.features([2024, 2025, 2026, 2027], keep_players=(2027,))
    G, P = _score(F[(F.season_type == "regular-season") & (F.home_ind == 1)].assign(scored=True), PL, p, 2027)
    return G, P


def summ(G, P) -> dict:
    o = {"n": len(G), "mae_close": float(G.ae_close.mean()), "brier": float(G.brier.mean()),
         "brier_close": float(((G.p_close - G.y) ** 2).mean()), "mae_actual": float((G.mu - G.margin).abs().mean()),
         "n_pl": len(P), "pts_mae": float(P.ae.mean())}
    q = P.dropna(subset=["line_open"])
    o["n_prop"] = len(q)
    o["pts_mae_propset"] = float(q.ae.mean()) if len(q) else np.nan
    o["prop_open_mae"] = float((q.line_open - q.pts).abs().mean()) if len(q) else np.nan
    o["pts_mae_vs_prop_open"] = float((q.pts_proj - q.line_open).abs().mean()) if len(q) else np.nan
    return o


def paired(a: pd.DataFrame, b: pd.DataFrame, key, col, cluster="event_id", n=4000, seed=0):
    """mean(b - a) over common rows, cluster bootstrap by game."""
    m = a[key + [col]].merge(b[key + [col]], on=key, suffixes=("_a", "_b")).dropna()
    m["d"] = m[f"{col}_b"] - m[f"{col}_a"]
    g = m.groupby(cluster).d.agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    k = rng.integers(0, len(g), (n, len(g)))
    bs = s[k].sum(1) / c[k].sum(1)
    return float(m.d.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975)), int(len(m))


def brown_rating(cfg, mode="v21") -> dict:
    """Live-state ratings (seasons 2024-2027, as of 2026-10-31) for a few usage creators."""
    set_cfg(cfg, mode)
    from datetime import date
    d = date(2026, 10, 31)
    g, box = ns.load_hist([2024, 2025, 2026, 2027], d)
    past = g[g.date < d]
    st, _ = ns.walk(past, box[box.event_id.isin(set(past.event_id))], lambda *a: None)
    pr = ns.priors(box, g, 2027)
    out = {}
    for nm in ("Jaylen Brown", "Cade Cunningham", "Paolo Banchero", "Ja Morant", "Alperen Sengun", "Anthony Edwards",
               "Jalen Johnson", "Nikola Jokic", "Shai Gilgeous-Alexander", "Giannis Antetokounmpo"):
        ids = [a for a, n in st.names.items() if n == nm]
        if ids:
            out[nm] = round(ns.player_rates(st, ids[0], d.toordinal(), pr)["r"], 2)
    out["_beta_tov_fga"] = [round(pr["bpm"]["beta"][7], 2), round(pr["bpm"]["beta"][1], 2)]
    return out


def tune() -> int:
    log = []
    best = dict(BASE)
    t0 = time.time()

    def run(cfg, why):
        G, P = eval_dev(cfg, 2024)
        s = summ(G, P)
        log.append({"cfg": {k: cfg[k] for k in BASE}, "why": why, **s})
        print(f"[{time.time() - t0:5.0f}s] {why:<14} {ns.impact_tag():<46} MAE {s['mae_close']:.4f}  Brier {s['brier']:.4f}"
              f"  ptsMAE {s['pts_mae']:.4f}", flush=True)
        return s

    def sweep(key, vals, metric="mae_close", why=None):
        res = []
        for v in vals:
            c = dict(best, **{key: v})
            res.append((run(c, why or key)[metric], v))
        best[key] = min(res, key=lambda z: z[0])[1]
        print(f"   -> {key} = {best[key]}", flush=True)

    run(best, "v2-baseline")
    # A: BPM shrink (usage factor, ridge) + the impact blend / state knobs, on spread MAE vs close; two passes
    for pas in (1, 2):
        sweep("u", [1.0, 0.9, 0.75, 0.6])
        sweep("lam", [None, 16.0, 4.0, 1.0])
        sweep("A", [0.25, 0.5, 0.75])
        sweep("G", [0.1, 0.2, 0.3])
        sweep("K", [800.0, 1500.0, 3000.0])
        sweep("KSHRINK", [500.0, 1000.0, 2000.0])
        sweep("HP", [180.0, 270.0, 365.0, 540.0])
    # B recency (stat projections only), KROLE, C arena: player-projection knobs, on player points MAE vs actual
    sweep("c", [0.0, 0.5, 1.0, 2.0, 4.0], metric="pts_mae")
    if best["c"]:
        sweep("hs", [7.0, 14.0, 30.0], metric="pts_mae")
        sweep("c", [0.0, 0.5, 1.0, 2.0, 4.0, 8.0], metric="pts_mae")
    sweep("KROLE", [50.0, 75.0, 150.0, 300.0], metric="pts_mae")
    sweep("KA", [0.0, 500.0, 1000.0, 2000.0, 8000.0], metric="pts_mae")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "v21_tune_2024.json").write_text(json.dumps({"best": best, "log": log}, indent=1, default=str))
    print("BEST", json.dumps(best))
    return 0


def confirm() -> int:
    tb = json.loads((OUT / "v21_tune_2024.json").read_text())["best"]
    # component variants, each ADDED to v2 alone, plus the full tuned v2.1
    A_keys = ("u", "lam", "A", "G", "K", "KSHRINK", "HP")
    var = {"v2": dict(BASE),
           "A_shrinkBPM": dict(BASE, **{k: tb[k] for k in A_keys}),
           "B_recency": dict(BASE, c=tb["c"], hs=tb["hs"]) if tb["c"] else None,
           "C_arena": dict(BASE, KA=tb["KA"] or 8000.0),
           "v21_all": dict(tb)}
    var = {k: v for k, v in var.items() if v is not None}
    R = {}
    for nm, cfg in var.items():
        mode = "v2" if nm == "v2" else "v21"
        R[nm] = {"2024": eval_dev(cfg, 2024, mode), "2025": eval_dev(cfg, 2025, mode), "2027": eval_27(cfg, mode)}
        print(nm, {S: {k: round(v, 4) if isinstance(v, float) else v for k, v in summ(*R[nm][S]).items()}
                   for S in R[nm]}, flush=True)
    out = {"tuned": tb, "variants": {k: {kk: (vv if vv is not None else "None") for kk, vv in v.items()}
                                     for k, v in var.items()}, "summary": {}, "paired_vs_v2": {}}
    for nm in R:
        out["summary"][nm] = {S: summ(*R[nm][S]) for S in R[nm]}
        if nm == "v2":
            continue
        out["paired_vs_v2"][nm] = {}
        for S in R[nm]:
            g0, p0 = R["v2"][S]
            g1, p1 = R[nm][S]
            out["paired_vs_v2"][nm][S] = {
                "mae_close": paired(g0, g1, ["event_id"], "ae_close"),
                "brier": paired(g0, g1, ["event_id"], "brier"),
                "pts_mae": paired(p0, p1, ["event_id", "athlete_id"], "ae")}
            x = out["paired_vs_v2"][nm][S]
            print(f"  {nm:<12} {S}: dMAE {x['mae_close'][0]:+.4f} [{x['mae_close'][1]:+.4f},{x['mae_close'][2]:+.4f}]"
                  f"  dBrier {x['brier'][0]:+.5f} [{x['brier'][1]:+.5f},{x['brier'][2]:+.5f}]"
                  f"  dPtsMAE {x['pts_mae'][0]:+.4f} [{x['pts_mae'][1]:+.4f},{x['pts_mae'][2]:+.4f}] n={x['pts_mae'][3]}")
    out["brown"] = {nm: brown_rating(cfg, "v2" if nm == "v2" else "v21") for nm, cfg in var.items()}
    print(json.dumps(out["brown"], indent=0))
    (OUT / "v21_confirm.json").write_text(json.dumps(out, indent=1, default=str))
    return 0


def minutes() -> int:
    """Early-season minutes check (team games 1-10) + KM (shrink EWMA minutes to last season's mpg) tuned on 2023-24
    games 1-10 minutes MAE, confirmed on 2024-25 and 2026-27 (paired, game-cluster bootstrap) under the same rule."""
    tb = json.loads((OUT / "v21_tune_2024.json").read_text())["best"]
    out = {"tune_2024": {}, "check": {}, "paired": {}}
    early = lambda M: M[M.gn <= 10]
    best, bv = 0.0, None
    for km in (0.0, 1.0, 2.0, 5.0, 10.0, 20.0):
        M = early(minutes_rows(dict(tb, KM=km), 2024))
        out["tune_2024"][km] = float(M.ae.mean())
        print(f"KM {km:>4}: 2023-24 games 1-10 minutes MAE {M.ae.mean():.3f} (n={len(M)})", flush=True)
        if bv is None or M.ae.mean() < bv:
            best, bv = km, float(M.ae.mean())
    out["KM_best"] = best
    print(f"-> KM = {best}")
    for S in (2024, 2025, 2027):
        M0 = minutes_rows(dict(tb, KM=0.0), S)
        M1 = minutes_rows(dict(tb, KM=best), S)
        for nm, M in (("KM0", M0), (f"KM{best:g}", M1)):
            e = early(M)
            tab = e.groupby("tier").agg(n=("ae", "size"), proj=("min_proj", "mean"), act=("min_act", "mean"),
                                        mae=("ae", "mean"))
            rtab = e.groupby("role").agg(n=("ae", "size"), mae=("ae", "mean"), bias=("err", "mean"))
            out["check"][f"{S}_{nm}"] = {"early_mae": float(e.ae.mean()), "late_mae": float(M[M.gn > 10].ae.mean()),
                                         "n_early": len(e), "by_tier": tab.round(3).reset_index().to_dict("records"),
                                         "by_role": rtab.round(3).reset_index().to_dict("records")}
            print(f"\n{S} {nm}: games 1-10 minutes MAE {e.ae.mean():.3f} (n={len(e)}), games 11+ "
                  f"{M[M.gn > 10].ae.mean():.3f}\n" + tab.round(2).to_string() + "\n" + rtab.round(2).to_string())
        if best and S > 2024:
            ev = eval_27 if S == 2027 else (lambda c: eval_dev(c, S))
            g0, p0 = ev(dict(tb, KM=0.0))
            g1, p1 = ev(dict(tb, KM=best))
            out["paired"][f"{S}_game"] = {"mae_close": paired(g0, g1, ["event_id"], "ae_close"),
                                          "brier": paired(g0, g1, ["event_id"], "brier"),
                                          "pts_mae": paired(p0, p1, ["event_id", "athlete_id"], "ae")}
            print(f"  game-level KM{best:g} - KM0 {S}: {out['paired'][f'{S}_game']}")
        if best:
            for lab, f in (("early", early), ("all", lambda M: M)):
                out["paired"][f"{S}_{lab}"] = paired(f(M0), f(M1), ["event_id", "athlete_id"], "ae")
                print(f"  paired minutes MAE KM{best:g} - KM0 ({lab}) {S}: {out['paired'][f'{S}_{lab}']}")
    (OUT / "v21_minutes.json").write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "tune"
    sys.exit({"tune": tune, "confirm": confirm, "minutes": minutes}[cmd]())
