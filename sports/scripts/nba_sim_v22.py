"""nba_sim v2.2 tuning + validation harness: PLAYER minutes map (role/starter, top-end slope, cap). DEV ONLY, no orders.

    ../.venv_quant/bin/python scripts/nba_sim_v22.py tune       # coordinate search on dev 2023-24 only -> v22_tune_2024.json
    ../.venv_quant/bin/python scripts/nba_sim_v22.py confirm    # each component + combo vs v2.1 on 2024-25 and 2026-27

Pre-registered rule: notes/nba_sim.md "v2.2 PRE-REGISTRATION" (commit 08fc20e, before any tuning run).
The map acts on the v2.1 ACT-set projections (x["m"] in the feature cache), exactly as side_inputs applies it live
(nba_sim.adjust_minutes), so the team stack and the v2.1 caches are untouched. 2025-26 is never scored.
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

import nba_sim as ns
import nba_sim_v21 as v21

OUT = ROOT / "results" / "nba_sim"
OFF = {"bB": 0.0, "bS": 0.0, "b15": 0.0, "sT": 0.0, "CAP2": 42.0}
COMP = {"R": ("bB", "bS", "b15"), "T": ("sT",), "C": ("CAP2",)}


def v21_cfg() -> dict:
    tb = json.loads((OUT / "v21_tune_2024.json").read_text())["best"]
    return dict(tb, KM=1.0)                      # = PRESETS["v21"] (KM added by the 10-31 minutes study)


_PL = {}


def load(S: int):
    """v2.1 feature cache (F, PL) for scored season S (2024 = 2023-24, 2025 = 2024-25, 2027 = 2026-27 walk-forward)."""
    if S not in _PL:
        v21.set_cfg(v21_cfg(), "v21")
        F, PL = (ns.features(list(range(2021, S + 1)), keep_players=(S,)) if S < 2027 else
                 ns.features([2024, 2025, 2026, 2027], keep_players=(2027,)))
        gi = F.set_index("event_id")
        _, box = ns.load_hist([S - 1])
        pm = box.groupby("athlete_id").minutes.agg(["mean", "count"])
        pm = pm[pm["count"] >= 10]["mean"].to_dict()
        keys = [k for k in PL if k[0] in gi.index and int(gi.at[k[0], "season"]) == S]
        gn = {k: int(gi.at[k[0], f"n_{k[1]}"]) + 1 for k in keys}
        _PL[S] = (PL, keys, gn, pm)
    return _PL[S]


def rows(S: int, mcfg: dict) -> pd.DataFrame:
    PL, keys, gn, pm = load(S)
    for k, v in OFF.items():
        ns.IMPACT[k] = float(mcfg.get(k, v))
    out = []
    for key in keys:
        x = PL[key]
        roles = {a: x["R"][a]["role"] for a in x["m"]}
        m1 = ns.adjust_minutes(x["m"], roles)
        for a, m0 in x["m"].items():
            act = x["act"].get(a)
            if act is None:
                continue
            out.append((str(key[0]), str(a), gn[key], roles[a], pm.get(a, np.nan), m0, m1[a], act[0],
                        m0 * x["R"][a]["pts"], m1[a] * x["R"][a]["pts"], act[1]))
    for k, v in OFF.items():
        ns.IMPACT[k] = v
    M = pd.DataFrame(out, columns=["event_id", "athlete_id", "gn", "role", "prev_mpg", "m0", "m1", "min_act",
                                   "pts0", "pts1", "pts"])
    M["tier"] = pd.cut(M.prev_mpg, [0, 15, 25, 32, 60], labels=["<15", "15-25", "25-32", "32+"]).astype(str)
    M.loc[M.prev_mpg.isna(), "tier"] = "none"
    M["ae_min0"], M["ae_min1"] = (M.m0 - M.min_act).abs(), (M.m1 - M.min_act).abs()
    M["ae_pts0"], M["ae_pts1"] = (M.pts0 - M.pts).abs(), (M.pts1 - M.pts).abs()
    return M


def paired(M: pd.DataFrame, a: str, b: str, n=4000, seed=0):
    d = (M[b] - M[a]).to_numpy()
    g = pd.DataFrame({"e": M.event_id.to_numpy(), "d": d}).groupby("e").d.agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    k = np.random.default_rng(seed).integers(0, len(g), (n, len(g)))
    bs = s[k].sum(1) / c[k].sum(1)
    return [float(d.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975)), int(len(d))]


def metrics(M: pd.DataFrame) -> dict:
    P = M[M.m0 >= 15]                            # player set fixed by the v2.1 projection (pre-registered)
    return {"min_mae0": float(M.ae_min0.mean()), "min_mae1": float(M.ae_min1.mean()),
            "pts_mae0": float(P.ae_pts0.mean()), "pts_mae1": float(P.ae_pts1.mean()),
            "d_min": paired(M, "ae_min0", "ae_min1"), "d_pts": paired(P, "ae_pts0", "ae_pts1"),
            "d_min_early": paired(M[M.gn <= 10], "ae_min0", "ae_min1")}


def bias_table(M: pd.DataFrame, col: str) -> dict:
    e = M.assign(err0=M.m0 - M.min_act, err1=M.m1 - M.min_act)
    out = {}
    for lab, x in (("early", e[e.gn <= 10]), ("all", e)):
        t = x.groupby(col).agg(n=("err0", "size"), bias0=("err0", "mean"), bias1=("err1", "mean"),
                               mae0=("ae_min0", "mean"), mae1=("ae_min1", "mean"))
        out[lab] = t.round(3).reset_index().to_dict("records")
    return out


def tune() -> int:
    t0 = time.time()
    log = []
    grids = {"bB": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0], "bS": [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
             "b15": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0], "sT": [0.0, 0.1, 0.2, 0.3, 0.5, 0.75],
             "CAP2": [36.0, 38.0, 40.0, 42.0]}

    def score(c):
        M = rows(2024, c)
        return float(M.ae_min1.mean())

    best = {}
    for comp, ks in COMP.items():                # each component tuned alone on top of v2.1
        cur = dict(OFF)
        for _ in range(2):
            for k in ks:
                res = [(score(dict(cur, **{k: v})), v) for v in grids[k]]
                cur[k] = min(res)[1]
                log.append({"comp": comp, "key": k, "res": res})
                print(f"[{time.time() - t0:4.0f}s] {comp} {k}: " + "  ".join(f"{v:g}:{s:.4f}" for s, v in res), flush=True)
        best[comp] = {k: cur[k] for k in ks}
    combo = dict(OFF)                            # combination: start from the component optima, one more pass
    for comp in COMP:
        combo.update(best[comp])
    for _ in range(2):
        for k in OFF:
            res = [(score(dict(combo, **{k: v})), v) for v in grids[k]]
            combo[k] = min(res)[1]
            log.append({"comp": "combo", "key": k, "res": res})
            print(f"[{time.time() - t0:4.0f}s] combo {k}: " + "  ".join(f"{v:g}:{s:.4f}" for s, v in res), flush=True)
    base = score(OFF)
    print(f"v2.1 2023-24 minutes MAE {base:.4f}; combo {score(combo):.4f} {combo}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "v22_tune_2024.json").write_text(json.dumps({"components": best, "combo": combo, "v21_mae": base,
                                                        "log": log}, indent=1))
    return 0


def passes(r: dict) -> bool:
    """Pre-registered keep rule on {season: metrics}: for minutes AND points, point estimate < 0 in both OOS seasons
    and the 95% CI upper bound < 0 in at least one."""
    ok = True
    for m in ("d_min", "d_pts"):
        pe = [r[S][m][0] < 0 for S in ("2025", "2027")]
        ci = [r[S][m][2] < 0 for S in ("2025", "2027")]
        ok &= all(pe) and any(ci)
    return bool(ok)


def confirm() -> int:
    T = json.loads((OUT / "v22_tune_2024.json").read_text())
    var = {f"{c}_alone": dict(OFF, **T["components"][c]) for c in COMP}
    var["combo"] = dict(T["combo"])
    out = {"tuned": T["components"], "combo": T["combo"], "results": {}, "pass": {}, "bias": {}}
    for nm, c in var.items():
        out["results"][nm] = {}
        for S in (2024, 2025, 2027):
            M = rows(S, c)
            out["results"][nm][str(S)] = metrics(M)
            if nm == "combo":
                out["bias"][str(S)] = {"tier": bias_table(M, "tier"), "role": bias_table(M, "role")}
        out["pass"][nm] = passes(out["results"][nm])
        r = out["results"][nm]
        print(f"\n{nm} {c}  PASS={out['pass'][nm]}")
        for S in r:
            x = r[S]
            print(f"  {S}: minutes MAE {x['min_mae0']:.3f} -> {x['min_mae1']:.3f}  d {x['d_min'][0]:+.4f} "
                  f"[{x['d_min'][1]:+.4f}, {x['d_min'][2]:+.4f}] n={x['d_min'][3]}   pts MAE {x['pts_mae0']:.3f} -> "
                  f"{x['pts_mae1']:.3f}  d {x['d_pts'][0]:+.4f} [{x['d_pts'][1]:+.4f}, {x['d_pts'][2]:+.4f}]"
                  f"  early-min d {x['d_min_early'][0]:+.4f} [{x['d_min_early'][1]:+.4f}, {x['d_min_early'][2]:+.4f}]")
    for S, b in out["bias"].items():
        print(f"\ncombo bias {S} (proj - actual):")
        for k in ("tier", "role"):
            for lab in ("early", "all"):
                print(f"  {k} {lab}: " + "; ".join(f"{d[k]} n{d['n']} {d['bias0']:+.2f}->{d['bias1']:+.2f}"
                                                   for d in b[k][lab]))
    (OUT / "v22_confirm.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "tune"
    sys.exit({"tune": tune, "confirm": confirm}[cmd]())
