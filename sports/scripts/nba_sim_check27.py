"""2026-27 walk-forward check of nba_sim (legacy v1 vs v2 vs v2.1 impact) against Draft Kings closes (ESPN lines_2027).

    ../.venv_quant/bin/python scripts/nba_sim_check27.py            # all 2026-27 regular-season home games to date

Point-in-time: player/team state for each game uses only games dated before it (nba_sim.features). Team-stack params
are the frozen dev fits (2022-23..2024-25: results/nba_sim_params.json for legacy, nba_sim_params_v2.json for v2);
nothing is fit on 2026-27 or 2025-26 here. 2025-26 box rows enter only as player/team STATE for 2026-27 predictions,
as the live `game` command does. ACT = who actually played (injury-report proxy, lookahead on late scratches);
STRICT = the previous game's players. Unlike live, the full rotation is last-10-games (no ESPN roster feed history).
Descriptive only; not a bet.
"""
from __future__ import annotations

import json
import sys
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


def score(F: pd.DataFrame, p: dict, season: int, lines: pd.DataFrame, only_scored=False) -> dict:
    te = F[(F.season == season) & (F.season_type == "regular-season") & (F.home_ind == 1)]
    if only_scored:
        te = te[te.scored]
    te = te.merge(lines, on="event_id").dropna(subset=["mu0", "home_spread_close", "dAbs_act", "dAbs_strict"])
    close = -te.home_spread_close.to_numpy(float)
    ph_c, _, ov = devig_two_way(te.home_ml_close.to_numpy(float), te.away_ml_close.to_numpy(float))
    ph_c = np.where((ov >= 0) & (ov <= .10), ph_c, np.nan)
    y = (te.margin > 0).to_numpy(float)
    out = {"n": int(len(te))}
    for v in ("act", "strict"):
        cm = p[v]["margin"]
        sim = (cm["const"] + cm["mu0"] * te.mu0 + cm["dPfull"] * te.dPfull + cm["dAbs"] * te[f"dAbs_{v}"]).to_numpy(float)
        ps = norm.cdf(sim / p[v]["sigma_m"])
        b, _ = ns.ols(sim[:, None], close)
        ok = np.isfinite(ph_c)
        lg = lambda q: np.log(np.clip(q, 1e-4, 1 - 1e-4) / (1 - np.clip(q, 1e-4, 1 - 1e-4)))
        bp, _ = ns.ols(lg(ps[ok])[:, None], lg(ph_c[ok]))
        fav = np.abs(close) >= 6
        out[v] = {"mae_close": float(np.abs(sim - close).mean()), "bias": float((sim - close).mean()),
                  "slope_close_on_sim": float(b[1]), "slope_logit_pclose_on_psim": float(bp[1]),
                  "mae_close_fav6": float(np.abs(sim - close)[fav].mean()),
                  "signed_gap_fav6": float(np.mean((np.abs(sim) - np.abs(close))[fav])),
                  "n_fav6": int(fav.sum()),
                  "brier_sim": float(np.mean((ps - y) ** 2)), "brier_close": float(np.nanmean((ph_c - y) ** 2)),
                  "mae_p_vs_close": float(np.nanmean(np.abs(ps - ph_c))),
                  "err": sim - close, "ps": ps}
    out["y"], out["pc"] = y, ph_c
    return out


def main() -> int:
    L = pd.read_parquet(nsm.ESPN / "lines_2027.parquet")
    L = L[~L.provider.str.lower().isin(nsm.NOT_BOOKS)]
    L = L[L.provider == L.provider.value_counts().index[0]].drop_duplicates("event_id")
    L = L[["event_id", "home_spread_close", "home_ml_close", "away_ml_close"]]
    R = {}
    for mode in ("legacy", "v2", "v21"):
        ns.set_impact(mode)
        F, _ = ns.features([2024, 2025, 2026, 2027])
        base = json.loads(ns.params_path().read_text())
        # STRICT-set stack coefficients are not in the params file: refit them on the same dev features/seasons
        Fd, _ = ns.features(list(range(2021, 2026)), keep_players=(2025,))
        p = {"act": base, "strict": ns.fit_team(Fd, base["train_seasons"], var="strict")}
        R[mode] = score(F, p, 2027, L)
        x = R[mode]
        for v in ("act", "strict"):
            o = x[v]
            print(f"{mode:6} {v:6} n={x['n']}  MAE vs close {o['mae_close']:.2f} (bias {o['bias']:+.2f})  slope close~sim "
                  f"{o['slope_close_on_sim']:.2f}  logit-slope p_close~p_sim {o['slope_logit_pclose_on_psim']:.2f}  "
                  f"|close|>=6 (n={o['n_fav6']}): MAE {o['mae_close_fav6']:.2f}, |sim|-|close| {o['signed_gap_fav6']:+.2f}  "
                  f"Brier sim {o['brier_sim']:.4f} close {o['brier_close']:.4f}  mean|p_sim-p_close| {o['mae_p_vs_close']:.3f}")
    rng = np.random.default_rng(0)
    for v in ("act", "strict"):
        for a_, b_ in (("v2", "legacy"), ("v21", "v2")):
            d = np.abs(R[a_][v]["err"]) - np.abs(R[b_][v]["err"])
            bs = d[rng.integers(0, len(d), (4000, len(d)))].mean(1)
            print(f"paired MAE {a_} - {b_} ({v}): {d.mean():+.3f} [{np.quantile(bs, .025):+.3f}, {np.quantile(bs, .975):+.3f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
