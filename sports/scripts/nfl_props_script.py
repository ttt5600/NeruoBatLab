"""L4 -- NFL player-prop sides by category x game script (DEV ONLY; 2025 SEALED, never read).

    ../.venv_quant/bin/python scripts/nfl_props_script.py inventory   # what the props files hold (no outcomes)
    ../.venv_quant/bin/python scripts/nfl_props_script.py dev [--ledger]

PRE-REGISTRATION (written 2026-10-25 after inspecting ONLY props/lines/games schemas and market
counts -- no outcome, box score or graded file had been joined to any line when this was frozen)
------------------------------------------------------------------------------------------------
Data reality found before freezing (props_2022..2024, no 2025 file opened):
  * 2022 + 2023: ESPN carries NO yardage/reception/attempt props -- only TD-scorer "yes" markets,
    QB interceptions O/U and tackles O/U. The yardage hypothesis cannot be tested there.
  * 2024: one book (ESPN BET), all yardage markets, 100% line_open + dec_open on both sides.
    `line`/`dec` carry last_updated stamps that are mostly AFTER kickoff -> untrusted, NOT used.
  => Dev sample = 2024 only. The brief's halves (2022 vs 2023-24) are impossible; halves are
     H1 = games before 2024-11-04 (weeks 1-9) vs H2 = later (weeks 10-18 + postseason).

Price: OPENING line and price (line_open, dec_open), both sides required at the same line_open.
Game script: the same book's OPENING home spread / total (lines_2024 home_spread_open, total_open)
-- known when the openers are posted. Team spread = home spread for the home player, negated for
the away player. Player team comes from the box-score team_id (players who did not play are void,
as books void them).
One observation per (event, athlete, market): if several open pairs exist, keep the pair whose
de-vigged over prob is closest to 0.5 (the main line). Drop pairs with overround outside
[1.00, 1.20], and pushes.
QB = athlete with a passing-yards, passing-attempts or pass-completions prop in that event
(props-only definition, no outcomes). Non-QB = everyone else.

Cells (8, fixed; side fixed; thresholds fixed):
  C1 FAV_QB_PASSYDS_U   total passing yards,  team spread <= -6.5           UNDER
  C2 FAV_QB_ATT_U       total passing attempts, team spread <= -6.5         UNDER
  C3 DOG_RB_RUSHYDS_U   total rushing yards, non-QB, team spread >= +6.5    UNDER
  C4 DOG_REC_O          total receptions, team spread >= +6.5               OVER
  C5 LOWTOT_PASSYDS_U   total passing yards, game total <= 40.5             UNDER
  C6 LOWTOT_RECYDS_U    total receiving yards, game total <= 40.5           UNDER
  C7 LOWTOT_RUSHYDS_U   total rushing yards (all), game total <= 40.5       UNDER
  C8 RB_RECYDS_U        total receiving yards, non-QB athletes who also have a rushing-yards
                        prop that game (RB proxy), any script (PFF in-sample winner)    UNDER
Primary metric: residual = 1{side wins} - de-vigged (proportional) open prob of that side.
Test: mean residual, cluster-robust SE by event_id, two-sided normal p.
Secondary: ROI per 1u at dec_open (same clustering).
PASS (per cell) = Bonferroni p < 0.05/8 on the residual AND BH q < 0.05 over the 8 cells AND
mean residual > 0 AND ROI > 0 AND residual > 0 in both halves. Even a pass is one book, one
season: it would earn a sealed 2025 ask, nothing more.
Descriptive, not trials: blind-under baseline per market (all scripts), raw under hit rate vs
de-vigged under prob, so the reader sees how much of any "under" result is skew already priced.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sports import settle  # noqa: E402

D = ROOT / "data" / "espn" / "nfl"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "nfl_props_script"
DEV = (2022, 2023, 2024)
SEALED = 2025
H_SPLIT = pd.Timestamp("2024-11-04", tz="UTC")
BIG, LOWTOT = 6.5, 40.5
QB_MKTS = {"total passing yards", "total passing attempts", "total pass completions"}
CELLS = ["C1_FAV_QB_PASSYDS_U", "C2_FAV_QB_ATT_U", "C3_DOG_RB_RUSHYDS_U", "C4_DOG_REC_O",
         "C5_LOWTOT_PASSYDS_U", "C6_LOWTOT_RECYDS_U", "C7_LOWTOT_RUSHYDS_U", "C8_RB_RECYDS_U"]
BAR = 0.05 / len(CELLS)


def _read(kind: str, season: int) -> pd.DataFrame:
    assert season != SEALED and season in DEV, f"season {season} is sealed / not dev"
    return pd.read_parquet(D / f"{kind}_{season}.parquet")


def inventory() -> None:
    for y in DEV:
        p = _read("props", y)
        print(y, p.provider.unique().tolist())
        print(p.groupby("market").agg(n=("dec", "size"), open=("dec_open", lambda s: s.notna().mean())).to_string())


def pairs(season: int) -> pd.DataFrame:
    p = _read("props", season)
    p = p[p.side.isin(["over", "under"]) & p.line_open.notna() & p.dec_open.notna() & p.athlete_id.notna()].copy()
    p["market_n"] = p.market.map(settle.normalise)
    k = ["event_id", "athlete_id", "market_n", "line_open"]
    o = p[p.side == "over"].groupby(k, as_index=False).dec_open.first().rename(columns={"dec_open": "o"})
    u = p[p.side == "under"].groupby(k, as_index=False).dec_open.first().rename(columns={"dec_open": "u"})
    x = o.merge(u, on=k)
    x["over_round"] = 1 / x.o + 1 / x.u
    x = x[(x.over_round >= 1.0) & (x.over_round <= 1.20)].copy()
    x["p_over"] = (1 / x.o) / x.over_round
    x["d"] = (x.p_over - 0.5).abs()
    x = x.sort_values("d").drop_duplicates(["event_id", "athlete_id", "market_n"]).drop(columns="d")
    qb = set(zip(p.event_id[p.market_n.isin(QB_MKTS)], p.athlete_id[p.market_n.isin(QB_MKTS)]))
    rb = set(zip(p.event_id[p.market_n == "total rushing yards"], p.athlete_id[p.market_n == "total rushing yards"]))
    x["qb"] = [(e, a) in qb for e, a in zip(x.event_id, x.athlete_id)]
    x["has_rush"] = [(e, a) in rb for e, a in zip(x.event_id, x.athlete_id)]
    return x.reset_index(drop=True)


def attach(x: pd.DataFrame, season: int) -> pd.DataFrame:
    """Join game script, then (only here) outcomes."""
    g = _read("games", season)[["event_id", "start_utc", "season_type", "home_team_id", "away_team_id"]]
    ln = _read("lines", season)
    ln = ln[ln.provider.str.contains("ESPN BET")].drop_duplicates("event_id")[["event_id", "home_spread_open", "total_open"]]
    b = _read("box", season)
    team = b.dropna(subset=["athlete_id"]).groupby(["event_id", "athlete_id"]).team_id.agg(lambda s: s.mode().iat[0]).rename("team_id").reset_index()
    x = x.merge(g, on="event_id").merge(ln, on="event_id", how="left").merge(team, on=["event_id", "athlete_id"], how="left")
    x["home"] = x.team_id.astype(str) == x.home_team_id.astype(str)
    x["team_spread"] = np.where(x.team_id.isna(), np.nan, np.where(x.home, x.home_spread_open, -x.home_spread_open))
    x["actual"] = settle._actual("nfl", x.rename(columns={}), b).to_numpy()
    x = x[x.actual.notna()].copy()
    x["push"] = x.actual == x.line_open
    x = x[~x.push].copy()
    x["over_won"] = (x.actual > x.line_open).astype(float)
    x["half"] = np.where(x.start_utc < H_SPLIT, "H1", "H2")
    return x


def side_cols(x: pd.DataFrame, side: str) -> pd.DataFrame:
    y = x.copy()
    if side == "over":
        y["won"], y["p"], y["dec"] = y.over_won, y.p_over, y.o
    else:
        y["won"], y["p"], y["dec"] = 1 - y.over_won, 1 - y.p_over, y.u
    y["resid"] = y.won - y.p
    y["profit"] = np.where(y.won == 1, y.dec - 1, -1.0)
    return y


def cl_mean(y: pd.DataFrame, col: str):
    """Mean with cluster-robust (by event) SE; two-sided normal p."""
    from scipy.stats import norm
    n = len(y)
    if n < 2:
        return dict(n=n, mean=None, se=None, p=None)
    m = y[col].mean()
    s = (y[col] - m).groupby(y.event_id).sum()
    G = len(s)
    se = np.sqrt((s ** 2).sum() * G / max(G - 1, 1)) / n
    return dict(n=int(n), games=int(G), mean=float(m), se=float(se), p=float(2 * norm.sf(abs(m / se))) if se > 0 else None)


def cell(x: pd.DataFrame, name: str) -> pd.DataFrame:
    m, ts, tot = x.market_n, x.team_spread, x.total_open
    sel = {
        "C1_FAV_QB_PASSYDS_U": ((m == "total passing yards") & (ts <= -BIG), "under"),
        "C2_FAV_QB_ATT_U": ((m == "total passing attempts") & (ts <= -BIG), "under"),
        "C3_DOG_RB_RUSHYDS_U": ((m == "total rushing yards") & ~x.qb & (ts >= BIG), "under"),
        "C4_DOG_REC_O": ((m == "total receptions") & (ts >= BIG), "over"),
        "C5_LOWTOT_PASSYDS_U": ((m == "total passing yards") & (tot <= LOWTOT), "under"),
        "C6_LOWTOT_RECYDS_U": ((m == "total receiving yards") & (tot <= LOWTOT), "under"),
        "C7_LOWTOT_RUSHYDS_U": ((m == "total rushing yards") & (tot <= LOWTOT), "under"),
        "C8_RB_RECYDS_U": ((m == "total receiving yards") & ~x.qb & x.has_rush, "under"),
    }[name]
    return side_cols(x[sel[0].fillna(False)], sel[1])


def bh(ps):
    ps = np.asarray(ps, float)
    o = np.argsort(ps)
    q = ps[o] * len(ps) / (np.arange(len(ps)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def dev() -> int:
    x = attach(pairs(2024), 2024)
    res = {"n_pairs": int(len(x)), "games": int(x.event_id.nunique()), "team_missing": float(x.team_spread.isna().mean()),
           "cells": {}, "baseline_under": {}}
    for name in CELLS:
        y = cell(x, name)
        r = {"resid": cl_mean(y, "resid"), "roi": cl_mean(y, "profit"), "hit": float(y.won.mean()) if len(y) else None,
             "p_devig": float(y.p.mean()) if len(y) else None, "overround": float(y.over_round.mean()) if len(y) else None,
             "halves": {h: [cl_mean(y[y.half == h], "resid")["mean"], cl_mean(y[y.half == h], "profit")["mean"], int((y.half == h).sum())]
                        for h in ("H1", "H2")}}
        res["cells"][name] = r
    q = bh([res["cells"][c]["resid"]["p"] for c in CELLS])
    for c, qq in zip(CELLS, q):
        r = res["cells"][c]
        r["bh_q"] = float(qq)
        both = all(v[0] is not None and v[0] > 0 for v in r["halves"].values())
        r["passed"] = bool(r["resid"]["p"] < BAR and qq < 0.05 and r["resid"]["mean"] > 0 and r["roi"]["mean"] > 0 and both)
    for m, g in x.groupby("market_n"):
        y = side_cols(g, "under")
        res["baseline_under"][m] = {"n": int(len(y)), "under_hit": float(y.won.mean()), "under_p_devig": float(y.p.mean()),
                                    "resid": cl_mean(y, "resid")["mean"], "resid_p": cl_mean(y, "resid")["p"],
                                    "roi": cl_mean(y, "profit")["mean"], "overround": float(y.over_round.mean())}
    print(json.dumps(res, indent=1, default=float))
    (ROOT / "data" / "lab").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "lab" / "nfl_props_script_results.json").write_text(json.dumps(res, indent=1, default=float))
    if "--ledger" in sys.argv:
        with LEDGER.open("a") as f:
            for i, c in enumerate(CELLS, 1):
                r = res["cells"][c]
                f.write(json.dumps({
                    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"nfl_props_script_{c}",
                    "sha": hashlib.sha256(f"nfl_props_script:{c}".encode()).hexdigest()[:12], "dataset": DATASET,
                    "metric": "resid_vs_devig_open", "trial": c, "hypothesis": f"pre-registered in scripts/nfl_props_script.py ({c})",
                    "source": "lead brief 2026-10-25: L4 NFL prop sides by category x game script (2024 ESPN BET openers)",
                    "status": "ok", "trial_number": i, "study_trials": len(CELLS), "bonferroni_bar": BAR,
                    "bets": r["resid"]["n"], "mean": r["resid"]["mean"], "roi": r["roi"]["mean"], "p": r["resid"]["p"],
                    "bh_q": r["bh_q"], "per_fold": r["halves"], "passed": r["passed"]}, default=float) + "\n")
        print(f"ledger += {len(CELLS)} rows")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "dev"
    raise SystemExit(inventory() if cmd == "inventory" else dev())
