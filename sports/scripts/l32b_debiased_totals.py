"""L32b dev scorer: NBA opening totals with a point-in-time LEVEL correction. Pre-registration:
notes/l32b_debiased_totals.md (committed before this was run). DEV only (2023-24, 2024-25 = T5's games).

    ../.venv_quant/bin/python scripts/l32b_debiased_totals.py dev [--ledger]

c(g) = mean(T2_strict - OPEN) over the last N=100 same-season games with both, on ET dates strictly before g's
(>= 30 required, else skip).  Pick OVER if T2 - c - OPEN >= 3, UNDER if <= -3; graded vs OPEN at no-vig open.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import nba_simple_model as M  # noqa: E402  (read-only import; frozen code)

N_WIN, N_MIN, THR = 100, 30, 3.0
DATASET = "l32b_debiased_totals"
OUT = ROOT / "results" / "l32b_debiased_totals"


def level_correction(d: pd.DataFrame, pred: str, line: str) -> pd.Series:
    """Point-in-time c per row: same season, strictly earlier ET date, last N_WIN games with pred & line."""
    c = pd.Series(np.nan, index=d.index)
    for s, g in d.groupby("season"):
        g = g[g[pred].notna() & g[line].notna()].sort_values("start_utc")
        gap = (g[pred] - g[line]).to_numpy(float)
        dates = g["date"].to_numpy()
        for dt in np.unique(dates):
            prior = gap[dates < dt][-N_WIN:]
            if len(prior) >= N_MIN:
                c.loc[g.index[dates == dt]] = prior.mean()
    return c


def boot(v, seed=0):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), (4000, len(v)))].mean(1)
    return {"n": int(len(v)), "roi": float(v.mean()), "lo": float(np.quantile(bs, .025)),
            "hi": float(np.quantile(bs, .975)), "p": float(np.mean(bs <= 0))}


def summarize(x: pd.DataFrame) -> dict:
    r = {"novig": boot(x.pnl_novig), "m110": boot(x.pnl_110),
         "win_rate_ex_push": float(x.win[~x.push].mean()), "overs": int((x.side > 0).sum()),
         "unders": int((x.side < 0).sum()), "clv_pts": float(x.clv.mean()),
         "per_season": {str(s): {"n": int(len(g)), "novig": float(g.pnl_novig.mean()), "m110": float(g.pnl_110.mean())}
                        for s, g in x.groupby("season")},
         "by_side": {k: {"n": int(len(g)), "novig": float(g.pnl_novig.mean())}
                     for k, g in x.groupby(np.where(x.side > 0, "over", "under"))}}
    return r


def cmd_dev(ledger: bool) -> int:
    wf = pd.read_pickle(M.OUTD / "wf.pkl")
    d = M.markets(wf)
    assert set(d.season.unique()) <= {2024, 2025}, d.season.unique()
    d["c"] = level_correction(d, "T2_strict", "total_open")
    d["T2_adj"] = d.T2_strict - d.c
    have = d[d.c.notna()].copy()
    print(f"dev games {len(d)}; with opener {d.total_open.notna().sum()}; with c {len(have)}")
    print("c by season: " + ", ".join(f"{s}: mean {g.c.mean():+.2f} min {g.c.min():+.2f} max {g.c.max():+.2f}"
                                       for s, g in have.groupby("season")))
    xb = M.bets(have, "tot", "T2_adj", "total_open", "tot_close_same", "pf_over_open", THR)
    x0 = M.bets(have, "tot", "T2_strict", "total_open", "tot_close_same", "pf_over_open", THR)   # original T5 rule
    res = {"rule": {"N": N_WIN, "min": N_MIN, "thr": THR}, "games_with_c": int(len(have)),
           "L32b": summarize(xb), "T5_same_games": summarize(x0)}
    b = res["L32b"]
    b["passed"] = bool(b["novig"]["p"] < .05 and b["novig"]["roi"] > 0 and
                       all(v["novig"] > 0 for v in b["per_season"].values()))
    too_good = b["novig"]["roi"] > .10 and b["novig"]["n"] > 200
    res["too_good_flag"] = bool(too_good)
    for k in ("L32b", "T5_same_games"):
        r = res[k]
        print(f"\n{k}: n {r['novig']['n']}  overs {r['overs']} unders {r['unders']}  win {r['win_rate_ex_push']:.3f}  "
              f"CLV {r['clv_pts']:+.2f} pts")
        for m in ("novig", "m110"):
            q = r[m]
            print(f"  ROI {m:5s} {q['roi']:+.4f}  [{q['lo']:+.4f}, {q['hi']:+.4f}]  p {q['p']:.4f}")
        print("  per season " + json.dumps({s: {kk: round(vv, 4) for kk, vv in v.items()} for s, v in r["per_season"].items()}))
        print("  by side " + json.dumps({s: {kk: round(vv, 4) for kk, vv in v.items()} for s, v in r["by_side"].items()}))
    print(f"\nL32b PASS: {b['passed']}   too-good flag: {too_good}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dev.json").write_text(json.dumps(res, indent=1))
    xb[["event_id", "season", "date", "home", "away", "T2_strict", "c", "T2_adj", "total_open", "tot_close_same",
        "total", "open_book", "pf_over_open", "side", "win", "push", "pnl_novig", "pnl_110", "clv"]].to_csv(
        OUT / "dev_bets.csv", index=False)
    if ledger:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=12", "HEAD"], capture_output=True,
                             text=True).stdout.strip()
        row = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": "L32b_debiased_open_totals",
               "sha": sha, "dataset": DATASET,
               "hypothesis": "T2 minus trailing-100 same-season mean(T2-open), |adj-open|>=3, beats OPEN at no-vig",
               "source": "lead ask 2026-11-01 (L32 over-bias)", "status": "ok", "split": "DEV 2023-24, 2024-25",
               "trial_number": 1, "study_trials": 1, "n": b["novig"]["n"], "roi_novig": b["novig"]["roi"],
               "ci_novig": [b["novig"]["lo"], b["novig"]["hi"]], "p": b["novig"]["p"], "roi_110": b["m110"]["roi"],
               "ci_110": [b["m110"]["lo"], b["m110"]["hi"]], "overs": b["overs"], "unders": b["unders"],
               "per_season": b["per_season"], "t5_same_games_roi_novig": res["T5_same_games"]["novig"]["roi"],
               "passed": b["passed"], "verdict": "PASS dev" if b["passed"] else "DEAD on dev"}
        with open(ROOT / "lab" / "ledger.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print("ledger row written")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "dev":
        print(__doc__)
        sys.exit(1)
    sys.exit(cmd_dev("--ledger" in sys.argv))
