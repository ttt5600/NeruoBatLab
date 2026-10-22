"""NBA player-prop UNDERS from an empirical distribution vs the de-vigged opening market (research top-2). DEV only.

    python scripts/nba_props_dist.py dev [--ledger]

Why (notes/research_prop_models.md s.1b, s.5 #1-#2): prop stats are right-skewed and books set lines near the MEDIAN, so
a MEAN projection (our P1, -3.3%/130k) leans over. Use the player's empirical distribution instead, compare P(under) to
the market's de-vigged P(under), and bet unders (the side the public does not chase; overs lost 4-7% in every market).

Pre-registered 2026-10-21, before scoring (dev 2023-25 only; 2026 sealed):
  markets  : points, rebounds, assists, 3-point FG, PRA (as P1)
  model    : p_emp = share of the player's previous 20 games (>= 10, strictly before tip) with stat < line_open
             (stat == line counted as half: push)
  market   : at each book with BOTH over and under dec_open at the same line_open: p_mkt = (1/du) / (1/du + 1/do)
  P2_under8 : p_emp - p_mkt >= 0.08 at that book -> UNDER; take the best under dec_open across qualifying books;
              one bet per (player, game, market). Players with mean minutes < 20 over the same 20 games excluded
              (NBA 2026-10-16 under-restrictions hit < 12-min players; 20 keeps clear of them).
  1 trial -> bar p < 0.05 (dataset nba_props_dist), one-sided bootstrap over GAMES. Pushes refunded.
Descriptive only (not trials): ROI by market and by skew = (rolling mean - rolling median) / line.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from nba_props_model import D, DEV, MK, history

LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "nba_props_dist"


def rolling_cols(h: pd.DataFrame) -> pd.DataFrame:
    """Per (player, game): the previous-20-game arrays per market + mean minutes (strictly prior)."""
    rows = []
    for aid, g in h.groupby("athlete_id"):
        g = g.sort_values("start_utc")
        vals = {mk: g[cols].sum(axis=1).to_numpy() for mk, cols in MK.items()}
        mins = g.minutes.fillna(0).to_numpy()
        ev = g.event_id.to_numpy()
        for i in range(len(g)):
            lo = max(0, i - 20)
            if i - lo < 10:
                continue
            rows.append({"event_id": ev[i], "athlete_id": aid, "min20": mins[lo:i].mean(),
                         **{f"h::{mk}": vals[mk][lo:i] for mk in MK}})
    return pd.DataFrame(rows)


def bets(seasons) -> tuple[pd.DataFrame, pd.DataFrame]:
    h = history(seasons)
    r = rolling_cols(h)
    gr = pd.concat([pd.read_parquet(D / f"graded_{s}.parquet").assign(season=s) for s in seasons], ignore_index=True)
    gr = gr[gr.market_n.isin(MK) & gr.line_open.notna() & gr.dec_open.notna() & (gr.dec_open > 1) & gr.actual.notna()]
    piv = gr.pivot_table(index=["event_id", "athlete_id", "market_n", "provider", "line_open", "season"], columns="side",
                         values="dec_open", aggfunc="first").reset_index()
    piv = piv.dropna(subset=["over", "under"])
    act = gr.drop_duplicates(["event_id", "athlete_id", "market_n", "line_open"])[["event_id", "athlete_id", "market_n", "line_open", "actual"]]
    piv = piv.merge(act, on=["event_id", "athlete_id", "market_n", "line_open"], how="left")
    piv = piv.merge(r, on=["event_id", "athlete_id"], how="inner")
    piv = piv[piv.min20 >= 20]
    hist = [row[f"h::{mk}"] for mk, row in zip(piv.market_n, piv.to_dict("records"))]
    L = piv.line_open.to_numpy()
    piv["p_emp"] = [np.mean(x < l) + 0.5 * np.mean(x == l) for x, l in zip(hist, L)]
    piv["skew"] = [(np.mean(x) - np.median(x)) / l if l > 0 else np.nan for x, l in zip(hist, L)]
    piv["p_mkt"] = (1 / piv.under) / (1 / piv.under + 1 / piv.over)
    piv = piv.drop(columns=[c for c in piv.columns if c.startswith("h::")])
    cand = piv[piv.p_emp - piv.p_mkt >= 0.08]
    x = cand.sort_values("under", ascending=False).drop_duplicates(["event_id", "athlete_id", "market_n"])
    push = x.actual == x.line_open
    x = x.assign(pnl=np.where(push, 0.0, np.where(x.actual < x.line_open, x.under - 1, -1.0)))
    return x, piv


def cmd_dev() -> int:
    x, piv = bets(DEV)
    ev = x.groupby("event_id").pnl.sum().to_numpy()
    rng = np.random.default_rng(0)
    boots = np.array([rng.choice(ev, len(ev)).sum() for _ in range(3000)])
    per = x.groupby("season").pnl.mean().round(4)
    r = {"bets": int(len(x)), "games": int(len(ev)), "staked": float(len(x)), "units": float(x.pnl.sum()), "roi": float(x.pnl.mean()),
         "ci_lo": float(np.quantile(boots, .025)), "ci_hi": float(np.quantile(boots, .975)), "p": float(np.mean(boots <= 0)),
         "folds_positive": int((per > 0).sum()), "per_fold": per.to_dict()}
    print(f"P2_under8 DEV bets {r['bets']} games {r['games']} units {r['units']:+.1f} ROI {r['roi']:+.2%} "
          f"95%CI [{r['ci_lo']:+.0f}, {r['ci_hi']:+.0f}] p {r['p']:.4f} seasons+ {r['folds_positive']}/3 per-season {r['per_fold']}")
    print("avg under dec", round(x.under.mean(), 3), " mean p_emp - p_mkt", round((x.p_emp - x.p_mkt).mean(), 3))
    print("\nDESCRIPTIVE (not trials) -- by market:")
    print(x.groupby("market_n").pnl.agg(["size", "mean"]).round(4).to_string())
    x["skew_bin"] = pd.qcut(x["skew"], 4, duplicates="drop")
    print("\nDESCRIPTIVE -- by skew quartile ((mean-median)/line):")
    print(x.groupby("skew_bin", observed=True).pnl.agg(["size", "mean"]).round(4).to_string())
    base = piv.copy()
    base["pnl_u"] = np.where(base.actual == base.line_open, 0.0, np.where(base.actual < base.line_open, base.under - 1, -1.0))
    print(f"\nbaseline: ALL unders at these books/lines (>=20-min players) ROI {base.pnl_u.mean():+.2%} on {len(base)}")
    if "--ledger" in sys.argv:
        prior = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
        shas = {q["sha"] for q in prior if q.get("status") == "ok" and q["dataset"] == DATASET}
        sha = hashlib.sha256(b"nba_props_dist:P2_under8").hexdigest()[:12]
        shas.add(sha)
        with LEDGER.open("a") as fh:
            fh.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": "nba_props_P2_under8",
                                 "sha": sha, "dataset": DATASET, "hypothesis": "pre-registered in scripts/nba_props_dist.py (P2)",
                                 "source": "research_prop_models.md top-2 (skew/median, unders)", "status": "ok",
                                 "trial_number": len(shas), **r}, default=str) + "\n")
        print("appended to ledger")
    return 0


def cmd_sealed() -> int:
    """R_REB sealed test (pre-registered 2026-10-21 in STATE.md; user yes 2026-10-21). Scored ONCE.

    P2 rule unchanged, total rebounds only, season 2026 only. History is loaded from 2025+2026 so the last-20 window
    crosses the season boundary exactly as it did inside dev (2024 rolled from 2023). PASS = ROI > 0 AND one-sided
    bootstrap p < 0.05 over games (3000 resamples, seed 0, as dev).
    """
    if "--confirm" not in sys.argv or "R_REB" not in sys.argv:
        print("refusing: needs `sealed --confirm R_REB` (user's explicit yes)")
        return 1
    if any('"strategy": "nba_props_R_REB_sealed"' in l for l in LEDGER.read_text().splitlines()):
        print("refusing: R_REB already scored (see ledger); sealed data is scored once")
        return 1
    x, _ = bets([2025, 2026])
    x = x[(x.season == 2026) & (x.market_n == "total rebounds")]
    ev = x.groupby("event_id").pnl.sum().to_numpy()
    rng = np.random.default_rng(0)
    boots = np.array([rng.choice(ev, len(ev)).sum() for _ in range(3000)])
    r = {"bets": int(len(x)), "games": int(len(ev)), "staked": float(len(x)), "units": float(x.pnl.sum()), "roi": float(x.pnl.mean()),
         "ci_lo": float(np.quantile(boots, .025)), "ci_hi": float(np.quantile(boots, .975)), "p": float(np.mean(boots <= 0)),
         "win_rate": float((x.actual < x.line_open).mean()), "avg_under_dec": float(x.under.mean())}
    r["passed"] = bool(r["roi"] > 0 and r["p"] < 0.05)
    with LEDGER.open("a") as fh:
        fh.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": "nba_props_R_REB_sealed",
                             "sha": hashlib.sha256(b"nba_props_dist:R_REB").hexdigest()[:12], "dataset": "nba_props_dist_sealed2026",
                             "hypothesis": "pre-registered 2026-10-21 STATE.md SEALED ASK R_REB", "source": "user yes 2026-10-21",
                             "status": "ok", "sealed": True, **r}, default=str) + "\n")
    print(f"R_REB SEALED 2026: bets {r['bets']} games {r['games']} units {r['units']:+.1f} ROI {r['roi']:+.2%} "
          f"95%CI units [{r['ci_lo']:+.0f}, {r['ci_hi']:+.0f}] p {r['p']:.4f} win {r['win_rate']:.3f} "
          f"avg dec {r['avg_under_dec']:.3f} -> {'PASS' if r['passed'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"dev": cmd_dev, "sealed": cmd_sealed}.get(cmd, lambda: print(__doc__) or 1)())
