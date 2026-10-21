"""NBA player props vs a rolling-form model, at OPENING prices (QUEUE M4 props half). DEV only.

    python scripts/nba_props_model.py dev [--ledger]

Data: data/espn/nba/graded_<season>.parquet (multi-book: MGM, DraftKings, ESPN BET, Caesars NJ, SugarHouse, PointsBet;
line_open/dec_open + graded actual) and box_<season>.parquet (per-player stats), games_<season>.parquet (start_utc).
ESPN season label = end year (2024 = 2023-24). Trap: last_updated is often AFTER tip (hours_before < 0), so the current
dec/line can be in-game; we use ONLY line_open / dec_open (pre-game).

Pre-registered 2026-10-21, before scoring:
  markets : total points, total rebounds, total assists, total 3-point field goals, total points, rebounds, and assists
  model   : player's mean of that stat over his previous 10 games (strictly before this game's start_utc, any opponent),
            >= 5 prior games; minutes are not modelled (pure form).
  P1 form15 : model >= 1.15 * line_open -> OVER; model <= 0.85 * line_open -> UNDER; bet at the BEST dec_open across
              books for that (player, game, market, side, line_open); one bet per (player, game, market).
  split   : DEV = seasons 2023-2025; SEALED = 2026 (frozen-protocol sealed season; user's yes).
  1 trial -> bar p < 0.05, one-sided bootstrap over GAMES. Pushes refunded.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd

D = ROOT / "data" / "espn" / "nba"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "nba_props_model"
MK = {"total points": ["points"], "total rebounds": ["rebounds"], "total assists": ["assists"],
      "total 3-point field goals": ["threePointFieldGoalsMade"],
      "total points, rebounds, and assists": ["points", "rebounds", "assists"]}
DEV = [2023, 2024, 2025]


def history(seasons) -> pd.DataFrame:
    rows = []
    for s in seasons:
        b = pd.read_parquet(D / f"box_{s}.parquet")
        g = pd.read_parquet(D / f"games_{s}.parquet")[["event_id", "start_utc"]]
        b = b[b.stat.isin(["points", "rebounds", "assists", "threePointFieldGoalsMade", "minutes"])]
        w = b.pivot_table(index=["event_id", "athlete_id"], columns="stat", values="value", aggfunc="first").reset_index()
        w = w.merge(g, on="event_id", how="inner")
        rows.append(w)
    h = pd.concat(rows, ignore_index=True)
    h = h[h.minutes.fillna(0) > 0]  # played
    h["start_utc"] = pd.to_datetime(h.start_utc, utc=True)
    return h.sort_values(["athlete_id", "start_utc"])


def form(h: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean of the previous 10 games (shifted -> strictly prior), per stat combo."""
    out = h[["event_id", "athlete_id", "start_utc"]].copy()
    g = h.groupby("athlete_id")
    for mk, cols in MK.items():
        v = h[cols].sum(axis=1)
        out[mk] = v.groupby(h.athlete_id).transform(lambda x: x.shift(1).rolling(10, min_periods=5).mean())
    return out


def bets(seasons) -> pd.DataFrame:
    h = history(seasons)
    f = form(h).melt(id_vars=["event_id", "athlete_id", "start_utc"], var_name="market_n", value_name="model")
    gr = pd.concat([pd.read_parquet(D / f"graded_{s}.parquet").assign(season=s) for s in seasons], ignore_index=True)
    gr = gr[gr.market_n.isin(MK) & gr.line_open.notna() & gr.dec_open.notna() & (gr.dec_open > 1) & gr.actual.notna()]
    gr = gr.merge(f, on=["event_id", "athlete_id", "market_n"], how="inner")
    gr = gr[gr.model.notna()]
    over = (gr.side == "over") & (gr.model >= 1.15 * gr.line_open)
    under = (gr.side == "under") & (gr.model <= 0.85 * gr.line_open)
    x = gr[over | under]
    x = x.sort_values("dec_open", ascending=False).drop_duplicates(["event_id", "athlete_id", "market_n"])
    win = np.where(x.side == "over", x.actual > x.line_open, x.actual < x.line_open)
    push = x.actual == x.line_open
    x = x.assign(pnl=np.where(push, 0.0, np.where(win, x.dec_open - 1, -1.0)))
    return x


def cmd_dev() -> int:
    x = bets(DEV)
    ev = x.groupby("event_id").pnl.sum().to_numpy()
    rng = np.random.default_rng(0)
    boots = np.array([rng.choice(ev, len(ev)).sum() for _ in range(3000)])
    per = x.groupby("season").pnl.agg(["size", "mean"]).round(4)
    r = {"bets": int(len(x)), "games": int(len(ev)), "staked": float(len(x)), "units": float(x.pnl.sum()),
         "roi": float(x.pnl.mean()), "ci_lo": float(np.quantile(boots, .025)), "ci_hi": float(np.quantile(boots, .975)),
         "p": float(np.mean(boots <= 0)), "folds_positive": int((per["mean"] > 0).sum()), "per_fold": per["mean"].to_dict()}
    print(f"P1_form15 DEV bets {r['bets']} games {r['games']} units {r['units']:+.1f} ROI {r['roi']:+.2%} "
          f"95%CI [{r['ci_lo']:+.0f}, {r['ci_hi']:+.0f}] p {r['p']:.4f} seasons+ {r['folds_positive']}/3")
    print(per.to_string())
    print(x.groupby(["market_n", "side"]).pnl.agg(["size", "mean"]).round(4).to_string())
    print("avg dec_open", round(x.dec_open.mean(), 3), "| books:", x.provider.value_counts().head(5).to_dict())
    if "--ledger" in sys.argv:
        prior = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
        shas = {q["sha"] for q in prior if q.get("status") == "ok" and q["dataset"] == DATASET}
        sha = hashlib.sha256(b"nba_props_model:P1_form15").hexdigest()[:12]
        shas.add(sha)
        with LEDGER.open("a") as fh:
            fh.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": "nba_props_P1_form15",
                                 "sha": sha, "dataset": DATASET, "hypothesis": "pre-registered in scripts/nba_props_model.py (P1)",
                                 "source": "user: player patterns / models on past info", "status": "ok",
                                 "trial_number": len(shas), **r}, default=str) + "\n")
        print("appended to ledger")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"dev": cmd_dev}.get(cmd, lambda: print(__doc__) or 1)())
