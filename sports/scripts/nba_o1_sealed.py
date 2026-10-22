"""SEALED test O1: frozen simple NBA team-rating model (Model A of nba_player_sim.py) vs the OPENING moneyline.

    ../.venv_quant/bin/python scripts/nba_o1_sealed.py --confirm     # ONE shot (user said yes 2026-10-22)

Pre-registration (notes/player_sim.md s5, written 2026-10-22 before any sealed look):
  - Model A frozen: H 60 days, lambda 5, sigma 12.5, refit every ET date on all earlier games (2021-22 onward), no tuning.
  - Season 2026 (= 2025-26) regular season, not neutral site, both teams >= 5 earlier games (same `scored` filter).
  - Bet the side the model rates >= 0.06 above the de-vigged opener (proportional de-vig, overround in [0, 0.10]).
  - PRIMARY = close-implied EV at the opening price = de-vig same-book CLOSE prob of the bet side x opening decimal
    odds - 1. One-sided bootstrap over games (4000), PASS = mean > 0 and p < 0.05.
  - Secondary: realised ROI at the opening price; CLV (close prob - open prob of bet side).

AMENDMENT 2026-10-22, BEFORE any sealed outcome, model output or price comparison was looked at (only a provider x
month COUNT of rows with open+close moneylines in lines_2026 was viewed):
  ESPN BET openers exist only for 2025-10 .. 2025-12-0x (ESPN BET -> theScore Bet; ESPN's odds feed moved to
  DraftKings), "Draft Kings" covers 2025-11 .. 2026-06. Scoring ESPN BET only would leave ~60-80 bets.
  PRIMARY opener = ESPN BET where it has open+close, else Draft Kings open/close (CLV/close always the SAME book as the
  open). The ESPN-BET-only subset is reported as a secondary line, not a second promotion.
Data: data/espn/nba/{games,box,lines}_2026.parquet; sha256 of games/lines printed and compared with the frozen
MANIFEST (data/frozen/2026-10-14/MANIFEST.json).
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nba_player_sim as nps  # noqa: E402

from sports.market import american_to_decimal, devig_two_way  # noqa: E402

ROOT = nps.ROOT
LOCKBOX = ROOT / "lab" / "lockbox.jsonl"
H, LAM, SIGMA, X = 60, 5, 12.5, 0.06
SEASON = 2026


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def openers(ev: pd.Series) -> pd.DataFrame:
    ln = pd.read_parquet(nps.ESPN / f"lines_{SEASON}.parquet")
    out = []
    for prov in ("ESPN BET", "Draft Kings"):
        b = ln[ln.provider == prov].dropna(subset=["home_ml_open", "away_ml_open", "home_ml_close", "away_ml_close"])
        po, _, ovo = devig_two_way(b.home_ml_open, b.away_ml_open)
        pc, _, ovc = devig_two_way(b.home_ml_close, b.away_ml_close)
        b = b.assign(p_open=po, p_close=pc, ovo=ovo, ovc=ovc, dec_oh=american_to_decimal(b.home_ml_open),
                     dec_oa=american_to_decimal(b.away_ml_open), book=prov)
        b = b[(b.ovo >= 0) & (b.ovo <= .10) & (b.ovc >= 0) & (b.ovc <= .10)].drop_duplicates("event_id")
        out.append(b[["event_id", "book", "p_open", "p_close", "dec_oh", "dec_oa"]])
    o = pd.concat(out)                     # ESPN BET first -> kept by drop_duplicates
    return o[o.event_id.isin(ev)].drop_duplicates("event_id", keep="first")


def boot(v: np.ndarray, rng) -> dict:
    bs = np.array([rng.choice(v, len(v)).mean() for _ in range(4000)])
    return {"n": int(len(v)), "mean": float(v.mean()), "ci_lo": float(np.quantile(bs, .025)),
            "ci_hi": float(np.quantile(bs, .975)), "p": float(np.mean(bs <= 0))}


def main() -> int:
    if "--confirm" not in sys.argv:
        print("sealed: needs --confirm and the user's explicit yes")
        return 1
    if any(json.loads(l).get("strategy") == "nba_O1_open_model" for l in LOCKBOX.read_text().splitlines() if l.strip()):
        print("O1 already scored -- never re-score")
        return 1
    man = json.loads((ROOT / "data" / "frozen" / "2026-10-14" / "MANIFEST.json").read_text())
    fs = json.dumps(man)
    for f in ("games", "lines"):
        h = sha(nps.ESPN / f"{f}_{SEASON}.parquet")
        print(f"{f}_{SEASON} sha256 {h[:12]} in frozen MANIFEST: {h in fs}")

    nps.ESPN_DEV = [2022, 2023, 2024, 2025, SEASON]      # history + the sealed season (load_games reads this global)
    g = nps.load_games()
    g["mu_A"] = nps.ridge_preds(g, H, LAM)
    g["p_A"] = norm.cdf(g.mu_A / SIGMA)
    d = g[g.scored & (g.season == SEASON)].dropna(subset=["p_A"]).copy()
    d = d.merge(openers(d.event_id), on="event_id", how="inner")
    d["y"] = (d.margin > 0).astype(int)
    print(f"sealed games scored {int((g.scored & (g.season == SEASON)).sum())}; with opener {len(d)} "
          f"({d.book.value_counts().to_dict()})")

    diff = d.p_A - d.p_open
    x = d[diff.abs() >= X].copy()
    hs = (x.p_A - x.p_open >= X).to_numpy()
    dec = np.where(hs, x.dec_oh, x.dec_oa)
    pc = np.where(hs, x.p_close, 1 - x.p_close)
    po = np.where(hs, x.p_open, 1 - x.p_open)
    won = np.where(hs, x.y == 1, x.y == 0)
    x["ev_close"], x["roi"], x["clv"] = pc * dec - 1, np.where(won, dec - 1, -1.0), pc - po
    rng = np.random.default_rng(0)
    prim = boot(x.ev_close.to_numpy(), rng)
    roi = boot(x.roi.to_numpy(), np.random.default_rng(1))
    clv = boot(x.clv.to_numpy(), np.random.default_rng(2))
    verdict = "PASS" if prim["mean"] > 0 and prim["p"] < 0.05 else "FAIL"
    print(f"\nO1 bets {len(x)} (home {int(hs.sum())}, away {int((~hs).sum())}; by book {x.book.value_counts().to_dict()})")
    print(f"PRIMARY close-implied EV at open {prim['mean']:+.4f} CI [{prim['ci_lo']:+.4f},{prim['ci_hi']:+.4f}] "
          f"p {prim['p']:.4f} -> {verdict}")
    print(f"  realised ROI at open {roi['mean']:+.4f} CI [{roi['ci_lo']:+.4f},{roi['ci_hi']:+.4f}] p {roi['p']:.4f}; "
          f"units {x.roi.sum():+.2f}; win {won.mean():.3f} @ avg dec {dec.mean():.3f}")
    print(f"  CLV (prob pts) {clv['mean']:+.4f} CI [{clv['ci_lo']:+.4f},{clv['ci_hi']:+.4f}] p {clv['p']:.4f}")
    sub = {}
    for bk, s in x.groupby("book"):
        sub[bk] = {"n": int(len(s)), "ev_close": float(s.ev_close.mean()), "roi": float(s.roi.mean()),
                   "clv": float(s.clv.mean())}
        print(f"  secondary {bk:11s} n {len(s):4d} EV {s.ev_close.mean():+.4f} ROI {s.roi.mean():+.4f} CLV {s.clv.mean():+.4f}")
    x["m"] = pd.to_datetime(x.start_utc).dt.strftime("%Y-%m")
    per = x.groupby("m").ev_close.sum().round(3).to_dict()
    print("  EV units by month", per)
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": "nba_O1_open_model",
           "sha": hashlib.sha256(b"nba_O1_open_model").hexdigest()[:12], "dataset": "nba_player_sim",
           "promotion_number": 1, "bets": len(x), "staked": float(len(x)), "units": float(x.roi.sum()),
           "roi": roi["mean"], "roi_p": roi["p"], "ev_close": prim["mean"], "ci_lo": prim["ci_lo"],
           "ci_hi": prim["ci_hi"], "p": prim["p"], "clv": clv["mean"], "p_clv": clv["p"], "by_book": sub,
           "per_fold": per, "verdict": verdict}
    with LOCKBOX.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print("lockbox: appended")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
