"""Causal version of the rain+wind unders oracle: weather KNOWN AT KICKOFF only.

    python scripts/nfl_weather_nowcast.py            # print results
    python scripts/nfl_weather_nowcast.py --ledger   # also append the trials to lab/ledger.jsonl

The oracle (nfl_weather_oracle.py) used ERA5 weather observed DURING the game,
which no bettor has. Here the signal is the stadium weather over the 3 hours up
to kickoff (ERA5 hours labelled k-2..k, precipitation = accumulation ending at
the label, so all of it is over by the floored kickoff hour). A bettor taking
the closing total just before kickoff can see this (radar, gauges). True NWP
forecasts are not usable for dev: Open-Meteo's forecast archives start 2021+,
inside the 2020-25 lockbox.

Pre-registered before looking (2026-10-16), dev 2006-2019 only, bet the under:
  T1 pre_rain_wind_unders : pre precip >= 1 mm and pre wind mean >= 12 mph
  T2 pre_rain_unders      : pre precip >= 1 mm
(wind-only is ledger trial 1, wind_unders, already scored -- not repeated.)
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from sports import market, nfl
from nfl_game_weather import STADIUMS, hourly

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "lab" / "ledger.jsonl"
DEV = (2006, 2019)
TRIALS = {
    "pre_rain_wind_unders": ("Rain + wind visible before kickoff (3h pre-kickoff precip >= 1 mm and wind mean >= 12 mph, "
                             "ERA5 at the stadium) predicts the under beyond the closing total. Causal version of the "
                             "observed-weather oracle (+32.7%/70).", lambda d: (d.pre_precip >= 1) & (d.pre_wind >= 12)),
    "pre_rain_unders": ("Rain visible before kickoff (3h pre-kickoff precip >= 1 mm) predicts the under beyond the "
                        "closing total.", lambda d: d.pre_precip >= 1),
}


def frame() -> pd.DataFrame:
    raw = nfl.raw_games()
    # same filter + per stadium-season request span as nfl_game_weather.py -> cached responses
    raw = raw[(raw.season >= 2006) & (raw.season <= 2025) & raw.roof.isin(["outdoors", "open"])]
    g = nfl.from_raw(raw).assign(stadium_id=raw["stadium_id"].to_numpy())
    rows = []
    for sid, grp in g.groupby("stadium_id"):
        lat, lon = STADIUMS[sid]
        for season, sg in grp.groupby("season"):
            if not DEV[0] <= season <= DEV[1]:
                continue
            t0, t1 = sg["start_utc"].min(), sg["start_utc"].max() + pd.Timedelta(hours=5)
            h = hourly(lat, lon, t0.date().isoformat(), t1.date().isoformat())
            for _, r in sg.iterrows():
                if pd.isna(r["start_utc"]):
                    continue
                k = r["start_utc"].floor("h")
                pre, during = h.loc[k - pd.Timedelta(hours=2):k], h.loc[k:k + pd.Timedelta(hours=3)]
                if len(pre) < 3 or len(during) < 3:
                    continue
                rows.append({"game_id": r["game_id"], "pre_precip": pre["precipitation"].sum(),
                             "pre_wind": pre["wind_speed_10m"].mean(),
                             "game_precip": during["precipitation"].sum(),
                             "game_wind": during["wind_speed_10m"].mean()})
    wx = pd.DataFrame(rows)
    full = nfl.raw_games()
    d = full.merge(wx, on="game_id").dropna(subset=["total_line", "under_odds", "home_score"])
    d["resid"] = d["home_score"] + d["away_score"] - d["total_line"]
    return d


def score(s: pd.DataFrame, rng) -> dict:
    dec = np.asarray(market.american_to_decimal(s["under_odds"]), float)
    pnl = np.where(s.resid < 0, dec - 1, np.where(s.resid > 0, -1.0, 0.0))
    boots = np.array([rng.choice(pnl, len(pnl)).sum() for _ in range(5000)])
    per = pd.Series(pnl, index=s.index).groupby(s.season).sum().round(2)
    return {"bets": int(len(pnl)), "staked": float(len(pnl)), "units": float(pnl.sum()), "roi": float(pnl.mean()),
            "ci_lo": float(np.quantile(boots, 0.025)), "ci_hi": float(np.quantile(boots, 0.975)),
            "p": float(np.mean(boots <= 0)), "folds_positive": int((per > 0).sum()),
            "per_fold": {str(k): v for k, v in per.items()}}


def main() -> int:
    d = frame()
    print(f"dev outdoor games with pre-kickoff weather + close: {len(d)}")
    rain_pre, rain_game = d.pre_precip >= 1, d.game_precip >= 1
    print(f"pre-kickoff rain >=1mm: {rain_pre.sum()} games; of those, in-game rain >=1mm: "
          f"{(rain_pre & rain_game).sum()} ({(rain_pre & rain_game).sum() / max(rain_pre.sum(), 1):.0%}); "
          f"in-game rain games caught in advance: {(rain_pre & rain_game).sum()}/{rain_game.sum()}\n")
    prior = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    shas = {r["sha"] for r in prior if r.get("status") == "ok" and r["dataset"] == "nfl_total"}
    recs = []
    for name, (hyp, rule) in TRIALS.items():
        s = d[rule(d)]
        res = score(s, np.random.default_rng(0))
        sha = hashlib.sha256((name + hyp).encode()).hexdigest()[:12]
        shas.add(sha)
        recs.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": name, "sha": sha,
                     "dataset": "nfl_total", "hypothesis": hyp, "source": "scripts/nfl_weather_nowcast.py",
                     "status": "ok", "trial_number": len(shas), **res})
        print(f"{name:22s} bets {res['bets']:4d}  resid {s.resid.mean():+.2f}  units {res['units']:+6.1f}  "
              f"ROI {res['roi']:+6.1%}  95%CI units [{res['ci_lo']:+.1f}, {res['ci_hi']:+.1f}]  "
              f"p {res['p']:.4f}  seasons+ {res['folds_positive']}/{len(res['per_fold'])}")
    n = len(shas)
    print(f"\nnfl_total trials now {n} -> bar p < {0.05 / n:.4f}")
    if "--ledger" in sys.argv:
        with LEDGER.open("a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"appended {len(recs)} trials to {LEDGER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
