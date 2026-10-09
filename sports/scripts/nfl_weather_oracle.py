"""Oracle test: does PERFECT knowledge of in-game weather beat the NFL closing total?

    python scripts/nfl_weather_oracle.py

Dev seasons 2006-2019 only (2020-2025 is the lab lockbox). Weather is observed
(ERA5), so this is an upper bound on any forecast-based strategy.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sports import market, nfl

DATA = Path(__file__).resolve().parents[1] / "data"
rng = np.random.default_rng(0)


def under_roi(d):
    dec = np.asarray(market.american_to_decimal(d["under_odds"]), float)
    resid = d["resid"].to_numpy()
    pnl = np.where(resid < 0, dec - 1, np.where(resid > 0, -1.0, 0.0))
    boots = [rng.choice(pnl, len(pnl)).mean() for _ in range(2000)]
    return len(pnl), pnl.sum(), pnl.mean(), np.mean(np.array(boots) <= 0)


def main() -> int:
    raw = nfl.raw_games()
    raw = raw[(raw.season >= 2006) & (raw.season <= 2019)]
    wx = pd.read_parquet(DATA / "nfl_game_wx.parquet")
    d = raw.merge(wx, on="game_id").dropna(subset=["total_line", "under_odds", "home_score"])
    d["resid"] = d["home_score"] + d["away_score"] - d["total_line"]
    print(f"dev outdoor games with weather + total: {len(d)}; mean resid {d.resid.mean():+.2f} pts\n")

    buckets = {
        "all outdoor": d.index == d.index,
        "dry": d.wx_precip_mm < 0.1,
        "any rain >=1mm": d.wx_precip_mm >= 1,
        "heavy rain >=5mm": d.wx_precip_mm >= 5,
        "wet >=2 of 4 hours": d.wx_wet_hours >= 2,
        "snow >=0.5cm": d.wx_snow_cm >= 0.5,
        "wind mean >=15mph": d.wx_wind_mean >= 15,
        "gust max >=30mph": d.wx_gust_max >= 30,
        "rain + wind>=12": (d.wx_precip_mm >= 1) & (d.wx_wind_mean >= 12),
        "cold <25F": d.wx_temp_mean < 25,
    }
    print(f"{'bucket':22s} {'n':>5s} {'resid':>7s} {'under units':>11s} {'ROI':>7s} {'p(ROI<=0)':>9s}")
    for name, m in buckets.items():
        s = d[m]
        if len(s) < 15:
            print(f"{name:22s} {len(s):5d}  (too few)")
            continue
        n, u, roi, p = under_roi(s)
        print(f"{name:22s} {n:5d} {s.resid.mean():+7.2f} {u:+11.1f} {roi:+7.1%} {p:9.3f}")

    # how much of the weather effect does the line already carry?
    import statsmodels.formula.api as smf
    d["pts"] = d["home_score"] + d["away_score"]
    d["rain"], d["snow"] = (d.wx_precip_mm >= 1).astype(int), (d.wx_snow_cm >= 0.5).astype(int)
    raw_fit = smf.ols("pts ~ rain + snow + wx_wind_mean", d).fit()
    beyond = smf.ols("resid ~ rain + snow + wx_wind_mean", d).fit()
    print("\neffect on total points (raw)      :", {k: f"{v:+.2f} (se {raw_fit.bse[k]:.2f})" for k, v in raw_fit.params.items() if k != "Intercept"})
    print("effect BEYOND the closing line   :", {k: f"{v:+.2f} (se {beyond.bse[k]:.2f})" for k, v in beyond.params.items() if k != "Intercept"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
