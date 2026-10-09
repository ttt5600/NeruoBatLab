"""Systematic scan of MLB situational factors (bullpen fatigue first) vs CLOSING lines.

    python scripts/mlb_scan.py        # -> lab/reports/mlb_scan.csv

Odds: SBR closing moneyline + total, joined to MLB Stats API games (score-checked).
Dev seasons 2015-2020 only; 2021 is the lab lockbox and is dropped here.
Features come from boxscores of games played on EARLIER dates, so everything is
known before first pitch (the day's lineup/starter is public by then).

Markets:
  total : residual = runs - closing total; rule bets over/under
  ml    : residual = home_win - devigged home prob; rule bets home/away ML
Benjamini-Hochberg across every test, as in nfl_scan.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats

from sports import market, sbr

DATA = Path(__file__).resolve().parents[1] / "data"
OUT = Path(__file__).resolve().parents[1] / "lab" / "reports" / "mlb_scan.csv"
DEV = range(2015, 2021)


def team_day_features(pit: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Per (team, date): bullpen + starter workload from strictly earlier dates."""
    p = pit.copy()
    p["date"] = pd.to_datetime(p["date"])
    rel = p[~p["started"].astype(bool)]
    daily = rel.groupby(["team", "date"]).agg(bp_pitches=("pitches", "sum"), bp_n=("pitcher_id", "nunique"))
    # relievers who pitched on a date -> did they also pitch the day before?
    r2 = rel[["team", "date", "pitcher_id"]].drop_duplicates()
    prev = r2.assign(date=r2["date"] + pd.Timedelta(days=1))
    b2b = r2.merge(prev, on=["team", "date", "pitcher_id"]).groupby(["team", "date"]).size().rename("bp_b2b")
    daily = daily.join(b2b).fillna({"bp_b2b": 0})
    out = []
    for team, g in games.groupby("team"):
        dates = g["date"].sort_values().unique()
        d = daily.loc[team] if team in daily.index.get_level_values(0) else pd.DataFrame()
        for dt in dates:
            w1 = d.loc[(d.index >= dt - pd.Timedelta(days=1)) & (d.index < dt)] if len(d) else d
            w3 = d.loc[(d.index >= dt - pd.Timedelta(days=3)) & (d.index < dt)] if len(d) else d
            out.append({"team": team, "date": dt,
                        "bp_pitches_1d": w1["bp_pitches"].sum() if len(w1) else 0.0,
                        "bp_pitches_3d": w3["bp_pitches"].sum() if len(w3) else 0.0,
                        "bp_arms_3d": w3["bp_n"].sum() if len(w3) else 0.0,
                        # relievers who pitched on BOTH of the last two days (likely unavailable today)
                        "bp_b2b_tired": w1["bp_b2b"].sum() if len(w1) else 0.0})
    return pd.DataFrame(out)


def starter_features(pit: pd.DataFrame) -> pd.DataFrame:
    s = pit[pit["started"].astype(bool)].copy()
    s["date"] = pd.to_datetime(s["date"])
    s = s.sort_values(["pitcher_id", "date"])
    g = s.groupby("pitcher_id")
    s["sp_rest"] = (s["date"] - g["date"].shift(1)).dt.days
    s["sp_prev_pitches"] = g["pitches"].shift(1)
    return s[["game_pk", "team", "sp_rest", "sp_prev_pitches"]]


def main() -> int:
    mlb = pd.read_parquet(DATA / "mlb_games.parquet")
    pit = pd.read_parquet(DATA / "mlb_pitching.parquet")
    j = pd.concat([sbr.join_to_mlb(sbr.season(y), mlb) for y in DEV], ignore_index=True)
    j = j[j["score_match"]].copy()
    j["game_pk"] = j["game_id"].astype(int)
    first_box = pd.to_datetime(pit["date"]).min()
    j = j[j["date"] >= first_box + pd.Timedelta(days=4)]       # need 3 days of history
    print(f"dev games with odds + box history: {len(j)} ({j.season.min()}-{j.season.max()})")

    long = pd.concat([j[["date", "home"]].rename(columns={"home": "team"}),
                      j[["date", "away"]].rename(columns={"away": "team"})])
    tf = team_day_features(pit, long.drop_duplicates())
    sf = starter_features(pit)
    for side in ("home", "away"):
        j = j.merge(tf.add_prefix(f"{side}_").rename(columns={f"{side}_team": side, f"{side}_date": "date"}),
                    on=[side, "date"], how="left")
        j = j.merge(sf.rename(columns={"team": side}).rename(columns=lambda c: f"{side}_{c}" if c.startswith("sp_") else c),
                    on=["game_pk", side], how="left")
    # day game after a night game (team played last night, today starts before 17:00 local-ish)
    st = pd.to_datetime(j["start_utc"], utc=True)
    j["local_hour"] = (st.dt.tz_convert("America/New_York").dt.hour)
    sched = pd.concat([j[["date", "home", "local_hour"]].rename(columns={"home": "team"}),
                       j[["date", "away", "local_hour"]].rename(columns={"away": "team"})])
    night = set(map(tuple, sched[sched["local_hour"] >= 18][["team", "date"]].astype(str).to_numpy()))
    for side in ("home", "away"):
        yday = (j["date"] - pd.Timedelta(days=1)).astype(str)
        j[f"{side}_day_after_night"] = [float((t, y) in night and h < 16) for t, y, h in
                                        zip(j[side], yday, j["local_hour"])]

    feats = ["bp_pitches_1d", "bp_pitches_3d", "bp_arms_3d", "bp_b2b_tired", "sp_rest", "sp_prev_pitches",
             "day_after_night"]
    X = {}
    for f in feats:
        X[f"diff_{f}"] = j[f"home_{f}"] - j[f"away_{f}"]
        X[f"sum_{f}"] = j[f"home_{f}"] + j[f"away_{f}"]
    X = pd.DataFrame(X)
    p_home, _, _ = market.devig_two_way(j["ml_close_home"], j["ml_close_away"])
    home_win = (j["home_score"] > j["away_score"]).astype(float)
    ys = {"total": (j["home_score"] + j["away_score"] - j["ou_close"]).to_numpy(float),
          "ml": (home_win - p_home).to_numpy(float)}
    rows = []
    for c in X.columns:
        x = X[c].to_numpy(float)
        for mkt, y in ys.items():
            if mkt == "ml" and c.startswith("sum_"):
                continue
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 300 or np.nanstd(x[ok]) == 0:
                continue
            r = stats.linregress(x[ok], y[ok])
            xb, sub, yy = x[ok], j[ok], y[ok]
            if len(np.unique(xb)) <= 5:
                centre = stats.mode(xb, keepdims=False).mode
                flag = xb != centre
            else:
                centre = np.median(xb)
                flag = ((xb <= np.quantile(xb, 0.2)) | (xb >= np.quantile(xb, 0.8))) & (xb != centre)
            sub, yy, up = sub[flag], yy[flag], (r.slope * (xb[flag] - centre)) > 0
            if mkt == "total":
                won = np.where(up, yy > 0, yy < 0)
                odds = np.where(up, sub["over_odds"], sub["under_odds"])
                push = yy == 0
            else:
                hw = (sub["home_score"] > sub["away_score"]).to_numpy()
                won = np.where(up, hw, ~hw)
                odds = np.where(up, sub["ml_close_home"], sub["ml_close_away"])
                push = np.zeros(len(won), bool)
            dec = np.asarray(market.american_to_decimal(pd.Series(odds)), float)
            pnl = np.where(push, 0.0, np.where(won, dec - 1, -1.0))
            pnl = pnl[np.isfinite(pnl)]
            rows.append({"feature": c, "market": mkt, "n": int(ok.sum()), "slope": r.slope,
                         "t": r.slope / r.stderr, "p": r.pvalue,
                         "rule": ("home/over" if r.slope > 0 else "away/under") + " when x high",
                         "rule_n": len(pnl), "rule_units": pnl.sum(), "rule_roi": pnl.mean() if len(pnl) else np.nan})
    res = pd.DataFrame(rows)
    p = res["p"].to_numpy()
    o = np.argsort(p)
    q = np.empty_like(p)
    q[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    res["q_bh"] = np.minimum(q, 1)
    res = res.sort_values("p")
    res.to_csv(OUT, index=False)
    pd.set_option("display.width", 200)
    print(f"{len(res)} tests; BH q<0.10: {(res.q_bh < 0.10).sum()}")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
