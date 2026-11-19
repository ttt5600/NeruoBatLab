"""M1: P(team made FGs >= k), k=1..4, logistic on implied pts (+^2) + indoor. Fit 2023-2025 nflverse (amended pre-join) (prereg notes/kalshi_fg_markets.md)."""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
OUT = Path(__file__).resolve().parent
PBP = Path.home() / ".claude/quant-kalshi/data/pbp"
s = pd.read_csv(Path.home() / ".claude/quant-kalshi/data/nfl_schedule.csv")
def team_games(sch):
    h = sch.assign(team=sch.home_team, implied=(sch.total_line + sch.spread_line) / 2, is_home=1)
    a = sch.assign(team=sch.away_team, implied=(sch.total_line - sch.spread_line) / 2, is_home=0)
    t = pd.concat([h, a])
    t["indoor"] = t.roof.isin(["dome", "closed"]).astype(int)
    return t[["game_id", "season", "week", "gameday", "gametime", "team", "is_home", "implied", "indoor", "roof", "home_team", "away_team"]]
def fg_counts(seasons):
    out = []
    for y in seasons:
        p = pd.read_parquet(PBP / f"play_by_play_{y}.parquet", columns=["game_id", "posteam", "field_goal_result", "play_type"])
        m = p[p.field_goal_result == "made"].groupby(["game_id", "posteam"]).size().rename("fg").reset_index()
        g = p[["game_id"]].drop_duplicates()
        out.append((m, g))
    return out
def X(t):
    return np.c_[t.implied, t.implied ** 2 / 10, t.indoor]
def fit():
    tr = team_games(s[(s.season >= 2023) & (s.season <= 2025) & s.home_score.notna() & s.spread_line.notna()])
    parts = fg_counts(range(2023, 2026))
    m = pd.concat([a for a, _ in parts]); games = pd.concat([b for _, b in parts])
    tr = tr[tr.game_id.isin(games.game_id)]
    tr = tr.merge(m, left_on=["game_id", "team"], right_on=["game_id", "posteam"], how="left").drop(columns="posteam")
    tr["fg"] = tr.fg.fillna(0).astype(int)
    assert tr.duplicated(["game_id", "team"]).sum() == 0
    models = {k: LogisticRegression(C=1e6, max_iter=2000).fit(X(tr), (tr.fg >= k).astype(int)) for k in range(1, 5)}
    return tr, models
def predict(models, t):
    P = np.column_stack([models[k].predict_proba(X(t))[:, 1] for k in range(1, 5)])
    P = np.minimum.accumulate(P, axis=1)
    return pd.DataFrame(P, columns=[f"p{k}" for k in range(1, 5)], index=t.index)
if __name__ == "__main__":
    tr, models = fit()
    print("train team-games", len(tr), "fg dist", tr.fg.value_counts().sort_index().to_dict(), "mean", round(tr.fg.mean(), 3))
    pr = predict(models, tr)
    for k in range(1, 5):
        print(f"k={k} base {(tr.fg>=k).mean():.3f} pred mean {pr[f'p{k}'].mean():.3f}")
    # in-sample calibration by implied bins
    tr["bin"] = pd.cut(tr.implied, [0, 17, 20, 23, 26, 29, 50])
    print(pd.concat([tr, pr], axis=1).groupby("bin", observed=True).apply(lambda g: pd.Series({"n": len(g), **{f"obs{k}": (g.fg >= k).mean() for k in range(1, 5)}, **{f"p{k}": g[f"p{k}"].mean() for k in range(1, 5)}})).round(3).to_string())
    # 2026 predictions (all games with lines), no outcomes used
    t26 = team_games(s[(s.season == 2026) & s.spread_line.notna()])
    t26 = pd.concat([t26, predict(models, t26)], axis=1)
    t26.to_parquet(OUT / "preds_2026.parquet")
    print("2026 team-games predicted", len(t26))
    # holdout sanity: fit 2018-2023, score 2024-25
