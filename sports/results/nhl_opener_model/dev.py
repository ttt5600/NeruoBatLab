"""Dev book check (descriptive): 2023-24 + 2024-25 vs ESPN BET total open/close. Model at the game's ET date with the
actual starter. c = trailing mean(m - open_med) over prior 100 games in-season (>= 30). Bet |g| >= k."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
OUT = Path(__file__).parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(OUT)); sys.path.insert(0, str(OUT.parent / "kalshi_openers_xleague"))
import model as M  # noqa: E402
from q1_timing import nhl_book_median  # noqa: E402

rows = []
for y in (2024, 2025):  # ESPN labels: 2024 = 2023-24, 2025 = 2024-25 (2026 = 2025-26 SEALED, not read)
    g = pd.read_parquet(ROOT / "data/espn/nhl" / f"games_{y}.parquet")
    g = g[(g.season_type.astype(str).str.contains("regular|2", regex=True)) & (g.status == "STATUS_FINAL") & (~g.neutral)]
    ln = pd.read_parquet(ROOT / "data/espn/nhl" / f"lines_{y}.parquet")
    ln = ln[ln.provider == "ESPN BET"].drop_duplicates("event_id")
    g = g.merge(ln[["event_id", "total_open", "over_odds_open", "under_odds_open", "total_close", "over_odds_close",
                    "under_odds_close"]], on="event_id")
    g["date"] = pd.to_datetime(g.start_utc, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    for r in g.sort_values("start_utc").itertuples():
        h, a = M.ESPN2MP.get(r.home, r.home), M.ESPN2MP.get(r.away, r.away)
        sh = M.STARTS[(M.STARTS.playerTeam == h) & (M.STARTS.date == r.date)]
        sa = M.STARTS[(M.STARTS.playerTeam == a) & (M.STARTS.date == r.date)]
        if sh.empty or sa.empty:
            continue
        try:
            lh, la, m, bh, ba = M.predict(h, a, r.date, {sh.playerId.iloc[0]: 1.0}, {sa.playerId.iloc[0]: 1.0})
        except KeyError:
            continue
        rows.append({"season": y, "event_id": r.event_id, "date": r.date, "home": h, "away": a, "lh": lh, "la": la, "m": m,
                     "final": r.home_score + r.away_score,
                     "open_med": nhl_book_median(r.total_open, r.over_odds_open, r.under_odds_open),
                     "close_med": nhl_book_median(r.total_close, r.over_odds_close, r.under_odds_close)})
d = pd.DataFrame(rows).dropna(subset=["m", "open_med", "close_med"]).sort_values("date").reset_index(drop=True)
d["c"] = np.nan
for s in d.season.unique():
    ix = d.index[d.season == s]
    diff = (d.m - d.open_med).loc[ix]
    for i in ix:
        prior = diff[(d.date.loc[ix] < d.date[i])].tail(100)
        if len(prior) >= 30:
            d.at[i, "c"] = prior.mean()
d["g"] = d.m - d.c - d.open_med
d.to_parquet(OUT / "dev_rows.parquet")


def boot(v, cl, n=4000, seed=0):
    v = pd.Series(np.asarray(v, float)); cl = pd.Series(np.asarray(cl))
    ok = v.notna(); v, cl = v[ok].values, cl[ok].values
    u = np.unique(cl); idx = {c: np.where(cl == c)[0] for c in u}
    rng = np.random.default_rng(seed); bs = []
    for _ in range(n):
        pick = rng.choice(u, len(u)); ii = np.concatenate([idx[c] for c in pick]); bs.append(v[ii].mean())
    bs = np.array(bs)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float((bs <= 0).mean())


res = {"n_games": int(len(d)), "n_with_c": int(d.c.notna().sum()),
       "mae_model_debiased": float((d.m - d.c - d.final).abs().mean()), "mae_open_med": float((d.open_med - d.final).abs().mean()),
       "mae_close_med": float((d.close_med - d.final).abs().mean()), "c_mean": float(d.c.mean()),
       "corr_gap_vs_move": float(np.corrcoef(d.g.dropna(), (d.close_med - d.open_med)[d.g.notna()])[0, 1])}
for k in (0.25, 0.4, 0.5, 0.6):
    b = d[d.g.abs() >= k]
    sg = np.sign(b.g)
    res[f"k{k}"] = {"n": int(len(b)), "over_share": float((sg > 0).mean()),
                    "clv_goals": boot(sg * (b.close_med - b.open_med), b.date),
                    "by_season_clv": {int(s): float((sg * (b.close_med - b.open_med))[b.season == s].mean()) for s in b.season.unique()}}
# stack vs close: (final - close_med) on gap to close (same c)
x = d.dropna(subset=["c"]); gc = (x.m - x.c - x.close_med).values; yv = (x.final - x.close_med).values
slope = float(np.polyfit(gc, yv, 1)[0])
rng = np.random.default_rng(0); u = x.date.unique(); grp = {k: np.where(x.date.values == k)[0] for k in u}; sl = []
for _ in range(2000):
    ii = np.concatenate([grp[k] for k in rng.choice(u, len(u))]); sl.append(np.polyfit(gc[ii], yv[ii], 1)[0])
res["stack_close_slope"] = [slope, float(np.percentile(sl, 2.5)), float(np.percentile(sl, 97.5))]
gO = (x.m - x.c - x.open_med).values; yO = (x.final - x.open_med).values; slo = []
for _ in range(2000):
    ii = np.concatenate([grp[k] for k in rng.choice(u, len(u))]); slo.append(np.polyfit(gO[ii], yO[ii], 1)[0])
res["stack_open_slope"] = [float(np.polyfit(gO, yO, 1)[0]), float(np.percentile(slo, 2.5)), float(np.percentile(slo, 97.5))]
json.dump(res, open(OUT / "dev.json", "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
