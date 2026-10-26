"""DEV-only (ESPN seasons 2023-2025 = 2022-23..2024-25) descriptive calibration of the sheet's 'game luck' metric:
does early-season 3P luck (same definition as nba_tonight_sheet.py) predict actual total - closing total?
Not a ledger trial; informs how much weight the sheet's luck column deserves."""
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
sys.path.insert(0, '/Users/jonathanwang/Desktop/vocalizations_lab/.claude/worktrees/quant-trader/sports/scripts')
import nba_tonight_sheet as S
E = S.ESPN_DIR
K3 = 750.0
out = []
for season in (2023, 2024, 2025):
    gp, bp = pd.read_parquet(E / f"games_{season-1}.parquet"), pd.read_parquet(E / f"box_{season-1}.parquet")
    g, b = pd.read_parquet(E / f"games_{season}.parquet"), pd.read_parquet(E / f"box_{season}.parquet")
    P = S.team_table(S.team_games(gp, bp, pd.Timestamp("2100-01-01", tz="UTC")))
    lg = P.m3.sum() / P.a3.sum()
    t = S.team_games(g, b, pd.Timestamp("2100-01-01", tz="UTC")).sort_values("start_utc")
    cols = ["threePointFieldGoalsMade", "threePointFieldGoalsAttempted", "o_threePointFieldGoalsMade", "o_threePointFieldGoalsAttempted"]
    t = t.dropna(subset=cols)
    for c in cols:   # to-date BEFORE this game
        t["cum_" + c] = t.groupby("team")[c].cumsum() - t[c]
    t["gp"] = t.groupby("team").cumcount()
    t["luck_off"] = np.nan
    pr = t.team.map(P.p3).fillna(lg)
    m3, a3, om3, oa3 = (t["cum_" + c] for c in cols)
    gpn = t.gp.replace(0, np.nan)
    s_off = (m3 + K3 * pr) / (a3 + K3)
    s_def = (om3 + K3 * lg) / (oa3 + K3)
    t["luck_off"] = (m3 / a3 - s_off) * a3 / gpn * 3
    t["luck_def"] = (om3 / oa3 - s_def) * oa3 / gpn * 3
    gg = t.groupby("event_id").agg(luck=("luck_off", "sum"), luckd=("luck_def", "sum"), gpmin=("gp", "min"),
                                   total=("pts", "sum"), n=("team", "size"))
    gg = gg[gg.n == 2]
    gg["game_luck"] = 0.5 * (gg.luck + gg.luckd)
    L = pd.read_parquet(E / f"lines_{season}.parquet")
    L["event_id"] = L.event_id.astype(str)
    L["close"] = L.total_close.fillna(L.total)
    L["open"] = L.total_open
    pref = L[L.provider.isin(["DraftKings", "ESPN BET"])].sort_values("provider", ascending=False)  # ESPN BET first? keep one
    cl = L.groupby("event_id").agg(close=("close", "median"), open=("open", "median"))
    gg = gg.join(cl, how="inner")
    gg["season"] = season
    out.append(gg)
d = pd.concat(out).dropna(subset=["game_luck", "close"])
d["res"] = d.total - d.close
d["mv"] = d.close - d.open
print("n games", len(d))
x = d[(d.gpmin >= 1) & (d.gpmin <= 5)]
for thr in (-4, -6, -8):
    y = x[x.game_luck <= thr]
    print(f"EARLY cold <= {thr}: n {len(y)} mean res {y.res.mean():+.2f} over% {(y.res > 0).mean():.3f} by season "
          + str(y.groupby('season').res.agg(['mean', 'size']).round(2).to_dict('index')))
print("EARLY corr(luck, close - league mean close)", round(x.game_luck.corr(x.close - x.close.mean()), 3))
for lo, hi in ((1, 5), (6, 15), (16, 90), (1, 90)):
    x = d[(d.gpmin >= lo) & (d.gpmin <= hi)]
    m = sm.OLS(x.res, sm.add_constant(x.game_luck)).fit(cov_type="HC1")
    m2 = sm.OLS(x.mv.dropna(), sm.add_constant(x.loc[x.mv.notna(), "game_luck"])).fit(cov_type="HC1")
    print(f"gpmin {lo}-{hi}: n {len(x)} sd(luck) {x.game_luck.std():.2f}  res~luck slope {m.params.game_luck:+.3f} (se {m.bse.game_luck:.3f}, p {m.pvalues.game_luck:.3f})"
          f"  | open->close move ~luck slope {m2.params.game_luck:+.3f} (se {m2.bse.game_luck:.3f}) n {int(m2.nobs)}")
    mid = x[(x.game_luck > -4) & (x.game_luck < 4)]
    print(f"    baseline mean res {x.res.mean():+.2f}; mid-luck n {len(mid)} mean {mid.res.mean():+.2f}; by season "
          + str(x.groupby('season').res.mean().round(2).to_dict()))
    # bucket: |luck| >= 4
    for side, msk in (("luck<=-4 (cold)", x.game_luck <= -4), ("luck>=+4 (hot)", x.game_luck >= 4)):
        y = x[msk]
        if len(y):
            print(f"    {side}: n {len(y)} mean res {y.res.mean():+.2f} (se {y.res.std()/np.sqrt(len(y)):.2f}), over% {(y.res > 0).mean():.3f}")
