import sys
ROOT = "/Users/jonathanwang/Desktop/vocalizations_lab/.claude/worktrees/quant-trader/sports"
sys.path.insert(0, ROOT + "/scripts"); sys.path.insert(0, ROOT)
import nba_referees as N, numpy as np, pandas as pd
from sports.espn import summary

l = pd.read_csv(N.L2M, low_memory=False)
l = l[l.decision.isin(["IC", "INC", "CC"])].copy()
l["benefit"] = np.where(l.decision == "INC", l.committing_side, l.disadvantaged_side)
l = l[l.committing_side.isin(["home", "away"])]
l["gid"] = l.gid.astype(str)
eg = pd.concat([pd.read_parquet(N.DATA / f"games_{s}.parquet") for s in (2022, 2023, 2024, 2025)])
eg = eg[eg.season_type != "preseason"].copy()
eg["date"] = eg.start_utc.dt.tz_convert("US/Eastern").dt.strftime("%Y-%m-%d")
eg["home3"] = eg.home.replace(N.ESPN2NBA)
eg = eg.drop_duplicates(["date", "home3"]).set_index(["date", "home3"]).event_id
l["home3"] = l.home.replace({"BRK": "BKN", "PHO": "PHX", "CHO": "CHA"})
lc = l[l.season.isin([2022, 2023, 2024, 2025])].copy()
lc["pn"] = lc.period.astype(str).str.extract(r"Q(\d)")[0].astype(float)
lc["t"] = lc.time.map(N._clock)
rows, cache = [], {}
for r in lc.itertuples():
    eid = eg.get((r.date, r.home3))
    if eid is None or np.isnan(r.pn) or np.isnan(r.t):
        rows.append((np.nan, np.nan)); continue
    if eid not in cache:
        cache[eid] = summary("nba", eid).get("plays") or []
    hs, as_ = N._score_before(cache[eid], int(r.pn), r.t)
    rows.append((hs if hs is not None else np.nan, as_ if as_ is not None else np.nan))
lc["hs"] = [float(a) for a, b in rows]; lc["as_"] = [float(b) for a, b in rows]
lc = lc.dropna(subset=["hs"]); lc["margin"] = (lc.hs - lc.as_).abs(); lc = lc[lc.margin > 0]
lc["trail"] = np.where(lc.hs < lc.as_, "home", "away"); lc["tb"] = lc.benefit == lc.trail
lc["committer_trailing"] = lc.committing_side == lc.trail
print("overall", round(lc.tb.mean(), 3), len(lc))
print("errors where the trailing team is the 'committing' side:", round(lc.committer_trailing.mean(), 3))
print(lc.groupby(["decision", "committer_trailing"]).tb.agg(["mean", "size"]).round(3))
print(lc.groupby(pd.cut(lc.margin, [0, 1, 2, 3, 5, 99])).tb.agg(["mean", "size"]).round(3))
print(lc.groupby(lc.pn > 4).tb.agg(["mean", "size"]).round(3))
print(lc.groupby("season").tb.agg(["mean", "size"]).round(3))
print(lc.groupby("trail").tb.agg(["mean", "size"]).round(3))
ct = lc.call_type.fillna("?")
top = ct.value_counts().head(12).index
print(lc[ct.isin(top)].groupby(ct).tb.agg(["mean", "size"]).sort_values("size", ascending=False).round(3))

lc.to_parquet("/tmp/l2m_lc.parquet")
for name, m in [("all", lc.t >= 0), ("Q4 >60s left", (lc.pn == 4) & (lc.t > 60)), ("excl. Personal/Take fouls", ~lc.call_type.fillna("").str.contains("Personal"))]:
    x = lc[m & lc.decision.isin(["IC", "CC"])]
    h = x.groupby("committer_trailing").decision.value_counts().unstack()
    h["wrong"] = h.IC / (h.IC + h.CC)
    y = lc[m & lc.decision.isin(["INC", "CC"])]
    g = y.groupby("committer_trailing").decision.value_counts().unstack()
    g["miss"] = g.INC / (g.INC + g.CC)
    print(name, "| wrong-whistle lead %.3f (n %d) trail %.3f (n %d)" % (h.wrong[False], h.IC[False]+h.CC[False], h.wrong[True], h.IC[True]+h.CC[True]),
          "| miss lead %.3f trail %.3f" % (g.miss[False], g.miss[True]))
# game-cluster bootstrap on wrong-whistle diff (all)
x = lc[lc.decision.isin(["IC", "CC"])].copy(); x["ic"] = (x.decision == "IC").astype(float)
agg = x.groupby(["gid", "committer_trailing"]).ic.agg(["sum", "size"]).unstack(fill_value=0)
rng = np.random.default_rng(1); n = len(agg); A = agg.values
d = []
for _ in range(5000):
    s_ = A[rng.integers(0, n, n)].sum(0)
    # columns: sum False, sum True, size False, size True
    d.append(s_[0] / s_[2] - s_[1] / s_[3])
print("wrong-whistle lead-minus-trail 95% CI", np.quantile(d, [.025, .975]).round(4), "share<=0", np.mean(np.array(d) <= 0))
