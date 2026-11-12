"""game_release_dose: does the post-launch totals residual scale with launch hype? (notes/game_release_dose.md)

    ../.venv_quant/bin/python -I scripts/game_release_dose.py build     # dev frame (labels only, no outcome stats)
    ../.venv_quant/bin/python -I scripts/game_release_dose.py power     # slope / effect_VI SE from counts (iid, SD 1)
    ../.venv_quant/bin/python -I scripts/game_release_dose.py score     # DEV: b0, b1, effect_VI with cluster bootstrap
    ../.venv_quant/bin/python -I scripts/game_release_dose.py sealed    # SEALED PREP: builds the edgelab frame and prints
                                                                        #   COUNTS ONLY (no outcome is aggregated/printed)
Dev games come from results/game_release_slump/all_games.parquet (prior agent, dev seasons only), cut at 2022-12-31.
Sealed games (2023-01-01..2026-10-31) come from ESPN games + lines (real over/under prices).
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path.home() / "agent-harness" / "lib")); sys.path.insert(0, str(ROOT / "scripts"))
import traps  # noqa: E402

PRIOR = ROOT / "results" / "game_release_slump" / "all_games.parquet"
OUT = ROOT / "results" / "game_release_dose"
DEV_END = pd.Timestamp("2022-12-31")
SEALED = (pd.Timestamp("2023-01-01"), pd.Timestamp("2026-10-31"))
H_MIN, H_BET = 250e6, 600e6           # inclusion; sealed-trial threshold (both fixed in the pre-registration)
H_REF, H_VI = 1e9, 3e9                # centring (GTA V 3-day) and the GTA VI extrapolation assumption
SD = {"NFL": 13.43, "NBA": 17.80, "NHL": 2.24, "CFB": 16.49}  # prior agent, all dev games (direction-free)

# (title, US release, H in USD within <= 7 days, basis) -- FROZEN before outcomes; sources in notes/game_release_dose.md
LAUNCHES: list[tuple[str, str, float, str]] = []
from game_release_dose_hype import LAUNCHES  # noqa: E402,F811  (separate file so the table is diffable / frozen)


def clusters(launches=None) -> pd.DataFrame:
    L = pd.DataFrame(launches or LAUNCHES, columns=["title", "date", "H", "basis"])
    L["date"] = pd.to_datetime(L["date"])
    L = L[L.H >= H_MIN].sort_values("date").reset_index(drop=True)
    cid, start, out = -1, None, []
    for _, r in L.iterrows():
        if start is None or (r.date - start).days > 6:
            cid += 1; start = r.date
        out.append(cid)
    L["cid"] = out
    C = L.groupby("cid").agg(start=("date", "min"), H=("H", "sum"), titles=("title", " + ".join)).reset_index()
    C["cluster"] = C.titles
    return C


def label(g: pd.DataFrame, C: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, c in C.iterrows():
        k = (g["date"] - c.start).dt.days
        m = k.between(-21, 34)
        s = g[m].copy(); s["cluster"], s["k"], s["H"] = c.cluster, k[m], c.H
        rows.append(s)
    x = pd.concat(rows, ignore_index=True)
    x["W"] = x.k.between(0, 6)
    x["near"] = x.k.between(0, 13)
    x["ctrl"] = x.k.between(-21, -8) | x.k.between(21, 34)
    busy = set(x.loc[x.near, "game_id"])                       # inside ANY cluster's days 0..13 -> never control
    x = x[x.W | (x.ctrl & ~x.game_id.isin(busy))]
    x = x.sort_values(["game_id", "k"])
    x = pd.concat([x[x.W].drop_duplicates("game_id", keep="first"),  # clusters start > 6 days apart: a game is in <= 1 window
                   x[~x.W]], ignore_index=True)
    x = x[~(~x.W & x.duplicated("game_id", keep=False))]          # control game shared by two clusters -> drop
    x["season_key"] = x.league + "|" + x.season.astype(str)
    # control must be same league-season as the cluster's window games
    ws = x[x.W].groupby("cluster").season_key.unique().apply(set).to_dict()
    x = x[x.W | x.apply(lambda r: r.season_key in ws.get(r.cluster, set()), axis=1)]
    return x.reset_index(drop=True)


def dev_games() -> pd.DataFrame:
    g = pd.read_parquet(PRIOR)
    g = g[g.date <= DEV_END].copy()
    g["date"] = pd.to_datetime(g.date).astype("datetime64[ns]")
    return g


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    C = clusters()
    C_dev = C[C.start <= DEV_END]
    x = label(dev_games(), C_dev)
    x.to_parquet(OUT / "dev_frame.parquet", index=False)
    C.to_csv(OUT / "clusters.csv", index=False)
    t = x.groupby(["cluster", "league"]).agg(win=("W", "sum"), ctrl=("W", lambda s: int((~s).sum()))).unstack(fill_value=0)
    print(C.to_string()); print(t.to_string())
    print(x.groupby("league").W.agg(win="sum", n="size"))


def design(x):
    """Demean within cluster x league (absorbs alpha); regressors W and W*(lnH - lnH_ref)."""
    x = x.copy()
    x["z"] = (x.final - x.line) / x.league.map(SD)
    x["xw"] = x.W * (np.log(x.H) - np.log(H_REF))
    x["g"] = x.cluster + "|" + x.league
    for c in ("z", "W", "xw"):
        x[c + "_d"] = x[c] - x.groupby("g")[c].transform("mean")
    return x


def ols(x):
    X = x[["W_d", "xw_d"]].to_numpy(float); y = x["z_d"].to_numpy(float)
    XtX = X.T @ X
    b = np.linalg.solve(XtX, X.T @ y)
    return b, X, y, XtX


def cr1(x, b, X, y, XtX):
    e = y - X @ b
    G = x.cluster.to_numpy(); us = np.unique(G)
    meat = np.zeros((2, 2))
    for u in us:
        m = G == u
        s = X[m].T @ e[m]; meat += np.outer(s, s)
    n, k, gN = len(y), 2 + x.g.nunique(), len(us)
    adj = gN / (gN - 1) * (n - 1) / (n - k)
    inv = np.linalg.inv(XtX)
    return adj * inv @ meat @ inv, gN


def power():
    x = design(pd.read_parquet(OUT / "dev_frame.parquet"))
    X = x[["W_d", "xw_d"]].to_numpy(float)
    V = np.linalg.inv(X.T @ X)                     # iid, unit variance (z is in SD units)
    a = np.array([1.0, np.log(H_VI / H_REF)])
    se1, seVI = np.sqrt(V[1, 1]), np.sqrt(a @ V @ a)
    print(f"rows {len(x)} window {int(x.W.sum())} clusters {x.cluster.nunique()} lnH range "
          f"{np.log(x.H.min()):.2f}..{np.log(x.H.max()):.2f} (ln $1B = {np.log(1e9):.2f}, ln H_VI {np.log(H_VI):.2f})")
    print(f"iid SE b1 {se1:.4f} -> MDE {2.8 * se1:.3f} SD per log-unit; SE effect_VI {seVI:.4f} -> MDE {2.8 * seVI:.3f} SD")
    print("(cluster-level heterogeneity widens both; prior agent's 14-cluster bootstrap CI half-width was ~0.055 SD)")


def score(B=4000, seed=0):
    x = design(pd.read_parquet(OUT / "dev_frame.parquet"))
    a = np.array([1.0, np.log(H_VI / H_REF)])
    b, X, y, XtX = ols(x)
    V, G = cr1(x, b, X, y, XtX)
    from scipy import stats
    tq = stats.t.ppf(0.975, G - 1)
    rng = np.random.default_rng(seed)
    cl = x.cluster.unique(); parts = {c: x[x.cluster == c] for c in cl}
    bs = []
    for _ in range(B):
        pick = rng.choice(cl, len(cl))
        q = pd.concat([parts[c].assign(cluster=f"{c}#{i}") for i, c in enumerate(pick)], ignore_index=True)
        q = design(q.drop(columns=[c for c in q.columns if c.endswith("_d")] + ["z", "xw", "g"]))
        try:
            bs.append(ols(q)[0])
        except np.linalg.LinAlgError:
            continue
    bs = np.array(bs); vi = bs @ a
    effVI = a @ b; seVI = np.sqrt(a @ V @ a)
    L = []
    L.append(f"rows {len(x)} (window {int(x.W.sum())}) clusters {G} leagues {x.league.value_counts().to_dict()}")
    L.append(f"b0 (window effect at $1B) {b[0]:+.4f} SD  boot [{np.percentile(bs[:, 0], 2.5):+.4f},{np.percentile(bs[:, 0], 97.5):+.4f}]"
             f"  CR1 [{b[0] - tq * np.sqrt(V[0, 0]):+.4f},{b[0] + tq * np.sqrt(V[0, 0]):+.4f}]")
    L.append(f"b1 (per ln-unit of H) {b[1]:+.4f} SD  boot [{np.percentile(bs[:, 1], 2.5):+.4f},{np.percentile(bs[:, 1], 97.5):+.4f}]"
             f"  CR1 [{b[1] - tq * np.sqrt(V[1, 1]):+.4f},{b[1] + tq * np.sqrt(V[1, 1]):+.4f}]  boot P(b1<0) {np.mean(bs[:, 1] < 0):.3f}")
    for hv in (2e9, 3e9, 5e9):
        aa = np.array([1.0, np.log(hv / H_REF)]); v = bs @ aa
        L.append(f"effect at H=${hv / 1e9:.0f}B (EXTRAPOLATION) {aa @ b:+.4f} SD  boot [{np.percentile(v, 2.5):+.4f},"
                 f"{np.percentile(v, 97.5):+.4f}]  CR1 [{aa @ b - tq * np.sqrt(aa @ V @ aa):+.4f},{aa @ b + tq * np.sqrt(aa @ V @ aa):+.4f}]")
    # descriptive per-cluster D (window - control, SD units) vs lnH
    per = x.groupby("cluster").apply(lambda q: pd.Series({
        "H_$M": q.H.iloc[0] / 1e6, "n_win": int(q.W.sum()),
        "D_sd": (q.loc[q.W, "z"].mean() - q.loc[~q.W, "z"].mean()) if q.W.any() and (~q.W).any() else np.nan}))
    per = per.sort_values("H_$M")
    L.append(per.round(3).to_string())
    succ = (np.percentile(bs[:, 1], 97.5) < 0) and (effVI <= -0.06)
    L.append(f"PRE-REG VERDICT: {'SUCCESS' if succ else 'KILL'} (b1 CI excludes 0 and < 0: "
             f"{np.percentile(bs[:, 1], 97.5) < 0}; effect_VI {effVI:+.4f} <= -0.06: {effVI <= -0.06})")
    L.append(f"GTA-VI-scale most-negative plausible effect (2.5th pct boot, $3B): {np.percentile(vi, 2.5):+.4f} SD")
    # secondary: all dev games, league x season FE + league x ISO-week FE (report only)
    g = dev_games(); C = clusters(); C = C[C.start <= DEV_END]
    g["z"] = (g.final - g.line) / g.league.map(SD)
    g["xw"] = 0.0; g["W"] = 0.0
    for _, c in C.iterrows():
        m = (g.date - c.start).dt.days.between(0, 6) & (g.W == 0)
        g.loc[m, "W"] = 1.0; g.loc[m, "xw"] = np.log(c.H / H_REF)
    g["fe1"] = g.league + "|" + g.season.astype(str); g["fe2"] = g.league + "|" + g.date.dt.isocalendar().week.astype(str)
    cols = ["z", "W", "xw"]
    d = g[cols].astype(float).copy()
    for _ in range(30):                              # alternating projections for two-way FE
        for fe in ("fe1", "fe2"):
            d = d - d.groupby(g[fe]).transform("mean")
    Xs = d[["W", "xw"]].to_numpy(); bs2 = np.linalg.lstsq(Xs, d.z.to_numpy(), rcond=None)[0]
    L.append(f"SECONDARY (all {len(g)} dev games, league x season + league x ISO-week FE): b0 {bs2[0]:+.4f} b1 {bs2[1]:+.4f}"
             f" effect_VI {bs2 @ a:+.4f} (point only)")
    txt = "\n".join(L); print(txt)
    (OUT / "score.txt").write_text(txt)
    np.save(OUT / "boot.npy", bs)


# ------------------------------------------------------------------------------------------------- sealed prep
PROV = ["ESPN BET", "DraftKings", "Draft Kings", "consensus"]


def espn_games(lo, hi) -> pd.DataFrame:
    fs = []
    for lg, name in (("nba", "NBA"), ("nfl", "NFL"), ("nhl", "NHL"), ("cfb", "CFB")):
        for f in sorted(glob.glob(str(ROOT / f"data/espn/{lg}/games_20*.parquet"))):
            if "full" in f:
                continue
            gm = pd.read_parquet(f)
            lf = f.replace("games_", "lines_")
            if not Path(lf).exists():
                continue
            ln = pd.read_parquet(lf, columns=["event_id", "provider", "total", "over_odds", "under_odds"])
            ln = ln.dropna(subset=["total", "under_odds"])
            ln["pr"] = ln.provider.map({p: i for i, p in enumerate(PROV)}).fillna(99)
            ln = ln.sort_values(["event_id", "pr"]).drop_duplicates("event_id")
            m = traps.checked_merge(gm, ln, on="event_id", how="inner", validate="one_to_one")
            m["league"] = name
            fs.append(m)
    g = pd.concat(fs, ignore_index=True)
    g = g[g.season_type.isin(["regular-season", "post-season", "play-in-season"])]
    g = g[g.status.astype(str).str.contains("FINAL|final|Final", regex=True)] if "status" in g else g
    st = pd.to_datetime(g.start_utc, utc=True)
    g["date"] = st.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize().astype("datetime64[ns]")
    g = g[(g.date >= lo) & (g.date <= hi)].copy()
    g["season"] = g.season
    g["line"] = g.total.astype(float)
    g["final"] = g.home_score.astype(float) + g.away_score.astype(float)
    am = g.under_odds.astype(float)
    g["dec_under"] = np.where(am > 0, 1 + am / 100, 1 + 100 / am.abs())
    g["price_real"] = True
    g["game_id"] = g.league + "|" + g.event_id.astype(str)
    return g[["date", "season", "home", "away", "line", "final", "dec_under", "game_id", "league", "price_real"]]


def edgelab_rows(x, era):
    x = x[x.W & (x.H >= H_BET)].copy()
    r = x.final - x.line
    return pd.DataFrame({"date": x.date, "game_id": x.game_id, "sel": "under", "league": x.league, "event": x.cluster,
                         "k": x.k, "H": x.H, "line": x.line, "dec_under": x.dec_under, "price_real": x.price_real,
                         "era": era, "won": (r < 0).astype(int), "push": (r == 0).astype(int), "final": x.final})


def sealed():
    """Builds results/game_release_dose/edgelab_frame.parquet (dev + sealed rows, H >= H_BET) for loader
    game_release_hype. Prints COUNTS ONLY; outcome columns are written (hidden by the vault) but never aggregated."""
    C = clusters()
    Cs = C[(C.start >= SEALED[0]) & (C.start <= SEALED[1])]
    gs = espn_games(SEALED[0] - pd.Timedelta(days=30), SEALED[1])
    xs = label(gs, Cs)
    xs = xs[xs.date >= SEALED[0]]
    xd = pd.read_parquet(OUT / "dev_frame.parquet")
    if "dec_under" not in xd:
        raise SystemExit("dev frame lacks dec_under")
    e = pd.concat([edgelab_rows(xd, "dev"), edgelab_rows(xs, "sealed")], ignore_index=True)
    e["date"] = pd.to_datetime(e.date)
    e.to_parquet(OUT / "edgelab_frame.parquet", index=False)
    print("sealed clusters (H >= $250M):"); print(Cs[["start", "H", "titles"]].to_string())
    print("\nbet rows (H >= $600M) by era x league:"); print(e.groupby(["era", "league"]).size().unstack(fill_value=0))
    print("\nsealed bet rows by cluster:"); print(e[e.era == "sealed"].groupby("event").size().to_string())
    n = int((e.era == "sealed").sum())
    se_win = np.sqrt(0.25 / max(n, 1))
    print(f"\nsealed n {n}: MDE under win-rate (80%, 2-sided .05, iid) {2.8 * se_win * 100:.1f} pp "
          f"= ROI ~{2.8 * se_win * 1.91 * 100:.1f}% at -110; break-even needs +2.4 pp")
    for eff_pp in (2.4, 4.0):
        z = eff_pp / 100 / se_win - 1.96
        from scipy import stats
        print(f"  power to detect +{eff_pp} pp under win-rate (iid): {stats.norm.cdf(z):.2f}")


if __name__ == "__main__":
    {"build": build, "power": power, "score": score, "sealed": sealed}[sys.argv[1]]()
