"""NBA hype (preseason win total) vs form, early season. Pre-registration: notes/nba_hype_vs_form.md (read it first).

    ../.venv_quant/bin/python scripts/nba_hype_vs_form.py [--ledger]

One run: features (ET date < game date only), thresholds frozen from DEV prices/form/totals with NO outcomes, then
outcomes joined, DEV scored, SEALED (2025-26) scored only for DEV passes. No orders, no network.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import edge_math as em  # noqa: E402
import nba_early_season_bias as nesb  # noqa: E402  (reused loaders: games, team_long, form_asof, lookahead_test)

ESPN = ROOT / "data" / "espn" / "nba"
WT = ROOT / "data" / "preseason" / "nba_win_totals_sportsoddshistory.csv"
OUT = ROOT / "results" / "nba_hype_vs_form"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DEV = (2022, 2023, 2024, 2025)
HALVES = ((2022, 2023), (2024, 2025))
SEALED = 2026
K_TRIALS = 4
BAR = 0.05 / K_TRIALS
NB = 4000
SEED = 2026
GP_LO, GP_HI = 3, 19
KSH = 20.0
POPULAR = {"LAL", "GS", "BOS", "NY", "CHI", "PHI"}
F1_X, F1_H, F1_GP = 5.3659, 2.06, (3, 8)


# ------------------------------------------------------------------------------------------------ data
def wt_map():
    w = pd.read_csv(WT)
    return {(int(r.season), r.team): float(r.win_total) for r in w.itertuples()}


def units_map(season):
    """OLS full-season MOV ~ (win pct - .5) over all team-seasons in ESPN seasons < season (from 2020)."""
    xs, ys = [], []
    for s in range(2020, season):
        g = nesb.games(s)
        tl = nesb.team_long(g)
        t = tl.groupby("team").agg(mov=("m", "mean"), wp=("m", lambda v: (v > 0).mean()), n=("m", "size"))
        t = t[t.n >= 40]  # drops All-Star / exhibition entities
        xs += list(t.wp - 0.5); ys += list(t.mov)
    b, a = np.polyfit(xs, ys, 1)
    return float(a), float(b), len(xs)


def lines_earliest(season):
    ln = pd.read_parquet(ESPN / f"lines_{season}.parquet")
    ln["event_id"] = ln.event_id.astype(str)
    close = nesb.lines(season).rename(columns={"home_ml": "c_hml", "away_ml": "c_aml", "spread": "c_spread"})
    close = close[["event_id", "provider", "c_hml", "c_aml", "c_spread"]]
    if ln.home_ml_open.notna().any():
        o = ln[ln.provider.isin(["DraftKings", "Draft Kings", "ESPN BET"])].dropna(
            subset=["home_ml_open", "away_ml_open", "home_spread_open"]).copy()
        o["pri"] = (o.provider == "ESPN BET").astype(int)
        o = o.sort_values("pri").drop_duplicates("event_id")
        o = o[["event_id", "provider", "home_ml_open", "away_ml_open", "home_spread_open"]].rename(
            columns={"provider": "o_provider"})
        x = close.merge(o, on="event_id", how="outer")
    else:
        x = close.copy()
        x["o_provider"] = None
        x["home_ml_open"] = x["away_ml_open"] = x["home_spread_open"] = np.nan
    has_o = x.home_ml_open.notna()
    x["price"] = np.where(has_o, "open", "close")
    x["book"] = np.where(has_o, x.o_provider, x.provider)
    x["hml"] = np.where(has_o, x.home_ml_open, x.c_hml)
    x["aml"] = np.where(has_o, x.away_ml_open, x.c_aml)
    x["M"] = -np.where(has_o, x.home_spread_open, x.c_spread)
    x["M_close"] = -x.c_spread
    return x[["event_id", "price", "book", "hml", "aml", "M", "M_close", "c_hml", "c_aml"]]


def build(season, WTM):
    g = nesb.games(season)
    tl = nesb.team_long(g)
    a, b, nfit = units_map(season)
    rows = []
    for d, gd in g.groupby("date"):
        f = nesb.form_asof(tl, d)
        for r in gd.itertuples():
            row = {"event_id": r.event_id, "date": d, "home": r.home, "away": r.away}
            for side, t in (("h", r.home), ("a", r.away)):
                row[f"gp_{side}"] = int(f.loc[t].gp) if t in f.index else 0
                row[f"adj_{side}"] = f.loc[t].adj if t in f.index else np.nan
                wt = WTM.get((season, t), np.nan)
                row[f"wt_{side}"] = wt
                row[f"E_{side}"] = a + b * (wt / 82 - 0.5)
            rows.append(row)
    x = pd.DataFrame(rows)
    x["season"] = season
    x = x.merge(lines_earliest(season), on="event_id", how="left")
    meta = {"units_a": a, "units_b": b, "units_n": nfit}
    return x, g, meta


def features(x):
    x = x.copy()
    for s in ("h", "a"):
        x[f"gap_{s}"] = x[f"adj_{s}"] - x[f"E_{s}"]
        w = x[f"gp_{s}"] / (x[f"gp_{s}"] + KSH)
        x[f"r_{s}"] = x[f"E_{s}"] + w * x[f"gap_{s}"]
    x["D"] = x.gap_h - x.gap_a
    Hs = x.groupby("season").M.mean()
    x["Hs"] = x.season.map(Hs)
    x["P"] = x.Hs + x.r_h - x.r_a
    x["Z"] = x.P - x.M
    x["Z0"] = x.Hs + x.adj_h - x.adj_a - x.M  # form-only twin (K = 0)
    f1 = (x.gp_h.between(*F1_GP)) & (x.gp_a.between(*F1_GP))
    R = (x.adj_h - x.adj_a) - (x.M - F1_H)
    x["F1_side"] = np.where(f1 & (R >= F1_X), "home", np.where(f1 & (R <= -F1_X), "away", ""))
    return x


def window(x):
    w = x[x.gp_h.between(GP_LO, GP_HI) & x.gp_a.between(GP_LO, GP_HI)]
    return w.dropna(subset=["adj_h", "adj_a", "E_h", "E_a", "M", "hml", "aml"]).copy()


# ------------------------------------------------------------------------------------------------ scoring
def bet_ml(b, home):
    b = b.copy()
    b["side"] = np.where(home, "home", "away")
    b["odds"] = np.where(home, b.hml, b.aml)
    b["win"] = np.where(home, b.homewin, ~b.homewin.astype(bool)).astype(bool)
    dec = np.array([em.american_to_decimal(o) for o in b.odds])
    b["profit"] = np.where(b.win, dec - 1, -1.0)
    return b


def boot_clustered(v, clus, stat="mean", xv=None):
    rng = np.random.default_rng(SEED)
    clus = np.asarray(clus)
    u, inv = np.unique(clus, return_inverse=True)
    groups = [np.where(inv == i)[0] for i in range(len(u))]
    out = np.empty(NB)
    for k in range(NB):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        if stat == "mean":
            out[k] = v[idx].mean()
        else:
            out[k] = np.polyfit(xv[idx], v[idx], 1)[0]
    return out


def summarize(b, halves=HALVES):
    if len(b) == 0:
        return {"bets": 0, "passed": False}
    v = b.profit.to_numpy()
    bs = boot_clustered(v, b.date.astype(str))
    out = {"bets": int(len(v)), "dates": int(b.date.nunique()), "units": float(v.sum()), "roi": float(v.mean()),
           "ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "p": float(np.mean(bs <= 0)),
           "boot_mean": float(bs.mean()), "win_rate": float(b.win.mean()), "sd": float(v.std(ddof=1)),
           "price_mix": b.price.value_counts().to_dict(),
           "by_season": {int(s): {"n": int(len(y)), "roi": float(y.profit.mean())} for s, y in b.groupby("season")}}
    out["halves"] = [float(b[b.season.isin(h)].profit.mean()) if b.season.isin(h).any() else None for h in halves]
    return out


def slope_summary(w, halves=HALVES):
    y = (w.margin - w.M).to_numpy(float)
    xv = w.D.to_numpy(float)
    sl = float(np.polyfit(xv, y, 1)[0])
    bs = boot_clustered(y, w.date.astype(str), "slope", xv)
    hs = [float(np.polyfit(w[w.season.isin(h)].D, (w.margin - w.M)[w.season.isin(h)], 1)[0])
          if w.season.isin(h).any() else None for h in halves]
    return {"n": int(len(w)), "dates": int(w.date.nunique()), "slope": sl,
            "ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "p": float(np.mean(bs <= 0)),
            "boot_mean": float(bs.mean()), "halves": hs, "D_sd": float(xv.std())}


def passed(s, bar):
    key = "slope" if "slope" in s else "roi"
    return bool(s.get("bets", s.get("n", 0)) > 0 and s["p"] < bar and all(h is not None and h > 0 for h in s["halves"]))


def trials(w, X, G):
    res, bets = {}, {}
    b1 = bet_ml(w[w.Z.abs() >= X], (w.Z >= X)[w.Z.abs() >= X])
    bets["H1"] = b1; res["H1"] = summarize(b1)
    res["H2"] = slope_summary(w)
    faded_gap = np.where(b1.side == "home", b1.gap_a, b1.gap_h)
    b3 = b1[faded_gap <= -G]
    bets["H3"] = b3; res["H3"] = summarize(b3)
    ph, pa = w.home.isin(POPULAR), w.away.isin(POPULAR)
    w4 = w[ph ^ pa]
    b4 = bet_ml(w4, w4.away.isin(POPULAR))  # bet home when the away team is the popular one
    bets["H4"] = b4; res["H4"] = summarize(b4)
    return res, bets


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", action="store_true")
    ap.add_argument("--build-only", action="store_true", help="stop before outcomes are joined")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    WTM = wt_map()
    xs, checks = [], {}
    for s in DEV + (SEALED,):
        x, g, meta = build(s, WTM)
        x["season"] = s
        checks[s] = {**meta, "lookahead_mismatches": nesb.lookahead_test(s, x.rename(columns={}), g),
                     "games": int(len(g)), "wt_missing_teams": int(x[x.wt_h.isna()].home.nunique()),
                     "price_mix": x.price.value_counts().to_dict(), "books": x.book.value_counts().to_dict()}
        y = x.dropna(subset=["M", "hml"])
        y = y[(y.M != 0) & (y.hml != y.aml)]
        checks[s]["spread_ml_sign_agree"] = float(((y.M > 0) == (y.hml < y.aml)).mean())
        xs.append(x)
    print("checks", json.dumps(checks, indent=1, default=str))
    assert all(c["lookahead_mismatches"] == 0 for c in checks.values()), "LOOKAHEAD TEST FAILED"
    x = features(pd.concat(xs, ignore_index=True))
    dev_w = window(x[x.season.isin(DEV)])
    X = float(np.quantile(dev_w.Z.abs(), 2 / 3))
    X0 = float(np.quantile(dev_w.Z0.abs(), 2 / 3))
    G = float(np.median(np.abs(np.concatenate([dev_w.gap_h, dev_w.gap_a]))))
    n1 = int((dev_w.Z.abs() >= X).sum())
    mde = (2.24 + 0.84) * 1.33 / np.sqrt(n1)
    print(f"FROZEN (no outcomes): X={X:.3f} pts  X0(form-only)={X0:.3f}  G={G:.3f}  DEV window games {len(dev_w)}; "
          f"H1 bets {n1}; MDE (80% power, one-sided .0125) ROI ~ {mde:+.1%}")

    if a.build_only:
        print(dev_w.groupby("season").agg(n=("Z", "size"), H=("Hs", "first"), Zsd=("Z", "std"), Dsd=("D", "std"),
                                          bets=("Z", lambda z: int((z.abs() >= X).sum()))))
        print(dev_w[["season", "date", "home", "away", "gp_h", "gp_a", "wt_h", "E_h", "adj_h", "gap_h", "r_h",
                     "M", "P", "Z", "price", "book", "hml", "aml"]].sample(6, random_state=0).to_string())
        return 0
    # ---- outcomes joined here
    oc = pd.concat([nesb.games(s)[["event_id", "margin"]] for s in DEV + (SEALED,)])
    x = x.merge(oc, on="event_id", how="left")
    x = x[x.margin.notna() & (x.margin != 0)].copy()
    x["homewin"] = x.margin > 0
    dw = window(x[x.season.isin(DEV)])
    res, bets = trials(dw, X, G)
    for k in res:
        res[k]["passed"] = passed(res[k], BAR)
    # descriptive extras
    b1 = bets["H1"]
    ov = b1[b1.F1_side == b1.side]
    f1_rows = dw[dw.F1_side != ""]
    extra = {"F1_overlap": {"H1_bets": int(len(b1)), "also_F1_same_side": int(len(ov)),
                            "F1_bets_in_window": int(len(f1_rows)),
                            "F1_opposite_side": int(((b1.F1_side != "") & (b1.F1_side != b1.side)).sum()),
                            "H1_not_F1_roi": float(b1[b1.F1_side != b1.side].profit.mean())}}
    b0 = bet_ml(dw[dw.Z0.abs() >= X0], (dw.Z0 >= X0)[dw.Z0.abs() >= X0])
    extra["form_only_twin"] = summarize(b0)
    extra["H1_vs_twin_shared"] = int(len(set(b1.event_id + b1.side) & set(b0.event_id + b0.side)))
    print(json.dumps({"dev": res, "extra": extra}, indent=1, default=str))
    for k, b in bets.items():
        b.to_csv(OUT / f"bets_dev_{k}.csv", index=False)
    dw.to_csv(OUT / "dev_window.csv", index=False)

    pas = [k for k in res if res[k]["passed"]]
    sealed = {}
    if pas:
        sw = window(x[x.season == SEALED])
        sres, sbets = trials(sw, X, G)
        sbar = 0.05 / len(pas)
        for k in pas:
            s = sres[k]
            key = "slope" if k == "H2" else "roi"
            s["passed"] = bool(s.get("bets", s.get("n", 0)) > 0 and s["p"] < sbar and s[key] > 0)
            sealed[k] = s
            if k in sbets:
                sbets[k].to_csv(OUT / f"bets_sealed_{k}.csv", index=False)
        print("SEALED", json.dumps(sealed, indent=1, default=str))
    else:
        print("SEALED not scored: no DEV pass")
    (OUT / "score.json").write_text(json.dumps({"X": X, "X0": X0, "G": G, "checks": checks, "dev": res,
                                                "extra": extra, "sealed": sealed}, indent=1, default=str))
    if a.ledger:
        sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        names = {"H1": "nhvf_H1_shrinkage_vs_line_ml", "H2": "nhvf_H2_resid_on_gap_slope",
                 "H3": "nhvf_H3_fade_overhyped_ml", "H4": "nhvf_H4_popular_fade_ml"}
        with LEDGER.open("a") as fh:
            for i, k in enumerate(("H1", "H2", "H3", "H4"), 1):
                r = {"ts": ts, "strategy": names[k], "sha": sha, "dataset": "nba_hype_vs_form",
                     "hypothesis": "pre-registered in notes/nba_hype_vs_form.md; early-season NBA lines anchored to "
                                   "preseason win totals (games 4-20)", "source": "user hypothesis 10-30",
                     "status": "ok", "split": "DEV 2021-22..2024-25", "trial_number": i, "study_trials": K_TRIALS,
                     "bonferroni_bar": BAR, "price": "earliest (open 23-24/24-25, close 21-22/22-23)", **res[k]}
                r["verdict"] = "PASS_DEV" if res[k]["passed"] else "FAIL"
                fh.write(json.dumps(r, default=str) + "\n")
            for k, s in sealed.items():
                r = {"ts": ts, "strategy": names[k] + "_sealed", "sha": sha, "dataset": "nba_hype_vs_form",
                     "status": "ok", "split": "SEALED 2025-26 games 4-20", "study_trials": len(pas),
                     "bonferroni_bar": 0.05 / len(pas), **s, "verdict": "PASS_SEALED" if s["passed"] else "FAIL_SEALED"}
                fh.write(json.dumps(r, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
