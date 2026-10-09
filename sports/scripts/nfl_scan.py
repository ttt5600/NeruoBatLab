"""Systematic scan of NFL situational factors against the CLOSING spread and total.

    python scripts/nfl_scan.py        # -> lab/reports/nfl_scan.csv

Every feature uses only games played BEFORE the one being priced (history from
1999 warms up the trailing stats). Tests run on dev seasons 2006-2019 only; the
2020-2025 lockbox is never touched. For each feature x and market:

  * residual test  : OLS slope of (outcome - closing line) on x, t and p
  * betting test   : a fixed, pre-declared rule (bet against the side the
                     feature says is overrated) at the real closing odds

p-values get a Benjamini-Hochberg correction over EVERY test in the scan, so
a feature that only "works" because 60 were tried does not survive.
A scan hit is a hypothesis, not a strategy: it still has to go through
lab/run.py (dev, with the trial count) and then the lockbox.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats

from sports import market, nfl

OUT = Path(__file__).resolve().parents[1] / "lab" / "reports" / "nfl_scan.csv"
DEV = range(2006, 2020)


def team_games(g: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game, from that team's point of view."""
    implied_home = (g["total_line"] + g["spread_line"]) / 2  # market's expected home points
    sides = []
    for side, opp, sign in (("home", "away", 1), ("away", "home", -1)):
        sides.append(pd.DataFrame({
            "game_id": g["game_id"], "season": g["season"], "week": g["week"], "date": g["gameday"],
            "team": g[f"{side}_team"], "opp": g[f"{opp}_team"], "is_home": side == "home",
            "qb": g[f"{side}_qb_id"], "coach": g[f"{side}_coach"], "rest": g[f"{side}_rest"],
            "pf": g[f"{side}_score"], "pa": g[f"{opp}_score"],
            "line": sign * g["spread_line"],  # >0 = this team favored by that many
            "imp_pf": implied_home if side == "home" else g["total_line"] - implied_home,
            "total_line": g["total_line"]}))
    t = pd.concat(sides, ignore_index=True).sort_values(["team", "date", "game_id"]).reset_index(drop=True)
    t["imp_pa"] = t["total_line"] - t["imp_pf"]
    t["ats"] = (t["pf"] - t["pa"]) - t["line"]        # cover margin vs closing spread
    t["off_res"] = t["pf"] - t["imp_pf"]              # scored vs market expectation
    t["def_res"] = t["pa"] - t["imp_pa"]              # allowed vs market expectation
    return t


def trailing(t: pd.DataFrame) -> pd.DataFrame:
    by = t.groupby("team", group_keys=False)
    f = pd.DataFrame(index=t.index)
    for k in (1, 3, 5):
        for c in ("ats", "off_res", "def_res", "pa", "pf"):
            f[f"{c}_l{k}"] = by[c].transform(lambda s: s.shift(1).rolling(k, min_periods=k).mean())
    # season-to-date ATS (prior games this season)
    f["ats_std"] = t.groupby(["team", "season"])["ats"].transform(lambda s: s.shift(1).expanding().mean())
    # QB: primary = most frequent starter over the team's previous 8 games
    def primary(s):
        out, hist = [], []
        for q in s:
            out.append(pd.Series(hist[-8:]).mode().iloc[0] if len(hist) >= 4 else None)
            hist.append(q)
        return pd.Series(out, index=s.index)
    prim = by["qb"].transform(primary)
    f["backup_qb"] = (prim.notna() & (t["qb"] != prim)).astype(float)
    f["qb_change"] = (by["qb"].shift(1).notna() & (t["qb"] != by["qb"].shift(1))).astype(float)
    f["new_coach"] = (t["coach"] != t.groupby("team")["coach"].shift(1)).astype(float).where(t["week"] == 1)
    f["new_coach"] = f.groupby([t["team"], t["season"]])["new_coach"].transform("max")
    f["rest"] = t["rest"]
    f["off_bye"] = (t["rest"] >= 13).astype(float)
    f["short_week"] = (t["rest"] <= 5).astype(float)
    f["prev_margin"] = by["pf"].shift(1) - by["pa"].shift(1)
    f["blowout_loss_prev"] = (f["prev_margin"] <= -21).astype(float)
    f["blowout_win_prev"] = (f["prev_margin"] >= 21).astype(float)
    return f


def game_level(g: pd.DataFrame, t: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    tf = pd.concat([t[["game_id", "is_home"]], f], axis=1)
    h = tf[tf.is_home].drop(columns="is_home").set_index("game_id").add_prefix("h_")
    a = tf[~tf.is_home].drop(columns="is_home").set_index("game_id").add_prefix("a_")
    d = g.set_index("game_id").join(h).join(a)
    feats = {}
    for c in f.columns:
        feats[f"diff_{c}"] = d[f"h_{c}"] - d[f"a_{c}"]   # spread: home minus away
        feats[f"sum_{c}"] = d[f"h_{c}"] + d[f"a_{c}"]    # total: both teams
    # referee: trailing over-rate (total residual) over the ref's prior games, min 16
    d = d.sort_values("gameday")
    tres = d["home_score"] + d["away_score"] - d["total_line"]
    feats["ref_total_res"] = tres.groupby(d["referee"]).transform(lambda s: s.shift(1).expanding(16).mean())
    feats["ref_home_ats"] = (d["result"] - d["spread_line"]).groupby(d["referee"]).transform(
        lambda s: s.shift(1).expanding(16).mean())
    # market-structure features
    feats["spread_line"] = d["spread_line"]
    feats["home_dog"] = (d["spread_line"] < 0).astype(float)
    feats["total_line"] = d["total_line"]
    feats["div_game"] = d["div_game"].astype(float)
    feats["week"] = d["week"].astype(float)
    feats["playoff"] = (d["game_type"] != "REG").astype(float)
    feats["outdoor_cold"] = ((d["roof"] == "outdoors") & (d["temp"] < 35)).astype(float)
    feats["wind15"] = ((d["roof"] == "outdoors") & (d["wind"] >= 15)).astype(float)
    X = pd.DataFrame(feats)
    return d, X


def bet_pnl(won, push, odds):
    dec = np.asarray(market.american_to_decimal(odds), float)
    return np.where(push, 0.0, np.where(won, dec - 1, -1.0))


def main() -> int:
    raw = nfl.raw_games()
    raw = raw[raw["home_score"].notna() & raw["spread_line"].notna() & raw["total_line"].notna()]
    t = team_games(raw)
    f = trailing(t)
    d, X = game_level(raw, t, f)
    dev = d["season"].isin(DEV).to_numpy()
    d, X = d[dev], X[dev]
    spread_res = (d["result"] - d["spread_line"]).to_numpy(float)          # >0 home covered
    total_res = (d["home_score"] + d["away_score"] - d["total_line"]).to_numpy(float)
    rows = []
    for c in X.columns:
        x = X[c].to_numpy(float)
        for mkt, y in (("spread", spread_res), ("total", total_res)):
            if mkt == "spread" and c.startswith("sum_"):
                continue
            if mkt == "total" and c.startswith("diff_") and not c.endswith(("_qb", "_change")):
                continue
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 200 or np.nanstd(x[ok]) == 0:
                continue
            r = stats.linregress(x[ok], y[ok])
            # rule: discrete (<=5 values) -> games with x != its mode; continuous -> top and
            # bottom quintiles. Each game is bet in the direction slope * (x - centre) points.
            xb = x[ok]
            vals = np.unique(xb)
            if len(vals) <= 5:
                centre = stats.mode(xb, keepdims=False).mode
                flag = xb != centre
            else:
                centre = np.median(xb)
                flag = ((xb <= np.quantile(xb, 0.2)) | (xb >= np.quantile(xb, 0.8))) & (xb != centre)
            sub = d[ok][flag]
            yy = y[ok][flag]
            up = (r.slope * (xb[flag] - centre)) > 0           # True = bet home / over
            won = np.where(up, yy > 0, yy < 0)
            if mkt == "spread":
                odds = np.where(up, sub["home_spread_odds"], sub["away_spread_odds"])
                side = "home" if r.slope > 0 else "away"
            else:
                odds = np.where(up, sub["over_odds"], sub["under_odds"])
                side = "over" if r.slope > 0 else "under"
            side += " when x high"
            pnl = bet_pnl(won, yy == 0, pd.Series(odds).fillna(-110))
            rows.append({"feature": c, "market": mkt, "n": int(ok.sum()), "slope": r.slope,
                         "t": r.slope / r.stderr, "p": r.pvalue, "rule_side": side, "rule_n": len(pnl),
                         "rule_units": pnl.sum(), "rule_roi": pnl.mean() if len(pnl) else np.nan})
    res = pd.DataFrame(rows)
    # Benjamini-Hochberg over all residual tests
    p = res["p"].to_numpy()
    order = np.argsort(p)
    q = np.empty_like(p)
    q[order] = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    res["q_bh"] = np.minimum(q, 1)
    res = res.sort_values("p")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    pd.set_option("display.width", 200)
    print(f"{len(res)} tests on {dev.sum()} dev games (2006-2019); BH q<0.10: {(res.q_bh < 0.10).sum()}")
    print(res.head(25).to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
