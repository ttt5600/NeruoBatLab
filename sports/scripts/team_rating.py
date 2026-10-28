"""NBA team rating: prior (last season, opponent-adjusted, + roster changes) blended with season-to-date. PRICING AID ONLY.

    ../.venv_quant/bin/python scripts/team_rating.py DEN [--date YYYY-MM-DD] [--no-refresh]
    rating DEN          (wrapper in /usr/local/bin)

Units: points of net margin per game vs an average team on a neutral floor.
  r_S        = season-S ridge rating: margin = h*home + r_home - r_away (unweighted, lambda 1, sum-to-zero by penalty).
  sigma      = game-noise SD (residual SD of that fit, dev seasons 2022-2025 = 2021-22..2024-25).
  b, tau     = r_S ~ b * r_{S-1} over S = 2023..2025; tau^2 = residual var - mean sampling var (sigma^2/games).
               NOTE tau is fit with a prior that has NO roster term, so it is an upper-ish bound for the live prior.
  PRIOR      = b * r_last + roster delta, roster delta = player-value strength (frozen BPM-lite in nba_simple_model,
               evaluated today) of the current full roster minus that of the team's last-10-games set of last season.
  SEASON     = mean over this season's games of (margin - h*home_sign + PRIOR_opp) = opponent-adjusted margin.
  3P LUCK    = shrunk 3P% as in nba_tonight_sheet.py (K=750 attempts; offence prior = own last-season 3P%, defence
               prior = league last-season 3P%); luck pts/g = (raw - shrunk) * 3PA/g * 3; adj = SEASON - off + def luck.
  BLEND      = w * SEASON_luckadj + (1 - w) * PRIOR, w = n / (n + (sigma/tau)^2).
  TONIGHT    = rating + availability adj (ESPN injury feed: strength(full minus OUT) - strength(full)), both teams,
               + h for the home team; win prob = Phi(spread / sigma). Kalshi KXNBAGAME mid shown alongside.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import norm

import nba_simple_model as nsm

ESPN = nsm.ESPN
ALIAS = {"NOP": "NO", "GSW": "GS", "NYK": "NY", "SAS": "SA", "UTA": "UTAH", "WAS": "WSH", "PHO": "PHX", "BRK": "BKN"}
K3 = 750.0


def games(season: int) -> pd.DataFrame:
    g = pd.read_parquet(ESPN / f"games_{season}.parquet")
    g = g[(g.season_type == "regular-season") & (g.status == "STATUS_FINAL")].copy()
    g["margin"] = g.home_score - g.away_score
    g["hi"] = (~g.neutral.astype(bool)).astype(float)
    return g


def ridge(g: pd.DataFrame, lam: float = 1.0):
    teams = sorted(set(g.home) | set(g.away))
    ix = {t: i for i, t in enumerate(teams)}
    n, k = len(g), len(teams)
    X = np.zeros((n, k + 1))
    X[np.arange(n), g.home.map(ix)] = 1
    X[np.arange(n), g.away.map(ix)] = -1
    X[:, k] = g.hi
    y = g.margin.to_numpy(float)
    beta = np.linalg.solve(X.T @ X + np.diag([lam] * k + [1e-6]), X.T @ y)
    res = y - X @ beta
    sig = float(np.sqrt((res ** 2).sum() / (n - k - 1)))
    gp = pd.concat([g.home, g.away]).value_counts()
    return pd.Series(beta[:k], index=teams), float(beta[k]), sig, gp


def fit_params(cur: int) -> dict:
    R, H, S, GP = {}, {}, {}, {}
    for s in range(2022, cur):
        R[s], H[s], S[s], GP[s] = ridge(games(s))
    sigma = float(np.mean([S[s] for s in range(2022, 2026)]))
    x, y, sv = [], [], []
    for s in range(2023, 2026):            # DEV only (2026 = 2025-26 sealed: live prior use only)
        c = R[s].index.intersection(R[s - 1].index)
        x += list(R[s - 1][c]); y += list(R[s][c]); sv += list(S[s] ** 2 / GP[s][c])
    x, y = np.array(x), np.array(y)
    b = float((x @ y) / (x @ x))
    tau = float(np.sqrt(max(np.var(y - b * x) - np.mean(sv), 1.0)))
    h = float(np.mean([H[s] for s in range(2022, 2026)]))
    return {"sigma": sigma, "tau": tau, "b": b, "h": h, "r_last": R[cur - 1], "h_last": H[cur - 1]}


def box_team(season: int, g: pd.DataFrame) -> pd.DataFrame:
    b = pd.read_parquet(ESPN / f"box_{season}.parquet", columns=["event_id", "team_id", "stat", "value"])
    b = b[b.stat.isin(["threePointFieldGoalsMade", "threePointFieldGoalsAttempted"])]
    tb = b.pivot_table(index=["event_id", "team_id"], columns="stat", values="value", aggfunc="sum").reset_index()
    tb.columns = ["event_id", "team_id", "a3", "m3"]
    tb["event_id"], tb["team_id"] = tb.event_id.astype(str), tb.team_id.astype(str)
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        x = pd.DataFrame({"event_id": g.event_id.astype(str), "team": g[side], "team_id": g[f"{side}_team_id"].astype(str),
                          "opp": g[opp], "opp_id": g[f"{opp}_team_id"].astype(str),
                          "margin": g.margin * (1 if side == "home" else -1), "hsign": g.hi * (1 if side == "home" else -1),
                          "start_utc": g.start_utc})
        rows.append(x)
    t = pd.concat(rows, ignore_index=True)
    t = t.merge(tb, on=["event_id", "team_id"], how="left")
    t = t.merge(tb.rename(columns={"team_id": "opp_id", "a3": "oa3", "m3": "om3"}), on=["event_id", "opp_id"], how="left")
    return t


def roster_deltas(d, cur, quiet=True):
    """team -> (P_now_full, P_lastseason, roster delta, avail adj, out names, today's game row or None, warnings)."""
    import nba_simple_live as nsl
    fz = json.loads(nsm.FROZEN.read_text())
    st = nsl.base_state(d, [cur - 1, cur], fz, quiet)
    strength = nsl.strength_fn(st["ST"], fz["bpm"], d.toordinal())
    g = pd.concat([pd.read_parquet(ESPN / f"games_{s}.parquet") for s in (cur - 1, cur)])
    tid = dict(zip(g.home, g.home_team_id.astype(str)))
    roster = nsl.rosters(sorted(set(tid.values())), quiet)
    inj, ok = nsl.injuries(quiet)
    out = {}
    for T, i in tid.items():
        hist = st["TH"].get(T, [])
        if not hist:
            continue
        ncur = st["n_cur"].get(T, 0)
        last = hist[:len(hist) - ncur][-10:]
        P_last = strength(set().union(*last))[0] if last else np.nan
        w = []
        full_pre, prev, full, sel, sel_q, avail, info = nsl._side_sets(T, i, d.toordinal(), st, roster, inj, set(), w)
        P_full = strength(full)[0]
        P_av = strength(avail)[0]
        out[T] = {"P_full": P_full, "P_last": P_last, "roster": P_full - P_last, "avail": P_av - P_full,
                  "out": info["out"], "q": info["questionable"], "warn": w}
    return out, st["today"], ok


def kalshi_mid(d, home, away):
    import nba_tonight_sheet as nts
    kg = nts.kalshi_series("KXNBAGAME")
    k = nts.kalshi_game(d, home, away, kg)
    f = lambda s: (k[s]["bid"] + k[s]["ask"]) / 2 if s in k and np.isfinite(k[s]["bid"]) and np.isfinite(k[s]["ask"]) else None
    return f("home"), f("away"), k


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("team")
    ap.add_argument("--date")
    ap.add_argument("--no-refresh", action="store_true")
    a = ap.parse_args(argv)
    T = ALIAS.get(a.team.upper(), a.team.upper())
    d = pd.Timestamp(a.date).date() if a.date else pd.Timestamp.now(tz="US/Eastern").date()
    cur = d.year + 1 if d.month >= 8 else d.year
    if not a.no_refresh:
        try:
            import nba_simple_live as nsl
            nsl.refresh_season(cur, d, True)
        except Exception as ex:
            print(f"(ESPN refresh failed: {str(ex)[:60]}; using disk)")
    P = fit_params(cur)
    sig, tau, b, h = P["sigma"], P["tau"], P["b"], P["h"]
    print(f"fitted (dev 2021-22..2024-25): game sigma {sig:.1f} pts, prior tau {tau:.1f} pts, carry-over b {b:.2f}, "
          f"home edge {h:.1f}  -> (sigma/tau)^2 = {(sig / tau) ** 2:.1f} games")
    rd, today, inj_ok = roster_deltas(d, cur)
    if T not in rd:
        print(f"unknown team {T}; known: {' '.join(sorted(rd))}")
        return 2
    prior = {t: b * float(P["r_last"].get(t, 0.0)) + (v["roster"] if np.isfinite(v["roster"]) else 0) for t, v in rd.items()}

    # season to date
    g = games(cur)
    g = g[pd.to_datetime(g.start_utc).dt.tz_convert("US/Eastern").dt.date < d]
    t = box_team(cur, g)
    x = t[t.team == T].copy()
    n = len(x)
    x["adj"] = x.margin - h * x.hsign + x.opp.map(prior).fillna(0)
    raw, opp_adj = (x.margin.mean() if n else np.nan), (x.adj.mean() if n else np.nan)
    # 3P luck
    prev = box_team(cur - 1, games(cur - 1))
    p3_prev = prev[prev.team == T].m3.sum() / prev[prev.team == T].a3.sum()
    lg_prev = prev.m3.sum() / prev.a3.sum()
    off = dfl = 0.0
    if n:
        m3, a3, om3, oa3 = x.m3.sum(), x.a3.sum(), x.om3.sum(), x.oa3.sum()
        off = (m3 / a3 - (m3 + K3 * p3_prev) / (a3 + K3)) * a3 / n * 3
        dfl = (om3 / oa3 - (om3 + K3 * lg_prev) / (oa3 + K3)) * oa3 / n * 3
    luck_adj = opp_adj - off + dfl if n else np.nan
    w = n / (n + (sig / tau) ** 2)
    blend = w * luck_adj + (1 - w) * prior[T] if n else prior[T]
    v = rd[T]
    print(f"\n{T} rating as of {d}  (pts/game vs average team, neutral floor)")
    print(f"  last season ridge          {P['r_last'].get(T, np.nan):+5.1f}   x b {b:.2f} = {b * P['r_last'].get(T, 0):+5.1f}")
    print(f"  roster change (player model) {v['roster']:+5.1f}   (strength now {v['P_full']:+.1f} vs last-season set {v['P_last']:+.1f})")
    print(f"  PRIOR                      {prior[T]:+5.1f}")
    print(f"  season raw margin          {raw:+5.1f}   over {n} games")
    print(f"  opponent-adjusted          {opp_adj:+5.1f}")
    print(f"  3P luck  off {off:+.1f}  def {dfl:+.1f}  -> luck-adjusted {luck_adj:+5.1f}")
    print(f"  weight on season  n/(n+{(sig / tau) ** 2:.1f}) = {w:.0%}")
    print(f"  BLENDED RATING             {blend:+5.1f}")
    print(f"  availability today         {v['avail']:+5.1f}   OUT: {', '.join(v['out']) or '-'}"
          f"{'   Q: ' + ', '.join(v['q']) if v['q'] else ''}{'' if inj_ok else '   (INJURY FEED DOWN)'}")
    gm = today[(today.home == T) | (today.away == T)] if len(today) else today
    if gm is None or gm.empty:
        print("  no game today")
        return 0
    r = gm.iloc[0]
    home, away = r.home, r.away
    opp = away if home == T else home

    def full(tm):
        xo = t[t.team == tm].copy()
        no = len(xo)
        if not no:
            return prior.get(tm, 0.0)
        xo["adj"] = xo.margin - h * xo.hsign + xo.opp.map(prior).fillna(0)
        p3p = prev[prev.team == tm].m3.sum() / max(prev[prev.team == tm].a3.sum(), 1)
        m3, a3, om3, oa3 = xo.m3.sum(), xo.a3.sum(), xo.om3.sum(), xo.oa3.sum()
        o_ = (m3 / a3 - (m3 + K3 * p3p) / (a3 + K3)) * a3 / no * 3
        d_ = (om3 / oa3 - (om3 + K3 * lg_prev) / (oa3 + K3)) * oa3 / no * 3
        wo = no / (no + (sig / tau) ** 2)
        return wo * (xo.adj.mean() - o_ + d_) + (1 - wo) * prior.get(tm, 0.0)

    r_opp = full(opp)
    ro = rd.get(opp, {"avail": 0.0, "out": []})
    hs = h * float(not bool(r.neutral)) * (1 if home == T else -1)
    spread = (blend + v["avail"]) - (r_opp + ro["avail"]) + hs
    wp = norm.cdf(spread / sig)
    print(f"\nTONIGHT {away} @ {home}: {opp} blended {r_opp:+.1f}, avail {ro['avail']:+.1f} (OUT: {', '.join(ro['out']) or '-'})")
    print(f"  {T} {'by' if spread >= 0 else 'gets'} {abs(spread):.1f}  ->  P({T} win) = {wp:.0%}")
    try:
        ph, pa, k = kalshi_mid(d, home, away)
        pk = ph if home == T else pa
        if pk is not None:
            s = k["home" if home == T else "away"]
            print(f"  Kalshi KXNBAGAME {T}: bid {s['bid']:.2f} / ask {s['ask']:.2f}  mid {pk:.0%}   (model - mid {wp - pk:+.0%})")
        else:
            print("  Kalshi: no market found")
    except Exception as ex:
        print(f"  Kalshi fetch failed: {str(ex)[:60]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
