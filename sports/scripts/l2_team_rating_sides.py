"""L2: team_rating.py vs closing moneyline/spread on NBA 2025-26 (held out of the fit). Prereg: notes/l2_team_rating_sides.md

    ../.venv_quant/bin/python scripts/l2_team_rating_sides.py build     # point-in-time ratings only (no prices/outcomes)
    ../.venv_quant/bin/python scripts/l2_team_rating_sides.py parity    # compare to team_rating.main() on 2 dates
    ../.venv_quant/bin/python scripts/l2_team_rating_sides.py score [--ledger]

Ratings are rebuilt for every ET game date d from games dated < d with team_rating's own functions/formulas
(fit_params, games, box_team, nsl.strength_fn), no availability (no historical injury reports), roster = the script's
own no-roster-API fallback (last-10-games set).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from scipy.stats import norm

import edge_math as em
import nba_simple_model as nsm
import nba_simple_live as nsl
import team_rating as tr

CUR = 2026
OUT = ROOT / "results" / "l2_team_rating_sides"
RATINGS = OUT / "ratings_2026.parquet"
LEDGER = ROOT / "lab" / "ledger.jsonl"
K_TRIALS = 4
BAR = 0.05 / K_TRIALS
NB = 10000
SEED = 2026


# ------------------------------------------------------------------------------------------------ build (no prices)
def build() -> pd.DataFrame:
    P = tr.fit_params(CUR)
    sig, tau, b, h = P["sigma"], P["tau"], P["b"], P["h"]
    print(f"fit_params({CUR}): sigma {sig:.2f} tau {tau:.2f} b {b:.3f} h {h:.2f}  (seasons used: ridge 2022..{CUR - 1})")
    fz = json.loads(nsm.FROZEN.read_text())
    # player state, incrementally, exactly as nsl.player_state (date batches)
    gp = nsm.load_games([CUR - 1, CUR])
    box = nsm.load_box(gp, [CUR - 1, CUR])
    bx = {}
    for r in box.itertuples(index=False):
        bx.setdefault((r.event_id, r.team), []).append(r)
    ST, TH = {}, {}
    dn = np.array([d.toordinal() for d in gp.date])

    def absorb(i, d):
        r = gp.iloc[i]
        for side in ("home", "away"):
            T = r[side]
            ps = bx.get((r.event_id, T), [])
            for x in ps:
                v = np.array([getattr(x, c) for c in nsm.STATS], float)
                s = ST.get(x.athlete_id)
                if s is None:
                    s = {"s": np.zeros(len(nsm.STATS)), "m": 0.0, "t": d, "em": x.minutes}
                dec = 0.5 ** ((d - s["t"]) / nsm.HP)
                s["s"], s["m"], s["t"] = s["s"] * dec + v, s["m"] * dec + x.minutes, d
                s["em"] = 0.7 * s["em"] + 0.3 * x.minutes
                s["team"] = T
                ST[x.athlete_id] = s
            if ps:
                TH.setdefault(T, []).append({x.athlete_id for x in ps})

    g = tr.games(CUR)
    g["date"] = pd.to_datetime(g.start_utc).dt.tz_convert("US/Eastern").dt.date
    tall = tr.box_team(CUR, g)
    tall["date"] = pd.to_datetime(tall.start_utc).dt.tz_convert("US/Eastern").dt.date
    prev = tr.box_team(CUR - 1, tr.games(CUR - 1))
    lg_prev = prev.m3.sum() / prev.a3.sum()
    p3prev = {T: prev[prev.team == T].m3.sum() / max(prev[prev.team == T].a3.sum(), 1) for T in prev.team.unique()}
    rs_all = gp[(gp.season == CUR) & (gp.season_type == "regular-season")]
    done = 0
    order = np.argsort(dn, kind="stable")
    rows = []
    for d in sorted(g.date.unique()):
        do = d.toordinal()
        while done < len(order) and dn[order[done]] < do:     # absorb every game dated < d
            # date batches: all games of one date share the same decay date, same as player_state
            absorb(order[done], dn[order[done]])
            done += 1
        strength = nsl.strength_fn(ST, fz["bpm"], do)
        rs = rs_all[rs_all.date < d]
        n_cur = pd.concat([rs.home, rs.away]).value_counts().to_dict()
        prior, rdel = {}, {}
        for T in P["r_last"].index:
            hist = TH.get(T, [])
            if not hist:
                rdel[T] = np.nan
            else:
                ncur = n_cur.get(T, 0)
                last = hist[:len(hist) - ncur][-10:]
                P_last = strength(set().union(*last))[0] if last else np.nan
                full = {a for a in set().union(*hist[-10:]) if ST[a]["team"] == T}
                rdel[T] = strength(full)[0] - P_last
            prior[T] = b * float(P["r_last"].get(T, 0.0)) + (rdel[T] if np.isfinite(rdel[T]) else 0)
        t = tall[tall.date < d]

        def full_rating(tm):
            xo = t[t.team == tm].copy()
            no = len(xo)
            if not no:
                return prior.get(tm, 0.0), 0
            xo["adj"] = xo.margin - h * xo.hsign + xo.opp.map(prior).fillna(0)
            p3p = p3prev.get(tm, lg_prev)
            m3, a3, om3, oa3 = xo.m3.sum(), xo.a3.sum(), xo.om3.sum(), xo.oa3.sum()
            o_ = (m3 / a3 - (m3 + tr.K3 * p3p) / (a3 + tr.K3)) * a3 / no * 3
            d_ = (om3 / oa3 - (om3 + tr.K3 * lg_prev) / (oa3 + tr.K3)) * oa3 / no * 3
            wo = no / (no + (sig / tau) ** 2)
            return wo * (xo.adj.mean() - o_ + d_) + (1 - wo) * prior.get(tm, 0.0), no

        fullset = {T: {a for a in set().union(*TH.get(T, [set()])[-10:]) if ST[a]["team"] == T} for T in P["r_last"].index}
        for r in g[g.date == d].itertuples():
            rh, nh = full_rating(r.home)
            ra, na = full_rating(r.away)
            # POST-HOC DESCRIPTIVE ONLY (not a trial; LOOKAHEAD upper bound): availability = who actually played
            act = {}
            for side in ("home", "away"):
                T = getattr(r, side)
                played = {x.athlete_id for x in bx.get((r.event_id, T), [])}
                act[side] = (strength(played)[0] - strength(fullset[T])[0]) if played and fullset.get(T) else np.nan
            spread = rh - ra + h * float(not bool(r.neutral))
            rows.append({"event_id": str(r.event_id), "date": d, "start_utc": r.start_utc, "home": r.home, "away": r.away,
                         "neutral": bool(r.neutral), "r_home": rh, "r_away": ra, "n_home": nh, "n_away": na,
                         "prior_home": prior.get(r.home), "prior_away": prior.get(r.away),
                         "roster_home": rdel.get(r.home), "roster_away": rdel.get(r.away),
                         "spread_model": spread, "p_model": float(norm.cdf(spread / sig)), "sigma": sig,
                         "avail_act_home": act["home"], "avail_act_away": act["away"]})
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(RATINGS)
    print(f"wrote {RATINGS} ({len(out)} games, {out.date.min()}..{out.date.max()})")
    print(out[["p_model", "spread_model", "roster_home", "n_home"]].describe().round(3).to_string())
    return out


# ------------------------------------------------------------------------------------------------ parity
def parity(dates=("2025-12-15", "2026-02-20")) -> None:
    import contextlib
    import io
    import tempfile
    rt = pd.read_parquet(RATINGS)
    nsl.rosters = lambda *a, **k: {}
    nsl.injuries = lambda *a, **k: ({}, True)
    nsl.LIVE = Path(tempfile.mkdtemp())
    tr.kalshi_mid = lambda *a, **k: (None, None, {})
    for ds in dates:
        d = pd.Timestamp(ds).date()
        x = rt[rt.date == d].iloc[0]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            tr.main([x.home, "--date", ds, "--no-refresh"])
        txt = buf.getvalue()
        line = [l for l in txt.splitlines() if "BLENDED RATING" in l or "TONIGHT" in l or " by " in l or " gets " in l]
        print(f"{ds} {x.away}@{x.home}: mine r_home {x.r_home:+.2f} r_away {x.r_away:+.2f} spread {x.spread_model:+.2f} "
              f"p {x.p_model:.3f}\n   script: " + "\n   script: ".join(line))


# ------------------------------------------------------------------------------------------------ score
def market() -> pd.DataFrame:
    ln = pd.read_parquet(nsm.ESPN / f"lines_{CUR}.parquet")
    ln = ln[ln.provider.isin(["Draft Kings", "ESPN BET"])].copy()
    ln["pri"] = (ln.provider != "Draft Kings").astype(int)
    ln = ln.sort_values("pri").drop_duplicates("event_id")
    ln["event_id"] = ln.event_id.astype(str)
    return ln[["event_id", "provider", "home_ml_close", "away_ml_close", "home_spread_close", "home_spread_odds_close",
               "away_spread_odds_close"]]


def dv(a, b, method):
    out = []
    for x, y in zip(a, b):
        if not (np.isfinite(x) and np.isfinite(y)):
            out.append(np.nan)
            continue
        out.append(em.devig([x, y], method)[0])
    return np.array(out)


def boot(v, clus=None, seed=SEED):
    v = np.asarray(v, float)
    n = len(v)
    if n == 0:
        return {"bets": 0}
    rng = np.random.default_rng(seed)
    if clus is None:
        bs = v[rng.integers(0, n, (NB, n))].mean(1)
    else:
        c = pd.Series(v).groupby(np.asarray(clus))
        s, k = c.sum().to_numpy(), c.size().to_numpy()
        idx = rng.integers(0, len(s), (NB, len(s)))
        bs = s[idx].sum(1) / k[idx].sum(1)
    h1, h2 = v[: n // 2], v[n // 2:]
    return {"bets": n, "units": float(v.sum()), "roi": float(v.mean()), "ci_lo": float(np.quantile(bs, .025)),
            "ci_hi": float(np.quantile(bs, .975)), "p": float(np.mean(bs <= 0)), "half1": float(h1.mean()),
            "half2": float(h2.mean()), "half1_n": len(h1), "half2_n": len(h2), "boot_mean": float(bs.mean())}


def score(ledger: bool) -> int:
    import statsmodels.api as sm
    rt = pd.read_parquet(RATINGS)
    g = tr.games(CUR)
    g["event_id"] = g.event_id.astype(str)
    d = rt.merge(market(), on="event_id", how="left").merge(
        g[["event_id", "home_score", "away_score"]], on="event_id", how="left")
    assert d.event_id.is_unique
    d["y"] = (d.home_score > d.away_score).astype(float)
    d["margin"] = d.home_score - d.away_score
    for m in ("proportional", "power", "shin"):
        d[f"pc_{m}"] = dv(d.home_ml_close, d.away_ml_close, m)
        d[f"ps_{m}"] = dv(d.home_spread_odds_close, d.away_spread_odds_close, m)
    d = d.sort_values(["start_utc", "event_id"]).reset_index(drop=True)
    d["dec_h"] = [em.american_to_decimal(x) if np.isfinite(x) else np.nan for x in d.home_ml_close]
    d["dec_a"] = [em.american_to_decimal(x) if np.isfinite(x) else np.nan for x in d.away_ml_close]
    d["sdec_h"] = [em.american_to_decimal(x) if np.isfinite(x) else np.nan for x in d.home_spread_odds_close]
    d["sdec_a"] = [em.american_to_decimal(x) if np.isfinite(x) else np.nan for x in d.away_spread_odds_close]
    d["pcov_model"] = norm.cdf((d.spread_model + d.home_spread_close) / d.sigma)
    d["cover_m"] = d.margin + d.home_spread_close           # >0 home covers, 0 push
    ml = d.dropna(subset=["home_ml_close", "away_ml_close", "home_score"]).copy()
    print(f"games: ratings {len(rt)}, with ML close {len(ml)}, spread close "
          f"{d.dropna(subset=['home_spread_close', 'home_spread_odds_close', 'away_spread_odds_close']).shape[0]}")

    def ml_bets(x, thr, method="power"):
        gap = x.p_model - x[f"pc_{method}"]
        side = np.where(gap >= thr, 1, np.where(-gap >= thr, -1, 0))
        b = x[side != 0].copy()
        s = side[side != 0]
        win = np.where(s == 1, b.y == 1, b.y == 0)
        dec = np.where(s == 1, b.dec_h, b.dec_a)
        b["side"], b["pnl"] = np.where(s == 1, "home", "away"), np.where(win, dec - 1, -1.0)
        b["gap"] = np.abs(gap[side != 0])
        return b

    def spr_bets(x, thr, method="power"):
        x = x.dropna(subset=["home_spread_close", "sdec_h", "sdec_a", f"ps_{method}"])
        gap = x.pcov_model - x[f"ps_{method}"]
        side = np.where(gap >= thr, 1, np.where(-gap >= thr, -1, 0))
        b = x[side != 0].copy()
        s = side[side != 0]
        cm = b.cover_m.to_numpy()
        res = np.where(cm == 0, 0, np.where((cm > 0) == (s == 1), 1, -1))
        dec = np.where(s == 1, b.sdec_h, b.sdec_a)
        b["side"], b["pnl"] = np.where(s == 1, "home", "away"), np.where(res == 1, dec - 1, np.where(res == 0, 0.0, -1.0))
        b["gap"] = np.abs(gap[side != 0])
        return b

    trials = {
        "T1_ml_gap05": ml_bets(ml, 0.05),
        "T2_ml_gap03": ml_bets(ml, 0.03),
        "T3_ml_gap05_after10": ml_bets(ml[(ml.n_home >= 10) & (ml.n_away >= 10)], 0.05),
        "T4_spread_gap05": spr_bets(d.dropna(subset=["home_score"]), 0.05),
    }
    res = {}
    for i, (k, b) in enumerate(trials.items(), 1):
        r = boot(b.pnl.to_numpy())
        r["date_clustered"] = {kk: v for kk, v in boot(b.pnl.to_numpy(), b.date.astype(str).to_numpy()).items()
                               if kk in ("ci_lo", "ci_hi", "p")}
        r["passed"] = bool(r.get("p", 1) < BAR and r.get("half1", -1) > 0 and r.get("half2", -1) > 0)
        r["by_side"] = {s: {"n": int((b.side == s).sum()), "roi": float(b[b.side == s].pnl.mean())} for s in ("home", "away")}
        r["mean_gap"] = float(b.gap.mean()) if len(b) else None
        r["win_rate"] = float((b.pnl > 0).mean()) if len(b) else None
        r["sens_devig"] = {}
        for m in ("proportional", "shin"):
            if k.startswith("T4"):
                bb = spr_bets(d.dropna(subset=["home_score"]), 0.05, m)
            elif k.startswith("T3"):
                bb = ml_bets(ml[(ml.n_home >= 10) & (ml.n_away >= 10)], 0.05, m)
            else:
                bb = ml_bets(ml, 0.05 if k.startswith("T1") else 0.03, m)
            rb = boot(bb.pnl.to_numpy())
            r["sens_devig"][m] = {kk: rb.get(kk) for kk in ("bets", "roi", "ci_lo", "ci_hi", "p")}
        res[k] = r
        b.assign(trial=k)[["trial", "event_id", "date", "away", "home", "side", "p_model", "pc_power", "pcov_model",
                           "ps_power", "home_ml_close", "away_ml_close", "home_spread_close", "home_score", "away_score",
                           "gap", "pnl"]].to_csv(OUT / f"bets_{k}.csv", index=False)
        print(f"{k}: n {r['bets']}  ROI {r['roi']:+.4f}  [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  p {r['p']:.4f} "
              f"(bar {BAR})  halves {r['half1']:+.4f}/{r['half2']:+.4f}  win {r['win_rate']:.3f}  "
              f"date-clus [{r['date_clustered']['ci_lo']:+.4f},{r['date_clustered']['ci_hi']:+.4f}] "
              f"sens {json.dumps({m: round(v['roi'], 4) for m, v in r['sens_devig'].items()})}  PASS={r['passed']}")

    # information: logistic regression + Brier/log-loss
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    X = sm.add_constant(np.column_stack([lg(ml.pc_power), lg(ml.p_model)]))
    fit = sm.Logit(ml.y.to_numpy(), X).fit(disp=0)
    ci = fit.conf_int()
    rng = np.random.default_rng(SEED)
    bc = []
    for _ in range(2000):
        ix = rng.integers(0, len(ml), len(ml))
        try:
            bc.append(sm.Logit(ml.y.to_numpy()[ix], X[ix]).fit(disp=0).params[2])
        except Exception:
            pass
    info = {"n": int(len(ml)), "coef_close": float(fit.params[1]), "coef_model": float(fit.params[2]),
            "coef_model_ci": [float(ci[2][0]), float(ci[2][1])], "coef_model_boot_ci": [float(np.quantile(bc, .025)),
                                                                                      float(np.quantile(bc, .975))],
            "coef_close_ci": [float(ci[1][0]), float(ci[1][1])]}
    yv = ml.y.to_numpy()
    bm = lambda p: (p - yv) ** 2
    ll = lambda p: -(yv * np.log(np.clip(p, 1e-9, 1)) + (1 - yv) * np.log(np.clip(1 - p, 1e-9, 1)))
    for nm, f in (("brier", bm), ("logloss", ll)):
        a, c = f(ml.p_model.to_numpy()), f(ml.pc_power.to_numpy())
        dd = a - c
        bs = dd[rng.integers(0, len(dd), (NB, len(dd)))].mean(1)
        info[nm] = {"model": float(a.mean()), "close_power": float(c.mean()), "diff_model_minus_close": float(dd.mean()),
                    "ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
    info["brier_close_proportional"] = float(bm(ml.pc_proportional.to_numpy()).mean())
    print(f"\nlogit(y) = a + {info['coef_close']:.3f} logit(close) + {info['coef_model']:.3f} logit(model); model coef 95% CI "
          f"{info['coef_model_ci'][0]:+.3f}..{info['coef_model_ci'][1]:+.3f} (boot {info['coef_model_boot_ci'][0]:+.3f}.."
          f"{info['coef_model_boot_ci'][1]:+.3f}), n {info['n']}")
    for nm in ("brier", "logloss"):
        v = info[nm]
        print(f"{nm}: model {v['model']:.5f} close {v['close_power']:.5f} diff {v['diff_model_minus_close']:+.5f} "
              f"[{v['ci'][0]:+.5f}, {v['ci'][1]:+.5f}]")
    # calibration by gap bucket
    gap = ml.p_model - ml.pc_power
    edges = [-1, -.10, -.05, -.03, 0, .03, .05, .10, 1]
    ml["bucket"] = pd.cut(gap, edges, right=False)
    cal = ml.groupby("bucket", observed=False).agg(n=("y", "size"), p_model=("p_model", "mean"),
                                                   p_close=("pc_power", "mean"), home_win=("y", "mean"))
    print("\ncalibration by P_model - P_close (home side):")
    print(cal.round(3).to_string())
    info["calibration"] = {str(k): {c: (None if pd.isna(v) else float(v)) for c, v in row.items()}
                           for k, row in cal.iterrows()}
    # POST-HOC DESCRIPTIVE (added after the trials were scored; NOT a trial, LOOKAHEAD upper bound): availability from
    # who actually played. Answers "is the loss just missing injury info?" -- an injury report at tip is weaker than this.
    x = ml.dropna(subset=["avail_act_home", "avail_act_away"]).copy()
    x["p_act"] = norm.cdf((x.spread_model + x.avail_act_home - x.avail_act_away) / x.sigma)
    Xa = sm.add_constant(np.column_stack([lg(x.pc_power), lg(x.p_act)]))
    fa = sm.Logit(x.y.to_numpy(), Xa).fit(disp=0)
    ba = ml_bets(x.assign(p_model=x.p_act), 0.05)
    rb = boot(ba.pnl.to_numpy())
    ya = x.y.to_numpy()
    da = (x.p_act - ya) ** 2 - (x.pc_power - ya) ** 2
    bsa = da.to_numpy()[rng.integers(0, len(da), (NB, len(da)))].mean(1)
    info["D1_actual_availability_lookahead"] = {
        "n": int(len(x)), "coef_model": float(fa.params[2]), "coef_model_ci": [float(v) for v in fa.conf_int()[2]],
        "brier_model": float(((x.p_act - ya) ** 2).mean()), "brier_close": float(((x.pc_power - ya) ** 2).mean()),
        "brier_diff_ci": [float(np.quantile(bsa, .025)), float(np.quantile(bsa, .975))],
        "T1_analog": {kk: rb.get(kk) for kk in ("bets", "roi", "ci_lo", "ci_hi", "p", "half1", "half2")}}
    print(f"\nD1 (descriptive, LOOKAHEAD availability = who played): {json.dumps(info['D1_actual_availability_lookahead'])}")
    t1 = res["T1_ml_gap05"]
    killed = (info["coef_model_ci"][0] <= 0 <= info["coef_model_ci"][1]) and (t1["ci_lo"] <= 0 <= t1["ci_hi"])
    verdict = ("PASS" if any(r["passed"] for r in res.values()) else ("KILL" if killed else "FAIL (not killed)"))
    print(f"\nVERDICT: {verdict}  (kill = model coef CI incl 0 AND T1 ROI CI incl 0 -> {killed})")
    summary = {"trials": res, "info": info, "verdict": verdict, "killed": killed}
    (OUT / "score.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"wrote {OUT / 'score.json'}")
    if ledger:
        sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
        with LEDGER.open("a") as f:
            for i, (k, r) in enumerate(res.items(), 1):
                row = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"l2_{k}", "sha": sha,
                       "dataset": "l2_team_rating_sides",
                       "hypothesis": "pre-registered in notes/l2_team_rating_sides.md; team_rating.py (no availability) "
                                     "vs power-devig ESPN close, flat 1u at close",
                       "source": "registry L2", "status": "ok", "split": "HELDOUT_2025-26 (standing authority 10-29)",
                       "trial_number": i, "study_trials": K_TRIALS, "bonferroni_bar": BAR,
                       **{kk: v for kk, v in r.items() if kk != "boot_mean"}, "boot_mean": r.get("boot_mean"),
                       "info_coef_model": info["coef_model"], "info_coef_model_ci": info["coef_model_ci"],
                       "verdict": verdict}
                f.write(json.dumps(row, default=str) + "\n")
        print(f"appended {len(res)} rows to {LEDGER}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "parity", "score"])
    ap.add_argument("--ledger", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build()
    elif a.cmd == "parity":
        parity()
    else:
        return score(a.ledger)
    return 0


if __name__ == "__main__":
    sys.exit(main())
