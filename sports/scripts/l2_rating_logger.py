"""L2 forward logger: team_rating.py (incl. availability) vs Kalshi KXNBAGAME / Pinnacle / ESPN book, once per game
~60-90 min before tip. PAPER ONLY -- never places orders. Context: notes/l2_team_rating_sides.md (2025-26 backtest
without availability FAILED: the close beats this model; this logger asks whether the live model with injuries does).

    ../.venv_quant/bin/python scripts/l2_rating_logger.py tick [--date YYYY-MM-DD] [--force] [--no-refresh]
    ../.venv_quant/bin/python scripts/l2_rating_logger.py report

tick   : for each NBA game on the ET date that has not tipped, if tip is 60-90 min away (or --force: any time before tip)
         and the game is not yet in the log, append one row to data/l2_rating_log.jsonl: model P(home) (team_rating
         prior + season blend + 3P luck + ESPN-injury availability), Kalshi bid/ask both sides, Pinnacle ML fair
         (proportional + power), ESPN first-book ML. Not installed as a launchd agent (run it every ~10 min if wanted).
report : settles from ESPN finals; paper bet = the model side at the Kalshi ask + taker fee whenever model - ask > 0,
         ROI grouped by gap bucket; CLV = de-vigged ESPN close (power) on the model side minus the logged Pinnacle
         power-fair (else Kalshi mid) on that side.
"""
from __future__ import annotations

import argparse
import contextlib
import io
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
import team_rating as tr

LOG = ROOT / "data" / "l2_rating_log.jsonl"
WIN = (60, 90)          # minutes before tip
BUCKETS = [-1, 0.0, 0.03, 0.05, 0.10, 1]


def _now():
    return pd.Timestamp.now(tz="UTC")


def logged() -> list[dict]:
    if not LOG.exists():
        return []
    return [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]


def model_slate(d, refresh: bool):
    """Per game on ET date d: team_rating's blended ratings + availability -> spread, P(home). Same math as tr.main."""
    cur = d.year + 1 if d.month >= 8 else d.year
    if refresh:
        try:
            import nba_simple_live as nsl
            nsl.refresh_season(cur, d, True)
        except Exception as ex:
            print(f"(ESPN refresh failed: {str(ex)[:60]}; using disk)")
    P = tr.fit_params(cur)
    sig, tau, b, h = P["sigma"], P["tau"], P["b"], P["h"]
    with contextlib.redirect_stdout(io.StringIO()):
        rd, today, inj_ok = tr.roster_deltas(d, cur)
    prior = {t: b * float(P["r_last"].get(t, 0.0)) + (v["roster"] if np.isfinite(v["roster"]) else 0) for t, v in rd.items()}
    g = tr.games(cur)
    g = g[pd.to_datetime(g.start_utc).dt.tz_convert("US/Eastern").dt.date < d]
    t = tr.box_team(cur, g) if len(g) else pd.DataFrame(columns=["team"])
    prev = tr.box_team(cur - 1, tr.games(cur - 1))
    lg_prev = prev.m3.sum() / prev.a3.sum()

    def full(tm):
        xo = t[t.team == tm].copy()
        no = len(xo)
        if not no:
            return prior.get(tm, 0.0), 0
        xo["adj"] = xo.margin - h * xo.hsign + xo.opp.map(prior).fillna(0)
        p3p = prev[prev.team == tm].m3.sum() / max(prev[prev.team == tm].a3.sum(), 1)
        m3, a3, om3, oa3 = xo.m3.sum(), xo.a3.sum(), xo.om3.sum(), xo.oa3.sum()
        o_ = (m3 / a3 - (m3 + tr.K3 * p3p) / (a3 + tr.K3)) * a3 / no * 3
        d_ = (om3 / oa3 - (om3 + tr.K3 * lg_prev) / (oa3 + tr.K3)) * oa3 / no * 3
        wo = no / (no + (sig / tau) ** 2)
        return wo * (xo.adj.mean() - o_ + d_) + (1 - wo) * prior.get(tm, 0.0), no

    out = []
    for r in today.itertuples():
        rh, nh = full(r.home)
        ra, na = full(r.away)
        vh, va = rd.get(r.home, {}), rd.get(r.away, {})
        ah, aa = vh.get("avail", 0.0), va.get("avail", 0.0)
        spread = (rh + ah) - (ra + aa) + h * float(not bool(r.neutral))
        out.append({"event_id": str(r.event_id), "home": r.home, "away": r.away, "start_utc": str(r.start_utc),
                    "status": getattr(r, "status", None), "r_home": rh, "r_away": ra, "n_home": nh, "n_away": na,
                    "avail_home": ah, "avail_away": aa, "out_home": vh.get("out", []), "out_away": va.get("out", []),
                    "q_home": vh.get("q", []), "q_away": va.get("q", []), "spread_model": spread,
                    "p_model_home": float(norm.cdf(spread / sig)), "sigma": sig, "injury_feed_ok": bool(inj_ok)})
    return out


def pinnacle_ml() -> dict:
    import nba_tonight_sheet as nts
    try:
        from sports import live
        p = live.pinnacle("nba")
    except Exception as ex:
        print(f"(Pinnacle failed: {str(ex)[:60]})")
        return {}
    p = p[(~p.live) & (p.market == "moneyline")]
    out = {}
    for _, x in p.iterrows():
        hn, an = nts.nickname(x.home), nts.nickname(x.away)
        if hn and an:
            out[(nts.ABBR_OF_NICK[hn], nts.ABBR_OF_NICK[an])] = {
                "home_ml": x.price1, "away_ml": x.price2, "fair_prop_home": x.fair1, "fair_power_home": x.pfair1,
                "start": str(x.start)}
    return out


def espn_ml(eid: str) -> dict:
    try:
        from sports import espn
        import nba_simple_model as nsm
        ls = [x for x in espn.lines("nba", eid, cache=False)
              if x.get("provider") and str(x["provider"]).lower() not in nsm.NOT_BOOKS]
    except Exception:
        return {}
    for x in ls:
        if np.isfinite(x.get("home_ml") or np.nan) and np.isfinite(x.get("away_ml") or np.nan):
            return {"provider": x["provider"], "home_ml": x["home_ml"], "away_ml": x["away_ml"],
                    "home_ml_close": x.get("home_ml_close"), "away_ml_close": x.get("away_ml_close"),
                    "spread": x.get("spread")}
    return {}


def cmd_tick(a) -> int:
    d = pd.Timestamp(a.date).date() if a.date else pd.Timestamp.now(tz="US/Eastern").date()
    now = _now()
    done = {r["event_id"] for r in logged()}
    slate = model_slate(d, not a.no_refresh)
    todo = []
    for s in slate:
        mins = (pd.Timestamp(s["start_utc"]) - now).total_seconds() / 60
        s["mins_to_tip"] = round(mins, 1)
        if s["event_id"] in done:
            print(f"{s['away']}@{s['home']}: already logged")
        elif mins <= 0 or (s["status"] not in (None, "STATUS_SCHEDULED")):
            print(f"{s['away']}@{s['home']}: tipped/started ({mins:.0f} min, {s['status']}) -- not logged")
        elif not (WIN[0] <= mins <= WIN[1]) and not a.force:
            print(f"{s['away']}@{s['home']}: {mins:.0f} min to tip, outside {WIN} -- wait")
        else:
            todo.append(s)
    if not todo:
        return 0
    import nba_tonight_sheet as nts
    kg = nts.kalshi_series("KXNBAGAME")
    pin = pinnacle_ml()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        for s in todo:
            k = nts.kalshi_game(d, s["home"], s["away"], kg)
            kh, ka = k.get("home", {}), k.get("away", {})
            num = lambda v: None if v is None or not np.isfinite(v) else float(v)
            row = {"ts": now.isoformat(timespec="seconds"), "date": str(d), **s,
                   "in_window": bool(WIN[0] <= s["mins_to_tip"] <= WIN[1]),
                   "kalshi": {"home_bid": num(kh.get("bid", np.nan)), "home_ask": num(kh.get("ask", np.nan)),
                              "away_bid": num(ka.get("bid", np.nan)), "away_ask": num(ka.get("ask", np.nan))},
                   "pinnacle": pin.get((s["home"], s["away"])), "espn": espn_ml(s["event_id"])}
            kk = row["kalshi"]
            mid = (kk["home_bid"] + kk["home_ask"]) / 2 if kk["home_bid"] is not None and kk["home_ask"] is not None else None
            row["kalshi_mid_home"] = mid
            f.write(json.dumps(row, default=str) + "\n")
            pf = (row["pinnacle"] or {}).get("fair_power_home")
            print(f"LOGGED {s['away']}@{s['home']} ({s['mins_to_tip']:.0f} min to tip{'' if row['in_window'] else ', FORCED'}): "
                  f"model P(home) {s['p_model_home']:.3f} (spread {s['spread_model']:+.1f}, avail {s['avail_home']:+.1f}/"
                  f"{s['avail_away']:+.1f})  Kalshi home {kk['home_bid']}/{kk['home_ask']} away {kk['away_bid']}/"
                  f"{kk['away_ask']}  Pinnacle fair(power) home {pf if pf is None else round(pf, 3)}")
    return 0


def cmd_report(a) -> int:
    rows = logged()
    if not rows:
        print("no rows logged")
        return 0
    seasons = sorted({(pd.Timestamp(r["date"]).year + 1 if pd.Timestamp(r["date"]).month >= 8 else pd.Timestamp(r["date"]).year)
                      for r in rows})
    fin = pd.concat([pd.read_parquet(tr.ESPN / f"games_{s}.parquet") for s in seasons])
    fin = fin[fin.status == "STATUS_FINAL"]
    fin["event_id"] = fin.event_id.astype(str)
    res = dict(zip(fin.event_id, (fin.home_score > fin.away_score).astype(int)))
    out = []
    for r in rows:
        y = res.get(r["event_id"])
        pm, k = r["p_model_home"], r["kalshi"]
        side = None
        cands = []
        if k.get("home_ask") is not None:
            cands.append(("home", pm - k["home_ask"], k["home_ask"]))
        if k.get("away_ask") is not None:
            cands.append(("away", (1 - pm) - k["away_ask"], k["away_ask"]))
        if cands:
            side, gap, ask = max(cands, key=lambda c: c[1])
        else:
            gap = ask = None
        pnl = None
        if side and y is not None:
            cost = ask + em.fee_per_contract(ask, "kalshi")
            won = (y == 1) if side == "home" else (y == 0)
            pnl = ((1 - cost) if won else -cost) / cost
        clv = None
        e = espn_ml(r["event_id"]) if y is not None else {}
        hc, ac = e.get("home_ml_close"), e.get("away_ml_close")
        ref = (r.get("pinnacle") or {}).get("fair_power_home") or r.get("kalshi_mid_home")
        if side and hc is not None and ac is not None and np.isfinite(hc) and np.isfinite(ac) and ref is not None:
            pc = em.devig([hc, ac], "power")[0]
            clv = (pc - ref) if side == "home" else ((1 - pc) - (1 - ref))
        out.append({"date": r["date"], "game": f"{r['away']}@{r['home']}", "p_model": pm, "side": side, "gap": gap,
                    "ask": ask, "y": y, "pnl": pnl, "clv": clv, "in_window": r.get("in_window")})
    df = pd.DataFrame(out)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    s = df.dropna(subset=["pnl", "gap"])
    if len(s):
        s = s.assign(bucket=pd.cut(s.gap, BUCKETS, right=False))
        print("\nROI per $ at Kalshi ask + taker fee, by model-minus-ask gap (paper):")
        print(s.groupby("bucket", observed=True).agg(n=("pnl", "size"), roi=("pnl", "mean"), clv=("clv", "mean")).to_string())
        c = s.clv.dropna()
        if len(c) > 1:
            print(f"\nCLV (ESPN close power-fair minus logged reference, model side): mean {c.mean():+.4f}, "
                  f"se {c.std(ddof=1) / np.sqrt(len(c)):.4f}, n {len(c)}")
    print(f"\nrows {len(df)}, settled {df.y.notna().sum()}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tick", "report"])
    ap.add_argument("--date")
    ap.add_argument("--force", action="store_true", help="log every not-yet-tipped game regardless of the 60-90 min window")
    ap.add_argument("--no-refresh", action="store_true")
    a = ap.parse_args(argv)
    return cmd_tick(a) if a.cmd == "tick" else cmd_report(a)


if __name__ == "__main__":
    sys.exit(main())
