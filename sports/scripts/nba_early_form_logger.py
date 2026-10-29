"""F1 forward paper logger: early-season form vs market (frozen DEV T1 rule, both teams 3..8 games). PAPER ONLY, no orders.
Pre-registration: notes/nba_early_season_bias.md, section "FORWARD PAPER TEST F1".

    ../.venv_quant/bin/python scripts/nba_early_form_logger.py tick [--date YYYY-MM-DD] [--force] [--no-refresh]
    ../.venv_quant/bin/python scripts/nba_early_form_logger.py report

tick   : for each not-yet-tipped NBA game on the ET date where both teams have <= 8 games before the date, if tip is
         60-90 min away (or --force: any time >= 15 min before tip) and the game is not logged yet, append one row:
         gp, adjMOV, ESPN spread (market-implied margin), R, fire/side, Kalshi bid/ask both sides, ESPN ML.
report : settles from ESPN finals; ROI at Kalshi ask + taker fee with a bootstrap CI; CLV = ESPN close (power de-vig)
         on the bet side minus the ask (and minus the Kalshi mid).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

import edge_math as em
import nba_early_season_bias as nesb

LOG = ROOT / "data" / "nba_early_form_log.jsonl"
WIN = (60, 90)
MIN_FORCE = 15
X = 5.365908549009408          # frozen DEV T1 threshold (results/nba_early_season_bias/score.json "X")
H = 2.06                       # frozen: mean of DEV H_s 2.532 / 2.167 / 1.545 / 1.998
GP_MIN, GP_MAX = 3, 8


def logged() -> list[dict]:
    if not LOG.exists():
        return []
    return [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]


def season_of(d) -> int:
    return d.year + 1 if d.month >= 8 else d.year


def load_games(cur: int) -> pd.DataFrame:
    g = pd.read_parquet(nesb.ESPN / f"games_{cur}.parquet")
    g = g[g.season_type == "regular-season"].copy()
    g["event_id"] = g.event_id.astype(str)
    g["date"] = g.start_utc.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    g["margin"] = g.home_score - g.away_score
    return g


def slate(d, refresh: bool) -> list[dict]:
    cur = season_of(d)
    if refresh:
        try:
            import nba_simple_live as nsl
            nsl.refresh_season(cur, d, True)
        except Exception as ex:
            print(f"(ESPN refresh failed: {str(ex)[:60]}; using disk)")
    g = load_games(cur)
    fin = g[g.status == "STATUS_FINAL"]
    D = pd.Timestamp(d)
    f = nesb.form_asof(nesb.team_long(fin), D)       # only finals with ET date < d
    out = []
    for r in g[g.date == D].itertuples():
        gh = int(f.loc[r.home].gp) if r.home in f.index else 0
        ga = int(f.loc[r.away].gp) if r.away in f.index else 0
        out.append({"event_id": r.event_id, "home": r.home, "away": r.away, "start_utc": str(r.start_utc),
                    "status": r.status, "gp_h": gh, "gp_a": ga,
                    "adj_h": float(f.loc[r.home].adj) if gh else None, "adj_a": float(f.loc[r.away].adj) if ga else None,
                    "eligible": bool(GP_MIN <= gh <= GP_MAX and GP_MIN <= ga <= GP_MAX)})
    return out


def signal(s: dict, spread) -> dict:
    if s["adj_h"] is None or s["adj_a"] is None or spread is None or not np.isfinite(spread):
        return {"M": None, "form_gap": None, "R": None, "fires": False, "side": None}
    M = -float(spread)
    fg = s["adj_h"] - s["adj_a"]
    R = fg - (M - H)
    fires = bool(s["eligible"] and abs(R) >= X)
    return {"M": M, "form_gap": fg, "R": R, "fires": fires, "side": ("home" if R >= X else "away") if fires else None}


def cmd_tick(a) -> int:
    from l2_rating_logger import espn_ml
    d = pd.Timestamp(a.date).date() if a.date else pd.Timestamp.now(tz="US/Eastern").date()
    now = pd.Timestamp.now(tz="UTC")
    done = {r["event_id"] for r in logged()}
    todo = []
    for s in slate(d, not a.no_refresh):
        mins = (pd.Timestamp(s["start_utc"]) - now).total_seconds() / 60
        s["mins_to_tip"] = round(mins, 1)
        tag = f"{s['away']}@{s['home']} (gp {s['gp_a']}/{s['gp_h']})"
        if s["gp_h"] > GP_MAX or s["gp_a"] > GP_MAX:
            print(f"{tag}: a team has > {GP_MAX} games -- outside F1")
        elif s["event_id"] in done:
            print(f"{tag}: already logged")
        elif mins < (MIN_FORCE if a.force else WIN[0]) or s["status"] not in (None, "STATUS_SCHEDULED"):
            if mins > 0 and s["status"] in (None, "STATUS_SCHEDULED") and mins < WIN[0]:
                print(f"{tag}: {mins:.0f} min to tip, inside {MIN_FORCE if a.force else WIN[0]} min -- too late, not logged")
            else:
                print(f"{tag}: tipped/started ({mins:.0f} min, {s['status']}) -- not logged")
        elif mins > WIN[1] and not a.force:
            print(f"{tag}: {mins:.0f} min to tip, outside {WIN} -- wait")
        else:
            todo.append(s)
    if not todo:
        return 0
    import nba_tonight_sheet as nts
    kg = nts.kalshi_series("KXNBAGAME")
    num = lambda v: None if v is None or not np.isfinite(v) else float(v)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as fh:
        for s in todo:
            e = espn_ml(s["event_id"])
            sig = signal(s, e.get("spread"))
            k = nts.kalshi_game(d, s["home"], s["away"], kg)
            kh, ka = k.get("home", {}), k.get("away", {})
            kal = {"home_bid": num(kh.get("bid", np.nan)), "home_ask": num(kh.get("ask", np.nan)),
                   "away_bid": num(ka.get("bid", np.nan)), "away_ask": num(ka.get("ask", np.nan))}
            row = {"ts": now.isoformat(timespec="seconds"), "date": str(d), **s, **sig, "X": X, "H": H,
                   "in_window": bool(WIN[0] <= s["mins_to_tip"] <= WIN[1]), "kalshi": kal, "espn": e}
            fh.write(json.dumps(row, default=str) + "\n")
            ask = kal.get(f"{sig['side']}_ask") if sig["side"] else None
            fmt = lambda v: "NA" if v is None else f"{v:+.2f}"
            print(f"LOGGED {s['away']}@{s['home']} ({s['mins_to_tip']:.0f} min{'' if row['in_window'] else ', FORCED'}) "
                  f"gp {s['gp_a']}/{s['gp_h']} eligible {s['eligible']}: adjMOV away {fmt(s['adj_a'])} home {fmt(s['adj_h'])}"
                  f" | form gap (home) {fmt(sig['form_gap'])}, market margin M {fmt(sig['M'])} (ESPN {e.get('provider')} "
                  f"spread {e.get('spread')}), M-H {fmt(None if sig['M'] is None else sig['M'] - H)}, R {fmt(sig['R'])}"
                  f" -> {'FIRE ' + sig['side'].upper() + f' @ Kalshi ask {ask}' if sig['fires'] else 'no bet'} | Kalshi "
                  f"home {kal['home_bid']}/{kal['home_ask']} away {kal['away_bid']}/{kal['away_ask']}")
    return 0


def cmd_report(a) -> int:
    from l2_rating_logger import espn_ml
    rows = logged()
    if not rows:
        print("no rows logged")
        return 0
    cur = season_of(pd.Timestamp(rows[0]["date"]))
    if not a.no_refresh:
        try:
            import nba_simple_live as nsl
            nsl.refresh_season(cur, pd.Timestamp.now(tz="US/Eastern").date(), True)
        except Exception as ex:
            print(f"(ESPN refresh failed: {str(ex)[:60]})")
    g = load_games(cur)
    g = g[g.status == "STATUS_FINAL"]
    res = dict(zip(g.event_id, (g.home_score > g.away_score).astype(int)))
    out = []
    for r in rows:
        y = res.get(r["event_id"])
        side, k = r.get("side"), r["kalshi"]
        rec = {"date": r["date"], "game": f"{r['away']}@{r['home']}", "R": r.get("R"), "fires": r.get("fires"),
               "side": side, "mins": r["mins_to_tip"], "in_window": r["in_window"], "y": y, "pnl": None,
               "clv_ask": None, "clv_mid": None}
        ask = k.get(f"{side}_ask") if side else None
        if side and ask is not None:
            cost = ask + em.fee_per_contract(ask, "kalshi")
            rec["ask"] = ask
            if y is not None:
                won = (y == 1) if side == "home" else (y == 0)
                rec["pnl"] = ((1 - cost) if won else -cost) / cost
                e = espn_ml(r["event_id"])
                hc, ac = e.get("home_ml_close"), e.get("away_ml_close")
                if hc is not None and ac is not None and np.isfinite(hc) and np.isfinite(ac):
                    pc = em.devig([hc, ac], "power")[0]
                    ps = pc if side == "home" else 1 - pc
                    rec["clv_ask"] = ps - ask
                    b, aa = k.get(f"{side}_bid"), k.get(f"{side}_ask")
                    if b is not None:
                        rec["clv_mid"] = ps - (b + aa) / 2
        out.append(rec)
    df = pd.DataFrame(out)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    s = df[df.fires.fillna(False).astype(bool) & (df.mins >= MIN_FORCE)].dropna(subset=["pnl"])
    if len(s):
        v = s.pnl.to_numpy()
        bs = v[np.random.default_rng(2026).integers(0, len(v), (10000, len(v)))].mean(1)
        print(f"\nF1 primary: bets {len(v)}, ROI {v.mean():+.4f} 95% CI [{np.quantile(bs, .025):+.4f}, "
              f"{np.quantile(bs, .975):+.4f}], one-sided p {np.mean(bs <= 0):.4f}")
        w = s[s.in_window]
        if len(w):
            print(f"  in-window only: bets {len(w)}, ROI {w.pnl.mean():+.4f}")
        for c in ("clv_ask", "clv_mid"):
            x = s[c].dropna().to_numpy()
            if len(x) > 1:
                se = x.std(ddof=1) / np.sqrt(len(x))
                print(f"  {c}: mean {x.mean():+.4f} 95% CI [{x.mean() - 1.96 * se:+.4f}, {x.mean() + 1.96 * se:+.4f}] n {len(x)}")
    print(f"\nrows {len(df)}, fired {int(df.fires.fillna(False).sum())}, settled {df.y.notna().sum()}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tick", "report"])
    ap.add_argument("--date")
    ap.add_argument("--force", action="store_true", help=f"log any not-yet-tipped game >= {MIN_FORCE} min before tip")
    ap.add_argument("--no-refresh", action="store_true")
    a = ap.parse_args(argv)
    return cmd_tick(a) if a.cmd == "tick" else cmd_report(a)


if __name__ == "__main__":
    sys.exit(main())
