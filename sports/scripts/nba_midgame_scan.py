"""L37 NBA MID-GAME comeback scanner + FORWARD PAPER logger. Paper only: NO orders are ever placed by this script.

    python scripts/nba_midgame_scan.py fit       # (worktree only) fit the history model -> results/nba_midgame_fit.json
                                                 #   + data/nba_midgame_states.parquet (compact dev states, shipped to deploy)
    python scripts/nba_midgame_scan.py show      # live table: every live game, model vs empirical vs Kalshi, both sides
    python scripts/nba_midgame_scan.py tick      # launchd every 90 s: log window ticks, paper entries, settle
    python scripts/nba_midgame_scan.py report    # entries W-L, P&L/contract, mean edge; Kalshi-mid calibration table

User idea (2026-10-26): "comparable teams (strength) with similar closeness -- what probability do they come back once
the game starts". notes/nba_comebacks_matchups.md showed comebacks depend only on deficit x time x pregame spread, and
late_game_calibration.md showed Kalshi is calibrated in Q4 <= 6:00 close states. The untested region is Q2..Q4 6:00
with deficits 6-25; this logger tests it forward.

History model (dev seasons 2015-16..2024-25 ONLY; 2025-26+ never read; data/nba_ingame.parquet never read):
  states = data/nba_comebacks/states.parquet (per-minute ESPN pbp states, closing spread = home expected margin from
  ESPN pickcenter / OddsData). Trailer perspective: d = deficit >= 1, s = trailer's pregame expected margin
  (+ = trailer was favoured), tau = regulation time left / 2880. Probit
      P(trailer wins) = Phi(b . [1, d/sqrt(tau), s*sqrt(tau), d, s, tau, (d/sqrt(tau))^2])
  fit on el (elapsed s) 600..2640, d 1..35. Out-of-sample check: fit <= 2023-24, score 2024-25 (results json `oos`).
  Empirical check: games within el +-120 s, deficit +-2, s +-3 (one state per game, nearest el), Wilson 95% CI.
  Leader probability = 1 - trailer probability.

PRE-REGISTRATION (frozen 2026-10-26 ~01:00 PT, written BEFORE the first live tick; any change = a new trial name)
-----------------------------------------------------------------------------------------------------------------
Trial L37 "midgame model vs Kalshi" (forward paper, 1 contract, taker):
  Window: period 2 (any clock, incl. halftime = Q2 0:00), period 3 (any clock), period 4 with clock > 360 s.
    OT and Q1 never qualify. State = ESPN live scoreboard (period, status.clock, scores) at the tick.
  Deficit: 6 <= |home score - away score| <= 25.
  Pregame spread required: ESPN odds/pickcenter home spread cached while the game is `pre` (or from summary
    pickcenter at first live sight); fallback = home expected margin implied by the last pregame Kalshi mid
    (sigma from the dev fit). No spread -> the tick is logged but can never be an entry.
  For each side X in {trailer, leader}: edge_X = p_model_X - ask_X - 0.07*ask_X*(1-ask_X) (exact taker fee,
    unrounded), ask = that team's KXNBAGAME YES ask (KXNBAGAME-<ET date><AWAY><HOME>-<TEAM>), 0.01 <= ask <= 0.99.
  ENTRY: the first tick in a game where some side has edge_X >= 0.05 AND the empirical neighbour check agrees in
    sign (n_emp >= 30 and p_emp_X - ask_X - fee > 0). If both sides qualify (impossible unless the book is crossed)
    take the larger edge. ONE entry per game, 1 contract, hold to settlement, no exits/adds.
  Settlement: ESPN final (competitor winner); Kalshi market `result` stored as a cross-check.
    P&L/contract = 1{X won} - ask - fee.
  Also logged (measurement): EVERY in-window tick with deficit 6-25, with both sides' bid/ask, model, empirical
    (for calibration of Kalshi vs model in this region), whether or not an entry happens.
  Report: n, W-L, P&L per contract (exact fee; cent-rounded fee sensitivity), mean edge at entry, model-implied
    vs realised; calibration table: trailer Kalshi mid by bin vs realised trailer win rate and mean model, one tick
    per game per bin (the first), Wilson CIs.
  Scoring: descriptive until n >= 100 entries; then one-sided test of mean P&L > 0 (game-level t / bootstrap).
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path(os.environ.get("MIDGAME_DATA", ROOT / "data"))
FIT = ROOT / "results" / "nba_midgame_fit.json"
STATES = DATA / "nba_midgame_states.parquet"
CACHE = DATA / "nba_midgame_pregame.json"
TICKS = DATA / "nba_midgame_ticks.jsonl"
ENTRIES = DATA / "nba_midgame_entries.jsonl"
RESULTS = DATA / "nba_midgame_results.jsonl"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
ET = ZoneInfo("America/New_York")
ESPN2K = {"SA": "SAS", "GS": "GSW", "NY": "NYK", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}
DEV_MAX_SEASON = 2024  # 2024-25; 2025-26 is sealed
EDGE_MIN, EMP_MIN_N = 0.05, 30
D_LO, D_HI = 6, 25
EL_LO, EL_HI = 720, 2520  # Q2 start .. Q4 6:00 (elapsed regulation seconds)


# ----------------------------------------------------------------------------------------------- model
def feats(d, s, tau):
    d, s, tau = (np.asarray(x, dtype=float) for x in (d, s, tau))
    tau = np.maximum(tau, 1e-3)
    z = d / np.sqrt(tau)
    return np.column_stack([np.ones_like(d), z, s * np.sqrt(tau), d, s, tau, z ** 2])


def p_trailer(fit: dict, d, s, el):
    b = np.asarray(fit["coef"])
    return norm.cdf(feats(d, s, (2880 - np.asarray(el, dtype=float)) / 2880) @ b)


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def empirical(st: pd.DataFrame, d: float, s: float, el: float):
    q = st[(st.el - el).abs().le(120) & (st.deficit - d).abs().le(2) & (st.s_tr - s).abs().le(3)]
    if q.empty:
        return np.nan, 0, (np.nan, np.nan)
    q = q.assign(dist=(q.el - el).abs()).sort_values("dist").drop_duplicates("eid")
    k, n = int(q.y.sum()), len(q)
    return k / n, n, wilson(k, n)


def fit() -> int:
    import statsmodels.api as sm
    s = pd.read_parquet(ROOT / "data" / "nba_comebacks" / "states.parquet")
    s = s[s.season <= DEV_MAX_SEASON]
    assert s.season.max() <= DEV_MAX_SEASON
    s = s[(s.m != 0) & s.spread.notna()].copy()
    s["home_trails"] = s.m < 0
    s["deficit"] = s.m.abs()
    s["s_tr"] = np.where(s.home_trails, s.spread, -s.spread)  # spread = home expected margin here
    s["y"] = np.where(s.home_trails, s.home_won, 1 - s.home_won)
    d = s[s.el.between(600, 2640) & s.deficit.between(1, 35)].copy()
    d["tau"] = (2880 - d.el) / 2880

    def probit(x):
        return sm.Probit(x.y.values, feats(x.deficit, x.s_tr, x.tau)).fit(disp=0)

    def calib(x, p):
        x = x.assign(p=p)
        bins = [0, .02, .05, .10, .15, .20, .30, .40, .50, .60, 1.0]
        x["bin"] = pd.cut(x.p, bins, include_lowest=True)
        out = []
        for b, g in x.groupby("bin", observed=True):
            gg = g.drop_duplicates("eid")  # one state per game per bin for the CI
            lo, hi = wilson(int(gg.y.sum()), len(gg))
            out.append({"bin": str(b), "states": len(g), "games": len(gg), "model": round(g.p.mean(), 4),
                        "realised": round(g.y.mean(), 4), "realised_1pg": round(gg.y.mean(), 4),
                        "ci_1pg": [round(lo, 4), round(hi, 4)]})
        return out

    win = lambda x: x[x.el.between(EL_LO, EL_HI) & x.deficit.between(D_LO, D_HI)]
    tr, te = d[d.season <= 2023], d[d.season == 2024]
    m1 = probit(tr)
    te_w = win(te)
    p_te = m1.predict(feats(te_w.deficit, te_w.s_tr, te_w.tau))
    ll = lambda y, p: float(-np.mean(y * np.log(np.clip(p, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1))))
    # no-spread comparison (shows the strength term matters out of sample)
    m0 = sm.Probit(tr.y.values, feats(tr.deficit, 0 * tr.s_tr, tr.tau)[:, [0, 1, 3, 5, 6]]).fit(disp=0)
    p0 = m0.predict(feats(te_w.deficit, 0 * te_w.s_tr, te_w.tau)[:, [0, 1, 3, 5, 6]])
    # empirical neighbour estimate OOS (train-season neighbours) on a 1-per-game-per-period sample
    smp = te_w.sort_values("el").groupby(["eid", (te_w.el // 720)]).head(1)
    emp = [empirical(tr, r.deficit, r.s_tr, r.el)[0] for r in smp.itertuples()]
    smp = smp.assign(emp=emp, p=m1.predict(feats(smp.deficit, smp.s_tr, smp.tau))).dropna(subset=["emp"])
    # by period x deficit band
    te_w = te_w.assign(p=p_te, per=(te_w.el // 720 + 1).clip(upper=4),
                       dband=pd.cut(te_w.deficit, [5, 10, 15, 20, 25]))
    cell = []
    for (pr, db), g in te_w.groupby(["per", "dband"], observed=True):
        gg = g.drop_duplicates("eid")
        cell.append({"period": int(pr), "deficit": str(db), "states": len(g), "games": len(gg),
                     "model": round(g.p.mean(), 4), "realised": round(g.y.mean(), 4),
                     "ci_1pg": [round(v, 4) for v in wilson(int(gg.y.sum()), len(gg))]})
    m = probit(d)
    # pregame: P(home win) = Phi(hm / sigma), for the Kalshi-mid -> spread fallback
    g = s.drop_duplicates("eid")
    hm = np.where(g.home_trails, g.s_tr, -g.s_tr)
    pm = sm.Probit(g.home_won.values, hm.reshape(-1, 1)).fit(disp=0)
    out = {
        "trial": "L37", "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": "data/nba_comebacks/states.parquet seasons 2015-16..2024-25 (regular + playoffs)",
        "features": "[1, d/sqrt(tau), s*sqrt(tau), d, s, tau, (d/sqrt(tau))^2]; d deficit, s trailer pregame "
                    "expected margin, tau = (2880-el)/2880; probit; fit el 600..2640, d 1..35",
        "coef": [float(x) for x in m.params], "n_states": int(len(d)), "n_games": int(d.eid.nunique()),
        "coef_le2023": [float(x) for x in m1.params],
        "pregame_sigma": float(1 / pm.params[0]),
        "oos": {"fit": "<= 2023-24", "test": "2024-25, window el 720..2520, deficit 6..25",
                "states": int(len(te_w)), "games": int(te_w.eid.nunique()),
                "logloss_model": ll(te_w.y.values, p_te), "logloss_no_spread": ll(te_w.y.values, p0),
                "brier_model": float(np.mean((te_w.y - p_te) ** 2)), "brier_no_spread": float(np.mean((te_w.y - p0) ** 2)),
                "mean_model": float(np.mean(p_te)), "mean_realised": float(te_w.y.mean()),
                "calibration": calib(te_w.drop(columns="p"), p_te), "by_period_deficit": cell,
                "empirical_vs_model_1pgpp": {"n": int(len(smp)), "mean_emp": float(smp.emp.mean()),
                                             "mean_model": float(smp.p.mean()), "realised": float(smp.y.mean()),
                                             "logloss_emp": ll(smp.y.values, smp.emp.clip(.005, .995).values),
                                             "logloss_model": ll(smp.y.values, smp.p.values)}},
    }
    FIT.parent.mkdir(exist_ok=True)
    FIT.write_text(json.dumps(out, indent=1))
    d[d.el.between(EL_LO - 180, EL_HI + 180)][["eid", "season", "el", "deficit", "s_tr", "y"]].astype(
        {"el": "int16", "deficit": "int16", "s_tr": "float32", "y": "int8", "season": "int16"}).to_parquet(STATES, index=False)
    o = out["oos"]
    print(f"fit n_states {out['n_states']} games {out['n_games']}; pregame sigma {out['pregame_sigma']:.2f}")
    print(f"OOS 2024-25 window: {o['states']} states / {o['games']} games; logloss model {o['logloss_model']:.4f} "
          f"vs no-spread {o['logloss_no_spread']:.4f}; mean model {o['mean_model']:.4f} realised {o['mean_realised']:.4f}")
    print(pd.DataFrame(o["calibration"]).to_string(index=False))
    print(pd.DataFrame(cell).to_string(index=False))
    print("empirical vs model (1 per game-period):", o["empirical_vs_model_1pgpp"])
    return 0


# ----------------------------------------------------------------------------------------------- live
def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def jl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, default=lambda v: None if v != v else float(v)) + "\n")


def fee(p: float) -> float:
    return 0.07 * p * (1 - p)


def get(url: str, **params) -> dict:
    r = requests.get(url, params=params or None, timeout=15)
    r.raise_for_status()
    return r.json()


def game_state(e: dict) -> dict:
    c = e["competitions"][0]
    st = c["status"]
    clock = st.get("clock")
    if clock is None:
        dc = st.get("displayClock", "0:00")
        mm, ss = dc.split(":") if ":" in dc else ("0", dc)
        clock = float(mm) * 60 + float(ss)
    t = {x["homeAway"]: x for x in c["competitors"]}
    code = lambda s: ESPN2K.get(t[s]["team"]["abbreviation"], t[s]["team"]["abbreviation"])
    start = datetime.fromisoformat(e["date"].replace("Z", "+00:00"))
    odds = (c.get("odds") or [{}])[0]
    per = int(st.get("period") or 0)
    return {"eid": e["id"], "game": e.get("shortName"), "state": st["type"]["state"],
            "completed": bool(st["type"].get("completed")), "period": per, "clock_s": float(clock),
            "el": (per - 1) * 720 + (720 - float(clock)) if 1 <= per <= 4 else None,
            "home": code("home"), "away": code("away"),
            "hs": int(t["home"].get("score") or 0), "as": int(t["away"].get("score") or 0),
            "winner": next((code(s) for s in ("home", "away") if t[s].get("winner")), None),
            "start": start, "home_spread": odds.get("spread"), "odds_details": odds.get("details"),
            "event": f"KXNBAGAME-{start.astimezone(ET).strftime('%y%b%d').upper()}{code('away')}{code('home')}"}


def kalshi_book() -> dict:
    from sports import live
    k = live.kalshi("nba")
    return {r.ticker: {"ask": r.yes_ask, "bid": r.yes_bid} for r in k.itertuples()}


def load_cache() -> dict:
    try:
        return json.loads(CACHE.read_text())
    except Exception:  # noqa: BLE001
        return {}


def update_pregame(g: dict, cache: dict, book: dict | None, fitd: dict) -> None:
    c = cache.setdefault(g["eid"], {"game": g["game"]})
    if c.get("hm") is None and g["state"] == "pre" and g["home_spread"] is not None:  # live odds never used
        c.update(hm=-float(g["home_spread"]), src="espn_scoreboard", details=g["odds_details"], at=now())
    if g["state"] == "pre" and book:
        hq = book.get(f"{g['event']}-{g['home']}", {})
        if hq.get("ask") == hq.get("ask") and hq.get("bid") == hq.get("bid") and hq.get("ask") and hq.get("bid"):
            c.update(k_pre_mid_home=(hq["ask"] + hq["bid"]) / 2, k_pre_at=now())
    if c.get("hm") is None and g["state"] == "in" and not c.get("summary_tried"):
        c["summary_tried"] = True
        try:
            pc = (get(f"{ESPN}/summary", event=g["eid"]).get("pickcenter") or [{}])[0]
            if pc.get("spread") is not None:
                c.update(hm=-float(pc["spread"]), src="espn_pickcenter", details=pc.get("details"), at=now())
        except Exception as ex:  # noqa: BLE001
            c["summary_err"] = str(ex)[:200]
    if c.get("hm") is None and c.get("k_pre_mid_home"):
        p = min(max(c["k_pre_mid_home"], 0.02), 0.98)
        c.update(hm=fitd["pregame_sigma"] * float(norm.ppf(p)), src="kalshi_pregame_mid")


def evaluate(g: dict, cache: dict, book: dict, fitd: dict, st: pd.DataFrame) -> dict:
    mar = g["hs"] - g["as"]
    hm = cache.get(g["eid"], {}).get("hm")
    row = {"ts": now(), "eid": g["eid"], "game": g["game"], "period": g["period"], "clock_s": g["clock_s"],
           "el": g["el"], "hs": g["hs"], "as": g["as"], "home": g["home"], "away": g["away"],
           "home_exp_margin": hm, "spread_src": cache.get(g["eid"], {}).get("src")}
    if mar == 0 or g["el"] is None:
        row["note"] = "tied" if mar == 0 else "OT"
        return row
    tr, ld = (g["home"], g["away"]) if mar < 0 else (g["away"], g["home"])
    d = abs(mar)
    s = None if hm is None else (hm if mar < 0 else -hm)
    row.update(trailer=tr, leader=ld, deficit=d, s_tr=s,
               in_window=bool(EL_LO <= g["el"] <= EL_HI and g["period"] in (2, 3, 4) and D_LO <= d <= D_HI))
    tq, lq = book.get(f"{g['event']}-{tr}", {}), book.get(f"{g['event']}-{ld}", {})
    row.update(tr_ask=tq.get("ask"), tr_bid=tq.get("bid"), ld_ask=lq.get("ask"), ld_bid=lq.get("bid"))
    if s is not None:
        p = float(p_trailer(fitd, [d], [s], [g["el"]])[0])
        pe, n, (lo, hi) = empirical(st, d, s, g["el"])
        row.update(p_tr=round(p, 4), emp_tr=None if pe != pe else round(pe, 4), emp_n=n,
                   emp_ci=[None if lo != lo else round(lo, 4), None if hi != hi else round(hi, 4)])
        for side, ps, pe_s, q in (("tr", p, pe, tq), ("ld", 1 - p, (1 - pe) if pe == pe else pe, lq)):
            a = q.get("ask")
            if a is None or a != a or not (0.01 <= a <= 0.99):
                continue
            row[f"edge_{side}"] = round(ps - a - fee(a), 4)
            row[f"emp_edge_{side}"] = None if pe_s != pe_s else round(pe_s - a - fee(a), 4)
    return row


def entry_side(row: dict) -> str | None:
    if not row.get("in_window") or row.get("p_tr") is None:
        return None
    ok = [s for s in ("tr", "ld") if (row.get(f"edge_{s}") or -1) >= EDGE_MIN and row.get("emp_n", 0) >= EMP_MIN_N
          and (row.get(f"emp_edge_{s}") if row.get(f"emp_edge_{s}") is not None else -1) > 0]
    return max(ok, key=lambda s: row[f"edge_{s}"]) if ok else None


def scoreboard(day: str | None = None) -> list[dict]:
    sb = get(f"{ESPN}/scoreboard", **({"dates": day} if day else {}))
    return [game_state(e) for e in sb.get("events", [])]


def settle(games: dict, msgs: list) -> None:
    done = {r["eid"] for r in jl(RESULTS)}
    need = ({r["eid"]: r for r in jl(TICKS)} | {r["eid"]: r for r in jl(ENTRIES)})
    for eid, r in need.items():
        if eid in done:
            continue
        g = games.get(eid)
        if g is None:
            try:
                s = get(f"{ESPN}/summary", event=eid)
                hc = s["header"]["competitions"][0]
                g = game_state({"id": eid, "date": hc["date"], "shortName": r["game"], "competitions": [hc]})
            except Exception as ex:  # noqa: BLE001
                msgs.append(f"settle lookup failed {eid}: {ex}")
                continue
        if not g["completed"] or g["winner"] is None:
            continue
        kr = None
        try:
            kr = get(f"{KALSHI}/markets/{g['event']}-{g['home']}")["market"].get("result") or None
        except Exception:  # noqa: BLE001
            pass
        append(RESULTS, {"ts": now(), "eid": eid, "game": r["game"], "winner": g["winner"], "home": g["home"],
                         "final_home": g["hs"], "final_away": g["as"], "kalshi_home_result": kr})
        msgs.append(f"SETTLED {r['game']} winner {g['winner']} {g['hs']}-{g['as']} (kalshi home {kr})")


def tick(mock: str | None = None) -> int:
    results = {r["eid"] for r in jl(RESULTS)}
    open_e = [r for r in jl(ENTRIES) if r["eid"] not in results]
    h = datetime.now(ET).hour
    if not mock and not open_e and 3 <= h < 11:
        return 0
    games = {g["eid"]: g for g in (map(game_state, json.loads(Path(mock).read_text())["events"]) if mock else scoreboard())}
    nowu = datetime.now(timezone.utc)
    live_ = [g for g in games.values() if g["state"] == "in"]
    soon = [g for g in games.values() if g["state"] == "pre" and g["start"] - nowu < timedelta(minutes=75)]
    msgs: list[str] = []
    fitd = json.loads(FIT.read_text())
    cache = load_cache()
    for g in games.values():  # cache ESPN pregame spreads (cheap; no Kalshi call)
        if g["state"] == "pre":
            update_pregame(g, cache, None, fitd)
    if live_ or soon:
        book = kalshi_book()
        for g in soon:
            update_pregame(g, cache, book, fitd)
        if live_:
            st = pd.read_parquet(STATES)
            entered = {r["eid"] for r in jl(ENTRIES)}
            for g in live_:
                update_pregame(g, cache, book, fitd)
                row = evaluate(g, cache, book, fitd, st)
                if not row.get("in_window"):
                    continue
                append(TICKS, row)
                side = entry_side(row)
                if side and g["eid"] not in entered:
                    a = row[f"{side}_ask"]
                    e = {**row, "side": side, "team": row["trailer" if side == "tr" else "leader"], "ask": a,
                         "fee": round(fee(a), 5), "fee_rounded": math.ceil(round(100 * fee(a), 9)) / 100,
                         "p_model": row["p_tr"] if side == "tr" else round(1 - row["p_tr"], 4),
                         "edge": row[f"edge_{side}"], "contracts": 1}
                    append(ENTRIES, e)
                    entered.add(g["eid"])
                    msgs.append(f"ENTRY {g['game']} Q{g['period']} {g['clock_s']:.0f}s down {row['deficit']} "
                                f"{e['team']} ({side}) ask {a:.2f} model {e['p_model']:.3f} edge {e['edge']:+.3f} "
                                f"emp {row['emp_tr']} n {row['emp_n']}")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, indent=1))
    settle(games, msgs)
    if msgs:
        print("\n".join(msgs))
    return 0


def show() -> int:
    fitd = json.loads(FIT.read_text())
    games = scoreboard()
    cache = load_cache()
    book = kalshi_book()
    st = pd.read_parquet(STATES)
    rows = []
    for g in games:
        update_pregame(g, cache, book, fitd)
        hm = cache.get(g["eid"], {}).get("hm")
        if g["state"] != "in":
            rows.append({"game": g["game"], "state": g["state"] + (" " + g["start"].astimezone(ET).strftime("%H:%M ET")
                         if g["state"] == "pre" else ""), "home_exp_margin": hm})
            continue
        r = evaluate(g, cache, book, fitd, st)
        ci = r.get("emp_ci") or [None, None]
        rows.append({"game": g["game"], "state": f"Q{g['period']} {int(g['clock_s'] // 60)}:{int(g['clock_s'] % 60):02d}",
                     "score": f"{g['away']} {g['as']}-{g['hs']} {g['home']}", "home_exp_margin": hm,
                     "trailer": r.get("trailer"), "down": r.get("deficit"), "s_tr": r.get("s_tr"),
                     "P_tr_model": r.get("p_tr"), "P_tr_emp": r.get("emp_tr"), "emp_n": r.get("emp_n"),
                     "emp_ci": f"{ci[0]}-{ci[1]}" if ci[0] is not None else None,
                     "tr_bid/ask": f"{r.get('tr_bid')}/{r.get('tr_ask')}", "ld_bid/ask": f"{r.get('ld_bid')}/{r.get('ld_ask')}",
                     "edge_tr": r.get("edge_tr"), "edge_ld": r.get("edge_ld"),
                     "window": r.get("in_window"), "WOULD_ENTER": entry_side(r)})
    if not rows:
        print("no NBA games on today's ESPN scoreboard")
        return 0
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nedge = model P - ask - 0.07 p(1-p). Entry rule (L37): window Q2..Q4 6:00, down 6-25, edge >= .05, "
          "empirical (n>=30) same sign. Paper only.")
    return 0


def report() -> int:
    res = {r["eid"]: r for r in jl(RESULTS)}
    ents = jl(ENTRIES)
    print("L37 NBA mid-game forward paper (frozen 2026-10-26; edge >= .05 + empirical agrees; 1 contract, taker)")
    done = [(e, res[e["eid"]]) for e in ents if e["eid"] in res]
    print(f"entries {len(ents)} | settled {len(done)} | open {len(ents) - len(done)}")
    if done:
        won = [r["winner"] == e["team"] for e, r in done]
        pnl = [(1.0 if w else 0.0) - e["ask"] - e["fee"] for (e, _), w in zip(done, won)]
        pnl_r = [(1.0 if w else 0.0) - e["ask"] - e["fee_rounded"] for (e, _), w in zip(done, won)]
        n = len(done)
        sd = float(np.std(pnl, ddof=1)) if n > 1 else float("nan")
        print(f"W-L {sum(won)}-{n - sum(won)} | P&L/contract {np.mean(pnl):+.4f} (SE {sd / math.sqrt(n):.4f}) total "
              f"{sum(pnl):+.3f} | cent-rounded fee {np.mean(pnl_r):+.4f} | mean edge at entry "
              f"{np.mean([e['edge'] for e, _ in done]):+.4f} | mean model {np.mean([e['p_model'] for e, _ in done]):.3f} "
              f"mean ask {np.mean([e['ask'] for e, _ in done]):.3f} realised {np.mean(won):.3f}")
        for (e, r), w, p in zip(done, won, pnl):
            print(f"  {e['ts'][:16]} {e['game']:<12} Q{e['period']} {e['clock_s']:>4.0f}s down {e['deficit']:>2} "
                  f"{e['team']}({e['side']}) ask {e['ask']:.2f} model {e['p_model']:.3f} edge {e['edge']:+.3f} -> "
                  f"{'W' if w else 'L'} {p:+.3f}")
    t = pd.DataFrame(jl(TICKS))
    if t.empty:
        print("no window ticks yet")
        return 0
    t = t[t.eid.isin(res)].copy()
    print(f"window ticks {len(jl(TICKS))} over {pd.DataFrame(jl(TICKS)).eid.nunique()} games; settled games {t.eid.nunique()}")
    if t.empty:
        return 0
    t["y"] = [res[e]["winner"] == tr for e, tr in zip(t.eid, t.trailer)]
    t = t.dropna(subset=["tr_ask", "tr_bid"])
    t["mid"] = (t.tr_ask.astype(float) + t.tr_bid.astype(float)) / 2
    t["bin"] = pd.cut(t.mid, [0, .05, .10, .15, .20, .30, .40, .50, 1.0], include_lowest=True)
    out = []
    for b, g in t.groupby("bin", observed=True):
        g1 = g.sort_values("ts").drop_duplicates("eid")
        lo, hi = wilson(int(g1.y.sum()), len(g1))
        out.append({"trailer_mid": str(b), "games": len(g1), "mean_mid": round(g1.mid.mean(), 3),
                    "mean_model": round(g1.p_tr.astype(float).mean(), 3) if g1.p_tr.notna().any() else None,
                    "realised": round(g1.y.mean(), 3), "ci95": f"{lo:.3f}-{hi:.3f}"})
    print("Calibration (one tick per game per bin, first in bin): trailer Kalshi mid vs model vs realised")
    print(pd.DataFrame(out).to_string(index=False))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    if cmd == "fit":
        raise SystemExit(fit())
    if cmd == "tick":
        raise SystemExit(tick(sys.argv[sys.argv.index("--mock") + 1] if "--mock" in sys.argv else None))
    if cmd == "report":
        raise SystemExit(report())
    raise SystemExit(show())
