"""LIVE path for the frozen NBA simple model (scripts/nba_simple_model.py). PREDICTION ONLY -- never scores anything.

    ../.venv_quant/bin/python scripts/nba_simple_live.py [--date YYYY-MM-DD] [--json] [--out "Name,Name"]
                                                         [--no-refresh] [--no-cache]
    (also reached via `nba_simple_model.py today ...`)

    from nba_simple_live import predict; rows = predict("2026-10-26")      # list[dict], schema below

Nothing here changes the frozen model: coefficients, BPM-lite weights, stack fits, half-lives, ridge settings and the
feature formulas all come from nba_simple_model.py / results/nba_simple_model_frozen.json. What this file adds is the
INPUT plumbing the sealed backtest cannot have (it runs on history):

  PAINT    ESPN boxscore team stat `pointsInPaint` (summary endpoint; finished games cached on disk). Checked on 300
           random 2024-25 games: equal to stats.nba.com PTS_PAINT on 98.5% of team-games, max |diff| 2 pts, corr .9997.
           stats.nba.com is only tried for team-games ESPN lacks. A game whose teams have no paint history gets
           paint_source "missing" and a warning: S1 is then 0 (the frozen fillna), never silently.
  ROSTER   ESPN team roster endpoint (as of NOW). full roster = (players in the team's last 10 games who are still on
           the roster) + (roster players acquired from another team, last NBA game <= 400 d ago, not listed OUT)
           + (roster players with no NBA history, ONLY before the team's first game of the season; rating =
           replacement, the frozen default). Previous-game players who have since left the roster are dropped.
  INJURY   ESPN league injury feed (as of NOW). Out / Doubtful -> removed from tonight's set (kept in the full roster,
           so the frozen absence term dAbs/oAbs prices the loss). Questionable / Day-To-Day / GTD -> kept, and a
           second prediction with them removed is reported (`*_q_out`). Probable -> plays. `--out` = manual OUT.
  STRICT   tonight's set = players who played the team's previous game (the pre-registered STRICT rule), on the
           current roster, minus OUT/Doubtful/manual. Before a team's first game of the season the previous game is
           last April's, so the set is instead the full roster minus OUT (warned).
  PREREG   `model_*_prereg` = the exact pre-registered STRICT inputs (no roster, no injuries) -- what the sealed test
           would compute for the same date. The gap to `model_*` is what the live plumbing moved.

JSON / predict() schema (one dict per game on `date`, ET; stable -- add fields, never rename):
  date, event_id, start_utc, status, home, away,
  model_total, model_margin (home points minus away), model_home_wp (Phi(margin / frozen sigma)),
  model_total_q_out, model_margin_q_out, model_home_wp_q_out,       (Questionable players also out)
  model_total_prereg, model_margin_prereg,
  model_total_avail, model_margin_avail, model_home_wp_avail  (tonight = full roster minus OUT/Doubtful, i.e. the
      injury report decides; uses the frozen M2_act/T2_act fits, whose backtest used who-actually-played -- a
      lookahead proxy -- so this is NOT the pre-registered/sealed model; use it to see what STRICT's "missed last game
      = out" rule costs, e.g. stars rested on a back-to-back),
  n_out_home, n_out_away  (players removed by the injury feed or --out who are on the team's full roster),
  out_home, out_away, questionable_home, questionable_away, absent_home, absent_away (top-5-minutes players not in
  tonight's set for any reason), paint_source, market_total, market_total_open, market_spread_home, market_provider
  (ESPN core odds, first listed book; informational only, never a model input), warnings (list of strings).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import pickle
import re
import sys
import time
from datetime import date as Date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from scipy.stats import norm

import nba_simple_model as nsm
from nba_player_sim import ridge_preds
from sports import espn
from sports.http import fetch

LIVE = nsm.OUTD / "live"
SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
OUT_STATUS = {"out", "doubtful", "suspension", "suspended", "not with team", "inactive"}
Q_STATUS = {"questionable", "day-to-day", "day to day", "game-time decision", "gtd"}
ROSTER_TTL_H = 3.0
ACQ_MAX_DAYS = 400


def season_of(d: Date) -> int:
    return d.year + 1 if d.month >= 8 else d.year


def log(msg: str, quiet: bool) -> None:
    if not quiet:
        print(msg, file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------------ data freshness
def refresh_season(season: int, d: Date, quiet: bool) -> list[str]:
    """Incremental ESPN update of games_{season}/box_{season}: rescan scoreboards from the earliest not-final game
    before `d` (or d-2) through d, then pull box scores for final games that have none. Writes only if changed."""
    import collect
    notes = []
    gp, bp = nsm.ESPN / f"games_{season}.parquet", nsm.ESPN / f"box_{season}.parquet"
    g = pd.read_parquet(gp) if gp.exists() else pd.DataFrame()
    if len(g):
        gd = g.start_utc.dt.tz_convert("US/Eastern").dt.date
        stale = g[(gd < d) & (g.status != "STATUS_FINAL") & ~g.status.isin(["STATUS_POSTPONED", "STATUS_CANCELED"])]
        start = min([d - timedelta(days=2)] + list(gd[stale.index]))
    else:
        start = Date(season - 1, 10, 1)
    rows = []
    day = start
    while day <= d:
        for e in fetch(SITE + "/scoreboard", {"dates": day.strftime("%Y%m%d"), "limit": 1000}, cache=False).get("events", []):
            rows.append(espn.parse_event("nba", e))
        day += timedelta(days=1)
    new = pd.DataFrame(rows)
    if len(new):
        new = new[new.season == season]
    changed = False
    if len(new):
        if len(g):
            new = new.astype({c: g[c].dtype for c in new.columns if c in g.columns}, errors="ignore")
            key = ["event_id", "status", "home_score", "away_score", "start_utc"]
            old = g[g.event_id.isin(new.event_id)].set_index("event_id")[key[1:]].sort_index()
            nw = new.set_index("event_id")[key[1:]].sort_index()
            changed = len(old) != len(nw) or not old.reindex(nw.index).astype(str).equals(nw.astype(str))
            g = pd.concat([g[~g.event_id.isin(new.event_id)], new], ignore_index=True)
        else:
            g, changed = new, True
    if changed:
        g = g.sort_values("start_utc").reset_index(drop=True)
        with contextlib.redirect_stdout(sys.stderr):          # keep stdout clean for --json
            collect.save(g.copy(), gp)
        notes.append(f"games_{season} updated")
    b = pd.read_parquet(bp) if bp.exists() else pd.DataFrame(columns=["event_id"])
    gd = g.start_utc.dt.tz_convert("US/Eastern").dt.date
    need = g[(g.status == "STATUS_FINAL") & (gd < d) & ~g.event_id.astype(str).isin(set(b.event_id.astype(str)))]
    if len(need):
        add = []
        for eid in need.event_id.astype(str):
            add.extend(espn.box("nba", eid, cache=True))
        if add:
            b = pd.concat([b, pd.DataFrame(add)], ignore_index=True)
            with contextlib.redirect_stdout(sys.stderr):
                collect.save(b, bp)
            notes.append(f"box_{season} +{len(need)} games")
    log(f"  refresh {season}: {', '.join(notes) or 'up to date'}", quiet)
    return notes


# ---------------------------------------------------------------------------------------------------------- paint
def load_paint_live(g: pd.DataFrame, quiet: bool) -> tuple[pd.DataFrame, dict]:
    """(event_id, team, paint, src) for every FINAL game in g. ESPN boxscore first, stats.nba.com for the rest."""
    LIVE.mkdir(parents=True, exist_ok=True)
    cp = LIVE / "paint_espn.parquet"
    cache = pd.read_parquet(cp) if cp.exists() else pd.DataFrame(columns=["event_id", "team_id", "paint"])
    fin = g[g.status == "STATUS_FINAL"]
    have = set(cache.event_id.astype(str))
    todo = [e for e in fin.event_id.astype(str) if e not in have]
    add = []
    for eid in todo:
        try:
            s = espn.summary("nba", eid, cache=True)
        except Exception:
            continue
        for t in s.get("boxscore", {}).get("teams", []) or []:
            st = {x.get("name"): x.get("displayValue") for x in t.get("statistics", [])}
            v = pd.to_numeric(st.get("pointsInPaint"), errors="coerce")
            add.append({"event_id": eid, "team_id": str(t.get("team", {}).get("id")), "paint": float(v)})
    if add:
        cache = pd.concat([cache, pd.DataFrame(add)], ignore_index=True).drop_duplicates(["event_id", "team_id"], keep="last")
        cache.to_parquet(cp, index=False)
    cache = cache.assign(event_id=cache.event_id.astype(str), team_id=cache.team_id.astype(str))
    rows = []
    for side in ("home", "away"):
        x = fin[["event_id", f"{side}_team_id", side]].rename(columns={f"{side}_team_id": "team_id", side: "team"})
        x = x.assign(event_id=x.event_id.astype(str), team_id=x.team_id.astype(str))
        rows.append(x.merge(cache, on=["event_id", "team_id"], how="left"))
    p = pd.concat(rows, ignore_index=True)
    p["src"] = np.where(p.paint.notna(), "espn", None)
    miss = p.paint.isna()
    if miss.any():                               # stats.nba.com for what ESPN lacks (often times out from this Mac)
        t0 = time.time()
        sub = g[g.event_id.astype(str).isin(set(p.event_id[miss]))]
        try:
            alt = nsm.load_paint(sub, live=True).assign(event_id=lambda z: z.event_id.astype(str))
            p = p.merge(alt.rename(columns={"paint": "paint_nba"}), on=["event_id", "team"], how="left")
            fill = p.paint.isna() & p.paint_nba.notna()
            p.loc[fill, "paint"], p.loc[fill, "src"] = p.paint_nba[fill], "stats.nba.com"
            p = p.drop(columns="paint_nba")
        except Exception as ex:
            log(f"  stats.nba.com paint fallback failed: {str(ex)[:80]}", quiet)
        log(f"  paint: {int(miss.sum())} team-games not on ESPN; stats.nba.com tried ({time.time() - t0:.0f}s)", quiet)
    counts = p.src.fillna("missing").value_counts().to_dict()
    log(f"  PAINT SOURCE: {counts} team-games (ESPN boxscore pointsInPaint primary)", quiet)
    p = p.dropna(subset=["paint"])
    return p[["event_id", "team", "paint", "src"]], counts


# -------------------------------------------------------------------------------------------- rosters, injuries
def rosters(team_ids, quiet: bool) -> dict:
    """team_id -> set(athlete_id) from ESPN's roster endpoint, cached ROSTER_TTL_H hours."""
    LIVE.mkdir(parents=True, exist_ok=True)
    cp = LIVE / "rosters.json"
    c = {}
    try:
        c = json.loads(cp.read_text())
    except Exception:
        pass
    fresh = c.get("ts", 0) > time.time() - ROSTER_TTL_H * 3600
    out = {k: set(v) for k, v in c.get("teams", {}).items()} if fresh else {}
    miss = [t for t in team_ids if str(t) not in out]
    for t in miss:
        try:
            j = fetch(f"{SITE}/teams/{t}/roster", cache=False, retries=3, timeout=20)
            out[str(t)] = {str(a["id"]) for a in j.get("athletes", [])}
        except Exception as ex:
            log(f"  roster {t} failed: {str(ex)[:60]}", quiet)
    if miss:
        try:
            cp.write_text(json.dumps({"ts": time.time() if not fresh else c["ts"],
                                      "teams": {k: sorted(v) for k, v in out.items()}}))
        except Exception:
            pass
    return out


def injuries(quiet: bool) -> tuple[dict, bool]:
    """team_id -> {athlete_id: (name, 'OUT'|'Q', raw status)} from ESPN's league injury feed (as of now)."""
    try:
        j = fetch(f"{SITE}/injuries", cache=False, retries=3, timeout=20)
    except Exception as ex:
        log(f"  INJURY FEED FAILED: {str(ex)[:80]}", quiet)
        return {}, False
    out = {}
    for t in j.get("injuries", []):
        tid = str(t.get("id"))
        for i in t.get("injuries", []):
            a = i.get("athlete", {}) or {}
            m = None
            for l in a.get("links", []) or []:
                m = m or re.search(r"/id/(\d+)", l.get("href", ""))
            if not m:
                continue
            s = (i.get("status") or "").strip().lower()
            fs = (((i.get("details") or {}).get("fantasyStatus") or {}).get("abbreviation") or "").upper()
            if s in OUT_STATUS or fs == "OUT":
                k = "OUT"
            elif s in Q_STATUS or fs == "GTD":
                k = "Q"
            else:
                continue                                  # probable / active
            out.setdefault(tid, {})[m.group(1)] = (a.get("displayName"), k, i.get("status"))
    return out, True


def market_lines(eids) -> dict:
    """ESPN core odds per event (info only, never used by the model): current + open total/spread of the first book.
    The scoreboard drops odds once a game tips, the core odds endpoint keeps them."""
    out = {}
    for eid in eids:
        try:
            ls = espn.lines("nba", str(eid), cache=False)
        except Exception:
            ls = []
        ls = [x for x in ls if x.get("provider") and str(x.get("provider")).lower() not in nsm.NOT_BOOKS]
        if not ls:
            continue
        x = ls[0]
        num = lambda v: None if v is None or not np.isfinite(v) else float(v)
        out[str(eid)] = {"market_total": num(x.get("total")), "market_total_open": num(x.get("total_open")),
                         "market_spread_home": num(x.get("spread")), "market_provider": x.get("provider")}
    return out


# --------------------------------------------------------------------------------------------- player state (live)
def player_state(gp: pd.DataFrame, box: pd.DataFrame):
    """ST/TH exactly as nsm.player_pass leaves them after processing every game in gp (all dated before tonight)."""
    bx = {}
    for r in box.itertuples(index=False):
        bx.setdefault((r.event_id, r.team), []).append(r)
    ST, TH = {}, {}
    dn = np.array([d.toordinal() for d in gp.date])
    for d in np.unique(dn):
        for i in np.where(dn == d)[0]:
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
    return ST, TH


def strength_fn(ST, bt, d):
    """Same rate()/strength() as nsm.player_pass."""
    beta, repl, rppm = np.array(bt["beta"]), np.array(bt["repl48"]) / 48, bt["repl_ppm"]

    def rate(a):
        s = ST.get(a)
        if s is None:
            return bt["b0"] + repl @ beta * 48, rppm, 15.0
        dec = 0.5 ** ((d - s["t"]) / nsm.HP)
        mm = s["m"] * dec
        rt = (s["s"] * dec + nsm.KSHRINK * repl) / (mm + nsm.KSHRINK)
        ppm = (s["s"][0] * dec + nsm.KSHRINK * rppm) / (mm + nsm.KSHRINK)
        return bt["b0"] + rt @ beta * 48, ppm, s["em"]

    def strength(S):
        if not S:
            return np.nan, np.nan
        v = np.array([rate(a) for a in S])
        m = nsm.minutes_proj(v[:, 2].copy())
        return float((m * v[:, 0]).sum() / 240), float((m * v[:, 1]).sum())
    return strength


# --------------------------------------------------------------------------------------------------- build (cached)
def _sig(d: Date, seasons) -> str:
    h = hashlib.sha256(str(d).encode())
    for y in seasons:
        g = pd.read_parquet(nsm.ESPN / f"games_{y}.parquet")
        gd = g.start_utc.dt.tz_convert("US/Eastern").dt.date
        x = g[gd <= d].sort_values("event_id")
        fin = x[gd[x.index] < d]
        h.update(pd.util.hash_pandas_object(fin[["event_id", "status", "home_score", "away_score"]].astype(str),
                                            index=False).values.tobytes())
        h.update(",".join(sorted(x[gd[x.index] == d].event_id.astype(str))).encode())
        bp = nsm.ESPN / f"box_{y}.parquet"
        h.update(str(pd.read_parquet(bp, columns=["event_id"]).event_id.nunique() if bp.exists() else 0).encode())
    h.update(str(nsm.FROZEN.stat().st_mtime).encode())
    h.update(Path(__file__).read_bytes())
    h.update(Path(nsm.__file__).read_bytes())
    return h.hexdigest()[:16]


def base_state(d: Date, seasons, fz, quiet: bool, use_cache=True) -> dict:
    """Everything that depends only on games before d: today's rows with mu0/tot0/style, ST/TH, paint coverage."""
    LIVE.mkdir(parents=True, exist_ok=True)
    sig = _sig(d, seasons)
    cp = LIVE / f"state_{d}.pkl"
    if use_cache and cp.exists():
        try:
            st = pickle.loads(cp.read_bytes())
            if st.get("sig") == sig:
                log(f"  state cache hit ({cp.name})", quiet)
                return st
        except Exception:
            pass
    t0 = time.time()
    g = nsm.load_games(seasons, d)
    box = nsm.load_box(g, seasons)
    g["mu0"] = ridge_preds(g, nsm.H_TEAM, nsm.LAM, only_from=d.toordinal())   # identical rows for date d
    g["tot0"] = nsm.ridge_totals(g)
    paint, counts = load_paint_live(g, quiet)
    sf = nsm.style_pass(g, box, paint[["event_id", "team", "paint"]])
    past = g[g.date < d]
    ST, TH = player_state(past, box)
    names = dict(zip(box.athlete_id, box.name))
    today = g[g.date == d].merge(sf, on="event_id", how="left")
    cur = season_of(d)
    rs = past[(past.season == cur) & (past.season_type == "regular-season")]
    n_cur = pd.concat([rs.home, rs.away]).value_counts().to_dict()
    # paint coverage per team over its past games (what feeds opp_paint_share)
    pk = set(zip(paint.event_id.astype(str), paint.team))
    srcmap = dict(zip(zip(paint.event_id.astype(str), paint.team), paint.src))
    cov = {}
    for T in set(today.home) | set(today.away):
        ev = past[(past.home == T) | (past.away == T)]
        opp = [(str(r.event_id), r.away if r.home == T else r.home) for r in ev.itertuples()]
        srcs = [srcmap.get(k) for k in opp]
        cov[T] = (sum(s is not None for s in srcs), len(srcs), sorted({s for s in srcs if s}))
    st = {"sig": sig, "today": today, "ST": ST, "TH": TH, "names": names, "n_cur": n_cur, "paint_counts": counts,
          "paint_cov": cov, "built": datetime.now(timezone.utc).isoformat(), "secs": time.time() - t0}
    cp.write_bytes(pickle.dumps(st))
    for old in LIVE.glob("state_*.pkl"):
        if old != cp and old.stat().st_mtime < time.time() - 3 * 86400:
            old.unlink()
    log(f"  state built in {st['secs']:.0f}s ({len(past)} past games)", quiet)
    return st


# ------------------------------------------------------------------------------------------------------ predict
def _side_sets(T, tid, d, st, roster, inj, manual_out, warnings):
    ST, TH, names = st["ST"], st["TH"], st["names"]
    hist = TH.get(T, [])
    full_pre = {a for a in set().union(*hist[-10:]) if ST[a]["team"] == T} if hist else set()
    prev = hist[-1] if hist else set()
    R = roster.get(str(tid))
    ti = inj.get(str(tid), {})
    out_ids = {a for a, v in ti.items() if v[1] == "OUT"}
    q_ids = {a for a, v in ti.items() if v[1] == "Q"}
    man = {a for a in (R or set(full_pre) | set(prev)) if (names.get(a) or "").lower() in manual_out}
    opener = st["n_cur"].get(T, 0) == 0
    if R:
        acq = {a for a in R - full_pre if a in ST and ST[a]["team"] != T and d - ST[a]["t"] <= ACQ_MAX_DAYS
               and a not in out_ids}
        rook = {a for a in R if a not in ST} if opener else set()
        full = (full_pre & R) | acq | rook
        gone = sorted(names.get(a, a) for a in full_pre - R)
        if gone:
            warnings.append(f"{T}: dropped (no longer on roster): {', '.join(gone[:8])}{' ...' if len(gone) > 8 else ''}")
        if acq:
            warnings.append(f"{T}: added (acquired, prior rating): {', '.join(sorted(names.get(a, a) for a in acq))}")
    else:
        full = set(full_pre)
        warnings.append(f"{T}: ESPN roster unavailable -> last-10-games roster (may include departed players)")
    base = (full if opener else (prev & R if R else prev))
    if opener:
        warnings.append(f"{T}: first game of season -> tonight's set = current roster minus OUT (not last April's game)")
    sel = base - out_ids - man
    sel_q = sel - q_ids
    avail = full - out_ids - man                      # injury-report availability (ACTUAL proxy), Q players in
    nm = lambda S: sorted(names.get(a, ti.get(a, (a,))[0]) for a in S)
    starters = sorted(full, key=lambda a: -ST[a]["em"] if a in ST else 0)[:5]
    info = {"out": nm((out_ids | man) & (full | prev)), "n_out": len((out_ids | man) & (full | prev)),
            "questionable": nm(q_ids & (sel | avail)), "absent": nm([a for a in starters if a not in sel])}
    rested = [a for a in starters if a not in sel and a not in out_ids | q_ids | man]
    if rested:
        warnings.append(f"{T}: {', '.join(nm(rested))} missed the previous game but are NOT on the injury report: STRICT "
                        f"counts them OUT in model_*; model_*_avail counts them IN")
    return full_pre, prev, full, sel, sel_q, avail, info


def predict(date=None, refresh: bool = True, out_names=(), use_cache: bool = True, quiet: bool = True) -> list[dict]:
    """Frozen M2/T2 STRICT predictions for every NBA game on `date` (ET). See module docstring for the schema."""
    t_start = time.time()
    d = pd.Timestamp(date).date() if date else pd.Timestamp.now(tz="US/Eastern").date()
    now_et = pd.Timestamp.now(tz="US/Eastern").date()
    cur = season_of(d)
    seasons = [cur - 1, cur]
    assert all(y >= 2026 for y in seasons), "live path reads 2025-26 onward only"
    gwarn = []
    if refresh:
        try:
            refresh_season(cur, d, quiet)
        except Exception as ex:
            gwarn.append(f"ESPN refresh failed ({str(ex)[:60]}); using games/box on disk")
    fz = json.loads(nsm.FROZEN.read_text())
    st = base_state(d, seasons, fz, quiet, use_cache)
    t = st["today"]
    if t.empty:
        log(f"no games on {d}", quiet)
        return []
    if d != now_et:
        gwarn.append(f"roster + injury feeds are as of {now_et} (now), not {d}")
    tids = sorted(set(t.home_team_id.astype(str)) | set(t.away_team_id.astype(str)))
    roster = rosters(tids, quiet)
    inj, inj_ok = injuries(quiet)
    if not inj_ok:
        gwarn.append("ESPN injury feed unavailable: no injury exclusions applied (use --out)")
    odds = market_lines(t.event_id)
    manual = {s.strip().lower() for s in out_names if s.strip()}
    strength = strength_fn(st["ST"], fz["bpm"], d.toordinal())
    fits = fz["fits"]
    sig = fits["M2_strict"]["sigma"]
    res, matched = [], set()
    for _, r in t.iterrows():
        warn = list(gwarn)
        sets = {}
        for side in ("home", "away"):
            sets[side] = _side_sets(r[side], r[f"{side}_team_id"], d.toordinal(), st, roster, inj, manual, warn)
            matched |= {n.lower() for n in sets[side][-1]["out"]}

        def run(var):
            row = t.loc[[r.name]].copy()
            for side in ("home", "away"):
                full_pre, prev, full, sel, sel_q, avail, _ = sets[side]
                F, S = {"prereg": (full_pre, prev), "live": (full, sel), "q_out": (full, sel_q),
                        "avail": (full, avail)}[var]
                row[f"Pfull_{side}"], row[f"Ofull_{side}"] = strength(F)
                row[f"Psel_{side}"], row[f"Osel_{side}"] = strength(S)
            v = "act" if var == "avail" else "strict"
            row["dPfull"] = row.Pfull_home - row.Pfull_away
            row["Ofull"] = row.Ofull_home + row.Ofull_away
            row[f"dAbs_{v}"] = (row.Psel_home - row.Pfull_home) - (row.Psel_away - row.Pfull_away)
            row[f"oAbs_{v}"] = (row.Osel_home - row.Ofull_home) + (row.Osel_away - row.Ofull_away)
            mu = float(nsm.predict(fits[f"M2_{v}"], row).iloc[0])
            tot = float(nsm.predict(fits[f"T2_{v}"], row).iloc[0])
            return mu, tot, fits[f"M2_{v}"]["sigma"]

        mu, tot, _ = run("live")
        mu_q, tot_q, _ = run("q_out")
        mu_p, tot_p, _ = run("prereg")
        mu_a, tot_a, sig_a = run("avail")
        srcs = []
        for side in ("home", "away"):
            n_ok, n_all, s = st["paint_cov"].get(r[side], (0, 0, []))
            if n_ok == 0 or pd.isna(r.get(f"{side}_opp_paint_share")):
                srcs.append("missing")
            else:
                srcs.append("+".join(s) + ("" if n_ok == n_all else f" ({n_ok}/{n_all})"))
        if "missing" in srcs:
            paint_source = "missing"
            warn.append("PAINT FEATURE MISSING: S1 paint-defence term = 0 in model_margin (totals unaffected)")
        else:
            paint_source = srcs[0] if srcs[0] == srcs[1] else f"{srcs[0]} | {srcs[1]}"
        for side in ("home", "away"):
            for c in ("fg3_pct", "opp_fg3_pct", "pace"):
                if pd.isna(r.get(f"{side}_{c}")):
                    warn.append(f"{r[side]}: style stat {c} missing -> prediction may be NaN")
        o = odds.get(str(r.event_id), {})
        f = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 2)
        f3 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 3)
        res.append({
            "date": str(d), "event_id": str(r.event_id), "start_utc": pd.Timestamp(r.start_utc).isoformat(),
            "status": r.status, "home": r.home, "away": r.away,
            "model_total": f(tot), "model_margin": f(mu), "model_home_wp": f3(norm.cdf(mu / sig)),
            "model_total_q_out": f(tot_q), "model_margin_q_out": f(mu_q), "model_home_wp_q_out": f3(norm.cdf(mu_q / sig)),
            "model_total_prereg": f(tot_p), "model_margin_prereg": f(mu_p),
            "model_total_avail": f(tot_a), "model_margin_avail": f(mu_a), "model_home_wp_avail": f3(norm.cdf(mu_a / sig_a)),
            "n_out_home": sets["home"][-1]["n_out"], "n_out_away": sets["away"][-1]["n_out"],
            "out_home": sets["home"][-1]["out"], "out_away": sets["away"][-1]["out"],
            "questionable_home": sets["home"][-1]["questionable"], "questionable_away": sets["away"][-1]["questionable"],
            "absent_home": sets["home"][-1]["absent"], "absent_away": sets["away"][-1]["absent"],
            "paint_source": paint_source,
            "market_total": o.get("market_total"), "market_total_open": o.get("market_total_open"),
            "market_spread_home": o.get("market_spread_home"), "market_provider": o.get("market_provider"),
            "warnings": warn,
        })
    unmatched = manual - matched
    if unmatched:
        for x in res:
            x["warnings"].append(f"--out names not matched to any roster player: {', '.join(sorted(unmatched))}")
    log(f"  {len(res)} games predicted in {time.time() - t_start:.1f}s", quiet)
    return res


def print_table(res: list[dict], st_counts=None) -> None:
    if not res:
        return
    print(f"NBA {res[0]['date']}: frozen simple model (M2/T2 STRICT, live roster + ESPN injuries)")
    print(f"{'away @ home':14s} {'total':>6s} {'q_out':>6s} {'avail':>6s} {'prereg':>6s} {'mkt':>6s} {'margin':>7s} "
          f"{'P(home)':>7s} {'out h/a':>7s}  paint")
    for x in res:
        mk = x["market_total"]
        print(f"{x['away'] + ' @ ' + x['home']:14s} {x['model_total']:6.1f} {x['model_total_q_out']:6.1f} "
              f"{x['model_total_avail']:6.1f} {x['model_total_prereg']:6.1f} {mk if mk is not None else '-':>6} {x['model_margin']:+7.1f} "
              f"{x['model_home_wp']:7.3f} {x['n_out_home']:>3d}/{x['n_out_away']:<3d}  {x['paint_source']}")
        for side in ("home", "away"):
            if x[f"out_{side}"]:
                print(f"     {x[side]} OUT: {', '.join(x[f'out_{side}'])}")
            if x[f"questionable_{side}"]:
                print(f"     {x[side]} QUESTIONABLE (in 'total', out in 'q_out'): {', '.join(x[f'questionable_{side}'])}")
            if x[f"absent_{side}"]:
                print(f"     {x[side]} top-5-minutes not in tonight's set: {', '.join(x[f'absent_{side}'])}")
        for w in x["warnings"]:
            print(f"     ! {w}")
    print("\nmargin = home minus away; P(home) = Phi(margin / frozen sigma). total = Questionable players IN; q_out = OUT;"
          " avail = injury-report availability with the frozen ACTUAL fits (not the sealed model). prereg = pre-registered"
          " STRICT inputs (no roster/injury). mkt = ESPN core odds current total (info only). "
          "No bet rule is validated live (see notes/nba_simple_model.md).")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--json", action="store_true", help="print one JSON list to stdout (logs go to stderr)")
    ap.add_argument("--out", default="", help="comma-separated player names to treat as OUT tonight")
    ap.add_argument("--no-refresh", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args(argv)
    with contextlib.redirect_stdout(sys.stderr if a.json else sys.stdout):   # stdout carries only the JSON
        res = predict(a.date, refresh=not a.no_refresh, out_names=[s for s in a.out.split(",") if s.strip()],
                      use_cache=not a.no_cache, quiet=a.json)
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        print_table(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
