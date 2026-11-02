"""Kalshi NBA player-prop ladder FORWARD PAPER logger (nba_sim vs KXNBAPTS / KXNBAREB / KXNBAAST). PAPER ONLY:
reads public Kalshi + ESPN endpoints, places no orders, reads no keys. Pre-registration: notes/kalshi_prop_ladders.md.

    ../.venv_quant/bin/python scripts/kalshi_prop_logger.py tick      # price + log games tipping in 60-120 min
    ../.venv_quant/bin/python scripts/kalshi_prop_logger.py grade     # settle finished games from ESPN box + CLV
    ../.venv_quant/bin/python scripts/kalshi_prop_logger.py report    # picks, W-L, ROI (game-cluster CI), CLV, calibration

tick: for every NBA game on today's ESPN slate with tip 60-120 min away, still STATUS_SCHEDULED and not yet logged,
run the default nba_sim (fresh ESPN games/box + injury feed) and log EVERY two-sided rung (0 < bid, ask < 1) of the
three ladders to data/kalshi_prop_log.jsonl, with sim probability, bid/ask, volume and projected minutes; picks are
flagged by the FROZEN rule below. Games already tipped are never logged (ESPN status re-checked just before writing).
Run it every 10-15 min on game days; it is idempotent per game (a game is logged once, at its first in-window tick).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import requests

LOG = ROOT / "data" / "kalshi_prop_log.jsonl"
GRADES = ROOT / "data" / "kalshi_prop_grades.jsonl"
API = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = {"KXNBAPTS": "pts", "KXNBAREB": "reb", "KXNBAAST": "ast"}
BOX_STAT = {"pts": "points", "reb": "rebounds", "ast": "assists"}

# ---- FROZEN RULE (notes/kalshi_prop_ladders.md, committed before the first logged pick) -------------------------
EDGE = 0.08          # YES if sim >= ask + EDGE; NO if sim <= bid - EDGE
MIN_VOL = 100.0      # rung volume_fp (contracts traded since listing) >= MIN_VOL at log time
MIN_PROJ = 20.0      # player's projected minutes (sim, before jitter) >= MIN_PROJ
WIN_LO, WIN_HI = 60.0, 120.0     # minutes before ESPN tip
N_SIMS, SEED = 20000, 7
MIN_PICKS, MIN_GAMES = 150, 20  # evaluation point (pass / kill rule in the pre-reg)
FEE_K = 0.07         # Kalshi taker fee per contract = 0.07 p (1-p) (rounding ignored; 1-contract paper units)
# ------------------------------------------------------------------------------------------------------------------
K2E = {"GSW": "GS", "NYK": "NY", "NOP": "NO", "UTA": "UTAH", "SAS": "SA", "WAS": "WSH", "PHO": "PHX", "BRK": "BKN"}
_last = [0.0]


def kget(path: str, params: dict | None = None, tries: int = 6) -> dict:
    """Kalshi public GET, paced to <= 2 req/s, exponential backoff on 429 / 5xx."""
    for k in range(tries):
        w = 0.5 - (time.time() - _last[0])
        if w > 0:
            time.sleep(w)
        _last[0] = time.time()
        try:
            r = requests.get(API + path, params=params, timeout=20)
        except requests.RequestException:
            time.sleep(2 ** k)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(min(2 ** (k + 1), 30))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Kalshi GET {path} failed after {tries} tries")


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", "", s.replace("-", " "))
    toks = [t for t in s.split() if t not in ("jr", "sr", "ii", "iii", "iv", "v")]
    return " ".join(toks)


def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def read_log(path=None) -> list[dict]:
    path = LOG if path is None else path
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def append(recs: list[dict], path=None) -> None:
    path = LOG if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for r in recs:
            f.write(json.dumps(r, default=str) + "\n")


def kalshi_codes(espn: str) -> set:
    return {espn} | {k for k, e in K2E.items() if e == espn}


def find_event(series: str, d, away: str, home: str, events: dict) -> str | None:
    tag = pd.Timestamp(d).strftime("%y%b%d").upper()
    for et in events.get(series, []):
        suf = et.split("-")[-1]
        if not suf.startswith(tag):
            continue
        teams = suf[len(tag):]
        if any(teams == a + h for a in kalshi_codes(away) for h in kalshi_codes(home)):
            return et
    return None


def open_events() -> dict:
    out = {}
    for s in SERIES:
        evs, cur = [], None
        while True:
            j = kget("/events", {"series_ticker": s, "status": "open", "limit": 200, **({"cursor": cur} if cur else {})})
            evs += [e["event_ticker"] for e in j.get("events", [])]
            cur = j.get("cursor")
            if not cur:
                break
        out[s] = evs
    return out


def event_markets(et: str) -> list[dict]:
    ms, cur = [], None
    while True:
        j = kget("/markets", {"event_ticker": et, "limit": 1000, **({"cursor": cur} if cur else {})})
        ms += j.get("markets", [])
        cur = j.get("cursor")
        if not cur:
            break
    return ms


def parse_rung(m: dict):
    """-> (player name, k) from yes_sub_title 'Name: 20+'; k = threshold (stat >= k)."""
    t = m.get("yes_sub_title") or ""
    mm = re.match(r"^(.*):\s*(\d+)\+\s*$", t)
    if mm:
        return mm.group(1).strip(), int(mm.group(2))
    fs = m.get("floor_strike")
    return (t.split(":")[0].strip() or None), (int(round(float(fs) + 0.5)) if fs is not None else None)


def espn_status(eids) -> dict:
    """event_id -> (status name, start_utc) from today's + yesterday's ESPN scoreboard (fresh, no cache)."""
    import nba_simple_live as nsl
    from sports.http import fetch
    out = {}
    d = pd.Timestamp.now(tz="US/Eastern")
    for day in (d - pd.Timedelta(days=1), d):
        for e in fetch(nsl.SITE + "/scoreboard", {"dates": day.strftime("%Y%m%d"), "limit": 1000}, cache=False).get("events", []):
            st = ((e.get("status") or {}).get("type") or {}).get("name")
            out[str(e["id"])] = (st, e.get("date"))
    return {str(k): out.get(str(k)) for k in eids}


def flag(sim: float | None, bid: float, ask: float, vol: float, mproj: float | None) -> str | None:
    if sim is None or mproj is None or vol < MIN_VOL or mproj < MIN_PROJ:
        return None
    if sim >= round(ask + EDGE, 6) - 1e-9:             # rounding guard: 0.62 + 0.08 must equal 0.70
        return "YES"
    if sim <= round(bid - EDGE, 6) + 1e-9:
        return "NO"
    return None


def sim_game(away: str, home: str, d) -> tuple[dict, dict]:
    """-> ({norm name: {aid, name, team, min_proj, pts/reb/ast arrays}}, meta). Default nba_sim, no refresh (done once)."""
    import nba_sim as ns
    p = ns.load_params()
    L = ns.live_setup(away, home, d, False)
    S = ns.live_sides(L)
    res = ns.price(p, L["mu0"], L["tot0"], S["home"]["si"], S["away"]["si"], N_SIMS, SEED)
    out = {}
    for side, key in (("home", "players_h"), ("away", "players_a")):
        sp, x = res[key], S[side]
        if sp is None:
            continue
        mm = x["si"].get("m_pl") or x["si"]["m"]
        for j, aid in enumerate(sp["ids"]):
            nm = L["st"].names.get(aid) or x["cands"].get(aid, aid)
            out.setdefault(norm_name(nm), []).append({"aid": str(aid), "name": nm, "team": x["T"],
                                                      "min_proj": float(mm[aid]),
                                                      **{s: sp[s][:, j] for s in ("pts", "reb", "ast")}})
    # players on the roster but not in tonight's set (OUT / not in rotation): known, projected 0 minutes
    for side in ("home", "away"):
        x = S[side]
        for aid, nm in x["cands"].items():
            k = norm_name(nm or aid)
            if k not in out:
                out[k] = [{"aid": str(aid), "name": nm, "team": x["T"], "min_proj": 0.0, "absent": True}]
    meta = {"sim_tag": ns.IMPACT["mode"] + ":" + ns.impact_tag(), "p_home": res["p_home"],
            "out_home": S["home"]["out"], "out_away": S["away"]["out"], "q_home": S["home"]["q"],
            "q_away": S["away"]["q"], "warn": S["warn"][:12]}
    return out, meta


def cmd_tick(a) -> int:
    import nba_simple_live as nsl
    import nba_simple_model as nsm
    now = now_utc()
    d = now.tz_convert("US/Eastern").date()
    try:
        nsl.refresh_season(nsl.season_of(d), d, True)
    except Exception as ex:
        print(f"(ESPN refresh failed: {str(ex)[:80]}; using disk)", file=sys.stderr)
    g = pd.read_parquet(nsm.ESPN / f"games_{nsl.season_of(d)}.parquet")
    g = g[g.start_utc.dt.tz_convert("US/Eastern").dt.date == d].copy()
    g["mins_to_tip"] = (g.start_utc - now).dt.total_seconds() / 60
    done = {r["event_id"] for r in read_log() if r.get("type") == "game"}
    print(f"{now.tz_convert('US/Pacific'):%Y-%m-%d %H:%M} PT  slate {d}: "
          + ", ".join(f"{r.away}@{r.home} {r.mins_to_tip:+.0f}m {r.status.replace('STATUS_', '')}" for r in g.itertuples()))
    win = g[(g.mins_to_tip >= WIN_LO) & (g.mins_to_tip <= WIN_HI) & (g.status == "STATUS_SCHEDULED")
            & ~g.event_id.astype(str).isin(done)]
    if a.event:
        win = g[g.event_id.astype(str) == a.event]
    if win.empty:
        print(f"no game in the {WIN_LO:.0f}-{WIN_HI:.0f} min window (or already logged) -> nothing logged")
        return 0
    events = open_events()
    tickets = []
    for r in win.itertuples():
        eid = str(r.event_id)
        t0 = time.time()
        players, meta = sim_game(r.away, r.home, d)
        rungs, nmiss = [], set()
        snap = now_utc()
        for ser, stat in SERIES.items():
            et = find_event(ser, d, r.away, r.home, events)
            if et is None:
                print(f"  {r.away}@{r.home}: no open {ser} event")
                continue
            for m in event_markets(et):
                bid, ask = float(m.get("yes_bid_dollars") or 0), float(m.get("yes_ask_dollars") or 1)
                if not (bid > 0 and ask < 1 and ask > bid) or m.get("status") not in ("active", "open"):
                    continue
                nm, k = parse_rung(m)
                cand = players.get(norm_name(nm or ""), [])
                pl = cand[0] if len(cand) == 1 else None
                if pl is None:
                    nmiss.add(nm)
                sim = None
                if pl is not None:
                    sim = 0.0 if pl.get("absent") else float((pl[stat] >= k).mean())
                vol = float(m.get("volume_fp") or 0)
                mp = pl["min_proj"] if pl is not None else None
                rec = {"type": "rung", "ts": snap.isoformat(), "event_id": eid, "game": f"{r.away}@{r.home}",
                       "tip_utc": r.start_utc.isoformat(), "mins_to_tip": round(float((r.start_utc - snap).total_seconds() / 60), 1),
                       "ticker": m["ticker"], "event_ticker": et, "stat": stat, "player": nm, "k": k,
                       "athlete_id": pl["aid"] if pl else None, "team": pl["team"] if pl else None,
                       "min_proj": round(mp, 2) if mp is not None else None, "absent": bool(pl.get("absent")) if pl else None,
                       "sim": round(sim, 4) if sim is not None else None, "bid": bid, "ask": ask,
                       "mid": round((bid + ask) / 2, 4), "last": float(m.get("last_price_dollars") or 0),
                       "volume": vol, "oi": float(m.get("open_interest_fp") or 0),
                       "pick": flag(sim, bid, ask, vol, mp), "sim_tag": meta["sim_tag"],
                       "rule": {"E": EDGE, "V": MIN_VOL, "MINP": MIN_PROJ}}
                rungs.append(rec)
        st = espn_status([eid]).get(eid)
        if st is None or st[0] != "STATUS_SCHEDULED" or (r.start_utc - now_utc()).total_seconds() < 0:
            print(f"  {r.away}@{r.home}: ESPN status {st} -> NOT logged (tipped / not scheduled)")
            continue
        if a.dry:
            print(f"  [dry] would log {len(rungs)} rungs")
        else:
            append([{"type": "game", "ts": snap.isoformat(), "event_id": eid, "game": f"{r.away}@{r.home}",
                     "tip_utc": r.start_utc.isoformat(), "n_rungs": len(rungs),
                     "n_picks": sum(1 for x in rungs if x["pick"]), "unmatched": sorted(x for x in nmiss if x), **meta}]
                   + rungs)
        picks = [x for x in rungs if x["pick"]]
        tickets += picks
        print(f"  {r.away}@{r.home} tip {r.start_utc.tz_convert('US/Pacific'):%H:%M} PT: {len(rungs)} two-sided rungs, "
              f"{len(picks)} picks, unmatched names {sorted(x for x in nmiss if x)} ({time.time() - t0:.0f}s)")
        if meta["out_home"] or meta["out_away"]:
            print(f"    OUT(feed) {r.away}: {', '.join(meta['out_away']) or '-'};  {r.home}: {', '.join(meta['out_home']) or '-'}")
    if tickets:
        print("\nPAPER TICKET (1 unit each, NOT placed):")
        for x in sorted(tickets, key=lambda x: (x["game"], x["player"], x["stat"], x["k"])):
            px = x["ask"] if x["pick"] == "YES" else 1 - x["bid"]
            print(f"  {x['game']:<9} {x['pick']:<3} {x['player']:<24} {x['stat']} {x['k']}+  sim {x['sim']:.3f}  "
                  f"bid/ask {x['bid']:.2f}/{x['ask']:.2f}  pay {px:.2f}  vol {x['volume']:.0f}  min {x['min_proj']:.1f}  {x['ticker']}")
    return 0


def box_stats(eids: set) -> dict:
    import nba_simple_live as nsl
    import nba_simple_model as nsm
    d = now_utc().tz_convert("US/Eastern").date()
    try:
        nsl.refresh_season(nsl.season_of(d), d + pd.Timedelta(days=1), True)
    except Exception as ex:
        print(f"(ESPN refresh failed: {str(ex)[:80]})", file=sys.stderr)
    S = nsl.season_of(d)
    g = pd.read_parquet(nsm.ESPN / f"games_{S}.parquet")
    fin = set(g[g.status == "STATUS_FINAL"].event_id.astype(str)) & set(eids)
    b = pd.read_parquet(nsm.ESPN / f"box_{S}.parquet")
    b = b[b.event_id.astype(str).isin(fin)]
    out = {}
    for (e, aid), x in b.groupby([b.event_id.astype(str), b.athlete_id.astype(str)]):
        v = dict(zip(x.stat, x.value))
        out[(e, aid)] = v | {"name": x.name.iloc[0]}
    return {"final": fin, "box": out}


def clv_close(ticker: str, tip_utc: str) -> float | None:
    ts = int(pd.Timestamp(tip_utc).timestamp())
    j = kget("/markets/trades", {"ticker": ticker, "max_ts": ts, "limit": 1})
    tr = j.get("trades") or []
    return float(tr[0]["yes_price_dollars"]) if tr else None


def fee(p: float) -> float:
    return FEE_K * p * (1 - p)


def cmd_grade(a) -> int:
    R = [r for r in read_log() if r.get("type") == "rung"]
    graded = {g["ticker"] for g in read_log(GRADES) if not g.get("pending")}
    todo = [r for r in R if r["ticker"] not in graded]
    if not todo:
        print("nothing to grade")
        return 0
    B = box_stats({r["event_id"] for r in todo})
    by_name = {}
    for (e, aid), v in B["box"].items():
        by_name.setdefault((e, norm_name(v["name"])), []).append(aid)
    out = []
    for r in todo:
        e = r["event_id"]
        if e not in B["final"]:
            continue
        aid = r.get("athlete_id")
        if aid is None or (e, aid) not in B["box"]:
            c = by_name.get((e, norm_name(r["player"] or "")), [])
            aid = c[0] if len(c) == 1 else None
        v = B["box"].get((e, aid)) if aid else None
        mins = float(v.get("minutes", 0) or 0) if v else 0.0
        g = {"ticker": r["ticker"], "event_id": e, "graded_ts": now_utc().isoformat()}
        if v is None or mins <= 0:
            g.update(void=True, reason="DNP / not in box (Kalshi settles inactive players at a fair price)")
        else:
            val = float(v.get(BOX_STAT[r["stat"]], np.nan))
            y = int(val >= r["k"])
            g.update(void=False, stat_value=val, minutes=mins, y=y)
            if r.get("pick"):
                if r["pick"] == "YES":
                    cost = r["ask"] + fee(r["ask"])
                    g["profit"] = y - cost
                else:
                    cost = (1 - r["bid"]) + fee(1 - r["bid"])
                    g["profit"] = (1 - y) - cost
                g["cost"] = cost
                c = clv_close(r["ticker"], r["tip_utc"])
                g["close_yes"] = c
                if c is not None:
                    g["clv"] = (c - r["ask"]) if r["pick"] == "YES" else (r["bid"] - c)
        out.append(g)
    append(out, GRADES)
    print(f"graded {len(out)} rungs ({sum(1 for g in out if g.get('void'))} void); "
          f"{sum(1 for g in out if 'profit' in g)} picks")
    return 0


def frame() -> pd.DataFrame:
    R = pd.DataFrame([r for r in read_log() if r.get("type") == "rung"])
    G = pd.DataFrame(read_log(GRADES))
    if R.empty or G.empty:
        return pd.DataFrame()
    R = R.drop_duplicates("ticker", keep="first")
    G = G.drop_duplicates("ticker", keep="last")
    return R.merge(G.drop(columns=["event_id"]), on="ticker", how="inner")


def cluster_boot(df: pd.DataFrame, num: str, den: str | None, n=10000, seed=0):
    g = df.groupby("event_id").agg(s=(num, "sum"), c=((den or num), "sum" if den else "size"))
    s, c = g.s.to_numpy(float), g.c.to_numpy(float)
    k = np.random.default_rng(seed).integers(0, len(g), (n, len(g)))
    bs = s[k].sum(1) / c[k].sum(1)
    est = s.sum() / c.sum()
    return float(est), float(np.quantile(bs, .025)), float(np.quantile(bs, .975)), float((bs <= 0).mean())


def cmd_report(a) -> int:
    D = frame()
    if D.empty:
        print("no graded rungs yet")
        return 0
    S = D[~D.void.astype(bool)].copy()
    P = S[S.pick.notna() & S.profit.notna()]
    print(f"graded rungs {len(D)} (void {int(D.void.astype(bool).sum())}), games {D.event_id.nunique()}; "
          f"rule E={EDGE} V>={MIN_VOL:g} min>={MIN_PROJ:g}")
    if len(P):
        w = int(((P.pick == "YES") & (P.y == 1)).sum() + ((P.pick == "NO") & (P.y == 0)).sum())
        roi = cluster_boot(P, "profit", "cost")
        print(f"PICKS {len(P)} (YES {int((P.pick == 'YES').sum())} / NO {int((P.pick == 'NO').sum())}), "
              f"W-L {w}-{len(P) - w}, units {P.profit.sum():+.2f} on {P.cost.sum():.2f} staked")
        print(f"  ROI {roi[0]:+.3f}  95% CI [{roi[1]:+.3f}, {roi[2]:+.3f}]  one-sided p(ROI<=0) {roi[3]:.3f}  "
              f"(game-cluster bootstrap)")
        C = P.dropna(subset=["clv"]) if "clv" in P else P.iloc[:0]
        if len(C):
            cv = cluster_boot(C, "clv", None)
            print(f"  CLV (last trade before tip - entry) mean {cv[0]:+.4f} [{cv[1]:+.4f}, {cv[2]:+.4f}] "
                  f"p(CLV<=0) {cv[3]:.3f}  n={len(C)}")
        n = len(P)
        verdict = f"PENDING ({n} settled picks / {P.event_id.nunique()} games; evaluated at >= {MIN_PICKS} picks and >= {MIN_GAMES} games)"
        if n >= MIN_PICKS and P.event_id.nunique() >= MIN_GAMES:
            cvp = cluster_boot(C, "clv", None) if len(C) else (np.nan, np.nan, np.nan, 1.0)
            if roi[0] > 0 and roi[3] < 0.05 and cvp[0] > 0:
                verdict = "PASS"
            elif roi[2] < 0 or (cvp[0] < 0 and 1 - cvp[3] < 0.05):
                verdict = "KILL"
            else:
                verdict = "NOT DISTINGUISHABLE yet (keep logging; re-read pre-reg)"
        print(f"  verdict: {verdict}")
    # calibration of ALL logged rungs with a sim prob: sim vs Kalshi mid (same rows)
    Q = S.dropna(subset=["sim"]).copy()
    Q = Q[~Q.absent.fillna(False).astype(bool)]
    if len(Q):
        bins = [0, .05, .15, .25, .35, .45, .55, .65, .75, .85, .95, 1.0001]
        print(f"\nCALIBRATION, {len(Q)} non-void rungs with a sim price ({Q.event_id.nunique()} games)")
        print(f"  {'bin':<11}{'n_sim':>6}{'sim':>7}{'hit':>7}   {'n_mid':>6}{'mid':>7}{'hit':>7}")
        Q["bs"] = pd.cut(Q.sim, bins, right=False)
        Q["bm"] = pd.cut(Q.mid, bins, right=False)
        for b in Q.bs.cat.categories:
            x, z = Q[Q.bs == b], Q[Q.bm == b]
            print(f"  {str(b):<11}{len(x):>6}{(x.sim.mean() if len(x) else np.nan):>7.3f}{(x.y.mean() if len(x) else np.nan):>7.3f}"
                  f"   {len(z):>6}{(z.mid.mean() if len(z) else np.nan):>7.3f}{(z.y.mean() if len(z) else np.nan):>7.3f}")
        Q["b_sim"], Q["b_mid"] = (Q.sim - Q.y) ** 2, (Q.mid - Q.y) ** 2
        Q["d"] = Q.b_sim - Q.b_mid
        db = cluster_boot(Q, "d", None)
        ll = lambda p: -np.mean(Q.y * np.log(np.clip(p, .01, .99)) + (1 - Q.y) * np.log(np.clip(1 - p, .01, .99)))
        print(f"  Brier sim {Q.b_sim.mean():.4f} vs mid {Q.b_mid.mean():.4f}; diff {db[0]:+.4f} [{db[1]:+.4f}, {db[2]:+.4f}]"
              f" (game-cluster);  log-loss sim {ll(Q.sim):.4f} vs mid {ll(Q.mid):.4f}")
        for st in ("pts", "reb", "ast"):
            x = Q[Q.stat == st]
            if len(x):
                print(f"    {st}: n {len(x)}  Brier sim {x.b_sim.mean():.4f}  mid {x.b_mid.mean():.4f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tick")
    t.add_argument("--dry", action="store_true", help="price and print, write nothing")
    t.add_argument("--event", help="ESPN event id: price this game regardless of the window (use with --dry only)")
    sub.add_parser("grade")
    sub.add_parser("report")
    a = ap.parse_args(argv)
    if a.cmd == "tick" and a.event and not a.dry:
        ap.error("--event bypasses the pre-registered window: only allowed with --dry")
    return {"tick": cmd_tick, "grade": cmd_grade, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
