"""IN-GAME cross-venue lag logger: Pinnacle LIVE moneyline vs Kalshi in-game game-winner quotes. Paper only, no orders.

    python scripts/live_lag_logger.py run [--minutes 120] [--kal-every 10] [--pin-every 30]
    python scripts/live_lag_logger.py report [--files data/live_lag/*.jsonl]

Question (edgelab trial xv_live_lag_t03, family cross_venue, registered BEFORE the first logged tick): when a fresh
Pinnacle live price arrives, is the Kalshi ask stale enough to buy with positive EV after the taker fee?

FRESHNESS -- the thing that decides whether this test is even possible (probed 2026-11-05):
  Every guest.api.arcadia.pinnacle.com endpoint (league markets, sport live markets, matchup markets, related
  markets, matchup) is served from a Cloudflare cache with `Cache-Control: public, max-age=~905` -- live odds are
  up to ~15 MIN old when read. A unique query string is refused (HTTP 204), and we do not try to get around it. So
  each Pinnacle record carries its own as_of = the response's `last-modified` header (fallback Date - Age), and the
  rule only fires on a snapshot RECEIVED within FRESH_S seconds of its as_of (i.e. right after a cache refill).
  Four endpoints per live game refresh on independent ~15-min cycles, so a game gets a fresh Pinnacle view every
  few minutes, never every few seconds. A seconds-level "Pinnacle moved, Kalshi lagged" measurement is not possible
  from this source; what IS measurable is whether Kalshi is still away from Pinnacle when a fresh view lands.

Records (jsonl, one per line, data/live_lag/<UTC date>.jsonl):
  pin: recv, url, as_of, lm, date, age, cf, league, live_id, parent_id, home, away, p_home, p_away (American), status,
       version, max_risk, score_home, score_away, clock
  kal: recv, ticker, event, team, live_id, side (home/away), yes_bid, yes_ask, no_bid, no_ask, last
  tick: recv, n_live, n_matched, errors

THE REGISTERED RULE (spec edgelab/specs/xv_live_lag_t03.yaml; any change = a new trial):
  decision = a pin record with status open, both prices, recv - as_of <= FRESH_S (45 s);
  fair(side) = min(power de-vig, Shin de-vig) of the two-way moneyline (never proportional);
  Kalshi quote = latest kal record for that side's ticker with recv <= decision recv and age <= KAL_MAX_AGE (15 s);
  bet YES on that side's ticker if fair - ask - fee(ask) >= THRESH (0.03), 0.05 <= ask <= 0.95;
  price = ask + Kalshi taker fee (0.07 p (1-p), 1 contract, rounded UP to the cent);
  at most one bet per (game, side) per DEDUP_S (600 s);
  outcome = Kalshi settlement of the ticker; CLV = Kalshi mid at decision +60 s / +300 s minus ask.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from sports import live
from sports.live import PIN, PIN_HEADERS, KALSHI

OUT = ROOT / "data" / "live_lag"
LEAGUES = {  # name -> (pinnacle league id, pinnacle sport id, kalshi series)
    "nba": (487, 4, "KXNBAGAME"),
    "cbb": (493, 4, "KXNCAAMBGAME"),
    "nhl": (1456, 19, "KXNHLGAME"),
    "nfl": (889, 15, "KXNFLGAME"),
    "cfb": (880, 15, "KXNCAAFGAME"),
}
# registered rule constants (do not edit: a change is a NEW trial)
FRESH_S = 45
KAL_MAX_AGE = 15
THRESH = 0.03
ASK_LO, ASK_HI = 0.05, 0.95
DEDUP_S = 600
MOVE = 0.02   # descriptive only: a Pinnacle "move" = fair changed >= 2c vs the previous fresh view of that game


def now() -> datetime:
    return datetime.now(timezone.utc)


def iso(t: datetime) -> str:
    return t.isoformat(timespec="milliseconds")


# ------------------------------------------------------------------ de-vig (power + Shin; never proportional)

def implied(am: float) -> float:
    return 1 / live.am_to_dec(am)


def power_devig(q1: float, q2: float) -> tuple[float, float]:
    return live.devig_power(q1, q2)


def shin_devig(q1: float, q2: float) -> tuple[float, float]:
    """Shin (1993) for two outcomes: find z in [0, .5) so the Shin probabilities sum to 1."""
    q = np.array([q1, q2]); S = q.sum()

    def probs(z):
        return (np.sqrt(z * z + 4 * (1 - z) * q * q / S) - z) / (2 * (1 - z))
    lo, hi = 0.0, 0.5
    for _ in range(80):
        z = (lo + hi) / 2
        if probs(z).sum() > 1:
            lo = z
        else:
            hi = z
    p = probs((lo + hi) / 2)
    return float(p[0]), float(p[1])


def fair_pair(am_home: float, am_away: float) -> tuple[float, float]:
    q1, q2 = implied(am_home), implied(am_away)
    a1, a2 = power_devig(q1, q2)
    b1, b2 = shin_devig(q1, q2)
    return min(a1, b1), min(a2, b2)   # conservative for whichever side is bought


def taker_fee(p: float) -> float:
    return math.ceil(round(0.07 * p * (1 - p) * 100, 6)) / 100


# ------------------------------------------------------------------ collection

class Pin:
    """Per-URL poller that respects the CDN: a URL is re-read every pin_every s; each response keeps its own as_of."""

    def __init__(self, every: float):
        self.every = every
        self.last: dict[str, float] = {}

    def get(self, url: str):
        t = time.time()
        if t - self.last.get(url, 0) < self.every:
            return None
        self.last[url] = t
        r = live._s.get(url, headers=PIN_HEADERS, timeout=20)
        recv = now()
        if r.status_code != 200 or not r.content:
            return {"recv": recv, "status": r.status_code, "js": None}
        h = r.headers
        date = parsedate_to_datetime(h["Date"]) if h.get("Date") else recv
        lm = parsedate_to_datetime(h["last-modified"]) if h.get("last-modified") else None
        age = float(h.get("Age") or 0)
        as_of = lm if lm is not None else date - pd.Timedelta(seconds=age)
        return {"recv": recv, "status": 200, "js": r.json(), "as_of": as_of, "lm": lm, "date": date, "age": age,
                "cf": h.get("cf-cache-status")}


def live_games(mu: list) -> dict:
    """Live full-game matchups (child of a pre-game parent) whose two participants are the parent's two teams."""
    out = {}
    for m in mu:
        if not m.get("isLive") or m.get("type") != "matchup" or m.get("units", "Regular") != "Regular":
            continue
        par = m.get("parent") or {}
        side = {p["alignment"]: p["name"] for p in m.get("participants", []) if p.get("alignment") in ("home", "away")}
        pside = {p["alignment"]: p for p in par.get("participants", []) if p.get("alignment") in ("home", "away")}
        if set(side) != {"home", "away"} or len(m.get("participants", [])) != 2:
            continue
        if pside and {side["home"], side["away"]} != {pside.get("home", {}).get("name"), pside.get("away", {}).get("name")}:
            continue
        st = par.get("state") or {}
        out[m["id"]] = {"live_id": m["id"], "parent_id": m.get("parentId"), "home": side["home"], "away": side["away"],
                        "start": par.get("startTime") or m.get("startTime"),
                        "score_home": (pside.get("home", {}).get("state") or {}).get("score"),
                        "score_away": (pside.get("away", {}).get("state") or {}).get("score"),
                        "clock": json.dumps(st, sort_keys=True) if st else None}
    return out


def pin_rows(resp: dict, url: str, league: str, games: dict) -> list[dict]:
    rows = []
    for k in resp["js"] or []:
        g = games.get(k.get("matchupId"))
        if g is None or k.get("period") != 0 or k.get("type") != "moneyline" or k.get("isAlternate"):
            continue
        pr = {p["designation"]: p.get("price") for p in k.get("prices", [])}
        rows.append({"t": "pin", "recv": iso(resp["recv"]), "url": url.replace(PIN, ""), "as_of": iso(resp["as_of"]),
                     "lm": iso(resp["lm"]) if resp["lm"] else None, "date": iso(resp["date"]), "age": resp["age"],
                     "cf": resp["cf"], "league": league, **{c: g[c] for c in ("live_id", "parent_id", "home", "away",
                                                                                 "score_home", "score_away", "clock")},
                     "p_home": pr.get("home"), "p_away": pr.get("away"), "status": k.get("status"),
                     "version": k.get("version"),
                     "max_risk": next((l["amount"] for l in k.get("limits", []) if l["type"] == "maxRiskStake"), None)})
    return rows


def kalshi_markets(series: str) -> list[dict]:
    out, cursor = [], None
    while True:
        params = {"series_ticker": series, "status": "open", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        d = live._get(f"{KALSHI}/markets", params=params)
        out += d.get("markets", [])
        cursor = d.get("cursor")
        if not cursor or not d.get("markets"):
            return out


def match(games: dict, kms: list[dict]) -> dict:
    """ticker -> (live_id, side). Reuses live.match_kalshi (event-ticker date + city-prefix names)."""
    if not games or not kms:
        return {}
    pin = pd.DataFrame([{"pin_id": g["live_id"], "start": pd.Timestamp(g["start"]), "home": g["home"],
                         "away": g["away"], "market": "moneyline", "fair1": np.nan, "fair2": np.nan, "pfair1": np.nan,
                         "pfair2": np.nan, "price1": np.nan, "price2": np.nan, "hold": np.nan, "max_risk": np.nan}
                        for g in games.values()])
    kal = pd.DataFrame([{"ticker": m["ticker"], "event": m["event_ticker"], "team": m.get("yes_sub_title")}
                        for m in kms])
    m = live.match_kalshi(pin, kal)
    out = {}
    for _, r in m.iterrows():
        g = games[r["pin_id"]]
        side = "home" if live._norm(g["home"]).startswith(live._norm(r["team"])) else "away"
        out[r["ticker"]] = (int(r["pin_id"]), side)
    return out


def run(minutes: float, kal_every: float, pin_every: float, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pin = Pin(pin_every)
    t_end = time.time() + minutes * 60
    games_by_lg: dict[str, dict] = {lg: {} for lg in LEAGUES}
    tmap: dict[str, tuple] = {}
    last_match = 0.0
    while time.time() < t_end:
        t0 = time.time()
        recs, errs = [], []
        # Pinnacle: league matchups (live games + score), league + sport-live markets, per-game markets
        for lg, (lid, sid, ser) in LEAGUES.items():
            try:
                r = pin.get(f"{PIN}/leagues/{lid}/matchups")
                if r and r["js"] is not None:
                    games_by_lg[lg] = live_games(r["js"])
                g = games_by_lg[lg]
                if not g:
                    continue
                urls = [f"{PIN}/leagues/{lid}/markets/straight"]
                urls += [f"{PIN}/matchups/{i}/markets/straight" for i in g]
                urls += [f"{PIN}/matchups/{i}/markets/related/straight" for i in g]
                for u in urls:
                    r = pin.get(u)
                    if r and r["js"] is not None:
                        recs += pin_rows(r, u, lg, g)
            except Exception as e:  # noqa: BLE001 -- a logger must keep running; the error is logged
                errs.append(f"pin {lg}: {e!r}"[:200])
        # Pinnacle sport-level live feeds (one per sport, shared by leagues)
        for sid in sorted({v[1] for v in LEAGUES.values()}):
            allg = {k: v for lg, (l2, s2, _) in LEAGUES.items() if s2 == sid for k, v in games_by_lg[lg].items()}
            if not allg:
                continue
            try:
                u = f"{PIN}/sports/{sid}/markets/live/straight"
                r = pin.get(u)
                if r and r["js"] is not None:
                    lgof = {k: lg for lg in LEAGUES for k in games_by_lg[lg]}
                    for row in pin_rows(r, u, "?", allg):
                        row["league"] = lgof.get(row["live_id"], "?")
                        recs.append(row)
            except Exception as e:  # noqa: BLE001
                errs.append(f"pin sport {sid}: {e!r}"[:200])
        # Kalshi: every tick, every series with a live Pinnacle game
        for lg, (lid, sid, ser) in LEAGUES.items():
            g = games_by_lg[lg]
            if not g:
                continue
            try:
                kms = kalshi_markets(ser)
                recv = iso(now())
                if time.time() - last_match > 60 or not any(tmap.get(m["ticker"], (None,))[0] in g for m in kms):
                    tmap.update(match(g, kms))
                f = lambda m, k: float(m[k]) if m.get(k) not in (None, "") else None
                for m in kms:
                    if m["ticker"] not in tmap or tmap[m["ticker"]][0] not in g:
                        continue
                    lid_, side = tmap[m["ticker"]]
                    recs.append({"t": "kal", "recv": recv, "ticker": m["ticker"], "event": m["event_ticker"],
                                 "team": m.get("yes_sub_title"), "live_id": lid_, "side": side, "league": lg,
                                 "yes_bid": f(m, "yes_bid_dollars"), "yes_ask": f(m, "yes_ask_dollars"),
                                 "no_bid": f(m, "no_bid_dollars"), "no_ask": f(m, "no_ask_dollars"),
                                 "last": f(m, "last_price_dollars")})
            except Exception as e:  # noqa: BLE001
                errs.append(f"kal {lg}: {e!r}"[:200])
        if time.time() - last_match > 60:
            last_match = time.time()
        n_live = sum(len(g) for g in games_by_lg.values())
        live_ids = {k for g in games_by_lg.values() for k in g}
        recs.append({"t": "tick", "recv": iso(now()), "n_live": n_live,
                     "n_matched": len({v[0] for v in tmap.values() if v[0] in live_ids}), "errors": errs})
        with (out / f"{now():%Y-%m-%d}.jsonl").open("a") as fh:
            for r in recs:
                fh.write(json.dumps(r, default=str) + "\n")
        print(f"{now():%H:%M:%S} live={n_live} matched={recs[-1]['n_matched']} "
              f"pin={sum(r['t'] == 'pin' for r in recs)} kal={sum(r['t'] == 'kal' for r in recs)} err={len(errs)}",
              flush=True)
        if n_live == 0 and time.time() - t0 < kal_every:
            time.sleep(max(0, 60 - (time.time() - t0)))   # nothing live: idle at 1/min
        else:
            time.sleep(max(0, kal_every - (time.time() - t0)))


# ------------------------------------------------------------------ report

def load(files) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for f in files:
        with open(f) as fh:
            rows += [json.loads(l) for l in fh if l.strip()]
    df = pd.DataFrame(rows)
    if df.empty:
        return df, df, df
    for c in ("recv", "as_of"):
        if c in df:
            df[c] = pd.to_datetime(df[c], utc=True, format="ISO8601")
    pin = df[df["t"] == "pin"].copy()
    kal = df[df["t"] == "kal"].copy()
    ticks = df[df["t"] == "tick"].copy()
    return pin, kal, ticks


def kal_at(kq: pd.DataFrame, t: pd.Timestamp, max_age: float | None = None):
    """Latest Kalshi quote with recv <= t (point-in-time); None if older than max_age seconds."""
    i = kq["recv"].searchsorted(t, side="right") - 1
    if i < 0:
        return None
    r = kq.iloc[i]
    if max_age is not None and (t - r["recv"]).total_seconds() > max_age:
        return None
    return r


def settle(tickers) -> dict:
    out = {}
    for t in tickers:
        try:
            m = live._get(f"{KALSHI}/markets/{t}")["market"]
            out[t] = {"yes": 1.0, "no": 0.0}.get(m.get("result"))
        except Exception:  # noqa: BLE001
            out[t] = None
    return out


def report(files, do_settle: bool = True) -> dict:
    sys.path.insert(0, str(Path.home() / "agent-harness" / "lib"))
    import traps  # noqa: E402  timestamp-unit + lookahead checks
    pin, kal, ticks = load(files)
    if pin.empty or kal.empty:
        print("no pin/kal rows yet"); return {}
    for d, c in ((pin, "recv"), (pin, "as_of"), (kal, "recv")):
        traps.check_epoch_seconds(traps.epoch_seconds(d[c]), name=c)   # unit-safe, never astype(int64)//1e9
    pin = pin[pin["status"] == "open"].dropna(subset=["p_home", "p_away"]).copy()
    pin["lag_s"] = (pin["recv"] - pin["as_of"]).dt.total_seconds()
    fp = pin.apply(lambda r: fair_pair(r["p_home"], r["p_away"]), axis=1)
    pin["fair_home"] = [a for a, b in fp]; pin["fair_away"] = [b for a, b in fp]
    pin["hold"] = pin.apply(lambda r: implied(r["p_home"]) + implied(r["p_away"]) - 1, axis=1)
    fresh = pin[pin["lag_s"] <= FRESH_S].sort_values("recv")
    # one decision per distinct (game, as_of, prices): the same fresh body read twice is one view
    fresh = fresh.drop_duplicates(["live_id", "as_of", "p_home", "p_away"])
    kal = kal.sort_values("recv")
    kq = {t: g.reset_index(drop=True) for t, g in kal.groupby("ticker")}
    tick_of = {(r.live_id, r.side): r.ticker for r in kal.drop_duplicates(["live_id", "side"]).itertuples()}
    print(f"files {len(files)}; ticks {len(ticks)}; pin rows {len(pin)} (open, priced); kal rows {len(kal)}; "
          f"games {pin['live_id'].nunique()} pinnacle-live, {kal['live_id'].nunique()} matched to Kalshi")
    print(f"Pinnacle snapshot age at receipt (recv - last-modified), s: "
          f"median {pin['lag_s'].median():.0f}, p10 {pin['lag_s'].quantile(.1):.0f}, p90 {pin['lag_s'].quantile(.9):.0f}; "
          f"fresh (<= {FRESH_S}s) views: {len(fresh)} over {fresh['live_id'].nunique()} games; "
          f"median live hold {pin['hold'].median():.3f}")
    ev_rows, bets, last_bet = [], [], {}
    prev_fair: dict = {}
    for r in fresh.itertuples():
        moved = None
        if r.live_id in prev_fair:
            moved = r.fair_home - prev_fair[r.live_id]
        prev_fair[r.live_id] = r.fair_home
        for side, fair in (("home", r.fair_home), ("away", r.fair_away)):
            tk = tick_of.get((r.live_id, side))
            if tk is None:
                continue
            q = kal_at(kq[tk], r.recv, KAL_MAX_AGE)
            if q is None or q["yes_ask"] is None or pd.isna(q["yes_ask"]):
                continue
            assert q["recv"] <= r.recv  # point-in-time
            ask = float(q["yes_ask"]); mid = (float(q["yes_bid"] or 0) + ask) / 2
            edge = fair - ask - taker_fee(ask)
            row = {"live_id": r.live_id, "league": r.league, "side": side, "ticker": tk, "decided_at": r.recv,
                   "as_of": r.as_of, "pin_lag_s": r.lag_s, "fair": fair, "ask": ask, "mid": mid, "gap_mid": fair - mid,
                   "edge": edge, "pin_move": None if moved is None else (moved if side == "home" else -moved)}
            for dt in (10, 30, 60, 300):
                q2 = kal_at(kq[tk], r.recv + pd.Timedelta(seconds=dt), 30)
                row[f"ask_{dt}"] = None if q2 is None else q2["yes_ask"]
                row[f"mid_{dt}"] = None if q2 is None else (float(q2["yes_bid"] or 0) + float(q2["yes_ask"])) / 2
                row[f"stale_{dt}"] = None if q2 is None else bool(fair - q2["yes_ask"] - taker_fee(q2["yes_ask"]) >= THRESH)
            ev_rows.append(row)
            key = (r.live_id, side)
            if edge >= THRESH and ASK_LO <= ask <= ASK_HI and (
                    key not in last_bet or (r.recv - last_bet[key]).total_seconds() >= DEDUP_S):
                last_bet[key] = r.recv
                bets.append({**row, "price": ask + taker_fee(ask)})
    ev = pd.DataFrame(ev_rows)
    if ev.empty:
        print("no fresh Pinnacle view had a Kalshi quote <= 15 s old"); return {}
    traps.check_no_lookahead(ev.assign(f=ev["as_of"], e=ev["decided_at"]), "f", "e")
    print(f"\nfresh views x sides with a Kalshi quote: {len(ev)}; gap = fair - Kalshi mid (c): "
          f"median {100 * ev['gap_mid'].median():+.2f}, |gap| p50 {100 * ev['gap_mid'].abs().median():.2f}, "
          f"p90 {100 * ev['gap_mid'].abs().quantile(.9):.2f}, max {100 * ev['gap_mid'].abs().max():.2f}")
    print("edge = fair - ask - fee: share >= 0: %.3f, >= 1c: %.3f, >= 3c (signal): %.3f" % (
        (ev["edge"] >= 0).mean(), (ev["edge"] >= .01).mean(), (ev["edge"] >= THRESH).mean()))
    mv = ev.dropna(subset=["pin_move"])
    ep = mv[mv["pin_move"].abs() >= MOVE]
    print(f"\nPinnacle MOVE episodes (fair moved >= {100 * MOVE:.0f}c vs previous fresh view; side-level): {len(ep)}")
    if len(ep):
        # did Kalshi already reflect the move at decision? compare gap vs the move size
        ep = ep.assign(already=(ep["gap_mid"].abs() < 0.015))
        print(f"  Kalshi mid already within 1.5c of Pinnacle's new fair at receipt: {ep['already'].mean():.2f}")
        print(f"  |gap| at receipt (c): median {100 * ep['gap_mid'].abs().median():.2f}")
    sig = ev[ev["edge"] >= THRESH]
    print(f"\nsignal views (edge >= {100 * THRESH:.0f}c): {len(sig)}")
    for dt in (10, 30, 60):
        s = sig[f"stale_{dt}"].dropna()
        print(f"  Kalshi ask still >= 3c stale after {dt:>3d}s: {s.mean() if len(s) else float('nan'):.2f} (n {len(s)})")
    out = {"n_ev": len(ev), "n_sig": len(sig), "n_bets": len(bets)}
    b = pd.DataFrame(bets)
    if len(b):
        b["clv60"] = b["mid_60"] - b["ask"]; b["clv300"] = b["mid_300"] - b["ask"]
        b["paper_ev"] = b["fair"] - b["price"]
        if do_settle:
            res = settle(b["ticker"].unique())
            b["won"] = b["ticker"].map(res)
            b["pnl"] = np.where(b["won"].isna(), np.nan, b["won"] - b["price"])
        print(f"\nRULE bets (dedup {DEDUP_S}s): {len(b)} across {b['live_id'].nunique()} games")
        print(f"  paper EV/bet (fair - price): mean {100 * b['paper_ev'].mean():+.2f}c")
        print(f"  CLV vs Kalshi mid +60s: mean {100 * b['clv60'].mean():+.2f}c (n {b['clv60'].notna().sum()}); "
              f"+300s: {100 * b['clv300'].mean():+.2f}c (n {b['clv300'].notna().sum()})")
        if "pnl" in b and b["pnl"].notna().any():
            s = b.dropna(subset=["pnl"])
            print(f"  settled {len(s)}: P&L {s['pnl'].sum():+.2f} on {s['price'].sum():.2f} risked "
                  f"(ROI {s['pnl'].sum() / s['price'].sum():+.1%})")
        cols = ["decided_at", "league", "ticker", "fair", "ask", "price", "edge", "pin_lag_s", "mid_60", "mid_300"] + \
               (["won"] if "won" in b else [])
        print(b[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        b.to_csv(OUT / "bets_report.csv", index=False)
        out.update({"paper_ev": float(b["paper_ev"].mean())})
    ev.to_csv(OUT / "views_report.csv", index=False)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    a.add_argument("--minutes", type=float, default=120)
    a.add_argument("--kal-every", type=float, default=10)
    a.add_argument("--pin-every", type=float, default=30)
    a.add_argument("--out", default=str(OUT))
    b = sub.add_parser("report")
    b.add_argument("--files", nargs="*")
    b.add_argument("--no-settle", action="store_true")
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.minutes, args.kal_every, args.pin_every, Path(args.out))
    else:
        files = args.files or sorted(glob.glob(str(OUT / "*.jsonl")))
        report(files, not args.no_settle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
