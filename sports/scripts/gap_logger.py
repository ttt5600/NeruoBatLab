"""Kalshi vs Pinnacle GAP LOGGER: how often does a Kalshi game price drift away from Pinnacle's no-vig price,
and for how long? Paper/measurement only -- no orders.

    python scripts/gap_logger.py tick      # one pass (launchd runs this every 2 min)
    python scripts/gap_logger.py report    # gap episodes: count, duration, EV at open

Why (user 2026-10-19: "if Kalshi moves around shouldn't there be opportunities?"): one-off snapshots found 0 quotes
>= 1% EV (US sports 10-12, soccer 10-19, median gap 0.34c), but a gap that opens and closes inside 10 minutes
is invisible to a snapshot. This samples every 2 min and logs every pre-game contract side with EV >= 1% after
the taker fee, valued at the CONSERVATIVE de-vig (US 2-way: worse of multiplicative/power; soccer 3-way: power).
An EPISODE = consecutive ticks (gap <= 5 min) where the same contract side stays >= 1%.
Caveat: Pinnacle is the reference, so an episode can also be Pinnacle being stale, not Kalshi.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from sports import live
from sports.live import PIN, PIN_HEADERS, _get, am_to_dec, kalshi_fee

LOG = ROOT / "data" / "gap_log.jsonl"      # one line per flagged contract side per tick
TICKS = ROOT / "data" / "gap_ticks.jsonl"  # one line per tick: quotes seen per league (the denominator)
EV_MIN = 0.01
MIN_LEAD = pd.Timedelta(minutes=10)
SOCCER = {"epl": (1980, "KXEPLGAME"), "laliga": (2196, "KXLALIGAGAME"), "seriea": (2436, "KXSERIEAGAME"),
          "bund": (1842, "KXBUNDESLIGAGAME"), "ligue1": (2036, "KXLIGUE1GAME"), "efl": (1977, "KXEFLCHAMPIONSHIPGAME"),
          "mls": (2663, "KXMLSGAME"), "ucl": (2627, "KXUCLGAME"), "ered": (1928, "KXEREDIVISIEGAME")}


def _power(r):
    k = brentq(lambda k: (r ** k).sum() - 1, 0.3, 4.0)
    return r ** k


def _first(s):
    s = re.sub(r"[^a-z ]", "", str(s).lower().replace("&", "and"))
    for w in ("fc", "afc", "cf", "sc", "club", "de", "the", "united", "city"):
        s = re.sub(rf"\b{w}\b", "", s)
    return s.split()[0] if s.split() else s


def soccer(lg: str) -> pd.DataFrame:
    lid, ser = SOCCER[lg]
    mu = _get(f"{PIN}/leagues/{lid}/matchups", headers=PIN_HEADERS)
    mk = _get(f"{PIN}/leagues/{lid}/markets/straight", headers=PIN_HEADERS)
    live.LEAGUES.setdefault(lg, (lid, ser, None))
    k = live.kalshi(lg)
    games = {}
    for m in mu:
        if m.get("type") != "matchup" or m.get("parentId") or m.get("isLive"):
            continue
        side = {p["alignment"]: p["name"] for p in m.get("participants", [])}
        if "home" in side and "away" in side:
            games[m["id"]] = (pd.Timestamp(m["startTime"]), side["home"], side["away"])
    rows = []
    for x in mk:
        g = games.get(x["matchupId"])
        if g is None or x.get("type") != "moneyline" or x.get("period") != 0 or x.get("isAlternate") or x.get("status") != "open":
            continue
        pr = {p["designation"]: am_to_dec(p["price"]) for p in x["prices"]}
        if set(pr) != {"home", "draw", "away"} or k.empty:
            continue
        fair = dict(zip("HDA", _power(np.array([1 / pr["home"], 1 / pr["draw"], 1 / pr["away"]]))))
        hn, an = _first(g[1]), _first(g[2])
        cand = k[k.event.str.contains(g[0].strftime("%y%b%d").upper(), regex=False)]
        evs = [e for e, gg in cand.groupby("event")
               if any(hn in _first(t) for t in gg.team) and any(an in _first(t) for t in gg.team)]
        if len(evs) != 1:
            continue
        for _, r in cand[cand.event == evs[0]].iterrows():
            s = "D" if r.ticker.endswith("-TIE") else ("H" if hn in _first(r.team) else "A")
            rows.append({"ticker": r.ticker, "event": evs[0], "start": g[0], "fair_lo": fair[s], "fair_hi": fair[s],
                         "yes_bid": r.yes_bid, "yes_ask": r.yes_ask, "no_ask": r.no_ask})
    return pd.DataFrame(rows)


def us(lg: str) -> pd.DataFrame:
    m = live.match_kalshi(live.pinnacle(lg), live.kalshi(lg))
    if m.empty:
        return m
    return pd.DataFrame({"ticker": m.ticker, "event": m.event, "start": m.start,
                         "fair_lo": np.minimum(m.fair, m.pfair), "fair_hi": np.maximum(m.fair, m.pfair),
                         "yes_bid": m.yes_bid, "yes_ask": m.yes_ask, "no_ask": m.no_ask})


def cmd_tick() -> int:
    ts = live.now()
    seen, flagged = {}, []
    for lg in list(live.LEAGUES)[:6] + list(SOCCER):
        try:
            q = soccer(lg) if lg in SOCCER else us(lg)
        except Exception as e:  # one league down must not stop the others
            seen[lg] = f"err {e.__class__.__name__}"
            continue
        if q.empty:
            seen[lg] = 0
            continue
        q = q[(q.start - ts > MIN_LEAD) & q.yes_ask.between(0.02, 0.98) & q.no_ask.between(0.02, 0.98)]
        seen[lg] = int(len(q))
        ev_yes = (q.fair_lo - q.yes_ask - kalshi_fee(q.yes_ask)) / (q.yes_ask + kalshi_fee(q.yes_ask))
        ev_no = ((1 - q.fair_hi) - q.no_ask - kalshi_fee(q.no_ask)) / (q.no_ask + kalshi_fee(q.no_ask))
        for side, ev in (("yes", ev_yes), ("no", ev_no)):
            for (_, r), e in zip(q[ev >= EV_MIN].iterrows(), ev[ev >= EV_MIN]):
                flagged.append({"ts": ts.isoformat(), "league": lg, "ticker": r.ticker, "side": side, "ev": round(float(e), 4),
                                "ask": float(r.yes_ask if side == "yes" else r.no_ask), "fair_lo": round(float(r.fair_lo), 4),
                                "fair_hi": round(float(r.fair_hi), 4), "start": r.start.isoformat()})
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with TICKS.open("a") as f:
        f.write(json.dumps({"ts": ts.isoformat(), "seen": seen, "flagged": len(flagged)}) + "\n")
    if flagged:
        with LOG.open("a") as f:
            for r in flagged:
                f.write(json.dumps(r) + "\n")
        print(f"{ts:%F %T} flagged {len(flagged)}: " + ", ".join(f"{r['ticker']} {r['side']} {r['ev']:+.1%}" for r in flagged[:6]))
    return 0


def cmd_report() -> int:
    t = pd.read_json(TICKS, lines=True) if TICKS.exists() else pd.DataFrame()
    if t.empty:
        print("no ticks yet")
        return 0
    quotes = sum(sum(v for v in s.values() if isinstance(v, int)) for s in t.seen)
    print(f"ticks {len(t)} ({t.ts.min()} .. {t.ts.max()}); contract-quotes checked {quotes}; ticks with a flag "
          f"{int((t.flagged > 0).sum())}")
    if not LOG.exists():
        print("no gaps >= 1% EV logged")
        return 0
    g = pd.read_json(LOG, lines=True).sort_values(["ticker", "side", "ts"])
    g["new"] = (g.groupby(["ticker", "side"]).ts.diff() > pd.Timedelta(minutes=5)) | g.groupby(["ticker", "side"]).ts.diff().isna()
    g["ep"] = g.new.cumsum()
    ep = g.groupby("ep").agg(league=("league", "first"), ticker=("ticker", "first"), side=("side", "first"),
                             opened=("ts", "min"), closed=("ts", "max"), n=("ts", "size"), ev0=("ev", "first"), evmax=("ev", "max"))
    ep["minutes"] = (ep.closed - ep.opened).dt.total_seconds() / 60
    print(f"episodes {len(ep)}; median duration {ep.minutes.median():.0f} min (0 = seen on one tick only); "
          f"median EV at open {ep.ev0.median():+.1%}")
    print(ep.groupby("league").agg(episodes=("n", "size"), med_min=("minutes", "median"), med_ev=("ev0", "median")).round(3).to_string())
    print(ep.sort_values("evmax", ascending=False).head(10).to_string())
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"tick": cmd_tick, "report": cmd_report}.get(cmd, lambda: print(__doc__) or 1)())
