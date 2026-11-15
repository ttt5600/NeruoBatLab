"""Kalshi vs Pinnacle GAP LOGGER: how often does a Kalshi game price drift away from Pinnacle's no-vig price,
and for how long? Paper/measurement only -- no orders.

    python scripts/gap_logger.py tick      # one pass (launchd runs this every 2 min)
    python scripts/gap_logger.py report    # gap episodes: count, duration, EV at open

Why (user 2026-10-19: "if Kalshi moves around shouldn't there be opportunities?"): one-off snapshots found 0 quotes
>= 1% EV (US sports 10-12, soccer 10-19, median gap 0.34c), but a gap that opens and closes inside 10 minutes
is invisible to a snapshot. This samples every 2 min and logs every pre-game contract side with EV >= 1% after
the taker fee, valued at the CONSERVATIVE de-vig (US 2-way: worse of multiplicative/power; soccer 3-way: power).
An EPISODE = consecutive ticks (gap <= 5 min) where the same contract side stays >= 1%.
Caveat: Pinnacle is the reference, so an episode can also be Pinnacle being stale, not Kalshi -- the guest API is
served from a Cloudflare cache (max-age 905 s; Age headers of 230-770 s seen 2026-11-15), so every tick now logs
the quote age too.

Raw Pinnacle odds (added 2026-11-15, ops_gap_logger) go to data/gap_pin_odds.jsonl so Shin vs power de-vig and
Pinnacle-close CLV are computable from history:
  {"kind":"fetch", ts, league, age_s, cf_age, cf, last_modified, n, ids}   every tick, every league (ids = open games)
  {"kind":"odds",  ts, league, pin_id, start, home, away, event, prices, version, max_risk, age_s}
                    only when a game's prices/version/Kalshi mapping changed since the last logged row (lossless:
                    the price at time t is the last odds row <= t while the id is still in that tick's fetch row).
age_s = Date - last-modified of the markets response (how old Pinnacle's numbers were when we read them).
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
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
ODDS = ROOT / "data" / "gap_pin_odds.jsonl"   # raw Pinnacle odds + age (see docstring)
ODDS_LAST = ROOT / "data" / "gap_pin_last.json"  # pin_id -> fingerprint of the last odds row written
EV_MIN = 0.01
MIN_LEAD = pd.Timedelta(minutes=10)
SOCCER = {"epl": (1980, "KXEPLGAME"), "laliga": (2196, "KXLALIGAGAME"), "seriea": (2436, "KXSERIEAGAME"),
          "bund": (1842, "KXBUNDESLIGAGAME"), "ligue1": (2036, "KXLIGUE1GAME"), "efl": (1977, "KXEFLCHAMPIONSHIPGAME"),
          "mls": (2663, "KXMLSGAME"), "ucl": (2627, "KXUCLGAME"), "ered": (1928, "KXEREDIVISIEGAME")}


def _power(r):
    k = brentq(lambda k: (r ** k).sum() - 1, 0.3, 4.0)
    return r ** k


# ---------------------------------------------------------------- team matching (Kalshi <-> Pinnacle, soccer)
# Pre-2026-11-15 the matcher reduced each name to its FIRST token and tested SUBSTRING containment: "West Ham" and
# "West Bromwich" both became "west" (fake +146% EV row, WHU-WBA 10-31), and "le" (Le Mans) is a substring of
# "lille" (Lille NO priced off Le Mans' fair -> fake +47% on 11-15). Now: normalized token sets + an alias table,
# and an event matches only when the two Pinnacle teams map ONE-TO-ONE onto the two Kalshi teams on the same date.
STOP = {"fc", "afc", "cf", "sc", "sv", "ac", "club", "the", "de", "la", "calcio", "fk", "cd", "ud"}
ALIAS = {  # normalized full name -> canonical normalized name (applied to both venues before STOP removal)
    "psg": "paris st germain", "inter": "internazionale", "inter milan": "internazionale",
    "mgladbach": "monchengladbach", "borussia monchengladbach": "monchengladbach", "qpr": "queens park rangers",
    "eindhoven": "psv", "psv eindhoven": "psv", "enschede": "twente", "ga eagles": "go ahead eagles",
    "stade rennais": "rennes", "stade brest 29": "brest", "stade brest": "brest", "strasbourg alsace": "strasbourg",
    "hamburg": "hamburger", "hamburger sv": "hamburger", "bodoe glimt": "bodo glimt", "sabah masazir": "sabah",
    "sabah fk": "sabah", "los angeles f": "lafc", "los angeles fc": "lafc", "los angeles g": "la galaxy",
    "los angeles galaxy": "la galaxy", "new york rb": "new york red bulls", "wolves": "wolverhampton",
    "man utd": "manchester united", "man united": "manchester united", "man city": "manchester city",
    "spurs": "tottenham", "nottm forest": "nottingham forest", "west brom": "west bromwich",
}


def team_tokens(s) -> frozenset:
    s = re.sub("[´’‘]", "", str(s))  # M´gladbach -> Mgladbach (NFKD would turn ´ into a space)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().replace("&", " and ")
    s = re.sub(r"[.'`]", "", s)  # D.C. -> dc, Nott'm -> nottm
    s = " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())
    s = re.sub(r"\bsaint\b", "st", s)
    s = ALIAS.get(s, s)
    return frozenset(t for t in s.split() if t not in STOP and not t.isdigit())


def same_team(a, b) -> bool:
    """Token-set match: equal, or one name's tokens all inside the other's ("Newcastle" ~ "Newcastle United").
    Never a substring of a token, never a first-word match. Ambiguity is resolved at the event level."""
    x, y = team_tokens(a), team_tokens(b)
    return bool(x) and bool(y) and (x <= y or y <= x)


def _event_dates(start: pd.Timestamp) -> set:
    return {start.tz_convert(tz).strftime("%y%b%d").upper() for tz in ("UTC", "America/New_York", "Europe/London")}


def match_soccer_event(start: pd.Timestamp, home: str, away: str, k: pd.DataFrame):
    """-> (event, {ticker: 'H'|'D'|'A'}) or None.  k = live.kalshi() frame (ticker, event, team).
    Requires: Kalshi event date (in the ticker) == Pinnacle start date (UTC/ET/UK), exactly 2 team markets + 1 TIE,
    home matches exactly one Kalshi team, away exactly the other (a Kalshi team matching both Pinnacle teams fails),
    and exactly ONE event on those dates satisfies all of that."""
    if k.empty:
        return None
    day = k.event.str.split("-").str[1].str[:7]
    hits = []
    for ev, gg in k[day.isin(_event_dates(start))].groupby("event"):
        tie = gg[gg.ticker.str.endswith("-TIE")]
        teams = gg[~gg.ticker.str.endswith("-TIE")]
        if len(tie) != 1 or len(teams) != 2:
            continue
        mh = [same_team(home, t) for t in teams.team]
        ma = [same_team(away, t) for t in teams.team]
        if sum(mh) == 1 and sum(ma) == 1 and mh != ma:
            side = {tie.ticker.iloc[0]: "D"}
            for t, h in zip(teams.ticker, mh):
                side[t] = "H" if h else "A"
            hits.append((ev, side))
    return hits[0] if len(hits) == 1 else None


# ---------------------------------------------------------------- raw Pinnacle odds + cache age
HDRS: dict = {}  # url -> headers of the last Pinnacle GET (captured by a requests session hook)


def _hook(r, *a, **kw):
    if r.url.startswith(PIN):
        h = r.headers
        HDRS[r.url.split("?")[0]] = {"date": h.get("Date"), "last_modified": h.get("last-modified"),
                                     "age": h.get("Age"), "cf": h.get("cf-cache-status")}
    return r


if _hook not in live._s.hooks["response"]:
    live._s.hooks["response"].append(_hook)


def _age(lid: int) -> dict:
    """Age of the markets response: age_s = Date - last-modified (origin generation), plus Cloudflare's Age."""
    h = HDRS.get(f"{PIN}/leagues/{lid}/markets/straight", {})
    out = {"age_s": None, "cf_age": int(h["age"]) if str(h.get("age") or "").isdigit() else None,
           "cf": h.get("cf"), "last_modified": h.get("last_modified")}
    try:
        out["age_s"] = int((pd.Timestamp(h["date"]) - pd.Timestamp(h["last_modified"])).total_seconds())
    except Exception:
        pass
    return out


# ---------------------------------------------------------------- per-league quotes
def soccer(lg: str):
    """-> (quotes frame, raw odds rows)."""
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
    rows, odds = [], []
    for x in mk:
        g = games.get(x["matchupId"])
        if g is None or x.get("type") != "moneyline" or x.get("period") != 0 or x.get("isAlternate") or x.get("status") != "open":
            continue
        am = {p["designation"]: p["price"] for p in x["prices"]}
        if set(am) != {"home", "draw", "away"}:
            continue
        pr = {d: am_to_dec(v) for d, v in am.items()}
        hit = match_soccer_event(g[0], g[1], g[2], k)
        odds.append({"pin_id": x["matchupId"], "start": g[0].isoformat(), "home": g[1], "away": g[2],
                     "event": hit[0] if hit else None, "prices": {"H": am["home"], "D": am["draw"], "A": am["away"]},
                     "version": x.get("version"),
                     "max_risk": next((l["amount"] for l in x.get("limits", []) if l["type"] == "maxRiskStake"), None)})
        if hit is None:
            continue
        fair = dict(zip("HDA", _power(np.array([1 / pr["home"], 1 / pr["draw"], 1 / pr["away"]]))))
        ev, sides = hit
        for _, r in k[k.event == ev].iterrows():
            s = sides[r.ticker]
            rows.append({"ticker": r.ticker, "event": ev, "start": g[0], "fair_lo": fair[s], "fair_hi": fair[s],
                         "yes_bid": r.yes_bid, "yes_ask": r.yes_ask, "no_ask": r.no_ask})
    return pd.DataFrame(rows), odds


def us(lg: str):
    """-> (quotes frame, raw odds rows). Off-season leagues (no Pinnacle or no Kalshi markets) -> empty, no error."""
    pin = live.pinnacle(lg)
    if pin.empty or "market" not in pin:
        return pd.DataFrame(), []
    ml = pin[pin.market == "moneyline"]
    kal = live.kalshi(lg)
    m = live.match_kalshi(pin, kal) if not kal.empty and "event" in kal else pd.DataFrame()
    evmap = dict(zip(m.pin_id, m.event)) if not m.empty else {}
    odds = [{"pin_id": int(r.pin_id), "start": r.start.isoformat(), "home": r.home, "away": r.away,
             "event": evmap.get(r.pin_id), "prices": {"H": r.price1, "A": r.price2}, "version": None,
             "max_risk": r.max_risk, "live": bool(r.live)} for r in ml.itertuples()]
    if m.empty:
        return m, odds
    return pd.DataFrame({"ticker": m.ticker, "event": m.event, "start": m.start,
                         "fair_lo": np.minimum(m.fair, m.pfair), "fair_hi": np.maximum(m.fair, m.pfair),
                         "yes_bid": m.yes_bid, "yes_ask": m.yes_ask, "no_ask": m.no_ask}), odds


def _err(e: Exception) -> str:
    code = getattr(getattr(e, "response", None), "status_code", None)
    return f"err {e.__class__.__name__}" + (f" {code}" if code else "")


def _write_odds(ts: pd.Timestamp, lg: str, odds: list, out: list, last: dict) -> None:
    lid = live.LEAGUES[lg][0] if lg not in SOCCER else SOCCER[lg][0]
    age = _age(lid)
    out.append({"kind": "fetch", "ts": ts.isoformat(), "league": lg, **age, "n": len(odds),
                "ids": [o["pin_id"] for o in odds]})
    for o in odds:
        fp = json.dumps([o["prices"], o["version"], o["event"]], sort_keys=True, default=str)
        if last.get(str(o["pin_id"])) != fp:
            last[str(o["pin_id"])] = fp
            out.append({"kind": "odds", "ts": ts.isoformat(), "league": lg, **o, "age_s": age["age_s"]})


def cmd_tick() -> int:
    ts = live.now()
    seen, flagged, odds_out = {}, [], []
    try:
        last = json.loads(ODDS_LAST.read_text()) if ODDS_LAST.exists() else {}
    except Exception:
        last = {}
    for lg in list(live.LEAGUES)[:6] + list(SOCCER):
        try:
            q, odds = soccer(lg) if lg in SOCCER else us(lg)
        except Exception as e:  # one league down must not stop the others
            seen[lg] = _err(e)
            continue
        try:
            _write_odds(ts, lg, odds, odds_out, last)
        except Exception as e:  # the raw-odds side log must never break the gap log
            seen[f"{lg}_odds"] = _err(e)
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
    if odds_out:
        with ODDS.open("a") as f:
            for r in odds_out:
                f.write(json.dumps(r, default=str) + "\n")
        live_ids = {str(i) for r in odds_out if r["kind"] == "fetch" for i in r["ids"]}
        ODDS_LAST.write_text(json.dumps({k: v for k, v in last.items() if k in live_ids}))
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
    g = pd.read_json(LOG, lines=True)
    g["ts"] = pd.to_datetime(g["ts"], utc=True, format="ISO8601")
    g = g.sort_values(["ticker", "side", "ts"])
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
