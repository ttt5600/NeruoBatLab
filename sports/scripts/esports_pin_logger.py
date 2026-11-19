"""ESP1 READ-ONLY forward logger: Kalshi esports match-winner books vs Pinnacle esports moneylines, with quote AGE.
Paper/measurement only -- no orders, no keys. Pre-registration: notes/esports_pinnacle.md.

    python scripts/esports_pin_logger.py tick     # one pass (launchd every 120 s via qs_job.sh esports-pin)
    python scripts/esports_pin_logger.py settle   # Kalshi result for paper bets not yet settled
    python scripts/esports_pin_logger.py report   # coverage / freshness / signals / settled ROI

Every tick: Pinnacle E Sports (sport 12) matchups + markets/straight (Cloudflare-cached; we log Date, last-modified,
Age, cf-cache-status; age_s = Date - last-modified), Kalshi open markets of KXCS2GAME/KXLOLGAME/KXVALORANTGAME/
KXDOTA2GAME (kx_throttle), match them (game + start +-90 min + one-to-one token-set team match, unique), and for
matched matches starting within 6 h: full Kalshi orderbook (both markets) + Pinnacle prices + de-vigged fair.
Outputs (append-only) under data/esports_pin/:
  fetch.jsonl  one row per tick: headers/age, window counts, matched ids, unmatched Pinnacle + Kalshi (for alias audit)
  snaps.jsonl  one row per matched match per tick (Pinnacle odds, age, Kalshi books, fair, edge per side)
  bets.jsonl   ESP1 paper decisions: FIRST tick per match with pregame & age_s <= 120 & fair - ask - fee >= 0.03
  settle.jsonl settlement of each paper bet (yes 1 / no 0 / scalar = settlement_value)
Idle ticks (no Pinnacle or Kalshi match in window) still write the fetch row (that is the coverage denominator).
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from kx_throttle import kget  # noqa: E402

PIN = "https://guest.api.arcadia.pinnacle.com/0.1"
PIN_HEADERS = {"User-Agent": "Mozilla/5.0", "X-API-Key": "CmX2KcMrXuFmNg6YFbmTxE0y9CIrOi0R"}  # public web-app key
OUT = ROOT / "data" / "esports_pin"
STATE = OUT / "decided.json"
SERIES = {"cs2": "KXCS2GAME", "lol": "KXLOLGAME", "valorant": "KXVALORANTGAME", "dota2": "KXDOTA2GAME"}
WINDOW_AHEAD = timedelta(hours=6)
WINDOW_BEHIND = timedelta(minutes=15)   # keep logging just past start (close context)
START_TOL = timedelta(minutes=90)
PREGAME_CUT = timedelta(minutes=2)
FRESH_S = 120
EDGE_MIN = 0.03
ET = ZoneInfo("America/New_York")
MON = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def now() -> datetime:
    return datetime.now(timezone.utc)


def iso(t: datetime | None) -> str | None:
    return t.astimezone(timezone.utc).isoformat() if t else None


def game_of(league: str) -> str | None:
    l = league.lower()
    if l.startswith("cs2") or l.startswith("cs:go") or l.startswith("counter-strike"):
        return "cs2"
    if l.startswith("league of legends") or l.startswith("lol"):
        return "lol"
    if l.startswith("valorant"):
        return "valorant"
    if l.startswith("dota"):
        return "dota2"
    return None


def kalshi_fee(p: float, n: int = 10) -> float:
    """Per-contract taker fee on an n-lot, rounded up to the cent on the order (Kalshi quadratic 0.07)."""
    return math.ceil(round(100 * 0.07 * n * p * (1 - p), 9)) / (100 * n)


def am_to_prob(a: float) -> float:
    dec = 1 + (a / 100 if a > 0 else 100 / -a)
    return 1 / dec


def devig(p1: float, p2: float) -> dict:
    t = p1 + p2
    mult = (p1 / t, p2 / t)
    lo, hi = 0.3, 4.0
    for _ in range(80):  # power: p1^k + p2^k = 1
        k = (lo + hi) / 2
        if p1 ** k + p2 ** k > 1:
            lo = k
        else:
            hi = k
    k = (lo + hi) / 2
    pw = (p1 ** k, p2 ** k)
    return {"mult": mult, "power": pw, "cons": (min(mult[0], pw[0]), min(mult[1], pw[1]))}


# ---------------------------------------------------------------- team matching (whole-word token sets, no substrings)
STOP = {"team", "esports", "esport", "gaming", "gg", "club", "the", "e", "sports", "fc", "clan"}
ALIAS = {"natus vincere": "navi", "na vi": "navi", "ninjas in pyjamas": "nip", "virtus pro": "virtuspro",
         "virtus.pro": "virtuspro", "vp": "virtuspro", "mousesports": "mouz", "betboom": "betboom", "bb": "betboom",
         "evil geniuses": "eg", "team liquid": "liquid", "faze clan": "faze", "g2 esports": "g2", "fnatic": "fnatic",
         "cloud9": "cloud9", "c9": "cloud9", "team spirit": "spirit", "heroic": "heroic", "100 thieves": "100thieves",
         "100t": "100thieves", "paper rex": "paperrex", "prx": "paperrex", "sentinels": "sentinels",
         "gen g": "geng", "gen.g": "geng", "geng": "geng", "t1": "t1", "hanwha life": "hle", "hanwha life esports": "hle",
         "dplus kia": "dplus", "dk": "dplus", "kt rolster": "kt", "bilibili gaming": "blg", "top esports": "tes",
         "weibo gaming": "wbg", "jd gaming": "jdg", "anyone s legend": "al", "anyones legend": "al"}


def team_tokens(s) -> frozenset:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().replace("&", " and ")
    s = re.sub(r"[.'`´’]", "", s)
    s = " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())
    s = ALIAS.get(s, s)
    toks = frozenset(t for t in s.split() if t not in STOP)
    return toks or frozenset(s.split())


def same_team(a, b) -> str | None:
    """'exact' (equal token sets), 'subset' (one inside the other), or None. Never a token-substring match."""
    x, y = team_tokens(a), team_tokens(b)
    if not x or not y:
        return None
    if x == y:
        return "exact"
    if x <= y or y <= x:
        return "subset"
    return None


def kalshi_start(event_ticker: str) -> datetime | None:
    m = re.match(r"^[A-Z0-9]+-(\d{2})([A-Z]{3})(\d{2})(\d{2})(\d{2})", event_ticker)
    if not m or m.group(2) not in MON:
        return None
    y, mo, d, hh, mm = 2000 + int(m.group(1)), MON[m.group(2)], int(m.group(3)), int(m.group(4)), int(m.group(5))
    try:
        return datetime(y, mo, d, hh, mm, tzinfo=ET).astimezone(timezone.utc)
    except ValueError:
        return None


# ---------------------------------------------------------------- fetch
def pin_get(path: str):
    r = requests.get(f"{PIN}{path}", headers=PIN_HEADERS, timeout=30)
    r.raise_for_status()
    h = r.headers
    meta = {"date": h.get("Date"), "last_modified": h.get("last-modified"), "cf_age": h.get("Age"),
            "cf": h.get("cf-cache-status"), "age_s": None}
    try:
        meta["age_s"] = int((parsedate_to_datetime(h["Date"]) - parsedate_to_datetime(h["last-modified"])).total_seconds())
    except Exception:
        pass
    return r.json(), meta


def pinnacle(t0: datetime):
    mu, mu_meta = pin_get("/sports/12/matchups")
    mk, mk_meta = pin_get("/sports/12/markets/straight")
    games = {}
    for m in mu:
        if m.get("type") != "matchup" or m.get("parentId"):
            continue
        side = {p.get("alignment"): p.get("name") for p in m.get("participants", [])}
        g = game_of(m.get("league", {}).get("name", ""))
        if "home" not in side or "away" not in side or g is None:
            continue
        st = datetime.fromisoformat(m["startTime"].replace("Z", "+00:00"))
        if not (t0 - WINDOW_BEHIND <= st <= t0 + WINDOW_AHEAD):
            continue
        games[m["id"]] = {"pin_id": m["id"], "league": m["league"]["name"], "game": g, "start": st,
                          "home": side["home"], "away": side["away"], "bo": m.get("bestOfX"), "live": m.get("isLive"),
                          "ml": None, "maps": {}}
    for x in mk:
        g = games.get(x.get("matchupId"))
        if g is None or x.get("type") != "moneyline" or x.get("isAlternate"):
            continue
        am = {p.get("designation"): p.get("price") for p in x.get("prices", [])}
        if set(am) != {"home", "away"}:
            continue
        rec = {"home": am["home"], "away": am["away"], "version": x.get("version"), "status": x.get("status"),
               "cutoff": x.get("cutoffAt"),
               "max_risk": next((l["amount"] for l in x.get("limits", []) if l.get("type") == "maxRiskStake"), None)}
        if x.get("period") == 0:
            g["ml"] = rec
        else:
            g["maps"][str(x.get("period"))] = rec
    return games, mu_meta, mk_meta


def kalshi_events(t0: datetime):
    evs = {}
    for g, ser in SERIES.items():
        cur = None
        for _ in range(5):
            p = {"series_ticker": ser, "status": "open", "limit": 200}
            if cur:
                p["cursor"] = cur
            j = kget("/markets", p) or {}
            for m in j.get("markets", []):
                e = evs.setdefault(m["event_ticker"], {"event": m["event_ticker"], "game": g,
                                                       "start": kalshi_start(m["event_ticker"]), "mkts": []})
                e["mkts"].append({k: m.get(k) for k in ("ticker", "yes_sub_title", "yes_bid_dollars", "yes_ask_dollars",
                                                        "yes_bid_size_fp", "yes_ask_size_fp", "last_price_dollars",
                                                        "volume_fp", "open_interest_fp", "status")})
            cur = j.get("cursor")
            if not cur:
                break
    return {k: v for k, v in evs.items()
            if v["start"] is not None and t0 - WINDOW_BEHIND <= v["start"] <= t0 + WINDOW_AHEAD}


def match(games: dict, evs: dict):
    """-> list of (pin_id, event, {home: mkt, away: mkt}, quality). One-to-one teams, unique event, game + time."""
    out = []
    for pid, g in games.items():
        hits = []
        for ev, e in evs.items():
            if e["game"] != g["game"] or len(e["mkts"]) != 2 or abs(e["start"] - g["start"]) > START_TOL:
                continue
            mh = [same_team(g["home"], m["yes_sub_title"]) for m in e["mkts"]]
            ma = [same_team(g["away"], m["yes_sub_title"]) for m in e["mkts"]]
            if sum(bool(x) for x in mh) == 1 and sum(bool(x) for x in ma) == 1 and [bool(x) for x in mh] != [bool(x) for x in ma]:
                ih = [bool(x) for x in mh].index(True)
                q = "exact" if mh[ih] == "exact" and ma[1 - ih] == "exact" else "subset"
                hits.append((ev, {"home": e["mkts"][ih], "away": e["mkts"][1 - ih]}, q))
        if len(hits) == 1:
            out.append((pid, *hits[0]))
    used = [x[1] for x in out]
    return [x for x in out if used.count(x[1]) == 1]   # an event claimed by two Pinnacle matchups -> drop both


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def book(ticker: str) -> dict | None:
    j = kget(f"/markets/{ticker}/orderbook", {"depth": 10})
    if not j:
        return None
    ob = j.get("orderbook_fp") or j.get("orderbook") or {}
    return {k: ob.get(k) for k in ob if k in ("yes_dollars", "no_dollars", "yes", "no")}


# ---------------------------------------------------------------- commands
def append(name: str, rows: list) -> None:
    if not rows:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("a") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def cmd_tick() -> int:
    t0 = now()
    fetch = {"ts": iso(t0)}
    try:
        games, mu_meta, mk_meta = pinnacle(t0)
    except Exception as e:
        fetch["pin_err"] = f"{e.__class__.__name__}: {e}"[:200]
        games, mu_meta, mk_meta = {}, {}, {}
    fetch.update({"pin_matchups": mu_meta, "pin_markets": mk_meta, "n_pin": len(games)})
    try:
        evs = kalshi_events(t0)
    except Exception as e:
        fetch["kal_err"] = f"{e.__class__.__name__}: {e}"[:200]
        evs = {}
    fetch["n_kal"] = len(evs)
    pairs = match(games, evs) if games and evs else []
    age = mk_meta.get("age_s")
    try:
        decided = json.loads(STATE.read_text()) if STATE.exists() else {}
    except Exception:
        decided = {}
    snaps, bets = [], []
    for pid, ev, side, q in pairs:
        g = games[pid]
        row = {"ts": iso(t0), "pin_id": pid, "league": g["league"], "game": g["game"], "start": iso(g["start"]),
               "kal_start": iso(evs[ev]["start"]), "bo": g["bo"], "pin_live": g["live"], "home": g["home"],
               "away": g["away"], "ml": g["ml"], "maps": g["maps"], "age_s": age, "cf_age": mk_meta.get("cf_age"),
               "pin_last_modified": mk_meta.get("last_modified"), "event": ev, "match_q": q, "kal": {}}
        fair = None
        if g["ml"] and g["ml"].get("status") == "open":
            fair = devig(am_to_prob(g["ml"]["home"]), am_to_prob(g["ml"]["away"]))
            row["fair"] = {k: [round(v[0], 5), round(v[1], 5)] for k, v in fair.items()}
        for i, s in enumerate(("home", "away")):
            m = side[s]
            ask, asz = fnum(m.get("yes_ask_dollars")), fnum(m.get("yes_ask_size_fp"))
            k = {**m, "book": book(m["ticker"])}
            if fair is not None and ask is not None and 0 < ask < 1:
                k["fee"] = kalshi_fee(ask)
                k["edge"] = round(fair["cons"][i] - ask - k["fee"], 5)
            row["kal"][s] = k
        snaps.append(row)
        pregame = t0 < g["start"] - PREGAME_CUT and not g["live"]
        if str(pid) in decided or not pregame or age is None or age > FRESH_S or fair is None:
            continue
        cands = []
        for i, s in enumerate(("home", "away")):
            k = row["kal"][s]
            ask, asz = fnum(k.get("yes_ask_dollars")), fnum(k.get("yes_ask_size_fp"))
            if k.get("edge") is not None and 0.03 <= ask <= 0.97 and (asz or 0) >= 1 and k["edge"] >= EDGE_MIN:
                cands.append((k["edge"], s, k, i))
        if cands:
            e, s, k, i = max(cands)
            bets.append({"ts": iso(t0), "trial": "esp1_kalshi_esports_pin_fresh", "pin_id": pid, "event": ev,
                         "ticker": k["ticker"], "team": k["yes_sub_title"], "side": s, "ask": fnum(k["yes_ask_dollars"]),
                         "ask_size": fnum(k["yes_ask_size_fp"]), "fee": k["fee"], "fair_cons": fair["cons"][i],
                         "fair_mult": fair["mult"][i], "fair_power": fair["power"][i], "edge": e, "age_s": age,
                         "start": iso(g["start"]), "league": g["league"], "game": g["game"], "match_q": q,
                         "ml": g["ml"]})
            decided[str(pid)] = iso(t0)
    matched_p = {p for p, *_ in pairs}
    matched_e = {e for _, e, *_ in pairs}
    fetch["matched"] = [[p, e, q] for p, e, _, q in pairs]
    fetch["unmatched_pin"] = [[p, g["league"], g["home"], g["away"], iso(g["start"])] for p, g in games.items() if p not in matched_p]
    fetch["unmatched_kal"] = [[e, [m["yes_sub_title"] for m in v["mkts"]], iso(v["start"]),
                               [[m.get("yes_bid_dollars"), m.get("yes_ask_dollars")] for m in v["mkts"]]]
                              for e, v in evs.items() if e not in matched_e]
    fetch["n_bets"] = len(bets)
    append("fetch.jsonl", [fetch])
    append("snaps.jsonl", snaps)
    append("bets.jsonl", bets)
    if bets:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(decided))
        print(f"{iso(t0)} ESP1 paper bets: " + ", ".join(f"{b['ticker']} ask {b['ask']} edge {b['edge']:+.3f}" for b in bets))
    if pairs and not bets:
        print(f"{iso(t0)} matched {len(pairs)} (age {age}s); no signal")
    return 0


def read(name: str) -> list:
    p = OUT / name
    return [json.loads(l) for l in p.open()] if p.exists() else []


def cmd_settle() -> int:
    done = {r["ticker"] for r in read("settle.jsonl")}
    rows = []
    for b in read("bets.jsonl"):
        if b["ticker"] in done:
            continue
        j = kget(f"/markets/{b['ticker']}") or {}
        m = j.get("market", {})
        res = m.get("result")
        if m.get("status") not in ("settled", "finalized") or res not in ("yes", "no", "scalar"):
            continue
        val = 1.0 if res == "yes" else 0.0 if res == "no" else fnum(m.get("settlement_value_dollars"))
        cost = b["ask"] + b["fee"]
        rows.append({"ts": iso(now()), "ticker": b["ticker"], "pin_id": b["pin_id"], "result": res, "value": val,
                     "cost": cost, "pnl": None if val is None else round(val - cost, 5)})
        done.add(b["ticker"])
    append("settle.jsonl", rows)
    if rows:
        print(f"settled {len(rows)}: " + ", ".join(f"{r['ticker']} {r['result']} {r['pnl']}" for r in rows))
    return 0


def cmd_report() -> int:
    f = read("fetch.jsonl")
    if not f:
        print("no ticks yet")
        return 0
    ages = sorted(r["pin_markets"].get("age_s") for r in f if r.get("pin_markets", {}).get("age_s") is not None)
    days = {}
    for r in f:
        d = days.setdefault(r["ts"][:10], {"ticks": 0, "matched": set(), "bets": 0})
        d["ticks"] += 1
        d["matched"] |= {m[0] for m in r.get("matched", [])}
        d["bets"] += r.get("n_bets", 0)
    print(f"ticks {len(f)} ({f[0]['ts']} .. {f[-1]['ts']}); Pinnacle age_s median "
          f"{ages[len(ages) // 2] if ages else None}, fresh (<= {FRESH_S}s) share "
          f"{sum(a <= FRESH_S for a in ages) / max(len(ages), 1):.2f}")
    for d, v in sorted(days.items()):
        print(f"  {d}: ticks {v['ticks']}, distinct matched matches {len(v['matched'])}, ESP1 bets {v['bets']}")
    s = read("snaps.jsonl")
    edges = [k["edge"] for r in s for k in r["kal"].values() if k.get("edge") is not None]
    if edges:
        edges.sort()
        print(f"snap side-edges n {len(edges)}: median {edges[len(edges) // 2]:+.3f}, >= +0.03 share "
              f"{sum(e >= 0.03 for e in edges) / len(edges):.3f}")
    st = [r for r in read("settle.jsonl") if r["pnl"] is not None]
    if st:
        print(f"settled bets {len(st)}: pnl {sum(r['pnl'] for r in st):+.2f} on cost {sum(r['cost'] for r in st):.2f} "
              f"(ROI {sum(r['pnl'] for r in st) / sum(r['cost'] for r in st):+.1%}; no CI until n >= 30)")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"tick": cmd_tick, "settle": cmd_settle, "report": cmd_report}.get(cmd, lambda: print(__doc__) or 1)())
