"""TB1 READ-ONLY forward logger: Kalshi tennis match-winner books vs Pinnacle tennis moneylines (pre-match), with quote AGE.
Paper/measurement only -- no orders, no keys. Pre-registration: notes/esoccer_tennis.md "PRE-REGISTRATION Part B".

    python scripts/tennis_pin_logger.py tick                 # one pass
    python scripts/tennis_pin_logger.py loop MINUTES [EVERY] # bounded loop for a hand-run session (default every 120 s)
    python scripts/tennis_pin_logger.py settle               # Kalshi result for paper bets (and all snapped events)
    python scripts/tennis_pin_logger.py report               # coverage / freshness / gap distribution / settled ROI

NOT installed in launchd (agent esoccer_tennis; the lead decides deployment). OUT dir = $TENNIS_PIN_OUT or data/tennis_pin.
Every tick: Pinnacle Tennis (sport 33) matchups + markets/straight (Cloudflare-cached, max-age 905 s; age_s = Date -
last-modified), Kalshi OPEN markets of the six tennis match series (kx_throttle; 1 call per series page), match:
  series <-> Pinnacle league prefix; singles only; BOTH players one-to-one by accent-free whole-word token sets (equal, or the
  smaller set has >= 2 tokens and lies inside the larger; never a substring, never one shared token); each Kalshi surname from
  rules_primary ("<A> vs <B> professional tennis match in the <tournament round>") inside its matched Pinnacle name; Pinnacle
  start within [ticker date 00:00 ET - 12 h, + 36 h]; unique both ways.
Outputs (append-only): fetch.jsonl (one row per tick: headers, counts, unmatched lists for the alias audit),
snaps.jsonl (matched matches with Pinnacle start in [t0 - 15 min, t0 + 6 h]: Pinnacle prices, de-vig, Kalshi quotes, edges),
bets.jsonl (TB1 paper decisions), settle.jsonl (settlement of every snapped Kalshi market; bets are a subset)."""
from __future__ import annotations

import json
import math
import os
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
OUT = Path(os.environ.get("TENNIS_PIN_OUT", str(ROOT / "data" / "tennis_pin")))
STATE = OUT / "decided.json"
SERIES = ["KXATPCHALLENGERMATCH", "KXITFMATCH", "KXITFWMATCH", "KXWTACHALLENGERMATCH", "KXATPMATCH", "KXWTAMATCH"]
SNAP_AHEAD = timedelta(hours=6)
SNAP_BEHIND = timedelta(minutes=15)
WIN_LO, WIN_HI = timedelta(minutes=90), timedelta(minutes=10)   # decision window: [start - 90, start - 10]
FRESH_S = 120
EDGE_MIN = 0.03
ET = ZoneInfo("America/New_York")
MON = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def now() -> datetime:
    return datetime.now(timezone.utc)


def iso(t: datetime | None) -> str | None:
    return t.astimezone(timezone.utc).isoformat() if t else None


def series_of(league: str) -> str | None:
    l = league.lower()
    if l.startswith("atp challenger"):
        return "KXATPCHALLENGERMATCH"
    if l.startswith("itf men"):
        return "KXITFMATCH"
    if l.startswith("itf women"):
        return "KXITFWMATCH"
    if l.startswith("wta 125"):
        return "KXWTACHALLENGERMATCH"
    if l.startswith("atp ") or l.startswith("davis cup"):
        return "KXATPMATCH"
    if l.startswith("wta ") or l.startswith("billie jean king cup"):
        return "KXWTAMATCH"
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
    return {"mult": mult, "power": pw, "cons": (min(mult[0], pw[0]), min(mult[1], pw[1])), "overround": t}


def toks(s) -> frozenset:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'`´’]", "", s)
    return frozenset(re.sub(r"[^a-z0-9]+", " ", s).split())


def same_player(a, b) -> str | None:
    x, y = toks(a), toks(b)
    if not x or not y:
        return None
    if x == y:
        return "exact"
    small, big = (x, y) if len(x) <= len(y) else (y, x)
    if len(small) >= 2 and small <= big:
        return "subset"
    return None


RULE_RE = re.compile(r"wins the (.+?) vs (.+?) professional tennis match in the (.+?) after", re.I)


def kalshi_date(event_ticker: str) -> datetime | None:
    m = re.match(r"^[A-Z0-9]+-(\d{2})([A-Z]{3})(\d{2})", event_ticker)
    if not m or m.group(2) not in MON:
        return None
    try:
        return datetime(2000 + int(m.group(1)), MON[m.group(2)], int(m.group(3)), tzinfo=ET).astimezone(timezone.utc)
    except ValueError:
        return None


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
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


def pinnacle():
    mu, mu_meta = pin_get("/sports/33/matchups")
    mk, mk_meta = pin_get("/sports/33/markets/straight")
    games = {}
    for m in mu:
        if m.get("type") != "matchup" or m.get("parentId"):
            continue
        side = {p.get("alignment"): p.get("name") for p in m.get("participants", [])}
        lg = m.get("league", {}).get("name", "")
        ser = series_of(lg)
        if "home" not in side or "away" not in side or ser is None or "/" in side["home"] or "/" in side["away"]:
            continue
        st = datetime.fromisoformat(m["startTime"].replace("Z", "+00:00"))
        games[m["id"]] = {"pin_id": m["id"], "league": lg, "series": ser, "start": st, "home": side["home"],
                          "away": side["away"], "live": m.get("isLive"), "status": m.get("status"), "ml": None}
    for x in mk:
        g = games.get(x.get("matchupId"))
        if g is None or x.get("type") != "moneyline" or x.get("isAlternate") or x.get("period") != 0:
            continue
        am = {p.get("designation"): p.get("price") for p in x.get("prices", [])}
        if set(am) != {"home", "away"}:
            continue
        g["ml"] = {"home": am["home"], "away": am["away"], "version": x.get("version"), "status": x.get("status"),
                   "cutoff": x.get("cutoffAt"),
                   "max_risk": next((l["amount"] for l in x.get("limits", []) if l.get("type") == "maxRiskStake"), None)}
    return games, mu_meta, mk_meta


KEEP = ("ticker", "yes_sub_title", "yes_bid_dollars", "yes_ask_dollars", "yes_bid_size_fp", "yes_ask_size_fp",
        "last_price_dollars", "volume_fp", "open_interest_fp", "status", "open_time")


def kalshi_events():
    evs = {}
    ncall = 0
    for ser in SERIES:
        cur = None
        for _ in range(10):
            p = {"series_ticker": ser, "status": "open", "limit": 1000}
            if cur:
                p["cursor"] = cur
            j = kget("/markets", p) or {}
            ncall += 1
            for m in j.get("markets", []):
                e = evs.setdefault(m["event_ticker"], {"event": m["event_ticker"], "series": ser,
                                                       "date": kalshi_date(m["event_ticker"]), "mkts": [], "names": None,
                                                       "tourn": None})
                rm = RULE_RE.search(m.get("rules_primary") or "")
                if rm and e["names"] is None:
                    e["names"], e["tourn"] = [rm.group(1).strip(), rm.group(2).strip()], rm.group(3).strip()
                e["mkts"].append({k: m.get(k) for k in KEEP})
            cur = j.get("cursor")
            if not cur or not j.get("markets"):
                break
    return evs, ncall


def match(games: dict, evs: dict):
    """-> list of (pin_id, event, {home: mkt, away: mkt}, quality)."""
    out = []
    for pid, g in games.items():
        hits = []
        for ev, e in evs.items():
            if e["series"] != g["series"] or len(e["mkts"]) != 2 or e["date"] is None:
                continue
            if not (e["date"] - timedelta(hours=12) <= g["start"] <= e["date"] + timedelta(hours=36)):
                continue
            mh = [same_player(g["home"], m["yes_sub_title"]) for m in e["mkts"]]
            ma = [same_player(g["away"], m["yes_sub_title"]) for m in e["mkts"]]
            bh, ba = [bool(x) for x in mh], [bool(x) for x in ma]
            if sum(bh) != 1 or sum(ba) != 1 or bh == ba:
                continue
            ih = bh.index(True)
            # surname check: every Kalshi title surname token set must sit inside one of the two matched Pinnacle names
            if e["names"]:
                pin_t = [toks(g["home"]), toks(g["away"])]
                ok = all(any(toks(n) <= pt for pt in pin_t) for n in e["names"])
                if not ok:
                    continue
            q = "exact" if mh[ih] == "exact" and ma[1 - ih] == "exact" else "subset"
            hits.append((ev, {"home": e["mkts"][ih], "away": e["mkts"][1 - ih]}, q))
        if len(hits) == 1:
            out.append((pid, *hits[0]))
    used = [x[1] for x in out]
    return [x for x in out if used.count(x[1]) == 1]


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
        games, mu_meta, mk_meta = pinnacle()
    except Exception as e:
        fetch["pin_err"] = f"{e.__class__.__name__}: {e}"[:200]
        games, mu_meta, mk_meta = {}, {}, {}
    fetch.update({"pin_matchups": mu_meta, "pin_markets": mk_meta, "n_pin": len(games),
                  "n_pin_ml": sum(1 for g in games.values() if g["ml"])})
    try:
        evs, ncall = kalshi_events()
    except Exception as e:
        fetch["kal_err"] = f"{e.__class__.__name__}: {e}"[:200]
        evs, ncall = {}, 0
    fetch["n_kal"], fetch["kal_calls"] = len(evs), ncall
    pairs = match(games, evs) if games and evs else []
    age = mk_meta.get("age_s")
    try:
        decided = json.loads(STATE.read_text()) if STATE.exists() else {}
    except Exception:
        decided = {}
    snaps, bets = [], []
    for pid, ev, side, q in pairs:
        g = games[pid]
        if not (t0 - SNAP_BEHIND <= g["start"] <= t0 + SNAP_AHEAD):
            continue
        row = {"ts": iso(t0), "pin_id": pid, "league": g["league"], "series": g["series"], "start": iso(g["start"]),
               "pin_live": g["live"], "pin_status": g["status"], "home": g["home"], "away": g["away"], "ml": g["ml"],
               "age_s": age, "cf_age": mk_meta.get("cf_age"), "pin_last_modified": mk_meta.get("last_modified"),
               "event": ev, "tourn": evs[ev]["tourn"], "match_q": q, "kal": {}}
        fair = None
        if g["ml"] and g["ml"].get("status") == "open":
            fair = devig(am_to_prob(g["ml"]["home"]), am_to_prob(g["ml"]["away"]))
            row["fair"] = {k: ([round(v[0], 5), round(v[1], 5)] if isinstance(v, tuple) else round(v, 5)) for k, v in fair.items()}
        for i, s in enumerate(("home", "away")):
            m = dict(side[s])
            ask = fnum(m.get("yes_ask_dollars"))
            if fair is not None and ask is not None and 0 < ask < 1:
                m["fee"] = kalshi_fee(ask)
                m["edge"] = round(fair["cons"][i] - ask - m["fee"], 5)
            row["kal"][s] = m
        snaps.append(row)
        in_win = (g["start"] - WIN_LO <= t0 <= g["start"] - WIN_HI) and not g["live"]
        if str(pid) in decided or not in_win or age is None or age > FRESH_S or fair is None:
            continue
        cands = []
        for i, s in enumerate(("home", "away")):
            k = row["kal"][s]
            ask, asz = fnum(k.get("yes_ask_dollars")), fnum(k.get("yes_ask_size_fp"))
            if k.get("edge") is not None and 0.03 <= ask <= 0.97 and (asz or 0) >= 1 and k["edge"] >= EDGE_MIN:
                cands.append((k["edge"], s, k, i))
        if cands:
            e, s, k, i = max(cands, key=lambda c: c[0])
            bets.append({"ts": iso(t0), "trial": "tb1_kalshi_tennis_pin_fresh", "pin_id": pid, "event": ev,
                         "ticker": k["ticker"], "player": k["yes_sub_title"], "side": s, "ask": fnum(k["yes_ask_dollars"]),
                         "ask_size": fnum(k["yes_ask_size_fp"]), "fee": k["fee"], "fair_cons": fair["cons"][i],
                         "fair_mult": fair["mult"][i], "fair_power": fair["power"][i], "edge": e, "age_s": age,
                         "start": iso(g["start"]), "league": g["league"], "series": g["series"], "match_q": q,
                         "ml": g["ml"], "book": book(k["ticker"])})
            decided[str(pid)] = iso(t0)
    matched_p = {p for p, *_ in pairs}
    matched_e = {e for _, e, *_ in pairs}
    fetch["matched"] = [[p, e, q, iso(games[p]["start"])] for p, e, _, q in pairs]
    fetch["unmatched_pin"] = [[p, g["league"], g["home"], g["away"], iso(g["start"])] for p, g in games.items()
                              if p not in matched_p]
    fetch["unmatched_kal"] = [[e, v["series"], [m["yes_sub_title"] for m in v["mkts"]], v["tourn"]]
                              for e, v in evs.items() if e not in matched_e]
    fetch["n_bets"] = len(bets)
    append("fetch.jsonl", [fetch])
    append("snaps.jsonl", snaps)
    append("bets.jsonl", bets)
    if bets:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(decided))
    print(f"{iso(t0)} pin {len(games)} (ml {fetch['n_pin_ml']}) kal {len(evs)} matched {len(pairs)} snaps {len(snaps)} "
          f"bets {len(bets)} age {age}", flush=True)
    return 0


def cmd_loop() -> int:
    minutes = float(sys.argv[2])
    every = float(sys.argv[3]) if len(sys.argv) > 3 else 120.0
    end = time.time() + 60 * minutes
    while time.time() < end:
        t = time.time()
        try:
            cmd_tick()
        except Exception as e:  # keep the loop alive; the error is in the log
            print(f"{iso(now())} tick error {e.__class__.__name__}: {e}", flush=True)
        time.sleep(max(5.0, every - (time.time() - t)))
    return 0


def read(name: str) -> list:
    p = OUT / name
    return [json.loads(l) for l in p.open()] if p.exists() else []


def cmd_settle() -> int:
    done = {r["ticker"] for r in read("settle.jsonl")}
    tick = set()
    for r in read("snaps.jsonl"):
        for k in r["kal"].values():
            tick.add(k["ticker"])
    rows = []
    for t in sorted(tick - done):
        j = kget(f"/markets/{t}") or {}
        m = j.get("market", {})
        res = m.get("result")
        if m.get("status") not in ("settled", "finalized") or res not in ("yes", "no", "scalar"):
            continue
        val = 1.0 if res == "yes" else 0.0 if res == "no" else fnum(m.get("settlement_value_dollars"))
        rows.append({"ts": iso(now()), "ticker": t, "result": res, "value": val, "close_time": m.get("close_time")})
    append("settle.jsonl", rows)
    print(f"settled {len(rows)} new markets")
    return 0


def cmd_report() -> int:
    f = read("fetch.jsonl")
    if not f:
        print("no ticks yet")
        return 0
    ages = sorted(r["pin_markets"].get("age_s") for r in f if r.get("pin_markets", {}).get("age_s") is not None)
    print(f"ticks {len(f)} ({f[0]['ts']} .. {f[-1]['ts']}); Pinnacle age_s median {ages[len(ages) // 2] if ages else None}, "
          f"fresh share {sum(a <= FRESH_S for a in ages) / max(len(ages), 1):.2f}")
    s = read("snaps.jsonl")
    first = {}
    for r in s:   # first FRESH tick inside the decision window per match (the pre-registered gap sample)
        st = datetime.fromisoformat(r["start"])
        t = datetime.fromisoformat(r["ts"])
        if r["pin_id"] in first or r.get("fair") is None or r["age_s"] is None or r["age_s"] > FRESH_S or r["pin_live"]:
            continue
        if st - WIN_LO <= t <= st - WIN_HI:
            first[r["pin_id"]] = r
    gaps = {}
    for r in first.values():
        k = r["kal"]["home"]
        b, a = fnum(k.get("yes_bid_dollars")), fnum(k.get("yes_ask_dollars"))
        if b and a and 0 < b <= a < 1:
            gaps.setdefault(r["series"], []).append((b + a) / 2 - r["fair"]["mult"][0])
    for ser, g in sorted(gaps.items()):
        g.sort()
        print(f"  {ser}: n {len(g)} median gap {g[len(g) // 2]:+.3f} |gap| median {sorted(abs(x) for x in g)[len(g) // 2]:.3f}")
    print(f"bets {len(read('bets.jsonl'))}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"tick": cmd_tick, "loop": cmd_loop, "settle": cmd_settle, "report": cmd_report}.get(
        cmd, lambda: print(__doc__) or 1)())
