"""F4: forward PAPER test of the weather consensus-reject rule (E1/E2) on POLYMARKET US, with the same pick's Kalshi
price logged at the same moment. No orders, no keys, public GET endpoints only.

    python scripts/pmus_weather_paper.py tick       # launchd every 5 min: decide due city-nights, poll maker fills, settle
    python scripts/pmus_weather_paper.py settle     # fill in results (Polymarket + Kalshi) then report
    python scripts/pmus_weather_paper.py report
    python scripts/pmus_weather_paper.py validate   # TODAY's open markets, any hour, logs nothing (exercises the live path)

PRE-REGISTERED 2026-10-28 ~06:09Z (first tick 06:11:31Z), written before this script's first tick and before any F4 row existed
(notes/venue_expansion.md, "F4 forward paper logger"). Trial label: F4_pmus_weather_paper.

CITIES: the 6 Polymarket US daily-HIGH series that match a Kalshi series 1:1 (scripts/venues/xvenue_map.py: same NWS CLI
  station, same 2F brackets, 48/48 brackets matched 10-28): nychigh=KXHIGHNY (KNYC), mdwhigh=KXHIGHCHI (KMDW),
  miahigh=KXHIGHMIA (KMIA), laxhigh=KXHIGHLAX (KLAX), phlhigh=KXHIGHPHIL (KPHL), sfohigh=KXHIGHTSFO (KSFO).
RULE (frozen, identical code: scripts/kalshi_paper.py `rejected`, `live_forecasts`, MODELS, MARGIN=3F):
  for TOMORROW's brackets, each model's forecast max over hours <= 18:00 local (>= 15 hourly values); reject a bracket if
  ALL of GFS/ECMWF/ICON/GEM/NBM (>= 3 available) miss it by > 3F. Forecasts: the same live Open-Meteo call kalshi_paper
  makes (same function), fetched at our decision instant (kalshi_paper logs forecasts only for its picks, so they are
  re-fetched rather than shared; the source is recorded).
DECISION: once per city-night, at the first tick inside 22:00-23:59 CITY-local the night before (kalshi_paper's window).
QUOTE FILTER (per venue, unchanged): yes_bid > 0, yes_ask < 1, yes_ask - yes_bid <= 0.10, yes_ask > 0.01, on the LIVE
  book (never price-history display quotes -- they invent bids on empty books, the 10-22 trap).
VARIANTS (scored on Polymarket rows that pass the rule AND Polymarket's quote filter = "PM picks"):
  T  taker : buy NO at the ask = 1 - best YES bid, fee 0.0695*C*p*(1-p) (Polymarket US taker schedule, eff. 2026-10-19).
             1 unit of cost per pick: win -> 1/(p + fee) - 1, loss -> -1.
  M  maker : post-only NO bid at q = min(NO mid floored to the cent, NO ask - 1c) (= Kalshi E4/M3 price rule), skip if
             q <= 0 or NO ask - q > 5c; maker fee 0; rests until the market stops trading; no taker fallback.
             On Polymarket's single instrument a NO bid at q is a YES offer at L = 1 - q.
             FILL (primary) = "prints through": after the decision instant T, a public trade printed at a YES price
             strictly > L. Key-free evidence (the trade WebSocket needs an API key, so it is NOT used): the book's public
             `stats` polled every 5 min -- lastTradePx with lastTradeSetTime > T, and the session highPx when
             highSetTime > T or the session opened after T. Polling can miss a print between polls that the session
             high does not retain, so fills are a lower bound; this is stated, not corrected.
             Secondary (reported, not the primary): at-or-through (YES print >= L); E4-comparable size-through proxy
             (sum of sharesTraded increments over polls whose lastTradePx >= L, >= 10 contracts).
             Unfilled -> no bet (0 units).
  K  Kalshi same moment (context): for every forecast-rejected bracket, the identical Kalshi bracket's live quote,
             fee 0.07*p*(1-p), and depth, fetched in the same tick seconds apart.
LOGGED per forecast-rejected bracket (whether or not either venue's quote passes): both books' best bid/ask, depth within
  1c/2c of the best YES bid (= contracts a NO taker can take), top-5 YES bid levels, PM stats, both settlement results.
SCORING (report): T units/ROI with a city-night bootstrap; M fill rate, units over fills, and M - T over ALL PM picks
  (unfilled M = 0); PAIRED price question = on picks quoted on both venues, Kalshi NO taker cost minus PM NO taker cost
  (incl. fees), mean in cents with a bootstrap CI and PM-better/equal/worse counts; combined depth = PM + Kalshi depth_1c.
DECISION RULE (fixed now): after >= 100 settled PM picks AND >= 6 weeks, go live on Polymarket US at tier B
  (<= 0.5u/bet) iff T units > 0 (or M units > 0 for the maker route) AND the paired PM-minus-Kalshi taker cost is not
  worse than +0.5c on average. KILL early if >= 9 of the first 100 settled PM picks lose (binomial p ~ .02 at E1's ~4%
  loss rate). Power, stated before any row: at ~+1-3% ROI on ~0.96 NO and ~80 picks/month, ROI significance is NOT
  reachable in months; this test measures price, depth, fill rate and catastrophic failure, not the edge itself
  (the edge is E1/E2's sealed pass on the same rule).
"""
from __future__ import annotations

import fcntl
import json
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import requests

import kalshi_paper as kp  # frozen rule + forecasts + Kalshi depth (same code the live E1/E2 paper agent runs)
import polymarket_us as pm  # key-free Polymarket US gateway reader
from sports import wx

DATA = ROOT / "data"
LOG = DATA / "pmus_weather_paper.jsonl"
DONE = DATA / "pmus_weather_paper_done.json"
STATE = DATA / "pmus_weather_paper_state.json"
LOCK = DATA / "pmus_weather_paper.lock"
LABEL = "F4_pmus_weather_paper"
PM_FEE = 0.0695
CITIES = {  # pm city -> (Kalshi series, station, tz)
    "nychigh": ("KXHIGHNY", "KNYC", "America/New_York"), "mdwhigh": ("KXHIGHCHI", "KMDW", "America/Chicago"),
    "miahigh": ("KXHIGHMIA", "KMIA", "America/New_York"), "laxhigh": ("KXHIGHLAX", "KLAX", "America/Los_Angeles"),
    "phlhigh": ("KXHIGHPHIL", "KPHL", "America/New_York"), "sfohigh": ("KXHIGHTSFO", "KSFO", "America/Los_Angeles"),
}
H = {"User-Agent": "Mozilla/5.0 (research; read-only)"}


def pm_fee(p: float) -> float:
    return PM_FEE * p * (1 - p)


def k_fee(p: float) -> float:
    return wx.FEE_RATE * p * (1 - p)


def cent_floor(x: float) -> float:
    return math.floor(round(x * 100, 6)) / 100


def quote_ok(bid: float, ask: float) -> bool:
    return bid > 0 and ask < 1 and ask - bid <= 0.10 + 1e-9 and ask > 0.01


def maker_q(bid: float, ask: float) -> float | None:
    """E4/M3 price: NO bid at min(NO mid floored, NO ask - 1c); None if it would improve the NO ask by > 5c."""
    no_ask, no_bid = 1 - bid, 1 - ask
    q = round(min(cent_floor((no_ask + no_bid) / 2), no_ask - 0.01), 4)
    return None if q <= 0 or no_ask - q > 0.05 + 1e-9 else q


def depth(levels: list[tuple[float, float]]) -> tuple[float, float]:
    if not levels:
        return 0.0, 0.0
    b = levels[0][0]
    return (sum(q for p, q in levels if p >= b - 0.0101), sum(q for p, q in levels if p >= b - 0.0201))


def _px(stats: dict, k: str) -> float | None:
    v = (stats or {}).get(k)
    return float(v["value"]) if isinstance(v, dict) and v.get("value") is not None else None


def _ts(stats: dict, k: str) -> datetime | None:
    v = (stats or {}).get(k)
    if not v:
        return None
    v = v.rstrip("Z")
    if "." in v:
        head, frac = v.split(".", 1)
        v = f"{head}.{frac[:6]}"
    return datetime.fromisoformat(v).replace(tzinfo=timezone.utc)


def kget(path: str, params: dict | None = None, tries: int = 5) -> dict:
    """Kalshi public GET with backoff on 429/5xx (a 429 body has no 'markets' and would look like an empty event)."""
    for i in range(tries):
        r = requests.get(f"{wx.API}{path}", headers=H, timeout=30, params=params)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(1.5 * 2 ** i)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"kalshi {path}: gave up after {tries} tries")


def kalshi_event(series: str, day) -> dict:
    ev = f"{series}-{day.strftime('%y%b%d').upper()}"
    ms = kget("/markets", {"event_ticker": ev, "limit": 100})
    out = {}
    for m in ms.get("markets", []):
        f = lambda k: None if m.get(k) is None else float(m[k])  # noqa: E731
        out[(m.get("strike_type"), f("floor_strike"), f("cap_strike"))] = m
    return out


def _key(br) -> tuple:
    st, fl, ca = br
    return st, (None if fl != fl else fl), (None if ca != ca else ca)


def kalshi_levels(ticker: str) -> list[tuple[float, float]]:
    d = kget(f"/markets/{ticker}/orderbook")
    ob = d.get("orderbook_fp") or d.get("orderbook") or {}
    bids = ob.get("yes_dollars") or ob.get("yes") or []
    return sorted(((float(p) if float(p) <= 1 else float(p) / 100, float(q)) for p, q in bids), reverse=True)


def evaluate(city: str, day, now: datetime) -> tuple[list[dict], str]:
    series, station, tz = CITIES[city]
    ev = pm.temp_event(city, day)
    if not ev or not ev.get("markets"):
        return [], f"{city} {day}: no Polymarket event"
    fc = kp.live_forecasts(*wx.station_coords(station), tz, day)
    vals = [v for v in fc.values() if v is not None]
    if len(vals) < 3:
        return [], f"{city} {day}: only {len(vals)} model forecasts"
    try:
        kmap = kalshi_event(series, day)
    except Exception as e:  # Kalshi context is optional; the PM decision must not depend on it
        kmap, kerr = {}, repr(e)[:200]
    else:
        kerr = None
    local = now.astimezone(ZoneInfo(tz))
    rows = []
    for m in ev["markets"]:
        br = pm.parse_bracket(m["slug"])
        if br is None or not kp.rejected(br[0], br[1], br[2], vals):
            continue
        b = pm.book(m["slug"]) or {"bids": [], "offers": [], "state": None, "raw_stats": None}
        pbid = b["bids"][0][0] if b["bids"] else 0.0
        pask = b["offers"][0][0] if b["offers"] else 1.0
        p1, p2 = depth(b["bids"])
        pok = quote_ok(pbid, pask)
        q = maker_q(pbid, pask) if pok else None
        r = {"label": LABEL, "logged_at": now.isoformat(timespec="seconds"), "city": city, "station": station,
             "date": day.isoformat(), "late_min": (local.hour - kp.DECISION_HOUR) * 60 + local.minute,
             "slug": m["slug"], "strike": list(_key(br)), "forecasts": fc, "forecast_source": "live_open_meteo",
             "pm_state": b["state"], "pm_yes_bid": pbid, "pm_yes_ask": pask, "pm_depth_1c": p1, "pm_depth_2c": p2,
             "pm_bids_top5": b["bids"][:5], "pm_stats": b["raw_stats"], "pm_ok": pok,
             "pm_no": round(1 - pbid, 4), "pm_fee": round(pm_fee(1 - pbid), 4) if pok else None,
             "maker_q": q, "maker_L": None if q is None else round(1 - q, 4),
             "maker": None if q is None else {"through": False, "at_or_through": False, "vol_ge_L": 0.0,
                                              "post_high": None, "last_shares": None, "polls": 0, "open": True},
             "pm_result": None, "k_result": None, "k_err": kerr}
        km = kmap.get(_key(br))
        if km:
            kb, ka = float(km.get("yes_bid_dollars") or 0), float(km.get("yes_ask_dollars") or 1)
            kl = kalshi_levels(km["ticker"])
            k1, k2 = depth(kl)
            kok = quote_ok(kb, ka)
            r.update({"k_ticker": km["ticker"], "k_yes_bid": kb, "k_yes_ask": ka, "k_ok": kok, "k_no": round(1 - kb, 4),
                      "k_fee": round(k_fee(1 - kb), 4) if kok else None, "k_depth_1c": k1, "k_depth_2c": k2,
                      "k_bids_top5": kl[:5], "k_maker_q": maker_q(kb, ka) if kok else None})
        else:
            r.update({"k_ticker": None, "k_ok": False})
        rows.append(r)
    return rows, f"{city} {day}: {len(rows)} rejected brackets, PM picks {sum(r['pm_ok'] for r in rows)}, " \
                 f"Kalshi-quotable {sum(r['k_ok'] for r in rows)}"


def poll_maker(r: dict, now: datetime) -> None:
    """One poll of the public stats for a resting paper maker bid (YES offer at L) placed at r['logged_at']."""
    mk = r["maker"]
    T = datetime.fromisoformat(r["logged_at"])
    b = pm.book(r["slug"])
    if not b:
        return
    st, L = b.get("raw_stats") or {}, r["maker_L"]
    mk["polls"] += 1
    mk["last_poll"] = now.isoformat(timespec="seconds")
    hi, hi_t, op_t = _px(st, "highPx"), _ts(st, "highSetTime"), _ts(st, "openSetTime")
    if hi is not None and ((hi_t and hi_t > T) or (op_t and op_t > T)):
        mk["post_high"] = max(mk["post_high"] or 0.0, hi)
    lp, lt = _px(st, "lastTradePx"), _ts(st, "lastTradeSetTime")
    shares = float(st.get("sharesTraded") or 0)
    if lp is not None and lt and lt > T:
        mk["post_last_max"] = max(mk.get("post_last_max") or 0.0, lp)
        prev = mk["last_shares"]
        inc = shares - prev if prev is not None and shares >= prev else (shares if prev is not None else 0.0)
        if lp >= L - 1e-9:
            mk["vol_ge_L"] = round(mk["vol_ge_L"] + inc, 4)
    mk["last_shares"] = shares
    top = max(x for x in (mk.get("post_high"), mk.get("post_last_max"), 0.0) if x is not None)
    if top > L + 1e-9 and not mk["through"]:
        mk["through"], mk["through_seen_at"] = True, now.isoformat(timespec="seconds")
    if top >= L - 1e-9:
        mk["at_or_through"] = True
    if b.get("state") != "MARKET_STATE_OPEN":
        mk["open"] = False


def load() -> list[dict]:
    return [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()] if LOG.exists() else []


def save(rows: list[dict]) -> None:
    tmp = LOG.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(LOG)


def settle_rows(rows: list[dict]) -> int:
    n = 0
    for r in rows:
        if r["pm_result"] is None:
            mk = pm.market(r["slug"])
            r["pm_result"] = pm.winner(mk) if mk else None
            n += r["pm_result"] is not None
        if r.get("k_ticker") and r.get("k_result") is None:
            m = kget(f"/markets/{r['k_ticker']}").get("market", {})
            if m.get("result") in ("yes", "no"):
                r["k_result"] = m["result"]
    return n


def cmd_tick(validate: bool = False) -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w") as lk:
        try:
            fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another tick holds the lock; exiting")
            return 0
        now = datetime.now(timezone.utc)
        rows = load()
        done = set(json.loads(DONE.read_text())) if DONE.exists() else set()
        msgs, new = [], []
        for city, (_, _, tz) in CITIES.items():
            local = now.astimezone(ZoneInfo(tz))
            if not validate and local.hour not in (kp.DECISION_HOUR, kp.DECISION_HOUR + 1):
                continue
            day = (local + timedelta(days=0 if validate else 1)).date()
            cn = f"{city}-{day.isoformat()}"
            if not validate and cn in done:
                continue
            try:
                got, msg = evaluate(city, day, now)
            except Exception as e:
                msgs.append(f"{city} {day}: ERROR {e!r}"[:300])
                continue
            msgs.append(msg)
            if validate:
                for r in got:
                    print(f"VALIDATE {r['slug']} pm {r['pm_yes_bid']}/{r['pm_yes_ask']} ok={r['pm_ok']} q={r['maker_q']} "
                          f"| kalshi {r.get('k_yes_bid')}/{r.get('k_yes_ask')} ok={r['k_ok']} depth pm {r['pm_depth_1c']:.0f} "
                          f"k {r.get('k_depth_1c', 0):.0f} {r.get('k_ticker')} {r.get('k_err') or ''}")
                continue
            if "no Polymarket event" in msg or "model forecasts" in msg:
                continue  # retry on the next tick inside the window
            seen = {x["slug"] for x in rows}
            new += [r for r in got if r["slug"] not in seen]
            done.add(cn)
        for m in msgs:
            print(m)
        if validate:
            print("validation only, nothing logged")
            return 0
        rows += new
        polled = 0
        for r in rows:
            if r.get("maker") and r["maker"]["open"] and r["pm_result"] is None and r["logged_at"] != now.isoformat(timespec="seconds"):
                try:
                    poll_maker(r, now)
                    polled += 1
                except Exception as e:
                    print(f"poll {r['slug']}: {e!r}"[:200])
        settled = 0
        if now.minute < 5 or new:  # settle hourly (cheap) -- results post ~08-11 ET next day
            try:
                settled = settle_rows(rows)
            except Exception as e:
                print(f"settle: {e!r}"[:200])
        save(rows)
        DONE.write_text(json.dumps(sorted(done)))
        STATE.write_text(json.dumps({"label": LABEL, "last_tick": now.isoformat(timespec="seconds"),
                                     "new_rows": len(new), "rows": len(rows), "maker_polled": polled,
                                     "newly_settled": settled, "city_nights_done": len(done), "msgs": msgs}, indent=1))
        if new or msgs:
            print(f"{len(new)} new rows ({sum(r['pm_ok'] for r in new)} PM picks); polled {polled}; settled {settled}")
    return 0


def _boot_p(units_by_cluster: dict, n: int = 4000) -> tuple[float, float, float]:
    v = np.array(list(units_by_cluster.values()))
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(0)
    b = np.array([rng.choice(v, len(v)).sum() for _ in range(n)])
    return float(np.quantile(b, .025)), float(np.quantile(b, .975)), float(np.mean(b <= 0))


def cmd_report() -> int:
    rows = load()
    picks = [r for r in rows if r["pm_ok"]]
    print(f"{LABEL}: {len(rows)} forecast-rejected brackets on {len({(r['city'], r['date']) for r in rows})} city-nights; "
          f"PM picks {len(picks)}; Kalshi-quotable {sum(r['k_ok'] for r in rows)}; both {sum(r['pm_ok'] and r['k_ok'] for r in rows)}")
    st = [r for r in picks if r["pm_result"]]
    tu = {}
    for r in st:
        c = r["pm_no"] + r["pm_fee"]
        tu.setdefault(r["city"] + r["date"], 0.0)
        tu[r["city"] + r["date"]] += (1 / c - 1) if r["pm_result"] == "no" else -1.0
    lo, hi, p = _boot_p(tu)
    T = sum(tu.values())
    print(f"  T taker  : {len(st)} settled, {sum(r['pm_result'] == 'no' for r in st)} wins, {T:+.3f} units, "
          f"ROI {T / max(len(st), 1):+.2%}, CI [{lo:+.2f}, {hi:+.2f}] p {p:.3f}; avg NO {np.mean([r['pm_no'] for r in picks]) if picks else float('nan'):.3f}")
    mk = [r for r in picks if r.get("maker")]
    fills = [r for r in mk if r["maker"]["through"]]
    sfill = [r for r in fills if r["pm_result"]]
    M = sum((1 / r["maker_q"] - 1) if r["pm_result"] == "no" else -1.0 for r in sfill)
    Tm = sum(((1 / (r["pm_no"] + r["pm_fee"]) - 1) if r["pm_result"] == "no" else -1.0) for r in picks if r["pm_result"]
             and r.get("maker"))
    print(f"  M maker  : {len(mk)} posted (of {len(picks)} picks), through-fills {len(fills)} "
          f"({len(fills) / max(len(mk), 1):.0%}), at-or-through {sum(r['maker']['at_or_through'] for r in mk)}, "
          f"size>=10 proxy {sum(r['maker']['vol_ge_L'] >= 10 for r in mk)}; settled fills {len(sfill)}, "
          f"{sum(r['pm_result'] == 'yes' for r in sfill)} losses, {M:+.3f} units; M - T (posted picks) {M - Tm:+.3f}")
    both = [r for r in rows if r["pm_ok"] and r["k_ok"]]
    if both:
        d = np.array([(r["k_no"] + r["k_fee"]) - (r["pm_no"] + r["pm_fee"]) for r in both]) * 100
        rng = np.random.default_rng(1)
        bs = [rng.choice(d, len(d)).mean() for _ in range(4000)]
        print(f"  PAIRED   : {len(both)} picks quoted on both; Kalshi minus PM NO taker cost {d.mean():+.2f}c "
              f"[{np.quantile(bs, .025):+.2f}, {np.quantile(bs, .975):+.2f}]; PM cheaper {int((d > 0.005).sum())}, "
              f"equal {int((abs(d) <= 0.005).sum())}, Kalshi cheaper {int((d < -0.005).sum())}")
    only_pm = sum(r["pm_ok"] and not r["k_ok"] for r in rows)
    only_k = sum(r["k_ok"] and not r["pm_ok"] for r in rows)
    print(f"  coverage : PM-only quotable {only_pm}, Kalshi-only {only_k}")
    if picks:
        pd1 = [r["pm_depth_1c"] for r in picks]
        kd1 = [r.get("k_depth_1c", 0.0) for r in picks]
        print(f"  depth 1c : median PM {np.median(pd1):.0f}, Kalshi {np.median(kd1):.0f}, combined "
              f"{np.median(np.add(pd1, kd1)):.0f} contracts per PM pick")
    ks = [r for r in rows if r["k_ok"] and r.get("k_result")]
    KU = sum((1 / (r["k_no"] + r["k_fee"]) - 1) if r["k_result"] == "no" else -1.0 for r in ks)
    mism = sum(1 for r in rows if r["pm_result"] and r.get("k_result") and r["pm_result"] != r["k_result"])
    print(f"  Kalshi K : {len(ks)} settled same-moment Kalshi picks, {KU:+.3f} units; settlement mismatches PM vs Kalshi {mism}")
    if STATE.exists():
        s = json.loads(STATE.read_text())
        print(f"  last tick {s['last_tick']} (rows {s['rows']}, city-nights {s['city_nights_done']})")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "settle":
        rs = load()
        settle_rows(rs)
        save(rs)
        cmd = "report"
    raise SystemExit({"tick": cmd_tick, "validate": lambda: cmd_tick(validate=True), "report": cmd_report}[cmd]())
