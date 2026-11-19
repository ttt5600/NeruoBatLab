"""Paper-trade the Kalshi consensus-reject rule on live markets. No orders are placed.

    python scripts/kalshi_paper.py pick      # run ~10pm local: log tomorrow's picks + book depth
    python scripts/kalshi_paper.py settle    # any time later: fill in results from settled markets
    python scripts/kalshi_paper.py report

Rule (lockbox-validated, kalshi_reject_safe_check): for tomorrow's brackets in
each city, take each model's forecast max over hours <= 18:00 local; if ALL
of GFS, ECMWF, ICON, GEM, NBM (>= 3 available) miss the bracket by > 3F,
buy NO at 1 - best YES bid, if the YES ask > 1c and the quote is two-sided
with a spread <= 10c.

The point of paper trading is the one thing the backtest could not see:
how many contracts were actually resting at the price. Each pick records the
depth within 1c and 2c of the best price.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import requests

from sports import wx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import om_fetch  # noqa: E402  same open-meteo endpoint + params; adds TLS-stall timeout + in-window shared cache

LOG = Path(__file__).resolve().parents[1] / "data" / "kalshi_paper.jsonl"
DONE = LOG.with_name("kalshi_paper_done.json")  # city-nights already evaluated
LOST = LOG.with_name("kalshi_paper_lost.jsonl")  # city-nights whose whole decision window passed without a forecast
RETRY_GAP_S, RETRY_BUDGET_S = 90, 20 * 60  # `retry`: re-try failed cities every 90 s for up to 20 min after a tick
FAIL_EV: dict[str, str] = {}  # series -> event of its last FETCH_FAIL in this process (for the LOST alert)
MODELS = ["gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless", "ncep_nbm_conus"]
MARGIN = 3.0
DECISION_HOUR = 22  # city-local, the night before (matches lab/strategies + wx.decision_ts)
H = {"User-Agent": "Mozilla/5.0"}

# X1 (scripts/wx_expand.py) PASSED its sealed test 2026-10-19: same frozen rule on 17 more daily-HIGH series,
# +0.78%/2,464 bets p .0047 (2026-03..10). Paper-traded alongside the original 7 (cohort "x1"). Kept out of
# wx.CITIES on purpose: that dict defines the original kalshi_temp dataset and wx_expand's "already traded" set.
X1_CITIES = {
    "KXHIGHTATL": ("KATL", "America/New_York"), "KXHIGHTBOS": ("KBOS", "America/New_York"),
    "KXHIGHTDAL": ("KDFW", "America/Chicago"), "KXHIGHTDC": ("KDCA", "America/New_York"),
    "KXHIGHTEWR": ("KEWR", "America/New_York"), "KXHIGHTHOU": ("KHOU", "America/Chicago"),
    "KXHIGHTLV": ("KLAS", "America/Los_Angeles"), "KXHIGHTMIN": ("KMSP", "America/Chicago"),
    "KXHIGHTNOLA": ("KMSY", "America/Chicago"), "KXHIGHTOKC": ("KOKC", "America/Chicago"),
    "KXHIGHTPHX": ("KPHX", "America/Phoenix"), "KXHIGHTSAN": ("KSAN", "America/Los_Angeles"),
    "KXHIGHTSATX": ("KSAT", "America/Chicago"), "KXHIGHTSDF": ("KSDF", "America/Kentucky/Louisville"),
    "KXHIGHTSEA": ("KSEA", "America/Los_Angeles"), "KXHIGHTSFO": ("KSFO", "America/Los_Angeles"),
    "KXHIGHTTTN": ("KTTN", "America/New_York"),
}
PAPER_CITIES = {**wx.CITIES, **X1_CITIES}


def _retry(fn, tries: int = 3, wait: float = 5.0, sleep=None):
    """Call fn up to `tries` times on network/parse errors; re-raise the last one. open-meteo timed out on every
    decision-hour tick of the nights logged 11-13 and 11-14 UTC (notes/e4_ops_diag.md)."""
    import time
    sleep = sleep or time.sleep
    for i in range(tries):
        try:
            return fn()
        except (requests.RequestException, ValueError, KeyError):
            if i == tries - 1:
                raise
            sleep(wait * (i + 1))


def live_forecasts(lat, lon, tz, day, since=None, src=None):
    """since: start of this city's decision window. If open-meteo fails, a response for the SAME params fetched at or
    after `since` (by this job or e4p) is used instead (om_fetch). src: list that receives 'live' / 'cache@..'."""
    r, s = om_fetch.fetch_hourly(lat, lon, tz, MODELS, headers=H, since=since)
    if src is not None:
        src.append(s)
    times = [datetime.fromisoformat(t) for t in r["time"]]
    keep = [i for i, t in enumerate(times) if t.date() == day and t.hour <= 18]
    out = {}
    for m in MODELS:
        v = [r.get(f"temperature_2m_{m}", [None] * len(times))[i] for i in keep]
        v = [x for x in v if x is not None]
        out[m] = max(v) if len(v) >= 15 else None
    return out


def rejected(st, floor_s, cap_s, fcs):
    hi, lo = max(fcs), min(fcs)
    if st == "greater":
        return hi + MARGIN < floor_s + 1
    if st == "less":
        return lo - MARGIN > cap_s - 1
    return hi + MARGIN < floor_s or lo - MARGIN > cap_s


def book_depth(ticker):
    """Contracts available to BUY NO = resting YES bids. Depth within 1c/2c of best."""
    d = requests.get(f"{wx.API}/markets/{ticker}/orderbook", headers=H, timeout=30).json()
    ob = d.get("orderbook_fp") or d.get("orderbook") or {}
    bids = ob.get("yes_dollars") or ob.get("yes") or []
    levels = sorted(((float(p) if float(p) <= 1 else float(p) / 100, float(q)) for p, q in bids), reverse=True)
    if not levels:
        return None, 0.0, 0.0
    best = levels[0][0]
    d1 = sum(q for p, q in levels if p >= best - 0.0101)
    d2 = sum(q for p, q in levels if p >= best - 0.0201)
    return best, d1, d2


def cmd_pick(days_ahead: int = 1, log: bool = True, due_only: bool = False, only: set | None = None):
    """due_only: evaluate a city only in its DECISION window -- 22:00-23:59 CITY-local, the night before
    (the backtest decides at 22:00 city-local) -- and only once per city-night (DONE file).
    History: a single 22:03-PT run looked up the day AFTER tomorrow for ET/CT cities (10-15); then an
    exact-22:xx rule missed every city on 10-16 because launchd ran late (19:09, 20:17, ...) after sleep.
    The 23:xx grace hour keeps the same local date, so 'tomorrow' is unchanged; lateness is logged."""
    now = datetime.now(timezone.utc)
    seen = {json.loads(l)["ticker"] for l in LOG.read_text().splitlines()} if LOG.exists() else set()
    done = set(json.loads(DONE.read_text())) if DONE.exists() else set()
    picks, failed = [], []
    for series, (station, tz) in PAPER_CITIES.items():
        if only is not None and series not in only:
            continue
        local = now.astimezone(ZoneInfo(tz))
        if due_only and local.hour not in (DECISION_HOUR, DECISION_HOUR + 1):
            continue
        tomorrow = (local + timedelta(days=days_ahead)).date()
        ev = f"{series}-{tomorrow.strftime('%y%b%d').upper()}"
        if due_only and ev in done:
            continue
        late_min = (local.hour - DECISION_HOUR) * 60 + local.minute
        since = local.replace(hour=DECISION_HOUR, minute=0, second=0, microsecond=0) if due_only else None
        src: list[str] = []
        try:  # one city's network failure must not abort the other cities (11-13/11-14 UTC: whole nights lost)
            ms = _retry(lambda: requests.get(f"{wx.API}/markets", headers=H, timeout=30,
                                             params={"event_ticker": ev, "limit": 100}).json().get("markets", []))
            fc = _retry(lambda: live_forecasts(*wx.station_coords(station), tz, tomorrow, since=since, src=src)) \
                if ms else None
        except Exception as ex:  # noqa: BLE001 -- NOT marked done: the next run in the window retries this city
            failed.append(series)
            FAIL_EV[series] = ev
            print(f"FETCH_FAIL {series} {ev}: {type(ex).__name__}: {str(ex)[:160]}")
            continue
        if not ms:
            print(f"{series}: no open event {ev}")
            continue
        if src and src[-1] != "live":
            print(f"{series}: open-meteo failed; using the same-params response fetched in-window ({src[-1]})")
        vals = [v for v in fc.values() if v is not None]
        if len(vals) < 3:
            print(f"{series}: only {len(vals)} model forecasts; skipping")
            continue
        if due_only and log:
            done.add(ev)  # evaluated this city-night; a later run in the window must not re-pick it
        for m in ms:
            ask = float(m.get("yes_ask_dollars") or 0)
            bid = float(m.get("yes_bid_dollars") or 0)
            fl = float(m["floor_strike"]) if m.get("floor_strike") is not None else np.nan
            ca = float(m["cap_strike"]) if m.get("cap_strike") is not None else np.nan
            if not (bid > 0 and ask < 1 and ask - bid <= 0.10 and ask > 0.01):
                continue
            if not rejected(m.get("strike_type"), fl, ca, vals):
                continue
            if log and m["ticker"] in seen:  # launchd catch-up after sleep must not double-log
                continue
            try:
                best, d1, d2 = _retry(lambda: book_depth(m["ticker"]))
            except Exception as ex:  # noqa: BLE001 -- keep the pick (the executor re-reads the live book anyway)
                print(f"DEPTH_FAIL {m['ticker']}: {type(ex).__name__}")
                best, d1, d2 = None, 0.0, 0.0
            picks.append({"logged_at": now.isoformat(timespec="seconds"), "city": series, "event": ev,
                          "cohort": "x1" if series in X1_CITIES else "orig7",
                          "ticker": m["ticker"], "strike_type": m.get("strike_type"), "floor": fl, "cap": ca,
                          "forecasts": fc, "yes_bid": bid, "yes_ask": ask, "no_price": round(1 - bid, 4),
                          "fee_per_contract": round(wx.FEE_RATE * (1 - bid) * bid, 4),
                          "depth_1c": d1, "depth_2c": d2, "late_min": late_min, "result": None,
                          "fc_source": src[-1] if src else None})
    if log:
        with LOG.open("a") as f:
            for p in picks:
                f.write(json.dumps(p) + "\n")
        if due_only:
            DONE.write_text(json.dumps(sorted(done)))
    for p in picks:
        print(f"PICK {p['ticker']}: buy NO @ {p['no_price']:.2f}  depth {p['depth_1c']:.0f} (1c) / "
              f"{p['depth_2c']:.0f} (2c) contracts  forecasts {p['forecasts']}")
    print(f"{len(picks)} picks " + (f"logged to {LOG}" if log else "(validation only, not logged)"))
    if failed:
        print(f"FETCH_FAIL {len(failed)} cities: {','.join(failed)} (retried next run while in the 22-23h window)")
    return failed


def cmd_settle():
    rows = [json.loads(l) for l in LOG.read_text().splitlines()] if LOG.exists() else []
    for r in rows:
        if r["result"] is None:
            m = requests.get(f"{wx.API}/markets/{r['ticker']}", headers=H, timeout=30).json().get("market", {})
            if m.get("result") in ("yes", "no"):
                r["result"] = m["result"]
    LOG.write_text("".join(json.dumps(r) + "\n" for r in rows))
    cmd_report()


def cmd_report():
    rows = [json.loads(l) for l in LOG.read_text().splitlines()] if LOG.exists() else []
    done = [r for r in rows if r["result"]]
    units = sum((1 - r["no_price"] - r["fee_per_contract"]) / (r["no_price"] + r["fee_per_contract"])
                if r["result"] == "no" else -1.0 for r in done)
    for c in ("orig7", "x1"):
        cd = [r for r in done if r.get("cohort", "orig7") == c]
        cu = sum((1 - r["no_price"] - r["fee_per_contract"]) / (r["no_price"] + r["fee_per_contract"])
                 if r["result"] == "no" else -1.0 for r in cd)
        print(f"  {c}: {sum(r.get('cohort', 'orig7') == c for r in rows)} picks, {len(cd)} settled, {cu:+.2f} units")
    print(f"{len(rows)} paper picks, {len(done)} settled, {sum(r['result'] == 'no' for r in done)} wins, "
          f"{units:+.2f} units; median depth within 1c: "
          f"{np.median([r['depth_1c'] for r in rows]) if rows else 0:.0f} contracts")


def _local(series: str, now: datetime) -> datetime:
    return now.astimezone(ZoneInfo(PAPER_CITIES[series][1]))


def window_open(series: str, now: datetime | None = None) -> bool:
    return _local(series, now or datetime.now(timezone.utc)).hour in (DECISION_HOUR, DECISION_HOUR + 1)


def pending_due(now: datetime | None = None) -> list[str]:
    """Cities in their decision window whose city-night is not in DONE yet."""
    now = now or datetime.now(timezone.utc)
    done = set(json.loads(DONE.read_text())) if DONE.exists() else set()
    out = []
    for s in PAPER_CITIES:
        if window_open(s, now):
            ev = f"{s}-{(_local(s, now) + timedelta(days=1)).date().strftime('%y%b%d').upper()}"
            if ev not in done:
                out.append(s)
    return out


def cmd_retry(cands: list[str], sleep=time.sleep, clock=time.monotonic) -> list[str]:
    """Run by run.sh AFTER kalshi_exec has acted on the cities that worked, when the tick exited 3. Re-tries the
    pending cities every RETRY_GAP_S for up to RETRY_BUDGET_S. 11-17: NY and DEN failed all 3 tries in both window
    ticks while the other cities fetched fine seconds later, so tries spread over minutes beat 3 tries in ~1 min.
    A city still unfetched once its window is in its last hour (23:xx local) or shut is LOST: logged to LOST and a
    macOS alert (once per city-night). Same source and params only -- no fallback forecast."""
    t0 = clock()
    left = list(cands)
    while left and clock() - t0 < RETRY_BUDGET_S:
        sleep(RETRY_GAP_S)
        left = [s for s in left if window_open(s)]
        if not left:
            break
        print(f"RETRY {','.join(left)} at +{clock() - t0:.0f}s")
        cmd_pick(due_only=True, only=set(left))
        left = [s for s in left if s in pending_due()]   # decided (incl. no-pick nights) -> in DONE
        left = [s for s in left if s in FAIL_EV]          # no open event / few models -> not a fetch problem
    now = datetime.now(timezone.utc)
    done = set(json.loads(DONE.read_text())) if DONE.exists() else set()
    old = {json.loads(l)["event"] for l in LOST.read_text().splitlines()} if LOST.exists() else set()
    for s in sorted(FAIL_EV):
        ev = FAIL_EV[s]
        if ev in done or (window_open(s, now) and _local(s, now).hour == DECISION_HOUR):
            continue  # decided, or the 23:xx tick will still try it
        if ev not in old:
            with LOST.open("a") as f:
                f.write(json.dumps({"ts": now.isoformat(timespec="seconds"), "city": s, "event": ev}) + "\n")
            om_fetch.notify("E4 LOST city-night", f"{ev}: open-meteo failed for the whole 22-23h window; no pick")
        print(f"LOST {s} {ev}: open-meteo unavailable for the whole decision window (alerted)")
    return left


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "tick":  # hourly from launchd: pick any city in its decision window, then settle (idempotent)
        failed = cmd_pick(due_only=True)
        try:
            cmd_settle()
        finally:
            if failed:
                sys.exit(3)  # run.log shows "exit 3" while any city-night in its window is still unevaluated
    elif cmd == "retry":  # run.sh: after kalshi_exec, only when tick exited 3
        sys.exit(3 if cmd_retry(pending_due()) else 0)
    elif cmd == "validate":  # today's open markets: exercises the live path, logs nothing
        cmd_pick(days_ahead=0, log=False)
    else:
        {"pick": cmd_pick, "settle": cmd_settle, "report": cmd_report}[cmd]()
