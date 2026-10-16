"""Live signal engine: turn validated-ish rules into paper bets on Kalshi. NO ORDERS ARE PLACED.

    python scripts/live_signals.py tick      # run every ~15 min (launchd): evaluate due games, settle old picks
    python scripts/live_signals.py report    # paper P&L per signal
    python scripts/live_signals.py dry       # print what tick would do now, write nothing

Each signal is a function (now) -> list of candidate bets; the engine prices
each candidate on Kalshi (the only venue we may automate), logs a paper pick
with the price and the size resting at it, and settles it from Kalshi's result.
Every evaluated game is logged too (data/live_signals_eval.jsonl), bet or not,
so the live signal rate can be compared with the backtest.

Signals:
  nfl_pre_rain  -- outdoor NFL game with >= 1 mm rain at the stadium over the
                   last 3 completed hours, 10-50 min before kickoff -> buy NO on
                   the Kalshi "over X.5" contract nearest 50c (= the under).
                   Dev 2006-19 +12.4%/217, sealed 2020-25 +13.3%/101 p=.074 FAIL:
                   effect size replicated, sample too small -> forward paper test.
                   Live weather is Open-Meteo's forecast-model past hours, not
                   ERA5 as in the backtest: a known mismatch.
"""
from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd
import requests

from sports import nfl
from nfl_game_weather import STADIUMS

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
FORECAST = "https://api.open-meteo.com/v1/forecast"
PICKS = ROOT / "data" / "live_signals.jsonl"
EVALS = ROOT / "data" / "live_signals_eval.jsonl"
SCHED = ROOT / "data" / "nfl_schedule.csv"
FEE_RATE = 0.07
STADIUMS = {**STADIUMS, "RIO00": (-22.912, -43.230)}
H = {"User-Agent": "Mozilla/5.0"}
KALSHI_TEAM = {"LA": "LAR", "JAX": "JAC"}  # nflverse code -> Kalshi ticker code (seen 2026-10)


def _get(url, params=None):
    r = requests.get(url, params=params, headers=H, timeout=30)
    r.raise_for_status()
    return r.json()


def _read(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def _append(p: Path, rec: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def fee(p: float) -> float:
    return math.ceil(FEE_RATE * p * (1 - p) * 100) / 100  # per contract, rounded up to the cent


# ---------------------------------------------------------------- signals
def _nfl_schedule() -> pd.DataFrame:
    # refresh nflverse at most every 6 h: the tick runs every 15 min and kickoff times rarely move
    if not SCHED.exists() or time.time() - SCHED.stat().st_mtime > 6 * 3600:
        SCHED.parent.mkdir(parents=True, exist_ok=True)
        nfl.raw_games(cache=False).to_csv(SCHED, index=False)
    raw = pd.read_csv(SCHED)
    # roof is blank for many future games: use the stadium's usual roof from past seasons
    usual = raw.dropna(subset=["roof"]).groupby("stadium_id")["roof"].agg(lambda s: s.mode().iloc[0])
    raw["roof_eff"] = raw["roof"].fillna(raw["stadium_id"].map(usual))
    extra = raw.assign(book_total=raw["total_line"], away_code=raw["away_team"], home_code=raw["home_team"],
                       unplayed=raw["home_score"].isna())[["game_id", "stadium_id", "roof_eff", "book_total",
                                                            "away_code", "home_code", "unplayed"]]
    g = nfl.from_raw(raw).merge(extra, on="game_id", validate="1:1")  # by key: from_raw re-sorts rows
    return g[g.unplayed]


def _rain_last3h(lat, lon, now) -> tuple[float, float, list]:
    d = _get(FORECAST, {"latitude": lat, "longitude": lon, "hourly": "precipitation,wind_speed_10m",
                        "past_hours": 6, "forecast_hours": 1, "timezone": "UTC", "wind_speed_unit": "mph"})
    h = pd.DataFrame(d["hourly"])
    h["time"] = pd.to_datetime(h["time"], utc=True)
    done = h[h.time <= now.floor("h")].tail(3)  # label = accumulation over the hour ENDING at it
    return float(done.precipitation.sum()), float(done.wind_speed_10m.mean()), done.time.astype(str).tolist()


def nfl_pre_rain(now: pd.Timestamp) -> list[dict]:
    g = _nfl_schedule()
    due = g[(g.start_utc > now + pd.Timedelta(minutes=10)) & (g.start_utc <= now + pd.Timedelta(minutes=50))
            & g.roof_eff.isin(["outdoors", "open"])]
    out = []
    for _, r in due.iterrows():
        if r.stadium_id not in STADIUMS:
            out.append({"game": r.game_id, "skip": f"no coords for {r.stadium_id}"})
            continue
        precip, wind, hours = _rain_last3h(*STADIUMS[r.stadium_id], now)
        out.append({"game": r.game_id, "kickoff": r.start_utc, "stadium": r.stadium_id, "precip_3h_mm": precip,
                    "wind_3h_mph": wind, "hours": hours, "book_total": r.book_total, "fire": precip >= 1.0,
                    "kalshi_event": f"KXNFLTOTAL-{r.start_utc.tz_convert('America/New_York'):%y%b%d}".upper()
                                    + KALSHI_TEAM.get(r.away_code, r.away_code) + KALSHI_TEAM.get(r.home_code, r.home_code),
                    "side": "no_over"})
    return out


SIGNALS = {"nfl_pre_rain": nfl_pre_rain}


# ---------------------------------------------------------------- pricing / settlement
def price_under(event: str) -> dict | None:
    """The 'over X.5' contract whose YES mid is nearest 50c; we would buy NO at its no_ask."""
    ms = _get(f"{KALSHI}/markets", {"event_ticker": event, "limit": 200}).get("markets", [])
    best = None
    for m in ms:
        f = lambda k: float(m[k]) if m.get(k) not in (None, "") else math.nan
        yb, ya, na = f("yes_bid_dollars"), f("yes_ask_dollars"), f("no_ask_dollars")
        if math.isnan(yb) or math.isnan(ya) or math.isnan(na) or ya - yb > 0.05:
            continue
        mid = (yb + ya) / 2
        if best is None or abs(mid - 0.5) < abs(best["yes_mid"] - 0.5):
            best = {"ticker": m["ticker"], "line": m.get("floor_strike"), "yes_mid": mid, "no_ask": na,
                    "no_ask_size": m.get("no_ask_size_fp") or m.get("yes_bid_size_fp")}
    return best


def settle():
    picks, settled = _read(PICKS), {r["ticker"] for r in _read(PICKS) if r.get("type") == "settle"}
    for p in picks:
        if p.get("type") != "pick" or p["ticker"] in settled:
            continue
        m = _get(f"{KALSHI}/markets/{p['ticker']}").get("market", {})
        res = m.get("result")
        if res not in ("yes", "no"):
            continue
        cost = p["no_ask"] + p["fee"]
        pnl = (1 - cost) if res == "no" else -cost
        _append(PICKS, {"type": "settle", "ticker": p["ticker"], "signal": p["signal"], "result": res,
                        "pnl_per_contract": round(pnl, 4), "roi": round(pnl / cost, 4),
                        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        print(f"settled {p['ticker']}: {res} pnl {pnl:+.3f}/contract")


def tick(write: bool = True):
    now = pd.Timestamp.now(tz="UTC")
    evaluated = {(e["signal"], e["game"]) for e in _read(EVALS)}
    for name, fn in SIGNALS.items():
        for c in fn(now):
            if (name, c["game"]) in evaluated:
                continue
            rec = {"type": "eval", "signal": name, "ts": now.isoformat(), **c}
            if c.get("fire"):
                q = price_under(c["kalshi_event"])
                if q is None:
                    rec["skip"] = f"no two-sided Kalshi total for {c['kalshi_event']}"
                else:
                    pick = {"type": "pick", "signal": name, "game": c["game"], "ts": now.isoformat(), **q,
                            "fee": fee(q["no_ask"]), "precip_3h_mm": c["precip_3h_mm"], "book_total": c["book_total"]}
                    print(f"PAPER PICK {name} {c['game']}: buy NO {q['ticker']} @ {q['no_ask']:.2f} "
                          f"(+fee {pick['fee']:.2f}), rain {c['precip_3h_mm']:.1f} mm")
                    if write:
                        _append(PICKS, pick)
            print(f"eval {name} {c['game']}: " + ", ".join(f"{k}={rec[k]}" for k in
                  ("precip_3h_mm", "wind_3h_mph", "fire", "skip") if k in rec))
            if write:
                _append(EVALS, rec)
    if write:
        settle()


def report():
    picks = [p for p in _read(PICKS) if p.get("type") == "pick"]
    sets = {s["ticker"]: s for s in _read(PICKS) if s.get("type") == "settle"}
    evals = _read(EVALS)
    print(f"games evaluated: {len(evals)}; signals fired: {sum(bool(e.get('fire')) for e in evals)}; picks: {len(picks)}")
    for sig in sorted({p["signal"] for p in picks}):
        done = [sets[p["ticker"]] for p in picks if p["signal"] == sig and p["ticker"] in sets]
        open_ = sum(1 for p in picks if p["signal"] == sig and p["ticker"] not in sets)
        tot = sum(s["pnl_per_contract"] for s in done)
        print(f"  {sig}: settled {len(done)} ({sum(s['result'] == 'no' for s in done)} won), open {open_}, "
              f"P&L {tot:+.2f} per 1-contract stake each")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "dry"
    if cmd == "tick":
        tick()
    elif cmd == "dry":
        tick(write=False)
    elif cmd == "report":
        report()
    else:
        raise SystemExit(__doc__)
