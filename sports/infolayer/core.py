"""infolayer core: fetch helpers, data types, gap check, paper ledger. NO ORDER CODE (see notes/infolayer.md)."""
from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
LEDGER_DIR = Path.home() / ".claude/quant-kalshi/data/infolayer"  # outside ~/Desktop (TCC)
_last_call = [0.0]


def now() -> datetime:
    return datetime.now(timezone.utc)


def get(url: str, tries: int = 5, min_gap: float = 0.25):
    """GET JSON with a global throttle and 429 backoff."""
    for i in range(tries):
        wait = min_gap - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "infolayer/0.1"})
            return json.load(urllib.request.urlopen(req, timeout=30))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(2.0 * (i + 1))
                continue
            raise
    raise RuntimeError(f"429 x{tries}: {url}")


def kalshi_markets(series: str) -> list[dict]:
    out, cursor = [], None
    while True:
        u = f"{KALSHI}/markets?series_ticker={series}&status=open&limit=1000" + (f"&cursor={cursor}" if cursor else "")
        d = get(u)
        out += d.get("markets", [])
        cursor = d.get("cursor")
        if not cursor or not d.get("markets"):
            return out


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def phi(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def taker_fee(p: float) -> float:
    """Kalshi taker fee per contract (dollars) at price p."""
    return 0.07 * p * (1 - p)


@dataclass
class Observation:
    key: str                      # family-specific id, e.g. "ME-SEN"
    value: dict                   # the measured quantities
    source: str
    observed_at: str
    source_updated_at: str | None
    fresh: bool
    note: str = ""


@dataclass
class Flag:
    family: str
    ticker: str
    side: str                     # "yes" | "no"
    fair: float                   # fair prob of the side
    price: float                  # executable ask for the side
    ev: float                     # fair - price - fee, per $1 contract
    obs_key: str
    fresh: bool
    ts: str = field(default_factory=lambda: now().isoformat(timespec="seconds"))
    extra: dict = field(default_factory=dict)


def gap(family: str, m: dict, fair_yes: float, obs: Observation, min_ev: float) -> list[Flag]:
    """Compare a fair YES probability with the executable book on both sides."""
    yb, ya = f(m.get("yes_bid_dollars")), f(m.get("yes_ask_dollars"))
    flags = []
    for side, fair, price in (("yes", fair_yes, ya), ("no", 1 - fair_yes, None if yb is None else 1 - yb)):
        if price is None or not 0.01 <= price <= 0.99:
            continue
        ev = fair - price - taker_fee(price)
        if ev >= min_ev:
            flags.append(Flag(family, m["ticker"], side, round(fair, 4), round(price, 4), round(ev, 4), obs.key,
                              obs.fresh, extra={"yes_bid": yb, "yes_ask": ya}))
    return flags


def log(family: str, kind: str, payload: dict) -> None:
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    with (LEDGER_DIR / f"{family}.jsonl").open("a") as fh:
        fh.write(json.dumps({"kind": kind, "ts": now().isoformat(timespec="seconds"), **payload}) + "\n")


def record(family: str, obs: list[Observation], flags: list[Flag], markets_seen: int) -> None:
    for o in obs:
        log(family, "obs", asdict(o))
    for fl in flags:
        log(family, "flag", asdict(fl))
    log(family, "pass", {"n_obs": len(obs), "n_markets": markets_seen, "n_flags": len(flags),
                         "n_stale": sum(not o.fresh for o in obs)})
