"""Cross-process throttle for PUBLIC Kalshi market-data GETs (read-only; no keys, no orders).

The live bots share this IP with the paper loggers, so every forward logger / scanner (kx_forward.py, lx1_arb_scan.py,
the kalshi_openers_xleague scripts via kx.py) takes its send slot here: an fcntl lock on a shared stamp file holds the
time of the last request from ANY of these processes; the next one waits until last + MIN_GAP + jitter.
MIN_GAP 0.30 s + U(0, 0.10) jitter => at most ~3.3 requests/s in total, below the 4 req/s ceiling.

    from kx_throttle import kget      # kget("/markets", {"event_ticker": ev}) -> dict, or None on a 4xx
"""
from __future__ import annotations

import fcntl
import os
import random
import time
from pathlib import Path

import requests

API = "https://api.elections.kalshi.com/trade-api/v2"
MIN_GAP = 0.30
JITTER = 0.10
STAMP = Path(os.environ.get("KX_THROTTLE_FILE", str(Path.home() / ".claude" / "quant-sports" / "state" / "kalshi_throttle")))
_S = requests.Session()
_S.headers["User-Agent"] = "quant-sports-readonly-paper"


def slot() -> None:
    """Block until this process may send one request (shared across processes)."""
    STAMP.parent.mkdir(parents=True, exist_ok=True)
    with open(STAMP, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            txt = f.read().strip()
            last = float(txt) if txt else 0.0
            wait = last + MIN_GAP + random.uniform(0, JITTER) - time.time()
            if wait > 0:
                time.sleep(wait)
            f.seek(0)
            f.truncate()
            f.write(f"{time.time():.6f}")
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def kget(path: str, params: dict | None = None, tries: int = 6, raise_4xx: bool = False):
    """GET API+path. 429/5xx/network -> back off and retry; other 4xx -> None (or raise). Raises after `tries`."""
    url = path if path.startswith("http") else API + path
    for i in range(tries):
        slot()
        try:
            r = _S.get(url, params=params, timeout=60)
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            if i == tries - 1:
                r.raise_for_status()
            time.sleep(2 * (i + 1))
            continue
        if 400 <= r.status_code < 500:
            if raise_4xx:
                r.raise_for_status()
            return None
        return r.json()
    return None
