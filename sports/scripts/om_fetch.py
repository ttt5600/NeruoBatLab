"""open-meteo forecast fetch shared by the E4 pick logger (kalshi_paper.py) and e4p_paper.py (ops_robustness_1118).

The frozen E4 rule reads ONE source: api.open-meteo.com/v1/forecast with fixed params. This module does not add or
substitute a source. It adds two things around that one request:

1. a longer connect timeout. The 11-17 failures were "Read timed out. (read timeout=10)" with timeout=(10, 45): in
   urllib3 2.x that message with the CONNECT value comes from `_validate_conn`, i.e. the TLS handshake stalled after
   TCP connected (notes/ops_robustness_1118.md). A stalled handshake now gets 30 s instead of 10/15.
2. a cache of the last successful response per exact param set, shared across processes (E4 and e4p fetch the same
   URL for the same city minutes apart). A caller may fall back to it ONLY when the cached response was fetched
   inside the same decision window (fetched_at >= the caller's `since`, normally 22:00 city-local on D-1). That is a
   response from the same endpoint with the same params, fetched inside the window the rule decides in -- the same
   input the rule would have used had the caller's own request been the one that got through.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import requests

OM = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = (30, 60)  # (connect incl. TLS handshake, read)
CACHE_DIR = Path(os.environ.get("OM_CACHE_DIR", str(Path.home() / ".claude" / "quant-shared" / "om_cache")))


def params(lat, lon, tz, models) -> dict:
    return {"latitude": lat, "longitude": lon, "models": ",".join(models), "hourly": "temperature_2m",
            "temperature_unit": "fahrenheit", "timezone": tz, "forecast_days": 3}


def _path(p: dict) -> Path:
    k = hashlib.sha1(json.dumps(p, sort_keys=True, default=str).encode()).hexdigest()[:20]
    return CACHE_DIR / f"{k}.json"


def _save(p: dict, hourly: dict, now: datetime) -> None:
    try:  # never let a cache problem break a live fetch; tmp + rename so a reader never sees half a file
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=CACHE_DIR, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump({"fetched_at": now.isoformat(timespec="seconds"), "params": p, "hourly": hourly}, f)
        os.replace(tmp, _path(p))
    except OSError:
        pass


def cached(p: dict, since: datetime, now: datetime | None = None) -> tuple[dict, str] | None:
    """(hourly, fetched_at) if a response for exactly these params was fetched in [since, now], else None."""
    now = now or datetime.now(timezone.utc)
    try:
        c = json.loads(_path(p).read_text())
        t = datetime.fromisoformat(c["fetched_at"])
    except (OSError, ValueError, KeyError):
        return None
    if c.get("params") != json.loads(json.dumps(p, default=str)) or not (since <= t <= now):
        return None
    return c["hourly"], c["fetched_at"]


def fetch_hourly(lat, lon, tz, models, headers=None, since: datetime | None = None, get=None) -> tuple[dict, str]:
    """One open-meteo request -> (hourly dict, source). source = 'live' or 'cache@<fetched_at>'.
    On failure, falls back to a cached response fetched at/after `since` (same window); otherwise re-raises."""
    p = params(lat, lon, tz, models)
    get = get or requests.get
    try:
        hourly = get(OM, headers=headers, timeout=TIMEOUT, params=p).json()["hourly"]
    except (requests.RequestException, ValueError, KeyError):
        hit = cached(p, since) if since is not None else None
        if hit is None:
            raise
        return hit[0], f"cache@{hit[1]}"
    _save(p, hourly, datetime.now(timezone.utc))
    return hourly, "live"


def notify(title: str, msg: str) -> None:
    """macOS notification; never raises (same pattern as kalshi_exec.notify)."""
    import subprocess
    safe = lambda s: str(s).replace("\\", "").replace('"', "'")[:200]  # noqa: E731
    try:
        subprocess.run(["osascript", "-e", f'display notification "{safe(msg)}" with title "{safe(title)}"'],
                       timeout=5, capture_output=True)
    except Exception:  # noqa: BLE001
        pass
