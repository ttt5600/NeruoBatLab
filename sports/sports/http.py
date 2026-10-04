"""Polite, cached HTTP for public sports endpoints.

None of these APIs need a key, and none of them promise to keep serving us, so
the rules are: cache every response for a finished game forever, never hit a
host faster than a few requests a second, and back off on 429/5xx instead of
hammering. The cache is what makes a 2,000-request NBA pull a one-time cost.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode, urlparse

import requests

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"
HEADERS = {"User-Agent": "Mozilla/5.0 (research; sports backtest)"}
MIN_INTERVAL = 0.25  # seconds between requests to the same host

_last_hit: dict[str, float] = {}
_lock = threading.Lock()
_local = threading.local()


def _session() -> requests.Session:
    # requests.Session is not thread-safe; one per thread.
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


def _key(url: str, params: dict | None) -> str:
    full = url + ("?" + urlencode(sorted((params or {}).items())) if params else "")
    return hashlib.sha256(full.encode()).hexdigest()[:24]


def _throttle(host: str) -> None:
    """Reserve the next send slot for ``host``; at most one request per MIN_INTERVAL.

    Threads overlap network latency, not request rate: slots are handed out
    under a lock, so N threads still never exceed 1/MIN_INTERVAL per host.
    """
    with _lock:
        now = time.monotonic()
        slot = max(now, _last_hit.get(host, 0.0) + MIN_INTERVAL)
        _last_hit[host] = slot
    if slot > now:
        time.sleep(slot - now)


def fetch(url: str, params: dict | None = None, *, cache: bool = True,
          binary: bool = False, retries: int = 4, timeout: float = 60.0):
    """GET ``url``; JSON (default) or raw bytes. Cached on disk when ``cache``.

    Pass ``cache=False`` for anything that can still change -- today's
    scoreboard, an in-progress season -- or the cache will freeze it.
    """
    suffix = ".bin" if binary else ".json"
    path = CACHE_DIR / (_key(url, params) + suffix)
    if cache and path.exists():
        return path.read_bytes() if binary else json.loads(path.read_text())

    host = urlparse(url).netloc
    for attempt in range(retries):
        _throttle(host)
        try:
            r = _session().get(url, params=params, headers=HEADERS, timeout=timeout)
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
            continue
        if r.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
            time.sleep(2 ** (attempt + 1))
            continue
        r.raise_for_status()
        body = r.content if binary else r.json()
        if cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(suffix + ".tmp")
            if binary:
                tmp.write_bytes(body)
            else:
                tmp.write_text(json.dumps(body))
            tmp.replace(path)
        return body
    raise RuntimeError(f"unreachable: {url}")


def fetch_many(urls: list[str], workers: int = 4, **kw) -> list:
    """``fetch`` each URL with a small thread pool, results in input order."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda u: fetch(u, **kw), urls))
