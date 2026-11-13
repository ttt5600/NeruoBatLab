"""Shared helpers: Kalshi ladder -> implied line, anchor strike. Read-only public GETs."""
import json
import time
import urllib.parse
import urllib.request

import numpy as np

API = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = {  # (league, market) -> series
    ("nfl", "total"): "KXNFLTOTAL", ("nfl", "spread"): "KXNFLSPREAD",
    ("nhl", "total"): "KXNHLTOTAL",
    ("cfb", "total"): "KXNCAAFTOTAL", ("cfb", "spread"): "KXNCAAFSPREAD",
    ("cbb", "total"): "KXNCAAMBTOTAL", ("cbb", "spread"): "KXNCAAMBSPREAD",
}


try:  # ops_forward_wiring 11-13: shared cross-process Kalshi throttle (<= ~3.3 req/s over ALL paper jobs) when present
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.append(str(_P(__file__).resolve().parents[2] / "scripts"))
    from kx_throttle import slot as _slot
except Exception:  # noqa: BLE001
    _slot = None


def get(path, params=None, tries=8):
    url = f"{API}{path}" + (f"?{urllib.parse.urlencode(params)}" if params else "")
    for i in range(tries):
        try:
            _slot() if _slot else time.sleep(0.25)
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "x"}), timeout=30) as r:
                return json.loads(r.read())
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(3 * (i + 1))


def side_x(event_ticker, ticker, strike, mk):
    """Map a market to x on the 'over' axis: totals x = strike; spreads x on HOME margin
    (Kalshi event suffix = AWAY+HOME letters). Returns (x, sign) with P(axis > x) = mid if sign=+1 else 1-mid."""
    if mk == "total":
        return float(strike), 1
    letters = event_ticker.split("-")[-1][7:]  # drop YYMMMDD
    team = ticker.split("-")[-1].rstrip("0123456789")
    if letters.endswith(team):  # home team "wins by over s" -> P(H-A > s)
        return float(strike), 1
    return -float(strike), -1  # away by over s -> P(H-A < -s) -> P(H-A > -s) = 1 - p


def implied(points):
    """points: list of (x, p_over_x). Average duplicates, isotonic (decreasing) fit, interpolate the p=0.5 crossing."""
    from collections import defaultdict
    g = defaultdict(list)
    for x, p in points:
        if np.isfinite(p):
            g[float(x)].append(p)
    xs = sorted(g)
    if not xs:
        return np.nan
    blocks = [[np.mean(g[x]), 1, [x]] for x in xs]  # PAV for non-increasing
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0]:
            a, b = blocks[i], blocks[i + 1]
            w = a[1] + b[1]
            blocks[i] = [(a[0] * a[1] + b[0] * b[1]) / w, w, a[2] + b[2]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    fx = [(x, bl[0]) for bl in blocks for x in bl[2]]
    if fx[0][1] < 0.5 or fx[-1][1] > 0.5:
        return np.nan  # 0.5 outside the ladder
    for (x0, p0), (x1, p1) in zip(fx, fx[1:]):
        if p0 >= 0.5 >= p1:
            if p0 == p1:
                return (x0 + x1) / 2
            return x0 + (p0 - 0.5) * (x1 - x0) / (p0 - p1)
    return np.nan
