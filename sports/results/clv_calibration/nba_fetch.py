"""Read-only: NBA totals (KXNBATOTAL) settled ladders + hourly candles open->start+3h + ESPN book close, reusing
kalshi_openers_xleague/q1_fetch.py with outputs redirected here. Run from sports/ with
KX_LO=2026-10-20 KX_HI=2026-11-14 ../.venv_quant/bin/python results/clv_calibration/nba_fetch.py"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "kalshi_openers_xleague"))
import q1_fetch as q  # noqa: E402

q.OUT = HERE
q.CACHE = HERE / "cache"
q.CACHE.mkdir(exist_ok=True)
q.SERIES[("nba", "total")] = "KXNBATOTAL"
FIX = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS", "PHX": "PHX"}
_g0 = q.espn_games


def eg(lg):
    g = _g0(lg)
    for c in ("away", "home"):
        g[c] = g[c].replace(FIX)
    return g


q.espn_games = eg
_k0 = q.kalshi_settled


def ks(series):  # mixed ISO formats (with/without fractional seconds) break pd.to_datetime -> normalise
    ms = _k0(series)
    for m in ms:
        for f in ("open_time", "occurrence_datetime"):
            if m.get(f):
                m[f] = q.pd.Timestamp(m[f]).strftime("%Y-%m-%dT%H:%M:%SZ")
    return ms


q.kalshi_settled = ks
q.main("nba")
