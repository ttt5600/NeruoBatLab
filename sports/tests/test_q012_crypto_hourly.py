"""Lookahead and pricing guards for scripts/q012_crypto_hourly.py (synthetic spot, no network)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("q012", ROOT / "scripts" / "q012_crypto_hourly.py")
q = importlib.util.module_from_spec(spec)
sys.modules["q012"] = q
spec.loader.exec_module(q)


def _spot(n=3000, t0=1_700_000_000 - 1_700_000_000 % 60):
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
    c = t0 + 60 * np.arange(1, n + 1)
    s = pd.Series(close, index=c)
    r2 = np.log(s).diff() ** 2
    sig2 = 0.5 * r2.rolling(60, min_periods=50).mean() + 0.5 * r2.rolling(1440, min_periods=1200).mean()
    return pd.DataFrame({"S": close, "sig2": sig2.to_numpy(), "last_real": c.astype(float)}, index=c)


def test_spot_uses_only_closed_candles():
    sp = _spot()
    for t in sp.index[1500:1510]:
        for off in (0, 1, 30, 59):
            S, _, c = q.spot_at(sp, int(t + off))
            assert c <= t + off  # the candle closed at or before the decision time
            assert S == sp.loc[c, "S"]
            assert c == t  # never the candle that closes after t


def test_stale_spot_rejected():
    sp = _spot()
    t = int(sp.index[2000])
    sp.loc[t - 600:t, "last_real"] = float(t - 600)
    assert q.spot_at(sp, t) is None


def test_fade_trial_prices_no_at_one_minus_bid():
    dd = pd.DataFrame({"m": [15], "yes_bid": [0.08], "yes_ask": [0.09], "p_emp": [0.01], "event_ticker": ["E"],
                       "asset": ["BTC"], "close_ts": [0]})
    b = q.bets(dd, "T1_FADE_T15")
    assert len(b) == 1 and b.side.iloc[0] == "no"
    assert abs(b.ask.iloc[0] - 0.92) < 1e-12
    assert abs(b.edge.iloc[0] - (0.99 - 0.92 - 0.07 * 0.92 * 0.08)) < 1e-12


def test_no_bet_when_side_unbuyable():
    dd = pd.DataFrame({"m": [15], "yes_bid": [0.0], "yes_ask": [0.01], "p_emp": [0.0001], "event_ticker": ["E"],
                       "asset": ["BTC"], "close_ts": [0]})
    assert len(q.bets(dd, "T1_FADE_T15")) == 0  # NO ask would be 1.00
