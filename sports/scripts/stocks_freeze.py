"""Freeze daily OHLC for a fixed stock/ETF universe so every stock backtest reads the same bytes.

    python scripts/stocks_freeze.py            # writes data/stocks/frozen_<date>.parquet + .sha256

Universe is fixed here BEFORE any strategy is scored: broad index ETFs, the 9 original
SPDR sectors (all trade since 1998; XLRE/XLC are too young for the dev window), and
mega-cap tech (the "hub" nodes for the tech -> other-industries graph idea).
Prices are split/dividend adjusted (auto_adjust=True), so open/high/low/close are comparable.
"""
from __future__ import annotations

import hashlib
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "stocks"
UNIVERSE = ["SPY", "QQQ", "IWM",
            "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB",
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META"]
START, END = "2005-01-01", "2026-01-01"


def main():
    import yfinance as yf
    yf.set_tz_cache_location(os.environ.get("YF_CACHE", str(OUT / ".yfcache")))
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for t in UNIVERSE:  # one at a time: yfinance's sqlite cache locks under its own threads
        d = yf.download(t, start=START, end=END, auto_adjust=True, progress=False, threads=False)
        if d.empty:
            sys.exit(f"download failed for {t}; not freezing a partial universe")
        d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
        frames.append(d[["Open", "High", "Low", "Close", "Volume"]].assign(ticker=t).reset_index())
    df = pd.concat(frames, ignore_index=True).rename(columns=str.lower)
    path = OUT / f"frozen_{date.today().isoformat()}.parquet"
    df.to_parquet(path, index=False)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".sha256").write_text(sha + "\n")
    print(f"{len(df)} rows, {df.ticker.nunique()} tickers, {df.date.min().date()}..{df.date.max().date()} -> {path}  sha256 {sha[:12]}")


if __name__ == "__main__":
    main()
