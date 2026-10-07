#!/bin/bash
# After all cities are pulled: add multi-model forecasts, rebuild the Kalshi
# table, then a capped agent loop (2 iterations x 3 strategies) on it.
cd "$(dirname "$0")/.." || exit 1
PY=../.venv_quant/bin/python
while pgrep -f "scripts/pull_wx.py" >/dev/null; do sleep 60; done
$PY scripts/add_forecasts.py
rm -f data/lab/kalshi_temp.parquet
export LAB_FOCUS="dataset kalshi_temp (Kalshi daily-high brackets, 7 cities, real bid/ask at 10pm the day before, fee included in dec_odds). Ledger: a GFS-only error model lost -18% and a 4-model ensemble (sd 2.7F vs GFS 3.5F) lost -16%: the market's forecast is sharper than both, so do NOT just re-fit a global error model. Look for STRUCTURE the crowd may misprice: city-specific forecast biases (e.g. coastal LA/MIA vs continental DEN), season x city sd, disagreement between models (ensemble spread) as an uncertainty signal, brackets far from every model, thin-volume brackets, and market-implied distribution vs ensemble. Fit only on earlier quarters inside fit()."
bash lab/loop.sh strategy 2 3
