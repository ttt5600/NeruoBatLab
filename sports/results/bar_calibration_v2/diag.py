"""Diagnose placebo false-accept inflation: vary night ICC / month v / seasonality, live mix, rho1 .6 beta .2.
Run from sports/: ../.venv_quant/bin/python results/bar_calibration_v2/diag.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim2  # noqa: E402

p, mix = sim2.derive()
for icc, v, flat in [(0, 0, True), (0, 0, False), (0.031, 0, False), (0, 0.118, False), (0.031, 0.118, False)]:
    q = dict(p)
    q['icc_night'] = icc
    q['month_v_base'] = v
    if flat:
        q['season_profile'] = [1.0] * 12
    sim2._init(q, mix)
    out = sim2.nc_task(('live', 1.0, False, False, 3000, 7))[4]
    a = np.array([(r1, b, al, d) for r1, b, al, d, m, n in out])
    sel = (a[:, 0] == .6) & (a[:, 1] == .2)
    print(icc, v, 'flat' if flat else 'seasonal',
          {al: round(float((a[sel & (a[:, 2] == al), 3] == 1).mean()), 4) for al in (.05, .02, .01)}, flush=True)
