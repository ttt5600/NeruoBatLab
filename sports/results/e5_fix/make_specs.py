"""Write the 13 pre-registered e5_fix trials as edgelab specs (rule null: the scorer is results/e5_fix/proxy.py)."""
from pathlib import Path

import yaml

D = Path(__file__).parent / "specs"
D.mkdir(exist_ok=True)
base = ("E5 fix (notes/e5_fix.md): FX consensus-reject picks executed as a NO/YES bid at the best bid at 22:00 local D-1; "
        "Kalshi KXHIGH E1+X1 dev picks as timestamped proxy (fill iff >= 10 printed at <= q), FX daily lows as check. ")
trials = [(f"e5fix_f1_cut{h:02d}", f"F1: cancel at {h:02d}:00 station-local on event day D.") for h in (0, 6, 9, 12, 15)]
trials += [(f"e5fix_f2_obs_x{x}", f"F2: cancel when the METAR running max since local midnight D >= L - {x}F "
            "(above-side); below-side cancel 12:00 local unless cleared.") for x in (2, 4, 6)]
trials += [(f"e5fix_f3_bid_m{k}", f"F3: bid q0 - {k}c, rest to close.") for k in (2, 5, 10, 15, 20)]
for tid, h in trials:
    spec = {"id": tid, "family": "kalshi_weather_temp", "dataset": "kalshi_temp_maker+forecastex_paper", "rule": None,
            "params": {}, "hypothesis": base + h, "decision_time": "22:00 station-local on D-1",
            "price_model": "maker_tape", "split": "dev: kalshi E1/X1 dev picks; FX event_date < 2026-11-01",
            "min_n": 100, "kill": {"kind": "roi_ci_below_zero", "after_n": 100},
            "legacy": "new 2026-11-11 e5_fix agent (results/e5_fix/proxy.py, fx_check.py, fx_f2.py)"}
    (D / f"{tid}.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
print(len(trials))
