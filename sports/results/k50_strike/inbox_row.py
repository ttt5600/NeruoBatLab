import datetime
import json
from pathlib import Path

r = {"ts": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "agent": "k50_strike",
     "kind": "result", "edge": "K1/K-CFB/K-NFL (K50, LO)", "verdict": "weakened",
     "headline": ("K50 (model vs Kalshi own 50c strike) cuts cost to 0.53 but ROI still ~0: NBA -2.2% [-29,+25] n25 CLV "
                  "+0.88; CFB -7.3% [-30,+28] n71 CLV +0.82; NFL +11.6% n17 (3 dates) CLV +0.24 -> GO paper only. "
                  "Ladder-optimal LO dead (NBA -17%, NFL -36% [-71,-9], CFB -10%)."),
     "numbers": {"nba_k50": {"n": 25, "roi": -0.0224, "ci": [-0.291, 0.2456], "clv": 0.88, "cost": 0.5299},
                 "cfb_k50": {"n": 71, "roi": -0.0732, "ci": [-0.2951, 0.2766], "clv": 0.82, "cost": 0.53},
                 "nfl_k50": {"n": 17, "roi": 0.116, "clv": 0.235, "dates": 3},
                 "nba_lo": {"n": 33, "roi": -0.1725, "clv": -1.73}, "cfb_lo": {"n": 125, "roi": -0.1045, "clv": 0.02},
                 "nfl_lo": {"n": 30, "roi": -0.362, "ci": [-0.71, -0.0905], "clv": -2.68},
                 "breakeven_clv_pts": {"nba": 1.40, "cfb": 1.16}},
     "files": ["sports/notes/k50_strike.md", "sports/results/k50_strike/summary.json",
               "sports/results/k50_strike/clv_value.json", "sports/results/k50_strike/score.py"],
     "lesson": ("Old K1/K-CFB CLV (+1.4-1.5) was partly Kalshi's own move off the opener; vs Kalshi's 50c rung the "
                "models give ~0.85 pts = ~2c, below the ~1.2-1.4 pt taker break-even."),
     "memory": ("Kalshi totals taker needs ~1.2-1.4 pts CLV vs Kalshi's own 50c strike; L32b/L36 deliver ~0.85 -> "
                "ROI ~0 (notes/k50_strike.md). Ladder-optimal with model-as-truth normal loses.")}
open(Path(__file__).resolve().parents[2] / "notes/findings_inbox.jsonl", "a").write(json.dumps(r) + "\n")
print("ok")
