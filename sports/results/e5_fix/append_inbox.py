"""Append the e5_fix result row to notes/findings_inbox.jsonl (append-only)."""
import json
from pathlib import Path

row = {"ts": "2026-11-12T05:00:00Z", "agent": "e5_fix", "kind": "result", "edge": "E5", "verdict": "dead",
       "headline": "No fix rescues E5 FX maker bids: on IBKR ticks all 6 losers filled 7.5-13h local before any obs "
                   "trigger; F2 X=2 -10.8% (29 fills) vs B0 -8.9%; F3 lower bid -14%->-46% (losers fill at any price); "
                   "only F1 cancel-06:00 is >0 (+26%, 3 fills/6 days, ~$2/day, no CI)",
       "numbers": {"B0_ticks": {"fills": 32, "losses": 6, "roi": -0.089},
                   "F1_06": {"fills": 3, "losses": 0, "roi": 0.265},
                   "F1_09": {"fills": 7, "losses": 1, "roi": 0.015},
                   "F2_X2": {"fills": 29, "losses": 6, "roi": -0.108},
                   "F3_all693_k5": {"fills": 36, "roi": -0.221},
                   "proxy_dev_B0": {"roi": 0.030, "ci": [0.014, 0.046]}, "metar_settle_agree": "693/693"},
       "files": ["sports/notes/e5_fix.md", "sports/results/e5_fix/fx_tick_score.json",
                 "sports/results/e5_fix/fx_check.json", "sports/results/e5_fix/fx_f2.json",
                 "sports/results/e5_fix/proxy_dev.json", "sports/results/e5_fix/proxy_sealed.json"],
       "lesson": "FX weather bids are hit by day-D forecast traders (3-17F from losing), not obs watchers; lowering a "
                 "bid never sheds losers in a binary that goes to 0",
       "memory": "E5 FX fix attempt 11-11: F1/F2/F3 all fail on IBKR ticks; IBKR HMDS serves FORECASTX ticks ~1 wk "
                 "after expiry if the start time is near (12-h windows)"}
f = Path(__file__).resolve().parents[2] / "notes" / "findings_inbox.jsonl"
with open(f, "a") as fh:
    fh.write(json.dumps(row) + "\n")
print("appended")
