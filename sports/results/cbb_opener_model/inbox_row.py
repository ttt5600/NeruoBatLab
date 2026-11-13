import json
from pathlib import Path

row = {"ts": "2026-11-13T09:00:00Z", "agent": "cbb_opener_model", "kind": "result", "edge": "new (CBB totals)",
       "verdict": "weakened",
       "headline": "CBB de-biased totals model leads DK opener: CLV +0.83 [+0.69,+0.97] pts n2213 dev 2023-25, but no-vig ROI "
                   "+3.2% [-0.7,+7.0] p .054 (pre-reg FAIL narrowly), at DK price -1.4%; Kalshi 2026-27 k4 n67 ROI +16.3% "
                   "[-6.0,+37.3] with CLV +0.04 [-0.55,+0.66] (luck-shaped); Kalshi 50c ask+fee 0.541 > -110",
       "numbers": {"dev_n": 2213, "dev_clv": 0.83, "dev_roi_nv": 0.032, "dev_p": 0.054, "dev_roi_px": -0.014,
                   "kalshi_n": 67, "kalshi_roi": 0.163, "kalshi_clv": 0.04, "kalshi_cost": 0.541},
       "files": ["sports/notes/cbb_opener_model.md", "sports/results/cbb_opener_model/dev.json",
                 "sports/results/cbb_opener_model/kalshi.json", "sports/edgelab/specs/cbb_t1_kalshi_taker.yaml"],
       "lesson": "opener CLV again does not survive vig; Kalshi 50c rung ask+fee (0.541) is dearer than -110",
       "memory": "CBB totals model (results/cbb_opener_model) CLV +0.83 vs DK open but dev ROI fails p .054; "
                 "Kalshi taker forward paper cbb_t1_kalshi_taker-4dd8f3e6 unwired"}
with open(Path(__file__).resolve().parents[2] / "notes" / "findings_inbox.jsonl", "a") as f:
    f.write(json.dumps(row) + "\n")
