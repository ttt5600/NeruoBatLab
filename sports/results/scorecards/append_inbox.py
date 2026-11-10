import json
from datetime import datetime, timezone
from pathlib import Path

P = Path(__file__).resolve().parents[2] / "notes" / "findings_inbox.jsonl"
ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
rows = [
    {"ts": ts, "agent": "scorecards", "kind": "result", "edge": "L34", "verdict": "dead",
     "headline": "L34 NHL vs opener KILL RULE FIRED: 110 settled, mean CLV -0.0005 prob [-0.0044,+0.0033] <= 0 after 100 "
                 "(open_lines_logger.py:799); ml 57 ROI -26.2%, totals 53 ROI -1.5% (post-hoc split)",
     "numbers": {"n": 110, "clv": -0.0005, "clv_ci": [-0.0044, 0.0033], "roi_novig": -0.143},
     "files": ["sports/notes/SCORECARDS.md", "sports/notes/scorecards_agent.md"], "lesson": "", "memory": ""},
    {"ts": ts, "agent": "scorecards", "kind": "result", "edge": "NFLF", "verdict": "n/a",
     "headline": "NFLF runaway exits ALSO on 11-08 (every exit attempt filled; journal under-records). Exchange truth: "
                 "clean record 4-7 -$0.23 on $15.58 (journal 2-7 -$2.46); incident +$63.47 incl. unsettled BUFMIN marked "
                 "at last (MIN +54.35, PHI +35.56, BUF -22.95) -- luck, uncapped. All-fills P&L matches Kalshi "
                 "/portfolio/settlements on all 7 settled tickers",
     "numbers": {"clean_pnl": -0.23, "clean_wl": "4-7", "incident_pnl": 63.47, "journal_pnl": -2.46},
     "files": ["sports/notes/SCORECARDS.md", "sports/notes/scorecards_agent.md",
               "sports/results/scorecards/kalshi_cache.json", "sports/results/scorecards/check_settlements.py"],
     "lesson": "A journal of intended trades is not the ledger: reconcile against exchange fills (GET /portfolio/fills) nightly.",
     "memory": "NFLF exit-retry bug filled every exit attempt from day one (11-08), not just 11-09 BUFMIN; STATE's "
               "'NFLF 1-7 -$2.80' is journal-only."},
    {"ts": ts, "agent": "scorecards", "kind": "result", "edge": "n/a", "verdict": "n/a",
     "headline": "scripts/scorecards.py -> notes/SCORECARDS.md: 13 cards with verbatim kill rules + file:line. Fired: L34. "
                 "Near-kill: B3 F1 (paper CLV vs ask -0.0068 [-0.0120,-0.0016] n 29), B6 NFLF, Q004 NHL/NFL/NBA. Alive: "
                 "B1 L39 paper 22-0, L32b, L33. Too-early: B2, B4, B5 (live 2-3 -$2.04). Lock-in 99c queue at the horn "
                 "median 201,371 = 40x the frozen N=5,000",
     "numbers": {"lock_horn_queue_median": 201371, "lock_over_queue_median": 24866},
     "files": ["sports/scripts/scorecards.py", "sports/notes/SCORECARDS.md", "sports/notes/scorecards_agent.md"],
     "lesson": "notes/<topic>.md collides with notes/<TOPIC>.md on this case-insensitive Mac.",
     "memory": "Live scorecards vs pre-set kill rules: cd sports && ../.venv_quant/bin/python scripts/scorecards.py "
               "[--kalshi] -> notes/SCORECARDS.md"},
]
with P.open("a") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print("appended", len(rows))
