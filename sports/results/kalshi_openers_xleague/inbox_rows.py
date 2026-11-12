"""Append this agent's result rows to notes/findings_inbox.jsonl (append-only)."""
import json
from datetime import datetime, timezone
from pathlib import Path

INBOX = Path(__file__).resolve().parents[2] / "notes" / "findings_inbox.jsonl"
TS = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
R = "sports/results/kalshi_openers_xleague/"
rows = [
    dict(kind="result", edge="K-xleague timing", verdict="confirmed",
         headline="Kalshi ladders sit on the opener for NHL/CFB/CBB (first-tight D=|K-open|-|K-close|: NHL tot -0.048 goals "
                  "[-0.069,-0.026] n253; CFB tot -0.67 [-0.90,-0.43] n225, spr -0.47 [-0.70,-0.24]; CBB tot -1.07 [-1.33,-0.81] "
                  "n171, spr -0.42 [-0.65,-0.18]). NFL FAILS vs ESPN DK open (spr +1.42 [+0.85,+1.97], tot +0.38 [-0.17,+0.89]) "
                  "because the DK open is the lookahead line; vs Pinnacle's Sunday-night open Kalshi lists ~2 h earlier",
         numbers={"cfb_ft_h_prekick": 139, "cbb_ft_h_prekick": 19.5, "nhl_ft_h_prekick": 32, "nfl_list_h_prekick": 163,
                  "nhl_kal_minus_pin_h": -42.3, "nfl_kal_minus_pin_h": -2.2},
         files=["sports/notes/kalshi_openers_xleague.md", R + "q1_summary.json", R + "q1b_summary.json", R + "q1c_nhl_summary.json"],
         lesson="Kalshi occurrence_datetime = kickoff + 3 h (NFL); ESPN's DK NFL open is the lookahead line",
         memory="Kalshi college ladders (KXNCAAF*/KXNCAAMB*) list early (CFB ~150 h, CBB ~23 h pre-kick) and price the DK "
                "opener; NFL ladders list with Pinnacle's Sunday re-open"),
    dict(kind="result", edge="K-NFL", verdict="lead",
         headline="L33 picks priced on Kalshi at the logger tick: 14 settled 7-7, ROI@ask+fee -4.7% [-57.7,+46.3], CLV vs "
                  "bought strike +0.79 [+0.18,+1.43] -> GO forward paper (edgelab knfl_l33_kalshi_taker-445b123e)",
         numbers={"n": 14, "roi": -0.0473, "clv": 0.7857, "mean_cost": 0.5438},
         files=[R + "q2_nfl.py", R + "q2_nfl_summary_dev.json", "sports/edgelab/specs/knfl_l33_kalshi_taker.yaml"],
         lesson="", memory=""),
    dict(kind="result", edge="K-CFB", verdict="lead",
         headline="Frozen L36 CFB totals vs the Kalshi first-tight ladder (S = strike nearest DK open): 63 bets 35-28, "
                  "ROI@ask+fee -2.1% [-23.9,+20.3], CLV +1.49 [+0.86,+2.13] -> GO forward paper "
                  "(kcfb_l36_kalshi_taker-613b7d62). Mean cost 0.577: the open-line strike is already priced off 50c",
         numbers={"n": 63, "roi": -0.0207, "clv": 1.4921, "mean_cost": 0.5772, "posthoc_tight_strike_n": 43,
                  "posthoc_tight_strike_roi": 0.1143},
         files=[R + "q2_cfb.py", R + "q2_cfb_summary_dev.json", "sports/edgelab/specs/kcfb_l36_kalshi_taker.yaml"],
         lesson="buying the book-open strike after Kalshi has moved costs ~0.58; next prereg should gap vs Kalshi's 50c rung",
         memory=""),
    dict(kind="progress", edge="K-xleague tonight", verdict="n/a",
         headline="read-only tonight_logger.py running (PID 86585, nohup, exits 07:55Z 11-13): 7 Kalshi ladder series + "
                  "Pinnacle nfl/nhl/cfb/cbb every 10 min -> ticks.jsonl; summarize with q3_tonight.py after 08:00Z",
         numbers={}, files=[R + "tonight_logger.py", R + "q3_tonight.py"], lesson="", memory=""),
]
with open(INBOX, "a") as f:
    for r in rows:
        f.write(json.dumps({"ts": TS, "agent": "kalshi_openers_xleague", **r}) + "\n")
print("appended", len(rows))
