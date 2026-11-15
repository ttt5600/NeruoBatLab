"""Audit the OLD gap_logger soccer matcher (_first + substring) for team collisions.

Read-only GETs. For every soccer event in gap_log.jsonl (deployed copy) plus every currently open Kalshi soccer
event, fetch the Kalshi team names (yes_sub_title). Pinnacle names come from current matchups per league (names are
stable per club) -- each Kalshi team is mapped to its Pinnacle name with the NEW strict matcher, then the OLD
side-assignment rule is replayed:   side = "H" if _first(pin_home) in _first(kalshi_team) else "A".
An event is MIS-SIDED when that rule labels a team market with the wrong Pinnacle side.

Also lists every cross-team pair (pin name P, kalshi name K, different clubs) with _first(P) in _first(K) per league.

    python results/ops_gap_logger/audit_matcher.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

import gap_logger as G
from sports.live import KALSHI, PIN, PIN_HEADERS, _get

LOG = Path.home() / ".claude/quant-kalshi/data/gap_log.jsonl"
OUT = Path(__file__).resolve().parent


def old_first(s):  # verbatim copy of the pre-fix gap_logger._first
    import re
    s = re.sub(r"[^a-z ]", "", str(s).lower().replace("&", "and"))
    for w in ("fc", "afc", "cf", "sc", "club", "de", "the", "united", "city"):
        s = re.sub(rf"\b{w}\b", "", s)
    return s.split()[0] if s.split() else s


def main():
    series2lg = {ser: lg for lg, (_, ser) in G.SOCCER.items()}
    g = pd.read_json(LOG, lines=True)
    g = g[g.league.isin(list(G.SOCCER))]
    g["event"] = g.ticker.str.rsplit("-", n=1).str[0]
    events = set(g.event)
    # names.json = every Kalshi soccer market (all statuses, close >= 2025-10-09) per series, paginated, plus the
    # current Pinnacle participants per league -- built 2026-11-15 by the one-off fetch in notes/ops_gap_logger.md
    nj = json.load(open(OUT / "names.json"))
    kal_teams = nj["events"]  # event -> {ticker suffix: yes_sub_title}
    pin_names = {lg: set(v["pin"]) for lg, v in nj["names"].items()}
    print("gap_log soccer events missing from names.json:", sorted(events - set(kal_teams)))
    # cross-team pairs under the old rule
    pairs = []
    for ev, tm in kal_teams.items():
        lg = series2lg.get(ev.split("-")[0])
        if lg is None:
            continue
        for suf, k in tm.items():
            if suf == "TIE":
                continue
            for p in pin_names[lg]:
                same = G.same_team(p, k)
                if old_first(p) in old_first(k) and not same:
                    pairs.append((lg, p, k, old_first(p), old_first(k)))
    pairs = sorted(set(pairs))
    # replay per event
    bad = []
    for ev, tm in kal_teams.items():
        lg = series2lg.get(ev.split("-")[0])
        if lg is None:
            continue
        teams = {s: n for s, n in tm.items() if s != "TIE"}
        if len(teams) != 2:
            continue
        pmap = {}
        for s, n in teams.items():
            hits = [p for p in pin_names[lg] if G.same_team(p, n)]
            pmap[s] = hits[0] if len(hits) == 1 else n  # fall back to the Kalshi name itself
        # Kalshi soccer event tickers list the HOME code first (verified 138/138 on live Pinnacle matches 2026-11-15)
        code = ev.split("-")[1][7:]
        homes = [s for s in teams if code.startswith(s)]
        for home_s in homes[:1]:
            away_s = next(s for s in teams if s != home_s)
            hn, an = old_first(pmap[home_s]), old_first(pmap[away_s])
            found = any(hn in old_first(t) for t in teams.values()) and any(an in old_first(t) for t in teams.values())
            if not found:
                continue
            for s, n in teams.items():
                lab = "H" if hn in old_first(n) else "A"
                truth = "H" if s == home_s else "A"
                if lab != truth:
                    bad.append({"league": lg, "event": ev, "pin_home": pmap[home_s], "pin_away": pmap[away_s],
                                "kalshi_team": n, "labelled": lab, "truth": truth, "in_gap_log": ev in events,
                                "gap_rows": int((g.event == ev).sum())})
    bad = pd.DataFrame(bad)
    print(f"soccer events audited: {len(kal_teams)} (gap_log events {len(events)})")
    print(f"cross-team old-rule pairs: {len(pairs)}")
    for p in pairs:
        print("  ", p)
    print(bad.to_string() if len(bad) else "no mis-sided events")
    bad.to_csv(OUT / "missided_events.csv", index=False)
    json.dump({"pairs": pairs, "n_events": len(kal_teams), "n_gap_events": len(events)}, open(OUT / "audit.json", "w"), indent=1)


if __name__ == "__main__":
    main()
