"""The strategy registry: every betting claim the research agents find, graded.

    python lab/registry.py add '<json>'             # research agents write ONLY through this
    python lab/registry.py list [--testable] [--market M] [--sport S]
    python lab/registry.py topics                    # research queue
    python lab/registry.py topic-done ID "one-line summary"

An entry records a CLAIM, not a fact. ``evidence`` grades how much the claim
has been shown rather than asserted, so a forum post and a peer-reviewed paper
never look alike in the queue the backtest agents read from.
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

LAB = Path(__file__).resolve().parent
REG = Path(os.environ["LAB_REGISTRY"]) if os.environ.get("LAB_REGISTRY") else LAB / "registry" / "strategies.jsonl"
TOPICS = LAB / "registry" / "topics.json"

SPORTS = {"nfl", "nba", "mlb", "nhl", "ncaaf", "ncaab", "soccer", "tennis", "multi", "other"}
MARKETS = {"spread", "total", "moneyline", "player_prop", "team_prop", "futures", "live",
           "parlay_sgp", "exchange", "promo", "arbitrage", "middle", "other"}
EVIDENCE = {
    "peer_reviewed": "published, refereed study with data",
    "working_paper": "thesis, preprint or working paper with data",
    "practitioner_data": "a bettor/analyst showing a real record or reproducible backtest",
    "claim": "a stated edge without data (blog, tout, podcast)",
    "anecdote": "forum post or single example",
}
REQUIRED = {"title": str, "sport": str, "market": str, "mechanism": str, "claimed_edge": str,
            "evidence": str, "sources": list, "data_needed": str, "testable_with": list,
            "skeptic_note": str}
DATASETS = {"nfl_spread", "nfl_total", "nfl_moneyline", "mlb_k_props", "mlb_total"}


def _rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def _norm(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower())) - {"the", "a", "of", "in", "on", "and", "to", "for", "bet", "betting"}


def add(raw: str) -> int:
    try:
        e = json.loads(raw)
    except json.JSONDecodeError as x:
        print(f"REJECTED: not valid JSON ({x})")
        return 1
    errs = [f"{k} must be {t.__name__}" for k, t in REQUIRED.items() if not isinstance(e.get(k), t)]
    if e.get("sport") not in SPORTS:
        errs.append(f"sport must be one of {sorted(SPORTS)}")
    if e.get("market") not in MARKETS:
        errs.append(f"market must be one of {sorted(MARKETS)}")
    if e.get("evidence") not in EVIDENCE:
        errs.append(f"evidence must be one of {sorted(EVIDENCE)}")
    if not e.get("sources") or not all(str(u).startswith("http") for u in e.get("sources", [])):
        errs.append("sources must be a non-empty list of URLs you actually opened or saw in results")
    bad = set(e.get("testable_with", [])) - DATASETS
    if bad:
        errs.append(f"testable_with entries must be from {sorted(DATASETS)} (got {sorted(bad)}); [] if none")
    if errs:
        print("REJECTED:\n  " + "\n  ".join(errs))
        return 1
    rows = _rows(REG)
    t = _norm(e["title"] + " " + e["mechanism"])
    for r in rows:
        u = _norm(r["title"] + " " + r["mechanism"])
        if r["market"] == e["market"] and r["sport"] == e["sport"] and len(t & u) / max(1, len(t | u)) > 0.6:
            print(f"DUPLICATE of {r['id']} ({r['title']}); add new sources to your notes instead")
            return 2
    e["id"] = f"S{len(rows) + 1:03d}"
    e["added"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    REG.parent.mkdir(parents=True, exist_ok=True)
    with REG.open("a") as f:
        f.write(json.dumps(e) + "\n")
    print(f"added {e['id']}: {e['title']}")
    return 0


def merge(path: str) -> int:
    """Re-add entries written elsewhere (a cloud agent's file) through the same
    validation and dedupe. Their ids are discarded and reassigned here."""
    counts = {0: 0, 1: 0, 2: 0}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            e = json.loads(line)
            e.pop("id", None), e.pop("added", None)
            counts[add(json.dumps(e))] += 1
    print(f"merged {path}: {counts[0]} added, {counts[2]} duplicates, {counts[1]} rejected")
    return 0


def cmd_list(argv: list[str]) -> int:
    rows = _rows(REG)
    if "--testable" in argv:
        rows = [r for r in rows if r["testable_with"]]
    for flag in ("--market", "--sport"):
        if flag in argv:
            v = argv[argv.index(flag) + 1]
            rows = [r for r in rows if r[flag[2:]] == v]
    for r in rows:
        print(f"{r['id']} [{r['sport']}/{r['market']}/{r['evidence']}] {r['title']}")
        print(f"     mechanism: {r['mechanism'][:220]}")
        print(f"     claimed: {r['claimed_edge'][:160]} | testable: {r['testable_with'] or '-'} | data: {r['data_needed'][:120]}")
    print(f"{len(rows)} entries")
    return 0


def cmd_topics() -> int:
    for t in json.loads(TOPICS.read_text()):
        print(f"{t['id']} [{t['status']}] {t['topic']}" + (f"  -> {t.get('summary', '')}" if t.get("summary") else ""))
    return 0


def next_topic() -> dict | None:
    return next((t for t in json.loads(TOPICS.read_text()) if t["status"] == "pending"), None)


def topic_done(tid: str, summary: str) -> int:
    ts = json.loads(TOPICS.read_text())
    for t in ts:
        if t["id"] == tid:
            t["status"], t["summary"] = "done", summary[:300]
            TOPICS.write_text(json.dumps(ts, indent=1))
            print(f"{tid} marked done")
            return 0
    print(f"no topic {tid}")
    return 1


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    cmd, rest = argv[0], argv[1:]
    if cmd == "add" and rest:
        return add(rest[0])
    if cmd == "merge" and rest:
        return merge(rest[0])
    if cmd == "list":
        return cmd_list(rest)
    if cmd == "topics":
        return cmd_topics()
    if cmd == "topic-done" and len(rest) >= 2:
        return topic_done(rest[0], rest[1])
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
