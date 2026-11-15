"""One-off patch of sports/scripts/lead_exec.py (ops_e4_prep): LEAD_TSA_SIDES. Each anchor must match exactly once."""
from pathlib import Path

p = Path(__file__).resolve().parents[2] / "scripts" / "lead_exec.py"
s = p.read_text()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:80], s.count(a))
    s = s.replace(a, b)


rep('''            contracts = max(1, floor($1 / (q + fee))), <= $8 per KXTSAW event (= week). Kill: 2 losses in first 30 fills.
''', '''            contracts = max(1, floor($1 / (q + fee))), <= $8 per KXTSAW event (= week). Kill: 2 losses in first 30 fills.
            .env LEAD_TSA_SIDES=yes (default yes,no; 2026-11-15 notes/tsa_fill_review.md, ops_e4_prep.md): live copy
            skips paper posts on the other side (status skip_tsa_side_off, $0, so they no longer use the $8 week cap).
            The paper logger (e1x/tsa_paper.jsonl) is not touched.
''')
rep('''def cmd_run(argv: list[str] | None = None, client: kx.Kalshi | None = None, now: datetime | None = None) -> int:
    argv = sys.argv[2:] if argv is None else argv
''', '''def tsa_sides(e: dict) -> set[str]:
    """Sides of TSA paper posts the live copy mirrors (.env LEAD_TSA_SIDES, default both)."""
    out = {x.strip().lower() for x in e.get("LEAD_TSA_SIDES", "yes,no").split(",") if x.strip()}
    if not out <= {"yes", "no"}:
        raise ValueError(f"LEAD_TSA_SIDES must be a subset of yes,no; got {e.get('LEAD_TSA_SIDES')!r}")
    return out


def cmd_run(argv: list[str] | None = None, client: kx.Kalshi | None = None, now: datetime | None = None) -> int:
    argv = sys.argv[2:] if argv is None else argv
''')
rep('''    for item in todo + games:
        if "strategy" in item:
            rec = decide_maker(item, client, mode, now, state, cash)
''', '''    sides = tsa_sides(e)
    for item in todo + games:
        if item.get("strategy") == "TSA" and item.get("side") not in sides:
            rec = {**_base(item, mode, now), "status": "skip_tsa_side_off"}
        elif "strategy" in item:
            rec = decide_maker(item, client, mode, now, state, cash)
''')
p.write_text(s)
print("patched", p)
