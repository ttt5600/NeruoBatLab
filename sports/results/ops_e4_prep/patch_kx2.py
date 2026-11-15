"""Second patch of sports/scripts/kalshi_exec.py (ops_e4_prep): KALSHI_X1_DAY_USD sub-cap so X1 cannot crowd orig7
out of the shared day/open caps. Each anchor must match exactly once."""
from pathlib import Path

p = Path(__file__).resolve().parents[2] / "scripts" / "kalshi_exec.py"
s = p.read_text()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:80], s.count(a))
    s = s.replace(a, b)


rep('''    KALSHI_X1_MAX_CONTRACTS (X1-only contract cap), KALSHI_PRICE_BAND_CAPS''',
    '''    KALSHI_X1_MAX_CONTRACTS (X1-only contract cap), KALSHI_X1_DAY_USD (X1 sub-cap on BOTH today's X1 spend and open
    X1 exposure, inside the shared day/open caps, so X1 cannot use up orig7's room), KALSHI_PRICE_BAND_CAPS''')
rep('''            "x1_kill_since": e.get("KALSHI_X1_KILL_SINCE") or None}
''', '''            "x1_kill_since": e.get("KALSHI_X1_KILL_SINCE") or None,
            "x1_day": float(e["KALSHI_X1_DAY_USD"]) if e.get("KALSHI_X1_DAY_USD") else None}
''')
rep('''def decide(p: dict, client: Kalshi, mode: str, cfg: dict, now: datetime, spent: float, opn: float,
           bal: float | None) -> dict:
''', '''def decide(p: dict, client: Kalshi, mode: str, cfg: dict, now: datetime, spent: float, opn: float,
           bal: float | None, x1_used: tuple[float, float] = (0.0, 0.0)) -> dict:
''')
rep('''    budget = min(cap["order"], cap["day"] - spent, cap["open"] - opn, bal if bal is not None else math.inf)
''', '''    budget = min(cap["order"], cap["day"] - spent, cap["open"] - opn, bal if bal is not None else math.inf)
    x1_room = math.inf
    if rec["cohort"] == "x1" and cfg.get("x1_day") is not None:
        x1_room = cfg["x1_day"] - max(x1_used)
        budget = min(budget, x1_room)
''')
rep('''    if style == "maker":
        return _decide_maker(rec, client, mode, cfg, ask, no_bid, budget, bal, close)
''', '''    if x1_room < ask - 0.05:  # not even one contract at the lowest maker price (ask - 5c) -> X1 sub-cap is full
        return {**rec, "status": "skip_x1_cap"}
    if style == "maker":
        return _decide_maker(rec, client, mode, cfg, ask, no_bid, budget, bal, close)
''')
rep('''    spent, opn = spend_and_open(state, now)
    for p in fresh:
        rec = decide(p, client, mode, cfg, now, spent, opn, bal)
''', '''    spent, opn = spend_and_open(state, now)
    x1s, x1o = spend_and_open({t: o for t, o in state.items() if o.get("cohort") == "x1"}, now)
    for p in fresh:
        rec = decide(p, client, mode, cfg, now, spent, opn, bal, (x1s, x1o))
''')
rep('''        spent, opn = spent + c, opn + c
        if bal is not None:
''', '''        spent, opn = spent + c, opn + c
        if rec.get("cohort") == "x1":
            x1s, x1o = x1s + c, x1o + c
        if bal is not None:
''')
rep('''            f" (kill since {cfg['x1_kill_since'] or 'all'}); price bands "''',
    '''            f" (kill since {cfg['x1_kill_since'] or 'all'}), x1 day/open ${cfg['x1_day'] or 'shared'}; price bands "''')
p.write_text(s)
print("patched", p)
