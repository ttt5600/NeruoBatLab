"""One-off patch of sports/scripts/kalshi_exec.py (ops_e4_prep). Each anchor must match exactly once."""
from pathlib import Path

p = Path(__file__).resolve().parents[2] / "scripts" / "kalshi_exec.py"
s = p.read_text()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:80], s.count(a))
    s = s.replace(a, b)


rep('''  * every fill / partial fill / error / kill / loss stop -> macOS notification (KALSHI_NOTIFY=0 turns it off)
''', '''  * every fill / partial fill / error / kill / loss stop -> macOS notification (KALSHI_NOTIFY=0 turns it off)
  * per-cohort / per-price contract caps (all default OFF; added 2026-11-15, notes/ops_e4_prep.md), applied on top of
    KALSHI_MAX_CONTRACTS in demo/live: KALSHI_X1_FRAC (bankroll share per X1 pick, default 0.005),
    KALSHI_X1_MAX_CONTRACTS (X1-only contract cap), KALSHI_PRICE_BAND_CAPS="0.97:24,0:19" (contracts by NO price at
    the order: the highest floor <= price applies). X1 kill (only while x1 is in KALSHI_COHORTS): settled X1 fills
    since KALSHI_X1_KILL_SINCE (ISO; unset = all) -> >= 2 losses in the first 30, or ROI < 0 once >= 60 -> a sticky
    cohort_kill row is journaled and X1 picks become skip_cohort_killed (orig7 keeps trading).
''')
rep('''MAKER_FREE_FEE_TYPES = ("quadratic",)           # General Trading Fees Table: no maker fee
''', '''MAKER_FREE_FEE_TYPES = ("quadratic",)           # General Trading Fees Table: no maker fee
X1_KILL_FIRST, X1_KILL_LOSSES, X1_KILL_ROI_N = 30, 2, 60  # X1 restart kill (notes/ops_e4_prep.md)
''')
rep('''            "exec_style": _style(e.get("KALSHI_EXEC_STYLE", "taker")), "ab": e.get("KALSHI_AB", "0") == "1"}
''', '''            "exec_style": _style(e.get("KALSHI_EXEC_STYLE", "taker")), "ab": e.get("KALSHI_AB", "0") == "1",
            # 2026-11-15 (ops_e4_prep): X1 restart at minimum size + E4 price-band cap; all off unless set
            "x1_frac": f("KALSHI_X1_FRAC", TIER_FRAC["x1"]),
            "x1_max_contracts": int(e["KALSHI_X1_MAX_CONTRACTS"]) if e.get("KALSHI_X1_MAX_CONTRACTS") else None,
            "price_band_caps": parse_band_caps(e.get("KALSHI_PRICE_BAND_CAPS", "")),
            "x1_kill_since": e.get("KALSHI_X1_KILL_SINCE") or None}


def parse_band_caps(v: str) -> list[tuple[float, int]]:
    """"0.97:24,0:19" -> [(0.97, 24), (0.0, 19)] sorted by floor, highest first. Empty -> [] (off)."""
    out = []
    for part in (v or "").split(","):
        if not part.strip():
            continue
        lo, n = part.split(":")
        lo, n = float(lo), int(n)
        if not 0 <= lo < 1 or n < 0:
            raise ValueError(f"KALSHI_PRICE_BAND_CAPS entry {part!r}: need 0 <= floor < 1 and contracts >= 0")
        out.append((lo, n))
    return sorted(out, reverse=True)


def contract_cap(cfg: dict, cohort: str, price: float, mode: str) -> int:
    """Max contracts for one order: KALSHI_MAX_CONTRACTS, the X1 cap, and the price band. dry = uncapped (as before).
    A price below every band floor is uncapped by the bands."""
    if mode == "dry":
        return 10 ** 9
    n = cfg["max_contracts"]
    if cohort == "x1" and cfg.get("x1_max_contracts") is not None:
        n = min(n, cfg["x1_max_contracts"])
    for lo, cap in cfg.get("price_band_caps") or []:
        if price >= lo - 1e-9:
            n = min(n, cap)
            break
    return n


def _utc(s: str) -> datetime:
    d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def x1_kill_verdict(state: dict, since: str | None) -> str | None:
    """Pre-set X1 restart kill on this mode's settled X1 fills (orders placed at/after `since`): >= X1_KILL_LOSSES
    losses within the first X1_KILL_FIRST, or ROI < 0 once there are >= X1_KILL_ROI_N. None = keep going."""
    lo = _utc(since) if since else None
    rows = sorted((o for o in state.values() if o.get("cohort") == "x1" and o.get("settled") and held(o) > 0
                   and (lo is None or _utc(o["ts"]) >= lo)), key=lambda o: _utc(o["ts"]))
    losses = sum(o.get("result") == "yes" for o in rows[:X1_KILL_FIRST])
    if losses >= X1_KILL_LOSSES:
        return f"X1: {losses} losses within the first {min(len(rows), X1_KILL_FIRST)} settled fills"
    if len(rows) >= X1_KILL_ROI_N:
        pnl, cost = sum(o["pnl"] for o in rows), sum(committed(o) for o in rows)
        if cost and pnl / cost < 0:
            return f"X1: ROI {pnl / cost:+.2%} after {len(rows)} settled fills"
    return None


def x1_killed(hist: list[dict], mode: str, since: str | None) -> bool:
    """Sticky: a cohort_kill row for this mode and this KALSHI_X1_KILL_SINCE keeps X1 off (a new SINCE = a new restart)."""
    return any(r.get("event") == "cohort_kill" and r.get("cohort") == "x1" and r.get("mode") == mode
               and r.get("since") == since for r in hist)
''')
rep('''    frac = cfg.get("orig7_frac", TIER_FRAC["orig7"]) if cohort == "orig7" else TIER_FRAC.get(cohort or "x1", 0.005)
''', '''    frac = (cfg.get("orig7_frac", TIER_FRAC["orig7"]) if cohort == "orig7" else
            cfg.get("x1_frac", TIER_FRAC["x1"]) if cohort in (None, "x1") else TIER_FRAC.get(cohort, 0.005))
''')
rep('''    if rec["cohort"] not in cfg.get("cohorts", {"orig7", "x1"}):
        return {**rec, "status": "skip_cohort_paused"}
''', '''    if rec["cohort"] not in cfg.get("cohorts", {"orig7", "x1"}):
        return {**rec, "status": "skip_cohort_paused"}
    if rec["cohort"] in cfg.get("killed_cohorts", ()):
        return {**rec, "status": "skip_cohort_killed"}
''')
rep('''    n = min(n, int(depth * cfg["depth_frac"]), cfg["max_contracts"] if mode != "dry" else 10 ** 9)
''', '''    ct = contract_cap(cfg, rec["cohort"], ask, mode)
    n = min(n, int(depth * cfg["depth_frac"]), ct)
    if mode != "dry" and ct != cfg["max_contracts"]:
        rec = {**rec, "ct_cap": ct}
''')
rep('''    n = min(n, cfg["max_contracts"] if mode != "dry" else 10 ** 9)
''', '''    ct = contract_cap(cfg, rec["cohort"], q, mode)
    n = min(n, ct)
    if mode != "dry" and ct != cfg["max_contracts"]:
        rec = {**rec, "ct_cap": ct}
''')
rep('''    state = fold(journal(), mode)
    b = cfg["bankroll"]
    if b is not None and mode != "dry":
''', '''    state = fold(journal(), mode)
    if "x1" in cfg["cohorts"]:
        since = cfg.get("x1_kill_since")
        if x1_killed(journal(), mode, since):
            cfg["killed_cohorts"] = {"x1"}
        else:
            why = x1_kill_verdict(state, since)
            if why:
                append({"event": "cohort_kill", "ts": now.isoformat(timespec="seconds"), "mode": mode, "cohort": "x1",
                        "since": since, "reason": why})
                notify(f"Kalshi {mode} X1 cohort KILLED", f"{why}; orig7 continues", cfg)
                cfg["killed_cohorts"] = {"x1"}
    b = cfg["bankroll"]
    if b is not None and mode != "dry":
''')
rep('''            f" 7d -${WEEK_LOSS_FRAC * b:.2f}")
''', '''            f" 7d -${WEEK_LOSS_FRAC * b:.2f}")
        add("PASS", "cohorts / contract caps",
            f"cohorts {','.join(sorted(cfg['cohorts']))}; x1 max ct {cfg['x1_max_contracts'] or cfg['max_contracts']}"
            f" (kill since {cfg['x1_kill_since'] or 'all'}); price bands "
            + (", ".join(f">={lo:.2f}:{n}" for lo, n in cfg["price_band_caps"]) or "off"))
''')
p.write_text(s)
print("patched", p)
