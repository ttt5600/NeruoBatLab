"""DFS pick'em legs vs Pinnacle player-prop fair probabilities. RESEARCH / PAPER ONLY.

    python scripts/pickem.py tables              # exact break-even per leg for fixed payout tables
    python scripts/pickem.py snapshot [--min 0.02] [--save]   # Sleeper Picks vs Pinnacle props, now
    python scripts/pickem.py log                 # same, append every matched leg to data/pickem/log.jsonl (paper)
    python scripts/pickem.py grade               # (not built yet: needs box scores; see notes/soft_venues.md)

Idea (the one that passed for us, soccer S12): a soft price vs the sharp Pinnacle price with its margin
removed by the POWER de-vig. Here the soft price is a pick'em leg.
  * Fixed-table venues (PrizePicks/Underdog standard): every leg pays the same, so a leg is worth taking iff
    its fair P(side) beats the table's per-leg break-even p* (solve sum_k C(n,k) p^k (1-p)^(n-k) M_k = 1).
  * Per-leg-multiplier venues (Sleeper Picks; Underdog has moved this way too): entry payout = product of the
    leg multipliers m_i, so with independent legs EV = prod(p_i m_i) and each leg's break-even is 1/m_i.
A leg "clears" when p_fair - p* >= --min (default 2 pts). p_fair is the CONSERVATIVE fair: min of the power and
multiplicative de-vigs (power takes more off a longshot, multiplicative more off a favourite).
Only exact line matches are used (same player, stat, line). A snapshot is DESCRIPTIVE, not a trial.

Sources (read-only, no login): Pinnacle guest API (sports/live.py); Sleeper api.sleeper.app/lines/available
(robots.txt allows all; public, no auth) + /v1/players/<sport> for names. PrizePicks is behind a DataDome
captcha and Underdog's over_under_lines answers 426 "upgrade required" to non-app clients -> not scraped.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import requests

from sports.live import PIN, PIN_HEADERS, LEAGUES, _get, am_to_dec, devig, devig_power

DATA = ROOT / "data" / "pickem"
SLEEPER_LINES = "https://api.sleeper.app/lines/available"

# Pinnacle "Total <X>" -> Sleeper wager_type, per league
STATMAP = {
    "nba": {"Points": "points", "Rebounds": "rebounds", "Assists": "assists", "Threes Made": "threes_made"},
    "nfl": {"Receiving Yards": "receiving_yards", "Receptions": "receptions", "Rushing Yards": "rushing_yards",
            "Passing Yards": "passing_yards", "Pass Completions": "pass_completions",
            "Pass Attempts": "passing_attempts", "Rush Attempts": "rushing_attempts",
            "Touchdown Passes": "passing_touchdowns", "Touchdowns": "anytime_touchdowns"},
    "nhl": {"Goals": "goals", "Assists": "assists", "Shots On Goal": "shots"},
    "cfb": {"Receiving Yards": "receiving_yards", "Rushing Yards": "rushing_yards", "Passing Yards": "passing_yards",
            "Receptions": "receptions"},
}

# Fixed payout tables: {name: {n_correct: multiplier}} for an n-pick entry. Verify before use (they change).
TABLES = {
    "PP power 2": (2, {2: 3.0}), "PP power 3": (3, {3: 6.0}), "PP power 4": (4, {4: 10.0}),
    "PP power 5": (5, {5: 20.0}), "PP power 6": (6, {6: 37.5}),
    "PP flex 3": (3, {3: 3.0, 2: 1.0}), "PP flex 4": (4, {4: 6.0, 3: 1.5}),
    "PP flex 5": (5, {5: 10.0, 4: 2.0, 3: 0.4}), "PP flex 6": (6, {6: 25.0, 5: 2.0, 4: 0.4}),
    # Underdog standard at 1.0x per-pick multipliers (BettingPros 2026-08-18; third-party, UNVERIFIED in-app)
    "UD std 2": (2, {2: 3.5}), "UD std 3": (3, {3: 6.5}), "UD std 4": (4, {4: 12.0}), "UD std 5": (5, {5: 20.0}),
    "UD std 6": (6, {6: 35.0}),
    "UD flex 3": (3, {3: 3.25, 2: 1.09}), "UD flex 4": (4, {4: 7.2, 3: 1.8}), "UD flex 5": (5, {5: 10.0, 4: 2.5}),
    "UD flex 6": (6, {6: 25.0, 5: 2.6, 4: 0.25}),
}
# PrizePicks tables are its pre-Aug-2025 house-banked "Power/Flex" (BettingPros 2026-08-18). PrizePicks has been
# peer-to-peer "Arena" only since 2025-08-22; whether Arena still pays these fixed multipliers is UNVERIFIED.


def breakeven(n: int, pay: dict[int, float]) -> float:
    """Per-leg p (iid legs) at which the entry's expected return is exactly 1."""
    f = lambda p: sum(comb(n, k) * p ** k * (1 - p) ** (n - k) * m for k, m in pay.items()) - 1
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return (lo + hi) / 2


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", "", s.replace("-", " "))
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def pin_props(league: str) -> pd.DataFrame:
    lid = LEAGUES[league][0]
    mu = _get(f"{PIN}/leagues/{lid}/matchups", headers=PIN_HEADERS)
    sp = {}
    for m in mu:
        if m.get("type") != "special" or (m.get("special") or {}).get("category") != "Player Props" or m.get("isLive"):
            continue
        d = m["special"]["description"]
        mm = re.match(r"^(.*?) Total (.+?)(?: \(.*\))?$", d)
        if not mm or mm.group(2) not in STATMAP[league]:
            continue
        side = {p["id"]: p["name"].lower() for p in m["participants"]}
        par = m.get("parent") or {}
        sp[m["id"]] = {"player": mm.group(1), "stat": STATMAP[league][mm.group(2)], "side": side,
                       "start": par.get("startTime") or m.get("startTime"),
                       "game": " @ ".join(p["name"] for p in sorted(par.get("participants", []),
                                                                      key=lambda p: p["alignment"] != "away"))}
    rows = []
    for k in _get(f"{PIN}/leagues/{lid}/markets/straight", headers=PIN_HEADERS):
        s = sp.get(k["matchupId"])
        if s is None or k.get("status", "open") != "open" or k.get("period") != 0:
            continue
        pr = {s["side"].get(p["participantId"]): p for p in k["prices"]}
        if not {"over", "under"} <= set(pr) or pr["over"].get("points") != pr["under"].get("points"):
            continue
        po, pu = 1 / am_to_dec(pr["over"]["price"]), 1 / am_to_dec(pr["under"]["price"])
        mo, _ = devig(po, pu)
        qo, _ = devig_power(po, pu)
        rows.append({"league": league, "player": s["player"], "key": norm(s["player"]), "stat": s["stat"],
                     "line": float(pr["over"]["points"]), "pin_over": pr["over"]["price"],
                     "pin_under": pr["under"]["price"], "hold": po + pu - 1,
                     "fair_over_mult": mo, "fair_over_pow": qo, "start": s["start"], "game": s["game"],
                     "max_risk": next((l["amount"] for l in k.get("limits", []) if l["type"] == "maxRiskStake"), None)})
    return pd.DataFrame(rows)


def sleeper_names(sport: str) -> dict[str, str]:
    f = DATA / "players" / f"{sport}.json"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(f"https://api.sleeper.app/v1/players/{sport}", timeout=120)
        r.raise_for_status()
        f.write_bytes(r.content)
    d = json.loads(f.read_text())
    out = {}
    for pid, p in d.items():
        nm = p.get("full_name") or f"{p.get('first_name', '')} {p.get('last_name', '')}"
        out[str(pid)] = nm
    return out


def sleeper_lines() -> pd.DataFrame:
    r = requests.get(SLEEPER_LINES, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    rows = []
    for mkt in r.json():
        if mkt.get("status") != "active" or mkt.get("subject_type") != "player" or mkt.get("game_status") != "pre_game":
            continue
        for o in mkt.get("options", []):
            if o.get("status") != "active" or o.get("line_type") != "normal" or o.get("outcome") not in ("over", "under"):
                continue
            rows.append({"sport": o["sport"], "subject_id": str(o["subject_id"]), "stat": o["wager_type"],
                         "line": float(o["outcome_value"]), "side": o["outcome"],
                         "mult": float(o["payout_multiplier"]), "team": o.get("subject_team"),
                         "updated_at": mkt.get("updated_at")})
    return pd.DataFrame(rows)


def snapshot(min_edge: float) -> tuple[pd.DataFrame, dict]:
    sl = sleeper_lines()
    out, stats = [], {}
    for lg in STATMAP:
        pin = pin_props(lg)
        s = sl[sl["sport"] == lg].copy()
        if s.empty or pin.empty:
            stats[lg] = {"pin_props": len(pin), "sleeper_legs": len(s), "matched": 0}
            continue
        names = sleeper_names(lg)
        s["key"] = s["subject_id"].map(lambda i: norm(names.get(i, "")))
        m = s.merge(pin, on=["key", "stat"], how="inner", suffixes=("", "_pin"))
        # exact line, or a pick'em line that is BETTER for the side than Pinnacle's: then Pinnacle's fair P(side)
        # at its own line is a lower bound (P(X > 6.5) >= P(X > 7.5)). Worse-for-the-side lines are dropped.
        m = m[(m["line"] == m["line_pin"]) | ((m["side"] == "over") & (m["line"] < m["line_pin"]))
              | ((m["side"] == "under") & (m["line"] > m["line_pin"]))].copy()
        m["line_gap"] = (m["line"] - m["line_pin"]).abs()
        stats[lg] = {"pin_props": len(pin), "sleeper_legs": len(s), "matched": len(m),
                     "pin_props_with_sleeper_player_stat": int(pin.merge(s[["key", "stat"]].drop_duplicates(),
                                                                         on=["key", "stat"]).shape[0])}
        if m.empty:
            continue
        over = m["side"] == "over"
        m["fair_pow"] = m["fair_over_pow"].where(over, 1 - m["fair_over_pow"])
        m["fair_mult"] = m["fair_over_mult"].where(over, 1 - m["fair_over_mult"])
        m["fair"] = m[["fair_pow", "fair_mult"]].min(axis=1)
        m["be"] = 1 / m["mult"]
        m["edge_pts"] = m["fair"] - m["be"]
        m["ev_leg"] = m["fair"] * m["mult"] - 1
        out.append(m)
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return df, stats


def cmd_tables(_):
    print(f"{'table':14s} {'break-even p per leg':>22s}")
    for name, (n, pay) in TABLES.items():
        print(f"{name:14s} {breakeven(n, pay):22.4f}")


def cmd_snapshot(a):
    df, stats = snapshot(a.min)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print("snapshot", ts)
    for lg, s in stats.items():
        print(f"  {lg}: {s}")
    if df.empty:
        print("no matched legs")
        return df, ts
    print(f"matched legs {len(df)}; fair-vs-break-even (pts): median {df.edge_pts.median()*100:+.1f}, "
          f"p90 {df.edge_pts.quantile(.9)*100:+.1f}, max {df.edge_pts.max()*100:+.1f}")
    ex = df[df.line_gap == 0]
    print(f"  exact-line legs {len(ex)} (median edge {ex.edge_pts.median()*100:+.1f} pts); better-line legs "
          f"{len(df) - len(ex)} (fair is a LOWER bound there)")
    hit = df[df.edge_pts >= a.min].sort_values("edge_pts", ascending=False)
    print(f"legs clearing break-even by >= {a.min*100:.0f} pts: {len(hit)} of {len(df)}")
    cols = ["league", "player", "stat", "line", "line_pin", "side", "mult", "be", "fair", "edge_pts", "ev_leg", "pin_over",
            "pin_under", "max_risk", "start"]
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(hit[cols].head(40).round(3).to_string(index=False))
    if a.save:
        f = DATA / f"snapshot_{ts[:16].replace(':', '')}.csv"
        df.to_csv(f, index=False)
        print("saved", f)
    return df, ts


def cmd_log(a):
    df, stats = snapshot(a.min)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / "log.jsonl", "a") as fh:
        for r in df.to_dict("records"):
            fh.write(json.dumps({"ts": ts, **{k: r[k] for k in ["league", "player", "stat", "line", "line_pin", "side", "mult",
                                                                 "fair", "fair_pow", "fair_mult", "edge_pts",
                                                                 "pin_over", "pin_under", "start", "game"]}}) + "\n")
    print(ts, "logged", len(df), "legs;", int((df.edge_pts >= a.min).sum()) if len(df) else 0, "clear", stats)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tables")
    for c in ("snapshot", "log"):
        p = sub.add_parser(c)
        p.add_argument("--min", type=float, default=0.02)
        p.add_argument("--save", action="store_true")
    a = ap.parse_args()
    {"tables": cmd_tables, "snapshot": cmd_snapshot, "log": cmd_log}[a.cmd](a)


if __name__ == "__main__":
    main()
