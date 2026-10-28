"""Fast, dependency-light view for the live board (stdlib only: no pandas/scipy, so `lb` answers in well under 0.5 s).

    live_board.py game              -> every game in the cache, one line each (the `show` from cache)
    live_board.py game lakers       -> the per-game check block for one game, from data/live_board_latest.json
    live_board.py game POR@LAL --fresh   -> live pull of just that game (handled in live_board.py, rendered here)

`tick` (launchd) writes data/live_board_latest.json: one record per game with the raw state, Kalshi quotes, model,
history neighbours, pregame lines/totals and our open manual_bets positions. This module only RENDERS those records;
it computes nothing that needs a feed. Verdict rules are fixed (lead 2026-10-27): |model - mid| < 4c fair; >= 8c
"market likely knows something"; else small gap. Never a bet from the in-game model (sealed L28 + L37 both failed).
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("LIVE_BOARD_DATA", ROOT / "data"))
LATEST = DATA / "live_board_latest.json"
ET = ZoneInfo("America/New_York")
REG = {"nba": 2880, "nfl": 3600, "nhl": 3600}
LIM = {"nba": 4, "nfl": 4, "nhl": 3}
FAIR, BIG = 0.04, 0.08
STALE_S = 180  # older cache -> fall back to a fresh pull (agent ticks every 30 s)
HIST_DESC = {"nba": "dev seasons 2015-2024, time +-2 min, margin +-2, trailer spread +-3",
             "nfl": "dev seasons 2006-2019, time +-2 min, same margin, spread +-3",
             "nhl": "dev seasons 2017-2022, time +-2 min, same goal diff, pregame P +-0.08"}


def ok(x) -> bool:
    return x is not None and x == x


def clk(s) -> str:
    s = float(s or 0)
    return f"{int(s // 60)}:{int(s % 60):02d}"


def per_label(g: dict) -> str:
    p, lim = int(g.get("period") or 0), LIM[g["sport"]]
    if p > lim:
        return f"OT{p - lim if p - lim > 1 else ''}"
    return f"{'P' if g['sport'] == 'nhl' else 'Q'}{p}"


def load_latest() -> dict:
    try:
        return json.loads(LATEST.read_text())
    except Exception:  # noqa: BLE001
        return {}


def age_s(ts: str | None) -> float:
    if not ts:
        return 1e9
    return (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds()


# ------------------------------------------------------------------------------------------------ matching
def tokens(q: str) -> list[str]:
    return [t for t in re.split(r"\s*(?:@|,|\bvs\.?\b|\bat\b|\s)\s*", q.lower().strip()) if t]


def team_hit(tok: str, g: dict) -> str | None:
    """-> 'home' / 'away' / None. Exact code first, then name/city/nickname prefix or substring (>= 3 chars)."""
    for side in ("home", "away"):
        if tok == g[side].lower():
            return side
    for side in ("home", "away"):
        names = g.get(f"names_{side}") or []
        if len(tok) >= 3 and any(n.startswith(tok) or f" {tok}" in f" {n}" for n in names):
            return side
    return None


def match(recs: dict, q: str) -> list[tuple[str, str | None]]:
    """-> [(key, focus team code)], live first, then pre, then final."""
    toks = tokens(q)
    out = []
    for k, rec in recs.items():
        g = rec["g"]
        hits = [team_hit(t, g) for t in toks]
        if toks and all(hits):
            focus = g[hits[0]] if len(set(hits)) == 1 else g["home"]
            out.append((k, focus))
    order = {"in": 0, "pre": 1, "post": 2}
    out.sort(key=lambda x: (order.get(recs[x[0]]["g"]["state"], 3), recs[x[0]]["g"].get("start") or ""))
    return out


# ------------------------------------------------------------------------------------------------ render
def side_mid(b, a):
    return (b + a) / 2 if ok(b) and ok(a) and a > 0 else None


def verdict(rec: dict, focus: str) -> tuple[str, str]:
    """-> (verdict text, short)."""
    g, r = rec["g"], rec["r"]
    if g["state"] == "post":
        return "game over", "final"
    if g["state"] == "pre":
        return "not started -- in-game model not applicable", "pre"
    p, m = r.get("p_home"), r.get("mid_home")
    if not ok(p) or not ok(m):
        return "no model price (OT / no pregame line / no Kalshi market) -- no bet", "no model"
    d = abs(p - m)
    base = "fair, no bet" if d < FAIR else (
        "big gap: market likely knows something (injury/fouls/news) -- not value" if d >= BIG else "small gap, no bet")
    gaps = [(g["home"], r.get("gap_home")), (g["away"], r.get("gap_away"))]
    gaps = [x for x in gaps if ok(x[1])]
    tilt = ""
    if gaps:
        t, v = max(gaps, key=lambda x: x[1])
        tilt = (f"model tilt {v * 100:+.1f}c on {t} after fee" if v > 0 else "no side positive after fee")
    return (f"{base} ({tilt}{'; ' if tilt else ''}sealed L28/L37 in-game tests failed -- never a bet from this model)",
            base.split(":")[0].split(",")[0] if d >= BIG else base)


def render(rec: dict, focus: str | None = None) -> tuple[list[str], str]:
    """-> (block lines, one-line summary for notifications)."""
    g, r, pre, tot = rec["g"], rec["r"], rec.get("pre") or {}, rec.get("tot") or {}
    sp, home, away = g["sport"], g["home"], g["away"]
    F = focus if focus in (home, away) else home
    O = away if F == home else home
    hf = F == home
    hs, as_ = int(g.get("hs") or 0), int(g.get("as") or 0)
    L = []
    # 1. score / clock
    if g["state"] == "pre":
        st = datetime.fromisoformat(g["start"]).astimezone(ET)
        head = f"{away}@{home} not started (start {st:%H:%M} ET)."
    elif g["state"] == "post":
        w = home if hs > as_ else away
        head = f"FINAL: {w} wins {max(hs, as_)}-{min(hs, as_)} ({away}@{home})."
    else:
        lead = home if hs > as_ else away
        when = f"{clk(g['clock_s'])} {per_label(g)}"
        if sp == "nhl" and g.get("intermission"):
            when = f"intermission after {per_label(g)}"
        head = f"{lead} leads {max(hs, as_)}-{min(hs, as_)}, {when}." if hs != as_ else f"Tied {hs}-{as_}, {when}."
        if sp == "nfl" and g.get("pos"):
            head += f" {home if g['pos'] == 'home' else away} ball {g.get('down')}&{g.get('dist')} ({g.get('yl100')} yds to go)."
        if sp == "nhl" and g.get("pp"):
            head += f" {home if g['pp'] > 0 else away} power play."
        if sp == "nhl" and (g.get("en_home") or g.get("en_away")):
            head += " EMPTY NET."
    L.append(head)
    # 2. Kalshi
    hb, ha, ab, aa = r.get("h_bid"), r.get("h_ask"), r.get("a_bid"), r.get("a_ask")
    fb, fa, ob, oa = (hb, ha, ab, aa) if hf else (ab, aa, hb, ha)
    mF, mO = side_mid(fb, fa), side_mid(ob, oa)
    c = lambda x: "-" if not ok(x) else f"{x * 100:.0f}"  # noqa: E731
    if r.get("kalshi_event") and (ok(fb) or ok(fa)):
        L.append(f"Kalshi {F} {c(fb)}/{c(fa)}" + (f" (mid {mF:.1%})" if mF is not None else "") +
                 f"; {O} {c(ob)}/{c(oa)}" + (f" (mid {mO:.1%})" if mO is not None else "") + ".")
    else:
        L.append("Kalshi: no open game market matched" + (" (settled/closed)" if g["state"] == "post" else "") + ".")
    # 3. model
    p = r.get("p_home")
    pF = None if not ok(p) else (p if hf else 1 - p)
    flag = r.get("flag") or ""
    if g["state"] == "in":
        L.append(f"Our model: {F} {pF:.1%}" + (f" [{flag}]" if flag else "") + "." if pF is not None
                 else f"Our model: none ({flag or 'n/a'}).")
    # 4. history
    if g["state"] == "in":
        n, ph = r.get("hist_n"), r.get("hist_p")
        if n and ok(ph):
            mar = hs - as_
            ms = "tied" if mar == 0 else f"home {'up' if mar > 0 else 'down'} {abs(mar)}"
            cs = round(float(g["clock_s"]) / 10) * 10
            prs = (f"pregame home {-pre['hm']:+.1f}" if ok(pre.get("hm")) else "") if sp != "nhl" else (
                f"pregame P(home) {pre['p_pre']:.2f}" if ok(pre.get("p_pre")) else "")
            hF = ph if hf else 1 - ph
            L.append(f"History ({n:,} similar: {ms}, {per_label(g)} ~{clk(cs)}, {prs}) {F} wins {hF:.0%} "
                     f"[{HIST_DESC[sp]}].")
        else:
            L.append(f"History: no neighbour games ({'tied: NBA table is trailer-based' if hs == as_ and sp == 'nba' else 'n=0 or n/a'}).")
    # 5. pregame lines
    pl = []
    if sp != "nhl" and ok(pre.get("hm")):
        hm = pre["hm"]
        pl.append("pick'em" if hm == 0 else f"{home if hm > 0 else away} -{abs(hm):g}" + f" ({pre.get('src')})")
    if sp == "nhl" and ok(pre.get("p_pre")):
        pl.append(f"P({home}) {pre['p_pre']:.0%} ({pre.get('src')})")
    ptot = pre.get("total")
    if ok(ptot):
        pl.append(f"total {ptot:g} ({pre.get('total_src')})")
    if ok(tot.get("pin_last")):
        pl.append(f"Pinnacle last pre-tip total {tot['pin_last']:g}")
    if ok(tot.get("kal_pre")):
        pl.append(f"Kalshi pre-tip implied total {tot['kal_pre']:.1f}")
    L.append("Pregame: " + ("; ".join(pl) if pl else "no line cached") + ".")
    # 6. totals / pace
    T = hs + as_
    ref = tot.get("pin_last") if ok(tot.get("pin_last")) else ptot
    if g["state"] == "in":
        el = g.get("el")
        if ok(el) and el >= 120:
            proj = T * REG[sp] / el
            s = f"Total: {T} with {clk(REG[sp] - el)} left in regulation, pace ~{proj:.0f}"
            if ok(ref):
                d = proj - ref
                word = "on pace" if abs(d) < 4 else (f"{'over' if d > 0 else 'under'}" if abs(d) < 12
                                                     else f"way {'over' if d > 0 else 'under'}")
                s += f" vs pregame {ref:g} -> {word}"
            L.append(s + " (descriptive only: Kalshi live totals already price pace).")
        else:
            L.append(f"Total: {T} ({'OT' if el is None else 'too early for pace'}).")
    elif g["state"] == "post" and ok(ref):
        L.append(f"Total: final {T} vs pregame {ref:g} -> {'over' if T > ref else 'under' if T < ref else 'push'}.")
    # 7. our bets
    if rec.get("pos_err"):
        L.append(f"Our bets: unknown (orders API: {rec['pos_err'][:80]}).")
    elif not rec.get("pos"):
        L.append("Our bets on this game: none.")
    for q in rec.get("pos") or []:
        if q.get("filled"):
            now_ = q.get("now")
            tail = (f" -> {'settled' if q.get('settled') else 'mid'} {now_:.2f}, value ${q['filled'] * now_:.2f}, "
                    f"P&L {q['pnl']:+.2f}") if ok(now_) else " -> no quote"
            L.append(f"OUR BET: {q['side'].upper()} {q['label']} x{q['filled']:g} @ {q['entry']:.2f}{tail}.")
        if q.get("resting"):
            L.append(f"Resting order: {q['side'].upper()} {q['label']} x{q['resting']:g} @ {q['entry']:.2f} (unfilled).")
    # 8. verdict
    v, vs = verdict(rec, F)
    L.append(f"Verdict: {v}.")
    summ = head
    if g["state"] == "in":
        bits = []
        if mF is not None:
            bits.append(f"K {F} {mF:.0%}")
        if pF is not None:
            bits.append(f"model {pF:.0%}")
        if r.get("hist_n") and ok(r.get("hist_p")):
            bits.append(f"hist {(r['hist_p'] if hf else 1 - r['hist_p']):.0%}")
        el = g.get("el")
        if ok(el) and el >= 120:
            bits.append(f"tot {T} pace~{T * REG[sp] / el:.0f}" + (f" v {ref:g}" if ok(ref) else ""))
        summ += " | " + " ".join(bits) + f" | {vs}"
    for q in rec.get("pos") or []:
        if q.get("filled") and ok(q.get("pnl")):
            summ += f" | our {q['label']} {q['pnl']:+.2f}"
    return L, summ


# ------------------------------------------------------------------------------------------------ CLI
def show_all(d: dict) -> int:
    recs = d.get("games") or {}
    order = {"in": 0, "pre": 2, "post": 1}
    ks = sorted(recs, key=lambda k: (order.get(recs[k]["g"]["state"], 3), recs[k]["g"]["sport"], recs[k]["g"].get("start") or ""))
    live = [k for k in ks if recs[k]["g"]["state"] == "in"]
    print(f"live board cache as of {age_s(d.get('ts')):.0f}s ago: {len(live)} live, {len(recs)} games today")
    for k in ks:
        g = recs[k]["g"]
        if g["state"] == "pre" and age_s(g.get("start")) < -6 * 3600:
            continue
        print(f"  {g['sport'].upper()} {render(recs[k])[1]}")
    print("lb <team>  -> full check; lb <team> --fresh -> live pull. Decision support only, no orders.")
    return 0


def main(args: list[str]) -> int | None:
    """Return None to make live_board.py fall back to a fresh pull (no cache / stale / no match)."""
    d = load_latest()
    a = age_s(d.get("ts"))
    q = " ".join(x for x in args if not x.startswith("--"))
    if not q:
        if not d:
            print("no cache yet (data/live_board_latest.json); is the launchd tick running?")
            return 1
        return show_all(d)
    if not d or a > STALE_S:
        return None
    hits = match(d.get("games") or {}, q)
    if not hits:
        return None
    live = [h for h in hits if d["games"][h[0]]["g"]["state"] == "in"]
    for k, focus in (live or hits[:1]):
        rec = d["games"][k]
        lines, _ = render(rec, focus)
        print(f"[{rec['g']['sport'].upper()} {rec['g']['game']}] as of {a:.0f}s ago (lb {q} --fresh for a live pull)")
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    r = main(sys.argv[1:])
    raise SystemExit(1 if r is None else r)
