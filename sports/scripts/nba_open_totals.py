"""NBA OPENING-TOTALS LOGGER (L32 forward paper). Paper only -- no orders.

    python scripts/nba_open_totals.py tick      # one pass (launchd, every 10 min)
    python scripts/nba_open_totals.py report    # picks, W-L, ROI (no-vig / Kalshi ask), CLV
    python scripts/nba_open_totals.py show      # tracked games: open vs latest vs model

Why (user 2026-10-25, "yes L32, and build the opening totals logger"): the simple NBA model (scripts/nba_simple_model.py,
notes/nba_simple_model.md) beat OPENING totals on dev (model >= 3 pts off the open: 1,190 bets 55.5%, no-vig +8.7%) but
not the close. That edge only exists if we see the number when it is first posted. This logger catches the first
Pinnacle total we observe (10-min polling, so "open" = first sighting, not the book's true open) and the first Kalshi
KXNBATOTAL ladder, and paper-bets the model gap.

PRE-REGISTERED 2026-10-25 (before the first tick):
  OPEN       = first Pinnacle full-game main total seen for a game (period 0, not alternate), plus its prices.
  MODEL      = nba_simple_live.predict(date)["model_total"] (frozen T2 stack, STRICT availability + current ESPN
               rosters + ESPN injury feed); model_total_avail/_prereg/_q_out are logged alongside, never picked on,
               computed at the DECISION tick (first tick where both the open and a model total exist).
  RULE       = |model - OPEN line| >= 3.0 -> OVER if model > line else UNDER; one pick per game; flat 1u.
               Decision must happen >= 30 min before tip; later first sightings are logged but not picked.
  PRICES     = recorded at the decision tick: (a) the OPEN line at Pinnacle no-vig (proportional) = proxy for a
               resting Kalshi maker order at fair (~0 maker fee); (b) Pinnacle line/prices at the decision tick;
               (c) the Kalshi strike equal to the bet line if one exists: taker ask + fee.
  GRADING    = vs the OPEN line (push on exact integer); ROI at no-vig open, at -110, at Kalshi taker (when quoted).
  FIRST RUN  = games already listed at the logger's first tick are marked pre-existing and never picked.
  CLV        = last Pinnacle main total seen before tip minus OPEN line, signed in the bet's direction (points).
  BC VARIANT = (added 2026-10-26, before any real pick; MODELS_GUIDE.md: the totals model runs ~+2.3 pts above the close
               on dev, post-hoc bias-corrected +7.2% no-vig) bias = mean(model - last pre-tip Pinnacle total) over the
               most recent 50 started games this logger decided (needs >= 30); BC pick when |model - bias - OPEN| >= 3.0;
               graded/priced exactly like the main rule; logged as type pick_bc / settle_bc; judged separately.
  JUDGE      = after 200 settled picks: CLV > 0 (one-sided p < .05) is the leading indicator; kill if mean CLV <= 0
               after 100 picks. ROI needs ~1,000+ picks to say anything (55% vs 52.4% break-even).
Files: data/open_totals_state.json (per-game state), data/open_totals_picks.jsonl (picks + settlements).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from sports import live
from sports.live import KALSHI, _get, am_to_dec, devig, kalshi_fee

STATE = ROOT / "data" / "open_totals_state.json"
PICKS = ROOT / "data" / "open_totals_picks.jsonl"
GAP = 3.0
BC_MIN_N = 30   # BC variant needs >= 30 earlier decided games with a pre-tip close to estimate the level bias
BC_WINDOW = 50  # most recent games used for the bias


def level_bias(st: dict, t: pd.Timestamp) -> tuple[float | None, int]:
    """Mean (model - last pre-tip Pinnacle total) over the most recent BC_WINDOW games that have started."""
    xs = sorted((pd.Timestamp(g["start"]), g["decision"]["model"] - g["pin_last"]["line"]) for g in st.values()
                if (g.get("decision") or {}).get("model") is not None and g.get("pin_last")
                and pd.Timestamp(g["start"]) < t)[-BC_WINDOW:]
    return (float(np.mean([x for _, x in xs])), len(xs)) if xs else (None, 0)
MIN_LEAD = pd.Timedelta(minutes=30)
ET = "America/New_York"

# every abbreviation / city / nickname we meet -> nickname (lower)
TEAMS = {
    "atl": "hawks", "bos": "celtics", "bkn": "nets", "bk": "nets", "cha": "hornets", "chi": "bulls", "cle": "cavaliers",
    "dal": "mavericks", "den": "nuggets", "det": "pistons", "gs": "warriors", "gsw": "warriors", "hou": "rockets",
    "ind": "pacers", "lac": "clippers", "lal": "lakers", "mem": "grizzlies", "mia": "heat", "mil": "bucks",
    "min": "timberwolves", "no": "pelicans", "nop": "pelicans", "ny": "knicks", "nyk": "knicks", "okc": "thunder",
    "orl": "magic", "phi": "76ers", "phx": "suns", "pho": "suns", "por": "trail blazers", "sac": "kings",
    "sa": "spurs", "sas": "spurs", "tor": "raptors", "uta": "jazz", "utah": "jazz", "was": "wizards", "wsh": "wizards",
}
NICKS = sorted(set(TEAMS.values()), key=len, reverse=True)


def nick(name: str) -> str | None:
    s = str(name).lower().strip()
    if s in TEAMS:
        return TEAMS[s]
    for n in NICKS:
        if n in s:
            return n
    return None


def now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def key(date_et: str, home: str, away: str) -> str:
    return f"{date_et}|{away}@{home}"


# ---------------------------------------------------------------- sources

def pin_totals() -> list[dict]:
    d = live.pinnacle("nba")
    if d.empty:
        return []
    d = d[(d.market == "total") & (~d.live)]
    out = []
    for pid, g in d.groupby("pin_id"):
        r = g.iloc[(g.hold).argsort()].iloc[0] if len(g) > 1 else g.iloc[0]  # main line = lowest hold
        h, a = nick(r.home), nick(r.away)
        if not h or not a:
            continue
        start = pd.Timestamp(r.start)
        out.append({"key": key(start.tz_convert(ET).strftime("%Y-%m-%d"), h, a), "start": start.isoformat(),
                    "home": h, "away": a, "line": float(r.line), "over": float(r.price1), "under": float(r.price2),
                    "fair_over": float(r.fair1)})
    return out


def kalshi_ladders() -> dict[str, dict]:
    """event -> {key, strikes: {line: (yes_bid, yes_ask, no_bid, no_ask)}, implied}"""
    out, cursor = {}, None
    while True:
        p = {"series_ticker": "KXNBATOTAL", "status": "open", "limit": 1000}
        if cursor:
            p["cursor"] = cursor
        d = _get(f"{KALSHI}/markets", params=p)
        for m in d.get("markets", []):
            ev = m["event_ticker"]                       # KXNBATOTAL-26OCT26GSWDEN
            tail = ev.split("-")[1]
            try:
                date = datetime.strptime(tail[:7], "%y%b%d").strftime("%Y-%m-%d")
            except ValueError:
                continue
            a, h = nick(tail[7:10]), nick(tail[10:13])
            if not a or not h:
                continue
            f = lambda k: float(m[k]) if m.get(k) not in (None, "") else np.nan
            e = out.setdefault(ev, {"key": key(date, h, a), "strikes": {}})
            if m.get("floor_strike") is not None:
                e["strikes"][float(m["floor_strike"])] = (f("yes_bid_dollars"), f("yes_ask_dollars"),
                                                          f("no_bid_dollars"), f("no_ask_dollars"))
        cursor = d.get("cursor")
        if not cursor or not d.get("markets"):
            break
    for e in out.values():  # implied total: where P(over) mid crosses 0.5
        pts = sorted((k, (v[0] + v[1]) / 2) for k, v in e["strikes"].items()
                     if np.isfinite(v[0]) and np.isfinite(v[1]) and v[1] - v[0] <= 0.10)
        e["implied"] = None
        for (k1, p1), (k2, p2) in zip(pts, pts[1:]):
            if p1 >= 0.5 >= p2 and p1 != p2:
                e["implied"] = round(k1 + (p1 - 0.5) / (p1 - p2) * (k2 - k1), 2)
                break
    return out


def model_totals(date_et: str) -> dict[str, float]:
    try:
        from nba_simple_live import predict
        rows = predict(date_et)
    except Exception as e:  # noqa: BLE001 -- logged, never fatal
        print(f"model unavailable for {date_et}: {type(e).__name__}: {e}")
        return {}
    out = {}
    for r in rows:
        h, a = nick(r["home"]), nick(r["away"])
        if h and a and r.get("model_total") is not None:
            out[key(date_et, h, a)] = {"total": float(r["model_total"]), "warnings": r.get("warnings"),
                                       "avail": r.get("model_total_avail"), "prereg": r.get("model_total_prereg"),
                                       "q_out": r.get("model_total_q_out")}
    return out


def finals(date_et: str) -> dict[str, int]:
    d = _get("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
             params={"dates": date_et.replace("-", "")})
    out = {}
    for e in d.get("events", []):
        if not e["status"]["type"].get("completed"):
            continue
        c = {t["homeAway"]: t for t in e["competitions"][0]["competitors"]}
        h, a = nick(c["home"]["team"]["displayName"]), nick(c["away"]["team"]["displayName"])
        out[key(date_et, h, a)] = int(c["home"]["score"]) + int(c["away"]["score"])
    return out


# ---------------------------------------------------------------- tick

def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def cmd_tick() -> int:
    t = now()
    st = load_state()
    first_run = not st
    errs = []
    try:
        pins = pin_totals()
    except Exception as e:  # noqa: BLE001
        pins, _ = [], errs.append(f"pinnacle {type(e).__name__}")
    try:
        kal = kalshi_ladders()
    except Exception as e:  # noqa: BLE001
        kal, _ = {}, errs.append(f"kalshi {type(e).__name__}")

    for p in pins:
        g = st.setdefault(p["key"], {"home": p["home"], "away": p["away"], "start": p["start"]})
        if first_run:  # already listed before the logger existed: its first sighting is not an open
            g["decision"] = {"t": t.isoformat(), "pick": None, "why": "pre-existing at first logger run"}
        g["start"] = p["start"]
        snap = {"t": t.isoformat(), "line": p["line"], "over": p["over"], "under": p["under"], "fair_over": p["fair_over"]}
        if pd.Timestamp(p["start"]) > t:
            g.setdefault("pin_open", snap)
            g["pin_last"] = snap
    for ev, e in kal.items():
        g = st.get(e["key"])
        if g is None:
            continue
        snap = {"t": t.isoformat(), "implied": e["implied"], "event": ev}
        if pd.Timestamp(g["start"]) > t:
            g.setdefault("kal_open", snap)
            g["kal_last"] = snap
        g["_ladder"] = e["strikes"]  # transient, for pricing the pick below

    # decisions
    need = sorted({k.split("|")[0] for k, g in st.items() if "pin_open" in g and "decision" not in g
                   and pd.Timestamp(g["start"]) - t >= MIN_LEAD})
    models = {}
    for d in need:
        models.update(model_totals(d))
    new = 0
    for k, g in st.items():
        if "pin_open" not in g or "decision" in g:
            continue
        if pd.Timestamp(g["start"]) - t < MIN_LEAD:
            g["decision"] = {"t": t.isoformat(), "pick": None, "why": "first seen < 30 min before tip"}
            continue
        m = models.get(k)
        if m is None:
            continue
        o = g["pin_open"]
        gap = m["total"] - o["line"]
        dec = {"t": t.isoformat(), "model": round(m["total"], 2), "open_line": o["line"], "gap": round(gap, 2),
               "model_warnings": m.get("warnings"), "model_avail": m.get("avail"), "model_prereg": m.get("prereg"),
               "model_q_out": m.get("q_out"), "pick": None}
        # BC variant (pre-registered 2026-10-26): subtract the model's own recent level bias vs the Pinnacle close
        bias, nb = level_bias(st, t)
        dec.update(bias=None if bias is None else round(bias, 2), n_bias=nb, pick_bc=None)
        if bias is not None and nb >= BC_MIN_N:
            gap_bc = m["total"] - bias - o["line"]
            dec["gap_bc"] = round(gap_bc, 2)
            if abs(gap_bc) >= GAP:
                side_bc = "over" if gap_bc > 0 else "under"
                fo, _ = devig(1 / am_to_dec(o["over"]), 1 / am_to_dec(o["under"]))
                dec.update(pick_bc=side_bc, nv_prob_open_bc=round(fo if side_bc == "over" else 1 - fo, 4))
                with PICKS.open("a") as f:
                    f.write(json.dumps({"type": "pick_bc", "key": k, "start": g["start"], **dec}) + "\n")
        if abs(gap) >= GAP:
            side = "over" if gap > 0 else "under"
            f_over, _ = devig(1 / am_to_dec(o["over"]), 1 / am_to_dec(o["under"]))
            q = f_over if side == "over" else 1 - f_over
            dec.update(pick=side, nv_prob_open=round(q, 4), price_open_am=o[side],
                       pin_now_line=g["pin_last"]["line"], pin_now_am=g["pin_last"][side])
            lad = g.get("_ladder", {}).get(o["line"])
            if lad:
                ask = lad[1] if side == "over" else lad[3]
                if np.isfinite(ask):
                    dec.update(kal_ask=ask, kal_fee=round(kalshi_fee(ask), 4))
            with PICKS.open("a") as f:
                f.write(json.dumps({"type": "pick", "key": k, "start": g["start"], **dec}) + "\n")
            new += 1
        g["decision"] = dec
    for g in st.values():
        g.pop("_ladder", None)

    # settle
    open_keys = [k for k, g in st.items() if ((g.get("decision") or {}).get("pick") or (g.get("decision") or {}).get("pick_bc"))
                 and "final_total" not in g and t - pd.Timestamp(g["start"]) > pd.Timedelta(hours=3)]
    for d in sorted({k.split("|")[0] for k in open_keys}):
        try:
            fin = finals(d)
        except Exception as e:  # noqa: BLE001
            errs.append(f"espn {type(e).__name__}")
            continue
        for k in open_keys:
            if k in fin:
                g = st[k]
                dec = g["decision"]
                g["final_total"] = fin[k]
                diff = fin[k] - dec["open_line"]
                close = g["pin_last"]["line"]
                grade = lambda side: ("push" if diff == 0 else ("win" if (diff > 0) == (side == "over") else "loss"))
                cl = lambda side: (close - dec["open_line"]) * (1 if side == "over" else -1)
                with PICKS.open("a") as f:
                    if dec.get("pick"):
                        f.write(json.dumps({"type": "settle", "key": k, "final_total": fin[k], "result": grade(dec["pick"]),
                                            "close_line": close, "clv_pts": cl(dec["pick"])}) + "\n")
                    if dec.get("pick_bc"):
                        f.write(json.dumps({"type": "settle_bc", "key": k, "final_total": fin[k],
                                            "result": grade(dec["pick_bc"]), "close_line": close,
                                            "clv_pts": cl(dec["pick_bc"])}) + "\n")
    STATE.write_text(json.dumps(st, indent=1))
    msg = f"tick {t:%F %H:%M}Z games {len(st)} pins {len(pins)} kalshi_events {len(kal)} new_picks {new}"
    if errs:
        msg += " ERR " + ",".join(errs)
    if new or errs:
        print(msg)
    return 0


# ---------------------------------------------------------------- report

def _rows() -> pd.DataFrame:
    if not PICKS.exists():
        return pd.DataFrame()
    ev = [json.loads(l) for l in PICKS.read_text().splitlines() if l.strip()]
    p = pd.DataFrame([e for e in ev if e["type"] == "pick"]).drop_duplicates("key")
    s = pd.DataFrame([e for e in ev if e["type"] == "settle"])
    if p.empty:
        return p
    return p.merge(s.drop(columns="type"), on="key", how="left") if not s.empty else p


def cmd_report() -> int:
    st = load_state()
    dec = [g["decision"] for g in st.values() if "decision" in g]
    print(f"games tracked {len(st)}; with open {sum('pin_open' in g for g in st.values())}; "
          f"decided {len(dec)}; picks {sum(bool(d.get('pick')) for d in dec)}")
    gaps = [d["gap"] for d in dec if d.get("gap") is not None]
    if gaps:
        print(f"model - open: mean {np.mean(gaps):+.2f} sd {np.std(gaps):.2f} n {len(gaps)}  "
              f"(a mean far from 0 = level problem, not edge)")
    r = _rows()
    if r.empty:
        print("no picks yet")
        return 0
    s = r[r.get("result").notna()] if "result" in r else r.iloc[0:0]
    if not s.empty:
        w, l, pu = (s.result == "win").sum(), (s.result == "loss").sum(), (s.result == "push").sum()
        nv = np.where(s.result == "win", 1 / s.nv_prob_open - 1, np.where(s.result == "loss", -1.0, 0.0))
        m110 = np.where(s.result == "win", 100 / 110, np.where(s.result == "loss", -1.0, 0.0))
        print(f"settled {len(s)}: {w}-{l}-{pu}  ROI no-vig {nv.mean():+.1%}  ROI -110 {m110.mean():+.1%}  "
              f"CLV mean {s.clv_pts.mean():+.2f} pts (sd {s.clv_pts.std():.2f})")
        if "kal_ask" in s:
            k = s[s.kal_ask.notna()]
            if len(k):
                kr = np.where(k.result == "win", (1 - k.kal_ask - k.kal_fee) / (k.kal_ask + k.kal_fee),
                              np.where(k.result == "loss", -1.0, 0.0))
                print(f"  Kalshi taker-priced subset {len(k)}: ROI {kr.mean():+.1%}")
    cols = [c for c in ["key", "pick", "open_line", "model", "gap", "close_line", "final_total", "result", "clv_pts"]
            if c in r]
    print(r[cols].tail(25).to_string(index=False))
    return 0


def cmd_show() -> int:
    st = load_state()
    rows = []
    for k, g in sorted(st.items()):
        d = g.get("decision") or {}
        rows.append({"game": k, "pin_open": (g.get("pin_open") or {}).get("line"),
                     "pin_last": (g.get("pin_last") or {}).get("line"),
                     "kal_open": (g.get("kal_open") or {}).get("implied"),
                     "kal_last": (g.get("kal_last") or {}).get("implied"),
                     "model": d.get("model"), "pick": d.get("pick")})
    print(pd.DataFrame(rows).to_string(index=False) if rows else "no games")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"tick": cmd_tick, "report": cmd_report, "show": cmd_show}.get(cmd, lambda: print(__doc__) or 1)())
