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
  ADDENDUM Q004 (2026-10-28 15:30Z, written and deployed BEFORE any of these fields were collected; the rules above
  are unchanged -- this only ADDS fields and a separate paper pick type; helpers in scripts/kalshi_tight.py):
    (a) LISTING  = per KXNBATOTAL event: Kalshi open_time (min over the ladder), our first sighting, and the FIRST
                   TWO-SIDED anchor quote (anchor = two-sided strike whose YES mid is nearest 50c).
    (b) TIGHT    = first tick at which the anchor's YES spread <= 4c: time, strike, bid/ask, and displayed depth at
                   the best price from the orderbook endpoint (contracts and $ for YES ask, NO ask, YES bid).
                   Censored (no pick_tight, excluded from timing stats) when seen tight/two-sided at our FIRST sighting
                   of an event whose open_time is > 25 min older (we missed the moment). 10-min polling -> times are
                   upper bounds to within ~10 min.
    (c) DEPTH    = at the main rule's decision tick, for a pick with a matching Kalshi strike: contracts and $ on our
                   side at the best ask and within 1c of it (kal_depth_* fields on the pick row).
    (d) pick_tight = separate paper pick (type pick_tight / settle_tight in the picks file; state in
                   data/open_totals_tight.json). Decided at the first tick >= the event's uncensored first tight quote
                   at which (i) the game is in our Pinnacle state, (ii) a model total exists, (iii) tip is >= 30 min
                   away, (iv) the anchor is CURRENTLY tight. Line = the anchor strike S at that tick; SAME threshold:
                   |model - S| >= 3.0 -> over (YES) if model > S else under (NO). Price = that contract's ask at that
                   tick + exact taker fee (100-contract order, rounded up to the cent). One per game. Settled on the
                   final total vs S (no pushes; half-point strikes). CLV = last pre-tip Pinnacle main total - S, signed
                   in the bet's direction (points). Main-rule picks are unaffected.
    JUDGE (Q004) = after >= 100 settled pick_tight (pooled with the NHL/NFL logger's pick_tight in QUESTIONS.md Q004,
                   also read per sport): mean CLV > 0 AND ROI at ask+fee > 0 -> write a sealed ask; otherwise no ask.
Files: data/open_totals_state.json (per-game state), data/open_totals_picks.jsonl (picks + settlements),
       data/open_totals_tight.json (Q004 listing/tight records + pick_tight decisions).
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
import kalshi_tight as KT

STATE = ROOT / "data" / "open_totals_state.json"
PICKS = ROOT / "data" / "open_totals_picks.jsonl"
TIGHT = ROOT / "data" / "open_totals_tight.json"
TIGHT_SLACK = pd.Timedelta(minutes=25)
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
            e = out.setdefault(ev, {"key": key(date, h, a), "strikes": {}, "rows": []})
            e["rows"].append({"ticker": m["ticker"], "strike": None if m.get("floor_strike") is None
                              else float(m["floor_strike"]), "yes_bid": f("yes_bid_dollars"),
                              "yes_ask": f("yes_ask_dollars"), "no_ask": f("no_ask_dollars"),
                              "open_time": m.get("open_time")})
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
        g["_rows"] = e["rows"]
    T = json.loads(TIGHT.read_text()) if TIGHT.exists() else {"markets": {}, "picks": {}}
    for ev, e in kal.items():  # Q004 (a)/(b): listing + first tight, independent of Pinnacle
        T["markets"][ev] = {**KT.track(T["markets"].get(ev), e["rows"], t, TIGHT_SLACK, _get, KALSHI), "key": e["key"]}
    tight_due = {r["key"]: ev for ev, r in T["markets"].items() if r.get("first_tight")
                 and not r["first_tight"]["censored"] and r["key"] not in T["picks"] and r["key"] in st}

    # decisions
    need = sorted({k.split("|")[0] for k, g in st.items() if "pin_open" in g and "decision" not in g
                   and pd.Timestamp(g["start"]) - t >= MIN_LEAD}
                  | {k.split("|")[0] for k in tight_due if pd.Timestamp(st[k]["start"]) - t >= MIN_LEAD})
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
                tk = [r["ticker"] for r in g.get("_rows", []) if r["strike"] == o["line"]]
                if tk:  # Q004 (c): our side's depth at the decision tick
                    dd = KT.buy_depth(KT.book(_get, KALSHI, tk[0]), side == "over")
                    dec.update(kal_ticker=tk[0], **{f"kal_depth_{x}": v for x, v in dd.items()})
            with PICKS.open("a") as f:
                f.write(json.dumps({"type": "pick", "key": k, "start": g["start"], **dec}) + "\n")
            new += 1
        g["decision"] = dec
    # Q004 (d): pick_tight
    for k, ev in tight_due.items():
        g = st[k]
        if pd.Timestamp(g["start"]) - t < MIN_LEAD:
            T["picks"][k] = {"t": t.isoformat(), "event": ev, "pick": None, "why": "not decidable >= 30 min before tip"}
            continue
        e = kal.get(ev)
        a = KT.anchor(e["rows"]) if e else None
        m = models.get(k)
        if a is None or a["strike"] is None or a["yes_ask"] - a["yes_bid"] > KT.TIGHT_MAX + 1e-9 or m is None:
            continue
        gap = m["total"] - a["strike"]
        dec = {"t": t.isoformat(), "event": ev, "ticker": a["ticker"], "strike": a["strike"],
               "model": round(m["total"], 2), "gap": round(gap, 2), "yes_bid": a["yes_bid"], "yes_ask": a["yes_ask"],
               "no_ask": a["no_ask"], "first_tight_t": T["markets"][ev]["first_tight"]["t"], "pick": None}
        if abs(gap) >= GAP:
            side = "over" if gap > 0 else "under"
            ask = a["yes_ask"] if side == "over" else a["no_ask"]
            if np.isfinite(ask):
                dd = KT.buy_depth(KT.book(_get, KALSHI, a["ticker"]), side == "over")
                dec.update(pick=side, ask=ask, fee=KT.fee_exact(ask), **{f"depth_{x}": v for x, v in dd.items()})
                with PICKS.open("a") as f:
                    f.write(json.dumps({"type": "pick_tight", "key": k, "start": g["start"], **dec}) + "\n")
                new += 1
        T["picks"][k] = dec
    for g in st.values():
        g.pop("_ladder", None)
        g.pop("_rows", None)

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
    # settle pick_tight
    tp = [k for k, p in T["picks"].items() if p.get("pick") and "result" not in p
          and t - pd.Timestamp(st[k]["start"]) > pd.Timedelta(hours=3)]
    for d in sorted({k.split("|")[0] for k in tp}):
        try:
            fin = finals(d)
        except Exception as e:  # noqa: BLE001
            errs.append(f"espn {type(e).__name__}")
            continue
        for k in tp:
            if k in fin:
                p, g = T["picks"][k], st[k]
                res = "win" if (fin[k] > p["strike"]) == (p["pick"] == "over") else "loss"
                close = (g.get("pin_last") or {}).get("line")
                clv = None if close is None else (close - p["strike"]) * (1 if p["pick"] == "over" else -1)
                p.update(result=res, final_total=fin[k], roi=KT.roi(res, p["ask"], p["fee"]), close_line=close, clv=clv)
                with PICKS.open("a") as f:
                    f.write(json.dumps({"type": "settle_tight", "key": k, "final_total": fin[k], "result": res,
                                        "roi": p["roi"], "close_line": close, "clv_pts": clv}) + "\n")
    TIGHT.write_text(json.dumps(T, indent=1, default=str))
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
    T = json.loads(TIGHT.read_text()) if TIGHT.exists() else {"markets": {}, "picks": {}}
    dd = [p.get("depth_usd_1c") for p in T["picks"].values() if p.get("pick")]
    if not r.empty and "kal_depth_usd_1c" in r:
        dd += r.kal_depth_usd_1c.dropna().tolist()
    KT.report(T, [p for p in T["picks"].values() if "result" in p], dd, "NBA totals")
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
