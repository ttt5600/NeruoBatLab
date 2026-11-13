"""Forward paper loggers for the Kalshi-execution trials (READ-ONLY: public market-data GETs, no keys, no orders).

    python scripts/kx_forward.py k1      # L32b-K1  (edgelab l32b_k1_kalshi_taker-8d6fb950; notes/nba_l32b_exec.md)
    python scripts/kx_forward.py knfl    # K-NFL    (edgelab knfl_l33_kalshi_taker-445b123e; notes/kalshi_openers_xleague.md)
    python scripts/kx_forward.py kcfb    # K-CFB    (edgelab kcfb_l36_kalshi_taker-613b7d62) -- reads the forward outputs of
                                         #          results/kalshi_openers_xleague/{q1_fetch,q1_timing,q2_cfb}.py (KX_TAG=fwd)
    python scripts/kx_forward.py k50nfl  # K50-NFL  (edgelab k50_nfl_k50_kalshi-d7b7da0f; notes/k50_strike.md "Forward rule")
    python scripts/kx_forward.py status  # counts per file

Output (append-only, one JSON object per line; never rewritten): ~/.claude/quant-sports/data/kx_forward/<trial>.jsonl
(override KX_FWD_DIR). Idempotent: a decision gets ONE `pick` row (status bet / skip / no quote ...), a bet gets ONE
`settle` row once the game is final. Re-running only adds what is missing.

The trading rules are copied from the pre-registered dev scripts, not re-decided here:
  K1   = results/nba_l32b_exec/q2_score.py one(): L32b live pick (t >= 2026-11-12), S = strike nearest the Pinnacle open
         among two-sided strikes (tie: lower for over, higher for under), quote = last 1-min candle ending <= the L32b
         tick (<= 60 min old), bet iff model - c - S clears +/-3.0, YES ask (over) / 1 - YES bid (under) + exact taker
         fee (100 ct), graded on the ESPN final vs S, CLV = Pinnacle last line - S (signed).
  KNFL = results/kalshi_openers_xleague/q2_nfl.py: open_lines_logger L33 pick (t >= 2026-11-12), strikes within 3.01 of
         the Pinnacle open on the home-margin / total axis, last 1-min candle ending <= tick, favourable tie, gap at S
         >= 2.0 (spread) / 2.5 (total), ask + exact fee, graded on final vs S, CLV = Pinnacle close vs S signed.
  KCFB = q2_cfb.py raw variant (the primary), forward tag: copied row by row from q2_cfb_rows_fwd.csv.
Candles are historical and immutable, so pricing after the fact uses only pre-decision quotes (price_time <= decided_at).
Extra (descriptive, for edgelab's price-CLV): close_mid = same-side mid of the last candle ending <= event start.
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kx_throttle import kget  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SD = Path.home() / ".claude" / "quant-sports" / "data"
KD = Path.home() / ".claude" / "quant-kalshi" / "data"
OUTD = Path(os.environ.get("KX_FWD_DIR", str(SD / "kx_forward")))
XL = ROOT / "results" / "kalshi_openers_xleague"
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
FWD_FROM = pd.Timestamp(os.environ.get("KX_FWD_FROM", "2026-11-12"), tz="UTC")   # override ONLY for code checks on dev picks
NOW = lambda: pd.Timestamp.now(tz="UTC")  # noqa: E731


# ------------------------------------------------------------------ helpers
def fee(p, n=100):
    return math.ceil(round(n * 0.07 * p * (1 - p) * 100, 6)) / 100 / n


def ts(x) -> int:
    return int((pd.Timestamp(x) - EPOCH) // pd.Timedelta(seconds=1))


def iso(t: int) -> str:
    return (EPOCH + pd.Timedelta(seconds=int(t))).isoformat()


def px(c, k, f="close"):
    d = c.get(k) or {}
    v = d.get(f + "_dollars", d.get(f))
    if v is None:
        return np.nan
    v = float(v)
    return v / 100 if v > 1 else v


def read(trial):
    p = OUTD / f"{trial}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def append(trial, rows):
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / f"{trial}.jsonl").touch()            # the file exists once the job has run, even with nothing to log
    if not rows:
        return
    with open(OUTD / f"{trial}.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def clean(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, float)):
            v = None if not np.isfinite(v) else float(v)
        elif isinstance(v, np.integer):
            v = int(v)
        elif isinstance(v, np.bool_):
            v = bool(v)
        out[k] = v
    return out


def candles_1m(series, ticker, t0, t1):
    j = kget(f"/series/{series}/markets/{ticker}/candlesticks", {"start_ts": int(t0), "end_ts": int(t1), "period_interval": 1})
    return (j or {}).get("candlesticks", [])


def close_mid(series, ticker, start, side):
    """Same-side mid of the last 1-min candle ending <= event start with a two-sided quote (NaN if none)."""
    t = ts(start)
    cs = [c for c in candles_1m(series, ticker, t - 3600, t) if c["end_period_ts"] <= t]
    for c in reversed(cs):
        b, a = px(c, "yes_bid"), px(c, "yes_ask")
        if np.isfinite(b) and np.isfinite(a) and b > 0 and a < 1:
            m = (a + b) / 2
            return (m if side == "yes" else 1 - m), iso(c["end_period_ts"])
    return np.nan, None


def market_result(ticker):
    m = (kget(f"/markets/{ticker}") or {}).get("market", {})
    return m.get("result") or None, m.get("status")


# ------------------------------------------------------------------ K1 (NBA totals)
def k1():
    S = json.load(open(SD / "l32b_state.json"))
    L = json.load(open(KD / "open_totals_state.json"))
    have = read("k1")
    picked = {r["key"]: r for r in have if r["kind"] == "pick"}
    settled = {r["key"] for r in have if r["kind"] == "settle"}
    now = NOW()
    new = []
    for k, r in S.items():
        if not r.get("pick") or r.get("kind") != "live" or pd.Timestamp(r["t"]) < FWD_FROM or k in picked:
            continue
        T = pd.Timestamp(r["t"])
        if now < T + pd.Timedelta(minutes=3):
            continue                                    # the candle ending at the tick must be closed
        g = L.get(k) or {}
        base = dict(kind="pick", trial="k1", key=k, logged_at=now.isoformat(), decided_at=T.isoformat(),
                    event_start=r["start"], pick=r["pick"], open_line=r["open_line"], model=r["model"], c=r["c"])
        ev = (g.get("kal_open") or {}).get("event")
        if not ev:
            if now > pd.Timestamp(r["start"]):
                new.append({**base, "status": "no kalshi event"})
            continue
        ms = (kget("/markets", {"event_ticker": ev, "limit": 200}) or {}).get("markets", [])
        t = ts(T)
        q = {}
        for m in ms:
            fs = m.get("floor_strike")
            if fs is None or abs(float(fs) - g["pin_open"]["line"]) > 5.0:
                continue
            cs = [c for c in candles_1m("KXNBATOTAL", m["ticker"], t - 3600, t)
                  if t - 3600 <= c["end_period_ts"] <= t and np.isfinite(px(c, "yes_bid")) and np.isfinite(px(c, "yes_ask"))]
            if cs:
                c = cs[-1]
                q[float(fs)] = dict(ticker=m["ticker"], bid=px(c, "yes_bid"), ask=px(c, "yes_ask"), end=c["end_period_ts"])
        if not q:
            new.append({**base, "event": ev, "status": "no quote"}); continue
        q = {s: z for s, z in q.items() if z["bid"] > 0 and z["ask"] < 1}
        if not q:
            new.append({**base, "event": ev, "status": "no two-sided"}); continue
        ks = np.array(sorted(q))
        dist = np.abs(ks - r["open_line"])
        cand = ks[dist == dist.min()]
        Sx = float(cand.min() if r["pick"] == "over" else cand.max())
        adj = r["model"] - r["c"] - Sx
        ok = adj >= 3 if r["pick"] == "over" else adj <= -3
        z = q[Sx]
        side = "yes" if r["pick"] == "over" else "no"
        ask = z["ask"] if side == "yes" else 1 - z["bid"]
        new.append(clean({**base, "event": ev, "status": "bet" if ok else "gap<3 at S", "S": Sx, "ticker": z["ticker"],
                          "side": side, "adj_S": round(adj, 2), "yes_bid": z["bid"], "yes_ask": z["ask"], "ask": ask,
                          "fee": fee(ask), "quote_end": iso(z["end"]), "quote_age_min": (t - z["end"]) / 60,
                          "n_strikes_quoted": len(q)}))
    append("k1", new)
    picked.update({r["key"]: r for r in new})
    st = []
    for k, p in picked.items():
        if p.get("status") != "bet" or k in settled:
            continue
        r = S.get(k) or {}
        fin = r.get("final_total")
        if r.get("result") not in ("win", "loss", "push") or fin is None:
            continue
        Sx = p["S"]
        res = "void" if fin == Sx else ("yes" if fin > Sx else "no")
        close = ((L.get(k) or {}).get("pin_last") or {}).get("line")
        sgn = 1 if p["pick"] == "over" else -1
        cm, cm_t = close_mid("KXNBATOTAL", p["ticker"], p["event_start"], p["side"])
        kr, kst = market_result(p["ticker"])
        st.append(clean(dict(kind="settle", trial="k1", key=k, logged_at=NOW().isoformat(), ticker=p["ticker"],
                             side=p["side"], S=Sx, final=fin, result=res, won=(res == p["side"]) if res != "void" else None,
                             close_line=close, clv_S=None if close is None else (close - Sx) * sgn, close_mid=cm,
                             close_mid_t=cm_t, kalshi_result=kr, kalshi_status=kst)))
    append("k1", st)
    print(f"k1: +{len(new)} pick rows ({sum(r.get('status') == 'bet' for r in new)} bets), +{len(st)} settle rows; "
          f"file {OUTD / 'k1.jsonl'}")


# ------------------------------------------------------------------ K-NFL
NFL_SER = {"total": "KXNFLTOTAL", "spread": "KXNFLSPREAD"}
NFL_TH = {"total": 2.5, "spread": 2.0}


def side_x(event_ticker, ticker, strike, mk):     # = results/kalshi_openers_xleague/kx.py side_x
    if mk == "total":
        return float(strike), 1
    letters = event_ticker.split("-")[-1][7:]
    team = ticker.split("-")[-1].rstrip("0123456789")
    if letters.endswith(team):
        return float(strike), 1
    return -float(strike), -1


def knfl():
    rows = [json.loads(l) for l in open(KD / "open_lines_nfl_picks.jsonl") if l.strip()]
    picks = [r for r in rows if r.get("type") == "pick" and pd.Timestamp(r["t"]) >= FWD_FROM]
    settle = {(r["key"], r["market"]): r for r in rows if r.get("type") == "settle"}
    tight = json.load(open(KD / "open_lines_nfl_tight.json"))["markets"]
    have = read("knfl")
    bid = lambda r: f"{r['key']}|{r['market'] if 'market' in r else r['mk']}"  # noqa: E731
    picked = {r["bet_key"]: r for r in have if r["kind"] == "pick"}
    settled = {r["bet_key"] for r in have if r["kind"] == "settle"}
    now = NOW()
    new = []
    for p in picks:
        bk = bid(p)
        if bk in picked:
            continue
        T = pd.Timestamp(p["t"])
        if now < T + pd.Timedelta(minutes=3):
            continue
        mk, k = p["market"], p["key"]
        d, teams = k.split("|")
        a, h = teams.split("@")
        ev = next((e for e, m in tight.items() if m["mk"] == mk and m["date"] == d and set(m["teams"]) == {a, h}), None)
        base = dict(kind="pick", trial="knfl", bet_key=bk, key=k, mk=mk, logged_at=now.isoformat(),
                    decided_at=T.isoformat(), event_start=p["start"], pick=p["pick"], open_line=p["open_line"],
                    model=p["model"], logger_kal_ticker=p.get("kal_ticker"), logger_kal_side=p.get("kal_side"),
                    logger_kal_ask=p.get("kal_ask"))
        if not ev:
            if now > T + pd.Timedelta(hours=6):
                new.append({**base, "status": "no event"})
            continue
        t = ts(T)
        x_open = p["open_line"] if mk == "total" else -p["open_line"]
        q = []
        for m in (kget("/markets", {"event_ticker": ev, "limit": 200}) or {}).get("markets", []):
            if m.get("floor_strike") is None:
                continue
            x, sg = side_x(ev, m["ticker"], m["floor_strike"], mk)
            if abs(x - x_open) > 3.01:
                continue
            cs = [c for c in candles_1m(NFL_SER[mk], m["ticker"], t - 3600, t) if c["end_period_ts"] <= t]
            if not cs:
                continue
            c = cs[-1]
            b, aa = px(c, "yes_bid"), px(c, "yes_ask")
            if not (np.isfinite(b) and np.isfinite(aa)):
                continue
            q.append(dict(ticker=m["ticker"], x=x, sg=sg, bid=b, ask=aa, end=c["end_period_ts"]))
        q = [z for z in q if z["bid"] > 0 and z["ask"] < 1]
        if not q:
            new.append({**base, "event": ev, "status": "no quote"}); continue
        dmin = min(abs(z["x"] - x_open) for z in q)
        cand = [z for z in q if abs(abs(z["x"] - x_open) - dmin) < 1e-9]
        up = p["pick"] in ("over", "home")
        z = min(cand, key=lambda z: z["x"]) if up else max(cand, key=lambda z: z["x"])
        gap = p["model"] - z["x"]
        ok = gap >= NFL_TH[mk] if up else gap <= -NFL_TH[mk]
        buy_yes = (up and z["sg"] > 0) or ((not up) and z["sg"] < 0)
        ask = z["ask"] if buy_yes else 1 - z["bid"]
        new.append(clean({**base, "event": ev, "status": "bet" if ok else "skip gap", "S": z["x"], "sg": z["sg"],
                          "gap_S": gap, "ticker": z["ticker"], "side": "yes" if buy_yes else "no", "up": up,
                          "yes_bid": z["bid"], "yes_ask": z["ask"], "ask": ask, "fee": fee(ask),
                          "quote_end": iso(z["end"]), "quote_age_min": (t - z["end"]) / 60,
                          "spread_c": round((z["ask"] - z["bid"]) * 100)}))
    append("knfl", new)
    picked.update({r["bet_key"]: r for r in new})
    st = []
    for bk, p in picked.items():
        if p.get("status") != "bet" or bk in settled:
            continue
        s = settle.get((p["key"], p["mk"]))
        if not s or s.get("home_score") is None:
            continue
        hs, as_ = s["home_score"], s["away_score"]
        v = hs + as_ if p["mk"] == "total" else hs - as_
        x = p["S"]
        res = "void" if v == x else ("yes" if ((v > x) if p["sg"] > 0 else (v < x)) else "no")
        xc = s.get("close_line")
        xc = None if xc is None else (xc if p["mk"] == "total" else -xc)
        cm, cm_t = close_mid(NFL_SER[p["mk"]], p["ticker"], p["event_start"], p["side"])
        kr, kst = market_result(p["ticker"])
        st.append(clean(dict(kind="settle", trial="knfl", bet_key=bk, key=p["key"], mk=p["mk"],
                             logged_at=NOW().isoformat(), ticker=p["ticker"], side=p["side"], S=x, final=v, result=res,
                             won=(res == p["side"]) if res != "void" else None,
                             clv=None if xc is None else ((xc - x) if p["up"] else (x - xc)), close_mid=cm,
                             close_mid_t=cm_t, kalshi_result=kr, kalshi_status=kst)))
    append("knfl", st)
    print(f"knfl: forward picks in log {len(picks)}; +{len(new)} pick rows ({sum(r.get('status') == 'bet' for r in new)} "
          f"bets), +{len(st)} settle rows; file {OUTD / 'knfl.jsonl'}")


# ------------------------------------------------------------------ K50-NFL (lead 11-13; notes/k50_strike.md "Forward rule")
# Copied from results/k50_strike/score.py nfl() + ladder()/anchor()/decide() and q1_timing.py's first-tight loop:
#   T = later of the L33 model-cache time and Kalshi's first tight HOUR (anchor <= 4c, finite implied; hourly candles,
#   each strike's last quote carried <= 6 h), rounded UP to the hour; skip if kickoff - T < 30 min. Ladder at T from
#   hourly candles (<= 6 h old, two-sided). S50 = anchor (mid nearest 50c, then tighter spread, then nearest m). g = m - S50
#   (total model C / spread model B home margin); bet if |g| >= 2.5 total / 2.0 spread; buy axis > S50 (g > 0) or < S50
#   at the ask + exact taker fee. Grade: final vs S50; CLV = DK close (ESPN core odds, as q1_fetch book()) - S50 signed.
# Forward deviations (stated, not tuned): (1) candles are fetched for EVERY strike of the event (dev fetched strikes within
#   +-4 / +-5 of the DK open, unknown pre-game); the anchor is the ~50c rung either way. (2) a game is decided once, with
#   the earliest model cache that contains it. (3) forward = kickoff after registration (2026-11-13T08:08Z).
K50_FROM = pd.Timestamp(os.environ.get("KX_K50_FROM", "2026-11-13T08:08:11Z")).tz_localize(None).tz_localize("UTC")
K50_K = {"total": 2.5, "spread": 2.0}
ESPN_CODE = {"WAS": "WSH", "LA": "LAR"}


def hourly_candles(ev, t0, t1):
    ms = (kget("/markets", {"event_ticker": ev, "limit": 200}) or {}).get("markets", [])
    strike = {m["ticker"]: float(m["floor_strike"]) for m in ms if m.get("floor_strike") is not None}
    opent = min((ts(m["open_time"]) for m in ms if m.get("open_time")), default=t0)
    tk, rows = list(strike), []
    for i in range(0, len(tk), 20):
        j = kget("/markets/candlesticks", {"market_tickers": ",".join(tk[i:i + 20]), "start_ts": max(t0, opent - 3600),
                                           "end_ts": t1, "period_interval": 60}) or {}
        for mm in j.get("markets", []):
            for c in mm.get("candlesticks", []):
                rows.append({"event": ev, "ticker": mm["market_ticker"], "strike": strike[mm["market_ticker"]],
                             "end": c["end_period_ts"], "bid": px(c, "yes_bid"), "ask": px(c, "yes_ask")})
    return pd.DataFrame(rows, columns=["event", "ticker", "strike", "end", "bid", "ask"])


def k50_first_tight(C, mk, kick):
    """q1_timing.py run() first-tight loop on one event's hourly candles -> pd.Timestamp or None."""
    sys.path.insert(0, str(XL))
    from kx import implied
    latest = {}
    for t, cd in C.sort_values("end").groupby("end"):
        T = EPOCH + pd.Timedelta(seconds=int(t))
        if T > kick:
            break
        for rr in cd.itertuples():
            if pd.notna(rr.bid) and pd.notna(rr.ask):
                latest[rr.ticker] = rr
        cur = [x for x in latest.values() if t - x.end <= 6 * 3600 and x.bid > 0 and x.ask < 1]
        if not cur:
            continue
        pts = []
        for r in cur:
            x, sg = side_x(r.event, r.ticker, r.strike, mk)
            mid = (r.bid + r.ask) / 2
            pts.append((x, mid if sg > 0 else 1 - mid))
        a = sorted(cur, key=lambda r: abs((r.bid + r.ask) / 2 - .5))[0]
        if a.ask - a.bid <= 0.04 and np.isfinite(implied(pts)):
            return T
    return None


def k50_ladder(C, T, mk):     # = results/k50_strike/score.py ladder(..., max_age 6 h)
    t = ts(T)
    c = C[(C.end <= t) & (C.end >= t - 6 * 3600) & C.bid.notna() & C.ask.notna()]
    c = c.sort_values("end").groupby("ticker").tail(1)
    c = c[(c.bid > 0) & (c.ask < 1)]
    out = []
    for r in c.itertuples():
        x, sg = side_x(r.event, r.ticker, r.strike, mk)
        mid = (r.bid + r.ask) / 2
        out.append(dict(ticker=r.ticker, x=x, sg=sg, p_over=mid if sg > 0 else 1 - mid, spread=r.ask - r.bid,
                        up=r.ask if sg > 0 else 1 - r.bid, dn=1 - r.bid if sg > 0 else r.ask, bid=r.bid, ask=r.ask,
                        end=int(r.end), age_min=(t - r.end) / 60))
    return out


def espn_final_close(key, home, away, kick):
    """ESPN final (home - away scores) + DK close (q1_fetch book() provider priority) for one NFL game, or None."""
    sys.path.insert(0, str(ROOT))
    from sports import espn
    from sports.http import fetch
    d = pd.Timestamp(kick).tz_convert("America/New_York")
    for day in (d, d - pd.Timedelta(days=1)):
        evs = fetch(espn.SITE.format(path=espn.LEAGUES["nfl"]) + "/scoreboard", {"dates": day.strftime("%Y%m%d")},
                    cache=False).get("events", [])
        for e in evs:
            g = espn.parse_event("nfl", e)
            if g["home"] == ESPN_CODE.get(home, home) and g["away"] == ESPN_CODE.get(away, away):
                if g["status"] != "STATUS_FINAL":
                    return None
                L = sorted(espn.lines("nfl", str(g["event_id"]), cache=False) or [],
                           key=lambda r: (0 if "Draft" in r["provider"] else 1 if "ESPN" in r["provider"] else 2, r["provider"]))
                tc = next((r for r in L if pd.notna(r.get("total_open")) and pd.notna(r.get("total_close"))), None)
                sc = next((r for r in L if pd.notna(r.get("home_spread_open")) and pd.notna(r.get("home_spread_close"))), None)
                return dict(event_id=str(g["event_id"]), home_score=g["home_score"], away_score=g["away_score"],
                            total_close=tc["total_close"] if tc else None, book_tot=tc["provider"] if tc else None,
                            home_spread_close=sc["home_spread_close"] if sc else None, book_spr=sc["provider"] if sc else None)
    return None


def k50nfl():
    st = json.load(open(KD / "open_lines_nfl_state.json"))["_model_cache"]
    tight = json.load(open(KD / "open_lines_nfl_tight.json"))["markets"]
    games = {}
    for w, v in sorted(st.items(), key=lambda kv: kv[1]["t"]):
        for r in v["rows"]:
            games.setdefault(r["game_id"], dict(cache_t=v["t"], **r))
    have = read("k50nfl")
    picked = {r["bet_key"]: r for r in have if r["kind"] == "pick"}
    settled = {r["bet_key"] for r in have if r["kind"] == "settle"}
    now = NOW()
    new = []
    for gid, c in games.items():
        kick = pd.Timestamp(c["kickoff_utc"])
        if kick < K50_FROM:
            continue
        date_et = kick.tz_convert("America/New_York").strftime("%Y-%m-%d")
        for mk in ("total", "spread"):
            bk = f"{gid}|{mk}"
            if bk in picked:
                continue
            base = dict(kind="pick", trial="k50nfl", bet_key=bk, game_id=gid, mk=mk, logged_at=now.isoformat(),
                        event_start=kick.isoformat(), cache_t=c["cache_t"],
                        m=c["model_total"] if mk == "total" else c["model_spread_home"], home=c["home"], away=c["away"])
            ev = next((e for e, m in tight.items() if m["mk"] == mk and m["date"] == date_et
                       and set(m["teams"]) == {c["home"], c["away"]}), None)
            if not ev:
                if now > kick:
                    new.append({**base, "status": "no kalshi event"})
                continue
            if now < pd.Timestamp(c["cache_t"]) + pd.Timedelta(minutes=5):
                continue
            C = hourly_candles(ev, ts(now) - 21 * 86400, min(ts(now), ts(kick)))
            ft = k50_first_tight(C, mk, kick) if len(C) else None
            if ft is None:
                if now > kick:
                    new.append({**base, "event": ev, "status": "never tight"})
                continue
            T = max(pd.Timestamp(c["cache_t"]), ft)
            T = pd.Timestamp(math.ceil(ts(T) / 3600) * 3600, unit="s", tz="UTC")
            if kick - T < pd.Timedelta(minutes=30):
                new.append({**base, "event": ev, "first_tight": ft.isoformat(), "status": "T < 30 min"}); continue
            if now < T + pd.Timedelta(minutes=5):
                continue
            L = k50_ladder(C, T, mk)
            if not L:
                new.append({**base, "event": ev, "decided_at": T.isoformat(), "status": "no quote"}); continue
            m = base["m"]
            a = min(L, key=lambda z: (round(abs(z["p_over"] - .5), 6), round(z["spread"], 6), abs(z["x"] - m)))
            g = m - a["x"]
            up = g > 0
            ok = abs(g) >= K50_K[mk]
            ask = a["up"] if up else a["dn"]
            buy_yes = (up and a["sg"] > 0) or ((not up) and a["sg"] < 0)
            new.append(clean({**base, "event": ev, "first_tight": ft.isoformat(), "decided_at": T.isoformat(),
                              "status": "bet" if ok else "skip gap", "S": a["x"], "sg": a["sg"], "gap": g,
                              "ticker": a["ticker"], "side": "yes" if buy_yes else "no", "up": up, "pick": "up" if up else "down",
                              "S50_mid": a["p_over"], "yes_bid": a["bid"], "yes_ask": a["ask"], "ask": ask, "fee": fee(ask),
                              "quote_end": iso(a["end"]), "quote_age_min": a["age_min"],
                              "spread_c": round(a["spread"] * 100), "n_rungs": len(L)}))
    append("k50nfl", new)
    picked.update({r["bet_key"]: r for r in new})
    out = []
    for bk, p in picked.items():
        if p.get("status") != "bet" or bk in settled or now < pd.Timestamp(p["event_start"]) + pd.Timedelta(hours=4):
            continue
        fc = espn_final_close(p["game_id"], p["home"], p["away"], p["event_start"])
        if not fc or fc["home_score"] is None or not np.isfinite(fc["home_score"]):
            continue
        hs, as_ = fc["home_score"], fc["away_score"]
        v = hs + as_ if p["mk"] == "total" else hs - as_
        x = p["S"]
        res = "void" if v == x else ("yes" if ((v > x) if p["sg"] > 0 else (v < x)) else "no")
        cl = fc["total_close"] if p["mk"] == "total" else (None if fc["home_spread_close"] is None else -fc["home_spread_close"])
        ser = "KXNFLTOTAL" if p["mk"] == "total" else "KXNFLSPREAD"
        cm, cm_t = close_mid(ser, p["ticker"], p["event_start"], p["side"])
        kr, kst = market_result(p["ticker"])
        out.append(clean(dict(kind="settle", trial="k50nfl", bet_key=bk, game_id=p["game_id"], mk=p["mk"],
                              logged_at=NOW().isoformat(), ticker=p["ticker"], side=p["side"], S=x, final=v, result=res,
                              won=(res == p["side"]) if res != "void" else None, close_line=cl,
                              close_book=fc["book_tot"] if p["mk"] == "total" else fc["book_spr"],
                              clv=None if cl is None else ((cl - x) if p["up"] else (x - cl)), close_mid=cm,
                              close_mid_t=cm_t, kalshi_result=kr, kalshi_status=kst, espn_event_id=fc["event_id"])))
    append("k50nfl", out)
    print(f"k50nfl: games {len(games)} (kickoff >= {K50_FROM}: {sum(pd.Timestamp(g['kickoff_utc']) >= K50_FROM for g in games.values())}); "
          f"+{len(new)} pick rows ({sum(r.get('status') == 'bet' for r in new)} bets), +{len(out)} settle rows; "
          f"file {OUTD / 'k50nfl.jsonl'}")


# ------------------------------------------------------------------ K-CFB
def kcfb():
    f = XL / "q2_cfb_rows_fwd.csv"
    if not f.exists():
        print(f"kcfb: {f} missing (run the forward pipeline first)"); return
    R = pd.read_csv(f)
    if R.empty:
        print("kcfb: no forward rows yet"); return
    Q1 = pd.read_csv(XL / "q1_cfbfwd_rows.csv").set_index("event")
    K = pd.read_parquet(XL / "cfbfwd_kmarkets.parquet")
    C = pd.read_parquet(XL / "cfbfwd_candles.parquet")
    have = read("kcfb")
    picked = {r["event"] for r in have if r["kind"] == "pick"}
    now = NOW().isoformat()
    new = []
    R = R[R["variant"].isna() | (R["variant"] == "raw")] if "variant" in R else R
    for r in R.to_dict("records"):
        ev = r["event"]
        if ev in picked:
            continue
        picked.add(ev)
        start = pd.Timestamp(Q1.loc[ev, "occ"]) if ev in Q1.index else pd.NaT
        base = dict(kind="pick", trial="kcfb", event=ev, logged_at=now, status=r["status"],
                    event_start=None if pd.isna(start) else start.isoformat())
        if r["status"] != "bet":
            new.append(clean({**base, **{k: r.get(k) for k in ("S", "gap") if k in r}})); continue
        T = pd.Timestamp(r["T"])
        Sx = float(r["S"])
        tk = K[(K.event == ev) & (K.strike == Sx)].ticker
        side = "yes" if r["pick"] == "over" else "no"
        t = ts(T)
        c = C[(C.event == ev) & (C.strike == Sx) & (C.end <= t) & (C.end >= t - 6 * 3600) & C.bid.notna() & C.ask.notna()]
        qe = int(c.end.max()) if len(c) else None
        cc = C[(C.event == ev) & (C.strike == Sx) & (C.end <= ts(start)) & (C.bid > 0) & (C.ask < 1)].sort_values("end")
        cm = np.nan
        if len(cc):
            m = (cc.bid.iloc[-1] + cc.ask.iloc[-1]) / 2
            cm = m if side == "yes" else 1 - m
        ask = float(r["ask"])
        new.append(clean({**base, "decided_at": T.isoformat(), "S": Sx, "ticker": tk.iloc[0] if len(tk) else None,
                          "side": side, "pick": r["pick"], "pred": r["pred"], "open": r["open"], "gap": r["gap"],
                          "ask": ask, "fee": fee(ask), "cost_dev_script": r["cost"], "spread_c": r["spread_c"],
                          "quote_end": iso(qe) if qe else None}))
        fin = float(r["final"])
        res = "void" if fin == Sx else ("yes" if fin > Sx else "no")
        kr, kst = market_result(tk.iloc[0]) if len(tk) else (None, None)
        new.append(clean(dict(kind="settle", trial="kcfb", event=ev, logged_at=now, S=Sx, side=side, final=fin,
                              result=res, won=(res == side) if res != "void" else None, won_dev_script=r["won"],
                              close_line=r["close"], clv=r["clv"], close_mid=cm, kalshi_result=kr, kalshi_status=kst,
                              close_mid_note="last HOURLY candle ending <= kickoff (q1_fetch candles)")))
    append("kcfb", new)
    print(f"kcfb: forward rows {len(R)}; +{sum(x['kind'] == 'pick' for x in new)} pick rows "
          f"({sum(x.get('status') == 'bet' and x['kind'] == 'pick' for x in new)} bets); file {OUTD / 'kcfb.jsonl'}")


def status():
    for tr in ("k1", "knfl", "kcfb", "k50nfl"):
        rs = read(tr)
        p = [r for r in rs if r["kind"] == "pick"]
        print(tr, "pick rows", len(p), "bets", sum(r.get("status") == "bet" for r in p),
              "settled", sum(r["kind"] == "settle" for r in rs))


if __name__ == "__main__":
    {"k1": k1, "knfl": knfl, "kcfb": kcfb, "k50nfl": k50nfl, "status": status}[sys.argv[1]]()
