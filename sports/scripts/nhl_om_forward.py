"""Forward paper logger for NHL-om-A (edgelab nhl_om_a_kft-754cfc46; rule: notes/nhl_opener_model.md "Forward rule").
READ-ONLY: public Kalshi market-data GETs through kx_throttle, NHL API schedule, MoneyPuck files on disk. No orders.

    python scripts/nhl_om_forward.py tick     # decide new games at Kalshi's first-tight hour; settle finished bets
    python scripts/nhl_om_forward.py status

Output (append-only, never rewritten): ~/.claude/quant-sports/data/kx_forward/nhla.jsonl (override KX_FWD_DIR).
One `pick` row per game (status bet / skip gap / no c / never tight / no kalshi event ...), one `settle` row per bet.

Rule = results/nhl_opener_model/kalshi.py trial A, copied line by line (deviations in notes/ops_nhl_goalie.md):
  T     = first hourly candle end x <= tip - 2 h whose ladder anchor (two-sided rung with mid nearest 0.50, tie ->
          narrower spread; each rung's last candle within 6 h) has spread <= 4c.
  m     = model.py predict(home, away, D = ET date of T, projected goalies = last-10 start shares, b2b on game date)
          -- b2b from the NHL API schedule (team plays on game date - 1), not from MoneyPuck results.
  L_K   = kx.implied ladder median at T over all two-sided rungs.
  c     = mean(m - L_K) over the last 100 (by decision time) pool rows with tip < T, >= 30 required (else no bet);
          pool = kalshi.py trial-A dev rows (c_seed_A.csv) + this log's decisions.
  g     = m - c - L_K; |g| >= 0.40 -> buy S50 over (YES at ask) / under (NO at 1 - bid) + exact taker fee (100 ct).
  grade = Kalshi result of that rung. CLV_c = rung mid at the last hourly candle <= tip minus entry mid, signed (cents).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from kx_throttle import kget  # noqa: E402
from kx_forward import EPOCH, OUTD, append, clean, fee, hourly_candles, iso, market_result, read, ts  # noqa: E402

OM = ROOT / "results" / "nhl_opener_model"
XL = ROOT / "results" / "kalshi_openers_xleague"
SEED = Path(os.environ.get("NHLA_SEED", str(ROOT / "results" / "ops_nhl_goalie" / "c_seed_A.csv")))
TRIAL = os.environ.get("NHLA_TRIAL", "nhla")
FROM = pd.Timestamp(os.environ.get("NHLA_FROM", "2026-11-15"))          # forward = ET game date >= this
K = 0.40
KX_CODE = {"SJS": "SJ", "LAK": "LA", "TBL": "TB", "NJD": "NJ"}           # NHL API / MoneyPuck -> Kalshi event code
NOW = lambda: pd.Timestamp(os.environ["NHLA_NOW"], tz="UTC") if os.environ.get("NHLA_NOW") else pd.Timestamp.now(tz="UTC")  # noqa: E731
ET = "America/New_York"


def nhl_get(url):
    for i in range(4):
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "quant-sports-readonly-paper"})
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        import time
        time.sleep(3 * (i + 1))
    raise RuntimeError(url)


def schedule(d0: pd.Timestamp, d1: pd.Timestamp) -> pd.DataFrame:
    """NHL API regular-season games with ET date in [d0, d1] (weekly endpoint)."""
    rows, d = [], d0
    while d <= d1:
        j = nhl_get(f"https://api-web.nhle.com/v1/schedule/{d.strftime('%Y-%m-%d')}")
        for wk in j.get("gameWeek", []):
            for g in wk.get("games", []):
                if g.get("gameType") != 2:
                    continue
                rows.append(dict(gameId=g["id"], date=wk["date"], tip=g["startTimeUTC"], home=g["homeTeam"]["abbrev"],
                                 away=g["awayTeam"]["abbrev"], state=g.get("gameState"),
                                 hs=g["homeTeam"].get("score"), as_=g["awayTeam"].get("score")))
        d += pd.Timedelta(days=7)
    s = pd.DataFrame(rows).drop_duplicates("gameId")
    s["tip"] = pd.to_datetime(s.tip, utc=True)
    s["date"] = pd.to_datetime(s.date)
    return s.sort_values("tip").reset_index(drop=True)


def event_of(g) -> str:
    d = g.tip.tz_convert(ET).strftime("%y%b%d").upper()
    return f"KXNHLTOTAL-{d}{KX_CODE.get(g.away, g.away)}{KX_CODE.get(g.home, g.home)}"


def ladder(cd, h):  # = kalshi.py ladder()
    sys.path.insert(0, str(XL))
    from kx import implied
    x = cd[(cd.end <= h) & (cd.end > h - 6 * 3600)].sort_values("end").groupby("ticker").tail(1)
    x = x[(x.bid > 0) & (x.ask < 1) & x.bid.notna() & x.ask.notna()]
    if x.empty:
        return np.nan, None, x
    x = x.assign(mid=(x.bid + x.ask) / 2, spr=x.ask - x.bid)
    a = x.assign(d=(x.mid - .5).abs()).sort_values(["d", "spr"]).iloc[0]
    return implied(list(zip(x.strike, x.mid))), a, x


def c_of(pool: list, T: int):
    p = sorted([r for r in pool if r["tip"] < T and np.isfinite(r["diff"])], key=lambda r: r["Tdec"])[-100:]
    return (float(np.mean([r["diff"] for r in p])) if len(p) >= 30 else np.nan), len(p)


def tick():
    now = NOW()
    t_now = ts(now)
    have = read(TRIAL)
    picked = {r["key"]: r for r in have if r["kind"] == "pick"}
    settled = {r["key"] for r in have if r["kind"] == "settle"}
    today = now.tz_convert(ET).tz_localize(None).normalize()
    S = schedule(min(FROM, today) - pd.Timedelta(days=7), today + pd.Timedelta(days=4))
    played = set(zip(S.date, S.home)) | set(zip(S.date, S.away))
    # c pool: dev seed (only games before the forward start, so a replay does not double count) + logged decisions
    sd = pd.read_csv(SEED)
    sd = sd[pd.to_datetime(sd.tip, unit="s", utc=True).dt.tz_convert(ET).dt.tz_localize(None).dt.normalize() < FROM]
    pool = [dict(tip=int(r.tip), Tdec=int(r.Tdec), diff=float(r.m - r.L_K)) for r in sd.itertuples()]
    pool += [dict(tip=ts(r["event_start"]), Tdec=ts(r["decided_at"]), diff=r["m"] - r["L_K"])
             for r in picked.values() if r.get("m") is not None and r.get("L_K") is not None]
    M = None
    new = []
    for g in S[(S.date >= FROM) & (S.tip <= now + pd.Timedelta(days=4))].itertuples():
        key = event_of(g)
        if key in picked:
            continue
        tip = ts(g.tip)
        base = dict(kind="pick", trial=TRIAL, key=key, gameId=int(g.gameId), logged_at=now.isoformat(),
                    event_start=g.tip.isoformat(), home=g.home, away=g.away, game_date=g.date.strftime("%Y-%m-%d"))
        ms = (kget("/markets", {"event_ticker": key, "limit": 200}) or {}).get("markets", [])
        if not ms:
            if t_now > tip:
                new.append({**base, "status": "no kalshi event"})
            continue
        C = hourly_candles(key, t_now - 21 * 86400, min(t_now, tip))
        C = C[C.end <= min(t_now, tip)]
        hrs = sorted(C.end.unique()) if len(C) else []
        tA = next((int(x) for x in hrs if x <= tip - 7200 and (lambda L: L[1] is not None and L[1].spr <= 0.04)(ladder(C, x))), None)
        if tA is None:
            if t_now > tip - 7200 + 3600:   # the last eligible hourly candle (<= tip - 2 h) has closed
                new.append({**base, "status": "never tight", "n_hours": len(hrs)})
            continue
        D = (EPOCH + pd.Timedelta(seconds=tA)).tz_convert(ET).tz_localize(None).normalize()
        if M is None:
            sys.path.insert(0, str(OM))
            import model as M  # noqa: N811  (heavy: loads all MoneyPuck CSVs once per run)
            M.played_prev = lambda team, d: (pd.Timestamp(d) - pd.Timedelta(days=1), team) in played  # schedule b2b
        prev_days = S[(S.date < D) & S.state.isin(["OFF", "FINAL"])].date
        last_day = prev_days.max() if len(prev_days) else None
        mp_ok = last_day is None or bool((M.T.date == last_day).any())
        if not mp_ok and t_now < tip + 12 * 3600:
            print(f"{key}: waiting for MoneyPuck to include {last_day.date()} (decision hour {iso(tA)})")
            continue
        gh, ga = M.projected(g.home, D), M.projected(g.away, D)
        try:
            lh, la, m, bh, ba = M.predict(g.home, g.away, D, gh, ga, b2b_date=g.date)
        except KeyError as e:
            new.append({**base, "status": f"model miss {e}"}); continue
        L, an, x = ladder(C, tA)
        c, npool = c_of(pool, tA)
        gap = m - c - L
        row = {**base, "decided_at": iso(tA), "h_before": (tip - tA) / 3600, "D": D.strftime("%Y-%m-%d"),
               "m": m, "lh": lh, "la": la, "b2b_h": bh, "b2b_a": ba, "L_K": L, "c": c, "c_pool_n": npool, "g": gap,
               "proj_h": {str(k): round(v, 2) for k, v in gh.items()}, "proj_a": {str(k): round(v, 2) for k, v in ga.items()},
               "mp_stale": not mp_ok, "n_rungs": len(x), "S50": float(an.strike), "ticker": an.ticker,
               "yes_bid": float(an.bid), "yes_ask": float(an.ask), "spread_c": round(float(an.spr) * 100),
               "quote_end": iso(int(an.end))}
        pool.append(dict(tip=tip, Tdec=tA, diff=m - L))
        if not np.isfinite(c):
            new.append(clean({**row, "status": "no c"})); continue
        if abs(gap) < K:
            new.append(clean({**row, "status": "skip gap"})); continue
        over = gap > 0
        side = "yes" if over else "no"
        ask = float(an.ask) if over else 1 - float(an.bid)
        new.append(clean({**row, "status": "bet", "pick": "over" if over else "under", "side": side, "ask": ask,
                          "fee": fee(ask), "entry_mid": float(an.mid)}))
    append(TRIAL, new)
    picked.update({r["key"]: r for r in new})
    # settle
    st = []
    fin = {event_of(g): g for g in S.itertuples()}
    for k, p in picked.items():
        if p.get("status") != "bet" or k in settled or t_now < ts(p["event_start"]) + 3 * 3600:
            continue
        kr, kst = market_result(p["ticker"])
        if kr not in ("yes", "no"):
            continue
        tip = ts(p["event_start"])
        C = hourly_candles(k, tip - 4 * 86400, tip)
        C = C[C.end <= tip]
        LC = ladder(C, int(C.end.max()))[0] if len(C) else np.nan
        q = C[(C.ticker == p["ticker"]) & (C.bid > 0) & (C.ask < 1)].sort_values("end")
        cm = float((q.bid.iloc[-1] + q.ask.iloc[-1]) / 2) if len(q) else np.nan
        sg = 1 if p["side"] == "yes" else -1
        g = fin.get(k)
        final = (g.hs + g.as_) if g is not None and g.state in ("OFF", "FINAL") and g.hs is not None else None
        st.append(clean(dict(kind="settle", trial=TRIAL, key=k, logged_at=NOW().isoformat(), ticker=p["ticker"],
                             side=p["side"], S=p["S50"], final=final, result=kr, kalshi_result=kr, kalshi_status=kst,
                             won=(kr == p["side"]), close_mid_yes=cm,
                             close_mid=(cm if sg > 0 else 1 - cm) if np.isfinite(cm) else None,
                             close_mid_t=iso(int(q.end.iloc[-1])) if len(q) else None,
                             clv_c=sg * (cm - p["entry_mid"]) * 100 if np.isfinite(cm) else None,
                             L_close=LC, clv_g=sg * (LC - p["L_K"]) if np.isfinite(LC) else None)))
    append(TRIAL, st)
    print(f"{TRIAL}: games in schedule {len(S)}; +{len(new)} pick rows ({sum(r.get('status') == 'bet' for r in new)} bets: "
          f"{[r['key'] for r in new if r.get('status') == 'bet']}), +{len(st)} settle rows; file {OUTD / (TRIAL + '.jsonl')}")


def status():
    rs = read(TRIAL)
    p = [r for r in rs if r["kind"] == "pick"]
    from collections import Counter
    print(TRIAL, "pick rows", len(p), dict(Counter(r.get("status") for r in p)), "settled", sum(r["kind"] == "settle" for r in rs))


if __name__ == "__main__":
    {"tick": tick, "status": status}[sys.argv[1]]()
