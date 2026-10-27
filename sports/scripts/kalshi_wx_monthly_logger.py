"""L39 FORWARD PAPER logger: Kalshi monthly rain / snow "unreachable" NO (W1/W1m, S1/S1m). NO ORDERS.

    python scripts/kalshi_wx_monthly_logger.py tick [--force]   # launchd hourly at :05; decides each city ONCE per day
    python scripts/kalshi_wx_monthly_logger.py settle           # score picks whose market has settled (tick also does it)
    python scripts/kalshi_wx_monthly_logger.py report           # n, W/L, P&L per contract + units, taker / maker, by series

The rule is the FROZEN one in scripts/kalshi_wx_winter.py (docstring) and is imported from there, not re-typed:
  window   last 7 days of the month (end-6 <= d <= end), decision T_d = 22:00 STATION-LOCAL on day d.
           launchd fires every hour at :05; a city is decided at the first tick whose local time is 22:00-23:59 on d
           (23:xx = grace hour after sleep, same local date; lateness is recorded). Missed both hours = no decision.
  state    OBS_d = ACIS observed total over days 1..d-1 (any missing day -> no decision that day, logged).
           F_m,d = winter.forecasts(): Open-Meteo previous-runs, day d+j at lead j+1, 5 models, need >= 3;
           FMAX = max. Snow = 10 x liquid.
  pick     X - OBS_d > 2*FMAX + margin (0.5 in rain, 1.0 in snow), market active, tradeable quote
           (yes_bid > 0, yes_ask < 1, spread <= 10c, yes_ask > 0.01) -> paper NO.
           Taker: NO ask = 1 - yes_bid, cost = a + 0.07 a (1-a).  Maker (W1m/S1m, the lead variant): resting NO bid at
           q = min(floor_cent(NO mid), NO ask - 1c); fill judged at settle from the public tape (>= 1 contract printed
           at no_price <= q after T_d and before close). One pick per bracket per month (the first qualifying d).
  Quotes here are a live snapshot of the market endpoint (+ orderbook depth), where the backtest used the last
  hourly candle close -- the only live/backtest difference. Station = winter.rain_station / winter.SNOW.

--force: evaluate every in-window city NOW regardless of local hour. Writes ONLY to the preview file (never to
picks or the once-per-day state), so an off-schedule quote cannot pre-empt the 22:00 decision.

Files (data/, gitignored; deployed copy ~/.claude/quant-kalshi/data/):
  kalshi_wx_monthly_picks.jsonl    paper picks (settle fills result / fill / pnl in place)
  kalshi_wx_monthly_evals.jsonl    every bracket evaluated at a decision (audit trail, qualifying or not)
  kalshi_wx_monthly_state.json     last tick, per event-day decision status
  kalshi_wx_monthly_preview.jsonl  --force runs only
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

import kalshi_wx_winter as W
from sports import wx
from sports.http import fetch

D = ROOT / "data"
PICKS = D / "kalshi_wx_monthly_picks.jsonl"
EVALS = D / "kalshi_wx_monthly_evals.jsonl"
STATE = D / "kalshi_wx_monthly_state.json"
PREVIEW = D / "kalshi_wx_monthly_preview.jsonl"
MARGIN = {"rain": 0.50, "snow": 1.0}
TZS = ["America/New_York", "America/Chicago", "America/Denver", "America/Phoenix", "America/Los_Angeles"]
DEC_HOURS = (W.wx.DECISION_HOUR, W.wx.DECISION_HOUR + 1)  # 22:00-23:59 local


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def append(p: Path, rows: list[dict]) -> None:
    if rows:
        D.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")


def in_window(day) -> bool:
    end = (pd.Timestamp(day) + pd.offsets.MonthEnd(0)).normalize()
    return end - pd.Timedelta(days=6) <= pd.Timestamp(day).normalize() <= end


def f(x) -> float:
    return float(x) if x not in (None, "") else np.nan


def depth(ticker: str) -> dict:
    """Resting size: YES bids (= NO you can TAKE at 1 - yes_bid) and NO bids (queue ahead of a maker NO bid)."""
    try:
        ob = fetch(f"{wx.API}/markets/{ticker}/orderbook", cache=False)
        ob = ob.get("orderbook_fp") or ob.get("orderbook") or {}
    except Exception as ex:  # noqa: BLE001
        return {"depth_err": str(ex)[:80]}
    out = {}
    for side in ("yes", "no"):
        lv = sorted(((float(p) if float(p) <= 1 else float(p) / 100, float(q))
                     for p, q in (ob.get(f"{side}_dollars") or ob.get(side) or [])), reverse=True)
        best = lv[0][0] if lv else np.nan
        out[f"{side}_bid_best"] = best
        out[f"{side}_bid_size"] = sum(q for p, q in lv if p >= best - 1e-9) if lv else 0.0
        out[f"{side}_bid_size_2c"] = sum(q for p, q in lv if p >= best - 0.0201) if lv else 0.0
    return out


# ------------------------------------------------------------------------------------------------ discovery
def open_ladders(month_code: str) -> list[dict]:
    """Open `greater` markets of every monthly rain/snow series for event month `month_code` (e.g. '26OCT')."""
    series = [(s, "rain") for s in W.rain_series()] + [(s, "snow") for s in W.SNOW]
    out = []
    for s, kind in series:
        cur = None
        while True:
            p = {"series_ticker": s, "status": "open", "limit": 1000}
            if cur:
                p["cursor"] = cur
            d = fetch(f"{wx.API}/markets", p, cache=False)
            for m in d.get("markets", []):
                if m.get("strike_type") == "greater" and m["event_ticker"].split("-")[-1] == month_code:
                    sid = W.rain_station(m.get("rules_primary", "")) if kind == "rain" else W.SNOW[s]
                    out.append({**m, "series": s, "kind": kind, "sid": sid})
            cur = d.get("cursor")
            if not cur or not d.get("markets"):
                break
    return out


# ------------------------------------------------------------------------------------------------ tick
def decide(ev: str, g: list[dict], lat, lon, tz, d: pd.Timestamp, now: datetime, force: bool,
           picked: set[str]) -> tuple[dict, list[dict], list[dict]]:
    kind, sid, series = g[0]["kind"], g[0]["sid"], g[0]["series"]
    end = (d + pd.offsets.MonthEnd(0)).normalize()
    st = {"event": ev, "d": str(d.date()), "sid": sid, "tz": tz, "decided_utc": now.isoformat(timespec="seconds"),
          "local": now.astimezone(ZoneInfo(tz)).strftime("%F %H:%M"), "markets": len(g)}
    var = "pcpn" if kind == "rain" else "snow"
    ob = W.acis(sid, str(d.replace(day=1).date()), str((d - pd.Timedelta(days=1)).date()))
    if ob.empty or len(ob) != d.day - 1 or ob[var].isna().any():
        st |= {"status": "no_decision", "reason": f"ACIS incomplete ({0 if ob.empty else int(ob[var].isna().sum())} "
                                                  f"missing of {d.day - 1})"}
        return st, [], []
    obs = float(ob[var].sum())
    fc = W.forecasts(lat, lon, tz, end)
    if kind == "snow":
        fc["F"] = fc["F"] * 10.0
    fd = fc[fc.d == d].set_index("model")["F"]
    fcs = {m: (float(fd[m]) if m in fd and pd.notna(fd[m]) else None) for m in W.MODELS}
    have = [v for v in fcs.values() if v is not None]
    n, fmax = len(have), (max(have) if have else np.nan)
    st |= {"obs_d": obs, "n_models": n, "fmax": fmax}
    if n < 3:
        st |= {"status": "no_decision", "reason": f"only {n} models"}
        return st, [], []
    T = int(now.timestamp())
    evals, picks = [], []
    for m in sorted(g, key=lambda m: float(m["floor_strike"])):
        X = float(m["floor_strike"])
        yb, ya = f(m.get("yes_bid_dollars")), f(m.get("yes_ask_dollars"))
        close = int(pd.Timestamp(m["close_time"]).timestamp())
        gap = X - obs
        fires = gap > 2 * fmax + MARGIN[kind]
        tradeable = (m.get("status") == "active" and T < close and yb > 0 and ya < 1 and ya - yb <= 0.10 + 1e-9
                     and ya > 0.01)
        e = {"event": ev, "ticker": m["ticker"], "series": series, "kind": kind, "sid": sid, "d": str(d.date()),
             "days_left": (end - d).days, "T": T, "X": X, "obs_d": obs, "gap": gap, "fmax": fmax, "n_models": n,
             "threshold": 2 * fmax + MARGIN[kind], "fires": bool(fires), "yes_bid": yb, "yes_ask": ya,
             "tradeable": bool(tradeable), "already_picked": m["ticker"] in picked, "off_schedule": force}
        evals.append(e)
        if fires and tradeable and m["ticker"] not in picked:
            a = 1 - yb
            mid = 1 - (yb + ya) / 2
            q = round(min(np.floor(round(mid * 100, 6)) / 100, a - 0.01), 4)
            picks.append({**e, "ts": now.isoformat(timespec="seconds"), "local": st["local"], "side": "no",
                          "fc": fcs, "no_ask": a, "taker_fee": W.FEE * a * (1 - a),
                          "taker_cost": a + W.FEE * a * (1 - a), "no_bid": 1 - ya, "no_mid": mid, "maker_q": q,
                          "yes_bid_size_mkt": f(m.get("yes_bid_size_fp")), "volume_24h": f(m.get("volume_24h_fp")),
                          "volume": f(m.get("volume_fp")), "open_interest": f(m.get("open_interest_fp")),
                          "close_time": m["close_time"], **depth(m["ticker"])})
    st |= {"status": "decided", "fires": sum(e["fires"] for e in evals),
           "fires_tradeable": sum(e["fires"] and e["tradeable"] for e in evals), "picks": len(picks)}
    return st, evals, picks


def cmd_tick(force: bool = False) -> int:
    now = datetime.now(timezone.utc)
    state = json.loads(STATE.read_text()) if STATE.exists() else {"decisions": {}, "station": {}}
    state["last_tick"] = now.isoformat(timespec="seconds")
    if not force:
        cmd_settle(quiet=True)
    # any tz whose local date is in the window and (unless forced) in the decision hours?
    cand = [z for z in TZS if in_window(now.astimezone(ZoneInfo(z)).date())
            and (force or now.astimezone(ZoneInfo(z)).hour in DEC_HOURS)]
    if not cand:
        state["last_status"] = "idle (no station-local 22:00-23:59 inside a month's last 7 days)"
        STATE.write_text(json.dumps(state, indent=1, default=str))
        return 0
    month_codes = {pd.Timestamp(now.astimezone(ZoneInfo(z)).date()).strftime("%y%b").upper() for z in cand}
    mk = [m for c in month_codes for m in open_ladders(c)]
    picked = {p["ticker"] for p in jl(PICKS)}
    by_ev: dict[str, list] = {}
    for m in mk:
        by_ev.setdefault(m["event_ticker"], []).append(m)
    all_evals, all_picks, n_dec = [], [], 0
    for ev, g in sorted(by_ev.items()):
        sid = g[0]["sid"]
        if not sid:
            state["decisions"][ev] = {"status": "skip", "reason": "no station parsed from rules"}
            continue
        if sid not in state["station"]:
            try:
                state["station"][sid] = list(W.station_meta(sid))
            except Exception as ex:  # noqa: BLE001
                state["decisions"][ev] = {"status": "skip", "reason": f"no NWS station meta: {ex}"[:120]}
                continue
        lat, lon, tz = state["station"][sid]
        local = now.astimezone(ZoneInfo(tz))
        d = pd.Timestamp(local.date())
        if not in_window(d) or d.strftime("%y%b").upper() != ev.split("-")[-1]:
            continue
        if not force and local.hour not in DEC_HOURS:
            continue
        key = f"{ev}|{d.date()}"
        if not force and key in state["decisions"]:
            continue
        try:
            st, evals, picks = decide(ev, g, lat, lon, tz, d, now, force, picked)
        except Exception as ex:  # noqa: BLE001 -- one city's API failure must not kill the others; retried next tick
            print(f"{ev}: ERROR {ex}")
            continue
        n_dec += 1
        if not force:
            state["decisions"][key] = st
        all_evals += evals
        all_picks += picks
        picked |= {p["ticker"] for p in picks}
        print(f"{ev:22s} {st['local']} {st['status']:11s} obs {st.get('obs_d', np.nan):5.2f} "
              f"FMAX {st.get('fmax', np.nan):5.2f} (n={st.get('n_models', 0)}) fires {st.get('fires', 0)} "
              f"tradeable {st.get('fires_tradeable', 0)} picks {st.get('picks', 0)} {st.get('reason', '')}")
    for p in all_picks:
        print(f"  PICK NO {p['ticker']:24s} X {p['X']:5.2f} gap {p['gap']:5.2f} > {p['threshold']:4.2f}  "
              f"yes {p['yes_bid']:.2f}/{p['yes_ask']:.2f}  taker {p['taker_cost']:.4f}  maker q {p['maker_q']:.2f}  "
              f"depth NO-ask {p.get('yes_bid_size', np.nan):.0f} / NO-bid queue {p.get('no_bid_size', np.nan):.0f}")
    if force:
        append(PREVIEW, all_picks)
    else:
        append(PICKS, all_picks)
        append(EVALS, all_evals)
        state["last_status"] = f"decided {n_dec} event-days, {len(all_picks)} picks"
        STATE.write_text(json.dumps(state, indent=1, default=str))
    print(f"{'PREVIEW (not logged as picks)' if force else 'logged'}: {n_dec} event-days, {len(all_picks)} picks")
    return 0


# ------------------------------------------------------------------------------------------------ settle / report
def cmd_settle(quiet: bool = False) -> int:
    rows = jl(PICKS)
    now = datetime.now(timezone.utc)
    todo = [r for r in rows if "result" not in r and pd.Timestamp(r["close_time"]) < now]
    for r in todo:
        try:
            m = fetch(f"{wx.API}/markets/{r['ticker']}", cache=False)["market"]
        except Exception as ex:  # noqa: BLE001
            print(f"{r['ticker']}: {ex}")
            continue
        if m.get("result") not in ("yes", "no"):
            continue
        won = m["result"] == r["side"]
        close = int(pd.Timestamp(m["close_time"]).timestamp())
        t = W.tape(r["ticker"], int(r["T"]))
        hit = (t.ts > r["T"]) & (t.ts <= close) & (t.no_price <= r["maker_q"] + 1e-9)
        r |= {"result": m["result"], "won": won, "settled_close": m["close_time"],
              "maker_filled": bool(t.loc[hit, "count"].sum() >= 1),
              "pnl_taker": (1 - r["taker_cost"]) if won else -r["taker_cost"],
              "pnl_maker": (1 - r["maker_q"]) if won else -r["maker_q"]}
        if not quiet:
            print(f"settled {r['ticker']} -> {m['result']} ({'WIN' if won else 'LOSS'}) maker fill {r['maker_filled']}")
    if todo:
        PICKS.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    if not quiet:
        print(f"{len(todo)} past close checked; {sum('result' in r for r in rows)}/{len(rows)} settled")
    return 0


def cmd_report() -> int:
    rows = jl(PICKS)
    if not rows:
        print("no picks yet")
        return 0
    df = pd.DataFrame(rows)
    for c in ("result", "won", "maker_filled", "pnl_taker", "pnl_maker"):
        if c not in df:
            df[c] = np.nan
    df["month"] = df.event.str.split("-").str[-1]
    s = df[df.result.notna()].copy()
    s["won"] = s.won.astype(bool)
    s["mf"] = s.maker_filled.astype(bool)
    s["u_taker"] = np.where(s.won, 1 / s.taker_cost - 1, -1.0)
    s["u_maker"] = np.where(s.won, 1 / s.maker_q - 1, -1.0)

    def agg(g):
        fm = g[g.mf]
        return pd.Series({"n": len(g), "W": int(g.won.sum()), "L": int((~g.won).sum()),
                          "taker_c/ct": 100 * g.pnl_taker.sum(), "taker_units": g.u_taker.sum(),
                          "makerAll_c/ct": 100 * g.pnl_maker.sum(), "makerAll_units": g.u_maker.sum(),
                          "maker_filled": len(fm), "makerTape_c/ct": 100 * fm.pnl_maker.sum(),
                          "makerTape_units": fm.u_maker.sum()})
    print(f"L39 forward paper: {len(df)} picks, {len(s)} settled, {len(df) - len(s)} open "
          f"({df[df.result.isna()].groupby('month').size().to_dict()})")
    if len(s):
        tab = s.groupby(["kind", "series"]).apply(agg)
        tab.loc[("ALL", ""), :] = agg(s)
        print("P&L summed over picks: c/ct = cents per 1-contract pick; units = stake 1 unit of cost per pick")
        print(tab.round(2).to_string())
        if (~s.won).any():
            print("losers:", s.loc[~s.won, ["ticker", "d", "X", "obs_d", "fmax", "yes_bid", "yes_ask"]].to_string())
    o = df[df.result.isna()]
    if len(o):
        print("\nopen picks:")
        print(o[["d", "ticker", "X", "obs_d", "fmax", "yes_bid", "yes_ask", "taker_cost", "maker_q"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = {"tick": lambda: cmd_tick("--force" in sys.argv), "settle": cmd_settle, "report": cmd_report}.get(cmd)
    raise SystemExit(fn() if fn else (print(__doc__) or 1))
