"""FX1-fwd: forward PAPER log of the ForecastEx weather consensus-reject rule (edge E5) at LIVE public quotes.
No orders, no login, no keys: public GET endpoints only (IBKR's key-free ForecastTrader feed, Open-Meteo,
api.weather.gov, forecastex.com daily price files).

    python scripts/forecastex_paper.py tick       # launchd every 5 min: decide due station-nights, settle hourly
    python scripts/forecastex_paper.py settle     # fill outcomes from the public daily files, then report
    python scripts/forecastex_paper.py report
    python scripts/forecastex_paper.py validate   # tomorrow's listed contracts for every station, any hour, logs nothing
    python scripts/forecastex_paper.py q008-validate   # Q008 econ snapshot now (any hour), logs nothing

WHY: the sealed FX1 test (results/forecastex_wx_sealed.json, PASS-UNDERPOWERED 2026-10-28: T1 219 bets, +15.05u,
ROI +6.9%, mean cost 0.898) priced at D-1 LAST TRADES from daily files, which have no bid/ask. This log prices the same
rule at the quote that existed at the decision instant. Public quotes exist (checked 2026-10-28):
  www.interactivebrokers.com/tws.proxy/public/forecasttrader/prediction/contract/list?conid=<underlying>&exchange=FORECASTX
  returns per strike bid/ask for YES and NO (top of book) and open interest, but NO SIZE. Sizes are logged as null.

PRE-REGISTERED 2026-10-28, before the first tick. Trial label: FX1-fwd.
STATIONS: the 24 UH* daily-high markets of the sealed test (venues/forecastex_client.STATIONS), WU station K<code>.
RULE: frozen, identical code -- scripts/forecastex_wx.py `candidates` (sha256 checked at import, aborts if changed),
  MARGIN 3F, >= 3 of {gfs_seamless, ecmwf_ifs025, icon_seamless, gem_seamless, ncep_nbm_conus}. Forecasts: the live
  Open-Meteo call of scripts/kalshi_paper.py `live_forecasts` (same function the E1/E2 paper agent runs; max over hours
  <= 18:00 local, >= 15 values) at the K<code> coordinates. NOTE the backtest used Open-Meteo previous-runs lead-1;
  live uses the latest run at 22:00, as kalshi_paper does for E1.
DECISION: once per station-night, at the first tick inside 22:00-23:59 station-local on D-1, for contracts of event day D.
  Every rule candidate is logged with both instruments' bid/ask, whether or not a quote exists.
PRICES (scored on the bought side, fee $0.01/contract as in the sealed test, take only if cost < 1):
  A  at ask (primary): cost = side ask + 0.01; needs an ask. This is a price we could have taken.
  MID          : cost = (side bid + side ask)/2 + 0.01; needs both. Context, not executable.
  MK maker proxy: a resting bid at the side's best bid; FILLED iff the bought instrument's day-D daily-file low_price
     < that bid with pair_quantity > 0 (a print strictly through our price; daily files have no time stamps, so this is
     a proxy, labelled as such). cost = bid + 0.01. Unfilled = no bet.
  L  last trade (sealed T1 method on the same candidates): D-1 end_price of the bought instrument if it traded D-1.
SETTLEMENT: YES settlement_price exactly 0/1 in the latest daily file dated D..D+3, and only once file D+1 exists
  (the sealed rule's settlement, without the monotonicity filter, which needs all strikes; non-0/1 -> unsettled).
REPORT: per variant n, losses, units, ROI, station-day bootstrap p (forecastex_wx.score, unchanged), mean cost, vs the
  sealed T1 ROI +6.9% / mean cost 0.898. No decision is pre-authorised by this log: it informs whether the sealed edge
  survives quote prices before any IBKR order (notes/forecastex_setup.md). Read-out rule fixed now: after >= 100
  settled A-picks, A ROI > 0 = "survives at the ask"; A n << L n = "the edge lives where no ask exists" (maker-only).

ADDENDUM 2026-10-28 (pre-registered before its first tick; FX1-fwd above is unchanged): pick type `q008`, fund
question Q008 (QUESTIONS.md; dev evidence notes/q007_forecastex_breadth.md, exploratory, last trades: 64 bets / 57
events, 0 losses, ROI +16% at FX last + 1c, +4.7% at the Kalshi ask). Separate logs: data/forecastex_q008.jsonl (picks),
data/forecastex_q008_obs.jsonl (every matched contract with a Kalshi side mid >= 0.80, the H2/kill-rate denominator).
UNIVERSE: ForecastEx-listed (contract exchange == FORECASTX; the public list also carries KALSHI/CME-routed rows, which
  are dropped) CPIY, CPIC, MMCPI, PREMP, UNR, IJC, FF, matched to Kalshi KXCPIYOY, KXCPICOREYOY, KXCPI, KXPAYROLLS, KXU3,
  KXJOBLESSCLAIMS, KXFED (status open) on: Kalshi floor_strike == FX strike + offset (0, FF +0.125: the Q007 frozen map,
  outcome agreement 115/115) and Kalshi close_time date (America/Chicago) == FX expiration date, else +-1 day (exact
  preferred). Unmatched FX contracts are not traded.
TIMING: once per Chicago calendar day, at the first tick with Chicago hour == 15 (launchd ticks every 5 min), for every
  matched contract with 1 <= (FX expiration date - Chicago today) <= 30 days. FX quotes (public IBKR ForecastTrader
  contract/list, top of book, no size) and Kalshi quotes (public /markets yes_bid/yes_ask) are read in the same tick
  (seconds apart) and both logged with the tick timestamp.
SIGNAL: Kalshi two-sided (yes_bid > 0, yes_ask < 1, spread <= 0.20); kmid = (yes_bid + yes_ask)/2; favoured side YES
  if kmid >= 0.5 else NO, side mid m = max(kmid, 1-kmid). Pick iff m >= 0.85 AND the FX ask of that side exists (own
  level, else 1 - the other side's bid, as FX1-fwd) AND FX ask <= m - 0.02 AND FX ask + 0.01 < 1. ONE pick per FX
  contract ever (first signal; later days skip it). DEVIATION from the Q008 text: ">= 10 contracts at the ask" cannot
  be checked (the public feed has no size); size is logged null, the filter is dropped, and fill size is unknown.
PRICE: cost = FX side ask + $0.01 fee, 1 unit. No FX ask -> no pick (the obs row records `fx_ask_missing`).
  Counterfactual on the same picks: Kalshi side ask (YES ask, or 1 - yes_bid) + taker fee 0.07 p (1-p).
SETTLEMENT: FX YES settlement_price exactly 0/1 in the forecastex.com daily price file dated E..E+3 (E = FX expiration),
  matched on (product, expiration date, strike), only once file E+1 exists.
REPORT (`report`, section Q008): n, W-L, units, ROI at FX ask, event-clustered (product+expiration) bootstrap p, mean
  cost; the same picks at the Kalshi ask; qualifying-ask rate = share of obs contract-days with m >= 0.85 that met the
  FX filter.
JUDGE: early KILL any time if >= 3 losses in the first 30 settled picks. At 60 settled picks: KILL if ROI at FX ask <= 0
  or the qualifying-ask rate < 25% (dev gap was a last-trade artefact); otherwise CONTINUE toward the Q008 pass bar
  (>= 100 bets / 40 events, one-sided p < .05, ROI > 0, Wilson-95 upper loss rate < 1 - mean cost). Paper only; no
  decision to trade is pre-authorised by this log.
"""
from __future__ import annotations

import fcntl
import hashlib
import io
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import requests

FX_SHA = "8a6edbb5c86d1492778c41f4c9ed95d4da18aa9304fbb8ca6024a97182a88f92"
_fx_path = ROOT / "scripts" / "forecastex_wx.py"
if hashlib.sha256(_fx_path.read_bytes()).hexdigest() != FX_SHA:
    raise SystemExit(f"{_fx_path} differs from the frozen sealed-test code (sha256 {FX_SHA[:8]}...); refusing to run")
import forecastex_wx as fx  # noqa: E402  frozen rule + scoring
import kalshi_paper as kp  # noqa: E402  live forecasts (same code as the E1/E2 paper agent)
from venues.forecastex_client import STATIONS, ForecastEx  # noqa: E402

TRIAL = "FX1-fwd"
DATA = ROOT / "data"
LOG = DATA / "forecastex_paper.jsonl"
STATE = DATA / "forecastex_paper_state.json"
FILES = DATA / "forecastex_paper_files"
LOCK = DATA / "forecastex_paper.lock"
SEALED = ROOT / "results" / "forecastex_wx_sealed.json"
FEE = fx.FEE
H = {"User-Agent": "Mozilla/5.0"}


def load() -> list[dict]:
    return [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()] if LOG.exists() else []


def save_all(rows: list[dict]) -> None:
    tmp = LOG.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    tmp.replace(LOG)


def state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {"done": [], "meta": {}, "last_settle": None}


def station_meta(st: dict, code: str) -> tuple[float, float, str]:
    if code not in st["meta"]:
        st["meta"][code] = list(fx.station_meta(code))  # api.weather.gov K<code>: lat, lon, tz
    lat, lon, tz = st["meta"][code]
    return float(lat), float(lon), tz


def evaluate(fxc: ForecastEx, st: dict, code: str, day: date, now: datetime) -> tuple[list[dict], str]:
    strikes = [c for c in fxc.strikes(code, max_age=0).values() if c["day"] == day]
    if not strikes:
        return [], f"{code} {day}: no listed contracts"
    lat, lon, tz = station_meta(st, code)
    fc = kp.live_forecasts(lat, lon, tz, day)
    df = pd.DataFrame([{"ticker": c["ticker"], "X": c["strike"], "monotone": True,
                        **{f"fcsafe_{m}": fc.get(m) for m in kp.MODELS}} for c in strikes])
    df[[f"fcsafe_{m}" for m in kp.MODELS]] = df[[f"fcsafe_{m}" for m in kp.MODELS]].astype(float)
    cand = fx.candidates(df)
    by = {c["ticker"]: c for c in strikes}
    out = []
    for _, r in cand.iterrows():
        c = by[r.ticker]
        s = "yes" if r.side == "YES" else "no"
        o = "no" if s == "yes" else "yes"
        # side quotes, with the pair complement where the side's own level is missing
        bid = c[f"bid_{s}"] if c[f"bid_{s}"] is not None else (round(1 - c[f"ask_{o}"], 6) if c[f"ask_{o}"] is not None else None)
        ask = c[f"ask_{s}"] if c[f"ask_{s}"] is not None else (round(1 - c[f"bid_{o}"], 6) if c[f"bid_{o}"] is not None else None)
        out.append({"kind": "pick", "trial": TRIAL, "ts": now.isoformat(timespec="seconds"), "code": code,
                    "event_date": day.isoformat(), "ticker": r.ticker, "X": float(r.X), "side": r.side,
                    "fc": fc, "n_models": int(sum(v is not None for v in fc.values())),
                    "bid_yes": c["bid_yes"], "ask_yes": c["ask_yes"], "bid_no": c["bid_no"], "ask_no": c["ask_no"],
                    "side_bid": bid, "side_ask": ask, "size_bid": None, "size_ask": None,  # public feed has no size
                    "open_interest": c["open_interest"], "conid_yes": c["conid_yes"], "conid_no": c["conid_no"],
                    "quote_source": "ibkr_public_forecasttrader", "n_strikes": len(strikes)})
    nm = int(sum(v is not None for v in fc.values()))
    tag = "" if nm >= 3 else " TOO FEW MODELS"
    return out, f"{code} {day}: {len(strikes)} strikes, {len(out)} rule picks, {nm} models{tag} {fc}"


def cmd_tick(validate: bool = False) -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w") as lk:
        try:
            fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        now = datetime.now(timezone.utc)
        st, rows, fxc = state(), load(), ForecastEx(env={})
        done, new, msgs = set(st["done"]), [], []
        for code in STATIONS:
            try:
                tz = station_meta(st, code)[2]
            except Exception as e:
                msgs.append(f"{code}: station meta ERROR {e!r}"[:200])
                continue
            local = now.astimezone(ZoneInfo(tz))
            if not validate and local.hour not in (kp.DECISION_HOUR, kp.DECISION_HOUR + 1):
                continue
            day = (local + timedelta(days=1)).date()
            if validate and not any(c["day"] == day for c in fxc.strikes(code).values()):
                day = local.date()  # validate off-hours: tomorrow may not be listed yet, exercise today's book
            key = f"{code}:{day.isoformat()}"
            if not validate and key in done:
                continue
            try:
                got, msg = evaluate(fxc, st, code, day, now)
            except Exception as e:
                msgs.append(f"{code} {day}: ERROR {e!r}"[:300])
                continue
            msgs.append(msg)
            if validate:
                for r in got:
                    print(f"VALIDATE {r['ticker']} {r['side']} bid {r['side_bid']} ask {r['side_ask']} OI {r['open_interest']}")
                continue
            if "no listed contracts" in msg or "TOO FEW MODELS" in msg:
                continue  # retry on the next tick inside the window
            new += got
            done.add(key)
        if not validate:
            try:  # Q008 addendum: isolated so it can never block FX1-fwd
                msgs += q008_tick(fxc, st, now)
            except Exception as e:
                msgs.append(f"q008: ERROR {e!r}"[:300])
        for m in msgs:
            print(m)
        if validate:
            return 0
        if new:
            with LOG.open("a") as f:
                for r in new:
                    f.write(json.dumps(r, default=str) + "\n")
        st["done"] = sorted(done)
        last = st.get("last_settle")
        if not last or now - datetime.fromisoformat(last) > timedelta(hours=1):
            settle(now)
            st["last_settle"] = now.isoformat(timespec="seconds")
        STATE.write_text(json.dumps(st, indent=1))
    return 0


# ---------------------------------------------------------------------- settlement from the public daily files
def day_file(d: date, today: date) -> pd.DataFrame | None:
    """UH rows of the forecastex.com daily price file dated d (cached once d < today)."""
    f = FILES / f"uh_{d:%Y%m%d}.csv"
    if f.exists():
        return pd.read_csv(f)
    if d >= today:
        return None
    r = requests.get(fx.URL, params={"type": "prices", "date": f"{d:%Y%m%d}"}, headers=H, timeout=120)
    if r.status_code != 200 or not r.text.strip():
        return None
    t = pd.read_csv(io.StringIO(r.text))
    if t.empty:
        return None
    t = t[t.event_contract.astype(str).str.startswith("UH")]
    FILES.mkdir(parents=True, exist_ok=True)
    t.to_csv(f, index=False)
    return t


def settle(now: datetime | None = None) -> int:
    now = now or datetime.now(timezone.utc)
    today = now.date()
    rows = load()
    changed = 0
    for r in rows:
        if r.get("kind") != "pick" or r.get("yes_won") is not None:
            continue
        D = date.fromisoformat(r["event_date"])
        if day_file(D + timedelta(days=1), today) is None:
            continue
        inst = "NO" if r["side"] == "NO" else "YES"
        fin = None
        for k in range(3, -1, -1):
            t = day_file(D + timedelta(days=k), today)
            if t is None:
                continue
            y = t[(t.event_contract == r["ticker"]) & (t.subtype == "YES")]
            if len(y) and float(y.settlement_price.iloc[-1]) in (0.0, 1.0):
                fin = float(y.settlement_price.iloc[-1])
                break
        if fin is None:
            continue
        prev, dd = day_file(D - timedelta(days=1), today), day_file(D, today)
        def row(t):
            if t is None:
                return None
            x = t[(t.event_contract == r["ticker"]) & (t.subtype == inst)]
            return x.iloc[-1] if len(x) else None
        p, q = row(prev), row(dd)
        r["yes_won"] = fin
        r["last_d1"] = float(p.end_price) if p is not None and p.pair_quantity > 0 else None
        r["low_d"] = float(q.low_price) if q is not None and q.pair_quantity > 0 else None
        r["settled_ts"] = now.isoformat(timespec="seconds")
        changed += 1
    if changed:
        save_all(rows)
    print(f"settled {changed} new picks")
    try:
        q008_settle(now)
    except Exception as e:
        print(f"q008 settle ERROR {e!r}"[:300])
    return changed


# ---------------------------------------------------------------------- report
def variant(t: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "A":
        price = t.side_ask
    elif name == "MID":
        price = (t.side_bid + t.side_ask) / 2
    elif name == "MK":
        price = t.side_bid.where(t.low_d < t.side_bid)  # filled iff a day-D print went strictly through our bid
    else:
        price = t.last_d1
    d = t.assign(price=price)
    d = d[d.price.notna() & (d.price > 0) & (d.price + FEE < 1)].copy()
    d["cost"] = d.price + FEE
    d["won"] = np.where(d.side == "NO", d.yes_won == 0, d.yes_won == 1)
    d["pnl"] = np.where(d.won, 1 / d.cost - 1, -1.0)
    return d


def cmd_report() -> int:
    rows = [r for r in load() if r.get("kind") == "pick"]
    if not rows:
        print("no FX1-fwd picks logged yet")
        q008_report()
        return 0
    t = pd.DataFrame(rows)
    for c in ("side_bid", "side_ask", "yes_won", "last_d1", "low_d"):
        t[c] = pd.to_numeric(t.get(c), errors="coerce")
    t["event_date"] = pd.to_datetime(t.event_date)
    s = t[t.yes_won.notna()]
    print(f"{TRIAL}: {len(t)} picks logged ({t.event_date.min().date()}..{t.event_date.max().date()}, "
          f"{t.code.nunique()} stations), {len(s)} settled; quotes: side ask present {t.side_ask.notna().mean():.0%}, "
          f"bid present {t.side_bid.notna().mean():.0%} (public feed: top of book, no size)")
    sealed = json.loads(SEALED.read_text())["trials"]["T1_end"] if SEALED.exists() else {}
    if sealed:
        print(f"sealed T1 (last trade) estimate: ROI {sealed['roi']:+.1%}, mean cost {sealed['mean_cost']:.3f}, n {sealed['bets']}")
    labels = {"A": "at ask (primary)", "MID": "at mid (context)", "MK": "maker@bid, day-D print-through PROXY",
              "L": "D-1 last trade (sealed T1 method)"}
    for k, lab in labels.items():
        d = variant(s, k)
        if d.empty:
            print(f"  {k:3s} {lab}: 0 bets")
            continue
        r = fx.score(d, np.random.default_rng(0))
        print(f"  {k:3s} {lab}: n {r['bets']} losses {r['losses']} units {r['units']:+.2f} ROI {r['roi']:+.2%} "
              f"mean cost {r['mean_cost']:.3f} p {r['p']:.3f}")
    q008_report()
    return 0


# ====================================================================== Q008 (ADDENDUM 2026-10-28, see docstring)
Q_LOG = DATA / "forecastex_q008.jsonl"
Q_OBS = DATA / "forecastex_q008_obs.jsonl"
Q_TZ = ZoneInfo("America/Chicago")
Q_HOUR = 15
KAPI = "https://api.elections.kalshi.com/trade-api/v2"
# FX product -> (Kalshi series, Kalshi floor_strike - FX strike); frozen from the Q007 match (outcome agreement checked)
Q_MAP = {"CPIY": ("KXCPIYOY", 0.0), "CPIC": ("KXCPICOREYOY", 0.0), "MMCPI": ("KXCPI", 0.0), "PREMP": ("KXPAYROLLS", 0.0),
         "UNR": ("KXU3", 0.0), "IJC": ("KXJOBLESSCLAIMS", 0.0), "FF": ("KXFED", 0.125)}
Q_MIN, Q_GAP, Q_MAXSPR = 0.85, 0.02, 0.20


def _jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def _num(v):
    try:
        return None if v is None or v == "" else float(v)
    except (TypeError, ValueError):
        return None


def q008_snapshot(fxc: ForecastEx, now: datetime) -> tuple[list[dict], list[dict], list[str]]:
    """One snapshot of every matched pair with 1-30 d to release -> (obs rows with side mid >= .80, signals, msgs)."""
    today = now.astimezone(Q_TZ).date()
    ts = now.isoformat(timespec="seconds")
    tree = fxc._pub("/category/tree")["categories"]
    und = {m["symbol"]: int(m["conid"]) for cat in tree.values() for m in cat.get("markets", [])
           if m.get("exchange") == "FORECASTX" and m.get("symbol") in Q_MAP}
    obs, sig, msgs = [], [], []
    for prod, (series, off) in Q_MAP.items():
        if prod not in und:
            msgs.append(f"q008 {prod}: not in public category tree")
            continue
        d = fxc._pub("/prediction/contract/list", {"conid": str(und[prod]), "exchange": "FORECASTX"})
        fxrows = []
        for cc in d.get("complex_contracts", []):
            k = (cc.get("contracts") or [{}])[0]
            if k.get("exchange") != "FORECASTX" or cc.get("strike") is None or not cc.get("expiration"):
                continue
            E = datetime.strptime(str(cc["expiration"]), "%Y%m%d").date()
            if 1 <= (E - today).days <= 30:
                fxrows.append((E, float(cc["strike"]), cc, k))
        if not fxrows:
            msgs.append(f"q008 {prod}: 0 FX contracts 1-30 d out")
            continue
        r = requests.get(f"{KAPI}/markets", params={"series_ticker": series, "status": "open", "limit": 1000},
                         headers=H, timeout=30)
        r.raise_for_status()
        kidx = {}
        for m in r.json().get("markets", []):
            if m.get("floor_strike") is None or not m.get("close_time"):
                continue
            cd = pd.Timestamp(m["close_time"]).tz_convert(Q_TZ).date()
            kidx[(cd, round(float(m["floor_strike"]), 4))] = m
        nm = ns = 0
        for E, X, cc, k in fxrows:
            ks = round(X + off, 4)
            m = next((kidx[(E + timedelta(days=dd), ks)] for dd in (0, -1, 1) if (E + timedelta(days=dd), ks) in kidx), None)
            if m is None:
                continue
            nm += 1
            kb, ka = _num(m.get("yes_bid_dollars")), _num(m.get("yes_ask_dollars"))
            if kb is None or ka is None or kb <= 0 or ka >= 1 or ka - kb > Q_MAXSPR:
                continue
            kmid = (kb + ka) / 2
            side = "YES" if kmid >= 0.5 else "NO"
            s, o = side.lower(), ("no" if side == "YES" else "yes")
            mside = max(kmid, 1 - kmid)
            if mside < 0.80:
                continue
            fb = {x: _num(cc.get(x)) for x in ("bid_yes", "ask_yes", "bid_no", "ask_no")}
            fask = fb[f"ask_{s}"] if fb[f"ask_{s}"] is not None else (round(1 - fb[f"bid_{o}"], 6) if fb[f"bid_{o}"] is not None else None)
            fbid = fb[f"bid_{s}"] if fb[f"bid_{s}"] is not None else (round(1 - fb[f"ask_{o}"], 6) if fb[f"ask_{o}"] is not None else None)
            kask_side = ka if side == "YES" else round(1 - kb, 6)
            qual = mside >= Q_MIN and fask is not None and fask <= mside - Q_GAP + 1e-9 and fask + FEE < 1
            row = {"kind": "q008_obs", "ts": ts, "product": prod, "fx_exp": E.isoformat(), "X": X,
                   "fx_key": f"{prod}|{E.isoformat()}|{X:g}", "days_to": (E - today).days, "side": side,
                   "kmid_side": round(mside, 4), "k_ticker": m["ticker"], "k_yes_bid": kb, "k_yes_ask": ka,
                   "k_ask_side": kask_side, **{f"fx_{x}": v for x, v in fb.items()}, "fx_side_bid": fbid,
                   "fx_side_ask": fask, "fx_size_ask": None, "fx_open_interest": cc.get("open_interest"),
                   "conid_yes": k.get("conid_yes"), "conid_no": k.get("conid_no"), "qualifies": bool(qual),
                   "fx_ask_missing": fask is None}
            obs.append(row)
            if qual:
                sig.append({**row, "kind": "q008", "trial": "Q008-fwd", "cost": round(fask + FEE, 4),
                            "k_cost": round(kask_side + 0.07 * kask_side * (1 - kask_side), 4)})
                ns += 1
        msgs.append(f"q008 {prod}: {len(fxrows)} FX contracts 1-30 d, {nm} matched to {series}, {ns} signals")
    return obs, sig, msgs


def q008_tick(fxc: ForecastEx, st: dict, now: datetime, validate: bool = False) -> list[str]:
    local = now.astimezone(Q_TZ)
    day = local.date().isoformat()
    if not validate and (local.hour != Q_HOUR or day in st.setdefault("q008_days", [])):
        return []
    obs, sig, msgs = q008_snapshot(fxc, now)
    picked = {r["fx_key"] for r in _jl(Q_LOG)}
    new = []
    for r in sig:
        if r["fx_key"] not in picked:
            picked.add(r["fx_key"])
            new.append(r)
    if validate:
        for r in obs:
            print(f"Q008-VALIDATE {r['fx_key']} {r['side']} kmid {r['kmid_side']} K-ask {r['k_ask_side']} "
                  f"FX bid/ask {r['fx_side_bid']}/{r['fx_side_ask']} {'SIGNAL' if r['qualifies'] else ''}"
                  f"{' (new)' if r in new or any(n['fx_key'] == r['fx_key'] for n in new) else ''}")
        return msgs
    with Q_OBS.open("a") as f:
        for r in obs:
            f.write(json.dumps(r, default=str) + "\n")
    with Q_LOG.open("a") as f:
        for r in new:
            f.write(json.dumps(r, default=str) + "\n")
    st["q008_days"] = sorted(set(st["q008_days"]) | {day})
    return msgs + [f"q008 {day}: {len(obs)} obs (side mid >= .80), {len(sig)} signals, {len(new)} new picks"]


def econ_file(d: date, today: date) -> pd.DataFrame | None:
    """Q008 product rows of the forecastex.com daily price file dated d (cached once d < today)."""
    f = FILES / f"econ_{d:%Y%m%d}.csv"
    if f.exists():
        return pd.read_csv(f)
    if d >= today:
        return None
    r = requests.get(fx.URL, params={"type": "prices", "date": f"{d:%Y%m%d}"}, headers=H, timeout=120)
    if r.status_code != 200 or not r.text.strip():
        return None
    t = pd.read_csv(io.StringIO(r.text))
    if t.empty:
        return None
    t = t[t.event_contract.astype(str).str.split("_").str[0].isin(list(Q_MAP))]
    FILES.mkdir(parents=True, exist_ok=True)
    t.to_csv(f, index=False)
    return t


def q008_settle(now: datetime) -> int:
    today = now.date()
    rows = _jl(Q_LOG)
    changed = 0
    for r in rows:
        if r.get("kind") != "q008" or r.get("yes_won") is not None:
            continue
        E = date.fromisoformat(r["fx_exp"])
        if econ_file(E + timedelta(days=1), today) is None:
            continue
        fin = None
        for k in range(3, -1, -1):
            t = econ_file(E + timedelta(days=k), today)
            if t is None or t.empty:
                continue
            p = t.event_contract.astype(str).str.split("_")
            y = t[(p.str[0] == r["product"]) & (pd.to_numeric(p.str[-1], errors="coerce").round(4) == round(r["X"], 4))
                  & (t.expiration_date.astype(str).str[:10] == r["fx_exp"]) & (t.subtype == "YES")]
            if len(y) and float(y.settlement_price.iloc[-1]) in (0.0, 1.0):
                fin = float(y.settlement_price.iloc[-1])
                r["fx_event_contract"] = str(y.event_contract.iloc[-1])
                break
        if fin is None:
            continue
        r["yes_won"] = fin
        r["settled_ts"] = now.isoformat(timespec="seconds")
        changed += 1
    if changed:
        tmp = Q_LOG.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(x, default=str) + "\n" for x in rows))
        tmp.replace(Q_LOG)
    print(f"q008 settled {changed} new picks")
    return changed


def _q_score(d: pd.DataFrame, cost: str, rng, B: int = 2000) -> dict:
    won = np.where(d.side == "NO", d.yes_won == 0, d.yes_won == 1)
    pnl = np.where(won, 1 / d[cost] - 1, -1.0)
    ev = (d["product"] + d.fx_exp).to_numpy()
    ue = np.unique(ev)
    g = [np.where(ev == e)[0] for e in ue]
    bs = np.array([pnl[np.concatenate([g[i] for i in rng.integers(0, len(ue), len(ue))])].mean() for _ in range(B)])
    return {"n": len(d), "events": len(ue), "w": int(won.sum()), "l": int((~won).sum()), "units": float(pnl.sum()),
            "roi": float(pnl.mean()), "p": float((bs <= 0).mean()), "mean_cost": float(d[cost].mean())}


def q008_report() -> None:
    picks, obs = [r for r in _jl(Q_LOG) if r.get("kind") == "q008"], _jl(Q_OBS)
    print(f"\nQ008-fwd (ForecastEx econ, Kalshi-favoured side >= {Q_MIN} with FX ask <= Kalshi mid - {Q_GAP}): "
          f"{len(picks)} picks logged, {len(obs)} obs rows")
    if obs:
        o = pd.DataFrame(obs)
        f = o[o.kmid_side >= Q_MIN]
        if len(f):
            print(f"  Kalshi-favoured (>= {Q_MIN}) contract-days {len(f)}: FX ask present {(~f.fx_ask_missing).mean():.0%}, "
                  f"qualifying-ask rate {f.qualifies.mean():.0%} (kill if < 25% at the 60-pick read); median FX ask - "
                  f"Kalshi mid {(f.fx_side_ask - f.kmid_side).median():+.3f}")
    if not picks:
        return
    t = pd.DataFrame(picks)
    t["yes_won"] = pd.to_numeric(t.get("yes_won"), errors="coerce")
    s = t[t.yes_won.notna()]
    print(f"  settled {len(s)} / {len(t)}; by product {t['product'].value_counts().to_dict()}")
    if s.empty:
        return
    a, k = _q_score(s, "cost", np.random.default_rng(0)), _q_score(s, "k_cost", np.random.default_rng(0))
    print(f"  at FX ask + $0.01 (primary): n {a['n']} ({a['events']} events) W-L {a['w']}-{a['l']} units {a['units']:+.2f} "
          f"ROI {a['roi']:+.2%} p {a['p']:.3f} mean cost {a['mean_cost']:.3f}")
    print(f"  same picks at Kalshi ask + 0.07p(1-p): ROI {k['roi']:+.2%} units {k['units']:+.2f} mean cost {k['mean_cost']:.3f}")
    first30 = s.sort_values("ts").head(30)
    l30 = int((np.where(first30.side == "NO", first30.yes_won != 0, first30.yes_won != 1)).sum())
    verdict = ("KILL (>= 3 losses in first 30)" if l30 >= 3 else
               "60-pick read due: KILL if ROI <= 0 or qualifying-ask rate < 25%, else CONTINUE" if len(s) >= 60 else
               f"running ({len(s)}/60 to the judge read)")
    print(f"  losses in first 30: {l30}; status: {verdict}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "tick":
        raise SystemExit(cmd_tick())
    if cmd == "validate":
        raise SystemExit(cmd_tick(validate=True))
    if cmd == "q008-validate":
        for m in q008_tick(ForecastEx(env={}), {}, datetime.now(timezone.utc), validate=True):
            print(m)
        raise SystemExit(0)
    if cmd == "settle":
        settle()
        raise SystemExit(cmd_report())
    if cmd == "report":
        raise SystemExit(cmd_report())
    print(__doc__)
    raise SystemExit(1)
