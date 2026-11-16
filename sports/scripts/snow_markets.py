"""Snow markets (agent snow_markets; prereg sports/notes/snow_markets.md, written before any climatology/ensemble pull).

Read-only: Kalshi public GETs via kx_throttle, ACIS StnData (keyless), Open-Meteo ensemble API (keyless). No orders.

  ../.venv_quant/bin/python scripts/snow_markets.py clim     # ACIS climatology cache -> results/snow_markets/acis_*.json
  ../.venv_quant/bin/python scripts/snow_markets.py log      # one daily snapshot: TG1 (until 11-26) + SD1/SD2 -> jsonl

Rules (frozen in the notes):
  TG1  P = w*E + (1-w)*C ; E = share of GEFS(31)+ECMWF-ENS(51) members with Nov-26 LST-day snowfall > 0 at the station,
       C = share of station-days Nov19..Dec3 1991..2025 with snow >= trace. w = .5 (lead>=7), .75 (3..6), .9 (<=2).
  SD1/SD2  FV = mean over seasons 1996-97..2025-26 of floor_cent(clip((N_obs + R_s - floor)/(cap-floor), 0, 1)),
       R_s = qualifying days over the remaining window in season s (SD1: N_obs = 0, whole window, before Dec 1).
  Fire (taker): YES if P - (ya + fee) >= .10 ; NO if (1-P) - (1-yb + fee) >= .10 ; fee = .07 p (1-p) exact.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kx_throttle import kget  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "snow_markets"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "snow_log.jsonl"
MARGIN = 0.10
TG_EVENT = "KXSNOWTHANKSGIVING-26NOV26"
TG_DATE = dt.date(2026, 11, 26)
SD_EVENT = "KXSNOWDAYS-01DEC26-01APR27"
SD_START, SD_END = dt.date(2026, 12, 1), dt.date(2027, 4, 1)
CLIM_SEASONS = list(range(1996, 2026))   # season s = Dec 1 s .. Apr 1 s+1  (1996-97 .. 2025-26)

TZ = {"E": "America/New_York", "C": "America/Chicago", "M": "America/Denver", "P": "America/Los_Angeles"}
LST_OFF = {"E": -5, "C": -6, "M": -7, "P": -8}
ZONE = {**{s: "E" for s in "BOS PWM PVD NYC EWR TTN PHL PIT BWI DCA CLE CMH DTW IND ATL CLT CVG".split()},
        **{s: "C" for s in "ORD MKE MSN MSP DSM AUS DFW HOU MCI BNA".split()},
        "DEN": "M", "SEA": "P", "PDX": "P"}
# KXSNOWDAYS ticker suffix -> station (from rules_primary, census.json)
SD_STN = {"ATL": "ATL", "AUS": "AUS", "BAL": "BWI", "BOS": "BOS", "CLT": "CLT", "CHI": "ORD", "CVG": "CVG", "CLE": "CLE",
          "DFW": "DFW", "DEN": "DEN", "DTW": "DTW", "HOU": "HOU", "IND": "IND", "KC": "MCI", "MSP": "MSP", "NASH": "BNA",
          "NYC": "NYC", "PHL": "PHL", "PIT": "PIT", "PDX": "PDX", "SEA": "SEA", "DC": "DCA"}


def fee(p: float) -> float:
    return 0.07 * p * (1 - p)


def floor_cent(x: float) -> float:
    return math.floor(round(x * 100, 6)) / 100


def acis(stn: str, sdate: str, edate: str) -> dict:
    q = {"sid": "K" + stn, "sdate": sdate, "edate": edate, "elems": "snow", "meta": "name,ll,sids"}
    r = requests.get("https://data.rcc-acis.org/StnData", params={"params": json.dumps(q)}, timeout=120)
    r.raise_for_status()
    return r.json()


def snowval(v: str):
    """ACIS snow string -> (amount_in or None if missing, is_trace)."""
    if v in ("M", "", None):
        return None, False
    if v == "T":
        return 0.0, True
    v = v.rstrip("ASE")  # accumulated / estimated flags
    try:
        return float(v), False
    except ValueError:
        return None, False


def stations() -> set[str]:
    c = json.loads((OUT / "census.json").read_text())
    s = {m["custom_strike"]["station"][1:] for m in c["KXSNOWTHANKSGIVING"]["events"][0]["markets"]}
    return s | set(SD_STN.values())


def cmd_clim() -> int:
    for stn in sorted(stations()):
        f = OUT / f"acis_{stn}.json"
        if f.exists():
            continue
        d = acis(stn, "1991-01-01", str(dt.date.today() - dt.timedelta(days=1)))
        f.write_text(json.dumps(d))
        print(stn, d.get("meta", {}).get("name"), len(d.get("data", [])))
    return 0


def load(stn: str) -> tuple[dict, dict]:
    d = json.loads((OUT / f"acis_{stn}.json").read_text())
    return d["meta"], {dt.date.fromisoformat(a): snowval(b) for a, b in d["data"]}


def tg_clim(stn: str) -> tuple[float, int]:
    _, s = load(stn)
    hit = n = 0
    for y in range(1991, 2026):
        for k in range(15):
            day = dt.date(y, 11, 19) + dt.timedelta(days=k)
            amt, tr = s.get(day, (None, False))
            if amt is None:
                continue
            n += 1
            hit += int(tr or amt > 0)
    return hit / n, n


def season_counts(stn: str, thr: float, frm_md: tuple[int, int] | None = None) -> list[int]:
    """Qualifying-day counts per climatological season over [frm .. Apr 1]; frm default Dec 1. Season dropped if > 5 missing."""
    _, s = load(stn)
    out = []
    for y in CLIM_SEASONS:
        a = dt.date(y, 12, 1)
        if frm_md:
            m, dd = frm_md
            a = dt.date(y if m == 12 else y + 1, m, dd)
        b = dt.date(y + 1, 4, 1)
        n = miss = 0
        day = a
        while day <= b:
            amt, _ = s.get(day, (None, False))
            if amt is None:
                miss += 1
            elif amt >= thr - 1e-9:
                n += 1
            day += dt.timedelta(days=1)
        if miss <= 5:
            out.append(n)
    return out


def payout(n: int, lo: float, hi: float) -> float:
    return floor_cent(min(max((n - lo) / (hi - lo), 0.0), 1.0))


def ensemble_frac(lat: float, lon: float, zone: str) -> tuple[float | None, int]:
    """Share of GEFS+ECMWF-ENS members with snowfall > 0 during Nov 26 LST (fixed UTC offset)."""
    off = LST_OFF[zone]
    t0 = dt.datetime(2026, 11, 26) - dt.timedelta(hours=off)       # LST midnight in UTC
    r = requests.get("https://ensemble-api.open-meteo.com/v1/ensemble", params={
        "latitude": lat, "longitude": lon, "hourly": "snowfall", "models": "gfs025,ecmwf_ifs025",
        "start_date": "2026-11-26", "end_date": "2026-11-27", "timezone": "GMT"}, timeout=120)
    r.raise_for_status()
    h = r.json()["hourly"]
    times = [dt.datetime.fromisoformat(t) for t in h["time"]]
    # hourly snowfall at time t = preceding hour sum -> hour ending t; LST day = (t0, t0+24h]
    idx = [i for i, t in enumerate(times) if t0 < t <= t0 + dt.timedelta(hours=24)]
    yes = n = 0
    for k, v in h.items():
        if not k.startswith("snowfall"):
            continue
        # keys: snowfall[_memberNN]_ncep_gefs025 (31) / snowfall[_memberNN]_ecmwf_ifs025_ensemble (51). A member counts only
        # if all 24 LST hours are present (GEFS 0.25 deg stops at 240 h, so it joins once the day is inside its horizon).
        # Fixed 2026-11-16 after the first snapshot (which used the 50 ECMWF perturbed members only; no fires, no outcome).
        if not (k.endswith("_ncep_gefs025") or k.endswith("_ecmwf_ifs025_ensemble")):
            continue
        vals = [v[i] for i in idx]
        if len(idx) < 24 or any(x is None for x in vals):
            continue
        n += 1
        yes += int(sum(x or 0 for x in vals) > 0)
    return (yes / n if n else None), n


def decide(p: float, yb: float, ya: float) -> str | None:
    if ya is not None and 0 < ya < 1 and p - (ya + fee(ya)) >= MARGIN:
        return "yes"
    na = 1 - yb if yb is not None else None
    if na is not None and 0 < na < 1 and (1 - p) - (na + fee(na)) >= MARGIN:
        return "no"
    return None


def fl(x):
    return None if x in (None, "") else float(x)


def cmd_log() -> int:
    now = dt.datetime.now(dt.timezone.utc)
    today = now.date()
    rows = []
    fired = set()
    if LOG.exists():
        for ln in LOG.read_text().splitlines():
            r = json.loads(ln)
            if r.get("fire"):
                fired.add((r["ticker"], r["fire"]))
    # TG1
    if today <= TG_DATE:
        lead = (TG_DATE - today).days
        w = 0.5 if lead >= 7 else (0.75 if lead >= 3 else 0.9)
        for m in (kget("/markets", {"event_ticker": TG_EVENT, "limit": 100}) or {}).get("markets", []):
            stn = m["custom_strike"]["station"][1:]
            meta, _ = load(stn)
            lon, lat = meta["ll"]
            e, ne = ensemble_frac(lat, lon, ZONE[stn])
            c, nc = tg_clim(stn)
            p = w * e + (1 - w) * c if e is not None else None
            yb, ya = fl(m.get("yes_bid_dollars")), fl(m.get("yes_ask_dollars"))
            f = decide(p, yb, ya) if p is not None else None
            first = f is not None and (m["ticker"], f) not in fired
            rows.append({"ts": now.isoformat(), "rule": "TG1", "ticker": m["ticker"], "stn": stn, "lead": lead, "w": w,
                         "E": e, "n_members": ne, "C": c, "n_clim": nc, "P": p, "yb": yb, "ya": ya,
                         "status": m.get("status"), "vol": fl(m.get("volume_fp")), "fire": f, "first_fire": first,
                         "cost": (ya + fee(ya)) if f == "yes" else ((1 - yb) + fee(1 - yb)) if f == "no" else None})
    # SD1 (before Dec 1) / SD2 (in season)
    if today <= SD_END:
        rule = "SD1" if today <= SD_START else "SD2"
        for m in (kget("/markets", {"event_ticker": SD_EVENT, "limit": 100}) or {}).get("markets", []):
            suf = m["ticker"].split("-")[-1]
            stn = SD_STN[suf]
            thr = 1.0 if "at least 1.0 inches" in m["rules_primary"] else 0.1
            lo, hi = float(m["custom_strike"]["scalar_floor"]), float(m["custom_strike"]["scalar_cap"])
            nobs, ok = 0, True
            if rule == "SD2":
                _, s = load_fresh(stn)
                day = SD_START
                while day < today:
                    amt, _ = s.get(day, (None, False))
                    if amt is None:
                        ok = False
                        break
                    nobs += int(amt >= thr - 1e-9)
                    day += dt.timedelta(days=1)
                rem = season_counts(stn, thr, (today.month, today.day))
            else:
                rem = season_counts(stn, thr)
            fv = sum(payout(nobs + r, lo, hi) for r in rem) / len(rem) if (ok and len(rem) >= 25) else None
            yb, ya = fl(m.get("yes_bid_dollars")), fl(m.get("yes_ask_dollars"))
            f = decide(fv, yb, ya) if fv is not None else None
            first = f is not None and (m["ticker"], f) not in fired
            pinned = nobs >= hi
            rows.append({"ts": now.isoformat(), "rule": rule, "ticker": m["ticker"], "stn": stn, "thr": thr, "floor": lo,
                         "cap": hi, "n_obs": nobs, "n_seasons": len(rem), "FV": fv, "yb": yb, "ya": ya,
                         "mid": (yb + ya) / 2 if yb is not None and ya is not None else None,
                         "vol": fl(m.get("volume_fp")), "pinned_yes": pinned, "fire": f, "first_fire": first,
                         "cost": (ya + fee(ya)) if f == "yes" else ((1 - yb) + fee(1 - yb)) if f == "no" else None})
    with LOG.open("a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    for r in rows:
        print(r["rule"], r["ticker"].split("-")[-1], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
              if k in ("E", "C", "P", "FV", "yb", "ya", "fire", "first_fire", "n_obs")})
    return 0


def load_fresh(stn: str):
    """Current-season ACIS (not cached) for SD2."""
    d = acis(stn, str(SD_START), str(dt.date.today() - dt.timedelta(days=1)))
    return d["meta"], {dt.date.fromisoformat(a): snowval(b) for a, b in d["data"]}


if __name__ == "__main__":
    sys.exit({"clim": cmd_clim, "log": cmd_log}[sys.argv[1]]())
