"""Forward PAPER logger for PW1 (pw1_power_week_maker-25925930; rule edgelab/rules/power_week.py). No order code.

    python scripts/power_week.py tick   [--out DIR]   # one snapshot of every open weekly rung + inputs -> power_week_ticks.jsonl
    python scripts/power_week.py settle [--out DIR]   # closed tickers: result, CLV mark, tape -> _results / _tape.jsonl

Default DIR = data/live next to this script's parent. The deployed copy (~/.claude/quant-sports) is the writer of record.
Sources (all keyless): MISO docs.misoenergy.org marketreports da_exante / rt_lmp_prelim (day files, EST hour-ending);
ERCOT cdr html dam_spp / real_time_spp. Settlement is MISO rtlmpfinal / ERCOT NP6-905-CD; prelim MISO is the in-week
observation only. PJM excluded (settlement feed requires an API key).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import phys_ladders as P  # noqa: E402
from edgelab.rules import power_week as PW  # noqa: E402

H = {"User-Agent": "Mozilla/5.0"}
API = P.API
CACHE = ROOT / "data" / "power_week_cache"
SERIES = {"KXMISOINDW": "miso", "KXERCOTNORTHW": "ercot"}
EST = timezone(timedelta(hours=-5))
ET = ZoneInfo("America/New_York")
CT = ZoneInfo("America/Chicago")
MON = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}
MISO_NODE = "INDIANA.HUB"
ERCOT_HUB = "HB_NORTH"


def _fetch(url, key, complete):
    """text or None on 404; complete days are cached."""
    f = CACHE / key
    if f.exists():
        return f.read_text()
    for i in range(4):
        try:
            r = requests.get(url, headers=H, timeout=60)
        except requests.RequestException:
            time.sleep(2 + 2 * i)
            continue
        if r.status_code in (404, 400):
            return None
        if r.status_code >= 500 or r.status_code == 429:
            time.sleep(2 + 2 * i)
            continue
        r.raise_for_status()
        if complete:
            CACHE.mkdir(parents=True, exist_ok=True)
            f.write_text(r.text)
        return r.text
    return None


# (interval_start_utc_epoch, seconds, price) rows for one day
def miso_day(d: date, kind: str, now_ts: int):
    name = {"rt": "rt_lmp_prelim", "da": "da_exante_lmp"}[kind]
    complete = datetime(d.year, d.month, d.day, tzinfo=EST).timestamp() + 86400 + 4 * 3600 < now_ts
    txt = _fetch(f"https://docs.misoenergy.org/marketreports/{d:%Y%m%d}_{name}.csv", f"miso_{name}_{d:%Y%m%d}.csv", complete)
    if not txt and kind == "rt":  # prelim files are only kept for a few days; older days live under the final report
        txt = _fetch(f"https://docs.misoenergy.org/marketreports/{d:%Y%m%d}_rt_lmp_final.csv", f"miso_rt_lmp_final_{d:%Y%m%d}.csv", complete)
    if not txt:
        return []
    lines = txt.splitlines()
    i = next((j for j, s in enumerate(lines) if s.startswith("Node,Type,Value")), None)
    if i is None:
        return []
    hdr = lines[i].split(",")
    rows = []
    for s in lines[i + 1:]:
        p = s.split(",")
        if p[0] != MISO_NODE or p[2] != "LMP":
            continue
        for h, v in zip(hdr[3:], p[3:]):
            m = re.match(r"HE (\d+)", h)
            if not m or v.strip() == "":
                continue
            he = int(m.group(1))
            t0 = int(datetime(d.year, d.month, d.day, tzinfo=EST).timestamp()) + (he - 1) * 3600
            rows.append((t0, 3600, float(v)))
    return rows


def _html_table(txt):
    """first table with <th> headers -> DataFrame of strings; stdlib only (deployed venv has no lxml)."""
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", txt, re.S | re.I)
    cells = [[re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, re.S | re.I)] for r in rows]
    cells = [c for c in cells if c]
    if len(cells) < 2:
        return None
    w = len(cells[0])
    return pd.DataFrame([c for c in cells[1:] if len(c) == w], columns=cells[0])


def ercot_day(d: date, kind: str, now_ts: int):
    name = {"rt": "real_time_spp", "da": "dam_spp"}[kind]
    complete = datetime(d.year, d.month, d.day, tzinfo=CT).timestamp() + 86400 + 4 * 3600 < now_ts
    txt = _fetch(f"https://www.ercot.com/content/cdr/html/{d:%Y%m%d}_{name}.html", f"ercot_{name}_{d:%Y%m%d}.html", complete)
    if not txt:
        return []
    col = "Interval Ending" if kind == "rt" else "Hour Ending"
    df = _html_table(txt)
    if df is None or ERCOT_HUB not in df.columns or col not in df.columns:
        return []
    base = datetime(d.year, d.month, d.day, tzinfo=CT)
    rows = []
    sec = 900 if kind == "rt" else 3600
    for he, v in zip(df[col], pd.to_numeric(df[ERCOT_HUB], errors="coerce")):
        if not np.isfinite(v):
            continue
        s = re.sub(r"\D", "", str(he).split(".")[0]).zfill(2)
        s = s if len(s) >= 3 else s + "00"
        s = s.zfill(4)
        end = base + timedelta(hours=int(s[:2]), minutes=int(s[2:]))
        rows.append((int(end.timestamp()) - sec, sec, float(v)))
    return rows


DAY = {"miso": miso_day, "ercot": ercot_day}


def _range(a: date, b: date):
    while a <= b:
        yield a
        a += timedelta(days=1)


def inputs(iso: str, ws: int, we: int, now: int) -> dict:
    fn = DAY[iso]
    wd0 = datetime.fromtimestamp(ws, ET).date() - timedelta(days=1)
    wd1 = datetime.fromtimestamp(we, ET).date() + timedelta(days=1)
    rt = [r for d in _range(wd0, min(wd1, datetime.fromtimestamp(now, ET).date())) for r in fn(d, "rt", now)]
    done = sorted({r for r in rt if ws <= r[0] < we and r[0] + r[1] <= now})
    sec = 3600 if iso == "miso" else 900
    H_ = int((we - ws) / sec)
    da = [r for d in _range(datetime.fromtimestamp(now, ET).date(), wd1) for r in fn(d, "da", now)]
    fut = [v for t, s, v in da if ws <= t < we and t >= now]
    hist = [r for d in _range(datetime.fromtimestamp(ws, ET).date() - timedelta(days=14), datetime.fromtimestamp(ws, ET).date() - timedelta(days=1))
            for r in fn(d, "rt", now)]
    hv = [v for t, s, v in hist if t + s <= ws]
    return {"H": H_, "n_done": len(done), "a_sum": float(sum(v for _, _, v in done)),
            "da_mean": float(np.mean(fut)) if fut else None, "m14": float(np.mean(hv)) if hv else None,
            "inputs_ts": max((t + s for t, s, _ in done), default=ws), "n_hist": len(hv)}


def open_markets():
    out = []
    for series, iso in SERIES.items():
        d = P._get(f"{API}/markets", {"series_ticker": series, "status": "open", "limit": 1000})
        for m in d.get("markets", []):
            ev = m["event_ticker"]
            g = re.search(r"-(\d\d)([A-Z]{3})(\d\d)$", ev)
            if not g:
                continue
            ds = date(2000 + int(g.group(3)), MON[g.group(2)], int(g.group(1)))
            ws = int(datetime(ds.year, ds.month, ds.day, 17, tzinfo=ET).timestamp())
            we = int((datetime(ds.year, ds.month, ds.day, 17, tzinfo=ET) + timedelta(days=7)).timestamp())
            out.append({"series": series, "iso": iso, "ticker": m["ticker"], "event_ticker": ev, "date": ds.isoformat(),
                        "K": float(m["floor_strike"]) if m.get("floor_strike") is not None else None,
                        "strike_type": m.get("strike_type"), "close_ts": P._iso(m["close_time"]), "ws": ws, "week_end_ts": we,
                        "yes_bid": float(m["yes_bid_dollars"]) if m.get("yes_bid_dollars") not in (None, "") else None,
                        "yes_ask": float(m["yes_ask_dollars"]) if m.get("yes_ask_dollars") not in (None, "") else None})
    return out


def tick(out: Path):
    now = int(datetime.now(timezone.utc).timestamp())
    mk = open_markets()
    inp = {}
    rows = []
    for m in mk:
        k = (m["iso"], m["ws"], m["week_end_ts"])
        if k not in inp:
            inp[k] = inputs(m["iso"], m["ws"], m["week_end_ts"], now)
        rows.append({**m, **inp[k], "T": now})
    out.mkdir(parents=True, exist_ok=True)
    f = out / "power_week_ticks.jsonl"
    with open(f, "a") as fo:
        for r in rows:
            fo.write(json.dumps(r) + "\n")
    print(datetime.now(timezone.utc).isoformat(), len(rows), "rungs ->", f)
    for k, v in inp.items():
        print(" ", k[0], datetime.fromtimestamp(k[1], ET).date(), {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items()})
    df = pd.DataFrame(rows)
    if len(df):
        s = PW.select(df)
        print("rule picks this tick:", len(s))
        if len(s):
            print(s[["ticker", "K", "side", "q", "yes_bid", "yes_ask", "n_done", "H"]].to_string(index=False))


def settle(out: Path):
    f = out / "power_week_ticks.jsonl"
    if not f.exists():
        print("no ticks")
        return
    t = pd.read_json(f, lines=True)
    now = int(time.time())
    done = t.groupby("ticker").agg(series=("series", "first"), event=("event_ticker", "first"), close=("close_ts", "first"),
                                   first_t=("T", "min"))
    done = done[done["close"] < now]
    res, tape = [], []
    for ser, g in done.groupby("series"):
        mk = P.markets(ser).set_index("ticker")
        for ev, ge in g.groupby("event"):
            t0, t1 = int(ge["first_t"].min()) - 3600, int(ge["close"].max()) + 3600
            cs = P.candles(ser, ev, list(ge.index), t0, t1)
            for tk, r in ge.iterrows():
                if tk not in mk.index:
                    continue
                mb, ma = P.quote(cs.get(tk, []), int(r["close"]) - 3600)
                res.append({"ticker": tk, "result": mk.loc[tk, "result"], "expiration_value": float(mk.loc[tk, "expiration_value"]),
                            "mark_yes_bid": mb, "mark_yes_ask": ma})
                tape += P.tape(tk, int(r["first_t"]), int(r["close"]) + 3600)
    with open(out / "power_week_results.jsonl", "w") as fo:
        for r in res:
            fo.write(json.dumps(r, default=float) + "\n")
    with open(out / "power_week_tape.jsonl", "w") as fo:
        for r in tape:
            fo.write(json.dumps(r, default=float) + "\n")
    print(len(res), "settled tickers;", len(tape), "trades")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tick", "settle"])
    ap.add_argument("--out", default=str(ROOT / "data" / "live"))
    a = ap.parse_args()
    {"tick": tick, "settle": settle}[a.cmd](Path(a.out))
