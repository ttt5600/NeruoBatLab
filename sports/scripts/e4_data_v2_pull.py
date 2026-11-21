"""e4_data_v2 step 3 data pulls (read-only public GETs; NO outcomes are joined here). DEV calendar only:
dates <= 2025-06-30 for every station, plus 2026-01-01..2026-02-28 (X1 dev) for the 17 X1 stations. Nothing in
2025-07-01..2025-12-31 or after 2026-02-28 is requested (E1/X1 sealed calendar).

    python -I scripts/e4_data_v2_pull.py mos     # IEM MOS archive: GFS (MAV), NAM (MET), NBS -> results/e4_data_v2/mos_<M>.parquet
    python -I scripts/e4_data_v2_pull.py acis    # ACIS daily maxt (official climate record) -> results/e4_data_v2/acis_maxt.parquet
    python -I scripts/e4_data_v2_pull.py cli     # IEM CLI archive (as-issued NWS CLI products) -> results/e4_data_v2/iem_cli.parquet
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd
import requests

from kalshi_paper import PAPER_CITIES
from sports import wx

OUT = ROOT / "results" / "e4_data_v2"
IEM = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"
DEV_END = pd.Timestamp("2025-06-30")
ORIG7 = [wx.CITIES[s][0] for s in wx.CITIES]
STATIONS = ORIG7 + [v[0] for k, v in PAPER_CITIES.items() if v[0] not in ORIG7]
START = {"GFS": 2018, "NAM": 2018, "NBS": 2020}


def windows(st):
    ws = [(pd.Timestamp(f"{y}-01-01"), min(pd.Timestamp(f"{y}-12-31"), DEV_END)) for y in range(2018, 2026)]
    if st not in ORIG7:
        ws.append((pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-28")))
    return ws


def get(params):
    for k in range(8):
        try:
            r = requests.get(IEM, params=params, timeout=300)
            if r.ok and r.text.startswith("runtime"):
                return r.text
            msg = r.text[:80]
        except Exception as e:  # noqa: BLE001
            msg = repr(e)[:80]
        print(f"  retry {k} {params['station']} {params['model']} {params['sts'][:10]}: {msg}", flush=True)
        time.sleep(30 * (k + 1))
    return None


def cmd_mos() -> int:
    for model in (sys.argv[2:] or ["NBS", "GFS", "NAM"]):
        f = OUT / f"mos_{model}.parquet"
        have = pd.read_parquet(f) if f.exists() else pd.DataFrame()
        done = set(have["_win"]) if len(have) else set()
        parts = [have] if len(have) else []
        for st in STATIONS:
            for a, b in windows(st):
                if a.year < START[model] and a.year != 2026:
                    continue
                key = f"{st}|{a.date()}"
                if key in done:
                    continue
                txt = get({"station": st, "model": model, "sts": f"{a:%Y-%m-%d}T00:00Z", "ets": f"{b:%Y-%m-%d}T23:59Z",
                           "format": "csv"})
                if txt is None:
                    print(f"FAIL {key} {model}", flush=True)
                    continue
                d = pd.read_csv(io.StringIO(txt), low_memory=False)
                col = "txn" if model == "NBS" else "n_x"
                keep = ["runtime", "ftime", "station", "model", col] + (["xnd"] if model == "NBS" else [])
                if col not in d.columns:
                    d = pd.DataFrame(columns=keep)
                d = d[d[col].notna()][keep].copy() if len(d) else pd.DataFrame(columns=keep)
                d["_win"] = key
                parts.append(d)
                pd.concat(parts, ignore_index=True).to_parquet(f, index=False)
                print(f"{model} {key}: {len(d)} rows", flush=True)
                time.sleep(6)
    return 0


def cmd_acis() -> int:
    rows = []
    for st in STATIONS:
        sid = st[1:]
        for a, b in [(pd.Timestamp("2015-01-01"), DEV_END)] + ([(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-28"))] if st not in ORIG7 else []):
            q = {"sid": sid, "sdate": str(a.date()), "edate": str(b.date()), "elems": "maxt", "meta": "name,sids"}
            r = requests.get("https://data.rcc-acis.org/StnData", params={"params": json.dumps(q)}, timeout=120).json()
            nm = r.get("meta", {}).get("name")
            for dte, v in r.get("data", []):
                rows.append({"station": st, "date": pd.Timestamp(dte), "acis_maxt": v, "acis_name": nm})
            print(st, sid, nm, len(r.get("data", [])), flush=True)
            time.sleep(1)
    d = pd.DataFrame(rows)
    d["acis_val"] = pd.to_numeric(d.acis_maxt.astype(str).str.rstrip("A"), errors="coerce")
    d.to_parquet(OUT / "acis_maxt.parquet", index=False)
    return 0


def cmd_cli() -> int:
    rows = []
    for st in ORIG7:                       # years 2021-2024 only: whole-year requests never touch the sealed calendar
        for y in range(2021, 2025):
            u = f"https://mesonet.agron.iastate.edu/json/cli.py?station={st}&year={y}"
            for k in range(6):
                try:
                    r = requests.get(u, timeout=120)
                    if r.ok:
                        js = r.json()
                        break
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(20 * (k + 1))
            else:
                print("FAIL", st, y, flush=True)
                continue
            for x in js.get("results", []):
                dte = pd.Timestamp(x.get("valid"))
                if dte > DEV_END and not (pd.Timestamp("2026-01-01") <= dte <= pd.Timestamp("2026-02-28")):
                    continue
                rows.append({"station": st, "date": dte, "cli_high": x.get("high"), "product": x.get("product"),
                             "cli_high_time": x.get("high_time")})
            print(st, y, len(js.get("results", [])), flush=True)
            time.sleep(5)
    d = pd.DataFrame(rows)
    d["cli_high"] = pd.to_numeric(d.cli_high, errors="coerce")
    d["cli_high_time"] = d.cli_high_time.astype(str)
    d.to_parquet(OUT / "iem_cli.parquet", index=False)
    return 0


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"mos": cmd_mos, "acis": cmd_acis, "cli": cmd_cli}.get(c, lambda: print(__doc__) or 1)())
