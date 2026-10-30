"""Q011 step 1: download + parse the NBA official injury reports (ak-static.cms.nba.com) for 2025-26 and 2026-27.

    ../.venv_quant/bin/python scripts/q011_reports.py fetch      # resumable; ~4 req/s; cache data/q011/reports/*.json
    ../.venv_quant/bin/python scripts/q011_reports.py table      # -> data/q011/report_rows.parquet + report_index.parquet

URL pattern (verified 2026-10-29 by probing):
  2025-10-21 .. 2025-12-22 : hourly   Injury-Report_YYYY-MM-DD_HHAM|PM.pdf  (label HH, but the PDF header says HH:30 and
                              HTTP Last-Modified is HH:30 ET -> the label is NOT the publication time)
  2025-12-22 .. now         : 15-min   Injury-Report_YYYY-MM-DD_HH_MMAM|PM.pdf
Only game days (ESPN games_2026/2027, regular + post season, start <= now) are fetched, labels 07:00 .. 23:45 ET.
PDFs are NOT kept (size); each report's parsed rows + header time + HTTP Last-Modified are cached as JSON.
"""
from __future__ import annotations

import io
import json
import re
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUTD = ROOT / "data" / "q011"
REP = OUTD / "reports"
BASE = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_"
UA = {"User-Agent": "Mozilla/5.0 (research; sports backtest)"}
STATUSES = {"Out", "Questionable", "Doubtful", "Probable", "Available"}
TEAMS = {  # full name with spaces removed -> NBA/Kalshi code
    "AtlantaHawks": "ATL", "BostonCeltics": "BOS", "BrooklynNets": "BKN", "CharlotteHornets": "CHA", "ChicagoBulls": "CHI",
    "ClevelandCavaliers": "CLE", "DallasMavericks": "DAL", "DenverNuggets": "DEN", "DetroitPistons": "DET",
    "GoldenStateWarriors": "GSW", "HoustonRockets": "HOU", "IndianaPacers": "IND", "LAClippers": "LAC",
    "LosAngelesClippers": "LAC", "LosAngelesLakers": "LAL", "MemphisGrizzlies": "MEM", "MiamiHeat": "MIA",
    "MilwaukeeBucks": "MIL", "MinnesotaTimberwolves": "MIN", "NewOrleansPelicans": "NOP", "NewYorkKnicks": "NYK",
    "OklahomaCityThunder": "OKC", "OrlandoMagic": "ORL", "Philadelphia76ers": "PHI", "PhoenixSuns": "PHX",
    "PortlandTrailBlazers": "POR", "SacramentoKings": "SAC", "SanAntonioSpurs": "SAS", "TorontoRaptors": "TOR",
    "UtahJazz": "UTA", "WashingtonWizards": "WAS",
}
_lock = threading.Lock()
_last = [0.0]
INTERVAL = 0.25


def labels(day: pd.Timestamp) -> list[str]:
    d = day.strftime("%Y-%m-%d")
    out = []
    # format switched on 2025-12-22 (that day has both: 2 hourly files, then 15-min files)
    hourly, q15 = day <= pd.Timestamp("2025-12-22"), day >= pd.Timestamp("2025-12-22")
    for h in range(7, 24):
        hh, ap = (h - 12 if h > 12 else h), ("PM" if h >= 12 else "AM")
        if hourly:
            out.append(f"{d}_{hh:02d}{ap}")
        if q15:
            out += [f"{d}_{hh:02d}_{m:02d}{ap}" for m in (0, 15, 30, 45)]
    return out


def game_days() -> list[pd.Timestamp]:
    days = set()
    now = pd.Timestamp.now(tz="UTC")
    for y in (2026, 2027):
        g = pd.read_parquet(ROOT / "data" / "espn" / "nba" / f"games_{y}.parquet")
        g = g[g.season_type.isin(["regular", "post", "playin", "play-in"]) | ~g.season_type.isin(["preseason"])]
        g = g[g.season_type != "preseason"]
        g = g[g.start_utc <= now]
        days |= set(g.start_utc.dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None))
    return sorted(days)


def get(url: str):
    for a in range(6):
        with _lock:
            now = time.monotonic()
            slot = max(now, _last[0] + INTERVAL)
            _last[0] = slot
        if slot > now:
            time.sleep(slot - now)
        try:
            r = requests.get(url, headers=UA, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** a)
            continue
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** (a + 1))
            continue
        return r
    return None


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())


def parse(pdf: bytes) -> dict:
    import pdfplumber
    rows, header, team, matchup, gdate, gtime, notsub = [], None, None, None, None, None, []
    with pdfplumber.open(io.BytesIO(pdf)) as p:
        for pg in p.pages:
            words = pg.extract_words(x_tolerance=1.5)
            lines: dict[float, list] = {}
            for w in words:
                k = round(w["top"], 0)
                lines.setdefault(k, []).append(w)
            for top in sorted(lines):
                ws = sorted(lines[top], key=lambda w: w["x0"])
                txt = " ".join(w["text"] for w in ws)
                if txt.startswith("Injury Report:"):
                    header = txt.replace("Injury Report:", "").strip()
                    continue
                if txt.startswith("Game Date") or txt.startswith("Page"):
                    continue
                col = lambda lo, hi: "".join(w["text"] for w in ws if lo <= w["x0"] < hi)
                c_date, c_time, c_match = col(0, 115), col(115, 195), col(195, 260)
                c_team, c_player, c_status = col(260, 420), col(420, 580), col(580, 660)
                if re.fullmatch(r"\d\d/\d\d/\d{4}", c_date):
                    gdate = c_date
                if re.match(r"\d\d:\d\d", c_time):
                    gtime = c_time
                if re.fullmatch(r"[A-Z]{2,3}@[A-Z]{2,3}", c_match):
                    matchup = c_match
                if c_team:
                    if c_team in TEAMS:
                        team = TEAMS[c_team]
                    else:  # wrapped team name: try prefix match
                        cand = [v for k, v in TEAMS.items() if k.startswith(c_team) or c_team.startswith(k)]
                        if cand:
                            team = cand[0]
                if "NOTYETSUBMITTED" in (c_player + c_status + col(660, 2000)).replace(" ", "").upper():
                    notsub.append({"gdate": gdate, "matchup": matchup, "team": team})
                    continue
                if c_status in STATUSES and c_player:
                    rows.append({"gdate": gdate, "gtime": gtime, "matchup": matchup, "team": team,
                                 "player": c_player, "status": c_status})
    return {"header": header, "rows": rows, "notsub": notsub}


def one(label: str) -> str:
    f = REP / f"{label}.json"
    if f.exists():
        return "cached"
    r = get(f"{BASE}{label}.pdf")
    if r is None:
        return "neterr"
    rec = {"label": label, "http": r.status_code, "last_modified": r.headers.get("Last-Modified")}
    if r.status_code == 200:
        try:
            rec.update(parse(r.content))
        except Exception as e:  # keep going; record the failure
            rec["parse_error"] = repr(e)[:300]
    f.write_text(json.dumps(rec))
    return str(r.status_code)


def cmd_fetch() -> int:
    REP.mkdir(parents=True, exist_ok=True)
    todo = [lab for d in game_days() for lab in labels(d)]
    print(f"{len(todo)} report labels on {len(game_days())} game days", flush=True)
    n, t0, counts = 0, time.time(), {}
    with ThreadPoolExecutor(6) as ex:
        for s in ex.map(one, todo):
            counts[s] = counts.get(s, 0) + 1
            n += 1
            if n % 500 == 0:
                print(f"{n}/{len(todo)} {counts} {time.time() - t0:.0f}s", flush=True)
    print("done", counts, flush=True)
    return 0


def cmd_table() -> int:
    idx, rows = [], []
    for f in sorted(REP.glob("*.json")):
        r = json.loads(f.read_text())
        idx.append({"label": r["label"], "http": r["http"], "last_modified": r.get("last_modified"), "header": r.get("header"),
                    "n_rows": len(r.get("rows", [])), "n_notsub": len(r.get("notsub", [])), "parse_error": r.get("parse_error")})
        for x in r.get("rows", []):
            rows.append({"label": r["label"], **x})
        for x in r.get("notsub", []):
            rows.append({"label": r["label"], **x, "player": None, "status": "NOT_SUBMITTED"})
    idx = pd.DataFrame(idx)
    idx["lm"] = pd.to_datetime(idx.last_modified, utc=True, format="%a, %d %b %Y %H:%M:%S GMT", errors="coerce")
    idx["hdr"] = pd.to_datetime(idx.header, format="%m/%d/%y %I:%M %p", errors="coerce").dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    idx.to_parquet(OUTD / "report_index.parquet", index=False)
    rows = pd.DataFrame(rows)
    rows["pkey"] = rows.player.fillna("").map(lambda s: norm("".join(reversed(s.split(",", 1)))) if s else "")
    rows.to_parquet(OUTD / "report_rows.parquet", index=False)
    ok = idx[idx.http == 200]
    print(f"labels {len(idx)}  200: {len(ok)}  403/other: {(idx.http != 200).sum()}  parse errors: {idx.parse_error.notna().sum()}")
    print(f"rows {len(rows)}  header parsed {ok.hdr.notna().sum()}/{len(ok)}  last-modified {ok.lm.notna().sum()}/{len(ok)}")
    lag = (ok.lm - ok.hdr).dt.total_seconds() / 60
    print("Last-Modified minus PDF header time (min):", lag.describe().round(2).to_dict())
    return 0


if __name__ == "__main__":
    sys.exit({"fetch": cmd_fetch, "table": cmd_table}[sys.argv[1]]())
