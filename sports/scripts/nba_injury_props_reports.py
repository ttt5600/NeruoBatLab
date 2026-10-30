"""Official NBA injury reports for the DEV prop seasons 2022-23 .. 2024-25 (news side only, no prices/outcomes).

    ../.venv_quant/bin/python scripts/nba_injury_props_reports.py fetch   # resumable, <= 2 req/s, cache data/nba_injury_props/reports/
    ../.venv_quant/bin/python scripts/nba_injury_props_reports.py table   # -> data/nba_injury_props/report_rows.parquet, report_index.parquet

Same source as Q011 (ak-static.cms.nba.com/referee/injury/Injury-Report_YYYY-MM-DD_HHAM|PM.pdf, hourly era). The
2022-25 PDFs use a different column layout than 2025-26, so columns are located from the header row
("Game Date | Game Time | Matchup | Team | Player Name | Current Status | Reason") on each page.
Labels fetched: 07AM 09AM 11AM 01PM 03PM 05PM 07PM 09PM 11PM ET on every ET game day and the day before it.
Publication time = max(HTTP Last-Modified, PDF header time) (Q011 rule).
"""
from __future__ import annotations

import io
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q011_reports import STATUSES, TEAMS, norm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTD = ROOT / "data" / "nba_injury_props"
REP = OUTD / "reports"
BASE = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_"
UA = {"User-Agent": "Mozilla/5.0 (research; sports backtest)"}
SEASONS = [2023, 2024, 2025]
HOURS = [7, 9, 11, 13, 15, 17, 19, 21, 23]
INTERVAL = 0.5  # <= 2 req/s
_lock = threading.Lock()
_last = [0.0]


def game_days() -> list[pd.Timestamp]:
    days = set()
    for s in SEASONS:
        g = pd.read_parquet(ROOT / "data" / "espn" / "nba" / f"games_{s}.parquet")
        g = g[(g.season == s) & (g.season_type != "preseason")]
        days |= set(pd.to_datetime(g.start_utc, utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None))
    return sorted(days)


def labels() -> list[str]:
    gd = game_days()
    days = sorted(set(gd) | {d - pd.Timedelta(days=1) for d in gd})
    out = []
    for d in days:
        for h in HOURS:
            hh, ap = (h - 12 if h > 12 else h), ("PM" if h >= 12 else "AM")
            out.append(f"{d:%Y-%m-%d}_{hh:02d}{ap}")
    return out


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
            time.sleep(2 ** (a + 2))
            continue
        return r
    return None


def parse(pdf: bytes) -> dict:
    import pdfplumber
    rows, header, team, matchup, gdate, gtime, notsub = [], None, None, None, None, None, []
    bounds = None
    with pdfplumber.open(io.BytesIO(pdf)) as p:
        for pg in p.pages:
            words = pg.extract_words(x_tolerance=1.5)
            lines: dict[float, list] = {}
            for w in words:
                lines.setdefault(round(w["top"], 0), []).append(w)
            for top in sorted(lines):
                ws = sorted(lines[top], key=lambda w: w["x0"])
                txt = " ".join(w["text"] for w in ws)
                if txt.startswith("Injury Report:"):
                    header = txt.replace("Injury Report:", "").strip()
                    continue
                if txt.startswith("Game Date"):
                    x = {w["text"]: w["x0"] for w in ws}
                    try:
                        xs = [x["Time"] - 25, x["Matchup"] - 3, x["Team"] - 3, x["Player"] - 3, x["Current"] - 3, x["Reason"] - 3]
                    except KeyError:
                        continue
                    # Game Time column header "Game Time": the 2nd "Game" word starts the column
                    g2 = [w["x0"] for w in ws if w["text"] == "Game"]
                    if len(g2) >= 2:
                        xs[0] = g2[1] - 3
                    bounds = xs
                    continue
                if txt.startswith("Page") or bounds is None:
                    continue
                b = bounds
                col = lambda lo, hi: "".join(w["text"] for w in ws if lo <= w["x0"] < hi)
                c_date, c_time, c_match = col(0, b[0]), col(b[0], b[1]), col(b[1], b[2])
                c_team, c_player, c_status = col(b[2], b[3]), col(b[3], b[4]), col(b[4], b[5])
                if re.fullmatch(r"\d\d/\d\d/\d{4}", c_date):
                    gdate = c_date
                if re.match(r"\d\d:\d\d", c_time):
                    gtime = c_time
                if re.fullmatch(r"[A-Z]{2,3}@[A-Z]{2,3}", c_match):
                    matchup = c_match
                if c_team:
                    if c_team in TEAMS:
                        team = TEAMS[c_team]
                    else:
                        cand = [v for k, v in TEAMS.items() if k.startswith(c_team) or c_team.startswith(k)]
                        if cand:
                            team = cand[0]
                if "NOTYETSUBMITTED" in (c_player + c_status + col(b[5], 5000)).replace(" ", "").upper():
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
        except Exception as e:
            rec["parse_error"] = repr(e)[:300]
    f.write_text(json.dumps(rec))
    return str(r.status_code)


def cmd_fetch(shard: str | None = None) -> int:
    """shard 'i/n': this process takes labels[i::n] and paces at n x INTERVAL (n processes together <= 2 req/s;
    one process is parse-bound at ~0.4 labels/s)."""
    global INTERVAL
    REP.mkdir(parents=True, exist_ok=True)
    todo = labels()
    if shard:
        i, n = map(int, shard.split("/"))
        todo, INTERVAL = todo[i::n], INTERVAL * n
    print(f"{len(todo)} labels", flush=True)
    n, t0, counts = 0, time.time(), {}
    with ThreadPoolExecutor(3) as ex:
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
    idx["pub"] = idx[["lm", "hdr"]].max(axis=1)
    idx.to_parquet(OUTD / "report_index.parquet", index=False)
    rows = pd.DataFrame(rows)
    rows["pkey"] = rows.player.fillna("").map(lambda s: norm("".join(reversed(s.split(",", 1)))) if s else "")
    rows.to_parquet(OUTD / "report_rows.parquet", index=False)
    ok = idx[idx.http == 200]
    print(f"labels {len(idx)}  200: {len(ok)}  other: {(idx.http != 200).sum()}  parse errors: {idx.parse_error.notna().sum()}"
          f"  200 with 0 rows: {(ok.n_rows == 0).sum()}")
    print(f"rows {len(rows)}  header parsed {ok.hdr.notna().sum()}/{len(ok)}  last-modified {ok.lm.notna().sum()}/{len(ok)}")
    lag = (ok.lm - ok.hdr).dt.total_seconds() / 60
    print("Last-Modified minus PDF header time (min):", lag.describe().round(2).to_dict())
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "fetch":
        sys.exit(cmd_fetch(sys.argv[2] if len(sys.argv) > 2 else None))
    sys.exit(cmd_table())
