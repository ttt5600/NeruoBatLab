"""Official free data APIs, keyless unless noted. Endpoints mirror what the fund already uses:
  FRED    fred.stlouisfed.org/graph/fredgraph.csv?id=...   (sports/scripts/lx1_arb_scan.py fed_upper_bound)
  Fed Reg federalregister.gov/api/v1/documents.json        (sports/results/count_nowcast/eo_fr.py params)
  BLS     api.bls.gov/publicAPI/v1 (keyless, 25 queries/day) or v2 with BLS_API_KEY  [new wrapper, no fund fetcher]
  BEA     apps.bea.gov/api/data (needs BEA_API_KEY, free registration) -- otherwise BEA series come via FRED mirrors
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone

from ..questions import Question
from . import Tool, ToolResult

# concept regex -> FRED series (id, description). First match per concept wins; at most 3 series per question.
FRED_MAP = [
    (r"10[- ]year treasury|10y treasury|ten-year (treasury|yield)", "DGS10", "10-year Treasury constant-maturity yield, %"),
    (r"2[- ]year treasury|two-year (treasury|yield)", "DGS2", "2-year Treasury yield, %"),
    (r"30[- ]year (fixed )?mortgage|mortgage rate", "MORTGAGE30US", "Freddie Mac 30-year fixed mortgage rate, %"),
    (r"federal funds|fed funds|fomc|target range", "DFEDTARU", "Fed funds target range upper limit, %"),
    (r"\bSOFR\b", "SOFR", "Secured overnight financing rate, %"),
    (r"unemployment rate", "UNRATE", "Unemployment rate (BLS via FRED), %"),
    (r"nonfarm payroll|payrolls|jobs report", "PAYEMS", "All employees, total nonfarm (thousands)"),
    (r"initial (jobless )?claims", "ICSA", "Initial jobless claims (weekly, SA)"),
    (r"\bCPI\b|consumer price index|inflation", "CPIAUCSL", "CPI-U all items index, SA"),
    (r"\bPCE\b|personal consumption expenditure", "PCEPI", "PCE price index"),
    (r"\bGDP\b|gross domestic product", "A191RL1Q225SBEA", "Real GDP, % change at annual rate (BEA via FRED)"),
    (r"S&P ?500|\bSPX\b", "SP500", "S&P 500 index close"),
    (r"\bVIX\b|volatility index", "VIXCLS", "CBOE VIX close"),
    (r"\bWTI\b|crude oil", "DCOILWTICO", "WTI crude spot, $/bbl"),
    (r"\bbrent\b", "DCOILBRENTEU", "Brent crude spot, $/bbl"),
    (r"euro.{0,12}dollar|EUR/?USD", "DEXUSEU", "USD per EUR"),
    (r"\byen\b|USD/?JPY", "DEXJPUS", "JPY per USD"),
    (r"bitcoin", "CBBTCUSD", "Coinbase bitcoin price, USD"),
    (r"retail sales", "RSAFS", "Advance retail sales, $M"),
    (r"housing starts", "HOUST", "Housing starts, thousands SAAR"),
]
FRED_EXPLICIT = re.compile(r"fred\.stlouisfed\.org/series/([A-Z0-9_]+)|FRED (?:series )?(?:ID )?\(?([A-Z][A-Z0-9_]{3,})\)?")


class FREDTool(Tool):
    name = "fred"
    supports_as_of = True
    URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    def series_for(self, q: Question) -> list[tuple[str, str]]:
        blob = q.text_blob()
        out: list[tuple[str, str]] = []
        for m in FRED_EXPLICIT.finditer(blob):
            sid = m.group(1) or m.group(2)
            if sid and sid not in [s for s, _ in out]:
                out.append((sid, "series named in the question"))
        for pat, sid, desc in FRED_MAP:
            if re.search(pat, blob, re.I) and sid not in [s for s, _ in out]:
                out.append((sid, desc))
        return out[:3]

    def matches(self, q: Question) -> bool:
        return bool(self.series_for(q))

    def run(self, q: Question, as_of: datetime | None = None) -> ToolResult:
        end = (as_of or datetime.now(timezone.utc)).date()
        lines, data = [], {}
        for sid, desc in self.series_for(q):
            txt = self.http_get(self.URL, {"id": sid, "cosd": (end - timedelta(days=500)).isoformat()}, as_json=False)
            rows = []
            for ln in txt.strip().splitlines()[1:]:
                parts = ln.split(",")
                if len(parts) < 2 or parts[1] in (".", ""):
                    continue
                d = parts[0]
                if as_of is not None and d >= end.isoformat():     # observation must predate as_of's day
                    continue
                rows.append((d, float(parts[1])))
            if not rows:
                continue
            data[sid] = rows[-24:]
            tail = rows[-8:]
            lines.append(f"FRED {sid} ({desc}), latest {rows[-1][0]} = {rows[-1][1]:g}; recent: "
                         + ", ".join(f"{d}: {v:g}" for d, v in tail))
        if not lines:
            return ToolResult(self.name, False, error="no FRED rows")
        return ToolResult(self.name, True, "\n".join(lines), data=data, asof=end.isoformat())


BLS_MAP = [
    (r"\bcore (CPI|inflation)\b", "CUSR0000SA0L1E", "CPI-U less food and energy, SA"),
    (r"\bCPI\b|consumer price index", "CUSR0000SA0", "CPI-U all items, SA"),
    (r"unemployment rate", "LNS14000000", "Unemployment rate, SA"),
    (r"nonfarm payroll|payrolls", "CES0000000001", "Total nonfarm employment, SA (thousands)"),
    (r"\bPPI\b|producer price", "WPSFD4", "PPI final demand, SA"),
    (r"average hourly earnings", "CES0500000003", "Average hourly earnings, private, SA"),
]


class BLSTool(Tool):
    name = "bls"
    supports_as_of = True

    def series_for(self, q: Question) -> list[tuple[str, str]]:
        blob = q.text_blob()
        if not re.search(r"\bBLS\b|Bureau of Labor Statistics|\bCPI\b|payroll|unemployment rate|\bPPI\b|hourly earnings", blob, re.I):
            return []
        out = []
        for pat, sid, desc in BLS_MAP:
            if re.search(pat, blob, re.I):
                out.append((sid, desc))
        return out[:2]

    def matches(self, q: Question) -> bool:
        return bool(self.series_for(q))

    def run(self, q: Question, as_of: datetime | None = None) -> ToolResult:
        key = os.environ.get("BLS_API_KEY")
        lines, data = [], {}
        for sid, desc in self.series_for(q):
            if key:
                j = self.http_get(f"https://api.bls.gov/publicAPI/v2/timeseries/data/{sid}", {"registrationkey": key})
            else:
                j = self.http_get(f"https://api.bls.gov/publicAPI/v1/timeseries/data/{sid}")
            if j.get("status") != "REQUEST_SUCCEEDED":
                lines.append(f"BLS {sid}: {j.get('status')} {j.get('message')}")
                continue
            obs = ((j.get("Results") or {}).get("series") or [{}])[0].get("data") or []
            rows = []
            for o in obs:
                per = o.get("period", "")
                if not per.startswith("M") or per == "M13":
                    continue
                ym = f"{o['year']}-{per[1:]}"
                rows.append((ym, float(o["value"])))
            rows.sort()
            if as_of is not None:     # a month is released ~2 weeks after it ends; keep months >= 45 days old
                cut = (as_of - timedelta(days=45)).strftime("%Y-%m")
                rows = [r for r in rows if r[0] <= cut]
            if not rows:
                continue
            data[sid] = rows[-24:]
            mom = (rows[-1][1] / rows[-2][1] - 1) * 100 if len(rows) >= 2 and rows[-2][1] else None
            yoy = (rows[-1][1] / rows[-13][1] - 1) * 100 if len(rows) >= 13 and rows[-13][1] else None
            extra = (f"; m/m {mom:+.2f}%" if mom is not None else "") + (f"; y/y {yoy:+.2f}%" if yoy is not None else "")
            lines.append(f"BLS {sid} ({desc}), latest {rows[-1][0]} = {rows[-1][1]:g}{extra}; recent: "
                         + ", ".join(f"{d}: {v:g}" for d, v in rows[-6:]))
        return ToolResult(self.name, bool(data), "\n".join(lines), data=data,
                          error=None if data else "no BLS rows")


class BEATool(Tool):
    name = "bea"
    patterns = [r"\bBEA\b|Bureau of Economic Analysis", r"\bGDP\b|gross domestic product"]
    supports_as_of = True

    def matches(self, q: Question) -> bool:
        return bool(os.environ.get("BEA_API_KEY")) and super().matches(q)

    def run(self, q: Question, as_of: datetime | None = None) -> ToolResult:
        end = (as_of or datetime.now(timezone.utc))
        years = ",".join(str(y) for y in range(end.year - 2, end.year + 1))
        j = self.http_get("https://apps.bea.gov/api/data/", {
            "UserID": os.environ["BEA_API_KEY"], "method": "GetData", "DataSetName": "NIPA", "TableName": "T10101",
            "Frequency": "Q", "Year": years, "ResultFormat": "JSON"})
        rows = [(r["TimePeriod"], float(str(r["DataValue"]).replace(",", ""))) for r in
                ((j.get("BEAAPI") or {}).get("Results") or {}).get("Data", []) if str(r.get("LineNumber")) == "1"]
        rows.sort()
        if as_of is not None:   # a quarter's advance estimate comes ~4 weeks after it ends; keep quarters ending >= 40 days ago
            def q_end(tp: str) -> datetime:
                y, qq = int(tp[:4]), int(tp[-1])
                m = qq * 3
                return datetime(y + (m // 12), (m % 12) + 1, 1, tzinfo=timezone.utc)
            rows = [r for r in rows if q_end(r[0]) + timedelta(days=40) <= as_of]
        if not rows:
            return ToolResult(self.name, False, error="no BEA rows")
        txt = "BEA NIPA T10101 line 1, real GDP % change SAAR: " + ", ".join(f"{p}: {v:+.1f}" for p, v in rows[-6:])
        return ToolResult(self.name, True, txt, data={"gdp": rows[-8:]}, asof=rows[-1][0])


class FederalRegisterTool(Tool):
    name = "federal_register"
    patterns = [r"executive orders?", r"Federal Register", r"presidential (memorand|proclamation)", r"\bproclamations?\b"]
    supports_as_of = True
    URL = "https://www.federalregister.gov/api/v1/documents.json"

    def run(self, q: Question, as_of: datetime | None = None) -> ToolResult:
        end = (as_of or datetime.now(timezone.utc)).date()
        start = end - timedelta(days=120)
        params = {"conditions[presidential_document_type][]": "executive_order", "conditions[type][]": "PRESDOCU",
                  "conditions[signing_date][gte]": start.isoformat(), "per_page": 200, "order": "newest",
                  "fields[]": ["executive_order_number", "signing_date", "publication_date", "title"]}
        j = self.http_get(self.URL, params)
        docs = [d for d in (j.get("results") or []) if not as_of or (d.get("publication_date") or "9999") < end.isoformat()]
        if not docs:
            return ToolResult(self.name, True, f"Federal Register: no executive orders published {start}..{end}.",
                              data={"n": 0}, asof=end.isoformat())
        by_week: dict[str, int] = {}
        for d in docs:
            sd = datetime.fromisoformat(d["signing_date"]).date()
            wk = (sd - timedelta(days=sd.weekday())).isoformat()
            by_week[wk] = by_week.get(wk, 0) + 1
        lines = [f"Federal Register executive orders signed since {start} (published before {end}): {len(docs)} total.",
                 "  by signing week (Mon): " + ", ".join(f"{w}: {n}" for w, n in sorted(by_week.items())[-10:]),
                 "  latest: " + "; ".join(f"EO {d.get('executive_order_number')} signed {d.get('signing_date')}: "
                                         f"{(d.get('title') or '')[:80]}" for d in docs[:5])]
        return ToolResult(self.name, True, "\n".join(lines), data={"n": len(docs), "by_week": by_week},
                          asof=end.isoformat())
