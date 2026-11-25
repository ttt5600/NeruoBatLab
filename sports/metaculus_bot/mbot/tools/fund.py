"""Wrappers around the fund's existing read-only nowcast fetchers (sports/scripts/*.py), imported by file path so
the fund code is reused as-is and never modified:

  source_shift_watch.aaa(state)       AAA national averages page (gasprices.aaa.com), all grades
  source_shift_watch.parse_cli(...)   NWS CLI climate-report parser (daily max/min, final vs partial)
  source_shift_watch.openrouter(st)   OpenRouter rankings frontend endpoints (token chart, default/organic/all views)
  source_shift_watch.noaa(state)      NOAA NCEI global land+ocean monthly anomaly CSV
  e1x_tsa_logger.daily()              TSA checkpoint throughput: fund CSV history + Kalshi KXTSAD first prints
                                      (tsa.gov itself returns 403 to scripts; the fund does not work around that)

Those functions take a scratch `state` dict (their revision/alert bookkeeping); we pass a fresh dict and swallow
their ALERT prints. If the scripts directory is missing (e.g. a CI runner without the fund repo), the tools report
'unavailable' and the forecast continues without them.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..config import SPORTS_DIR
from ..questions import Question
from . import Tool, ToolResult


class FundUnavailable(RuntimeError):
    pass


def load_fund_module(name: str, scripts_dir: str | Path | None = None):
    scripts = Path(scripts_dir) if scripts_dir else SPORTS_DIR / "scripts"
    path = scripts / f"{name}.py"
    if not path.exists():
        raise FundUnavailable(f"{path} not found")
    key = f"_mbot_fund_{name}_{abs(hash(str(path.resolve())))}"
    if key in sys.modules:
        return sys.modules[key]
    if str(scripts) not in sys.path:      # e1x_tsa_logger imports its sibling e1_family_scan
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(key, None)
        raise
    return mod


def _quiet(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*a, **kw)


class _FundTool(Tool):
    module = "source_shift_watch"

    def __init__(self, scripts_dir=None, http_get=None):
        super().__init__(http_get)
        self.scripts_dir = scripts_dir

    def mod(self):
        return load_fund_module(self.module, self.scripts_dir)


# ---------------------------------------------------------------------------------------------------------- AAA gas
class AAAGasTool(_FundTool):
    name = "aaa_gas"
    patterns = [r"\bAAA\b", r"\bgas(oline)? price", r"price of (regular )?(gas|gasoline)", r"\bdiesel\b.*\bprice"]

    def run(self, q: Question, as_of: datetime | None = None) -> ToolResult:
        res = _quiet(self.mod().aaa, {})
        if not isinstance(res, dict) or res.get("error"):
            return ToolResult(self.name, False, error=f"aaa fetch: {res.get('error') if isinstance(res, dict) else res}")
        rows = res.get("rows") or {}
        grades = [g for g in (res.get("grades") or []) if g.strip()]
        lines = [f"AAA national average prices (USD/gal), page 'Price as of' {res.get('asof')}:"]
        # The fund's header scrape returns the grade columns FIRST plus a stray header from another table
        # (live page 2026-11-25: Regular, Mid-Grade, Premium, Diesel, E85, Price; 5 values per row). Align from the
        # left, as the fund itself does (it reads Regular as row[0]).
        for label, vals in rows.items():
            g = grades[:len(vals)] if len(grades) >= len(vals) else [f"grade{i}" for i in range(len(vals))]
            lines.append(f"  {label}: " + ", ".join(f"{gn} ${v}" for gn, v in zip(g, vals)))
        return ToolResult(self.name, True, "\n".join(lines), data=res, asof=res.get("asof"))


# ---------------------------------------------------------------------------------------------------------- weather
# city regex -> (NWS station, CLI location). First seven = the fund's E4 cities (sports/sports/wx.py CITIES).
CITY_STATIONS = {
    r"new york|nyc|central park|manhattan": ("KNYC", "NYC"),
    r"chicago|midway": ("KMDW", "MDW"),
    r"miami": ("KMIA", "MIA"),
    r"austin": ("KAUS", "AUS"),
    r"denver": ("KDEN", "DEN"),
    r"los angeles|\bL\.?A\.?\b": ("KLAX", "LAX"),
    r"philadelphia": ("KPHL", "PHL"),
    r"boston": ("KBOS", "BOS"),
    r"seattle": ("KSEA", "SEA"),
    r"san francisco": ("KSFO", "SFO"),
    r"washington,? d\.?c|reagan national": ("KDCA", "DCA"),
    r"atlanta": ("KATL", "ATL"),
    r"houston": ("KIAH", "IAH"),
    r"dallas": ("KDFW", "DFW"),
    r"phoenix": ("KPHX", "PHX"),
    r"las vegas": ("KLAS", "LAS"),
    r"minneapolis": ("KMSP", "MSP"),
    r"new orleans": ("KMSY", "MSY"),
}


class WeatherTool(_FundTool):
    name = "nws_weather"
    patterns = [r"temperature", r"\bdegrees\b|°\s?F|fahrenheit", r"\bsnow(fall)?\b", r"\brain(fall)?\b|precipitation",
                r"\bNWS\b|National Weather Service|heat index|wind chill"]
    NWS = "https://api.weather.gov"

    def matches(self, q: Question) -> bool:
        return super().matches(q) and bool(self._cities(q))

    def _cities(self, q: Question) -> list[tuple[str, str, str]]:
        import re
        blob = q.text_blob()
        return [(pat, st, cli) for pat, (st, cli) in CITY_STATIONS.items() if re.search(pat, blob, re.I)][:2]

    def run(self, q: Question, as_of: datetime | None = None) -> ToolResult:
        parse_cli = self.mod().parse_cli
        lines, data = [], {}
        for pat, station, cli in self._cities(q):
            st = self.http_get(f"{self.NWS}/stations/{station}")
            lon, lat = st["geometry"]["coordinates"][:2]
            pt = self.http_get(f"{self.NWS}/points/{lat:.4f},{lon:.4f}")
            fc = self.http_get(pt["properties"]["forecast"])
            periods = (fc.get("properties") or {}).get("periods") or []
            lines.append(f"NWS forecast for {station} ({cli}), issued {fc.get('properties', {}).get('updateTime', '?')}:")
            for p in periods[:8]:
                lines.append(f"  {p.get('name')}: {p.get('temperature')}°{p.get('temperatureUnit')}, {p.get('shortForecast')}")
            lst = self.http_get(f"{self.NWS}/products/types/CLI/locations/{cli}")
            reps = []
            for prod in (lst.get("@graph") or [])[:3]:
                txt = self.http_get(f"{self.NWS}/products/{prod['id']}")
                if txt.get("productText"):
                    reps.append(parse_cli(txt["productText"], prod.get("issuanceTime")))
            for c in reps:
                lines.append(f"  NWS CLI {cli} {c.get('date')}: max {c.get('max')} min {c.get('min')} "
                             f"({'final' if c.get('final') else 'partial: ' + str(c.get('valid'))})")
            data[station] = {"forecast": periods[:8], "cli": reps}
        return ToolResult(self.name, bool(lines), "\n".join(lines), data=data,
                          asof=datetime.now(timezone.utc).isoformat(timespec="minutes"))


# ---------------------------------------------------------------------------------------------------------- TSA
class TSATool(_FundTool):
    name = "tsa_throughput"
    module = "e1x_tsa_logger"
    patterns = [r"\bTSA\b", r"checkpoint (travel|throughput|numbers)", r"passengers? screened"]
    supports_as_of = True

    def run(self, q: Question, as_of: datetime | None = None) -> ToolResult:
        s = _quiet(self.mod().daily)         # pandas Series: date -> travelers screened
        if as_of is not None:
            s = s[s.index < as_of.replace(tzinfo=None) - timedelta(hours=12)]
        if len(s) < 14:
            return ToolResult(self.name, False, error="too little TSA history")
        last = s.iloc[-10:]
        t7 = float(s.iloc[-7:].mean())
        ly_idx = [d - timedelta(days=364) for d in s.index[-7:]]
        ly = [float(s.loc[d]) for d in ly_idx if d in s.index]
        yoy = (t7 / (sum(ly) / len(ly)) - 1) * 100 if len(ly) == 7 else None
        lines = ["TSA checkpoint travel numbers (fund series: CSV history + Kalshi KXTSAD first prints):"]
        lines += [f"  {d.date()} ({d.strftime('%a')}): {int(v):,}" for d, v in last.items()]
        lines.append(f"  trailing 7-day mean {t7:,.0f}" + (f"; vs same weekdays 364 days earlier {yoy:+.1f}%" if yoy is not None else ""))
        return ToolResult(self.name, True, "\n".join(lines), data={"last": {str(d.date()): float(v) for d, v in last.items()},
                          "t7": t7, "yoy_pct": yoy}, asof=str(s.index[-1].date()))


# ---------------------------------------------------------------------------------------------------------- OpenRouter
class OpenRouterTool(_FundTool):
    name = "openrouter_rankings"
    patterns = [r"openrouter"]

    def run(self, q: Question, as_of: datetime | None = None) -> ToolResult:
        res = _quiet(self.mod().openrouter, {})
        keep = {k: res.get(k) for k in ("chart_default", "chart_organic", "chart_all", "default_matches", "top_model",
                                        "organic_over_all")}
        txt = "OpenRouter rankings (frontend endpoints; tokens in trillions, weekly bars; default/organic/all views):\n"
        txt += json.dumps(keep, default=str)[:2500]
        return ToolResult(self.name, True, txt, data=keep, asof=(keep.get("chart_default") or {}).get("cachedAt"))


# ---------------------------------------------------------------------------------------------------------- NOAA
class NOAAGlobalTool(_FundTool):
    name = "noaa_global_temp"
    patterns = [r"global (surface |mean )?temperature", r"\bNCEI\b", r"warmest (year|month|on record)",
                r"temperature anomaly"]
    supports_as_of = True

    def run(self, q: Question, as_of: datetime | None = None) -> ToolResult:
        res = _quiet(self.mod().noaa, {})
        last = res.get("last24") or {}
        if as_of is not None:   # a month's value is published ~2 weeks after it ends: keep months >= 45 days old
            cut = (as_of - timedelta(days=45)).strftime("%Y%m")
            last = {k: v for k, v in last.items() if k <= cut}
        if not last:
            return ToolResult(self.name, False, error=str(res.get("error") or "no rows"))
        lines = ["NOAA NCEI global land+ocean temperature anomaly (degC vs 1901-2000), latest months:"]
        lines += [f"  {k[:4]}-{k[4:]}: {v:+.2f}" for k, v in sorted(last.items())[-12:]]
        return ToolResult(self.name, True, "\n".join(lines), data={"last": last}, asof=max(last))
