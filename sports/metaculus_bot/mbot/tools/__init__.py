"""Free official-data research tools. Each tool decides from the question text whether it applies, fetches with
read-only public GETs (honest User-Agent), and returns a short text summary for the prompt plus structured data
for the log. A tool failure never stops a forecast: it is logged and the pipeline continues without it.

Backtest safety: when `as_of` is given (scoring a resolved question), tools that cannot cut their data at `as_of`
are skipped, so today's values never leak into a past question. Tools that can cut still return the *latest
vintage* of each observation (FRED/BLS revisions), a small residual lookahead noted in the README.

A tool may set ToolResult.resolves (True/False) only when the official number that decides the question is
already published; that is the one case where the binary clip widens. No shipped tool sets it yet (needs a
per-question threshold parser) -- see README "What is stubbed".
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import requests

from ..config import USER_AGENT
from ..questions import Question


@dataclass
class ToolResult:
    tool: str
    ok: bool
    summary: str = ""
    data: dict = field(default_factory=dict)
    asof: str | None = None
    resolves: bool | None = None
    error: str | None = None
    seconds: float = 0.0

    def log_row(self) -> dict:
        return {"tool": self.tool, "ok": self.ok, "asof": self.asof, "resolves": self.resolves,
                "error": self.error, "seconds": round(self.seconds, 2), "summary": self.summary[:1500]}


def default_http_get(url: str, params: dict | None = None, timeout: float = 30.0, as_json: bool = True):
    r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    r.raise_for_status()
    return r.json() if as_json else r.text


class Tool:
    name = "tool"
    patterns: list[str] = []
    supports_as_of = False

    def __init__(self, http_get: Callable | None = None):
        self.http_get = http_get or default_http_get
        self._rx = [re.compile(p, re.I) for p in self.patterns]

    def matches(self, q: Question) -> bool:
        blob = q.text_blob()
        return any(rx.search(blob) for rx in self._rx)

    def run(self, q: Question, as_of: datetime | None = None) -> ToolResult:  # pragma: no cover - interface
        raise NotImplementedError


def run_tools(q: Question, tools: list[Tool], as_of: datetime | None = None) -> list[ToolResult]:
    out = []
    for t in tools:
        if not t.matches(q):
            continue
        if as_of is not None and not t.supports_as_of:
            out.append(ToolResult(t.name, False, error="skipped in backtest: cannot cut data at as_of"))
            continue
        t0 = time.time()
        try:
            r = t.run(q, as_of)
        except Exception as e:  # noqa: BLE001 -- a broken source must never cost a forecast
            r = ToolResult(t.name, False, error=f"{type(e).__name__}: {str(e)[:200]}")
        r.seconds = time.time() - t0
        out.append(r)
    return out


def default_tools(cfg=None, http_get: Callable | None = None) -> list[Tool]:
    from .fund import AAAGasTool, NOAAGlobalTool, OpenRouterTool, TSATool, WeatherTool
    from .official import BEATool, BLSTool, FederalRegisterTool, FREDTool
    scripts_dir = getattr(cfg, "fund_scripts_dir", None)
    return [AAAGasTool(scripts_dir, http_get), WeatherTool(scripts_dir, http_get), TSATool(scripts_dir, http_get),
            OpenRouterTool(scripts_dir, http_get), NOAAGlobalTool(scripts_dir, http_get),
            FREDTool(http_get), BLSTool(http_get), BEATool(http_get), FederalRegisterTool(http_get)]
