"""Research step: AskNews (free bot-maker tier, only when ASKNEWS_CLIENT_ID/ASKNEWS_SECRET exist) + the free
official-data tools. Returns the text handed to every sample plus structured rows for the ledger."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime

from .questions import Question
from .tools import Tool, ToolResult, run_tools


@dataclass
class Research:
    text: str
    tool_results: list[ToolResult] = field(default_factory=list)
    news_status: str = "off"

    @property
    def resolves(self) -> bool | None:
        """True/False only if a data tool says the deciding number is already published (never the LLM)."""
        vals = {r.resolves for r in self.tool_results if r.ok and r.resolves is not None}
        return vals.pop() if len(vals) == 1 else None

    def log_rows(self) -> list[dict]:
        return [r.log_row() for r in self.tool_results]


class AskNews:
    """Thin wrapper over the official `asknews` SDK (optional dependency). Two searches, as in the Metaculus template
    README: latest news + the 60-day news-knowledge archive."""

    def __init__(self, client=None):
        self._client = client

    @staticmethod
    def available() -> bool:
        return bool(os.environ.get("ASKNEWS_CLIENT_ID") and os.environ.get("ASKNEWS_SECRET"))

    def _get(self):
        if self._client is None:
            from asknews_sdk import AskNewsSDK  # optional: pip install asknews
            self._client = AskNewsSDK(client_id=os.environ["ASKNEWS_CLIENT_ID"],
                                      client_secret=os.environ["ASKNEWS_SECRET"], scopes=["news"])
        return self._client

    def search(self, query: str) -> str:
        c = self._get()
        hot = c.news.search_news(query=query, n_articles=6, return_type="both", strategy="latest news")
        hist = c.news.search_news(query=query, n_articles=6, return_type="both", strategy="news knowledge")
        return f"[AskNews latest]\n{getattr(hot, 'as_string', hot)}\n\n[AskNews archive]\n{getattr(hist, 'as_string', hist)}"


class Researcher:
    def __init__(self, tools: list[Tool] | None = None, news: AskNews | None = None, max_chars: int = 9000,
                 use_news: bool | None = None):
        self.tools = tools or []
        self.news = news or AskNews()
        self.use_news = AskNews.available() if use_news is None else use_news
        self.max_chars = max_chars

    def run(self, q: Question, as_of: datetime | None = None) -> Research:
        parts: list[str] = []
        results = run_tools(q, self.tools, as_of=as_of)
        for r in results:
            if r.ok and r.summary:
                parts.append(f"[data: {r.tool}, as of {r.asof}]\n{r.summary}")
        news_status = "off"
        if self.use_news and as_of is None:     # live news would leak the outcome into a backtest
            try:
                parts.append(self.news.search(q.title)[:6000])
                news_status = "ok"
            except Exception as e:  # noqa: BLE001
                news_status = f"error: {type(e).__name__}: {str(e)[:120]}"
        elif self.use_news:
            news_status = "skipped (backtest)"
        text = "\n\n".join(parts)
        if len(text) > self.max_chars:
            text = text[: self.max_chars] + "\n[... research truncated]"
        return Research(text=text, tool_results=results, news_status=news_status)
