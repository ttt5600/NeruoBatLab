"""Pluggable LLM providers + cost accounting.

Providers (MBOT_MODELS = comma-separated "provider:model" entries; samples cycle through them):
  mock:<name>          offline stub for tests / dry runs (cost 0). Never allowed to submit.
  anthropic:<model>    Anthropic Messages API through the official `anthropic` SDK (ANTHROPIC_API_KEY).
  openrouter:<model>   OpenAI-compatible chat completions at openrouter.ai (OPENROUTER_API_KEY; the Metaculus credits
                       form hands out OpenRouter keys).
  openai:<model>       OpenAI-compatible chat completions (OPENAI_API_KEY, OPENAI_BASE_URL optional).
  metaculus:<model>    Metaculus LLM proxy credits, authenticated with METACULUS_TOKEN. Endpoints copied from
                       forecasting-tools 0.3.7 general_llm.py: Anthropic-format at
                       https://llm-proxy.metaculus.com/proxy/anthropic, OpenAI-format at .../proxy/openai/v1.
                       UNVERIFIED: whether Metaculus still issues proxy credits this season.

Prices: Anthropic first-party list prices (claude-api skill table, cached 2026-10-06) and an OpenRouter snapshot
(prices_snapshot.json, public GET of openrouter.ai/api/v1/models on 2026-11-25). Unknown models are priced at a
deliberately high default so the spend cap errs on the safe side.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import requests

from .config import USER_AGENT

# USD per 1M tokens (input, output)
ANTHROPIC_PRICES = {
    "claude-fable-5-1": (10.0, 50.0), "claude-fable-5": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0), "claude-opus-5": (5.0, 25.0), "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0), "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0), "claude-sonnet-5": (2.0, 10.0), "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-5-5": (0.10, 0.50), "claude-haiku-4-5": (1.0, 5.0),
}
UNKNOWN_PRICE = (10.0, 50.0)
_SNAPSHOT = Path(__file__).with_name("prices_snapshot.json")
_OR_PRICES: dict[str, tuple[float, float]] | None = None

# models that take output_config.effort (Claude 4.6+ generation); older ones 400 on it
_EFFORT_MODELS = re.compile(r"claude-(fable|mythos|opus|sonnet|haiku)-(5|4-[6-9])")
# server-side refusal fallback (`fallbacks: "default"`, beta server-side-fallback-2026-07-01), Claude API only
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}


class LLMError(RuntimeError):
    pass


class LLMRefusal(LLMError):
    pass


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    stop_reason: str | None = None
    extra: dict = field(default_factory=dict)


def openrouter_prices() -> dict[str, tuple[float, float]]:
    global _OR_PRICES
    if _OR_PRICES is None:
        try:
            raw = json.loads(_SNAPSHOT.read_text())["models"]
            _OR_PRICES = {k: (float(v[0]), float(v[1])) for k, v in raw.items()}
        except (OSError, KeyError, ValueError):
            _OR_PRICES = {}
    return _OR_PRICES


def price_for(provider: str, model: str) -> tuple[float, float]:
    if provider == "mock":
        return (0.0, 0.0)
    m = model.split("/")[-1] if provider in ("anthropic", "metaculus") else model
    if m in ANTHROPIC_PRICES:
        return ANTHROPIC_PRICES[m]
    orp = openrouter_prices()
    if model in orp:
        return orp[model]
    dotted = re.sub(r"-(\d)-(\d)$", r"-\1.\2", m)          # claude-opus-5-5 <-> anthropic/claude-opus-5.5
    for k, v in orp.items():
        if k.endswith("/" + m) or k.endswith("/" + dotted):
            return v
    return UNKNOWN_PRICE


def cost_usd(provider: str, model: str, tokens_in: int, tokens_out: int) -> float:
    pin, pout = price_for(provider, model)
    return (tokens_in * pin + tokens_out * pout) / 1e6


def estimate_tokens(text: str) -> int:
    return int(len(text) / 3.0) + 50      # conservative: ~3 chars/token


class BaseLLM:
    provider = "base"

    def __init__(self, model: str, max_tokens: int = 16000, effort: str | None = "high"):
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    def worst_case_cost(self, prompt: str) -> float:
        """Upper bound used by the spend guard BEFORE the call: full prompt + every allowed output token."""
        return cost_usd(self.provider, self.model, estimate_tokens(prompt), self.max_tokens)

    def complete(self, prompt: str, system: str | None = None) -> LLMResponse:  # pragma: no cover - interface
        raise NotImplementedError


class MockLLM(BaseLLM):
    """Offline stub. `responder(prompt, call_index) -> text`, or a list of texts used in rotation."""
    provider = "mock"

    def __init__(self, model: str = "mock", responder: Callable[[str, int], str] | list[str] | None = None,
                 fake_cost_usd: float = 0.0, max_tokens: int = 2000, fail_on: set[int] | None = None):
        super().__init__(model, max_tokens=max_tokens, effort=None)
        self.responder = responder
        self.fake_cost_usd = fake_cost_usd
        self.calls: list[str] = []
        self.fail_on = fail_on or set()
        self._lock = threading.Lock()

    def worst_case_cost(self, prompt: str) -> float:
        return self.fake_cost_usd

    def complete(self, prompt: str, system: str | None = None) -> LLMResponse:
        with self._lock:
            i = len(self.calls)
            self.calls.append(prompt)
        if i in self.fail_on:
            raise LLMError(f"mock failure on call {i}")
        if callable(self.responder):
            text = self.responder(prompt, i)
        elif isinstance(self.responder, list) and self.responder:
            text = self.responder[i % len(self.responder)]
        else:
            text = default_mock_answer(prompt)
        return LLMResponse(text=text, model=self.model, provider="mock", input_tokens=estimate_tokens(prompt),
                           output_tokens=estimate_tokens(text), cost_usd=self.fake_cost_usd, stop_reason="end_turn")


def default_mock_answer(prompt: str) -> str:
    """A format-correct placeholder answer for dry runs without any key. Uninformative on purpose."""
    if "FORMAT: BINARY" in prompt:
        return "Mock reasoning (no LLM called).\nProbability: 50%"
    if "FORMAT: MULTIPLE_CHOICE" in prompt:
        opts = re.findall(r"^- OPTION: (.+)$", prompt, flags=re.M)
        share = round(100.0 / max(len(opts), 1), 2)
        return "Mock reasoning (no LLM called).\n" + "\n".join(f"{o}: {share}%" for o in opts)
    if "FORMAT: CONTINUOUS" in prompt:
        lo = re.search(r"^LOW_HINT: (.+)$", prompt, flags=re.M)
        hi = re.search(r"^HIGH_HINT: (.+)$", prompt, flags=re.M)
        if lo and hi:
            a, b = lo.group(1).strip(), hi.group(1).strip()
            if re.match(r"\d{4}-\d{2}-\d{2}", a):
                from datetime import datetime
                da, db = datetime.fromisoformat(a), datetime.fromisoformat(b)
                vals = [(da + (db - da) * f).date().isoformat() for f in (0.2, 0.3, 0.45, 0.55, 0.7, 0.8)]
            else:
                fa, fb = float(a), float(b)
                vals = [repr(round(fa + (fb - fa) * f, 6)) for f in (0.2, 0.3, 0.45, 0.55, 0.7, 0.8)]
            return "Mock reasoning (no LLM called).\n" + "\n".join(
                f"Percentile {p}: {v}" for p, v in zip((10, 20, 40, 60, 80, 90), vals))
    return "Probability: 50%"


class AnthropicLLM(BaseLLM):
    """Claude through the official `anthropic` Python SDK (optional dependency, imported lazily)."""
    provider = "anthropic"

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None,
                 extra_headers: dict | None = None, max_tokens: int = 16000, effort: str | None = "high",
                 use_fallbacks: bool = True, client=None, provider_label: str = "anthropic"):
        super().__init__(model, max_tokens=max_tokens, effort=effort)
        self.provider = provider_label
        self._client = client
        self._api_key, self._base_url, self._extra_headers = api_key, base_url, extra_headers
        self.use_fallbacks = use_fallbacks and base_url is None and model in _FALLBACK_MODELS

    def _get_client(self):
        if self._client is None:
            import anthropic  # optional dependency: `pip install anthropic`
            kw = {"max_retries": 2, "timeout": 600.0}
            if self._api_key:
                kw["api_key"] = self._api_key
            if self._base_url:
                kw["base_url"] = self._base_url
            if self._extra_headers:
                kw["default_headers"] = self._extra_headers
            self._client = anthropic.Anthropic(**kw)
        return self._client

    def request_kwargs(self, prompt: str, system: str | None = None) -> dict:
        kw: dict = {"model": self.model, "max_tokens": self.max_tokens,
                    "messages": [{"role": "user", "content": prompt}]}
        if system:
            kw["system"] = system
        if self.effort and _EFFORT_MODELS.search(self.model):
            kw["output_config"] = {"effort": self.effort}
        if self.use_fallbacks:
            kw["betas"] = ["server-side-fallback-2026-07-01"]
            kw["fallbacks"] = "default"
        return kw   # no temperature/top_p: rejected on current API versions

    def complete(self, prompt: str, system: str | None = None) -> LLMResponse:
        client = self._get_client()
        kw = self.request_kwargs(prompt, system)

        def call(kw):   # streamed (long thinking at high effort can outlast a plain request's timeout)
            api = client.beta.messages if "betas" in kw else client.messages
            with api.stream(**kw) as stream:
                return stream.get_final_message()
        try:
            resp = call(kw)
        except Exception as e:  # noqa: BLE001 -- downgrade once if the beta/fallback params are refused
            # 400 = API refused the beta; TypeError = an older SDK without the `fallbacks` kwarg (raised before sending)
            if "betas" in kw and (getattr(e, "status_code", None) == 400 or isinstance(e, TypeError)):
                self.use_fallbacks = False
                kw = self.request_kwargs(prompt, system)
                resp = call(kw)
            else:
                raise
        if getattr(resp, "stop_reason", None) == "refusal":
            raise LLMRefusal(f"refusal ({getattr(getattr(resp, 'stop_details', None), 'category', None)})")
        text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
        u = resp.usage
        tin = int(getattr(u, "input_tokens", 0) or 0)
        tcache_r = int(getattr(u, "cache_read_input_tokens", 0) or 0)
        tcache_w = int(getattr(u, "cache_creation_input_tokens", 0) or 0)
        tout = int(getattr(u, "output_tokens", 0) or 0)
        served = getattr(resp, "model", None) or self.model
        pin, pout = price_for(self.provider, served)
        cost = (tin * pin + tcache_w * pin * 1.25 + tcache_r * pin * 0.1 + tout * pout) / 1e6
        return LLMResponse(text=text, model=served, provider=self.provider, input_tokens=tin + tcache_r + tcache_w,
                           output_tokens=tout, cost_usd=cost, stop_reason=getattr(resp, "stop_reason", None))


class OpenAICompatLLM(BaseLLM):
    """OpenAI-compatible /chat/completions over plain HTTP (OpenRouter, OpenAI, Metaculus proxy)."""

    def __init__(self, model: str, api_key: str, base_url: str, provider_label: str = "openrouter",
                 auth_scheme: str = "Bearer", max_tokens: int = 16000, effort: str | None = "high",
                 http_post: Callable | None = None, timeout: float = 300.0):
        super().__init__(model, max_tokens=max_tokens, effort=effort)
        self.provider = provider_label
        self.base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"{auth_scheme} {api_key}", "Content-Type": "application/json",
                         "User-Agent": USER_AGENT}
        if provider_label == "openrouter":
            self._headers["X-Title"] = "mbot"
        self._post = http_post or requests.post
        self.timeout = timeout

    def body(self, prompt: str, system: str | None = None) -> dict:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        b: dict = {"model": self.model, "messages": msgs}
        if "api.openai.com" in self.base_url:
            b["max_completion_tokens"] = self.max_tokens
        else:
            b["max_tokens"] = self.max_tokens
        if self.effort and self.provider == "openrouter":
            b["reasoning"] = {"effort": self.effort if self.effort in ("low", "medium", "high") else "high"}
        return b

    def complete(self, prompt: str, system: str | None = None) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        body = self.body(prompt, system)
        last = None
        for attempt in range(3):
            try:
                r = self._post(url, json=body, headers=self._headers, timeout=self.timeout)
            except requests.RequestException as e:
                last = e
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                last = LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
                time.sleep(3 * (attempt + 1))
                continue
            if r.status_code >= 400:
                raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
            j = r.json()
            choice = (j.get("choices") or [{}])[0]
            text = ((choice.get("message") or {}).get("content")) or ""
            if isinstance(text, list):
                text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
            u = j.get("usage") or {}
            tin, tout = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
            reported = u.get("cost")
            cost = float(reported) if isinstance(reported, (int, float)) else cost_usd(self.provider, self.model, tin, tout)
            if choice.get("finish_reason") in ("content_filter",):
                raise LLMRefusal("content_filter")
            return LLMResponse(text=text, model=j.get("model") or self.model, provider=self.provider,
                               input_tokens=tin, output_tokens=tout, cost_usd=cost,
                               stop_reason=choice.get("finish_reason"))
        raise LLMError(f"gave up after retries: {last}")


METACULUS_PROXY_ANTHROPIC = "https://llm-proxy.metaculus.com/proxy/anthropic"
METACULUS_PROXY_OPENAI = "https://llm-proxy.metaculus.com/proxy/openai/v1"


def build_llm(entry: str, max_tokens: int = 16000, effort: str | None = "high") -> BaseLLM:
    """'provider:model' -> provider instance. Raises LLMError when the needed key is missing."""
    if ":" not in entry:
        raise LLMError(f"model entry {entry!r} must look like provider:model")
    provider, model = entry.split(":", 1)
    provider = provider.strip().lower()
    model = model.strip()
    env = os.environ.get
    if provider == "mock":
        return MockLLM(model or "mock")
    if provider == "anthropic":
        if not env("ANTHROPIC_API_KEY"):
            raise LLMError("ANTHROPIC_API_KEY not set")
        return AnthropicLLM(model, max_tokens=max_tokens, effort=effort)
    if provider == "openrouter":
        if not env("OPENROUTER_API_KEY"):
            raise LLMError("OPENROUTER_API_KEY not set")
        return OpenAICompatLLM(model, env("OPENROUTER_API_KEY"), "https://openrouter.ai/api/v1", "openrouter",
                               max_tokens=max_tokens, effort=effort)
    if provider == "openai":
        if not env("OPENAI_API_KEY"):
            raise LLMError("OPENAI_API_KEY not set")
        return OpenAICompatLLM(model, env("OPENAI_API_KEY"), env("OPENAI_BASE_URL") or "https://api.openai.com/v1",
                               "openai", max_tokens=max_tokens, effort=effort)
    if provider == "metaculus":
        tok = env("METACULUS_TOKEN")
        if not tok:
            raise LLMError("METACULUS_TOKEN not set (needed for the Metaculus proxy)")
        if "claude" in model or "anthropic" in model:
            return AnthropicLLM(model.split("/")[-1], api_key=tok, base_url=METACULUS_PROXY_ANTHROPIC,
                                extra_headers={"Authorization": f"Token {tok}"}, max_tokens=max_tokens,
                                effort=effort, provider_label="metaculus")
        return OpenAICompatLLM(model, tok, METACULUS_PROXY_OPENAI, "metaculus", auth_scheme="Token",
                               max_tokens=max_tokens, effort=None)
    raise LLMError(f"unknown provider {provider!r}")


def build_ensemble(spec: str, max_tokens: int = 16000, effort: str | None = "high") -> list[BaseLLM]:
    entries = [e.strip() for e in spec.split(",") if e.strip()]
    if not entries:
        raise LLMError("MBOT_MODELS is empty")
    return [build_llm(e, max_tokens=max_tokens, effort=effort) for e in entries]
