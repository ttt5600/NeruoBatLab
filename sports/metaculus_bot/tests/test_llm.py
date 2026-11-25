"""Provider adapters, tested with fake clients only: no network, no key, no spend."""
from types import SimpleNamespace as NS

import pytest

from mbot.llm import (AnthropicLLM, LLMError, LLMRefusal, MockLLM, OpenAICompatLLM, build_ensemble, build_llm,
                      cost_usd, price_for)


class FakeMessages:
    def __init__(self, resp, fail_first=None):
        self.resp, self.fail_first, self.calls = resp, fail_first, []

    def stream(self, **kw):
        self.calls.append(kw)
        if self.fail_first is not None:
            e, self.fail_first = self.fail_first, None
            raise e
        resp = self.resp

        class _S:
            def __enter__(s):
                return s

            def __exit__(s, *a):
                return False

            def get_final_message(s):
                return resp
        return _S()


def _anthropic_resp(text="Probability: 61%", stop="end_turn", model="claude-opus-5-5"):
    return NS(content=[NS(type="thinking", thinking="..."), NS(type="text", text=text)], stop_reason=stop, model=model,
              usage=NS(input_tokens=1000, output_tokens=2000, cache_read_input_tokens=0, cache_creation_input_tokens=0),
              stop_details=None)


def test_anthropic_request_shape_and_cost():
    beta = FakeMessages(_anthropic_resp())
    client = NS(messages=FakeMessages(None), beta=NS(messages=beta))
    llm = AnthropicLLM("claude-opus-5-5", client=client, max_tokens=16000, effort="high")
    r = llm.complete("q?", system="sys")
    kw = beta.calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["output_config"] == {"effort": "high"} and kw["system"] == "sys"
    assert kw["betas"] == ["server-side-fallback-2026-07-01"] and kw["fallbacks"] == "default"
    assert "temperature" not in kw and "thinking" not in kw
    assert r.text == "Probability: 61%"
    assert r.cost_usd == pytest.approx((1000 * 4 + 2000 * 20) / 1e6)       # $4 / $20 per MTok


def test_anthropic_downgrades_when_beta_refused():
    err = Exception("bad beta")
    err.status_code = 400
    plain = FakeMessages(_anthropic_resp())
    client = NS(messages=plain, beta=NS(messages=FakeMessages(None, fail_first=err)))
    llm = AnthropicLLM("claude-opus-5-5", client=client)
    llm.complete("q")
    assert "betas" not in plain.calls[0] and llm.use_fallbacks is False


def test_anthropic_downgrades_on_old_sdk_typeerror():
    plain = FakeMessages(_anthropic_resp())
    client = NS(messages=plain, beta=NS(messages=FakeMessages(None, fail_first=TypeError("unexpected kwarg 'fallbacks'"))))
    r = AnthropicLLM("claude-opus-5-5", client=client).complete("q")
    assert r.text == "Probability: 61%" and "fallbacks" not in plain.calls[0]


def test_anthropic_refusal_and_no_fallback_on_proxy():
    client = NS(messages=FakeMessages(_anthropic_resp(text="", stop="refusal")), beta=NS(messages=None))
    llm = AnthropicLLM("claude-opus-5-5", client=client, base_url="https://llm-proxy.metaculus.com/proxy/anthropic")
    assert llm.use_fallbacks is False          # proxies get plain requests
    with pytest.raises(LLMRefusal):
        llm.complete("q")


def test_haiku_has_no_fallback_but_has_effort():
    llm = AnthropicLLM("claude-haiku-5-5", client=object())
    kw = llm.request_kwargs("q")
    assert "betas" not in kw and kw["output_config"] == {"effort": "high"}


class FakeHTTP:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def __call__(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        status, body = self.responses.pop(0)
        return NS(status_code=status, json=lambda: body, text=str(body))


def test_openrouter_body_retry_and_reported_cost(monkeypatch):
    monkeypatch.setattr("mbot.llm.time.sleep", lambda s: None)
    ok = {"model": "openai/gpt-6.2-sol", "choices": [{"message": {"content": "Probability: 30%"}, "finish_reason": "stop"}],
          "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.0123}}
    http = FakeHTTP([(429, "slow down"), (200, ok)])
    llm = OpenAICompatLLM("openai/gpt-6.2-sol", "k", "https://openrouter.ai/api/v1", "openrouter", http_post=http)
    r = llm.complete("q", system="s")
    assert len(http.calls) == 2 and http.calls[0]["url"].endswith("/chat/completions")
    b = http.calls[0]["json"]
    assert b["max_tokens"] == 16000 and b["reasoning"] == {"effort": "high"} and b["messages"][0]["role"] == "system"
    assert http.calls[0]["headers"]["Authorization"] == "Bearer k"
    assert r.cost_usd == 0.0123 and r.text == "Probability: 30%"


def test_openai_uses_max_completion_tokens_and_4xx_raises():
    http = FakeHTTP([(401, {"error": "bad key"})])
    llm = OpenAICompatLLM("gpt-6.2-sol", "k", "https://api.openai.com/v1", "openai", http_post=http)
    with pytest.raises(LLMError):
        llm.complete("q")
    assert "max_completion_tokens" in http.calls[0]["json"] and "reasoning" not in http.calls[0]["json"]


def test_build_llm_key_gating(monkeypatch):
    with pytest.raises(LLMError):
        build_llm("anthropic:claude-opus-5-5")
    with pytest.raises(LLMError):
        build_llm("openrouter:openai/gpt-6.2-sol")
    with pytest.raises(LLMError):
        build_llm("nonsense")
    monkeypatch.setenv("METACULUS_TOKEN", "t")
    m = build_llm("metaculus:anthropic/claude-opus-5-5")
    assert isinstance(m, AnthropicLLM) and m.provider == "metaculus" and m.use_fallbacks is False
    o = build_llm("metaculus:openai/gpt-6.2-sol")
    assert isinstance(o, OpenAICompatLLM) and o._headers["Authorization"] == "Token t"
    ens = build_ensemble("mock:a, mock:b")
    assert [e.name for e in ens] == ["mock:a", "mock:b"]


def test_prices():
    assert price_for("anthropic", "claude-opus-5-5") == (4.0, 20.0)
    assert price_for("mock", "x") == (0.0, 0.0)
    assert price_for("openrouter", "anthropic/claude-haiku-5.5") == (0.1, 0.5)
    assert cost_usd("anthropic", "claude-sonnet-5-5", 1_000_000, 0) == pytest.approx(2.0)
    # an unknown model is priced pessimistically so the cap still binds
    assert price_for("anthropic", "claude-unknown-9") == (10.0, 50.0)


def test_mock_failures_and_rotation():
    m = MockLLM(responder=["Probability: 10%", "Probability: 20%"], fail_on={1})
    assert m.complete("x").text == "Probability: 10%"
    with pytest.raises(LLMError):
        m.complete("x")
    assert m.complete("x").text == "Probability: 10%"


def _sse(model="claude-opus-5-5", text="Probability: 33%"):
    import json as _j
    ev = [
        ("message_start", {"type": "message_start", "message": {
            "id": "msg_1", "type": "message", "role": "assistant", "model": model, "content": [],
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 1200, "output_tokens": 1, "cache_read_input_tokens": 0,
                      "cache_creation_input_tokens": 0}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                           "usage": {"output_tokens": 900}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {e}\ndata: {_j.dumps(d)}\n\n" for e, d in ev).encode()


def test_real_sdk_request_on_the_wire():
    """The installed `anthropic` SDK builds the request; an httpx MockTransport answers. Nothing leaves the process."""
    anthropic = pytest.importorskip("anthropic")
    try:                       # anthropic 2.x ships on `httpx2`; 0.x/1.x on `httpx`
        import httpx2 as httpx
    except ImportError:
        httpx = pytest.importorskip("httpx")
    import json as _j
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_sse())
    client = anthropic.Anthropic(api_key="sk-ant-test-not-real", max_retries=0,
                                 http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    llm = AnthropicLLM("claude-opus-5-5", client=client, max_tokens=16000, effort="high")
    r = llm.complete("Will X happen?", system="sys")
    body = _j.loads(seen[0].content)
    assert seen[0].url.path == "/v1/messages" and "server-side-fallback-2026-07-01" in seen[0].headers["anthropic-beta"]
    assert body["fallbacks"] == "default" and body["output_config"] == {"effort": "high"} and body["stream"] is True
    assert body["model"] == "claude-opus-5-5" and body["max_tokens"] == 16000 and "temperature" not in body
    assert r.text == "Probability: 33%" and r.input_tokens == 1200 and r.output_tokens == 900
    assert r.cost_usd == pytest.approx((1200 * 4 + 900 * 20) / 1e6)
