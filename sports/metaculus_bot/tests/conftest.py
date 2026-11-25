import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

BOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOT))
FIX = BOT / "tests" / "fixtures"

NOW = datetime(2026, 11, 30, 13, 0, tzinfo=timezone.utc)   # one hour after the fixtures open


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Any real HTTP call in a test is a bug: block requests and urllib at the socket-free level."""
    import requests

    def boom(*a, **k):
        raise AssertionError(f"network call attempted in a test: {a[:2]}")
    monkeypatch.setattr(requests.Session, "request", boom)
    monkeypatch.setattr(requests, "get", boom)
    monkeypatch.setattr(requests, "post", boom)
    for k in ("METACULUS_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "ASKNEWS_CLIENT_ID",
              "ASKNEWS_SECRET", "BEA_API_KEY", "BLS_API_KEY", "MBOT_MODELS", "MBOT_TOURNAMENTS", "MBOT_STATE_DIR"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def open_posts():
    return json.loads((FIX / "open_posts.json").read_text())["results"]


@pytest.fixture
def llm_out():
    return json.loads((FIX / "llm_outputs.json").read_text())


@pytest.fixture
def questions(open_posts):
    from mbot.questions import questions_from_post
    qs = {}
    for p in open_posts:
        for q in questions_from_post(p):
            qs[q.question_id] = q
    return qs


@pytest.fixture
def cfg(tmp_path):
    from mbot.config import Config
    c = Config()
    c.state_dir = tmp_path / "state"
    c.n_samples = 3
    c.use_data_tools = False
    c.monthly_cap_usd = 50.0
    return c
