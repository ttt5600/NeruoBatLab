"""Research tools against stub fund scripts and fake HTTP (no network)."""
import textwrap
from datetime import datetime, timezone

import pytest

from mbot.config import SPORTS_DIR
from mbot.questions import Question
from mbot.research import AskNews, Researcher
from mbot.tools import run_tools
from mbot.tools.fund import AAAGasTool, NOAAGlobalTool, TSATool, WeatherTool, load_fund_module
from mbot.tools.official import FederalRegisterTool, FREDTool


def _q(title, qid=1, typ="binary"):
    return Question(post_id=qid, question_id=qid, title=title, type=typ, status="open")


@pytest.fixture
def stub_scripts(tmp_path):
    d = tmp_path / "scripts"
    d.mkdir()
    (d / "source_shift_watch.py").write_text(textwrap.dedent('''
        print("module import side effect")          # must be swallowed / harmless
        def aaa(state):
            print("ALERT should be swallowed")
            # shape copied from the live page on 2026-11-25: 6 scraped headers (stray trailing "Price"), 5 values per row
            return {"asof": "2026-11-25", "decimals": 4,
                    "grades": ["Regular", "Mid-Grade", "Premium", "Diesel", "E85", "Price"],
                    "rows": {"Current": ["3.5866", "4.1249", "4.5141", "5.7645", "2.6765"],
                             "Week Ago": ["3.7502", "4.2844", "4.6759", "5.9594", "2.8560"]}}
        def noaa(state):
            return {"latest": "202610", "last24": {"202607": 1.21, "202608": 1.18, "202609": 1.25, "202610": 1.30}}
        def parse_cli(text, issued):
            return {"date": "2026-11-24", "max": 51, "min": 40, "final": True, "valid": None, "issued": issued}
        def openrouter(state):
            return {"chart_default": {"cachedAt": "2026-11-25"}, "top_model": "x"}
    '''))
    (d / "e1x_tsa_logger.py").write_text(textwrap.dedent('''
        import pandas as pd
        def daily():
            idx = pd.date_range("2025-10-01", "2026-11-24", freq="D")
            return pd.Series([2_000_000.0 + 1000 * i for i in range(len(idx))], index=idx)
    '''))
    return d


def test_aaa_tool_uses_fund_fetcher(stub_scripts, questions, capsys):
    t = AAAGasTool(stub_scripts)
    assert t.matches(questions[8001])
    r = t.run(questions[8001])
    assert r.ok and r.asof == "2026-11-25"
    assert "Current: Regular $3.5866, Mid-Grade $4.1249" in r.summary and "E85 $2.6765" in r.summary
    assert "Price $" not in r.summary                     # the stray header must not shift the labels
    assert "ALERT" not in capsys.readouterr().out


def test_tsa_tool_cuts_at_as_of(stub_scripts):
    pytest.importorskip("pandas")
    t = TSATool(stub_scripts)
    q = _q("Will TSA checkpoint throughput exceed 2.5 million on Dec 1?")
    assert t.matches(q)
    live = t.run(q)
    assert live.ok and live.asof == "2026-11-24" and "vs same weekdays 364 days earlier" in live.summary
    assert live.data["yoy_pct"] == pytest.approx((2_000_000 + 1000 * 416) / (2_000_000 + 1000 * 52) * 100 - 100, abs=0.05)
    past = t.run(q, as_of=datetime(2026, 6, 1, 12, tzinfo=timezone.utc))
    assert past.ok and past.asof == "2026-05-31"          # nothing on/after the as_of day leaks in
    assert past.data["yoy_pct"] is None                   # stub history too short for a year-ago comparison


def test_noaa_tool_cuts_at_as_of(stub_scripts):
    q = _q("Will October 2026 be the warmest on record per NOAA NCEI global temperature?")
    # a month counts once as_of >= its first day + 45 days (NOAA posts a month ~2 weeks after it ends)
    r = NOAAGlobalTool(stub_scripts).run(q, as_of=datetime(2026, 10, 10, tzinfo=timezone.utc))
    assert r.ok and r.asof == "202608"                    # September not yet published on Oct 10
    r = NOAAGlobalTool(stub_scripts).run(q, as_of=datetime(2026, 10, 20, tzinfo=timezone.utc))
    assert r.ok and r.asof == "202609"


def test_missing_fund_dir_never_breaks_a_forecast(tmp_path, questions):
    res = run_tools(questions[8001], [AAAGasTool(tmp_path / "nope")])
    assert len(res) == 1 and not res[0].ok and "not found" in res[0].error


def test_weather_needs_a_city_and_is_skipped_in_backtest(stub_scripts):
    t = WeatherTool(stub_scripts, http_get=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no fetch")))
    assert not t.matches(_q("Will the global temperature anomaly exceed 1.3C?"))
    q = _q("Will the high temperature in Central Park, New York exceed 60 degrees F on Dec 3?")
    assert t.matches(q)
    res = run_tools(q, [t], as_of=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert not res[0].ok and "skipped in backtest" in res[0].error


def test_weather_tool_with_fake_nws(stub_scripts):
    calls = []

    def fake_get(url, params=None, **kw):
        calls.append(url)
        if url.endswith("/stations/KNYC"):
            return {"geometry": {"coordinates": [-73.97, 40.78]}}
        if "/points/" in url:
            return {"properties": {"forecast": "https://api.weather.gov/gridpoints/OKX/33,37/forecast"}}
        if url.endswith("/forecast"):
            return {"properties": {"updateTime": "2026-11-25T10:00:00Z",
                                   "periods": [{"name": "Today", "temperature": 55, "temperatureUnit": "F", "shortForecast": "Sunny"}]}}
        if "/locations/NYC" in url:
            return {"@graph": [{"id": "abc", "issuanceTime": "2026-11-25T06:00:00Z"}]}
        if url.endswith("/products/abc"):
            return {"productText": "CLIMATE SUMMARY ..."}
        raise AssertionError(url)
    r = WeatherTool(stub_scripts, http_get=fake_get).run(_q("High temperature in New York City on Dec 3?"))
    assert r.ok and "Today: 55°F, Sunny" in r.summary and "max 51 min 40 (final)" in r.summary


def test_fred_csv_and_as_of_filter():
    csv = "observation_date,DGS10\n2026-11-20,4.10\n2026-11-21,.\n2026-11-23,4.15\n2026-11-24,4.18\n"
    seen = []

    def fake_get(url, params=None, as_json=True):
        seen.append((url, params, as_json))
        return csv
    t = FREDTool(fake_get)
    q = _q("What will the 10-year Treasury yield (FRED DGS10) be on Dec 10?")
    assert [s for s, _ in t.series_for(q)] == ["DGS10"]
    r = t.run(q, as_of=datetime(2026, 11, 24, 9, tzinfo=timezone.utc))
    assert r.ok and r.data["DGS10"][-1] == ("2026-11-23", 4.15)    # the as_of day itself is excluded
    assert seen[0][2] is False and seen[0][1]["id"] == "DGS10"
    assert FREDTool(fake_get).run(q).data["DGS10"][-1] == ("2026-11-24", 4.18)


def test_federal_register_counts_and_as_of():
    docs = {"results": [
        {"executive_order_number": 14400, "signing_date": "2026-11-20", "publication_date": "2026-11-25", "title": "B"},
        {"executive_order_number": 14399, "signing_date": "2026-11-17", "publication_date": "2026-11-20", "title": "A"},
        {"executive_order_number": 14398, "signing_date": "2026-11-10", "publication_date": "2026-11-13", "title": "Z"}]}
    t = FederalRegisterTool(lambda url, params=None, **k: docs)
    q = _q("How many executive orders will be signed between December 1 and December 14, 2026?")
    assert t.matches(q)
    assert t.run(q).data["n"] == 3
    r = t.run(q, as_of=datetime(2026, 11, 22, tzinfo=timezone.utc))
    assert r.data["n"] == 2 and r.data["by_week"] == {"2026-11-16": 1, "2026-11-09": 1}   # Monday-start weeks


class FakeNews:
    def __init__(self):
        self.n = 0

    def search(self, query):
        self.n += 1
        return "NEWS " + query


def test_researcher_news_off_in_backtest_and_truncates():
    news = FakeNews()
    big = type("Big", (FREDTool,), {"run": lambda self, q, as_of=None: __import__("mbot.tools", fromlist=["ToolResult"]).ToolResult(
        "fred", True, "x" * 50_000, asof="2026")})(lambda *a, **k: "")
    q = _q("What will the 10-year Treasury yield be?")
    r = Researcher(tools=[big], news=news, max_chars=1000, use_news=True).run(q)
    assert news.n == 1 and r.news_status == "ok" and len(r.text) < 1100 and r.text.endswith("truncated]")
    r = Researcher(tools=[], news=news, use_news=True).run(q, as_of=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert news.n == 1 and r.news_status == "skipped (backtest)"
    assert AskNews.available() is False                   # conftest removed the keys


def test_asknews_wrapper_with_fake_sdk():
    class N:
        def search_news(self, **kw):
            assert kw["return_type"] == "both" and kw["strategy"] in ("latest news", "news knowledge")
            return type("R", (), {"as_string": f"articles for {kw['query']} ({kw['strategy']})"})()
    a = AskNews(client=type("C", (), {"news": N()})())
    out = a.search("Example")
    assert "latest news" in out and "news knowledge" in out


@pytest.mark.skipif(not (SPORTS_DIR / "scripts" / "source_shift_watch.py").exists(), reason="fund scripts absent")
def test_real_fund_scripts_import_without_network():
    ssw = load_fund_module("source_shift_watch")
    for fn in ("aaa", "parse_cli", "openrouter", "noaa"):
        assert callable(getattr(ssw, fn))
    sample = ("\n000\nCDUS41 KOKX 250600\nCLINYC\n\nCLIMATE REPORT\nNATIONAL WEATHER SERVICE NEW YORK NY\n"
              "120 AM EST TUE NOV 25 2026\n\n...THE CENTRAL PARK NY CLIMATE SUMMARY FOR NOVEMBER 24 2026...\n\n"
              "TEMPERATURE (F)\n YESTERDAY\n  MAXIMUM         51    249 PM\n  MINIMUM         40    633 AM\n")
    c = ssw.parse_cli(sample, "2026-11-25T06:20:00+00:00")
    assert c["date"] == "2026-11-24" and c["max"] == 51 and c["min"] == 40 and c["final"] is True
    tsa = load_fund_module("e1x_tsa_logger")
    assert callable(tsa.daily)
