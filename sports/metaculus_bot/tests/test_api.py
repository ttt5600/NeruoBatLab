"""Metaculus REST client: request shapes (docs/openapi.yml), pagination, GET retry, POST never retried."""
import pytest

from mbot.metaculus_api import MetaculusAPI, MetaculusError, forecast_payload


class Rec:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def __call__(self, method, url, params=None, json=None):
        self.calls.append({"method": method, "url": url, "params": params, "json": json})
        return self.responses.pop(0)


def api(rec):
    return MetaculusAPI("tok", http=rec, min_interval_s=0, sleep=lambda s: None)


def test_list_params_and_pagination():
    page = lambda n, nxt: (200, {"results": [{"id": i} for i in range(n)], "next": nxt})  # noqa: E731
    rec = Rec([page(2, "x"), page(1, None)])
    posts = api(rec).list_posts("minibench", page_size=2)
    assert len(posts) == 3 and len(rec.calls) == 2
    p = rec.calls[0]["params"]
    assert p["tournaments"] == ["minibench"] and p["statuses"] == ["open"]
    assert "group_of_questions" in p["forecast_type"] and "binary" in p["forecast_type"]
    assert p["include_descriptions"] == "true" and rec.calls[1]["params"]["offset"] == 2


def test_get_retries_then_raises_on_4xx():
    rec = Rec([(429, "slow"), (503, "down"), (200, {"id": 5})])
    assert api(rec).get_post(5) == {"id": 5} and len(rec.calls) == 3
    with pytest.raises(MetaculusError) as e:
        api(Rec([(403, {"detail": "auth"})])).get_post(5)
    assert e.value.status == 403


def test_post_is_never_retried_and_payload_shape():
    rec = Rec([(502, "bad gateway")])
    with pytest.raises(MetaculusError):
        api(rec).post_forecast(11, forecast_payload("binary", 0.3))
    assert len(rec.calls) == 1
    body = rec.calls[0]["json"]
    assert body == [{"question": 11, "source": "api", "probability_yes": 0.3, "probability_yes_per_category": None,
                     "continuous_cdf": None}]
    rec = Rec([(201, {"id": 1})])
    api(rec).post_comment(7, "why")
    assert rec.calls[0]["json"] == {"text": "why", "parent": None, "included_forecast": True, "is_private": True,
                                    "on_post": 7}
    assert forecast_payload("multiple_choice", {"a": 0.5, "b": 0.5})["probability_yes_per_category"] == {"a": 0.5, "b": 0.5}
    with pytest.raises(ValueError):
        forecast_payload("conditional", 1)
