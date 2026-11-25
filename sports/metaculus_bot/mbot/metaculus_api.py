"""Metaculus REST client -- the official tournament-submission path, same endpoints and payloads as Metaculus'
metac-bot-template (main_with_no_framework.py, commit bcbd016b) and forecasting-tools 0.3.7 MetaculusClient:

  GET  /api/posts/?tournaments=<slug>&statuses=open&forecast_type=...   list open questions (needs a token)
  GET  /api/posts/<post_id>/                                           one post (re-check before posting)
  POST /api/questions/forecast/   [{"question": qid, "source": "api", probability_yes | probability_yes_per_category
                                    | continuous_cdf}]
  POST /api/comments/create/      {"text", "parent": null, "included_forecast": true, "is_private": true, "on_post"}

Every request carries `Authorization: Token <METACULUS_TOKEN>` (unauthenticated requests get 403 since 2026). The HTTP
layer is injectable so tests run on recorded fixtures without network.
"""
from __future__ import annotations

import time
from typing import Callable

import requests

from .config import API_BASE_URL, SUPPORTED_TYPES, USER_AGENT


class MetaculusError(RuntimeError):
    def __init__(self, msg: str, status: int | None = None):
        super().__init__(msg)
        self.status = status


def _requests_http(token: str | None, timeout: float = 60.0) -> Callable:
    sess = requests.Session()
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en"}
    if token:
        headers["Authorization"] = f"Token {token}"

    def http(method: str, url: str, params=None, json=None):
        r = sess.request(method, url, params=params, json=json, headers=headers, timeout=timeout)
        try:
            body = r.json()
        except ValueError:
            body = r.text
        return r.status_code, body
    return http


class MetaculusAPI:
    def __init__(self, token: str | None, base_url: str = API_BASE_URL, http: Callable | None = None,
                 min_interval_s: float = 1.0, sleep: Callable[[float], None] = time.sleep):
        self.base = base_url.rstrip("/")
        self.http = http or _requests_http(token)
        self.min_interval = min_interval_s
        self.sleep = sleep
        self._last = 0.0
        self.has_token = bool(token) or http is not None

    def _pace(self):
        dt = time.time() - self._last
        if dt < self.min_interval:
            self.sleep(self.min_interval - dt)
        self._last = time.time()

    def _get(self, path: str, params: dict | None = None, retries: int = 4) -> dict:
        last = None
        for attempt in range(retries):
            self._pace()
            try:
                status, body = self.http("GET", f"{self.base}{path}", params=params)
            except requests.RequestException as e:
                last = MetaculusError(f"GET {path}: {e}")
                self.sleep(min(2 ** attempt * 2, 30))
                continue
            if status == 429 or status >= 500:
                last = MetaculusError(f"GET {path}: HTTP {status}", status)
                self.sleep(min(2 ** attempt * 3, 60))
                continue
            if status >= 400:
                raise MetaculusError(f"GET {path}: HTTP {status}: {str(body)[:300]}", status)
            return body
        raise last or MetaculusError(f"GET {path} failed")

    def _post(self, path: str, payload) -> dict | str:
        """POSTs are NOT retried here: a retry after an ambiguous failure could double-submit. The runner re-reads
        the post and decides."""
        self._pace()
        status, body = self.http("POST", f"{self.base}{path}", json=payload)
        if status >= 400:
            raise MetaculusError(f"POST {path}: HTTP {status}: {str(body)[:300]}", status)
        return body

    # ------------------------------------------------------------------ reads
    def list_posts(self, tournament: str | int, statuses: str = "open", page_size: int = 50, max_pages: int = 20,
                   extra: dict | None = None) -> list[dict]:
        out: list[dict] = []
        for page in range(max_pages):
            # docs/openapi.yml: tournaments / statuses / forecast_type are arrays (repeated params) and the flag is
            # `include_descriptions`; the template's no-framework path sends the singular `include_description`, so
            # both are sent. group_of_questions is requested too: groups are unpacked into their sub-questions
            # (forecasting-tools does the same in "unpack_subquestions" mode).
            params = {"tournaments": [tournament],
                      "statuses": [statuses] if isinstance(statuses, str) else list(statuses),
                      "limit": page_size, "offset": page * page_size,
                      "forecast_type": [*SUPPORTED_TYPES, "group_of_questions"],
                      "include_descriptions": "true", "include_description": "true", "order_by": "-published_at"}
            if extra:
                params.update(extra)
            body = self._get("/posts/", params)
            res = body.get("results") or []
            out.extend(res)
            if len(res) < page_size or not body.get("next"):
                break
        return out

    def list_open_posts(self, tournament: str | int) -> list[dict]:
        return self.list_posts(tournament, "open")

    def get_post(self, post_id: int) -> dict:
        return self._get(f"/posts/{post_id}/")

    # ------------------------------------------------------------------ writes
    def post_forecast(self, question_id: int, payload: dict) -> dict | str:
        return self._post("/questions/forecast/", [{"question": int(question_id), "source": "api", **payload}])

    def post_comment(self, post_id: int, text: str) -> dict | str:
        return self._post("/comments/create/", {"text": text, "parent": None, "included_forecast": True,
                                                "is_private": True, "on_post": int(post_id)})


def forecast_payload(family: str, value) -> dict:
    """Payload body for /questions/forecast/ (template create_forecast_payload)."""
    if family == "binary":
        return {"probability_yes": float(value), "probability_yes_per_category": None, "continuous_cdf": None}
    if family == "multiple_choice":
        return {"probability_yes": None, "probability_yes_per_category": {k: float(v) for k, v in value.items()},
                "continuous_cdf": None}
    if family == "continuous":
        return {"probability_yes": None, "probability_yes_per_category": None, "continuous_cdf": [float(x) for x in value]}
    raise ValueError(family)
