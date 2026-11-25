"""Question model built from the Metaculus /api/posts/ JSON, plus the type router.

Field names follow docs/openapi.yml of github.com/Metaculus/metaculus (read 2026-11-25). The gateway sets hidden
fields to null rather than deleting keys, so every read tolerates None.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

CONTINUOUS_TYPES = ("numeric", "discrete", "date")


def parse_time(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


@dataclass
class Question:
    post_id: int
    question_id: int
    title: str
    type: str
    status: str | None = None
    description: str = ""
    resolution_criteria: str = ""
    fine_print: str = ""
    unit: str = ""
    label: str | None = None                  # group sub-question label
    options: list[str] = field(default_factory=list)
    range_min: float | None = None
    range_max: float | None = None
    zero_point: float | None = None
    open_lower_bound: bool = True
    open_upper_bound: bool = True
    inbound_outcome_count: int = 200
    nominal_min: float | None = None
    nominal_max: float | None = None
    open_time: datetime | None = None
    scheduled_close_time: datetime | None = None
    resolution: str | None = None
    my_latest_forecast: dict | None = None
    tournaments: list[str] = field(default_factory=list)
    question_weight: float | None = None
    community_prediction: list[float] | None = None   # forecast_values of the CP, only on accessible projects
    raw: dict = field(default_factory=dict, repr=False)
    my_score_data: dict = field(default_factory=dict, repr=False)   # my_forecasts.score_data (after resolution)

    # ------------------------------------------------------------------ derived
    @property
    def family(self) -> str:
        return route(self)

    @property
    def cdf_size(self) -> int:
        return int(self.inbound_outcome_count) + 1

    @property
    def already_forecast(self) -> bool:
        # Any forecast by this account counts, even a withdrawn one: bot tournaments allow one forecast per question.
        lf = self.my_latest_forecast
        return bool(lf) and lf.get("forecast_values") is not None

    @property
    def url(self) -> str:
        return f"https://www.metaculus.com/questions/{self.post_id}/"

    def minutes_left(self, now: datetime | None = None) -> float | None:
        if self.scheduled_close_time is None:
            return None
        now = now or datetime.now(timezone.utc)
        return (self.scheduled_close_time - now).total_seconds() / 60.0

    def text_blob(self) -> str:
        return " ".join(x for x in [self.title, self.label or "", self.description, self.resolution_criteria, self.fine_print] if x)

    # ------------------------------------------------------------------ parsing
    @classmethod
    def from_question_json(cls, post: dict, q: dict) -> "Question":
        scaling = q.get("scaling") or {}
        ioc = q.get("inbound_outcome_count") or scaling.get("inbound_outcome_count") or 200
        olb = q.get("open_lower_bound")
        oub = q.get("open_upper_bound")
        if olb is None:
            olb = scaling.get("open_lower_bound")
        if oub is None:
            oub = scaling.get("open_upper_bound")
        projects = post.get("projects") or {}
        slugs = [str(t.get("slug") or t.get("id")) for t in (projects.get("tournament") or [])]
        slugs += [str(t.get("slug") or t.get("id")) for t in (projects.get("question_series") or [])]
        mf = q.get("my_forecasts") or {}
        cp = None
        aggs = q.get("aggregations") or {}
        for method in ("recency_weighted", "unweighted"):
            latest = (aggs.get(method) or {}).get("latest") or {}
            if latest.get("forecast_values"):
                cp = latest["forecast_values"]
                break
        title = q.get("title") or post.get("title") or ""
        return cls(
            post_id=int(post["id"]),
            question_id=int(q["id"]),
            title=title,
            type=str(q.get("type")),
            status=q.get("status") or post.get("status"),
            description=q.get("description") or post.get("description") or "",
            resolution_criteria=q.get("resolution_criteria") or "",
            fine_print=q.get("fine_print") or "",
            unit=q.get("unit") or "",
            label=(q.get("label") or None),
            options=list(q.get("options") or []),
            range_min=_f(scaling.get("range_min")),
            range_max=_f(scaling.get("range_max")),
            zero_point=_f(scaling.get("zero_point")),
            open_lower_bound=bool(olb) if olb is not None else True,
            open_upper_bound=bool(oub) if oub is not None else True,
            inbound_outcome_count=int(ioc),
            nominal_min=_f(scaling.get("nominal_min")),
            nominal_max=_f(scaling.get("nominal_max")),
            open_time=parse_time(q.get("open_time") or post.get("open_time")),
            scheduled_close_time=parse_time(q.get("scheduled_close_time") or post.get("scheduled_close_time")),
            resolution=q.get("resolution"),
            my_latest_forecast=mf.get("latest"),
            tournaments=slugs,
            question_weight=_f(q.get("question_weight")),
            community_prediction=cp,
            raw=post,
            my_score_data=dict(mf.get("score_data") or {}),
        )


def _f(x: Any) -> float | None:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def questions_from_post(post: dict) -> list[Question]:
    """One post -> its forecastable questions. Notebooks and conditionals are skipped (not used in bot tournaments);
    group posts are unpacked into their sub-questions."""
    if "notebook" in post and post.get("notebook"):
        return []
    if post.get("conditional"):
        return []
    if post.get("group_of_questions"):
        group = post["group_of_questions"]
        out = []
        for q in group.get("questions") or []:
            qq = dict(q)
            for k in ("description", "resolution_criteria", "fine_print"):
                if not qq.get(k) and group.get(k):
                    qq[k] = group[k]
            out.append(Question.from_question_json(post, qq))
        return out
    q = post.get("question")
    return [Question.from_question_json(post, q)] if q else []


def route(q: Question) -> str:
    """Type router: binary / multiple_choice / continuous (numeric, discrete, date). Unknown -> 'unsupported'."""
    if q.type == "binary":
        return "binary"
    if q.type == "multiple_choice":
        return "multiple_choice" if len(q.options) >= 2 else "unsupported"
    if q.type in CONTINUOUS_TYPES:
        if q.range_min is None or q.range_max is None or not q.range_max > q.range_min:
            return "unsupported"
        return "continuous"
    return "unsupported"
