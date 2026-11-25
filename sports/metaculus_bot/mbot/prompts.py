"""Prompts, one per question family. Each ends with a fixed answer format that parsing.py reads with a regex.
Marker lines (FORMAT:, - OPTION:, LOW_HINT:/HIGH_HINT:) are machine-readable and also used by the mock LLM."""
from __future__ import annotations

from datetime import datetime, timezone

from .questions import Question

SYSTEM = ("You are a careful, well-calibrated forecaster. You reason from base rates and the status quo, weigh the "
          "evidence you are given, and never claim a question has resolved unless the evidence shows it.")

_COMMON = """Question: {title}{label}

Background:
{description}

Resolution criteria (not yet satisfied unless the evidence below shows otherwise):
{criteria}

Fine print:
{fine_print}

Research notes (news + official data pulled for this question; may be incomplete):
{research}

Today is {today} (UTC). The question closes for forecasting at {close}.
"""


def _fmt_bound(q: Question, v: float | None) -> str:
    if v is None:
        return "unknown"
    if q.type == "date":
        return datetime.fromtimestamp(v, tz=timezone.utc).date().isoformat()
    return repr(round(v, 6)) if abs(v) < 1e15 else f"{v:.6g}"


def _common(q: Question, research: str, today: datetime) -> str:
    return _COMMON.format(
        title=q.title, label=f" -- sub-question: {q.label}" if q.label else "",
        description=(q.description or "(none)")[:6000], criteria=(q.resolution_criteria or "(none)")[:4000],
        fine_print=(q.fine_print or "(none)")[:3000], research=research or "(no research available)",
        today=today.strftime("%Y-%m-%d %H:%M"),
        close=q.scheduled_close_time.strftime("%Y-%m-%d %H:%M") if q.scheduled_close_time else "unknown")


def binary_prompt(q: Question, research: str, today: datetime) -> str:
    return _common(q, research, today) + """
FORMAT: BINARY
Think it through briefly: (a) how long until the outcome is known, (b) the status-quo outcome if nothing changes,
(c) a concrete path to NO, (d) a concrete path to YES, (e) the base rate for this kind of event. The world usually
changes slowly; weight the status quo accordingly. Official data in the research notes outranks news commentary.

End with exactly one line:
Probability: NN%
"""


def mc_prompt(q: Question, research: str, today: datetime) -> str:
    opts = "\n".join(f"- OPTION: {o}" for o in q.options)
    return _common(q, research, today) + f"""
FORMAT: MULTIPLE_CHOICE
Options (copy the names exactly):
{opts}

Think it through briefly: (a) time until the outcome is known, (b) the status-quo option, (c) what would produce a
surprise. Leave some probability on every option that is not impossible.

End with one line per option, in the order above, exactly like:
<option name>: NN%
The percentages must sum to 100.
"""


def continuous_prompt(q: Question, research: str, today: datetime) -> str:
    lo, hi = _fmt_bound(q, q.range_min), _fmt_bound(q, q.range_max)
    lo_txt = (f"The outcome cannot be lower than {lo}." if not q.open_lower_bound
              else f"The question author thinks the value is unlikely to be below {lo}.")
    hi_txt = (f"The outcome cannot be higher than {hi}." if not q.open_upper_bound
              else f"The question author thinks the value is unlikely to be above {hi}.")
    if q.type == "date":
        fmt = "Write each value as a date YYYY-MM-DD (UTC)."
    else:
        unit = q.unit or "(infer from the question)"
        fmt = (f"Units: {unit}. Write plain numbers in these units (no scientific notation, no ranges)."
               + (" The outcome is a whole number." if q.type == "discrete" else ""))
    return _common(q, research, today) + f"""
FORMAT: CONTINUOUS
LOW_HINT: {lo}
HIGH_HINT: {hi}
{lo_txt}
{hi_txt}
{fmt}

Think it through briefly: (a) time until the outcome is known, (b) the value if nothing changes, (c) the value if
the current trend continues, (d) what experts or markets expect, (e) a surprise on the low side, (f) a surprise on
the high side. Good forecasters set wide 10-90 intervals.

End with exactly these six lines, values increasing:
Percentile 10: X
Percentile 20: X
Percentile 40: X
Percentile 60: X
Percentile 80: X
Percentile 90: X
"""


def build_prompt(q: Question, research: str, today: datetime) -> str:
    fam = q.family
    if fam == "binary":
        return binary_prompt(q, research, today)
    if fam == "multiple_choice":
        return mc_prompt(q, research, today)
    if fam == "continuous":
        return continuous_prompt(q, research, today)
    raise ValueError(f"unsupported question type {q.type}")
