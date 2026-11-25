"""Deterministic parsers for the final-answer lines the prompts ask for. No LLM is used to parse (the template uses a
second 'parser' LLM call; one public bot maker reported parser failures as their biggest error source, so a strict
regex on a fixed format, with the sample dropped on failure, is cheaper and auditable)."""
from __future__ import annotations

import re
from datetime import datetime, timezone

_NUM = r"[-+]?\$?\s?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?"
_SUFFIX = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9,
           "t": 1e12, "trillion": 1e12}

RESOLVED_CLAIM = re.compile(r"\b(has|have|had) already (been )?resolved\b|\balready resolved (yes|no)\b", re.I)


class ParseError(ValueError):
    pass


def _to_float(tok: str) -> float:
    return float(tok.replace("$", "").replace(",", "").replace(" ", ""))


def parse_binary(text: str) -> float:
    """Last 'Probability: NN%' line -> NN/100."""
    hits = re.findall(r"probability\s*[:=]\s*(\d{1,3}(?:\.\d+)?)\s*%", text, flags=re.I)
    if not hits:
        raise ParseError("no 'Probability: NN%' line")
    p = float(hits[-1]) / 100.0
    if not 0.0 <= p <= 1.0:
        raise ParseError(f"probability out of range: {hits[-1]}%")
    return p


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def parse_multiple_choice(text: str, options: list[str]) -> dict[str, float]:
    """Lines '<option>: NN%' (the last block wins). Every option must appear; values are renormalized to sum 1."""
    found: dict[str, float] = {}
    norm_opts = {_norm(o): o for o in options}
    for line in text.splitlines():
        m = re.match(r"^\s*(?:[-*]\s*)?(?:option\s*[_:]?\s*)?[\"']?(.+?)[\"']?\s*[:=]\s*(\d{1,3}(?:\.\d+)?)\s*%?\s*$", line, flags=re.I)
        if not m:
            continue
        key = _norm(m.group(1))
        if key in norm_opts:
            found[norm_opts[key]] = float(m.group(2))
    missing = [o for o in options if o not in found]
    if missing:
        raise ParseError(f"missing options: {missing[:5]}")
    vals = [found[o] for o in options]
    tot = sum(vals)
    if tot <= 0:
        raise ParseError("probabilities sum to 0")
    return {o: found[o] / tot for o in options}


def _parse_value(raw: str, is_date: bool) -> float:
    raw = raw.replace("\u2212", "-").strip().rstrip(".")
    if is_date:
        m = re.search(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?Z?)?", raw)
        if not m:
            raise ParseError(f"not a date: {raw!r}")
        s = m.group(0).replace(" ", "T").replace("Z", "")
        d = datetime.fromisoformat(s)
        return d.replace(tzinfo=timezone.utc).timestamp()
    m = re.search(rf"({_NUM})\s*(thousand|million|billion|trillion|bn|mn|[kKmMbBtT])?\b", raw)
    if not m:
        raise ParseError(f"not a number: {raw!r}")
    v = _to_float(m.group(1))
    suf = (m.group(2) or "").lower()
    # only scale by a suffix that is clearly a magnitude word (a bare 'm' after a unit-less number is ambiguous
    # with metres, so single letters are honoured only when glued to the number, e.g. '3.2B')
    if suf in ("thousand", "million", "billion", "trillion", "bn", "mn"):
        v *= _SUFFIX[suf]
    elif suf and re.search(rf"{re.escape(m.group(1).strip())}{suf}\b", raw, flags=re.I):
        v *= _SUFFIX.get(suf, 1.0)
    return v


def parse_percentiles(text: str, is_date: bool = False) -> list[tuple[float, float]]:
    """'Percentile NN: value' lines -> sorted [(NN/100, value)]. The last occurrence of each percentile wins."""
    out: dict[float, float] = {}
    for m in re.finditer(r"percentile\s*(\d{1,2}(?:\.\d+)?)\s*[:=]\s*([^\n]+)", text, flags=re.I):
        p = float(m.group(1)) / 100.0
        if not 0 < p < 1:
            continue
        out[p] = _parse_value(m.group(2), is_date)
    if len(out) < 2:
        raise ParseError("fewer than 2 percentile lines")
    pts = sorted(out.items())
    for (p0, v0), (p1, v1) in zip(pts, pts[1:]):
        if v1 < v0:
            raise ParseError(f"values not increasing in percentile ({p0}: {v0} > {p1}: {v1})")
    return pts


def claims_resolved(text: str) -> bool:
    """The LLM says the question is already decided. Logged and kept clipped: only a data tool may unclip."""
    return bool(RESOLVED_CLAIM.search(text))
