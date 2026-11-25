"""One question -> research -> N independent samples -> parse -> median -> guards -> validated payload + comment."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import aggregate as agg
from . import cdf as cdfm
from .config import Config
from .ledger import Ledger, SpendCapExceeded, SpendGuard
from .llm import BaseLLM, LLMRefusal
from .metaculus_api import forecast_payload
from .parsing import ParseError, claims_resolved, parse_binary, parse_multiple_choice, parse_percentiles
from .prompts import SYSTEM, build_prompt
from .questions import Question, route
from .research import Research, Researcher


@dataclass
class Sample:
    idx: int
    llm: str
    ok: bool = False
    value: object = None            # float | dict | list[(p, v)]
    cdf: list[float] | None = None
    error: str | None = None
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    claims_resolved: bool = False
    excerpt: str = ""

    def log(self) -> dict:
        v = self.value
        if isinstance(v, list):
            v = [[round(p, 4), x] for p, x in v]
        return {"idx": self.idx, "llm": self.llm, "ok": self.ok, "value": v, "error": self.error,
                "cost_usd": round(self.cost_usd, 6), "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
                "claims_resolved": self.claims_resolved}


@dataclass
class ForecastResult:
    question: Question
    family: str
    ok: bool = False
    error: str | None = None
    research: Research | None = None
    samples: list[Sample] = field(default_factory=list)
    final: object = None
    payload: dict | None = None
    comment: str = ""
    warnings: list[str] = field(default_factory=list)
    readable: str = ""

    @property
    def cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.samples)

    def log(self) -> dict:
        q = self.question
        fin = self.final
        if self.family == "continuous" and isinstance(fin, list):
            fin = {"cdf_len": len(fin), "summary": self.readable}
        return {"question_id": q.question_id, "post_id": q.post_id, "type": q.type, "family": self.family,
                "title": q.title[:200], "tournaments": q.tournaments, "ok": self.ok, "error": self.error,
                "inputs": {"research_chars": len(self.research.text) if self.research else 0,
                           "news": self.research.news_status if self.research else None,
                           "tools": self.research.log_rows() if self.research else []},
                "samples": [s.log() for s in self.samples], "final": fin, "readable": self.readable,
                "warnings": self.warnings, "cost_usd": round(self.cost_usd, 6)}


class _Reserve:
    """Thread-safe reservation on top of SpendGuard so parallel samples cannot jointly overshoot the cap."""

    def __init__(self, guard: SpendGuard | None):
        self.guard = guard
        self.lock = threading.Lock()
        self.reserved = 0.0

    def take(self, amount: float) -> None:
        if self.guard is None:
            return
        with self.lock:
            self.guard.check(self.reserved + amount)
            self.reserved += amount

    def settle(self, amount: float, row: dict) -> None:
        if self.guard is None:
            return
        with self.lock:
            self.reserved -= amount
            self.guard.record(row)


class Forecaster:
    def __init__(self, llms: list[BaseLLM], researcher: Researcher, cfg: Config, guard: SpendGuard | None = None,
                 ledger: Ledger | None = None, max_parallel: int = 5):
        if not llms:
            raise ValueError("need at least one LLM")
        self.llms, self.researcher, self.cfg, self.guard, self.ledger = llms, researcher, cfg, guard, ledger
        self.max_parallel = max_parallel
        self.budget_note: str | None = None

    # ------------------------------------------------------------------ one sample
    def _sample(self, i: int, q: Question, family: str, prompt: str, res: _Reserve, run_id: str | None) -> Sample:
        llm = self.llms[i % len(self.llms)]
        s = Sample(idx=i, llm=llm.name)
        worst = llm.worst_case_cost(prompt)
        res.take(worst)                      # raises SpendCapExceeded before any money is spent
        try:
            r = llm.complete(prompt, system=SYSTEM)
        except Exception as e:  # noqa: BLE001
            timeout_like = "timeout" in type(e).__name__.lower()
            cost = worst if timeout_like else 0.0      # a timed-out call may still be billed: count the worst case
            res.settle(worst, {"kind": "llm_error", "run_id": run_id, "question_id": q.question_id, "sample": i,
                               "llm": llm.name, "error": f"{type(e).__name__}: {str(e)[:200]}", "cost_usd": cost,
                               "refusal": isinstance(e, LLMRefusal)})
            s.error, s.cost_usd = f"{type(e).__name__}: {str(e)[:200]}", cost
            return s
        s.cost_usd, s.tokens_in, s.tokens_out = r.cost_usd, r.input_tokens, r.output_tokens
        res.settle(worst, {"kind": "llm_call", "run_id": run_id, "question_id": q.question_id, "sample": i,
                           "llm": llm.name, "served_model": r.model, "tokens_in": r.input_tokens,
                           "tokens_out": r.output_tokens, "cost_usd": r.cost_usd, "stop_reason": r.stop_reason})
        text = r.text or ""
        s.claims_resolved = claims_resolved(text)
        s.excerpt = text[-700:]
        try:
            if family == "binary":
                s.value = parse_binary(text)
            elif family == "multiple_choice":
                s.value = parse_multiple_choice(text, q.options)
            else:
                pts = parse_percentiles(text, is_date=(q.type == "date"))
                s.value = pts
                s.cdf = cdfm.percentiles_to_cdf(pts, cdfm.Spec.from_question(q))
            s.ok = True
        except (ParseError, ValueError) as e:
            s.error = f"parse: {e}"
        return s

    # ------------------------------------------------------------------ one question
    def forecast(self, q: Question, as_of: datetime | None = None, run_id: str | None = None) -> ForecastResult:
        family = route(q)
        out = ForecastResult(question=q, family=family)
        if family == "unsupported":
            out.error = f"unsupported question type {q.type!r}"
            return out
        out.research = self.researcher.run(q, as_of=as_of)
        today = as_of or datetime.now(timezone.utc)
        prompt = build_prompt(q, out.research.text, today)
        res = _Reserve(self.guard)
        n = max(1, self.cfg.n_samples)
        with ThreadPoolExecutor(max_workers=min(self.max_parallel, n)) as ex:
            futs = [ex.submit(self._sample, i, q, family, prompt, res, run_id) for i in range(n)]
            samples, cap_hit = [], None
            for f in futs:
                try:
                    samples.append(f.result())
                except SpendCapExceeded as e:
                    cap_hit = e
        out.samples = samples
        valid = [s for s in samples if s.ok]
        if cap_hit is not None and len(valid) < self.cfg.min_valid_samples:
            raise cap_hit
        if cap_hit is not None:
            out.warnings.append(f"spend cap reached mid-question; aggregated {len(valid)} samples")
        if len(valid) < max(1, self.cfg.min_valid_samples):
            out.error = f"only {len(valid)} valid samples of {n}: " + "; ".join(str(s.error) for s in samples if not s.ok)[:400]
            return out
        if any(s.claims_resolved for s in valid):
            out.warnings.append("an LLM sample claims the question already resolved; kept the normal clip")
        try:
            self._aggregate(out, q, family, valid)
        except ValueError as e:
            out.error = f"aggregation: {e}"
            return out
        out.payload = forecast_payload(family, out.final)
        out.comment = build_comment(out, self.llms, today)
        out.ok = True
        return out

    def _aggregate(self, out: ForecastResult, q: Question, family: str, valid: list[Sample]) -> None:
        cfg = self.cfg
        if family == "binary":
            p = agg.median_binary([s.value for s in valid])
            p = agg.extremize_binary(p, cfg.extremize)
            resolved = out.research.resolves if out.research else None
            lo, hi = (cfg.resolved_clip_lo, cfg.resolved_clip_hi) if resolved is not None else (cfg.binary_clip_lo, cfg.binary_clip_hi)
            if resolved is not None:
                out.warnings.append(f"data tool reports the outcome determined ({resolved}); clip widened to [{lo}, {hi}]")
            out.final = round(agg.clip_binary(p, lo, hi), 4)
            out.readable = f"P(yes) = {out.final:.3f} (raw median {p:.3f}, n={len(valid)})"
        elif family == "multiple_choice":
            m = agg.median_mc([s.value for s in valid], q.options)
            out.final = agg.floor_mc(m, cfg.mc_floor)
            out.readable = ", ".join(f"{o}: {v:.3f}" for o, v in out.final.items())
        else:
            spec = cdfm.Spec.from_question(q)
            med = cdfm.median_cdf([s.cdf for s in valid], spec)
            errs = cdfm.validate_cdf(med, spec)
            if errs:
                raise ValueError("; ".join(errs))
            out.final = med
            out.readable = cdfm.readable(med, spec)


def pick_models(cfg: Config, remaining_usd: float) -> tuple[str, int, str | None]:
    """Budget pacing. A skipped question scores 0 and prizes scale with the square of the summed score, so running
    out of budget mid-month is worse than forecasting the rest with a cheaper ensemble. Below the threshold, switch to
    MBOT_FALLBACK_MODELS (if set); the hard cap still applies to every call."""
    if cfg.fallback_spec and remaining_usd < cfg.fallback_below_usd:
        return cfg.fallback_spec, max(1, cfg.fallback_n_samples), (
            f"budget fallback: ${remaining_usd:.2f} left of the monthly cap < ${cfg.fallback_below_usd:.2f}")
    return cfg.provider_spec, cfg.n_samples, None


def build_comment(out: ForecastResult, llms: list[BaseLLM], today: datetime) -> str:
    """Private comment required on every forecast; it must reflect the actual reasoning and the final decision."""
    q = out.question
    valid = [s for s in out.samples if s.ok]
    lines = [f"# Forecast: {out.readable}", "",
             f"*Method*: {len(out.samples)} independent samples ({', '.join(sorted({l.name for l in llms}))}); "
             f"{len(valid)} parsed; aggregated by median"
             + ("; binary clipped to the configured range" if out.family == "binary" else "")
             + ("; each option floored then renormalized" if out.family == "multiple_choice" else "")
             + ("; pointwise-median CDF, validated against the API rules" if out.family == "continuous" else "")
             + f". Generated {today.strftime('%Y-%m-%d %H:%M')} UTC by an automated bot, no human in the loop.", ""]
    if out.research is not None:
        used = [f"{r.tool} (as of {r.asof})" for r in out.research.tool_results if r.ok]
        lines.append("*Research*: " + (", ".join(used) if used else "no official-data tool matched")
                     + f"; news: {out.research.news_status}.")
        lines.append("")
    if out.warnings:
        lines += [f"*Note*: {w}" for w in out.warnings] + [""]
    lines.append("## Samples")
    for s in out.samples:
        if s.ok:
            if out.family == "binary":
                v = f"{s.value:.3f}"
            elif out.family == "multiple_choice":
                v = ", ".join(f"{k}: {x:.2f}" for k, x in s.value.items())
            else:
                v = ", ".join(f"p{int(p * 100)}={x:.6g}" for p, x in s.value)
            lines.append(f"- [{s.idx}] {s.llm}: {v}")
            exc = " ".join(s.excerpt.split())[-400:]
            if exc:
                lines.append(f"  > ...{exc}")
        else:
            lines.append(f"- [{s.idx}] {s.llm}: dropped ({s.error})")
    text = "\n".join(lines)
    return text[:9500]
