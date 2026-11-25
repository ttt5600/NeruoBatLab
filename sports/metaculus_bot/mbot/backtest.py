"""Backtest harness: score the bot on already-resolved questions, with no submission.

1) fetch  -- pull resolved questions from the API into a local fixture dir (one JSON per post + index.jsonl).
   Needs METACULUS_TOKEN. Under the default "Restricted" API tier `resolution` and the question text are null for
   questions you never forecast, so use a project you have access to: the Bot Benchmarking list (project 32979,
   ~250 resolved questions "intended for training and evaluation purposes"; request the tier with Metaculus' Data
   Needs form) or bot-testing-area (32977). Metaculus' API terms forbid using its data to evaluate AI models
   without written permission -- project 32979 access is that permission; do not bulk-score other projects.
2) run    -- forecast each fixture as of its open time (no live news; data tools that cannot cut at that time are
   off) and score with the official formulas (Metaculus scoring/score_math.py, reimplemented):
     binary/MC baseline = 100 * ln(p_outcome * K) / ln(K)        (K = 2 for binary, number of options for MC)
     continuous baseline = 100 * ln(pmf[bucket] / b) / 2,  b = 0.05 out of range else (1 - 0.05 * open_bounds) / n
     rel_cp = 100 * ln(p_bot / p_cp)  (/2 for continuous) when the fixture carries a Community Prediction
   rel_cp is a proxy for the tournament peer score (which compares against the geometric mean of the other BOTS,
   not available offline). Report means with a question-level bootstrap 95% CI.

LOOKAHEAD WARNING: an LLM whose training data postdates a question's resolution can 'know' the answer. Only score
questions that opened after every ensemble model's knowledge cutoff (`--min-open-date`); otherwise a high score is
a leak, not skill.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import cdf as cdfm
from .forecaster import Forecaster
from .metaculus_api import MetaculusAPI
from .questions import Question, questions_from_post, route

UNSUCCESSFUL = {"annulled", "ambiguous"}


# ---------------------------------------------------------------------------------------------- fetch
def fetch_resolved(api: MetaculusAPI, project: str | int, out_dir: str | Path, limit: int = 100) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # with_cp: the Bot Benchmarking tier returns the current CP on project 32979, which gives the rel_cp proxy
    posts = api.list_posts(project, statuses="resolved", page_size=50, max_pages=max(1, math.ceil(limit / 50)),
                           extra={"with_cp": "true"})
    kept, null_res, null_text, other = 0, 0, 0, 0
    with open(out / "index.jsonl", "w") as idx:
        for post in posts:
            for q in questions_from_post(post):
                if route(q) == "unsupported":
                    other += 1
                    continue
                if q.resolution is None:
                    null_res += 1
                    continue
                if not (q.description or q.resolution_criteria):
                    null_text += 1
                    continue
                (out / f"{q.post_id}.json").write_text(json.dumps(post))
                idx.write(json.dumps({"post_id": q.post_id, "question_id": q.question_id, "type": q.type,
                                      "title": q.title, "open_time": str(q.open_time), "resolution": q.resolution,
                                      "has_cp": q.community_prediction is not None}) + "\n")
                kept += 1
            if kept >= limit:
                break
    return {"posts_listed": len(posts), "kept": kept, "resolution_null": null_res, "text_null": null_text,
            "unsupported": other, "out_dir": str(out)}


def load_fixtures(dir_: str | Path) -> list[Question]:
    qs = []
    for p in sorted(Path(dir_).glob("*.json")):
        try:
            post = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        qs.extend(questions_from_post(post))
    return qs


# ---------------------------------------------------------------------------------------------- scoring
def _resolution_value(q: Question) -> float:
    r = q.resolution
    if r == "below_lower_bound":
        return q.range_min - 1.0
    if r == "above_upper_bound":
        return q.range_max + 1.0
    if q.type == "date":
        return cdfm.date_to_value(r)
    return float(r)


def bucket_index(q: Question) -> int:
    spec = cdfm.Spec.from_question(q)
    loc = cdfm.nominal_to_location(_resolution_value(q), spec)
    n = spec.inbound
    if loc < 0:
        return 0
    if loc > 1:
        return n + 1
    if loc == 1:
        return n
    return max(int(loc * n + 1 - 1e-10), 1)


def outcome_prob(q: Question, final) -> tuple[float, float | None]:
    """(p assigned to the realized outcome, the CP's p for it or None). For continuous: pmf of the bucket."""
    fam = route(q)
    cp = q.community_prediction
    if fam == "binary":
        yes = q.resolution == "yes"
        p = final if yes else 1 - final
        pc = None
        if cp and len(cp) == 2:          # forecast_values = [1 - p_yes, p_yes]
            pc = cp[1] if yes else cp[0]
        elif cp and len(cp) == 1:
            pc = cp[0] if yes else 1 - cp[0]
        return p, pc
    if fam == "multiple_choice":
        i = q.options.index(q.resolution)
        p = final[q.resolution]
        pc = cp[i] if cp and len(cp) == len(q.options) else None
        return p, pc
    b = bucket_index(q)
    p = float(cdfm.cdf_to_pmf(final)[b])
    pc = float(cdfm.cdf_to_pmf(cp)[b]) if cp and len(cp) == q.cdf_size else None
    return p, pc


def score(q: Question, final) -> dict:
    fam = route(q)
    p, pc = outcome_prob(q, final)
    p = max(p, 1e-9)
    if fam == "binary":
        base = 100 * math.log(p * 2) / math.log(2)
    elif fam == "multiple_choice":
        k = len(q.options)
        base = 100 * math.log(p * k) / math.log(k)
    else:
        b = bucket_index(q)
        pmf_len = q.cdf_size + 1
        obc = int(q.open_lower_bound) + int(q.open_upper_bound)
        ref = 0.05 if b in (0, pmf_len - 1) else (1 - 0.05 * obc) / (pmf_len - 2)
        base = 100 * math.log(p / ref) / 2
    rel = None
    if pc:
        rel = 100 * math.log(p / max(pc, 1e-9)) * (0.5 if fam == "continuous" else 1.0)
    return {"baseline": base, "rel_cp": rel, "log_score": math.log(p), "p_outcome": p, "p_cp": pc}


def bootstrap_ci(x: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    a = np.asarray([v for v in x if v is not None], dtype=float)
    if len(a) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    return float(a.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------------------------------------- run
def run_backtest(fc: Forecaster, questions: list[Question], out_path: str | Path, limit: int = 50,
                 min_open_date: str | None = None, printer=print) -> dict:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    skipped = {"unresolved_or_annulled": 0, "before_min_open_date": 0, "forecast_failed": 0, "unsupported": 0}
    cut = datetime.fromisoformat(min_open_date).replace(tzinfo=timezone.utc) if min_open_date else None
    with open(out, "w") as f:
        for q in questions:
            if len(rows) >= limit:
                break
            if route(q) == "unsupported":
                skipped["unsupported"] += 1
                continue
            if q.resolution is None or q.resolution in UNSUCCESSFUL:
                skipped["unresolved_or_annulled"] += 1
                continue
            if cut and (q.open_time is None or q.open_time < cut):
                skipped["before_min_open_date"] += 1
                continue
            as_of = q.open_time or datetime.now(timezone.utc)
            fr = fc.forecast(q, as_of=as_of, run_id="backtest")
            if not fr.ok:
                skipped["forecast_failed"] += 1
                f.write(json.dumps({"question_id": q.question_id, "ok": False, "error": fr.error}) + "\n")
                continue
            sc = score(q, fr.final)
            row = {"question_id": q.question_id, "post_id": q.post_id, "type": q.type, "title": q.title[:120],
                   "as_of": as_of.isoformat(), "resolution": q.resolution, "readable": fr.readable,
                   "cost_usd": round(fr.cost_usd, 6), **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in sc.items()}}
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            printer(f"q{q.question_id} [{q.type}] baseline {sc['baseline']:+.1f}"
                    + (f" rel_cp {sc['rel_cp']:+.1f}" if sc["rel_cp"] is not None else "") + f"  {q.title[:60]!r}")
    m, lo, hi = bootstrap_ci([r["baseline"] for r in rows])
    rel = [r["rel_cp"] for r in rows if r["rel_cp"] is not None]
    rm, rlo, rhi = bootstrap_ci(rel)
    by_type = {}
    for t in sorted({r["type"] for r in rows}):
        v = [r["baseline"] for r in rows if r["type"] == t]
        by_type[t] = {"n": len(v), "mean_baseline": round(float(np.mean(v)), 2)}
    summary = {"n_scored": len(rows), "skipped": skipped,
               "baseline_mean": round(m, 2), "baseline_ci95": [round(lo, 2), round(hi, 2)],
               "rel_cp_n": len(rel), "rel_cp_mean": round(rm, 2) if rel else None,
               "rel_cp_ci95": [round(rlo, 2), round(rhi, 2)] if rel else None, "by_type": by_type,
               "cost_usd": round(sum(r["cost_usd"] for r in rows), 4), "scores_file": str(out),
               "min_open_date": min_open_date,
               "lookahead_note": "valid only if every question opened after all models' training cutoffs"}
    Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=1))
    return summary
