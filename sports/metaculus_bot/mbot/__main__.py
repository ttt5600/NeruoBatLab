"""CLI. Run from sports/metaculus_bot:  .venv/bin/python -m mbot <command> --help

  check-env                      which keys are set (never prints them), models parse, paths, spend so far
  run --dry-run [--questions-file F]   forecast and print, submit nothing
  run                            the unattended pass: forecast every open, unforecast question and submit
  status                         month-to-date spend vs cap, submissions, recent errors, actual $/question
  evaluate                       apply the pre-registered go/kill rule to Metaculus' own scores
  smoke --yes                    ONE tiny paid call per configured model (cents) to prove the keys work
  backtest fetch|run             score on resolved questions (see mbot/backtest.py)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone

from . import backtest as bt
from .config import BOT_BENCHMARK_PROJECT, Config
from .forecaster import Forecaster, pick_models
from .ledger import AlreadyRunning, Ledger, SpendGuard
from .llm import LLMError, build_ensemble, price_for
from .metaculus_api import MetaculusAPI
from .research import Researcher
from .runner import Runner, load_posts_file, run_locked
from .tools import default_tools


def _build(cfg: Config, args) -> tuple[Forecaster, Ledger, SpendGuard]:
    if getattr(args, "models", None):
        cfg.provider_spec = args.models
    if getattr(args, "n_samples", None):
        cfg.n_samples = args.n_samples
    if getattr(args, "state_dir", None):
        from pathlib import Path
        cfg.state_dir = Path(args.state_dir)
    if getattr(args, "no_tools", False):
        cfg.use_data_tools = False
    ledger = Ledger(cfg.ledger_path)
    guard = SpendGuard(ledger, cfg.monthly_cap_usd)
    cfg.provider_spec, cfg.n_samples, note = pick_models(cfg, guard.remaining())
    llms = build_ensemble(cfg.provider_spec, max_tokens=cfg.max_output_tokens, effort=cfg.effort)
    tools = default_tools(cfg) if cfg.use_data_tools else []
    researcher = Researcher(tools=tools, max_chars=cfg.research_chars)
    fc = Forecaster(llms, researcher, cfg, guard=guard, ledger=ledger)
    fc.budget_note = note
    return fc, ledger, guard


def cmd_check_env(cfg: Config, args) -> int:
    print("secrets present:", json.dumps(cfg.secrets_present()))
    print("models (MBOT_MODELS):", cfg.provider_spec)
    try:
        llms = build_ensemble(cfg.provider_spec, max_tokens=cfg.max_output_tokens, effort=cfg.effort)
        for l in llms:
            pin, pout = price_for(l.provider, l.model)
            print(f"  {l.name}: ${pin}/M in, ${pout}/M out; worst case per sample on a 12k-char prompt "
                  f"${l.worst_case_cost('x' * 12000):.3f}")
    except LLMError as e:
        print("  model setup error:", e)
    print("tournaments:", cfg.tournaments, "| samples:", cfg.n_samples, "| cap $", cfg.monthly_cap_usd)
    print("state dir:", cfg.state_dir, "| fund scripts:", cfg.fund_scripts_dir, "exists" if cfg.fund_scripts_dir.exists() else "MISSING")
    led = Ledger(cfg.ledger_path)
    print(f"spent this month: ${led.spent_month():.4f}; submitted questions: {len(led.submitted_question_ids())}")
    return 0


def cmd_run(cfg: Config, args) -> int:
    if args.max_questions:
        cfg.max_questions_per_run = args.max_questions
    fc, ledger, guard = _build(cfg, args)
    posts = load_posts_file(args.questions_file) if args.questions_file else None
    api = None
    if posts is None:
        if not cfg.metaculus_token:
            print("METACULUS_TOKEN is not set: the API needs it even to list questions. For an offline dry run pass "
                  "--questions-file tests/fixtures/open_posts.json", file=sys.stderr)
            return 2
        api = MetaculusAPI(cfg.metaculus_token)
    elif not args.dry_run:
        print("--questions-file is for dry runs only", file=sys.stderr)
        return 2
    try:
        runner = Runner(cfg, fc, ledger, api, dry_run=args.dry_run)
    except ValueError as e:            # e.g. the mock provider in a live run
        print(f"refusing: {e}", file=sys.stderr)
        return 2
    tournaments = args.tournament or cfg.tournaments
    try:
        summary = run_locked(cfg, runner, tournaments=tournaments, posts_override=posts)
    except AlreadyRunning as e:
        print(f"skip: {e}")
        return 0
    print(json.dumps(summary, indent=1))
    return 0


def cmd_status(cfg: Config, args) -> int:
    led = Ledger(cfg.ledger_path)
    rows = list(led.rows())
    now = datetime.now(timezone.utc)
    print(f"ledger: {cfg.ledger_path} ({len(rows)} rows)")
    print(f"spent {now:%Y-%m}: ${led.spent_month(now):.4f} of ${cfg.monthly_cap_usd:.2f} cap")
    print(f"submitted questions (all time): {len(led.submitted_question_ids())}")
    day = (now - timedelta(days=1)).isoformat()
    recent = [r for r in rows if r.get("ts", "") >= day]
    print("last 24h by kind:", dict(Counter(r.get("kind") for r in recent)))
    skips = Counter(r.get("reason") for r in recent if r.get("kind") == "skip")
    if skips:
        print("last 24h skip reasons:", dict(skips))
    errs = [r for r in recent if r.get("kind") in ("submit_error", "comment_error", "llm_error")]
    for r in errs[-5:]:
        print("  error:", r.get("kind"), r.get("question_id"), str(r.get("error"))[:160])
    fc_rows = [r for r in rows if r.get("kind") == "forecast" and r.get("ok") and not r.get("dry_run")]
    if fc_rows:
        per_q = sum(float(r.get("cost_usd") or 0) for r in fc_rows) / len(fc_rows)
        print(f"actual cost per live question: ${per_q:.3f} over {len(fc_rows)} questions "
              f"(x ~130 MiniBench questions/month = ${per_q * 130:.0f}/month vs cap ${cfg.monthly_cap_usd:.0f})")
    runs = [r for r in rows if r.get("kind") == "run_end"]
    if runs:
        print("last run:", json.dumps(runs[-1])[:400])
    return 0


def cmd_evaluate(cfg: Config, args) -> int:
    from . import evaluate as ev
    if not cfg.metaculus_token:
        print("METACULUS_TOKEN is required to read scores", file=sys.stderr)
        return 2
    led = Ledger(cfg.ledger_path)
    since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc) if args.since else ev.first_live_run(led)
    if since is None:
        print("no live run in the ledger yet; pass --since YYYY-MM-DD")
        return 2
    got = ev.collect(MetaculusAPI(cfg.metaculus_token), led, args.tournament or cfg.tournaments, since)
    res = ev.decide(got["eligible"], got["forecast"], got["scores"], ev.monthly_spend(led), cfg.monthly_cap_usd,
                    previous_verdict=args.previous)
    res["since"] = since.isoformat()
    print(json.dumps(res, indent=1, default=str))
    led.append({"kind": "evaluation", **res})
    return 0


def cmd_smoke(cfg: Config, args) -> int:
    if not args.yes:
        print("smoke makes ONE real, paid call per model (a few cents). Re-run with --yes to proceed.")
        return 2
    fc, ledger, guard = _build(cfg, args)
    prompt = "Reply with exactly: Probability: 42%"
    rc = 0
    for l in fc.llms:
        try:
            guard.check(l.worst_case_cost(prompt))
            r = l.complete(prompt)
            guard.record({"kind": "llm_call", "question_id": None, "llm": l.name, "served_model": r.model,
                          "tokens_in": r.input_tokens, "tokens_out": r.output_tokens, "cost_usd": r.cost_usd, "smoke": True})
            print(f"OK   {l.name} -> {r.text.strip()[:60]!r}  ${r.cost_usd:.5f}")
        except Exception as e:  # noqa: BLE001
            rc = 1
            print(f"FAIL {l.name}: {type(e).__name__}: {str(e)[:200]}")
    return rc


def cmd_backtest(cfg: Config, args) -> int:
    if args.bt_cmd == "fetch":
        if not cfg.metaculus_token:
            print("METACULUS_TOKEN is required to fetch resolved questions", file=sys.stderr)
            return 2
        api = MetaculusAPI(cfg.metaculus_token)
        print(json.dumps(bt.fetch_resolved(api, args.project, args.out, limit=args.limit), indent=1))
        return 0
    fc, ledger, guard = _build(cfg, args)
    qs = bt.load_fixtures(args.fixtures)
    summary = bt.run_backtest(fc, qs, args.out, limit=args.limit, min_open_date=args.min_open_date)
    print(json.dumps(summary, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mbot", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--models", help="override MBOT_MODELS, e.g. 'mock:m' or 'anthropic:claude-opus-5-5'")
        p.add_argument("--n-samples", type=int)
        p.add_argument("--state-dir", help="override MBOT_STATE_DIR (ledger + lock)")
        p.add_argument("--no-tools", action="store_true", help="disable the official-data tools")

    sub.add_parser("check-env")
    p = sub.add_parser("run")
    common(p)
    p.add_argument("--dry-run", action="store_true", help="forecast and print; never submit")
    p.add_argument("--tournament", action="append", help="slug or id; repeatable (default MBOT_TOURNAMENTS)")
    p.add_argument("--questions-file", help="offline posts JSON (dry runs only)")
    p.add_argument("--max-questions", type=int)
    sub.add_parser("status")
    p = sub.add_parser("evaluate", help="apply the pre-registered go/kill rule (mbot/evaluate.py)")
    p.add_argument("--tournament", action="append")
    p.add_argument("--since", help="YYYY-MM-DD; default = first live run in the ledger")
    p.add_argument("--previous", choices=["HOLD"], help="pass HOLD if the last evaluation was HOLD")
    p = sub.add_parser("smoke")
    common(p)
    p.add_argument("--yes", action="store_true")
    p = sub.add_parser("backtest")
    bsub = p.add_subparsers(dest="bt_cmd", required=True)
    f = bsub.add_parser("fetch")
    f.add_argument("--project", default=str(BOT_BENCHMARK_PROJECT))
    f.add_argument("--limit", type=int, default=100)
    f.add_argument("--out", default="backtest_fixtures")
    r = bsub.add_parser("run")
    common(r)
    r.add_argument("--fixtures", default="backtest_fixtures")
    r.add_argument("--limit", type=int, default=50)
    r.add_argument("--min-open-date", help="YYYY-MM-DD: only questions opened on/after this (post model cutoff)")
    r.add_argument("--out", default="backtest_results/scores.jsonl")

    args = ap.parse_args(argv)
    cfg = Config()
    if getattr(args, "state_dir", None):
        from pathlib import Path
        cfg.state_dir = Path(args.state_dir)
    return {"check-env": cmd_check_env, "run": cmd_run, "status": cmd_status, "smoke": cmd_smoke,
            "backtest": cmd_backtest, "evaluate": cmd_evaluate}[args.cmd](cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
