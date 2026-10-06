"""Score strategies. The only place a number about a strategy comes from.

    python lab/run.py schema  DATASET            # columns an agent can use
    python lab/run.py dev     lab/strategies/x.py  # walk-forward on DEV folds; appends ledger
    python lab/run.py ledger  [--dataset D]       # every trial so far
    python lab/run.py promote lab/strategies/x.py --confirm-lockbox   # HUMAN ONLY

Walk-forward: for each dev fold after the first few, fit() sees only earlier
folds (with outcomes), bet() sees the fold's rows with outcome columns removed.

Every dev run is a TRIAL and is counted, whether the agent likes the result or
not. The ledger is append-only. A dev result is a hypothesis; the lockbox,
scored once per strategy by a human, is the test -- and its bar rises with the
number of trials it took to find the strategy.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent))

import numpy as np
import pandas as pd

from lab import datasets as D
from lab import sandbox as S

LEDGER = Path(os.environ["LAB_LEDGER"]) if os.environ.get("LAB_LEDGER") else LAB / "ledger.jsonl"
LOCKBOX_LOG = LAB / "lockbox.jsonl"


def _load_meta(path: Path) -> dict:
    src = Path(path).read_text()
    S.check_source(src)
    ns: dict = {}
    import ast
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ("DATASET", "HYPOTHESIS", "SOURCE"):
            ns[node.targets[0].id] = ast.literal_eval(node.value)
    if ns["DATASET"] not in D.SPLITS:
        raise S.StrategyRejected(f"DATASET must be one of {sorted(D.SPLITS)}")
    return ns


def score(df: pd.DataFrame, stakes: pd.Series, fold_col: str) -> dict:
    s = stakes.reindex(df.index).fillna(0.0)
    profit = np.where(df["push"] == 1, 0.0,
                      np.where(df["won"] == 1, s * (df["dec_odds"] - 1), -s))
    df = df.assign(stake=s, profit=profit)
    bets = df[df["stake"] > 0]
    if bets.empty:
        return {"bets": 0, "staked": 0.0, "units": 0.0, "roi": 0.0, "ci_lo": 0.0, "ci_hi": 0.0,
                "p": 1.0, "folds_positive": 0, "per_fold": {}}
    by_day = bets.groupby("date")["profit"].sum().to_numpy()
    boots = np.random.default_rng(0).choice(by_day, (5000, len(by_day))).sum(axis=1)
    per_fold = bets.groupby(fold_col)["profit"].sum().round(2)
    return {"bets": int(len(bets)), "staked": float(bets["stake"].sum()),
            "units": float(bets["profit"].sum()),
            "roi": float(bets["profit"].sum() / bets["stake"].sum()),
            "ci_lo": float(np.percentile(boots, 2.5)), "ci_hi": float(np.percentile(boots, 97.5)),
            "p": float(np.mean(boots <= 0)),
            "folds_positive": int((per_fold > 0).sum()), "per_fold": {str(k): v for k, v in per_fold.items()}}


def walk_forward(path: Path, dataset: str, split: str) -> tuple[dict, pd.DataFrame]:
    fold_col, dev_folds, lock_folds = D.SPLITS[dataset]
    hidden = [c for c in D.OUTCOME_COLS]
    if split == "dev":
        df = D.load(dataset, "dev")
        folds = dev_folds
        test_folds = folds[D.MIN_TRAIN_FOLDS[dataset]:]
    else:
        df = pd.concat([D.load(dataset, "dev"), D.load(dataset, "lockbox")], ignore_index=True)
        folds = dev_folds + lock_folds
        test_folds = lock_folds
    stakes = []
    for f in test_folds:
        train = df[df[fold_col].isin(folds[:folds.index(f)])]
        slate = df[df[fold_col] == f].drop(columns=[c for c in hidden if c in df.columns])
        stakes.append(S.run_fold(path, train, slate))
    stakes = pd.concat(stakes)
    tested = df.loc[stakes.index]
    return score(tested, stakes, fold_col), tested.assign(stake=stakes)


def trials(dataset: str | None = None) -> list[dict]:
    if not LEDGER.exists():
        return []
    rows = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    # Only scored runs are trials; a strategy rejected before scoring saw no data.
    return [r for r in rows if r.get("status") == "ok" and (dataset is None or r["dataset"] == dataset)]


def cmd_dev(args) -> int:
    path = Path(args.strategy)
    meta: dict = {"DATASET": "?", "HYPOTHESIS": "?", "SOURCE": "?"}
    try:
        meta = _load_meta(path)
        res, _ = walk_forward(path, meta["DATASET"], "dev")
        status = "ok"
    except (S.StrategyRejected, SyntaxError) as e:
        print(f"REJECTED: {e}")
        res, status = {}, f"rejected: {str(e)[:300]}"
    n = len({r["sha"] for r in trials(meta.get("DATASET"))} | {S.source_hash(path)})
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": path.stem,
           "sha": S.source_hash(path), "dataset": meta.get("DATASET"),
           "hypothesis": meta.get("HYPOTHESIS"), "source": meta.get("SOURCE"),
           "status": status, "trial_number": n, **res}
    with LEDGER.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    if status != "ok":
        return 1
    bar = 0.05 / n
    print(f"{path.stem} on {meta['DATASET']} (DEV walk-forward, trial #{n} on this dataset)")
    print(f"  bets {res['bets']}  staked {res['staked']:.1f}u  units {res['units']:+.2f}  "
          f"ROI {res['roi']:+.2%}  95% CI [{res['ci_lo']:+.1f}, {res['ci_hi']:+.1f}]u  p {res['p']:.4f}")
    print(f"  folds positive {res['folds_positive']}/{len(res['per_fold'])}: {res['per_fold']}")
    print(f"  promotion bar after {n} trials: p < {bar:.5f} "
          f"-> {'MEETS BAR (still only a hypothesis until the lockbox)' if res['p'] < bar else 'does not meet bar'}")
    return 0


def cmd_ledger(args) -> int:
    rows = trials(args.dataset)
    if not rows:
        print("ledger empty")
        return 0
    df = pd.DataFrame(rows)
    cols = ["ts", "strategy", "dataset", "status", "bets", "units", "roi", "p", "folds_positive", "hypothesis"]
    df = df[[c for c in cols if c in df.columns]]
    with pd.option_context("display.width", 250, "display.max_colwidth", 70):
        print(df.tail(args.tail).to_string(index=False))
    for d, g in pd.DataFrame(rows).groupby("dataset"):
        print(f"{d}: {g['sha'].nunique()} distinct strategies tried -> promotion bar p < {0.05 / g['sha'].nunique():.5f}")
    return 0


def cmd_merge_ledger(args) -> int:
    """Append another ledger's trials (e.g. a cloud agent's) as origin=cloud.
    They count toward the promotion bar: the agent saw those results."""
    have = {(r["ts"], r["sha"]) for r in (json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip())} \
        if LEDGER.exists() else set()
    n = 0
    with LEDGER.open("a") as f:
        for line in Path(args.file).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if (r["ts"], r["sha"]) in have:
                continue
            r["origin"] = args.origin
            f.write(json.dumps(r) + "\n")
            n += 1
    print(f"merged {n} trials from {args.file}")
    return 0


def cmd_export_dev(args) -> int:
    """Write DEV rows only to lab/devdata/ -- what cloud agents get. No lockbox."""
    out = LAB / "devdata"
    out.mkdir(exist_ok=True)
    for name in D.SPLITS:
        df = D.load(name, "dev")
        df.to_parquet(out / f"{name}.parquet", index=False)
        print(f"{name}: {len(df)} dev rows -> devdata/{name}.parquet")
    return 0


def cmd_schema(args) -> int:
    print(json.dumps(D.schema(args.dataset), indent=1))
    return 0


def cmd_promote(args) -> int:
    if not args.confirm_lockbox:
        print("lockbox scoring is a human decision; pass --confirm-lockbox")
        return 1
    path = Path(args.strategy)
    meta = _load_meta(path)
    sha = S.source_hash(path)
    prior = [r for r in (json.loads(l) for l in LOCKBOX_LOG.read_text().splitlines())
             if r["sha"] == sha] if LOCKBOX_LOG.exists() else []
    if prior:
        print(f"this exact code was already scored on the lockbox ({prior[0]['ts']}); "
              "re-scoring would turn the lockbox into a dev set. Refusing.")
        return 1
    n_dev = len({r["sha"] for r in trials(meta["DATASET"])})
    n_promoted = 1 + sum(1 for r in (json.loads(l) for l in LOCKBOX_LOG.read_text().splitlines())
                         if r["dataset"] == meta["DATASET"]) if LOCKBOX_LOG.exists() else 1
    res, _ = walk_forward(path, meta["DATASET"], "lockbox")
    bar = 0.05 / n_promoted
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": path.stem,
           "sha": sha, "dataset": meta["DATASET"], "dev_trials": n_dev,
           "promotion_number": n_promoted, **res}
    with LOCKBOX_LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"LOCKBOX {path.stem} on {meta['DATASET']} (found after {n_dev} dev trials; "
          f"promotion #{n_promoted} on this dataset)")
    print(f"  bets {res['bets']}  units {res['units']:+.2f}  ROI {res['roi']:+.2%}  "
          f"95% CI [{res['ci_lo']:+.1f}, {res['ci_hi']:+.1f}]u  p {res['p']:.4f}  "
          f"folds positive {res['folds_positive']}/{len(res['per_fold'])}")
    print(f"  verdict: {'PASS' if res['p'] < bar and res['units'] > 0 else 'FAIL'} (bar p < {bar:.4f})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("dev"); a.add_argument("strategy"); a.set_defaults(fn=cmd_dev)
    a = sub.add_parser("ledger"); a.add_argument("--dataset"); a.add_argument("--tail", type=int, default=40)
    a.set_defaults(fn=cmd_ledger)
    a = sub.add_parser("schema"); a.add_argument("dataset"); a.set_defaults(fn=cmd_schema)
    a = sub.add_parser("merge-ledger"); a.add_argument("file"); a.add_argument("--origin", default="cloud")
    a.set_defaults(fn=cmd_merge_ledger)
    a = sub.add_parser("export-dev"); a.set_defaults(fn=cmd_export_dev)
    a = sub.add_parser("promote"); a.add_argument("strategy"); a.add_argument("--confirm-lockbox", action="store_true")
    a.set_defaults(fn=cmd_promote)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
