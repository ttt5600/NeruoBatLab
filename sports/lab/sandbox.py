"""Run an agent-written strategy without letting it reach data it should not see.

A strategy is a .py file defining:

    DATASET = "nfl_total"                    # one of datasets.SPLITS
    HYPOTHESIS = "one sentence: what edge and why the market misses it"
    SOURCE = "registry:<id> | <url> | agent"
    def fit(train: pd.DataFrame): ...        # gets outcome columns; returns anything
    def bet(model, slate: pd.DataFrame) -> pd.Series   # stake per row, 0..MAX_STAKE

Two layers keep the lockbox closed:
1. A static check rejects file, network and process access, dynamic imports
   and dunder tricks. Only numeric/data libraries may be imported.
2. The code runs in a child process that is handed exactly two frames -- the
   training folds and the outcome-stripped slate -- as temp files it reads
   through this module, never through a path the strategy chooses.

This is a guard against an agent that drifts, not a security boundary against
a determined adversary; every promoted strategy's code is read by a human
before its lockbox number means anything.
"""
from __future__ import annotations

import ast
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

MAX_STAKE = 3.0
TIMEOUT_S = 300
ALLOWED_IMPORTS = {"numpy", "pandas", "scipy", "sklearn", "math", "statistics", "itertools",
                   "functools", "collections", "dataclasses", "typing", "warnings"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
                   "getattr", "setattr", "delattr", "input", "breakpoint", "memoryview"}
FORBIDDEN_ATTR_PREFIXES = ("read_", "to_", "__")
FORBIDDEN_ATTRS = {"io", "os", "sys", "subprocess", "pathlib", "Path", "load", "save", "loadtxt",
                   "savetxt", "fromfile", "tofile", "genfromtxt", "DataReader", "system", "popen"}


class StrategyRejected(Exception):
    pass


def check_source(src: str) -> None:
    tree = ast.parse(src)
    have = {n.targets[0].id for n in tree.body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    missing = ({"DATASET", "HYPOTHESIS", "SOURCE"} - have) | ({"fit", "bet"} - funcs)
    if missing:
        raise StrategyRejected(f"missing required definitions: {sorted(missing)}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise StrategyRejected(f"import not allowed: {a.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS or node.level:
                raise StrategyRejected(f"import not allowed: from {node.module}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise StrategyRejected(f"name not allowed: {node.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRS or node.attr.startswith(FORBIDDEN_ATTR_PREFIXES):
                if node.attr not in ("to_numpy", "to_list", "to_dict", "to_frame", "to_series",
                                     "to_period", "to_timestamp", "to_datetime", "to_numeric"):
                    raise StrategyRejected(f"attribute not allowed: .{node.attr}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "/" in node.value and ("data" in node.value or node.value.startswith("/")):
                raise StrategyRejected("string literal looks like a file path")


def source_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def run_fold(strategy: Path, train: pd.DataFrame, slate: pd.DataFrame) -> pd.Series:
    """Fit on ``train``, stake ``slate`` in a child process. Returns stakes by slate index."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        train.to_parquet(td / "train.parquet")
        slate.to_parquet(td / "slate.parquet")
        cmd = [sys.executable, "-m", "lab._worker", str(Path(strategy).resolve()), str(td)]
        r = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], capture_output=True,
                           text=True, timeout=TIMEOUT_S)
        if r.returncode != 0:
            raise StrategyRejected(f"strategy crashed:\n{r.stderr[-2000:]}")
        stakes = pd.read_parquet(td / "stakes.parquet")["stake"]
    stakes.index = slate.index
    if stakes.isna().any() or (stakes < 0).any() or (stakes > MAX_STAKE).any():
        raise StrategyRejected(f"stakes must be finite and in [0, {MAX_STAKE}]")
    return stakes
