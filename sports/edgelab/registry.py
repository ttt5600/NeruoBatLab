"""Append-only, hash-chained trial registry (GOAL.md rules 1 and 2).

File: lab/edgelab_registry.jsonl (override with EDGELAB_REGISTRY for sandboxes/tests). One JSON event per line; each
line carries `prev` = sha256 of the previous line's text, so an edited or deleted line breaks verify().

Events: register | amend | access | evaluate | transition | user_approval | legacy_note.
A trial = (spec, rule source). trial_id = <spec id>-<sha256(spec_sha256 + rule_sha256)[:8]>.
  * re-registering byte-identical spec + rule -> the same trial id (not counted twice);
  * any change to the spec (rule, params, filters, threshold, split, price model...) or to the rule module's source
    -> a NEW trial id, counted in the family forever;
  * amend(trial_id, new spec) is allowed only while the trial has no `access` event; it keeps the id and the count.
Nothing is ever un-registered. Family bar = alpha / (sum of `n_trials` over every trial ever registered in the family).
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path

from . import ALPHA
from . import paths

REQUIRED_SPEC = ["id", "family", "hypothesis", "dataset", "rule", "decision_time", "price_model", "split", "min_n",
                 "kill"]
PRICE_MODELS = {"taker", "maker_tape", "book_quote", "mixed"}
HISTORICAL_ORIGINS = {"ledger_import", "lockbox_import", "fund_book_3d", "study_import"}


class RegistryError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def sha256_file(path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rule_source(rule: str | None) -> tuple[str | None, str | None]:
    """'edgelab.rules.kalshi_wx:e1_taker' -> (source file path, sha256 of the whole module file)."""
    if not rule:
        return None, None
    mod, _, fn = rule.partition(":")
    m = importlib.import_module(mod)
    if fn and not hasattr(m, fn):
        raise RegistryError(f"rule {rule!r}: module has no function {fn!r}")
    f = inspect.getsourcefile(m)
    return f, sha256_file(f)


def validate_spec(spec: dict, historical: bool = False) -> None:
    missing = [k for k in REQUIRED_SPEC if k not in spec]
    if historical:
        missing = [k for k in missing if k in ("id", "family", "hypothesis")]
    if missing:
        raise RegistryError(f"spec missing {missing}")
    if not historical:
        if spec["price_model"] not in PRICE_MODELS:
            raise RegistryError(f"price_model must be one of {sorted(PRICE_MODELS)}")
        if not isinstance(spec["min_n"], int) or spec["min_n"] < 1:
            raise RegistryError("min_n must be a positive int fixed before data access")
        if not isinstance(spec.get("kill"), dict) or "kind" not in spec["kill"]:
            raise RegistryError("kill must be a dict with a 'kind' (rule 7: kill rules are code too)")


class Registry:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else paths.REGISTRY

    # ------------------------------------------------------------------ raw log
    def events(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]

    def _last_line(self) -> str | None:
        if not self.path.exists():
            return None
        lines = [l for l in self.path.read_text().splitlines() if l.strip()]
        return lines[-1] if lines else None

    def append(self, ev: dict) -> dict:
        last = self._last_line()
        ev = {"ts": now(), **ev, "prev": sha256_text(last) if last else None}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(ev, default=str) + "\n")
        return ev

    def verify(self) -> dict:
        """Hash chain intact? Returns {'ok': bool, 'lines': n, 'first_break': line_no|None}."""
        lines = [l for l in self.path.read_text().splitlines() if l.strip()] if self.path.exists() else []
        prev = None
        for i, l in enumerate(lines):
            if json.loads(l).get("prev") != (sha256_text(prev) if prev else None):
                return {"ok": False, "lines": len(lines), "first_break": i + 1}
            prev = l
        return {"ok": True, "lines": len(lines), "first_break": None}

    # ------------------------------------------------------------------ trials
    def trials(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for e in self.events():
            if e["event"] == "register":
                out[e["trial_id"]] = dict(e)
            elif e["event"] == "amend" and e["trial_id"] in out:
                out[e["trial_id"]].update(spec=e["spec"], spec_sha256=e["spec_sha256"], rule_sha256=e["rule_sha256"],
                                          rule_path=e.get("rule_path"), amended_at=e["ts"])
        return out

    def trial(self, trial_id: str) -> dict:
        t = self.trials().get(trial_id)
        if t is None:
            raise RegistryError(f"unknown trial {trial_id!r}")
        return t

    def by_spec_id(self, spec_id: str) -> list[dict]:
        return [t for t in self.trials().values() if t["spec"]["id"] == spec_id]

    def register(self, spec: dict, origin: str = "edgelab", initial_state: str | None = None,
                 n_trials: int = 1, evidence: str | None = None) -> str:
        historical = origin in HISTORICAL_ORIGINS
        validate_spec(spec, historical=historical)
        if initial_state not in (None, "dead"):
            raise RegistryError("only historical imports may start in a non-registered state, and only as 'dead'")
        if initial_state and not historical:
            raise RegistryError("initial_state is for historical imports only")
        rule_path, rule_sha = rule_source(spec.get("rule")) if not historical else (None, None)
        spec_sha = sha256_text(canonical(spec))
        tid = f"{spec['id']}-{sha256_text(spec_sha + (rule_sha or ''))[:8]}"
        if tid in self.trials():
            return tid
        self.append({"event": "register", "trial_id": tid, "family": spec["family"], "spec": spec,
                     "spec_sha256": spec_sha, "rule_path": rule_path, "rule_sha256": rule_sha,
                     "git_head": paths.git_head(), "origin": origin, "n_trials": int(n_trials),
                     "evidence": evidence})
        if initial_state == "dead":
            self.append({"event": "transition", "trial_id": tid, "from": "registered", "to": "dead",
                         "via": "historical_import", "evidence": evidence})
        return tid

    def accesses(self, trial_id: str | None = None) -> list[dict]:
        return [e for e in self.events() if e["event"] == "access" and (trial_id is None or e["trial_id"] == trial_id)]

    def amend(self, trial_id: str, spec: dict) -> str:
        t = self.trial(trial_id)
        if self.accesses(trial_id):
            raise RegistryError(f"{trial_id} already accessed data ({self.accesses(trial_id)[0]['ts']}); "
                                "amendments are only allowed before the first data access -> register a NEW trial")
        if spec["id"] != t["spec"]["id"] or spec["family"] != t["spec"]["family"]:
            raise RegistryError("amend may not change the spec id or family")
        validate_spec(spec)
        rule_path, rule_sha = rule_source(spec.get("rule"))
        self.append({"event": "amend", "trial_id": trial_id, "spec": spec,
                     "spec_sha256": sha256_text(canonical(spec)), "rule_path": rule_path, "rule_sha256": rule_sha,
                     "git_head": paths.git_head()})
        return trial_id

    def check_unchanged(self, trial_id: str) -> dict:
        """Rule 1 at evaluation time: the rule module on disk must hash to what was registered (or amended)."""
        t = self.trial(trial_id)
        if t.get("origin") in HISTORICAL_ORIGINS:
            raise RegistryError(f"{trial_id} is a historical import (no runnable rule)")
        path, sha = rule_source(t["spec"]["rule"])
        ok = sha == t["rule_sha256"] and sha256_text(canonical(t["spec"])) == t["spec_sha256"]
        if not ok:
            raise RegistryError(f"{trial_id}: rule module {path} changed since registration "
                                f"({t['rule_sha256'][:12]} -> {(sha or '')[:12]}); register a NEW trial")
        return {"rule_path": path, "rule_sha256": sha}

    # ------------------------------------------------------------------ families / bars
    def family_counts(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for t in self.trials().values():
            c[t["family"]] = c.get(t["family"], 0) + int(t.get("n_trials", 1))
        return c

    def family_bar(self, family: str, alpha: float = ALPHA) -> tuple[float, int]:
        n = self.family_counts().get(family, 0)
        return (alpha / n if n else alpha), n

    # ------------------------------------------------------------------ state
    def state(self, trial_id: str) -> str:
        s = "registered"
        for e in self.events():
            if e["event"] == "transition" and e["trial_id"] == trial_id:
                s = e["to"]
        return s

    def evaluations(self, trial_id: str, split: str | None = None) -> list[dict]:
        return [e for e in self.events() if e["event"] == "evaluate" and e["trial_id"] == trial_id
                and (split is None or e["split"] == split)]
