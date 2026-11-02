"""Vault (GOAL.md rule 3): every data access goes through load(dataset, split, trial_id) and is logged in the registry.

  * a trial may only load ITS OWN dataset (the one in its registered spec);
  * dev: free; forward: free (data that did not exist when the strategy was frozen);
  * sealed: only if the trial's state is dev_pass (or, for a dataset with `dev: null`, the trial was registered before
    the sealed window opened), and only ONCE per trial -- a second unseal raises VaultError;
  * spent lockboxes (vault.yaml spent_unseals) may be replayed once by the listed spec id, flagged replay=True;
  * outcome columns (`hidden`) are stripped from the frame the rule sees and kept for resolve_outcomes().
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import yaml

from . import loaders, paths
from .registry import Registry, sha256_file

SPLITS = ("dev", "sealed", "forward")


class VaultError(PermissionError):
    pass


def config() -> dict:
    return yaml.safe_load(paths.VAULT_YAML.read_text())["datasets"]


def vault_sha() -> str:
    return sha256_file(paths.VAULT_YAML)


def window(ds: str, split: str):
    c = config()[ds]["splits"].get(split)
    if c is None:
        return None
    lo, hi = c
    return pd.Timestamp(lo), (pd.Timestamp(hi) if hi else None)


def _split_rows(df: pd.DataFrame, w) -> pd.DataFrame:
    d = pd.to_datetime(df["date"])
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_localize(None)
    m = d >= w[0]
    if w[1] is not None:
        m &= d <= w[1]
    out = df[m.to_numpy()].copy()
    out.attrs = dict(df.attrs)
    return out


def sealed_gate(reg: Registry, trial_id: str, ds: str) -> dict:
    """-> {'allowed': bool, 'replay': bool, 'why': str}. No side effects."""
    t = reg.trial(trial_id)
    cfg = config()[ds]
    prior = [a for a in reg.accesses(trial_id) if a["split"] == "sealed"]
    if prior:
        return {"allowed": False, "replay": False, "why": f"second unseal refused (first: {prior[0]['ts']})"}
    spent = [s for s in cfg.get("spent_unseals") or [] if s["spec_id"] == t["spec"]["id"]]
    state = reg.state(trial_id)
    if spent:
        return {"allowed": True, "replay": True, "why": f"replay of lockbox spent {spent[0]['when']} ({spent[0]['source']})",
                "gate_ok": state == "dev_pass" or (cfg["splits"].get("dev") is None)}
    if cfg["splits"].get("dev") is None:
        w = window(ds, "sealed")
        ok = w is not None and pd.Timestamp(t["ts"]).tz_localize(None) < w[0]
        return {"allowed": bool(ok), "replay": False, "gate_ok": bool(ok),
                "why": "prospective: registered before the sealed window" if ok else
                "dataset has no dev split and the trial was registered after the sealed window opened"}
    if state != "dev_pass":
        return {"allowed": False, "replay": False, "why": f"state is {state!r}; sealed needs dev_pass"}
    return {"allowed": True, "replay": False, "gate_ok": True, "why": "dev passed"}


def load(dataset: str, split: str, trial_id: str, reg: Registry | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """-> (frame for the rule [hidden columns removed], hidden outcome frame, access record)."""
    reg = reg or Registry()
    if split not in SPLITS:
        raise VaultError(f"split must be one of {SPLITS}")
    t = reg.trial(trial_id)
    if t["spec"]["dataset"] != dataset:
        raise VaultError(f"trial {trial_id} is registered on {t['spec']['dataset']!r}, not {dataset!r}")
    cfg = config()[dataset]
    w = window(dataset, split)
    if w is None:
        raise VaultError(f"{dataset} has no {split} split")
    gate = {"allowed": True, "replay": False}
    if split == "sealed":
        gate = sealed_gate(reg, trial_id, dataset)
        if not gate["allowed"]:
            raise VaultError(f"sealed refused for {trial_id}: {gate['why']}")
    lname = cfg["splits"].get(f"{split}_loader") or cfg["loader"]
    df, sources = loaders.LOADERS[lname]()
    rows = _split_rows(df, w)
    ocfg = outcome_cfg(dataset, split)
    hidden = [c for c in ocfg["hidden"] if c in rows.columns]
    rec = reg.append({"event": "access", "trial_id": trial_id, "dataset": dataset, "split": split,
                      "window": [str(w[0].date()), str(w[1].date()) if w[1] is not None else None],
                      "rows": int(len(rows)), "replay": bool(gate.get("replay")), "gate": gate.get("why"),
                      "vault_sha256": vault_sha(),
                      "sources": [{"path": s, "sha256": sha256_file(s)} for s in sources]})
    key = ocfg["outcome"]["key"]
    keys = [key] if isinstance(key, str) else list(key)
    hidden_df = rows[[c for c in keys if c in rows.columns] + hidden].copy()
    hidden_df.attrs = {"split": split}
    visible = rows.drop(columns=hidden)
    visible.attrs = dict(rows.attrs)
    return visible, hidden_df, {**rec, "gate_ok": gate.get("gate_ok", True)}


def outcome_cfg(dataset: str, split: str) -> dict:
    cfg = config()[dataset]
    if split == "forward" and "forward_outcome" in cfg:
        return {"outcome": cfg["forward_outcome"], "hidden": cfg.get("forward_hidden", [])}
    return {"outcome": cfg["outcome"], "hidden": cfg.get("hidden", [])}


def resolve_outcomes(dataset: str, bets: pd.DataFrame, hidden: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Attach `outcome` (1 / 0 / 0.5) to the rule's bet rows from the hidden columns. Rules may not set outcomes."""
    if "outcome" in bets.columns:
        raise VaultError("rules may not set `outcome`; the vault resolves it from the hidden columns")
    kind = outcome_cfg(dataset, hidden.attrs.get("split", "dev"))["outcome"]["kind"]
    b = bets.copy()
    if b.empty:
        b["outcome"] = pd.Series(dtype=float)
        return b
    if kind == "l32b_log":
        st = hidden[hidden["result"].notna()] if "result" in hidden else hidden.iloc[0:0]
        m = dict(zip(st["key"], st["result"]))
        b["outcome"] = [{"win": 1.0, "loss": 0.0, "push": 0.5}.get(m.get(k), np.nan) for k in b["bet_id"]]
        return b
    if kind == "won_column":
        m = dict(zip(hidden["ticker"], hidden["won"]))
        b["outcome"] = [float(m[k]) for k in b["bet_id"]]          # side must be the picked NO side
        if (b["side"] != "no").any():
            raise VaultError("won_column datasets settle the NO side only")
    elif kind == "won_push_columns":
        key = config()[dataset]["outcome"]["key"]
        h = hidden.assign(_k=hidden[key[0]].astype(str) + "|" + hidden[key[1]].astype(str))
        won = dict(zip(h["_k"], h["won"]))
        push = dict(zip(h["_k"], h["push"])) if "push" in h else {}
        ks = b["bet_id"].astype(str)
        b["outcome"] = [0.5 if push.get(k, 0) == 1 else float(won[k]) for k in ks]
    elif kind == "result_equals_side":
        m = dict(zip(hidden["ticker"], hidden["result"]))
        b["outcome"] = [1.0 if m[tk] == s else 0.0 for tk, s in zip(b["bet_id"].str.split("|").str[0], b["side"])]
    elif kind == "win_push_columns":
        h = hidden.assign(_k=hidden["event_id"].astype(str))
        win = dict(zip(h["_k"], h["win"]))
        push = dict(zip(h["_k"], h["push"]))
        b["outcome"] = [0.5 if bool(push[k]) else float(win[k]) for k in b["bet_id"].astype(str)]
    elif kind == "l2_side":
        h = hidden.assign(_k=hidden["trial"].astype(str) + "|" + hidden["event_id"].astype(str))
        hs, as_ = dict(zip(h["_k"], h["home_score"])), dict(zip(h["_k"], h["away_score"]))
        out = []
        for k, s in zip(b["bet_id"].astype(str), b["side"]):
            d = hs[k] - as_[k]
            out.append(0.5 if d == 0 else float((d > 0) == (s == "home")))
        b["outcome"] = out
    elif kind == "none":
        b["outcome"] = np.nan
    else:
        raise VaultError(f"unknown outcome kind {kind}")
    return b
