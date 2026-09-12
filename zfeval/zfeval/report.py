"""Assemble results into a JSON record, a markdown summary, and a regression comparison."""
from __future__ import annotations
import json, subprocess, datetime, platform
from pathlib import Path
import numpy as np


def provenance(ckpt_meta: dict | None = None, extra: dict | None = None):
    def sh(c):
        try:
            return subprocess.run(c, shell=True, capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except Exception:
            return None
    return dict(utc=datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                git_sha=sh("git rev-parse --short HEAD"),
                git_dirty=bool(sh("git status --porcelain")),
                host=platform.node(), python=platform.python_version(),
                checkpoint=ckpt_meta or {}, **(extra or {}))


class Report:
    def __init__(self, name, out_dir, ckpt_meta=None):
        self.name = name
        self.dir = Path(out_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.data = dict(name=name, provenance=provenance(ckpt_meta), sections={})

    def add(self, section, payload):
        self.data["sections"][section] = _jsonable(payload)
        return self

    def save(self):
        p = self.dir / "report.json"
        p.write_text(json.dumps(self.data, indent=2))
        (self.dir / "SUMMARY.md").write_text(self.markdown())
        return p

    def markdown(self):
        S = self.data["sections"]; P = self.data["provenance"]
        L = [f"# {self.name}", "",
             f"Generated {P['utc']} · git `{P.get('git_sha')}`"
             f"{' (dirty tree)' if P.get('git_dirty') else ''}"]
        ck = P.get("checkpoint") or {}
        if ck:
            L += ["", f"Checkpoint `{ck.get('path')}` — epoch {ck.get('epoch')}, "
                      f"step {ck.get('global_step')}, encoder fingerprint "
                      f"`{ck.get('encoder_fingerprint')}`"]
        if "scores" in S:
            L += ["", "## Detection scores", "",
                  "| eval | layer | n | majority | AUC | AP | acc |", "|---|---|---|---|---|---|---|"]
            for r in S["scores"]:
                L.append(f"| {r['split']} | {r.get('layer','-')} | {r['n']} | {r['majority']:.4f} "
                         f"| {r['auc']:.4f} | {r['ap']:.4f} | {r['acc']:.4f} |")
            L += ["", "_Accuracy is meaningless without the majority rate beside it._"]
        if "baselines" in S:
            L += ["", "## Baselines", "", "| eval | model | log-mel | log-energy | model − mel |",
                  "|---|---|---|---|---|"]
            for ev, d in S["baselines"].items():
                L.append(f"| {ev} | {d['model']:.4f} | {d['logmel']:.4f} | {d['energy']:.4f} "
                         f"| **{d['model']-d['logmel']:+.4f}** |")
            L += ["", "_If the model does not beat log-mel, the pretraining bought nothing._"]
        if "operating_points" in S:
            L += ["", "## What a false-alarm budget buys", ""]
            for ev, d in S["operating_points"].items():
                L.append(f"**{ev}** — " + ", ".join(
                    f"{k.replace('fa_','')} FA → {v['calls_found']} calls "
                    f"({v['recall']*100:.1f}%)" for k, v in d.items()))
        if "controls" in S:
            L += ["", "## Controls", ""]
            for k, v in S["controls"].items():
                ok = v.get("passed", v.get("matching_unbiased", v.get("energy_null_ok")))
                mark = {True: "PASS", False: "**FAIL**", None: ""}[ok]
                detail = ""
                if "median" in v and "n_below_0p80" in v:       # per_group has no pass/fail
                    detail = (f"median AUC {v['median']:.4f}, IQR [{v['q25']:.4f}, {v['q75']:.4f}], "
                              f"min {v['min']:.4f}, {v['n_below_0p80']}/{v['n_groups']} below 0.80")
                elif "mean_auc" in v:
                    detail = f"shuffled AUC {v['mean_auc']:.4f} ± {v['std']:.4f} (null 0.5)"
                elif "energy_win_rate" in v:
                    detail = (f"model wins {v['model_win_rate']:.4f} of {v['n_pairs']} "
                              f"equal-loudness pairs, energy {v['energy_win_rate']:.4f} (null 0.5)")
                elif "weighted_energy_auc" in v:
                    detail = (f"within-band model {v['weighted_model_auc']:.4f}, "
                              f"energy {v['weighted_energy_auc']:.4f} (null 0.5)")
                if "error" in v:
                    detail += f" — {v['error']}"
                L.append(f"- `{k}`: {mark}{' — ' if mark and detail else ''}{detail}")
        if "geometry" in S:
            L += ["", "## Embedding geometry", "",
                  "| layer | AUC | kNN-10 | silhouette | Fisher | PCA dim@90% |",
                  "|---|---|---|---|---|---|"]
            for l, d in S["geometry"].items():
                L.append(f"| {l} | {d['auc']:.4f} | {d['knn10_acc']:.4f} | {d['silhouette']:.3f} "
                         f"| {d['fisher_ratio']:.4f} | {d['pca_dim_90']} |")
        if "events" in S:
            L += ["", "## Onset / offset", "",
                  "| feature | collar-50 F1 | overlap F1 | P | R |", "|---|---|---|---|---|"]
            for k, d in S["events"].items():
                o = d["overall"]
                L.append(f"| {k} | {o['collar_f1']:.3f} | {o['overlap_f1']:.3f} "
                         f"| {o['collar_precision']:.3f} | {o['collar_recall']:.3f} |")
                if d.get("on_grid_edge"):
                    L.append(f"| | _grid edge: {', '.join(d['on_grid_edge'])} — widen it_ | | | |")
        L += ["", "## Files", ""] + [f"- `{p.name}`" for p in sorted(self.dir.glob("*")) ]
        return "\n".join(L)


def compare(new_path, base_path, out_path=None, tol=0.005):
    """Regression view: what moved between two runs, and by how much.

    The point of the suite is that retraining produces a comparable record. This prints the deltas
    so a regression is visible instead of being discovered later.
    """
    new = json.loads(Path(new_path).read_text())
    base = json.loads(Path(base_path).read_text())
    lines = [f"# {new['name']} vs {base['name']}", "",
             f"new: git `{new['provenance'].get('git_sha')}`, ckpt "
             f"`{(new['provenance'].get('checkpoint') or {}).get('encoder_fingerprint')}`",
             f"base: git `{base['provenance'].get('git_sha')}`, ckpt "
             f"`{(base['provenance'].get('checkpoint') or {}).get('encoder_fingerprint')}`", ""]
    ns = {(r["split"], r.get("layer")): r for r in new["sections"].get("scores", [])}
    bs = {(r["split"], r.get("layer")): r for r in base["sections"].get("scores", [])}
    common = sorted(set(ns) & set(bs), key=lambda k: (str(k[0]), str(k[1])))
    if common:
        lines += ["## Detection scores", "", "| eval | layer | base AUC | new AUC | delta |",
                  "|---|---|---|---|---|"]
        for k in common:
            d = ns[k]["auc"] - bs[k]["auc"]
            flag = "" if abs(d) < tol else (" **better**" if d > 0 else " **WORSE**")
            lines.append(f"| {k[0]} | {k[1] or '-'} | {bs[k]['auc']:.4f} | {ns[k]['auc']:.4f} "
                         f"| {d:+.4f}{flag} |")
    only_new = sorted(set(ns) - set(bs)); only_base = sorted(set(bs) - set(ns))
    if only_new or only_base:
        lines += ["", f"_present only in new: {only_new or 'none'}; "
                      f"only in base: {only_base or 'none'}_"]
    lines += ["", "Deltas are point estimates. A difference inside the bootstrap interval is not "
                  "a regression — run `metrics.paired_bootstrap` on the stored predictions before "
                  "acting on any row above."]
    txt = "\n".join(lines)
    if out_path:
        Path(out_path).write_text(txt)
    return txt


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if hasattr(o, "to_dict"):
        return o.to_dict()
    return o
