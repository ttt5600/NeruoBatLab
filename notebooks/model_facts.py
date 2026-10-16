#!/usr/bin/env python
"""Model facts: what each scored encoder IS -- init, corpus, targets, compute, size -- and which file each
value came from. There are no accuracies or other scores in this table, on purpose: it says what the
models are, so that a gap between two of them can be read against what actually differs.

    /Library/Frameworks/Python.framework/Versions/3.10/bin/python3 notebooks/model_facts.py
        -> notebooks/model_facts.csv, and prints the empty cells per column

Rows: every encoder tag that has call-type results -- each <tag>_calltype.json in analysis/ (except the
three files that are not one encoder's per-layer probe) plus the models in aves_variants_calltype.json.

Every value is read at run time:
  - JSON values by key (compute_budget.json, run15_compute.json, detection_variants.json, the call-type JSONs);
  - training-script values by parsing the train.py call in the slurm script (wrapper `export`s, positional
    arguments and `case` arms resolved the way bash would);
  - harness/build_board.py MODELS and harness/build_comparison.py META by evaluating those literals (no import,
    so nothing is written under harness/);
  - values that exist only in prose (findings, experiments.yaml `change` text, script comments) are written
    here once and passed through `said()`, which raises unless the exact text occurs in the cited file.
A cell that no source gives is left empty (NaN) and reported by `unknowns()`. Nothing is filled by analogy:
run16's measured audio per update is NOT copied to run18 even though run18 used run16's script shape.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
ANA = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
SLURM = ROOT / "pytorchAudio/examples/hubert/slurm"
TRAIN_PY = ROOT / "pytorchAudio/examples/hubert/train.py"
DATASETS_TSV = ROOT / "pytorchAudio/examples/hubert/savio/datasets.tsv"
EXPERIMENTS = ROOT / "harness/experiments.yaml"
FINDINGS = ROOT / "knowledge/findings"
OUT_CSV = Path(__file__).resolve().parent / "model_facts.csv"

FAM_SCRATCH, FAM_DAPT, FAM_AVES, FAM_HF = ("ours, from scratch", "DAPT from AVES",
                                           "downloaded AVES/BirdAVES", "off-the-shelf HF")
NOT_ONE_ENCODER = {"aves", "aves_variants", "finetune"}   # *_calltype.json files that are not an encoder's probe

COLUMNS = [
    "tag", "display_name", "family", "init", "training_corpus", "corpus_hours", "targets", "objective",
    "input", "updates", "warmup", "lr", "accumulate", "gpus", "gpus_requested", "audio_per_update_s",
    "audio_per_update_cap_s", "total_audio_h", "epochs", "mask", "seed", "params", "layers", "embed_dim",
    "saw_test_birds", "experiment", "finding", "baseline", "differs_from_baseline", "description", "notes",
    "source",
]
# what each column means; shown in the notebook
COLUMN_NOTES = {
    "init": "weights training started from (random = from scratch)",
    "training_corpus": "audio of THIS model's own training stage (a DAPT model's earlier AVES corpus is in `init`)",
    "corpus_hours": "hours of distinct audio in that corpus",
    "updates": "optimiser steps of the scored checkpoint's schedule (see notes for checkpoints saved short of it)",
    "gpus": "GPUs actually used = srun ranks (#SBATCH --ntasks / --ntasks-per-node), measured where a JSON has it",
    "gpus_requested": "GPUs reserved (#SBATCH --gres); run11/run15 reserved 4 and used 1",
    "audio_per_update_s": "MEASURED seconds of audio per optimiser step (checkpoint step counts / corpus size)",
    "audio_per_update_cap_s": "train.py --seconds-per-batch x GPUs x accumulation: the batch CAP, not realised",
    "total_audio_h": "MEASURED audio seen, counting repeats (compute, not information)",
    "epochs": "MEASURED passes over the corpus",
    "saw_test_birds": "trained on recordings of the birds in the call-type probe (finding 028)",
}

# ---------------------------------------------------------------------------------------------- sources
_TXT: dict[Path, str] = {}


def _text(p) -> str:
    p = Path(p)
    if p not in _TXT:
        _TXT[p] = p.read_text()
    return _TXT[p]


def rel(p) -> str:
    p = Path(p)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return "analysis/" + p.name if p.parent == ANA else str(p)


def finding_file(fid: str) -> Path:
    hits = sorted(FINDINGS.glob(f"{fid}-*.yaml"))
    if not hits:
        raise FileNotFoundError(f"no finding {fid}")
    return hits[0]


def said(path, s: str) -> str:
    """Return `s` after checking it occurs verbatim (whitespace-normalised) in `path`; raise otherwise."""
    norm = lambda x: " ".join(x.split())
    body = re.sub(r"(?m)^\s*#+", " ", _text(path))          # comment text may wrap over '# ' lines
    if str(path).endswith(".yaml"):                            # match the parsed text: raw YAML escapes quotes
        def strings(v):
            if isinstance(v, dict):
                return " ".join(strings(x) for x in v.values())
            if isinstance(v, list):
                return " ".join(strings(x) for x in v)
            return str(v)
        body += " " + strings(yaml.safe_load(_text(path)))
    if norm(s) not in norm(body):
        raise ValueError(f"{s!r} is not in {rel(path)} -- a fact here has gone stale")
    return s


_JSON: dict[str, dict] = {}


def jload(name: str) -> dict:
    """Parsed analysis JSON, cached (read-only: callers must not mutate it)."""
    if name not in _JSON:
        _JSON[name] = json.loads(_text(ANA / name))
    return _JSON[name]


def jget(name: str, *keys):
    v = jload(name)
    for k in keys:
        v = v[k]
    return v


def jsrc(name: str, *keys) -> str:
    return f"analysis/{name}:{'.'.join(keys)}"


def _harness_literals(path: Path, names: set[str]) -> dict:
    """Evaluate the named module-level assignments of a harness script without importing it."""
    def targets(node):
        for t in node.targets:
            yield from (t.elts if isinstance(t, ast.Tuple) else [t])
    tree = ast.parse(_text(path))
    body = [n for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in names for t in targets(n))]
    ns: dict = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), {}, ns)
    return ns


BOARD = {m[0]: dict(zip(("tag", "name", "fam", "what", "finding", "registry"), m))
         for m in _harness_literals(ROOT / "harness/build_board.py", {"MODELS"})["MODELS"]}
META = {k: dict(zip(("name", "family", "input", "objective", "data", "hours", "saw"), v))
        for k, v in _harness_literals(ROOT / "harness/build_comparison.py",
                                      {"ZF", "ZFFSD", "MIX", "META"})["META"].items()}
EXPS = yaml.safe_load(_text(EXPERIMENTS))["experiments"]
EXP_BY_ID = {e["id"]: e for e in EXPS}
EXP_BY_TAG = {e["tag"]: e for e in EXPS if e.get("tag")}


def said_exp(eid: str, field: str, s: str) -> str:
    """Like said(), against the PARSED experiments.yaml field (the raw file escapes quotes)."""
    norm = lambda x: " ".join(str(x).split())
    if norm(s) not in norm(EXP_BY_ID[eid].get(field, "")):
        raise ValueError(f"{s!r} is not in experiments.yaml:{eid}.{field} -- a fact here has gone stale")
    return s


def trainpy_default(flag: str) -> str:
    m = re.search(rf'"{re.escape(flag)}",\s*default=([^,\s]+),', _text(TRAIN_PY))
    if not m:
        raise ValueError(f"no default for {flag} in train.py")
    return m[1]


# ---------------------------------------------------------------------------------------------- slurm
def _resolve(s: str, var: dict, env: dict, seen: frozenset = frozenset()) -> str:
    """Expand ${X}, ${X:-d}, ${X:?msg}, ${X//pat/} and $X the way bash would, from env then script vars.
    Inside X's own definition (X="${X:-3}") X means the environment only, as in bash."""
    def val(name):
        if name in env:
            return env[name]
        if name in var and name not in seen:
            return _resolve(var[name], var, env, seen | {name})
        return None

    def brace(m):
        name, op, rest = re.match(r"(\w+)(?:(:-|:\?|//)(.*))?$", m[1]).groups()
        v = val(name)
        if op == ":-":
            return rest if not v else v
        if op == ":?" and v is None:
            raise ValueError(f"${{{name}}} is required: {rest}")
        return v or ""

    while re.search(r"\$\{[^${}]*\}", s):
        s = re.sub(r"\$\{([^${}]*)\}", brace, s, count=1)
    return re.sub(r"\$(\w+)", lambda m: val(m[1]) or "", s)


def recipe(script: str, pos: tuple = (), arm: str | None = None) -> dict:
    """Resolve what a training script passed to train.py. `script` is what sbatch ran: if it `exec`s
    another script, that one's call is read with this one's exports. `pos` are the sbatch positional
    arguments, `arm` the `case` arm they select."""
    t = _text(SLURM / script)
    sb = dict(re.findall(r"^#SBATCH --([\w-]+)=?(\S*)", t, re.M))
    env = dict(re.findall(r"^export (\w+)=(\S+)", t, re.M))
    chain = [script]
    m = re.search(r"^exec bash \S*/([\w.]+\.sh)", t, re.M)
    if m:
        chain.append(m[1])
        t = _text(SLURM / m[1])
    for i, p in enumerate(pos, 1):
        env[str(i)] = p
    var: dict = {}
    for k, v in re.findall(r"^\s*([A-Z][A-Z0-9_]*)=(.*)$", t, re.M):
        v = re.sub(r"\s+#.*$", "", v).strip()
        v = v[1:-1] if len(v) > 1 and v[0] == v[-1] == '"' else v
        var.setdefault(k, v)                       # first top-level assignment; later ones sit in if/else
    extra = []
    if arm is None and re.search(r'case "\$\{ARM\}" in', t):   # the arm the script defaults to
        arm = _resolve("${ARM}", var, env)
    if arm:
        a = re.search(rf"^\s*{arm}\)\s*(.*?);;", t, re.M)
        if not a:
            raise ValueError(f"{script}: no case arm {arm}")
        for k, v in re.findall(r"([A-Z][A-Z0-9_]*)=(\"[^\"]*\"|[^;\s]+)", a[1]):
            var[k] = v.strip('"')
        extra = re.findall(r"EXTRA\+=\((--[\w-]+)\)", a[1])
    call = re.search(r"^\s*srun python \S*train\.py\S*\s*\\\n((?:.*\\\n)*.*)", t, re.M)
    toks = call[1].replace("\\\n", " ").split()
    flags = {}
    for j, tok in enumerate(toks):
        if tok.startswith("--"):
            nxt = toks[j + 1] if j + 1 < len(toks) else ""
            flags[tok] = (True if (not nxt or nxt.startswith(("--", "&&", "||")) or "[@]" in nxt)
                          else _resolve(nxt.strip('"'), var, env))
    for f in extra:
        flags[f] = True
    seed_env = re.search(r'^SEED="\$\{(\w+):-\}"', t, re.M)
    mask_env = re.search(r'^MASK="\$\{(\w+):-\}"', t, re.M)
    seed = flags.get("--seed") or (env.get(seed_env[1]) if seed_env else None)
    mask = (flags.get("--mask-prob"), flags.get("--mask-length"))
    if mask_env and env.get(mask_env[1]):
        mask = tuple(env[mask_env[1]].split(","))
    init = flags.get("--init-weights") or (_resolve("${AVES}", var, env) if '"${START[@]}"' in toks else None)
    audio = _resolve("${MIX}", var, env) if "MIX" in var else flags["--dataset-path"]
    ranks = int(sb.get("ntasks-per-node") or sb.get("ntasks"))
    return dict(
        chain=chain, arm=arm, sbatch=sb, env=env, flags=flags, audio=audio, init=init,
        lr=float(flags["--learning-rate"]), updates=int(flags["--max-updates"]),
        warmup=int(flags["--warmup-updates"]), k=int(flags["--num-classes"]),
        accumulate=int(flags.get("--accumulate-grad-batches") or trainpy_default("--accumulate-grad-batches")),
        seed=int(seed) if seed else int(trainpy_default("--seed")), seed_default=not seed,
        mask=mask, freeze=bool(flags.get("--freeze-feature-extractor")),
        ranks=ranks, gres=int(sb["gres"].rsplit(":", 1)[1]),
        feature_weight=flags.get("--feature-weight"),
    )


def ssrc(r: dict, what: str) -> str:
    s = "slurm/" + " -> ".join(r["chain"])
    return f"{s}:{what}"


# ---------------------------------------------------------------------------------------------- corpora
def _corpora():
    cb, rc = "compute_budget.json", "run15_compute.json"
    f058, f060 = finding_file("058"), finding_file("060")
    zf_h = jget(cb, "run11", "corpus_hours")
    comb_h = jget(rc, "run15", "corpus_hours")
    said(DATASETS_TSV, "ZF + FSD50K, 763 files / 224.42 h")
    x1 = float(said(f058, "116.03")) + float(said(f058, "27.79"))
    said(f058, "mix_zf_fsd50k1 = run11 corpus 116.03 h + 27.79 h FSD50K labelled with run11's own k-means")
    e4, e5, e6 = EXP_BY_ID["E4"]["change"], EXP_BY_ID["E5"]["change"], EXP_BY_ID["E6"]["change"]
    return [  # (path fragment, corpus, short, hours, source)
        ("run5-4-26-full/data/spectrogram", "ZF colony (120 recordings)", "ZF", zf_h,
         jsrc(cb, "run11", "corpus_hours") + "; path = datasets.tsv zf_it1 (run11's training set)"),
        ("combined_zf_fsd/", "ZF colony + FSD50K (dev + eval)", "ZF+FSD50K", comb_h,
         jsrc(rc, "run15", "corpus_hours") + "; datasets.tsv combined_it1/combined_it2"),
        ("combined_aves_teacher/", "ZF colony + FSD50K (dev + eval)", "ZF+FSD50K", comb_h,
         jsrc(rc, "run15", "corpus_hours") + "; finding 059 'Same 224 h corpus'"),
        ("fsd_only_subset", "FSD50K rows of the combined corpus only", "FSD50K",
         float(re.search(r"only \((\d+\.\d+) h\)", e4)[1]), "experiments.yaml:E4.change"),
        ("zf_only_subset", "ZF rows of the combined corpus only", "ZF",
         float(re.search(r"only \((\d+\.\d+) h\)", e6)[1]), "experiments.yaml:E6.change"),
        ("mix_zf_fsd50k1", "ZF colony + FSD50K-eval replay x1 (" + said(f060, "19.3%") + " of the stream)",
         "ZF+FSD50K-eval replay", x1, "finding 058 method (116.03 h + 27.79 h); 19.3% from finding 060"),
        ("mix_zf_fsd50k3", "ZF colony + FSD50K-eval replay x3 (" + said_exp("E5", "change", "41.8%") + " of the stream)",
         "ZF+FSD50K-eval replay x3", float(re.search(r"of a (\d+\.\d+) h stream", e5)[1]),
         "experiments.yaml:E5.change"),
    ]


def corpus_of(path: str):
    for frag, name, short, hours, src in CORPORA:
        if frag in path:
            return name, short, hours, src
    raise KeyError(f"no corpus entry for {path}")


# ---------------------------------------------------------------------------------------------- targets
def _targets():
    cb = "compute_budget.json"
    k1 = f"k-means k={jget(cb, 'run11', 'num_clusters')} on {jget(cb, 'run11', 'target_feature')}"
    yid = lambda i: EXP_BY_ID[i]["change"]
    return {
        "run11": (k1, jsrc(cb, "run11", "target_feature") + "," + jsrc(cb, "run11", "num_clusters")),
        "run11_labels": ("run11's labels: " + k1, jsrc(cb, "run11", "target_feature")
                         + "; slurm DATA = run11's label dir (datasets.tsv zf_it1)"),
        "run11_replay": ("run11's spectrogram k-means k=100, FSD50K labelled with it ("
                         + said(finding_file("060"), "run11's k=100 k-means, not refit") + ")",
                         "finding 060 method; finding 058 method"),
        "combined_it1": ("spectrogram k-means k=200 (iteration 1)",
                         "datasets.tsv combined_it1: '" + said(DATASETS_TSV, "iteration-1 labels k=200") + "'"),
        "subset_it1": ("spectrogram k-means k=200 (iteration 1), the combined corpus's labels",
                       "experiments.yaml:E4.change/E6.change: '" + said_exp("E4", "change", "same k=200 vocabulary and labels") + "'"),
        "iter2_run16": ("iteration 2: run16 layer 6 k-means k=200",
                        "experiments.yaml:E1.change: '" + said_exp("E1", "change", "iteration-2 run16 layer 6 k=200") + "'"),
        "aves_l6": ("AVES-bio layer 6 k-means k=200",
                    "experiments.yaml:E3.change: '" + said_exp("E3", "change", "AVES layer 6 k=200 on the combined corpus") + "'"),
        "aves_l6_dapt": ("AVES-bio layer 6 k-means k=200 (run19's labels)",
                         "experiments.yaml:E17.change: '" + said_exp("E17", "change", "AVES layer 6 k=200 (combined_aves_teacher, run19's)") + "'"),
        "two_teacher": ("k-means k=200 on [run16 L6 | AVES L6], each standardised (two-teacher)",
                        "experiments.yaml:E19.change: '" + said_exp("E19", "change", "k=200 on [run16 L6 | AVES L6], each teacher standardised") + "'"),
        "aves_only_b": ("k-means k=200 on AVES L6 alone, standardised (two-teacher recipe, --only-b)",
                        "experiments.yaml:E21.change: '" + said_exp("E21", "change", "--only-b -- AVES layer 6 alone, same per-dimension standardisation") + "'"),
        "aves_bio_own": (f"k-means on {jget(cb, 'aves', 'target_feature')}",
                         jsrc(cb, "aves", "target_feature") + "," + jsrc(cb, "aves", "num_clusters")),
    }


CORPORA = _corpora()
TARGETS = _targets()

# ---------------------------------------------------------------------------------------------- recipes
# One entry per trained-here tag: the script sbatch ran (and the source naming it), positional args / case
# arm, the targets key, experiment id, baseline, the one-phrase difference, and finding(s) where the
# experiments.yaml / build_board.py fields do not carry them. Every phrase cites its source.
F = finding_file


def _round1(lr_str: str, step: int) -> dict:
    # the tag is dapt<LR without . and ->_step<N> (export_dapt_checkpoints.sh); check that rule holds
    said(SLURM / "train_dapt_aves_init.sh", 'TAG="${LR//[.-]/}"')
    said(SLURM / "export_dapt_checkpoints.sh", 'dapt${TAG}_step${step}.pt')
    return dict(script="train_dapt_aves_init.sh", script_src="finding 045 provenance", pos=(lr_str,),
                pos_src="finding 045 evidence ('lr " + lr_str + "') and the tag rule TAG=${LR//[.-]/}",
                targets="run11_labels", baseline="aves-base-bio",
                differs=f"continued on ZF only with run11's k=100 labels, lr {lr_str}, checkpoint at step {step:,}",
                differs_src="slurm/train_dapt_aves_init.sh (DATA = run11's labels; checkpoint every 2500); "
                            "finding 045 '" + said(F("045"), "AVES-init, run11's exact corpus and k=100 spectrogram labels") + "'",
                updates_override=step,
                finding="045, 047" if step == 15000 else "047", finding_src="finding 045 (step 15000) / 047 (all 12)")


RECIPES = {
    "run11": dict(script=jget("compute_budget.json", "run11", "sources", "script").rsplit("/", 1)[1],
                  script_src=jsrc("compute_budget.json", "run11", "sources", "script"), targets="run11",
                  finding="044", finding_src="finding 044 (run11's compute budget)"),
    "run15_combined": dict(script="train_run15_combined.sh", script_src="finding 051 method", targets="combined_it1",
                           baseline="run11",
                           differs="corpus ZF 116 h -> ZF+FSD50K 224.42 h, and k 100 -> 200",
                           differs_src="slurm/train_run15_combined.sh header: '" + said(
                               SLURM / "train_run15_combined.sh", "num-classes 100 -> 200") + "'"),
    "run16_compute4x": dict(script="train_run16_compute4x.sh", script_src="finding 052 method",
                            targets="combined_it1", baseline="run15_combined",
                            differs="the 4 requested GPUs actually used: 4x audio per update",
                            differs_src="slurm/train_run16_compute4x.sh header: '" + said(
                                SLURM / "train_run16_compute4x.sh",
                                "run16 = run15 with exactly ONE change: the four GPUs that every run already requests are finally USED") + "'"),
    "run16_seed2": dict(targets="combined_it1"),
    "run16_seed3": dict(targets="combined_it1"),
    "run17_accum2": dict(script="train_run17_accum2.sh", script_src="finding 054 method", targets="combined_it1",
                         baseline="run16_compute4x", differs="+ --accumulate-grad-batches 2: 2x audio per update",
                         differs_src="finding 054 method: '" + said(
                             F("054"), "train_run16_compute4x.sh + --accumulate-grad-batches 2") + "'"),
    "run18_iter2": dict(targets="iter2_run16", differs="targets: spectrogram k=200 -> run16 layer 6 k=200 (iteration 2)"),
    "run19_avesteacher": dict(targets="aves_l6", differs="targets: spectrogram k=200 -> AVES-bio layer 6 k=200"),
    "run20_fsdonly": dict(targets="subset_it1", baseline="run21_zfonly",
                          differs="corpus: ZF rows -> FSD50K rows of the combined corpus (same labels)"),
    "run21_zfonly": dict(targets="subset_it1", differs="corpus: ZF+FSD50K -> its ZF rows only (same labels)",
                         notes="scored weights are step " + said(F("061"), "93633") + " (the step-93750 checkpoint "
                               "was not kept; finding 061)"),
    "run29_avesrecipe": dict(targets="combined_it1", baseline="run17_accum2", experiment="E29a",
                             experiment_src="finding 073 method ('E29a')",
                             differs="AVES-bio's mask 0.8/10, lr 2e-4, warmup 32000, 100k updates",
                             finding="073", finding_src="finding 073 method (train_run29_avesrecipe.sh)",
                             notes="scored at " + said(F("073"), "epoch=78-step=99030")
                                   + " (step-100000 checkpoint not kept; finding 073)"),
    "run29_avesrecipe_s2": dict(script="train_run29_avesrecipe_seed2.sh",
                                script_src="experiments.yaml:E29.script ('seed 2 = train_run29_avesrecipe_seed2.sh')",
                                targets="combined_it1", baseline="run29_avesrecipe", experiment="E29b",
                                experiment_src="finding 073 method ('E29b')", differs="--seed 2",
                                differs_src="slurm/train_run29_avesrecipe_seed2.sh (export RUN29_SEED=2)",
                                finding="073", finding_src="finding 073 method"),
    # ---- DAPT round 2 (train_dapt_v2.sh arms, each one change against round 1's lr 5e-5 run)
    "daptreplay_5e5_step15000": dict(script="train_dapt_v2.sh", script_src="finding 058 method", arm="replay",
                                     targets="run11_replay", baseline="dapt5e5_step15000",
                                     differs="+ FSD50K-eval replay mixed into the ZF stream",
                                     differs_src="slurm/train_dapt_v2.sh header (replay arm)",
                                     finding="058", finding_src="finding 058"),
    "daptfreeze_5e5_step15000": dict(script="train_dapt_v2.sh", script_src="finding 058 evidence (freeze arm)",
                                     arm="freeze", targets="run11_labels", baseline="dapt5e5_step15000",
                                     differs="conv feature extractor frozen",
                                     differs_src="slurm/train_dapt_v2.sh freeze arm (--freeze-feature-extractor)",
                                     finding="058", finding_src="finding 058 evidence"),
    "daptlr_1e5_step15000": dict(script="train_dapt_v2.sh", script_src="finding 058 evidence (lr arm)", arm="lr",
                                 pos=("lr", said(F("058"), "1e-5")), pos_src="finding 058 evidence ('lr 1e-5')",
                                 targets="run11_labels", baseline="dapt5e5_step15000",
                                 differs="lr 5e-5 -> 1e-5", differs_src="slurm/train_dapt_v2.sh lr arm; finding 058",
                                 finding="058", finding_src="finding 058 evidence"),
    "daptshort_5e5_step2500": dict(script="train_dapt_v2.sh", script_src="finding 058 evidence (short arm)",
                                   arm="short", targets="run11_labels", baseline="dapt5e5_step2500",
                                   differs="schedule sized for 2,500 updates (warmup 125, full decay) vs a 15k "
                                           "schedule stopped at 2,500",
                                   differs_src="slurm/train_dapt_v2.sh header: '" + said(
                                       SLURM / "train_dapt_v2.sh", "2,500 updates with the schedule actually SIZED for 2,500") + "'",
                                   finding="058", finding_src="finding 058 evidence"),
    "daptreplay3_5e5_step15000": dict(targets="run11_replay", baseline="daptreplay_5e5_step15000",
                                      differs="the same FSD50K replay repeated 3x"),
    # ---- DAPT rounds 3-4
    "daptv3aves_5e5_step15000": dict(targets="aves_l6_dapt", baseline="daptreplay_5e5_step15000",
                                     differs="targets: run11 k=100 -> AVES-bio L6 k=200"),
    "daptv3aves_2e4_step15000": dict(targets="aves_l6_dapt", baseline="daptv3aves_5e5_step15000",
                                     differs="peak lr 5e-5 -> 2e-4"),
    "daptv3aves_2e4s2_step15000": dict(targets="aves_l6_dapt", baseline="daptv3aves_2e4_step15000", differs="--seed 2"),
    "daptv3aves_2e4m8_step15000": dict(targets="aves_l6_dapt", baseline="daptv3aves_2e4_step15000", experiment="E28a",
                                       experiment_src="harness/build_board.py MODELS name '(E28a)'",
                                       differs="masking 0.4/3 -> 0.8/10 (AVES-bio's)"),
    "daptv3aves_2e4s2m8_step15000": dict(script="train_dapt_v3_avesteacher_lr2e4_avesmask_seed2.sh",
                                         script_src="experiments.yaml:E28.script ('seed 2 = _avesmask_seed2.sh')",
                                         targets="aves_l6_dapt", baseline="daptv3aves_2e4s2_step15000", experiment="E28b",
                                         experiment_src="harness/build_board.py MODELS name '(E28b)'",
                                         differs="masking 0.4/3 -> 0.8/10 (AVES-bio's)",
                                         differs_src="harness/build_board.py MODELS: '" + BOARD["daptv3aves_2e4s2m8_step15000"]["what"] + "'",
                                         finding="071", finding_src="finding 071 method"),
    "daptv4two_2e4_step15000": dict(targets="two_teacher", baseline="daptv3aves_2e4_step15000",
                                    differs="targets: AVES L6 -> two-teacher [run16 L6 | AVES L6]"),
    "daptv4aveso_2e4_step15000": dict(targets="aves_only_b", baseline="daptv4two_2e4_step15000",
                                      differs="two-teacher recipe with AVES's block only (--only-b)"),
    "daptv4bird_2e4_step15000": dict(targets="two_teacher", baseline="daptv4two_2e4_step15000",
                                     baseline_src="harness/build_board.py MODELS: '" + BOARD["daptv4bird_2e4_step15000"]["what"] + "'",
                                     differs="init aves-base-bio -> birdaves-biox-base",
                                     differs_src="harness/build_board.py MODELS: '" + BOARD["daptv4bird_2e4_step15000"]["what"] + "'"),
    "daptv4two_2e4s2_step15000": dict(targets="two_teacher", baseline="daptv4two_2e4_step15000", differs="--seed 2"),
    "daptv4aveso_2e4s2_step15000": dict(targets="aves_only_b", baseline="daptv4aveso_2e4_step15000", differs="--seed 2"),
    "daptv4two45k_2e4_step45000": dict(targets="two_teacher", baseline="daptv4two_2e4_step15000",
                                       differs="45,000 updates instead of 15,000 (3x longer)"),
}
for _lr, _short in (("5e-5", "5e5"), ("1e-4", "1e4")):
    said(F("045"), f"lr {_lr}")
    for _step in (2500, 5000, 7500, 10000, 12500, 15000):
        RECIPES[f"dapt{_short}_step{_step}"] = _round1(_lr, _step)

# measured compute: tag -> (json, key)
MEASURED = {"run11": ("compute_budget.json", "run11"), "aves-base-bio": ("compute_budget.json", "aves"),
            "run15_combined": ("run15_compute.json", "run15"), "run16_compute4x": ("run15_compute.json", "run16"),
            "run17_accum2": ("run15_compute.json", "run17")}


# ---------------------------------------------------------------------------------------------- rows
def scored_tags() -> list[str]:
    tags = []
    for p in sorted(ANA.glob("*_calltype.json")):
        stem = p.name[: -len("_calltype.json")]
        if stem in NOT_ONE_ENCODER:
            continue
        tags.append(jload(p.name).get("tag", stem))            # run15_calltype.json -> run15_combined
    tags += [m for m in jget("aves_variants_calltype.json", "models") if m not in tags]
    return tags


def _calltype_json(tag: str) -> tuple[str, dict]:
    name = "run15_calltype.json" if tag == "run15_combined" else f"{tag}_calltype.json"
    return name, (jload(name) if (ANA / name).exists() else {})


class _Row(dict):
    def put(self, col, val, src):
        if val is None or (isinstance(val, str) and not val.strip()):
            return
        self[col] = val
        self.setdefault("_src", {})[col] = src


def _row(tag: str) -> dict:
    r = _Row(tag=tag)
    board, meta, exp = BOARD.get(tag), META.get(tag), EXP_BY_TAG.get(tag)
    ct_name, ct = _calltype_json(tag)
    av = jget("aves_variants_calltype.json", "models").get(tag)
    det = jload("detection_variants.json")
    spec = RECIPES.get(tag, {})

    # ---- identity
    if board:
        r.put("display_name", board["name"], "harness/build_board.py:MODELS")
    elif meta:
        r.put("display_name", meta["name"], "harness/build_comparison.py:META")
    elif tag.startswith("dapt"):
        r.put("display_name", f"DAPT round 1, lr {spec['pos'][0]}, step {spec['updates_override']:,}",
              "tag (export_dapt_checkpoints.sh: dapt<LR>_step<N>)")
    elif tag == "run11":
        r.put("display_name", "run11", "tag")
    fam = {"ours": FAM_SCRATCH, "DAPT": FAM_DAPT, "dapt": FAM_DAPT, "AVES": FAM_AVES, "aves": FAM_AVES,
           "other": FAM_HF}
    if meta:
        r.put("family", fam[meta["family"]], "harness/build_comparison.py:META.family")
    elif board:
        r.put("family", fam[board["fam"]], "harness/build_board.py:MODELS.fam")
    elif spec:
        r.put("family", FAM_DAPT if spec["script"].startswith("train_dapt") else FAM_SCRATCH,
              f"slurm/{spec['script']} (--init-weights)")
    if meta:
        r.put("input", meta["input"], "harness/build_comparison.py:META.input")
        r.put("objective", meta["objective"], "harness/build_comparison.py:META.objective")
        r.put("saw_test_birds", meta["saw"], "harness/build_comparison.py:META.saw_test_birds")
    if board and not board["what"].startswith("as above"):     # 'as above' only reads in the board's order
        r.put("description", board["what"], "harness/build_board.py:MODELS.what")
    if ct.get("hf_id"):
        r.put("description", f"{meta['name'] if meta else tag}: {ct['hf_id']}", f"analysis/{ct_name}:hf_id")

    # ---- experiment, finding, baseline
    if exp:
        r.put("experiment", exp["id"], "experiments.yaml:tag")
        r.put("finding", exp.get("finding"), f"experiments.yaml:{exp['id']}.finding")
    r.put("experiment", spec.get("experiment"), spec.get("experiment_src"))
    if "finding" not in r and board and board["finding"]:
        r.put("finding", board["finding"], "harness/build_board.py:MODELS.finding")
    if "finding" not in r:
        r.put("finding", spec.get("finding"), spec.get("finding_src"))
    if tag.startswith("hf-") or tag == "aves-base-nonbio":
        e25 = EXP_BY_ID["E25"]
        said_exp("E25", "script", "aves_variants_calltype.py (aves-base-nonbio)")
        r.put("experiment", "E25", "experiments.yaml:E25.script")
        r.put("finding", e25.get("finding"), "experiments.yaml:E25.finding")
    if tag == "aves-base-bio":
        r.put("finding", "044", "finding 044 (AVES-bio's compute budget)")
    if exp and not spec.get("baseline"):
        base = exp["baseline"].split()[0].rstrip(",;")
        base = {"E17": "daptv3aves_5e5_step15000", "E18": "daptv3aves_2e4_step15000",
                "E19": "daptv4two_2e4_step15000", "E21": "daptv4aveso_2e4_step15000"}.get(base, base)
        r.put("baseline", base, f"experiments.yaml:{exp['id']}.baseline")
    r.put("baseline", spec.get("baseline"), spec.get("baseline_src") or (
        f"experiments.yaml:{exp['id']}.baseline" if exp else spec.get("differs_src")))
    if spec.get("differs"):
        src = spec.get("differs_src") or (f"experiments.yaml:{exp['id']}.change: '{exp['change'][:90]}'" if exp else None)
        r.put("differs_from_baseline", spec["differs"], src)
    r.put("notes", spec.get("notes"), "see note text")

    # ---- training recipe (scripts)
    script = spec.get("script") or (exp["script"].split()[0].rsplit("/", 1)[1] if exp and exp.get("script") else None)
    if script:
        rc = recipe(script, spec.get("pos") or ((spec["arm"],) if spec.get("arm") else ()), spec.get("arm"))
        r.setdefault("_src", {})["script"] = script + " (named by " + (
            spec.get("script_src") or f"experiments.yaml:{exp['id']}.script") + ")"
        name, short, hours, csrc = corpus_of(rc["audio"])
        r.put("training_corpus", name, ssrc(rc, "--dataset-path/MIX") + " + " + csrc)
        r.put("corpus_hours", round(hours, 2), csrc)
        r["_short_corpus"] = short
        tk = spec.get("targets")
        if tk:
            r.put("targets", TARGETS[tk][0], TARGETS[tk][1] + f" (k={rc['k']} from --num-classes)")
        r.put("init", ("random" if not rc["init"] else Path(rc["init"]).name.replace(".torchaudio.pt", "")),
              ssrc(rc, "--init-weights" if rc["init"] else "no --init-weights"))
        r.put("updates", spec.get("updates_override", rc["updates"]), ssrc(rc, "--max-updates")
              + (" (checkpoint step from the tag)" if "updates_override" in spec else ""))
        r.put("warmup", rc["warmup"], ssrc(rc, "--warmup-updates"))
        r.put("lr", rc["lr"], ssrc(rc, "--learning-rate") + (f" (positional arg from {spec['pos_src']})"
                                                             if spec.get("pos_src") else ""))
        r.put("accumulate", rc["accumulate"], ssrc(rc, "--accumulate-grad-batches")
              if "--accumulate-grad-batches" in rc["flags"] else "train.py default --accumulate-grad-batches")
        r.put("gpus", rc["ranks"], ssrc(rc, "#SBATCH ntasks (srun ranks)"))
        r.put("gpus_requested", rc["gres"], ssrc(rc, "#SBATCH --gres"))
        spb = float(trainpy_default("--seconds-per-batch"))
        r.put("audio_per_update_cap_s", round(spb * rc["ranks"] * rc["accumulate"], 1),
              "train.py default --seconds-per-batch x gpus x accumulate")
        mp, ml = rc["mask"]
        if mp:
            r.put("mask", f"{mp} / {ml}", ssrc(rc, "--mask-prob/--mask-length"))
        else:
            said(TRAIN_PY, "None = the module's 0.4"), said(TRAIN_PY, "None = the module's 3")
            r.put("mask", "0.4 / 3 (module default)",
                  "train.py --mask-prob/--mask-length help: 'None = the module's 0.4' / 'None = the module's 3'")
        r.put("seed", rc["seed"], ("train.py --seed default (help: '" + said(TRAIN_PY, "Every run up to run18 used 1337")
                                   + "')") if rc["seed_default"] else ssrc(rc, "--seed"))
        if rc["freeze"]:
            r.put("notes", (r.get("notes", "") + "; " if r.get("notes") else "") + "conv feature extractor frozen",
                  ssrc(rc, "--freeze-feature-extractor"))
        r["_recipe"] = rc
        if tag in ("run11", "run15_combined"):
            r.put("notes", (r.get("notes", "") + "; " if r.get("notes") else "") + "reserved 4 GPUs, srun launched 1",
                  "finding 044 / 051 ('--gpus 4 under --ntasks=1')")
        if not meta and tag.startswith("dapt"):   # round-1 checkpoints are not in build_comparison.META
            r.put("input", META["run11"]["input"], "same train.py model as run11 (harness/build_comparison.py:META[run11])")
            r.put("objective", META["run11"]["objective"], "harness/build_comparison.py:META[run11].objective")
            r.put("saw_test_birds", META["run11"]["saw"], "trained on run11's corpus -> META[run11].saw_test_birds")
    elif meta:                                   # downloaded / off-the-shelf
        r.put("training_corpus", meta["data"], "harness/build_comparison.py:META.data")
        r.put("corpus_hours", meta["hours"], "harness/build_comparison.py:META.hours")
        r["_short_corpus"] = meta["data"]
        if "AST" in meta["name"]:
            r.put("init", "ImageNet", "harness/build_comparison.py:META.data ('" + meta["data"] + "')")

    # ---- measured compute
    if tag in MEASURED:
        js, key = MEASURED[tag]
        m = jget(js, key)
        ws = m.get("world_size_realised", m.get("world_size"))
        r.put("gpus", ws, jsrc(js, key, "world_size_realised" if "world_size_realised" in m else "world_size"))
        r.put("updates", m["updates"], jsrc(js, key, "updates"))
        r.put("audio_per_update_s", round(m["audio_seconds_per_update"], 2), jsrc(js, key, "audio_seconds_per_update"))
        r.put("total_audio_h", round(m["total_audio_hours"], 1), jsrc(js, key, "total_audio_hours"))
        ek = "epochs_over_corpus" if "epochs_over_corpus" in m else "epochs"
        r.put("epochs", round(m[ek], 2), jsrc(js, key, ek))
        r.put("corpus_hours", round(m["corpus_hours"], 2), jsrc(js, key, "corpus_hours"))
        if tag == "aves-base-bio":
            r.put("lr", m["peak_lr"], jsrc(js, key, "peak_lr"))
            r.put("warmup", m["warmup_updates"], jsrc(js, key, "warmup_updates"))
            r.put("accumulate", m["update_freq"], jsrc(js, key, "update_freq"))
            r.put("targets", *TARGETS["aves_bio_own"])
            r.put("notes", "fairseq epoch counter " + str(jget(js, "consistency_check", "aves", "recorded_epoch_counter"))
                  + " vs derived " + f"{m['epochs_over_corpus']:.2f}", jsrc(js, "consistency_check", "aves"))

    # ---- size
    if tag in det.get("params", {}):
        r.put("params", det["params"][tag], f"analysis/detection_variants.json:params.{tag}")
    elif ct.get("params"):
        r.put("params", ct["params"], f"analysis/{ct_name}:params")
    elif av:
        r.put("params", av["params"], f"analysis/aves_variants_calltype.json:models.{tag}.params")
    if ct.get("n_layers"):
        r.put("layers", ct["n_layers"], f"analysis/{ct_name}:n_layers")
        r.put("embed_dim", ct["dim"], f"analysis/{ct_name}:dim")
    elif av:
        r.put("layers", av["n_layers"], f"analysis/aves_variants_calltype.json:models.{tag}.n_layers")
        r.put("embed_dim", av["embed_dim"], f"analysis/aves_variants_calltype.json:models.{tag}.embed_dim")
    if av and tag == "run11" and "description" not in r:
        r.put("description", av.get("pretraining"), "analysis/aves_variants_calltype.json:models.run11.pretraining")
    return r


def _differs_from_exp(r: dict) -> None:
    """Fill differs_from_baseline from experiments.yaml `change` for E-runs whose spec gave no phrase."""
    exp = EXP_BY_TAG.get(r["tag"])
    if exp and "differs_from_baseline" not in r:
        r["differs_from_baseline"] = exp["change"].split(" (")[0][:80]
        r["_src"]["differs_from_baseline"] = f"experiments.yaml:{exp['id']}.change"


def facts(write_csv: bool = False) -> pd.DataFrame:
    """One row per scored encoder; see the module docstring. Columns in COLUMNS; `source` = col<-file:key,
    entries separated by ' || '."""
    rows = []
    for tag in scored_tags():
        r = _row(tag)
        _differs_from_exp(r)
        r["source"] = " || ".join(f"{c}<-{s}" for c, s in r.get("_src", {}).items())
        rows.append(r)
    df = pd.DataFrame(rows)
    keep = [c for c in COLUMNS if c in df.columns]
    df = df[keep + ["_short_corpus"]].copy()
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    for c in ("updates", "warmup", "accumulate", "gpus", "gpus_requested", "seed", "params", "layers", "embed_dim"):
        df[c] = df[c].astype("Int64")
    order = {FAM_SCRATCH: 0, FAM_DAPT: 1, FAM_AVES: 2, FAM_HF: 3}
    df = df.sort_values(["family", "tag"], key=lambda s: s.map(order) if s.name == "family" else s)
    df = df.set_index("tag", drop=False)
    if write_csv:
        df[COLUMNS].to_csv(OUT_CSV, index=False)
    return df


def cross_checks(df: pd.DataFrame) -> list[str]:
    """Places where two sources disagree about what a model is. Computed, not hand-listed."""
    out = []
    # 1. build_comparison.META's corpus string for DAPT vs the corpus the training script actually read
    for tag, m in META.items():
        if tag in df.index and m["family"] == "DAPT" and "FSD50K" in m["data"] \
                and "FSD50K" not in str(df.at[tag, "training_corpus"]):
            out.append(f"{tag}: build_comparison.META (-> docs/comparison/models.json) says '{m['data']}', "
                       f"but its script read '{df.at[tag, 'training_corpus']}' (no replay)")
    # 2. the call-type JSONs' `note` field: run15's description, copied to every later per-layer JSON
    src = _calltype_json("run15_combined")[1].get("note", "")
    keys = ["family", "init", "corpus_hours", "targets", "lr", "warmup", "updates", "mask"]
    ref = df.loc["run15_combined", keys]
    bad = [t for t in df.index if t != "run15_combined" and _calltype_json(t)[1].get("note") == src
           and any(str(df.at[t, k]) != str(ref[k]) for k in ("family", "init", "corpus_hours", "targets", "lr", "mask"))]
    if bad:
        out.append(f"{len(bad)} call-type JSONs carry run15's `note` ('{src}') although their init, corpus, targets, "
                   f"lr or masking differ from run15's: {', '.join(bad)}")
    # 3. run11's steps per epoch: compute_budget.json vs finding 051
    spe = jget("compute_budget.json", "run11", "steps_per_epoch")
    if said(finding_file("051"), "run11 5208 steps/epoch") and abs(spe - 5208) > 1:
        out.append(f"run11 steps/epoch: compute_budget.json {spe:.2f} (-> {jget('compute_budget.json', 'run11', 'audio_seconds_per_update'):.1f} s/update) "
                   "vs finding 051 '5208 steps/epoch ... 80.2 s of audio per step'")
    # 4. projected vs measured passes in run16's script header
    if said(SLURM / "train_run16_compute4x.sh", "run15 makes ~9.7 passes where run11 made ~18.7"):
        out.append(f"passes: train_run16_compute4x.sh header projects run15 ~9.7 / run11 ~18.7; measured "
                   f"{jget('run15_compute.json', 'run15', 'epochs'):.2f} (run15_compute.json) / "
                   f"{jget('compute_budget.json', 'run11', 'epochs_over_corpus'):.2f} (compute_budget.json)")
    # 5. GPU model for DAPT round 1
    if said(finding_file("045"), "1 L40S") and recipe("train_dapt_aves_init.sh", ("5e-5",))["sbatch"]["gres"].startswith("gpu:L40:"):
        out.append("DAPT round 1 GPU: finding 045 says '1 L40S'; train_dapt_aves_init.sh requests gres gpu:L40:1")
    # 6. replay corpus size
    if said(DATASETS_TSV, "FSD50K eval (27.9 h)") and said(finding_file("058"), "27.79 h"):
        out.append("FSD50K-eval replay: datasets.tsv says 27.9 h, findings 058/060 and experiments.yaml E5 say 27.79 h (rounding)")
    # 7. AVES-bio epochs: derived vs fairseq's counter
    cc = jget("compute_budget.json", "consistency_check", "aves")
    out.append(f"aves-base-bio epochs: derived {cc['derived_epochs']:.2f} vs fairseq counter {cc['recorded_epoch_counter']} "
               f"({cc['relative_disagreement']:.1%}; compute_budget.json passes it)")
    # 8. parameter counts: models.json fills ours/DAPT from run11's count; detection_variants.json has each tag's own
    mj = {r["tag"]: r for r in json.loads(_text(ROOT / "docs/comparison/models.json"))}
    diff = [t for t in df.index if t in mj and mj[t]["params_m"] is not None
            and pd.notna(df.at[t, "params"]) and abs(mj[t]["params_m"] - df.at[t, "params"] / 1e6) > 0.01]
    out.append("params: docs/comparison/models.json vs per-tag counts " + (f"disagree for {diff}" if diff else
               "agree to 0.01 M for every shared tag (models.json copies run11's count for ours/DAPT; "
               "detection_variants.json measures each)"))
    return out


def unknowns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Columns -> tags whose cell no source fills."""
    out = {}
    for c in COLUMNS:
        miss = df.index[df[c].isna()].tolist()
        if miss:
            out[c] = miss
    return out


def short_label(df: pd.DataFrame, tag: str, name: str | None = None, sep: str = "\n") -> str:
    """A legend label that carries what differs: how it was made, the corpus and its hours, compute, size,
    and the seed or targets where those are what separates it. `sep` joins the two halves."""
    r = df.loc[tag]
    name = name or r["display_name"]
    how = {FAM_SCRATCH: "from scratch", FAM_DAPT: f"continued from {r['init']}", FAM_AVES: "downloaded",
           FAM_HF: "off-the-shelf"}[r["family"]]
    corpus = r["_short_corpus"] if isinstance(r["_short_corpus"], str) else r["training_corpus"]
    head = f"{name} — {how}, {corpus}" + (f" {r['corpus_hours']:,.0f} h" if pd.notna(r["corpus_hours"]) else "")
    tail = []
    if r["family"] == FAM_HF:
        tail.append(str(r["objective"]))
    if pd.notna(r["gpus"]):
        tail.append(f"{r['gpus']} GPU")
    if pd.notna(r["total_audio_h"]):
        tail.append(f"{r['total_audio_h']:,.0f} h seen")
    if pd.notna(r["params"]):
        tail.append(f"{r['params'] / 1e6:.1f}M params")
    if r["family"] == FAM_DAPT and isinstance(r["targets"], str) and "two-teacher" in r["targets"]:
        tail.append("two-teacher targets")
    if pd.notna(r["seed"]) and r["seed"] != 1337:
        tail.append(f"seed {r['seed']}")
    return head + (sep + ", ".join(tail) if tail else "")


if __name__ == "__main__":
    df = facts(write_csv=True)
    print(f"wrote {OUT_CSV} ({len(df)} rows x {len(COLUMNS)} columns)")
    filled = df[COLUMNS].notna().sum()
    print("filled per column:", ", ".join(f"{c} {n}" for c, n in filled.items()))
    for c, tags in unknowns(df).items():
        print(f"  empty {c} ({len(tags)}): {', '.join(tags)}")
    print("\nsources that disagree:")
    for line in cross_checks(df):
        print("  -", line)
