#!/usr/bin/env python3
"""Build docs/model_board.html: every encoder with its scores, the experiment queue, the state of
the Savio automations, and the notebooks. Every number is read from the scoring JSONs at build time;
nothing is typed in. Rebuild after each finding:

    /Library/Frameworks/Python.framework/Versions/3.10/bin/python3 harness/build_board.py
"""
import datetime as dt
import html
import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
A = Path.home() / "zf_labelset/zf_detection_dataset_v1/analysis"
OUT = ROOT / "docs/model_board.html"
REGISTRY = ROOT / "pytorchAudio/examples/hubert/savio/datasets.tsv"
GITHUB = "https://github.com/ttt5600/NeruoBatLab/blob/UMAP/"
REF = "run16_compute4x"

# key, display name, family, what it is, finding, registry name (Savio folder)
MODELS = [
    ("run19_avesteacher", "run19", "ours", "run16's recipe, labels from AVES layer 6 (k=200)", "059", "run19_avesteacher"),
    ("run16_compute4x", "run16", "ours", "run15 on 4 GPUs: 4x the audio per update (reference)", "052", "run16"),
    ("run20_fsdonly", "run20", "ours", "run16's recipe on general audio only (FSD50K, 108 h)", "061", "run20_fsdonly"),
    ("run18_iter2", "run18", "ours", "run16's recipe, labels from run16 layer 6 (iteration 2)", "056", "run18"),
    ("run17_accum2", "run17", "ours", "run16 + 2x gradient accumulation", "054", "run17"),
    ("run16_seed2", "run16, seed 2", "ours", "run16 exactly, different random seed", "057", "run16_seed2"),
    ("run21_zfonly", "run21", "ours", "run16's recipe on zebra finch audio only (116 h)", "061", "run21_zfonly"),
    ("run15_combined", "run15", "ours", "run11's recipe on zebra finch + FSD50K (224 h), k=200", "051", "run15"),
    ("run11", "run11", "ours", "zebra finch only (116 h), spectrogram labels k=100, 1 GPU; the release model", "", "run11"),
    ("daptreplay3_5e5_step15000", "DAPT replay x3", "dapt", "AVES weights, continued on zebra finch with 41.8% general-audio replay", "060", "daptv2_replay_x3"),
    ("daptreplay_5e5_step15000", "DAPT replay x1", "dapt", "AVES weights, continued on zebra finch with 19.3% general-audio replay", "058", None),
    ("birdaves-biox-large", "BirdAVES biox-large", "aves", "downloaded; large, bird-heavy corpus", "", None),
    ("birdaves-biox-base", "BirdAVES biox-base", "aves", "downloaded; same size as ours, bird-heavy corpus", "", None),
    ("aves-base-core", "AVES core", "aves", "downloaded", "", None),
    ("aves-base-all", "AVES all", "aves", "downloaded", "", None),
    ("aves-base-bio", "AVES bio", "aves", "downloaded; the AVES in earlier headlines", "", None),
    ("birdaves-bioxn-large", "BirdAVES bioxn-large", "aves", "downloaded; large", "", None),
]


def load(name):
    p = A / name
    return json.loads(p.read_text()) if p.exists() else {}


def calltype():
    acc = {}
    for m in load("aves_variants_calltype.json").get("models", {}).items():
        acc[m[0]] = m[1]["best_11"]["acc"]
    for key, *_ in MODELS:
        d = load(f"{key}_calltype.json") or (load("run15_calltype.json") if key == "run15_combined" else {})
        if "acc11" in d:
            acc[key] = d["acc11"]
    return acc


def vs_ref():
    """X - run16 on 11-class call type, observed difference + bird-bootstrap interval, either direction."""
    out = {}
    for f in A.glob("*_bootstrap.json"):
        vs = json.loads(f.read_text()).get("vs", {})
        for a, row in vs.items():
            if "@" in a:
                continue
            for b, v in row.items():
                if a == REF and b not in out:
                    out[b] = (-v["delta"], -v["hi"], -v["lo"], v["resolved"])
                elif b == REF:
                    out[a] = (v["delta"], v["lo"], v["hi"], v["resolved"])
    return out


def registry():
    rows = {}
    for line in REGISTRY.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, kind, path, what = (line.split("\t") + ["", "", "", ""])[:4]
            rows[name] = path
    return rows


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def fmt(x, n=4):
    return "—" if x is None else f"{x:.{n}f}"


def forest(rows):
    """SVG forest plot: X - run16 with 95% intervals. rows = [(label, family, d, lo, hi, resolved)]."""
    lo_x = min(r[3] for r in rows); hi_x = max(r[4] for r in rows)
    lo_x = min(-0.01, (int(lo_x * 100) - 1) / 100); hi_x = max(0.01, (int(hi_x * 100) + 1) / 100)
    W, L, R, RH, T = 760, 190, 24, 24, 34
    H = T + RH * len(rows) + 30
    X = lambda v: L + (v - lo_x) / (hi_x - lo_x) * (W - L - R)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Call type difference from run16 with intervals">']
    t = round(lo_x, 2)
    while t <= hi_x + 1e-9:
        x = X(t)
        s.append(f'<line x1="{x:.1f}" y1="{T-8}" x2="{x:.1f}" y2="{H-26}" class="grid"/>')
        s.append(f'<text x="{x:.1f}" y="{H-10}" class="tick" text-anchor="middle">{t:+.2f}</text>')
        t = round(t + 0.01, 2)
    x0 = X(0)
    s.append(f'<line x1="{x0:.1f}" y1="{T-14}" x2="{x0:.1f}" y2="{H-26}" class="zero"/>')
    s.append(f'<text x="{x0:.1f}" y="{T-18}" class="tick" text-anchor="middle">run16</text>')
    for i, (label, fam, d, lo, hi, res) in enumerate(rows):
        y = T + RH * i + RH / 2
        s.append(f'<text x="{L-12}" y="{y+4:.1f}" class="lab" text-anchor="end">{html.escape(label)}</text>')
        s.append(f'<line x1="{X(lo):.1f}" y1="{y:.1f}" x2="{X(hi):.1f}" y2="{y:.1f}" class="ci {fam}"/>')
        fill = "dot-fill" if res else "dot-open"
        s.append(f'<circle cx="{X(d):.1f}" cy="{y:.1f}" r="5" class="{fam} {fill}"/>')
    s.append("</svg>")
    return "\n".join(s)


def chip(state):
    cls = {"done": "ok", "working": "ok", "submitted": "run", "running": "run", "trained": "run",
           "approved": "plan", "proposed": "idle", "needs-human": "warn", "stopped": "idle",
           "expired": "bad", "open issue": "bad", "partly tested": "warn", "failed": "bad"}.get(state, "idle")
    return f'<span class="chip {cls}">{html.escape(state)}</span>'


def main():
    now = dt.datetime.now().astimezone()
    acc, vs = calltype(), vs_ref()
    det = load("detection_variants.json").get("precommit", {})
    chick = load("chick_holdout_variants.json").get("precommit", {})
    reg = registry()

    # ---- models table + forest rows
    trs, frows = [], []
    for key, name, fam, what, finding, regname in MODELS:
        j = det.get(key, {}).get("joint", {})
        c = chick.get(key, {})
        v = vs.get(key)
        vtxt = "reference" if key == REF else (
            f'{v[0]:+.4f}' + ("" if v[3] else ' <span class="nd">n.d.</span>') + f'<br><span class="iv">[{v[1]:+.4f}, {v[2]:+.4f}]</span>'
            if v else "—")
        where = (f'<code>{html.escape(reg[regname].replace("/global/scratch/users/jonathanswang/", "…/"))}</code>'
                 if regname and regname in reg else ("<code>…/external/aves</code>" if key == "aves-base-bio"
                 else ("<code>…/external/dapt</code>" if fam == "dapt" else "Mac only")))
        ffile = next(iter(sorted((ROOT / "knowledge/findings").glob(f"{finding}-*.yaml"))), None) if finding else None
        flink = (f'<a href="{GITHUB}knowledge/findings/{ffile.name}">{finding}</a>' if ffile else "")
        trs.append(f'<tr class="{fam}"><td class="nm"><span class="sw {fam}"></span>{html.escape(name)}</td>'
                   f'<td class="what">{html.escape(what)}</td><td class="n">{fmt(acc.get(key))}</td>'
                   f'<td class="n vs">{vtxt}</td><td class="n">{fmt(j.get("zf", {}).get("auc"))}</td>'
                   f'<td class="n">{fmt(j.get("bp", {}).get("ap"))}</td><td class="n">{fmt(c.get("auc"))}</td>'
                   f'<td class="n">{flink}</td><td class="path">{where}</td></tr>')
        if v and key != REF:
            frows.append((name, fam, *v))
    frows.sort(key=lambda r: r[2], reverse=True)

    ours = {k: acc[k] for k, _, fam, *_ in MODELS if fam == "ours" and k in acc}
    best = max(ours, key=ours.get)
    best_name = next(n for k, n, *_ in MODELS if k == best)
    aves = [acc[k] for k, _, fam, *_ in MODELS if fam == "aves" and k in acc]
    seed = vs.get("run16_seed2")

    # ---- experiments
    q = yaml.safe_load((ROOT / "harness/experiments.yaml").read_text())["experiments"]
    running = [e for e in q if e.get("status") == "submitted"]
    erows = []
    for e in q:
        res = e.get("result") or e.get("question") or ""
        erows.append(f'<tr><td class="nm">{e["id"]}</td><td>{html.escape(e.get("title", ""))}</td>'
                     f'<td>{chip(e.get("status", ""))}</td><td class="n">{e.get("job") or ""}</td>'
                     f'<td class="n">{e.get("finding") or ""}</td><td class="what">{html.escape(res)}</td></tr>')

    # ---- live automation state
    cert = sh("ssh-keygen -L -f ~/.ssh/ssh_certs/brc_cert-cert.pub | sed -n 's/.*Valid: from .* to //p'")
    try:
        cert_ok = dt.datetime.fromisoformat(cert).astimezone() > now
    except ValueError:
        cert_ok = False
    wstate = json.loads((Path.home() / ".claude/savio-watch/state.json").read_text())
    jobscfg = json.loads((Path.home() / ".claude/savio-watch/jobs.json").read_text())
    hb = ROOT / "harness/.pi_heartbeat"
    hb_age = (now.timestamp() - hb.stat().st_mtime) / 3600 if hb.exists() else None
    lead_alive = hb_age is not None and hb_age < 2
    hb_txt = dt.datetime.fromtimestamp(hb.stat().st_mtime).strftime("%b %-d %H:%M") if hb.exists() else "never"
    nreg = len(reg)
    streak = int(wstate.get("ssh_fail_streak", 0))
    cards = [
        ("Savio login", "working" if cert_ok else "expired",
         f"12-hour SSH certificate, {'valid until' if cert_ok else 'expired'} {html.escape(cert[:16].replace('T', ' '))}. "
         "Renew with <code>sv login</code> (one PIN + code). Every automation below that touches Savio waits on this."),
        ("lab / sv commands", "working",
         "<code>sv train</code>, <code>sv status</code>, <code>sv log</code>, <code>sv gpu</code>, <code>sv notebook</code>, "
         "<code>sv data</code>, <code>sv share</code>. Training submissions run an import check first and go in a ledger."),
        ("Shared data", "working",
         f"{nreg} corpora, label sets and models registered in <code>datasets.tsv</code>, browsable under "
         "<code>…/lab_data</code>, read + write for Julie (jelie) and Bhavna (malladibhavna). New folders: <code>lab share</code>."),
        ("Storage", "working",
         "Home went from 47 GB (at its 50 GB cap) to 0.7 GB on Oct 4; conda envs, pip cache and JupyterLab live on scratch "
         "behind shortcuts. Env recipes in <code>~/env_specs</code>."),
        ("Job watcher", "working" if streak == 0 else "expired",
         f"Polls Savio every 5 min (launchd). Last poll {html.escape(str(wstate.get('last_poll', '?'))[:16].replace('T', ' '))}"
         + (f", {streak} failed in a row (login expired)." if streak else ", connected.")),
        ("Experiment harness", "working" if jobscfg.get("harness", {}).get("enabled") else "stopped",
         "Each poll, <code>tick.py</code> starts an agent when a run finishes, fails, or an approved experiment has a free slot "
         "(2 at a time, 6 agents a day). It stands down while the lead agent is alive."),
        ("Lead agent", "running" if lead_alive else "stopped",
         f"Long-lived <code>/loop</code> session (<code>harness/PI.md</code>). Last heartbeat {hb_txt}. "
         + ("" if lead_alive else "Restart: <code>caffeinate -is claude --permission-mode bypassPermissions</code>, then "
            "<code>/loop follow harness/PI.md</code>.")),
        ("Critic", "working",
         "A fresh agent tries to break every finding before it is recorded (<code>harness/CRITIC.md</code>). "
         "Findings 056–061 all came back accept-with-changes, changes applied."),
        ("Jupyter on Savio", "partly tested",
         "Kernel and the Mac-to-Savio tunnel are verified; the GPU notebook job itself has not run yet. "
         "Start one with <code>sv notebook HOURS</code>."),
        ("Duplicate agents", "open issue",
         "The usage watcher's auto-resume started a second agent in this project twice while the lead was running; "
         "both submitted the same job. Duplicates were cancelled. Fix: make auto-resume skip this project while the lead is alive."),
    ]
    cardhtml = "\n".join(
        f'<div class="card"><div class="ch"><h3>{t}</h3>{chip(s)}</div><p>{b}</p></div>' for t, s, b in cards)

    needs = []
    if not cert_ok:
        needs.append("Run <code>sv login</code>: the Savio certificate has expired.")
    nh = [e["id"] for e in q if e.get("status") == "needs-human"]
    if nh:
        needs.append(f"Decide on {', '.join(nh)} (second seeds) in <code>harness/experiments.yaml</code>.")
    needs.append("Approve the fix for duplicate agents.")

    # ---- notebooks
    nbrows = []
    for f in sorted((ROOT / "notebooks").glob("*.ipynb")):
        nb = json.loads(f.read_text())
        title, sub = f.stem, ""
        for c in nb["cells"]:
            if c["cell_type"] == "markdown":
                lines = [l.strip() for l in "".join(c["source"]).splitlines() if l.strip()]
                title = lines[0].lstrip("# ").strip() if lines else title
                sub = next((l for l in lines[1:] if not l.startswith("#")), "")
                break
        sub = sub.replace("**", "").replace("`", "")
        if len(sub) > 150:
            sub = sub[:147].rsplit(" ", 1)[0] + "…"
        date = dt.datetime.fromtimestamp(f.stat().st_mtime).strftime("%b %-d")
        nbrows.append(f'<tr><td class="nm"><a href="{GITHUB}notebooks/{f.name}">{html.escape(f.name)}</a></td>'
                      f'<td><strong>{html.escape(title)}</strong><br><span class="sub">{html.escape(sub)}</span></td>'
                      f'<td class="n">{date}</td></tr>')

    page = TEMPLATE
    subs = {
        "ASOF": now.strftime("%b %-d, %Y %H:%M"),
        "COHORT": "{n_clips:,} clips, {n_birds} birds".format(**load(f"{REF}_calltype.json")["cohort11"]),
        "BEST": f"{html.escape(best_name)} {acc[best]:.4f}",
        "BEST_NOTE": (f"AVES models score {min(aves):.3f}–{max(aves):.3f}; "
                      + (f"vs run16 {vs[best][0]:+.4f} [{vs[best][1]:+.4f}, {vs[best][2]:+.4f}], "
                         f"{'resolved' if vs[best][3] else 'not distinguishable'}" if best in vs and best != REF else "")),
        "SEED": f"{abs(seed[0]):.4f}" if seed else "—",
        "SEED_NOTE": (f"run16's second seed: {seed[0]:+.4f} [{seed[1]:+.4f}, {seed[2]:+.4f}]. Single runs cannot "
                      "separate smaller gaps." if seed else ""),
        "RUNNING": (", ".join(f'{e["id"]} ({e.get("job")})' for e in running) if running else "Nothing"),
        "RUNNING_NOTE": ("" if running else "The queue is empty; the next experiments wait for your decision."),
        "NEEDS": "".join(f"<li>{n}</li>" for n in needs),
        "FOREST": forest(frows),
        "MODEL_ROWS": "\n".join(trs),
        "EXP_ROWS": "\n".join(erows),
        "CARDS": cardhtml,
        "NB_ROWS": "\n".join(nbrows),
        "NB_LATEST": max(dt.datetime.fromtimestamp(f.stat().st_mtime) for f in (ROOT / "notebooks").glob("*.ipynb")).strftime("%b %-d"),
    }
    for k, v in subs.items():
        page = page.replace("{{" + k + "}}", v)
    OUT.write_text(page)
    print(f"wrote {OUT} ({len(MODELS)} models, {len(q)} experiments, {len(nbrows)} notebooks)")


TEMPLATE = (Path(__file__).resolve().parent / "board_template.html").read_text() if (
    Path(__file__).resolve().parent / "board_template.html").exists() else ""

if __name__ == "__main__":
    main()
