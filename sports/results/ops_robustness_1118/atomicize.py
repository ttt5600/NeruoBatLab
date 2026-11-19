"""Rewrite scripts/install_*.sh + qs_sync.sh deploy steps to atomic tmp + mv (ops_robustness_1118).
cp at a command position -> acp; `cat > "$DST/x" <<TAG` -> `awrite "$DST/x" <<TAG`; source atomic_deploy.sh after SRC=.
Plists ($LA/..., $PLIST) are left alone (launchd reads them only at bootstrap). Originals -> results/ops_robustness_1118/orig/.
    python atomicize.py [--dry]"""
import re
import shutil
import sys
from pathlib import Path

S = Path(__file__).resolve().parents[2] / "scripts"
ORIG = Path(__file__).resolve().parent / "orig"
SKIP = {"install_health_monitor_agent.sh", "install_kalshi_paper_agent.sh"}  # already tmp + mv
SRC_LINE = re.compile(r'^SRC="\$\(cd "\$\(dirname "\$0"\)/\.\." && pwd\)"\s*$', re.M)
SRCLINE_QS = re.compile(r'^SRC="\$\(cd "\$\(dirname "\$0"\)/\.\." && pwd\)"\s*$', re.M)
CP = re.compile(r'(^[ \t]*|\bdo |\|\| |&& |; )cp (?=")', re.M)
CAT = re.compile(r'cat > ("\$DST/[^"]+") *<<')
HOOK = '. "$SRC/scripts/atomic_deploy.sh"   # acp / awrite = tmp + mv deploys; never cp onto a live script (ops_robustness_1118)'
dry = "--dry" in sys.argv
ORIG.mkdir(exist_ok=True)
for p in sorted(S.glob("install_*.sh")) + [S / "qs_sync.sh"]:
    if p.name in SKIP:
        continue
    t = p.read_text()
    n_cp, n_cat = len(CP.findall(t)), len(CAT.findall(t))
    if "atomic_deploy.sh" in t or (n_cp == 0 and n_cat == 0):
        print(f"{p.name}: unchanged (cp {n_cp}, cat {n_cat}, hooked {'atomic_deploy.sh' in t})")
        continue
    m = SRC_LINE.search(t)
    assert m, p.name
    new = t[:m.end()] + "\n" + HOOK + t[m.end():]
    new = CP.sub(lambda mm: f"{mm.group(1)}acp ", new)
    new = CAT.sub(lambda mm: f"awrite {mm.group(1)} <<", new)
    print(f"{p.name}: cp->acp {n_cp}, cat->awrite {n_cat}")
    if not dry:
        if not (ORIG / p.name).exists():
            shutil.copy2(p, ORIG / p.name)
        p.write_text(new)
