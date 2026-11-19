"""Atomic deploy helper (ops_robustness_1118): copy src -> tmp in the destination dir, chmod, os.replace.
    python deploy.py SRC DST [MODE_OCTAL] [SRC DST MODE ...]   (triples)
Backs up an existing DST to <dst dir>/backups/ops_robustness_1118/ only if no backup exists yet (keeps the original)."""
import os
import shutil
import sys
import tempfile
from pathlib import Path

a = sys.argv[1:]
assert len(a) % 3 == 0, "triples: SRC DST MODE"
for i in range(0, len(a), 3):
    src, dst, mode = Path(a[i]), Path(a[i + 1]).expanduser(), int(a[i + 2], 8)
    if dst.exists():
        bdir = dst.parent / "backups" / "ops_robustness_1118"
        bdir.mkdir(parents=True, exist_ok=True)
        if not (bdir / dst.name).exists():
            shutil.copy2(dst, bdir / dst.name)
    fd, tmp = tempfile.mkstemp(dir=dst.parent, prefix=f".{dst.name}.", suffix=".tmp")
    os.close(fd)
    shutil.copyfile(src, tmp)
    os.chmod(tmp, mode)
    os.replace(tmp, dst)
    print(f"deployed {src} -> {dst} ({dst.stat().st_size} B, mode {oct(mode)})")
