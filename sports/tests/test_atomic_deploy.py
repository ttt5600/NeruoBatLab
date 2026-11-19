"""ops_robustness_1118: scripts/atomic_deploy.sh (acp / awrite) and why installers need it.

test_inplace_cp_corrupts_running_script reproduces the qs_job.sh 'syntax error near unexpected token ;;' mechanism;
test_acp_keeps_running_script_intact shows the tmp + mv deploy does not."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parents[1] / "scripts" / "atomic_deploy.sh"


def sh(cmd: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["/bin/bash", "-c", f'set -euo pipefail; . "{HELPER}"; {cmd}'], cwd=cwd,
                          capture_output=True, text=True)


def test_acp_to_dir_and_file_and_modes(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "dst").mkdir()
    a, b = tmp_path / "src" / "a.py", tmp_path / "src" / "b.sh"
    a.write_text("A")
    b.write_text("B")
    os.chmod(b, 0o755)
    r = sh('acp src/a.py src/b.sh dst/ && acp src/a.py dst/renamed.py && [ -f dst/a.py ]', tmp_path)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "dst" / "a.py").read_text() == "A" and (tmp_path / "dst" / "renamed.py").read_text() == "A"
    assert os.stat(tmp_path / "dst" / "b.sh").st_mode & 0o777 == 0o755
    assert not list((tmp_path / "dst").glob(".*.tmp.*"))
    r = sh("acp src/a.py src/b.sh dst/notadir.py", tmp_path)
    assert r.returncode != 0
    r = sh("acp src/missing.py dst/", tmp_path)
    assert r.returncode != 0 and not list((tmp_path / "dst").glob(".*.tmp.*"))


def test_awrite_heredoc_mode(tmp_path):
    r = sh('awrite w.sh <<EOF\necho hi\nEOF\nawrite p.plist <<EOF\nx\nEOF', tmp_path)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "w.sh").read_text() == "echo hi\n"
    assert os.stat(tmp_path / "w.sh").st_mode & 0o777 == 0o755
    assert os.stat(tmp_path / "p.plist").st_mode & 0o777 == 0o644
    os.chmod(tmp_path / "w.sh", 0o700)
    assert sh('awrite w.sh <<EOF\necho 2\nEOF', tmp_path).returncode == 0
    assert os.stat(tmp_path / "w.sh").st_mode & 0o777 == 0o700  # existing mode kept


OLD = 'sleep 1\ncase x in\n  x) echo old-ok ;;\nesac\n'
NEW = '# a longer header shifts every byte offset ...................................\nsleep 1\ncase x in\n  x) echo new-ok ;;\nesac\n'


def _run_while_replacing(tmp_path: Path, replace: str) -> subprocess.CompletedProcess:
    job = tmp_path / "job.sh"
    job.write_text(OLD)
    (tmp_path / "new.sh").write_text(NEW)
    p = subprocess.Popen(["/bin/bash", str(job)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(0.3)  # job is inside `sleep 1`, has read only the first line
    assert sh(replace, tmp_path).returncode == 0
    out, err = p.communicate(timeout=10)
    return subprocess.CompletedProcess(p.args, p.returncode, out, err)


def test_inplace_cp_corrupts_running_script(tmp_path):
    r = _run_while_replacing(tmp_path, "cp new.sh job.sh")
    assert "old-ok" not in r.stdout  # bash read the new bytes from the old offset: garbage / wrong branch


def test_brace_group_survives_inplace_cp(tmp_path):
    """qs_job.sh's body is one { ... } group: bash parses it whole before running, so even a plain cp mid-run is safe."""
    job = tmp_path / "job.sh"
    job.write_text("{\n" + OLD + "exit 0\n}\n")
    (tmp_path / "new.sh").write_text("{\n" + NEW + "exit 0\n}\n")
    p = subprocess.Popen(["/bin/bash", str(job)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(0.3)
    assert sh("cp new.sh job.sh", tmp_path).returncode == 0
    out, err = p.communicate(timeout=10)
    assert out.strip() == "old-ok" and p.returncode == 0, (out, err)


def test_qs_job_body_is_one_brace_group():
    lines = (HELPER.parent / "qs_job.sh").read_text().rstrip("\n").splitlines()
    code = [l for l in lines if l.strip() and not l.startswith("#")]
    assert code[0] == "{" and code[-1] == "}" and code[-2] == "exit 0"


def test_acp_keeps_running_script_intact(tmp_path):
    r = _run_while_replacing(tmp_path, "acp new.sh job.sh")
    assert r.stdout.strip() == "old-ok" and r.returncode == 0, (r.stdout, r.stderr)
    assert (tmp_path / "job.sh").read_text() == NEW
