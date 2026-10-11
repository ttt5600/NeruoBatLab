#!/bin/bash
# E10: download and prepare the BEANS benchmark with BEANS's own scripts, so the train/valid/test
# splits are exactly theirs, then check every prepared file against BEANS's published MD5s.
#
# Why BEANS's scripts and their pinned versions: the splits are random (train_test_split, seed 42);
# a different scikit-learn can split differently (we saw 1.7.2 vs 1.8.0 do it, finding 062). The MD5
# check on the split CSVs is the proof that ours equal theirs.
#
# Skips cbi and rfcx (Kaggle competitions: need the user's Kaggle account and accepted rules).
#     bash zfeval/experiments/beans_fetch.sh            # all 10 non-Kaggle tasks, bats first
#     bash zfeval/experiments/beans_fetch.sh bats dogs  # just these
set -uo pipefail
ROOT=${BEANS_ROOT:-$HOME/zf_labelset/external/beans}
COMMIT=b278b1d96b3fb7ad4d14eb7453fc61ab071bc897      # BEANS main, 2024-10-16
TASKS=${*:-bats watkins dogs humbugdb esc50 speech_commands dcase enabirds hiceas hainan_gibbons}

if [ ! -d "$ROOT/.git" ]; then
    git clone -q https://github.com/earthspecies/beans.git "$ROOT" && git -C "$ROOT" checkout -q "$COMMIT" || exit 1
fi
[ "$(git -C "$ROOT" rev-parse HEAD)" = "$COMMIT" ] || { echo "FATAL: $ROOT is not at $COMMIT" >&2; exit 1; }
cd "$ROOT" || exit 1

# BEANS pins python 3.8 / torch 1.12.1; 3.10 is the oldest python here and 1.12.1 has 3.10 wheels.
if [ ! -x .venv/bin/python ]; then
    /Library/Frameworks/Python.framework/Versions/3.10/bin/python3 -m venv .venv || exit 1
    .venv/bin/pip install -q numpy==1.23.5 pandas==1.4.3 scikit-learn==1.1.1 plumbum==1.7.2 \
        pyyaml==6.0 tqdm==4.64.0 memoization==0.4.0 torch==1.12.1 torchaudio==0.12.1 || exit 1
fi
command -v sox >/dev/null || { echo "FATAL: sox missing (dogs needs it): brew install sox" >&2; exit 1; }
export PYTHONPATH=$ROOT
PY=.venv/bin/python
mkdir -p data logs

for t in $TASKS; do
    [ -f "logs/$t.done" ] && { echo "== $t already done"; continue; }
    echo "== $t  $(date '+%H:%M:%S')"
    if [ "$t" = hiceas ]; then
        mkdir -p data/hiceas
        wget -q https://storage.googleapis.com/ml-bioacoustics-datasets/hiceas_1-20_minke-detection.zip -O data/hiceas/hiceas.zip \
            && unzip -q -o data/hiceas/hiceas.zip -d data/hiceas > "logs/$t.log" 2>&1
    else
        $PY "scripts/$t.py" > "logs/$t.log" 2>&1
    fi
    rc=$?
    [ $rc -eq 0 ] && touch "logs/$t.done" || echo "   FAILED rc=$rc, see $ROOT/logs/$t.log"
    du -sh data 2>/dev/null | sed 's/^/   data now /'
done

# MD5 check: every listed file we have. Split files (csv/jsonl) must all match -- they ARE the splits.
# Audio that BEANS resampled may differ by library version; that is reported, not fatal.
$PY - <<'PY'
import collections, hashlib, os, re
def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()
c = collections.defaultdict(collections.Counter); bad_splits = []
for line in open('data/file_hashes'):
    path, want = line.rstrip('\n').split('\t')
    ds = path.split('/')[1]
    kind = 'split' if re.search(r'\.(csv|jsonl)$', path) else 'audio'
    if not os.path.exists(path):
        c[ds][kind + ' missing'] += 1; continue
    ok = md5(path) == want
    c[ds][kind + (' ok' if ok else ' MISMATCH')] += 1
    if kind == 'split' and not ok:
        bad_splits.append(path)
for ds in sorted(c):
    if any('ok' in k or 'MISMATCH' in k for k in c[ds]):
        print(f'[md5] {ds:22s}', dict(sorted(c[ds].items())))
print('[md5] split files that differ from BEANS:', bad_splits or 'none')
PY
