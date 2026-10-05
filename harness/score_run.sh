#!/bin/bash
# Score a finished training run end to end, the way run16 and run17 were scored by hand:
#   1. Savio: export the final checkpoint to a stripped encoder (export_weights.py refuses to write
#      unless the rebuilt encoder reproduces the checkpoint's features bit-identically)
#   2. Mac:   copy the encoder, the training curves and the job log
#   3. Mac:   11-class call type + bird bootstrap, detection + BirdPark, chick holdout + bootstrap
# Every number the research agent may quote afterwards comes from the JSON files step 3 writes.
#
#   bash harness/score_run.sh TAG EXP_DIR [NUM_CLASSES] [JOBID]
#   DRY=1 bash harness/score_run.sh run17_accum2 /global/scratch/users/jonathanswang/temp_train_run17_accum2
#     (DRY=1 checks every input exists and prints the commands, changing nothing)
set -euo pipefail
TAG=${1:?usage: score_run.sh TAG EXP_DIR [NUM_CLASSES] [JOBID]}
EXP=${2:?usage: score_run.sh TAG EXP_DIR [NUM_CLASSES] [JOBID]}
NC=${3:-200}; JOB=${4:-}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=/Library/Frameworks/Python.framework/Versions/3.10/bin/python3     # the analysis interpreter
S=/global/scratch/users/jonathanswang
A=$HOME/zf_labelset/zf_detection_dataset_v1/analysis
DAPT=$HOME/zf_labelset/external/dapt
CONDA=/global/software/rocky-8.x86_64/manual/modules/langs/anaconda3/2024.02-1/etc/profile.d/conda.sh
CKDIR=$EXP/checkpoints_ZF_test_pipeline_hubert_pretrain_base
mkdir -p "$ROOT/harness/logs"
LOG=$ROOT/harness/logs/score_${TAG}_$(date +%Y%m%d-%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

# one SSH command at a time, never prompting; drop OpenSSH's post-quantum banner
ssh_s() { ssh -o BatchMode=yes savio-login "$@" 2> >(grep -v -iE "post-quantum|store now|upgraded|pq.html" >&2); }
run() { echo "+ $*"; [ -n "${DRY:-}" ] || "$@"; }

echo "== score $TAG  (exp $EXP, $NC classes, job ${JOB:-?})  $(date)"
CK=$(ssh_s "ls $CKDIR 2>/dev/null | grep -E '^epoch=[0-9]+-step=[0-9]+\.ckpt$' | sort -t= -k3 -n | tail -1")
[ -n "$CK" ] || { echo "FATAL: no epoch=*-step=*.ckpt in $CKDIR"; exit 2; }
echo "   final checkpoint: $CK"
ssh_s "test -f ~/export_weights.py && test -f $S/adultvoc_16k/BlaBla0506_110302-DC-01.wav" \
    || { echo "FATAL: ~/export_weights.py or the verification clip is missing on Savio"; exit 2; }
for f in run15_calltype run16_calltype_bootstrap detection_variants chick_holdout_variants; do
    [ -f "$ROOT/zfeval/experiments/$f.py" ] || { echo "FATAL: zfeval/experiments/$f.py missing"; exit 2; }
done
"$PY" -c "import torch, sklearn, yaml" || { echo "FATAL: analysis python incomplete"; exit 2; }

echo "== 1. export on Savio"
EXPORT="source $CONDA && conda activate hubert_env && python ~/export_weights.py --ckpt '$CKDIR/$CK' \
--out $S/external/dapt/$TAG.pt --num-classes $NC --run-tag $TAG --wav $S/adultvoc_16k/BlaBla0506_110302-DC-01.wav"
echo "+ ssh savio-login $EXPORT"
if [ -z "${DRY:-}" ]; then
    ssh_s "$EXPORT 2>&1 | grep -v WARNING | tail -6; test -s $S/external/dapt/$TAG.pt" \
        || { echo "FATAL: export failed or wrote nothing"; exit 3; }
fi

echo "== 2. copy to the Mac"
M=$ROOT/savio_artifacts
run mkdir -p "$DAPT" "$M/metrics/$TAG" "$M/logs" "$M/analysis"
run rsync -a "savio-login:$S/external/dapt/$TAG.pt" "$DAPT/"
for v in $(ssh_s "ls $EXP/lightning_logs 2>/dev/null | grep '^version_'"); do
    run rsync -a "savio-login:$EXP/lightning_logs/$v/metrics.csv" "$M/metrics/$TAG/${TAG}_${v}_metrics.csv" || true
done
[ -z "$JOB" ] || run rsync -a "savio-login:jobs/*_${JOB}.log" "$M/logs/" || true

echo "== 3. score"
cd "$ROOT"
run "$PY" zfeval/experiments/run15_calltype.py --tag "$TAG"
run "$PY" zfeval/experiments/run16_calltype_bootstrap.py --tag "$TAG"
run "$PY" zfeval/experiments/detection_variants.py --models "$TAG"
run "$PY" zfeval/experiments/chick_holdout_variants.py --models "$TAG"
run "$PY" zfeval/experiments/chick_holdout_variants.py --bootstrap --models "$TAG" --baseline run11

if [ -n "${DRY:-}" ]; then echo "DRY RUN OK: every input exists"; exit 0; fi
for f in "${TAG}_calltype.json" "${TAG}_bootstrap.json"; do
    [ -s "$A/$f" ] || { echo "FATAL: $A/$f was not written"; exit 4; }
done
for f in detection_variants chick_holdout_variants; do
    grep -q "\"$TAG" "$A/$f.json" || { echo "FATAL: $TAG missing from $f.json"; exit 4; }
done
cp "$A/${TAG}_calltype.json" "$A/${TAG}_bootstrap.json" "$A/detection_variants.json" \
   "$A/chick_holdout_variants.json" "$M/analysis/"
echo "SCORED $TAG -> $A/${TAG}_calltype.json, ${TAG}_bootstrap.json, detection_variants.json, chick_holdout_variants.json"
