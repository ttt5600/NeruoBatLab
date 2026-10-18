#!/bin/bash
# Score a finished training run end to end, the way run16 and run17 were scored by hand:
#   1. Savio: export the final checkpoint to a stripped encoder (export_weights.py refuses to write
#      unless the rebuilt encoder reproduces the checkpoint's features bit-identically)
#   2. Mac:   copy the encoder, the training curves and the job log
#   3. Mac:   11-class call type + bird bootstrap, detection + BirdPark, chick holdout + bootstrap
# Every number the research agent may quote afterwards comes from the JSON files step 3 writes.
#
# Call type is scored on the CPU (finding 076: the Mac's MPS attention is wrong for 2-8 frame clips, which is most
# call-type clips). run15_calltype.py still runs first because it writes the MPS cache + JSON that
# calltype_cpu_rescore.py records beside the CPU score; QUOTE ONLY calltype_cpu_rescore.json (models.TAG) and
# seed_pairs_cpu.json ("TAG | aves-base-bio", "TAG | run16_compute4x"). The label-free score uses the CPU caches too
# (labelfree_stability_cpu.json). The chick holdout still encodes on MPS, like every reference in that JSON
# (97 of 415 chick clips are 2-8 frames): compare chick numbers only with each other, never with a CPU number.
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
H=${SCORE_HOST:-savio-login}   # savio-dtn while the login node's python gives EIO (2026-10-11)
ssh_s() { ssh -o BatchMode=yes "$H" "$@" 2> >(grep -v -iE "post-quantum|store now|upgraded|pq.html" >&2); }
run() { echo "+ $*"; [ -n "${DRY:-}" ] || "$@"; }

echo "== score $TAG  (exp $EXP, $NC classes, job ${JOB:-?})  $(date)"
CK=$(ssh_s "ls $CKDIR 2>/dev/null | grep -E '^epoch=[0-9]+-step=[0-9]+\.ckpt$' | sort -t= -k3 -n | tail -1")
[ -n "$CK" ] || { echo "FATAL: no epoch=*-step=*.ckpt in $CKDIR"; exit 2; }
echo "   final checkpoint: $CK"
ssh_s "test -f ~/export_weights.py && test -f $S/adultvoc_16k/BlaBla0506_110302-DC-01.wav" \
    || { echo "FATAL: ~/export_weights.py or the verification clip is missing on Savio"; exit 2; }
for f in run15_calltype calltype_cpu_rescore seed_pairs_cpu detection_variants chick_holdout_variants calltype_labelfree; do
    [ -f "$ROOT/zfeval/experiments/$f.py" ] || { echo "FATAL: zfeval/experiments/$f.py missing"; exit 2; }
done
"$PY" -c "import torch, sklearn, yaml" || { echo "FATAL: analysis python incomplete"; exit 2; }

echo "== 1. export on Savio"
EXPORT="source $CONDA && conda activate hubert_env && python ~/export_weights.py --ckpt '$CKDIR/$CK' \
--out $S/external/dapt/$TAG.pt --num-classes $NC --run-tag $TAG --wav $S/adultvoc_16k/BlaBla0506_110302-DC-01.wav"
echo "+ ssh $H $EXPORT"
if [ -z "${DRY:-}" ]; then
    ssh_s "$EXPORT 2>&1 | grep -v WARNING | tail -6; test -s $S/external/dapt/$TAG.pt" \
        || { echo "FATAL: export failed or wrote nothing"; exit 3; }
fi

echo "== 2. copy to the Mac"
M=$ROOT/savio_artifacts
run mkdir -p "$DAPT" "$M/metrics/$TAG" "$M/logs" "$M/analysis"
run rsync -a "$H:$S/external/dapt/$TAG.pt" "$DAPT/"
for v in $(ssh_s "ls $EXP/lightning_logs 2>/dev/null | grep '^version_'"); do
    run rsync -a "$H:$EXP/lightning_logs/$v/metrics.csv" "$M/metrics/$TAG/${TAG}_${v}_metrics.csv" || true
done
[ -z "$JOB" ] || run rsync -a "savio-login:jobs/*_${JOB}.log" "$M/logs/" || true

echo "== 3. score"
cd "$ROOT"
run "$PY" zfeval/experiments/run15_calltype.py --tag "$TAG"                 # MPS cache + JSON (for the record only)
run "$PY" zfeval/experiments/calltype_cpu_rescore.py --only "$TAG"         # CPU call type: the number to quote
run "$PY" zfeval/experiments/seed_pairs_cpu.py "$TAG:aves-base-bio" "$TAG:run16_compute4x"   # CPU bird bootstrap
run "$PY" zfeval/experiments/detection_variants.py --models "$TAG"
run "$PY" zfeval/experiments/chick_holdout_variants.py --models "$TAG"
run "$PY" zfeval/experiments/chick_holdout_variants.py --bootstrap --models "$TAG" --baseline run11
run "$PY" zfeval/experiments/calltype_labelfree.py --cpu --only "$TAG"     # label-free column (E11), CPU caches

if [ -n "${DRY:-}" ]; then echo "DRY RUN OK: every input exists"; exit 0; fi
[ -s "$A/${TAG}_calltype.json" ] || { echo "FATAL: $A/${TAG}_calltype.json was not written"; exit 4; }
"$PY" - "$A" "$TAG" <<'PY' || exit 4
import json, sys
a, tag = sys.argv[1], sys.argv[2]
cpu = json.load(open(f"{a}/calltype_cpu_rescore.json"))["models"]
assert tag in cpu, f"FATAL: {tag} missing from calltype_cpu_rescore.json"
pairs = json.load(open(f"{a}/seed_pairs_cpu.json"))["pairs"]
for ref in ("aves-base-bio", "run16_compute4x"):
    assert f"{tag} | {ref}" in pairs, f"FATAL: '{tag} | {ref}' missing from seed_pairs_cpu.json"
c = cpu[tag]
print(f"   CPU call type {c['acc11_cpu']:.4f} at L{c['best_layer_cpu']} (MPS {c['acc11_mps']:.4f} at L{c['best_layer_mps']}, "
      f"{c['n_clips_changed']} clips changed)")
for ref in ("aves-base-bio", "run16_compute4x"):
    r = pairs[f"{tag} | {ref}"]
    print(f"   vs {ref}: {r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]{' resolved' if r['resolved'] else ' n.d.'}")
PY
for f in detection_variants chick_holdout_variants labelfree_stability_cpu; do
    grep -q "\"$TAG" "$A/$f.json" || { echo "FATAL: $TAG missing from $f.json"; exit 4; }
done
cp "$A/${TAG}_calltype.json" "$A/calltype_cpu_rescore.json" "$A/seed_pairs_cpu.json" "$A/detection_variants.json" \
   "$A/chick_holdout_variants.json" "$A/labelfree_stability_cpu.json" "$M/analysis/"
echo "SCORED $TAG -> call type: calltype_cpu_rescore.json + seed_pairs_cpu.json (CPU; ${TAG}_calltype.json is MPS, do"
echo "   not quote it); detection_variants.json; chick_holdout_variants.json (MPS, like all its references);"
echo "   labelfree_stability_cpu.json"
