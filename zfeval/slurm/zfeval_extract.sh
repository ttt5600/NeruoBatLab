#!/bin/bash
#SBATCH --job-name=zfeval
#SBATCH --account=fc_birdpow
#SBATCH --partition=savio3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --time=06:00:00
#SBATCH --qos=savio_lowprio
#SBATCH --output=zfeval_%j.log
#
#   sbatch --export=ALL,CKPT=<path>,TAG=run14,NUM_CLASSES=500 slurm/zfeval_extract.sh
#
# COMPLETED is not success: the script propagates the real exit code, and the extract stage
# aborts outright if the encoder does not fully load. Verify the artifact, not the SLURM state.
set -eo pipefail
source /global/software/rocky-8.x86_64/manual/modules/langs/anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate hubert_env
echo "Start: $(date)"
ZFEVAL=${ZFEVAL:-$HOME/vocalizations_lab/zfeval}
TAG=${TAG:-run}
NUM_CLASSES=${NUM_CLASSES:-100}
CONFIG=${CONFIG:-$ZFEVAL/config/datasets.yaml}
OUT=${OUT:-/global/scratch/users/jonathanswang/zfeval_runs/$TAG}
echo "TAG=$TAG NUM_CLASSES=$NUM_CLASSES CKPT=$CKPT"
python3 -u "$ZFEVAL/run_eval.py" extract \
    --config "$CONFIG" --ckpt "$CKPT" --out "$OUT" \
    --num-classes "$NUM_CLASSES" \
    --hubert-dir "$HOME/pytorchAudio/examples/hubert"
rc=$?
echo "End: $(date) rc=$rc"
# analyze needs no GPU, but running it here saves a round trip
python3 -u "$ZFEVAL/run_eval.py" analyze --out "$OUT" --config "$CONFIG" --name "$TAG"
exit $?
