#!/bin/bash
#SBATCH --job-name=holdout_train
#SBATCH --account=fc_birdpow
#SBATCH --partition=savio4_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40:4
#SBATCH --time=12:00:00
#SBATCH --qos=savio_lowprio
#SBATCH --output=holdout_train_%j.log

# run11's recipe, byte for byte, on the 100-recording held-out corpus.
# The ONLY intended difference from run11 is the corpus. Everything else -- lr 1e-4,
# 93750 updates, 3125 warmup, feature_weight 0, HUBERT_NORMALIZE_INPUT=0, k=100,
# virtual-chunk 20s, 4x L40 -- is copied from slurm/train_iter7_long_lowprio.sh so the
# comparison against run11 is not confounded by the recipe.
#
# lowprio is preemptible; resume with --resume-checkpoint last.ckpt if it gets bumped.
set -uo pipefail
export HUBERT_NORMALIZE_INPUT=0

source /global/software/rocky-8.x86_64/manual/modules/langs/anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate hubert_env

EXP=/global/scratch/users/jonathanswang/temp_train_holdout_lblred0613
DATA=/global/scratch/users/jonathanswang/temp_files/holdout_lblred0613/data/spectrogram/

echo "=============================================="
echo "HuBERT holdout: LblRed0613 excluded from pretraining"
echo "=============================================="
echo "Start: $(date)"
test -s "$DATA/label/label_train.pt" || { echo "labels missing -- run 01_preprocess first"; exit 1; }
wc -l "$DATA/tsv/ZF_test_pipeline_train.tsv"

python -c "
import torch
print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('GPUs:', torch.cuda.device_count())
"

srun python /global/home/users/jonathanswang/pytorchAudio/examples/hubert/train.py \
    --gpus 4 \
    --dataset-path "$DATA" \
    --exp-dir "$EXP" \
    --feature-type spectrogram \
    --dataset ZF_test_pipeline \
    --num-classes 100 \
    --max-updates 93750 \
    --warmup-updates 3125 \
    --learning-rate 0.0001 \
    --feature-weight 0 \
    --virtual-chunk-seconds 20
rc=$?

echo "End: $(date)  train rc=$rc"
# COMPLETED is not success -- a crash at argparse has reported 0:0 before. Check the artifact.
ls -l "$EXP"/checkpoints_*/ 2>/dev/null || { echo "NO CHECKPOINT DIR"; exit 1; }
test -n "$(ls -A "$EXP"/checkpoints_*/ 2>/dev/null)" || { echo "NO CHECKPOINTS WRITTEN"; exit 1; }
exit $rc
