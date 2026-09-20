#!/bin/bash -l
#SBATCH --job-name=holdout_prep
#SBATCH --account=fc_birdpow
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=holdout_prep_%j.log
#SBATCH --requeue

# Features + KMeans + labels for the 100-recording held-out corpus.
#
# --start-from features (NOT labels): the k-means codebook must be refit on the reduced corpus.
# Reusing run11's codebook would leak the held-out bird in through the pretraining target
# vocabulary, which is exactly the exposure this experiment is trying to remove.
set -euo pipefail

module load anaconda3
source activate hubert_env
export MPLBACKEND=Agg

EXP=/global/scratch/users/jonathanswang/temp_files/holdout_lblred0613/
HUB=/global/home/users/jonathanswang/pytorchAudio/examples/hubert

echo "=== holdout preprocess: $(date) ==="
test -s "$EXP/data/spectrogram/tsv/ZF_test_pipeline_train.tsv" || { echo "TSV missing -- run 00_build_tsv.sh"; exit 1; }
wc -l "$EXP/data/spectrogram/tsv/ZF_test_pipeline_train.tsv"

python "$HUB/preprocess.py" \
    --dataset ZF_test_pipeline \
    --root-dir /global/scratch/users/jelie/ZF_rec/train \
    --feat-type spectrogram \
    --exp-dir "$EXP" \
    --num-cluster 100 \
    --num-rank 1 \
    --layer-index 6 \
    --percent -1 \
    --kernel-size-ms 25 \
    --stride-ms 20 \
    --skip-vad \
    --skip-chunk \
    --use-gpu \
    --kmeans-backend gpu \
    --kmeans-subsample-frames 500000 \
    --kmeans-batch-size 65536 \
    --start-from features
rc=$?; [ $rc -ne 0 ] && { echo "preprocess FAILED rc=$rc"; exit $rc; }

echo "=== verify bridge ==="
python "$HUB/verify_bridge.py" \
    --exp-dir "$EXP" --feat-type spectrogram --dataset ZF_test_pipeline \
    --num-rank 1 --skip-dataset-load
rc=$?; [ $rc -ne 0 ] && { echo "verify_bridge FAILED rc=$rc"; exit $rc; }

# A SLURM state of COMPLETED does not mean the work ran -- check the artifact.
test -s "$EXP/data/spectrogram/label/label_train.pt" || { echo "NO LABELS PRODUCED"; exit 1; }
echo "=== done: $(date) ==="
ls -l "$EXP/data/spectrogram/label/" "$EXP/data/spectrogram/km_model/"
