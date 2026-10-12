#!/bin/bash
#SBATCH --job-name=beans_detect
#SBATCH --account=fc_birdpow
#SBATCH --partition=savio4_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40:1
#SBATCH --time=12:00:00
#SBATCH --qos=savio_lowprio
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=beans_detect_%j.log
#
#   sbatch zfeval/slurm/beans_detect.sh            (from /global/scratch/users/jonathanswang/beans_eval)
#
# E10, the four BEANS detection tasks (beans_detect.py), moved off the Mac where they would hold the
# GPU for about a day. Every model x split embedding is cached as it finishes, so a preempted job
# that comes back on --requeue skips what is done. Results: ~/zf_labelset/zf_detection_dataset_v1/
# analysis/beans_<task>.json (~/zf_labelset is a symlink to scratch).
#
# PYTHONNOUSERSITE=1: ~/.local holds files the login node could not read (2026-10-11, a Lustre OST
# its client had lost) and nothing here needs them; hubert_env alone has torch/torchaudio/sklearn.
set -uo pipefail
export PYTHONNOUSERSITE=1
source /global/software/rocky-8.x86_64/manual/modules/langs/anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate hubert_env
cd /global/scratch/users/jonathanswang/beans_eval/zfeval/experiments || exit 2
echo "== $(date)  $(hostname)  restart ${SLURM_RESTART_COUNT:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; assert torch.cuda.is_available(), 'no GPU'" || exit 2

rc=0
for t in ${TASKS:-hiceas enabirds dcase hainan-gibbons}; do
    echo "== $t  $(date +%T)"
    python -u beans_detect.py --task "$t" --batch 32 || { echo "FAILED $t"; rc=1; }
done
echo "== end $(date)  rc=$rc"
exit $rc
