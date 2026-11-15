#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=pope_res
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Resolution experiment (follow-up to E5/E6). Most POPE errors of this model are MISSES of small
# objects, made while it attends strongly to the image. If that is a perception limit, more image
# tokens per object should raise recall where the attention boost did not.
#
# The UNMODIFIED model (alpha = 0) answers POPE with three image budgets:
#     default  64-256 tokens (the budget of every other experiment)
#     576      every image resized to about 576 tokens
#     1024     every image resized to about 1,024 tokens
# All three use batch size 3, so they see identically composed batches and the largest budget
# stays well inside the V100's memory (the other experiments use batch 12 at 256 tokens).
# Nothing is tuned on these results; the default-budget run is the control for the other two.
#
# Submit from the repo root:
#     sbatch tc1/pope_resolution.sbatch          # POPE random split (3,000 questions per budget)
#     sbatch tc1/pope_resolution.sbatch all      # all three splits (9,000 per budget; may need a resubmit)
# Resubmitting resumes where it stopped. Afterwards:
#     python scripts/resolution_analysis.py      # table + results/analysis/resolution.json (no GPU)
# Output in ~/sc4001/results/pope/: res_default_b3/, res576_b3/, res1024_b3/

SPLIT=${1:-random}

source "$HOME/venvs/sc4001/bin/activate"   # created by tc1/setup_env.sh
set -eo pipefail

export SC4001_ROOT=$HOME/sc4001
export HF_HOME=$HOME/hf_cache
export HF_HUB_OFFLINE=1

echo "node $(hostname) | job $SLURM_JOB_ID | split $SPLIT | started $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Smoke test first (24 questions at the largest budget), so a memory problem shows in the first minutes.
SC4001_IMAGE_TOKENS=1024 python scripts/eval_pope.py --split random --limit 24 --batch-size 3 --tag res1024_b3_smoke

python scripts/eval_pope.py --split "$SPLIT" --batch-size 3 --tag res_default_b3
echo "default budget finished $(date)"
SC4001_IMAGE_TOKENS=576 python scripts/eval_pope.py --split "$SPLIT" --batch-size 3 --tag res576_b3
echo "576 tokens finished $(date)"
SC4001_IMAGE_TOKENS=1024 python scripts/eval_pope.py --split "$SPLIT" --batch-size 3 --tag res1024_b3
echo "1024 tokens finished $(date)"

python scripts/resolution_analysis.py
echo "finished $(date)"
