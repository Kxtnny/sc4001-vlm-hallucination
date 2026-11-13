#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=pope_rerun
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Reproducibility check on one TC1 V100: E0 sanity checks, then the UNMODIFIED model
# (alpha = 0) on POPE. Nothing is tuned or chosen from these results.
# On the V100 the model runs in float32 (DTYPE = "auto": no native bfloat16, and float16
# overflows), so a few answers may differ from the original bfloat16 run on the RTX 4060.
#
# Submit from the repo root:
#     sbatch tc1/pope_rerun.sbatch          # POPE random split (3,000 questions)
#     sbatch tc1/pope_rerun.sbatch all      # all three splits (9,000 questions)
# Resubmitting resumes where it stopped.
# Output: pope_rerun_<jobid>.out / .err in the repo root, and in ~/sc4001/results/:
#     sanity/sanity_report_tc1-v100.json
#     pope/rerun_tc1_v100/<split>.jsonl, <split>_metrics.json, run_meta.json
# Never use --tag baseline here: it would resume into (and mix with) the original results.

SPLIT=${1:-random}

source "$HOME/venvs/sc4001/bin/activate"   # created by tc1/setup_env.sh
set -eo pipefail

export SC4001_ROOT=$HOME/sc4001   # data/ and results/ live here. TC1 has NO backup: copy results off.
export HF_HOME=$HOME/hf_cache     # weights downloaded by tc1/setup_env.sh
export HF_HUB_OFFLINE=1           # do not depend on internet access from the compute node

echo "node $(hostname) | job $SLURM_JOB_ID | split $SPLIT | started $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

python scripts/sanity_check.py --tag tc1-v100
python scripts/eval_pope.py --split "$SPLIT" --tag rerun_tc1_v100
echo "finished $(date)"
