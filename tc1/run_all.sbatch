#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=run_all
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# All experiments E0-E6 + report tables (scripts/run_all.sh) on one TC1 V100.
# On a V100 the model runs in float32 (DTYPE = "auto"; the V100 has no native bfloat16 and
# float16 overflows). Every step resumes, so if the 6-hour limit is hit just submit again:
#     sbatch tc1/run_all.sbatch
# Results: ~/sc4001/results (TC1 has NO backup: copy them to Drive).

source "$HOME/venvs/sc4001/bin/activate"   # created by tc1/setup_env.sh
set -eo pipefail

export SC4001_ROOT=$HOME/sc4001
export HF_HOME=$HOME/hf_cache
export HF_HUB_OFFLINE=1
export PY=python

echo "node $(hostname) | job $SLURM_JOB_ID | started $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
bash scripts/run_all.sh
echo "finished $(date)"
