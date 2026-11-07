#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=e0_sanity
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# E0 sanity checks + speed benchmark on one TC1 V100.
# Submit from the repo root:   sbatch tc1/e0_sanity.sbatch
# Extra arguments go to sanity_check.py, e.g.:   sbatch tc1/e0_sanity.sbatch --dtype float32
# Output: e0_sanity_<jobid>.out / .err in the repo root, and
#         ~/sc4001/results/sanity/sanity_report_<gpu>.json

module load anaconda
source activate sc4001
set -eo pipefail

export SC4001_ROOT=$HOME/sc4001   # data/ and results/ live here. TC1 has NO backup: copy results to Drive.
export HF_HOME=$HOME/hf_cache     # weights downloaded by tc1/setup_env.sh
export HF_HUB_OFFLINE=1           # do not depend on internet access from the compute node

echo "node $(hostname) | job $SLURM_JOB_ID | started $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Fail fast with a clear message if this PyTorch build cannot run on the V100.
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "PyTorch sees no GPU (check --gres=gpu:1 and the torch install)"
major, minor = torch.cuda.get_device_capability(0)
print(f"torch {torch.__version__} | CUDA build {torch.version.cuda} | {torch.cuda.get_device_name(0)} sm_{major}{minor}")
x = torch.ones(64, 64, device="cuda", dtype=torch.float16)
assert float((x @ x)[0, 0]) == 64.0, "GPU matmul gave a wrong result"
print("GPU check OK")
EOF

python scripts/sanity_check.py "$@"
echo "finished $(date)"
