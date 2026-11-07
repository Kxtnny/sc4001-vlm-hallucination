#!/bin/bash
# One-time setup on the TC1 head node (CCDS-TC1). Run from the repo root:
#     bash tc1/setup_env.sh
#
# It only installs packages and downloads the model weights; it runs no model or
# GPU code (that is not allowed on the head node, and the head node has no GPU).
#
# Why a venv instead of `conda create`: on TC1, conda fails while unpacking
# ncurses into ~/.conda/pkgs ("Too many levels of symbolic links"), most likely
# because the home filesystem is case-insensitive. A plain Python venv needs no
# conda packages, so we build one on top of the course's Python
# (/tc1apps/2_conda_env/CZ4042_v6, Python 3.13 as of Nov 2026). We do NOT use that
# env's own packages: transformers==5.22.2 needs Python >= 3.11 and torch >= 2.6,
# and the venv gets its own torch.
#
# Why the cu126 wheels: TC1 has V100 GPUs (compute capability 7.0). Newer CUDA
# builds of PyTorch (CUDA 13) no longer include V100 kernels; CUDA 12.6 builds do.
# tc1/e0_sanity.sbatch checks this on the GPU node before running anything.

BASE_PYTHON=${BASE_PYTHON:-/tc1apps/2_conda_env/CZ4042_v6/bin/python}
VENV=${VENV:-$HOME/venvs/sc4001}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
set -eo pipefail

echo "Base Python: $BASE_PYTHON ($("$BASE_PYTHON" --version))"
if [ -x "$VENV/bin/python" ]; then
    echo "venv $VENV already exists, reusing it"
else
    mkdir -p "$(dirname "$VENV")"
    "$BASE_PYTHON" -m venv "$VENV"
fi
source "$VENV/bin/activate"
echo "Using $(which python) ($(python --version))"

python -m pip install --no-cache-dir --upgrade pip
python -m pip install --no-cache-dir "torch>=2.6" torchvision --index-url "$TORCH_INDEX"
python -m pip install --no-cache-dir -r requirements.txt
python -c "import torch, transformers; print('torch', torch.__version__, '(CUDA', torch.version.cuda, ') | transformers', transformers.__version__)"

# Download the weights now, so GPU jobs do not depend on internet access from the compute nodes.
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2-VL-2B-Instruct'))"

mkdir -p "$HOME/sc4001/data" "$HOME/sc4001/results"
echo
echo "Done. Next: copy the data folder from Drive to $HOME/sc4001/data (see tc1/README.md),"
echo "then submit:  sbatch tc1/e0_sanity.sbatch"
