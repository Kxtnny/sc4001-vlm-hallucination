#!/bin/bash
# One-time setup on the TC1 head node (CCDS-TC1). Run from the repo root:
#     bash tc1/setup_env.sh
#
# It only installs packages and downloads the model weights; it runs no model or
# GPU code (that is not allowed on the head node, and the head node has no GPU).
#
# Why our own env: transformers==5.22.2 needs Python >= 3.11 and torch >= 2.6,
# but the course env /tc1apps/2_conda_env/CZ4042_v6 has torch 2.5.1.
# Why the cu126 wheels: TC1 has V100 GPUs (compute capability 7.0). Newer CUDA
# builds of PyTorch (CUDA 13) no longer include V100 kernels; CUDA 12.6 builds do.
# tc1/e0_sanity.sbatch checks this on the GPU node before running anything.

ENV_NAME=${ENV_NAME:-sc4001}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}
export HF_HOME=${HF_HOME:-$HOME/hf_cache}

module load anaconda
set -eo pipefail

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "conda env '${ENV_NAME}' already exists, reusing it"
else
    conda create -y -n "$ENV_NAME" python=3.11
fi
source activate "$ENV_NAME"

python -m pip install --no-cache-dir "torch>=2.6" torchvision --index-url "$TORCH_INDEX"
python -m pip install --no-cache-dir -r requirements.txt
python -c "import torch, transformers; print('torch', torch.__version__, '(CUDA', torch.version.cuda, ') | transformers', transformers.__version__)"

# Download the weights now, so GPU jobs do not depend on internet access from the compute nodes.
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2-VL-2B-Instruct'))"

mkdir -p "$HOME/sc4001/data" "$HOME/sc4001/results"
echo
echo "Done. Next: copy the data folder from Drive to $HOME/sc4001/data (see tc1/README.md),"
echo "then submit:  sbatch tc1/e0_sanity.sbatch"
