"""Shared settings for every experiment.

Change values here (not inside scripts) so that every run uses the same model,
image size and prompts. Every script also writes the settings it used into a
run_meta.json next to its outputs, so the report can quote exact settings.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths. On Colab, ROOT is a folder on Google Drive so nothing is lost when the
# runtime disconnects. Override with the SC4001_ROOT environment variable.
# ---------------------------------------------------------------------------
ROOT = Path(os.environ.get("SC4001_ROOT", "/content/drive/MyDrive/sc4001"))
DATA_DIR = ROOT / "data"
RESULTS_DIR = Path(os.environ.get("SC4001_RESULTS", ROOT / "results"))   # override for smoke tests
FIGURE_DIR = RESULTS_DIR / "figures"     # report figures, built from saved JSON only

IMAGE_DIR = DATA_DIR / "images" / "val2014"
ANNOT_DIR = DATA_DIR / "annotations"
POPE_DIR = DATA_DIR / "pope"
SPLIT_DIR = DATA_DIR / "splits"

# ---------------------------------------------------------------------------
# Model. One model only (see project plan, Section 1).
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DTYPE = "auto"             # bfloat16 if the GPU supports it natively (A100, L4, RTX 30/40), else float32.
                           # NOT float16: Qwen2-VL-2B overflows (NaN logits) in float16 - see E0 / RUN_LOG.md.
ATTN_IMPL = "eager"        # Required: sdpa / flash attention do not return attention weights.

# Fixed image budget so every method sees the same number of image tokens.
# One image token covers a 28x28 pixel area (14px patches, merged 2x2).
# 256 * 28 * 28 = 200,704 px -> at most 256 image tokens per image.
MIN_PIXELS = 64 * 28 * 28
MAX_PIXELS = 256 * 28 * 28

# ---------------------------------------------------------------------------
# Prompts. Keep them here so the baseline and later variants are comparable.
# ---------------------------------------------------------------------------
POPE_SUFFIX = " Answer with yes or no."          # appended to the POPE question
POPE_MAX_NEW_TOKENS = 5
CAPTION_PROMPT = "Describe this image in detail."  # used for CHAIR (E2)
CAPTION_MAX_NEW_TOKENS = 256

# Batched greedy decoding (left padding). Every method uses the same batch size and
# order, so baseline and intervention see identically composed batches.
POPE_BATCH_SIZE = 12
CAPTION_BATCH_SIZE = 8

# ---------------------------------------------------------------------------
# Data splits (built once by scripts/prepare_data.py, then frozen).
# ---------------------------------------------------------------------------
SEED = 0
N_DEV_IMAGES = 200      # for choosing alpha / layers / prompts. Never test images.
N_CHAIR_IMAGES = 500    # CHAIR test images. Disjoint from POPE and dev.
POPE_SPLITS = ("random", "popular", "adversarial")
POPE_URL = "https://raw.githubusercontent.com/AoiDragon/POPE/main/output/coco/coco_pope_{split}.json"
COCO_IMAGE_URL = "http://images.cocodataset.org/val2014/{file_name}"
COCO_ANNOT_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"

# CHAIR synonym list from the original CHAIR repository (Rohrbach et al., 2018), used unchanged.
CHAIR_DIR = DATA_DIR / "chair"
CHAIR_SYNONYMS_URL = "https://raw.githubusercontent.com/LisaAnne/Hallucination/master/data/synonyms.txt"

# ---------------------------------------------------------------------------
# E4 intervention: attention to image tokens is multiplied by (1 + alpha) in a
# range of decoder layers, then each attention row is renormalised.
# The final setting is chosen on the dev set by scripts/sweep.py and written to
# results/sweep/selected.json; scripts read it from there (never from test data).
# ---------------------------------------------------------------------------
SWEEP_ALPHAS = (0.5, 1.0, 2.0, 4.0)
SWEEP_LAYER_RANGES = ("0-7", "8-15", "16-27", "0-27")
N_DEV_CAPTION_IMAGES = 100   # dev images used for caption-based tuning (subset of the 200 dev images)

# E5 comparison baseline: a cautious POPE prompt (chosen before seeing any test result).
CAUTIOUS_SUFFIX = " Answer yes only if the object is clearly visible in the image, otherwise answer no."
