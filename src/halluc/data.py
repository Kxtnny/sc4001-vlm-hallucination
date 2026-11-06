"""Loading POPE questions, COCO images and our frozen dev/test splits."""
import random
from pathlib import Path

from PIL import Image

from . import config
from .io_utils import load_json, read_jsonl


def coco_file_name(image_id):
    return f"COCO_val2014_{int(image_id):012d}.jpg"


def load_image(file_name, image_dir=None):
    path = Path(image_dir or config.IMAGE_DIR) / file_name
    return Image.open(path).convert("RGB")


def load_pope(split, pope_dir=None):
    """POPE questions for one split ("random", "popular", "adversarial", or "dev_random").

    Each row: {"question_id", "image", "text", "label"}; label is "yes" or "no".
    POPE files are JSON Lines even though they end in .json.
    """
    pope_dir = Path(pope_dir or config.POPE_DIR)
    name = "dev_pope_random.jsonl" if split == "dev_random" else f"coco_pope_{split}.json"
    rows = read_jsonl(pope_dir / name)
    if not rows:
        raise FileNotFoundError(f"No POPE questions at {pope_dir / name}. Run scripts/prepare_data.py first.")
    return rows


def subsample(rows, limit, seed=config.SEED):
    """A fixed random subset (same seed -> same subset every time), kept in file order.

    Because the subset is fixed, a --limit 50 timing run and a later --limit 300
    or full run share their first examples, so the resume logic reuses them.
    """
    if not limit or limit >= len(rows):
        return rows
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)  # same permutation for every limit -> nested subsets
    idx = sorted(order[:limit])
    return [rows[i] for i in idx]


def load_split(name, split_dir=None):
    """Image list for "dev", "chair_test" or "pope" (see prepare_data.py)."""
    return load_json(Path(split_dir or config.SPLIT_DIR) / f"{name}_images.json")
