"""Download data and build the frozen dev / test splits. Run ONCE for the whole team.

What it does:
  1. Downloads the three POPE COCO question files (random, popular, adversarial).
  2. Downloads COCO 2014 annotations and keeps instances_val2014.json and
     captions_val2014.json (needed for CHAIR and for the dev questions).
  3. Builds three disjoint image sets from COCO val2014 (fixed seed):
       pope        - the 500 images POPE uses (all 3 splits share them)  -> TEST
       chair_test  - 500 other images for CHAIR captioning (E2)         -> TEST
       dev         - 200 other images for choosing alpha/layers/prompts -> DEV
  4. Writes dev POPE-style questions (random-negative protocol) so the
     intervention can also be tuned on yes/no questions without touching POPE.
  5. Downloads the CHAIR synonym list from the original CHAIR repository.
  6. Downloads only the ~1,200 images we need (not the 6 GB val2014 zip).

Splits are never overwritten unless you pass --force-splits. That is on purpose:
changing them after you have tuned anything would leak test data into dev.

Usage:  python scripts/prepare_data.py            (about 5-10 min on Colab)
"""
import _path  # noqa: F401
import argparse
import json
import random
import shutil
import tempfile
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from halluc import config
from halluc.data import coco_file_name
from halluc.io_utils import load_json, read_jsonl, save_json


def download(url, dest):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dest)


def get_pope():
    for split in config.POPE_SPLITS:
        dest = config.POPE_DIR / f"coco_pope_{split}.json"
        download(config.POPE_URL.format(split=split), dest)
        rows = read_jsonl(dest)
        n_yes = sum(r["label"] == "yes" for r in rows)
        print(f"POPE {split}: {len(rows)} questions, {len({r['image'] for r in rows})} images, {n_yes} yes")


def get_annotations():
    needed = ["instances_val2014.json", "captions_val2014.json"]
    if all((config.ANNOT_DIR / n).exists() for n in needed):
        print("COCO annotations already present")
        return
    config.ANNOT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:  # local disk, not Drive: the zip is ~240 MB
        zpath = Path(tmp) / "annotations.zip"
        print("Downloading COCO 2014 annotations (~240 MB)...")
        download(config.COCO_ANNOT_URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            for n in needed:
                with z.open(f"annotations/{n}") as src, open(config.ANNOT_DIR / n, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    print("Saved", needed, "to", config.ANNOT_DIR)


def get_chair_synonyms():
    dest = config.CHAIR_DIR / "synonyms.txt"
    download(config.CHAIR_SYNONYMS_URL, dest)
    n = sum(1 for ln in dest.read_text(encoding="utf-8").splitlines() if ln.strip())
    print(f"CHAIR synonyms: {n} categories -> {dest}")


def build_splits(instances, force):
    out = {name: config.SPLIT_DIR / f"{name}_images.json" for name in ("pope", "dev", "chair_test")}
    if all(p.exists() for p in out.values()) and not force:
        print("Splits already exist (frozen). Use --force-splits only if nothing has been tuned yet.")
        return {k: load_json(p) for k, p in out.items()}

    pope_files = sorted({r["image"] for s in config.POPE_SPLITS
                         for r in read_jsonl(config.POPE_DIR / f"coco_pope_{s}.json")})
    # Candidate images: val2014 images with at least one annotated object, not used by POPE.
    with_objects = {a["image_id"] for a in instances["annotations"]}
    pope_set = set(pope_files)
    candidates = sorted(i for i in with_objects if coco_file_name(i) not in pope_set)
    random.Random(config.SEED).shuffle(candidates)
    dev_ids = sorted(candidates[:config.N_DEV_IMAGES])
    chair_ids = sorted(candidates[config.N_DEV_IMAGES:config.N_DEV_IMAGES + config.N_CHAIR_IMAGES])

    def pack(file_names, role):
        return {"role": role, "seed": config.SEED, "n": len(file_names), "images": file_names}

    splits = {
        "pope": pack(pope_files, "test (POPE)"),
        "dev": pack([coco_file_name(i) for i in dev_ids], "dev: tuning only"),
        "chair_test": pack([coco_file_name(i) for i in chair_ids], "test (CHAIR)"),
    }
    a, b, c = (set(splits[k]["images"]) for k in ("pope", "dev", "chair_test"))
    assert not (a & b) and not (a & c) and not (b & c), "splits overlap!"
    for k, p in out.items():
        save_json(splits[k], p)
        print(f"split {k}: {splits[k]['n']} images -> {p}")
    return splits


def build_dev_questions(instances, dev_files, per_image=3):
    """POPE-style yes/no questions on dev images, following POPE's *random* setting:
    positives are ground-truth COCO objects, negatives are drawn uniformly from
    object categories that are absent from the image."""
    dest = config.POPE_DIR / "dev_pope_random.jsonl"
    if dest.exists():
        print("Dev questions already exist:", dest)
        return
    cat_name = {c["id"]: c["name"] for c in instances["categories"]}
    objs = {}
    for a in instances["annotations"]:
        objs.setdefault(a["image_id"], set()).add(cat_name[a["category_id"]])
    rng = random.Random(config.SEED)
    all_names = sorted(cat_name.values())
    rows, qid = [], 0
    for fn in dev_files:
        present = sorted(objs[int(fn.split("_")[-1].split(".")[0])])
        pos = rng.sample(present, min(per_image, len(present)))
        neg = rng.sample([n for n in all_names if n not in present], len(pos))
        for name, label in [(n, "yes") for n in pos] + [(n, "no") for n in neg]:
            qid += 1
            rows.append({"question_id": qid, "image": fn,
                         "text": f"Is there a {name} in the image?", "label": label})
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Dev questions: {len(rows)} ({sum(r['label'] == 'yes' for r in rows)} yes) -> {dest}")


def get_images(file_names, workers=16):
    config.IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    def fetch(fn):
        dest = config.IMAGE_DIR / fn
        try:
            download(config.COCO_IMAGE_URL.format(file_name=fn), dest)
            Image.open(dest).verify()  # catches truncated downloads
            return None
        except Exception as e:
            dest.unlink(missing_ok=True)
            return f"{fn}: {e}"

    with ThreadPoolExecutor(workers) as ex:
        errors = [e for e in tqdm(ex.map(fetch, file_names), total=len(file_names), desc="images") if e]
    print(f"Images ready: {len(file_names) - len(errors)}/{len(file_names)} in {config.IMAGE_DIR}")
    if errors:
        print("Failed (re-run the script to retry):", *errors[:10], sep="\n  ")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force-splits", action="store_true", help="rebuild splits (only before any tuning!)")
    ap.add_argument("--skip-images", action="store_true")
    args = ap.parse_args()
    print("Data folder:", config.DATA_DIR)

    get_pope()
    get_annotations()
    get_chair_synonyms()
    instances = load_json(config.ANNOT_DIR / "instances_val2014.json")
    splits = build_splits(instances, args.force_splits)
    build_dev_questions(instances, splits["dev"]["images"])
    if not args.skip_images:
        all_files = sorted(set().union(*(s["images"] for s in splits.values())))
        get_images(all_files)


if __name__ == "__main__":
    main()
