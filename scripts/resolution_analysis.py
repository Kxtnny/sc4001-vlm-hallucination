"""Resolution experiment: does a larger image budget reduce POPE misses?

Reads the runs made by tc1/pope_resolution.sbatch (unmodified model, same questions and batches,
three image budgets) and compares each larger budget with the default-budget control:
  * accuracy, precision, recall, yes-ratio per split;
  * exact McNemar test on the same questions (two-sided, not corrected);
  * recall on present objects split by object size: "small" = the object's largest instance
    covers < 1% of the image (the E6 threshold), "larger" = the rest;
  * how many of the control's misses are recovered and how many correct "yes" answers are lost.
No model run. Output: results/analysis/resolution.json

Usage:  python scripts/resolution_analysis.py [--control res_default_b3] [--tags res576_b3 res1024_b3]
        python scripts/resolution_analysis.py --control baseline --tags intervention --out-name recall_by_size
            (recall by object size for the unmodified model and the attention boost; used in Section 5.2)
"""
import _path  # noqa: F401
import argparse
import math
from collections import defaultdict

from halluc import config
from halluc.data import coco_file_name
from halluc.io_utils import load_json, read_jsonl, save_json
from halluc.metrics import pope_metrics

SMALL_T = 0.01   # same threshold as scripts/failure_cases.py


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def largest_area_fraction(instances):
    img_area = {im["id"]: im["width"] * im["height"] for im in instances["images"]}
    cat_name = {c["id"]: c["name"] for c in instances["categories"]}
    best = defaultdict(float)
    for a in instances["annotations"]:
        key = (coco_file_name(a["image_id"]), cat_name[a["category_id"]])
        best[key] = max(best[key], a["area"] / img_area[a["image_id"]])
    return best


def object_of(question):
    return question.replace("Is there a ", "").replace("Is there an ", "").replace(" in the image?", "")


def recall(rows):
    return sum(r["pred"] == "yes" for r in rows) / len(rows) if rows else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--control", default="res_default_b3")
    ap.add_argument("--tags", nargs="+", default=["res576_b3", "res1024_b3"])
    ap.add_argument("--out-name", default="resolution", help="results/analysis/<out-name>.json")
    args = ap.parse_args()
    pope = config.RESULTS_DIR / "pope"
    area = largest_area_fraction(load_json(config.ANNOT_DIR / "instances_val2014.json"))
    out = {"control": args.control, "small_object_area": SMALL_T, "splits": {}}

    for split in config.POPE_SPLITS:
        ctrl = {r["question_id"]: r for r in read_jsonl(pope / args.control / f"{split}.jsonl")}
        if not ctrl:
            continue
        res = {}
        for tag in [args.control] + args.tags:
            rows = {r["question_id"]: r for r in read_jsonl(pope / tag / f"{split}.jsonl")}
            common = sorted(set(ctrl) & set(rows))
            if not rows or len(common) < len(ctrl):
                print(f"[{split}] {tag}: {len(common)} of {len(ctrl)} questions answered - skipped (incomplete)")
                continue
            assert all(rows[q]["text"] == ctrl[q]["text"] and rows[q]["label"] == ctrl[q]["label"] for q in common)
            sel = [rows[q] for q in common]
            m = pope_metrics(sel)
            pos = [r for r in sel if r["label"] == "yes"]
            small = [r for r in pos if area.get((r["image"], object_of(r["text"])), 0.0) < SMALL_T]
            large = [r for r in pos if area.get((r["image"], object_of(r["text"])), 0.0) >= SMALL_T]
            entry = {"n": len(sel), "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"],
                     "yes_ratio": m["yes_ratio"], "false_yes": m["fp"],
                     "recall_small": recall(small), "n_small": len(small),
                     "recall_larger": recall(large), "n_larger": len(large)}
            if tag != args.control:
                ok = lambda d, q: d[q]["pred"] == d[q]["label"]
                only_ctrl = sum(ok(ctrl, q) and not ok(rows, q) for q in common)
                only_new = sum(ok(rows, q) and not ok(ctrl, q) for q in common)
                pos_q = [q for q in common if ctrl[q]["label"] == "yes"]
                entry.update(control_only_right=only_ctrl, new_only_right=only_new,
                             mcnemar_p=mcnemar_exact(only_ctrl, only_new),
                             misses_recovered=sum(ctrl[q]["pred"] == "no" and rows[q]["pred"] == "yes" for q in pos_q),
                             hits_lost=sum(ctrl[q]["pred"] == "yes" and rows[q]["pred"] == "no" for q in pos_q))
            res[tag] = entry
            extra = ("" if tag == args.control else
                     f" | vs control: +{entry['new_only_right']}/-{entry['control_only_right']} answers, "
                     f"McNemar p = {entry['mcnemar_p']:.3f}; misses recovered {entry['misses_recovered']}, hits lost {entry['hits_lost']}")
            print(f"[{split}] {tag:16s} acc {100 * entry['accuracy']:.1f} prec {100 * entry['precision']:.1f} "
                  f"rec {100 * entry['recall']:.1f} yes {100 * entry['yes_ratio']:.1f} | recall small objects "
                  f"{100 * entry['recall_small']:.1f} (n={entry['n_small']}), larger {100 * entry['recall_larger']:.1f} "
                  f"(n={entry['n_larger']}){extra}")
        out["splits"][split] = res
    save_json(out, config.RESULTS_DIR / "analysis" / f"{args.out_name}.json")
    print("saved", config.RESULTS_DIR / "analysis" / f"{args.out_name}.json")


if __name__ == "__main__":
    main()
