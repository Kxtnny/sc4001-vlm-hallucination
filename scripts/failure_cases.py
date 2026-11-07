"""E6: failure analysis - where the intervention makes things worse or still hallucinates.

Compares the final intervention run with the baseline on the TEST sets (no new model
runs; reads saved JSON):

  CHAIR  "introduced": objects hallucinated with the intervention but not by the baseline
         "persistent": objects hallucinated by both
  POPE   "worse": questions the baseline got right and the intervention got wrong
         "still": questions both got wrong

Each case gets simple, checkable features, which are used to group cases by a likely cause:
  co-occurrence  the hallucinated / falsely-confirmed object often appears together with an
                 object that IS in the image (COCO statistics computed on val2014 images that
                 are in none of our test sets); threshold P(object | present object) >= 0.25
  small object   (POPE misses) the object's largest instance covers < 1% of the image
  late in caption (CHAIR) the mention is in the last third of the caption
  other          none of the above (read these by hand)
These rules are a first pass to speed up reading the cases, not a replacement for it:
the markdown file lists every sampled case so the team can check and re-label them.

Outputs: results/analysis/failure_cases.json, results/analysis/failure_cases.md,
         figures/e6_failure_cases.png

Usage:  python scripts/failure_cases.py [--intervention-tag intervention] [--n 16]
"""
import _path  # noqa: F401
import argparse
import random
import textwrap
from collections import Counter, defaultdict

from halluc import config
from halluc.chair import Chair
from halluc.data import coco_file_name, load_image, load_split
from halluc.io_utils import load_json, read_jsonl, save_json

COOC_T = 0.25
SMALL_T = 0.01


def cooccurrence_table(instances, chair, exclude):
    """P(B present | A present) over val2014 images not in `exclude`, by CHAIR category name."""
    cat_name = {c["id"]: chair.inverse_synonym[c["name"]] for c in instances["categories"]}
    objs = defaultdict(set)
    for a in instances["annotations"]:
        fn = coco_file_name(a["image_id"])
        if fn not in exclude:
            objs[fn].add(cat_name[a["category_id"]])
    count, pair = Counter(), Counter()
    for s in objs.values():
        for x in s:
            count[x] += 1
            for y in s:
                if x != y:
                    pair[(x, y)] += 1
    return lambda b, present: max([pair[(a, b)] / count[a] for a in present if count[a]] or [0.0])


def largest_area_fraction(instances):
    img_area = {im["id"]: im["width"] * im["height"] for im in instances["images"]}
    cat_name = {c["id"]: c["name"] for c in instances["categories"]}
    best = defaultdict(float)
    for a in instances["annotations"]:
        key = (coco_file_name(a["image_id"]), cat_name[a["category_id"]])
        best[key] = max(best[key], a["area"] / img_area[a["image_id"]])
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--intervention-tag", default="intervention")
    ap.add_argument("--n", type=int, default=16, help="cases to sample for the figure / markdown")
    args = ap.parse_args()
    R = config.RESULTS_DIR
    chair = Chair(config.CHAIR_DIR / "synonyms.txt")
    instances = load_json(config.ANNOT_DIR / "instances_val2014.json")
    test_imgs = set(load_split("chair_test")["images"]) | set(load_split("pope")["images"])
    cooc = cooccurrence_table(instances, chair, exclude=test_imgs)
    area = largest_area_fraction(instances)
    cats_by_img = defaultdict(set)
    for fn, n in area:
        cats_by_img[fn].add(chair.inverse_synonym[n])

    cases = []
    # ---- CHAIR ----
    base = {r["image"]: r for r in read_jsonl(R / "chair" / "baseline" / "chair_test_scored.jsonl")}
    intv = {r["image"]: r for r in read_jsonl(R / "chair" / args.intervention_tag / "chair_test_scored.jsonl")}
    for im in sorted(set(base) & set(intv)):
        b, v = base[im], intv[im]
        present = set(v["gt_nodes"])
        for m in v["mentions"]:
            if not m["hallucinated"]:
                continue
            kind = "introduced" if m["node"] not in b["hallucinated_nodes"] else "persistent"
            rel = m["start"] / max(1, len(v["caption"]))
            c = cooc(m["node"], present)
            cause = "co-occurrence" if c >= COOC_T else "late in caption" if rel > 2 / 3 else "other"
            cases.append({"bench": "CHAIR", "kind": kind, "image": im, "object": m["node"], "word": m["word"],
                          "cooccurrence": round(c, 3), "rel_pos": round(rel, 3), "cause": cause,
                          "baseline_caption": b["caption"], "intervention_caption": v["caption"],
                          "gt": sorted(present)})
    # Deduplicate (an object can be mentioned several times in one caption).
    seen, uniq = set(), []
    for c in cases:
        k = (c["image"], c["object"])
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    cases = uniq

    # ---- POPE ----
    for split in config.POPE_SPLITS:
        b = {r["question_id"]: r for r in read_jsonl(R / "pope" / "baseline" / f"{split}.jsonl")}
        v = {r["question_id"]: r for r in read_jsonl(R / "pope" / args.intervention_tag / f"{split}.jsonl")}
        for qid in sorted(set(b) & set(v)):
            rb, rv = b[qid], v[qid]
            if rv["pred"] == rv["label"]:
                continue
            kind = "worse" if rb["pred"] == rb["label"] else "still"
            obj = rv["text"].replace("Is there a ", "").replace("Is there an ", "").replace(" in the image?", "")
            gt = sorted(cats_by_img[rv["image"]])
            if rv["label"] == "no":   # false "yes": object is absent
                c = cooc(chair.inverse_synonym.get(obj, obj), set(gt))
                cause = "co-occurrence" if c >= COOC_T else "other"
                feat = {"cooccurrence": round(c, 3)}
            else:                     # false "no": object is present
                frac = area.get((rv["image"], obj), 0.0)
                cause = "small object" if frac < SMALL_T else "other"
                feat = {"area_fraction": round(frac, 4)}
            cases.append({"bench": f"POPE-{split}", "kind": kind, "image": rv["image"], "object": obj,
                          "label": rv["label"], "baseline_answer": rb["answer"],
                          "intervention_answer": rv["answer"], "cause": cause, "gt": gt, **feat})

    summary = defaultdict(Counter)
    for c in cases:
        summary[f"{c['bench']} {c['kind']}"][c["cause"]] += 1
    summary = {k: dict(v) for k, v in sorted(summary.items())}

    # Sample cases for reading: half from "made worse" (introduced/worse), half "still wrong".
    rng = random.Random(config.SEED)
    worse = [c for c in cases if c["kind"] in ("introduced", "worse")]
    still = [c for c in cases if c["kind"] in ("persistent", "still")]
    sample = rng.sample(worse, min(len(worse), args.n // 2))
    sample += rng.sample(still, min(len(still), args.n - len(sample)))

    save_json({"intervention_tag": args.intervention_tag, "n_cases": len(cases), "summary_by_cause": summary,
               "thresholds": {"cooccurrence": COOC_T, "small_object_area": SMALL_T},
               "sample": sample, "all_cases": cases}, R / "analysis" / "failure_cases.json")

    lines = ["# E6 failure cases (auto-grouped; check each by hand)", "",
             f"Intervention tag: `{args.intervention_tag}`. Counts by benchmark/kind and heuristic cause:", ""]
    for k, v in summary.items():
        lines.append(f"- **{k}**: " + ", ".join(f"{c} {n}" for c, n in sorted(v.items(), key=lambda x: -x[1])))
    lines += ["", "## Sampled cases", ""]
    for i, c in enumerate(sample, 1):
        lines.append(f"### {i}. {c['bench']} - {c['kind']} - `{c['object']}` ({c['cause']})")
        lines.append(f"Image `{c['image']}`; ground-truth objects: {', '.join(c['gt'])}")
        if c["bench"] == "CHAIR":
            lines.append(f"- baseline: {c['baseline_caption']}")
            lines.append(f"- intervention: {c['intervention_caption']}")
        else:
            lines.append(f"- label {c['label']}; baseline said {c['baseline_answer']!r}, "
                         f"intervention said {c['intervention_answer']!r}")
        lines.append("")
    (R / "analysis" / "failure_cases.md").write_text("\n".join(lines), encoding="utf-8")
    print("Summary:", summary)

    # Figure: up to 8 sampled cases with image + short text.
    import matplotlib.pyplot as plt
    from halluc.plots import _save
    show = sample[:8]
    if show:
        cols = 4
        rows = (len(show) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(7.4, 2.15 * rows), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, c in zip(axes.flat, show):
            img = load_image(c["image"])
            w, h = img.size                       # centre-crop to 4:3 so every panel has the same shape
            tw, th = (w, int(w * 3 / 4)) if w * 3 / 4 <= h else (int(h * 4 / 3), h)
            img = img.crop(((w - tw) // 2, (h - th) // 2, (w + tw) // 2, (h + th) // 2))
            img.thumbnail((320, 240))
            ax.imshow(img)
            if c["bench"] == "CHAIR":
                txt = f"CHAIR {c['kind']}: '{c['word']}' ({c['cause']})"
            else:
                txt = (f"{c['bench']}: {c['object']}? label {c['label']}, said "
                       f"{c['intervention_answer'][:3]} ({c['cause']})")
            ax.set_title("\n".join(textwrap.wrap(txt, 34)), fontsize=7)
        fig.tight_layout(h_pad=0.6, w_pad=0.4)
        _save(fig, config.FIGURE_DIR / "e6_failure_cases.png")
    print("Saved", R / "analysis" / "failure_cases.json", "and failure_cases.md")


if __name__ == "__main__":
    main()
