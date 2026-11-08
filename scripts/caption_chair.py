"""E2: caption generation + CHAIR scoring (also used for the E4 intervention runs).

Generates one caption per image with the prompt "Describe this image in detail."
(greedy decoding), then scores every caption with CHAIR.

Outputs (under results/chair/<tag>/):
  <split>.jsonl          one line per image: caption, generated token ids, settings
  <split>_scored.jsonl   CHAIR details per caption (object mentions, which are hallucinated)
  <split>_metrics.json   CHAIR-s, CHAIR-i, recall, average length
                         (with --limit N these two are named <split>_nN_scored.jsonl / _metrics.json)
  run_meta.json          settings, library versions, git commit

Resumable like eval_pope.py: images already in <split>.jsonl are skipped.

Examples:
  python scripts/caption_chair.py --split chair_test                    # E2 baseline (500 test images)
  python scripts/caption_chair.py --split dev                           # dev captions (E3 layer choice)
  python scripts/caption_chair.py --split chair_test --metrics-only --show 5
  python scripts/caption_chair.py --split dev --limit 100 --tag a1.0_L8-15 --alpha 1.0 --layers 8-15
  python scripts/caption_chair.py --split chair_test --tag intervention --selected   # final E4 setting
"""
import _path  # noqa: F401
import argparse
import time

from tqdm import tqdm

from halluc import config
from halluc.chair import Chair, chair_metrics, coco_ground_truth_tables
from halluc.data import load_image, load_split, subsample
from halluc.io_utils import JsonlWriter, load_json, read_jsonl, run_metadata, save_json, selected_setting, write_jsonl


def load_chair_scorer():
    """CHAIR scorer + per-image ground truth (COCO instances + reference captions)."""
    chair = Chair(config.CHAIR_DIR / "synonyms.txt")
    instances = load_json(config.ANNOT_DIR / "instances_val2014.json")
    captions = load_json(config.ANNOT_DIR / "captions_val2014.json")
    cats, caps = coco_ground_truth_tables(instances, captions)
    return chair, cats, caps


def score_records(records, chair, cats, caps):
    scored = []
    for r in records:
        gt = chair.ground_truth(cats.get(r["image"], set()), caps.get(r["image"], []))
        s = chair.score_caption(r["caption"], gt)
        scored.append({"image": r["image"], "caption": r["caption"], **s})
    return scored


def highlight(caption, mentions):
    """Caption with hallucinated object words in [[double brackets]] and correct ones in <angle brackets>."""
    out, last = [], 0
    for m in mentions:
        out.append(caption[last:m["start"]])
        word = caption[m["start"]:m["end"]]
        out.append(f"[[{word}]]" if m["hallucinated"] else f"<{word}>")
        last = m["end"]
    out.append(caption[last:])
    return "".join(out)


def run_captions(vlm, split, tag, limit=None, alpha=0.0, layers="none",
                 max_new_tokens=config.CAPTION_MAX_NEW_TOKENS, scorer=None, run_args=None,
                 batch_size=config.CAPTION_BATCH_SIZE):
    """Generate (resuming) and CHAIR-score captions for one split; returns (metrics, scored).

    vlm=None only rescores captions that are already saved.
    """
    images = subsample(load_split(split)["images"], limit)
    out_dir = config.RESULTS_DIR / "chair" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}.jsonl"
    done = {r["image"]: r for r in read_jsonl(out_path)}
    todo = [im for im in images if im not in done]
    print(f"[{split}/{tag}] {len(images)} images selected, {len(images) - len(todo)} done, {len(todo)} to run "
          f"(alpha={alpha}, layers={layers})")

    if todo and vlm is not None:
        save_json(run_metadata(run_args, model=config.MODEL_ID, max_pixels=config.MAX_PIXELS,
                               min_pixels=config.MIN_PIXELS, decoding="greedy", prompt=config.CAPTION_PROMPT,
                               alpha=alpha, layers=layers, max_new_tokens=max_new_tokens,
                               batch_size=batch_size, dtype_resolved=vlm.dtype),
                  out_dir / f"run_meta_{split}.json")
        t0 = time.time()
        with JsonlWriter(out_path) as w, tqdm(total=len(todo), desc=f"captions[{tag}]") as bar:
            for i in range(0, len(todo), batch_size):
                batch = todo[i:i + batch_size]
                outs = vlm.generate_batch([load_image(im) for im in batch], [config.CAPTION_PROMPT] * len(batch),
                                          max_new_tokens=max_new_tokens, alpha=alpha, layers=layers)
                for im, (caption, new_ids) in zip(batch, outs):
                    rec = {"image": im, "caption": caption, "token_ids": new_ids, "n_new_tokens": len(new_ids),
                           "hit_max_tokens": len(new_ids) >= max_new_tokens and new_ids[-1] not in vlm.eos_ids,
                           "alpha": alpha, "layers": layers, "prompt": config.CAPTION_PROMPT,
                           "batch_size": batch_size}
                    w.write(rec)
                    done[im] = rec
                bar.update(len(batch))
        print(f"{(time.time() - t0) / len(todo):.1f} s per caption")

    selected = [done[im] for im in images if im in done]
    if not selected:
        raise SystemExit(f"No captions in {out_path} yet.")
    chair, cats, caps = scorer or load_chair_scorer()
    scored = score_records(selected, chair, cats, caps)
    m = chair_metrics(scored)
    m.update(split=split, tag=tag, limit=limit, complete=len(selected) == len(images),
             alpha=selected[0]["alpha"], layers=selected[0]["layers"],
             avg_new_tokens=sum(r["n_new_tokens"] for r in selected) / len(selected),
             frac_hit_max_tokens=sum(r["hit_max_tokens"] for r in selected) / len(selected))
    # Derived files are named by subset size, so a --limit run never overwrites the full-split files.
    name = split if limit is None else f"{split}_n{limit}"
    save_json(m, out_dir / f"{name}_metrics.json")
    write_jsonl(scored, out_dir / f"{name}_scored.jsonl")
    flag = "" if m["complete"] else "  (incomplete)"
    print(f"CHAIR  tag={tag} split={split} n={m['n_captions']}{flag}: "
          f"CHAIR-s {m['chair_s']:.3f} | CHAIR-i {m['chair_i']:.3f} | recall {m['recall']:.3f} | "
          f"{m['avg_len_words']:.1f} words | {m['frac_hit_max_tokens']:.1%} hit the token limit")
    return m, scored


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="chair_test", help="chair_test | dev")
    ap.add_argument("--limit", type=int, default=None, help="fixed random subset of N images")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--alpha", type=float, default=0.0, help="E4: image-attention boost (0 = original model)")
    ap.add_argument("--layers", default="none", help="E4: decoder layers to boost, e.g. 8-15, 0-27, none")
    ap.add_argument("--selected", action="store_true", help="use alpha/layers from results/sweep/selected.json")
    ap.add_argument("--max-new-tokens", type=int, default=config.CAPTION_MAX_NEW_TOKENS)
    ap.add_argument("--batch-size", type=int, default=config.CAPTION_BATCH_SIZE)
    ap.add_argument("--dtype", default=config.DTYPE)
    ap.add_argument("--metrics-only", action="store_true", help="do not load the model; rescore saved captions")
    ap.add_argument("--show", type=int, default=0, help="print N captions with hallucinated words marked")
    args = ap.parse_args()

    if args.selected:
        args.alpha, args.layers = selected_setting(config.RESULTS_DIR)
    if args.tag == "baseline" and args.alpha != 0:
        raise SystemExit("Use a different --tag for intervention runs, so results do not mix.")
    if args.split not in ("chair_test", "dev"):
        raise SystemExit("--split must be chair_test or dev")

    vlm = None
    if not args.metrics_only:
        from halluc.model import VLM
        vlm = VLM(dtype=args.dtype)
    _, scored = run_captions(vlm, args.split, args.tag, args.limit, args.alpha, args.layers,
                             args.max_new_tokens, run_args=args, batch_size=args.batch_size)
    for s in scored[:args.show]:
        print(f"\n{s['image']}  hallucinated={s['hallucinated_nodes']}  gt={s['gt_nodes']}")
        print("  " + highlight(s["caption"], s["mentions"]))


if __name__ == "__main__":
    main()
