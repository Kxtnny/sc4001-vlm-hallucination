"""E1: POPE evaluation (accuracy, precision, recall, F1, yes-ratio per split).

Outputs (on Drive, under results/pope/<tag>/):
  <split>.jsonl          one line per question: question, raw answer, parsed answer, label
  <split>_metrics.json   metrics for the questions selected by --limit (or all)
  run_meta.json          settings, library versions, git commit

Resumable: questions already in <split>.jsonl are skipped, so after a Colab
disconnect just run the same command again.

Examples:
  python scripts/eval_pope.py --split random --limit 50          # timing run
  python scripts/eval_pope.py --split all --limit 300            # quick subset
  python scripts/eval_pope.py --split all                        # full E1 baseline
  python scripts/eval_pope.py --split dev_random --tag baseline   # dev questions (tuning)
  python scripts/eval_pope.py --split all --metrics-only         # recompute from saved JSON
  python scripts/eval_pope.py --split all --tag cautious \\
      --suffix " Answer yes only if the object is clearly visible, otherwise answer no."
"""
import _path  # noqa: F401
import argparse
import time

from tqdm import tqdm

from halluc import config
from halluc.data import load_image, load_pope, subsample
from halluc.io_utils import JsonlWriter, read_jsonl, run_metadata, save_json
from halluc.metrics import parse_yes_no, pope_metrics


def evaluate_split(vlm, split, args, out_dir):
    questions = subsample(load_pope(split), args.limit)
    out_path = out_dir / f"{split}.jsonl"
    done = {r["question_id"]: r for r in read_jsonl(out_path)}
    todo = [q for q in questions if q["question_id"] not in done]
    print(f"\n[{split}] {len(questions)} questions selected, {len(questions) - len(todo)} already done, "
          f"{len(todo)} to run")

    if todo and vlm is not None:
        cached_name, cached_img = None, None
        t0 = time.time()
        with JsonlWriter(out_path) as w:
            for q in tqdm(todo, desc=split):
                if q["image"] != cached_name:  # POPE asks 6 questions per image in a row
                    cached_name, cached_img = q["image"], load_image(q["image"])
                answer = vlm.generate(cached_img, q["text"] + args.suffix,
                                      max_new_tokens=config.POPE_MAX_NEW_TOKENS)
                pred, fallback = parse_yes_no(answer)
                rec = {**q, "prompt_suffix": args.suffix, "answer": answer, "pred": pred,
                       "parse_fallback": fallback}
                w.write(rec)
                done[q["question_id"]] = rec
        print(f"[{split}] {(time.time() - t0) / len(todo):.2f} s per question")

    selected = [done[q["question_id"]] for q in questions if q["question_id"] in done]
    m = pope_metrics(selected)
    m.update(split=split, tag=args.tag, limit=args.limit, complete=len(selected) == len(questions))
    save_json(m, out_dir / f"{split}_metrics.json")
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="all", help="random | popular | adversarial | dev_random | all")
    ap.add_argument("--limit", type=int, default=None, help="fixed random subset of N questions per split")
    ap.add_argument("--tag", default="baseline", help="name of this method/setting (results subfolder)")
    ap.add_argument("--suffix", default=config.POPE_SUFFIX, help="text appended to each question")
    ap.add_argument("--dtype", default=config.DTYPE)
    ap.add_argument("--metrics-only", action="store_true", help="do not load the model; recompute metrics")
    args = ap.parse_args()

    # Guard against mixing prompts inside one results folder.
    if args.tag == "baseline" and args.suffix != config.POPE_SUFFIX:
        raise SystemExit("Use a different --tag when changing --suffix, so results do not mix.")

    splits = list(config.POPE_SPLITS) if args.split == "all" else [args.split]
    out_dir = config.RESULTS_DIR / "pope" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    vlm = None
    if not args.metrics_only:
        from halluc.model import VLM
        vlm = VLM(dtype=args.dtype)
        save_json(run_metadata(args, model=config.MODEL_ID, max_pixels=config.MAX_PIXELS,
                               min_pixels=config.MIN_PIXELS, decoding="greedy"),
                  out_dir / "run_meta.json")

    rows = [evaluate_split(vlm, s, args, out_dir) for s in splits]

    print(f"\nPOPE results  (tag={args.tag}, limit={args.limit})")
    print(f"{'split':<12}{'n':>6}{'acc':>8}{'prec':>8}{'rec':>8}{'f1':>8}{'yes%':>8}{'fallbk':>8}")
    for m in rows:
        flag = "" if m["complete"] else "  (incomplete)"
        print(f"{m['split']:<12}{m['n']:>6}{m['accuracy']:>8.3f}{m['precision']:>8.3f}{m['recall']:>8.3f}"
              f"{m['f1']:>8.3f}{m['yes_ratio']:>8.3f}{m['n_parse_fallback']:>8}{flag}")


if __name__ == "__main__":
    main()
