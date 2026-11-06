"""E0: sanity checks. Run this before any real experiment, and again after
changing model-loading or attention code.

Checks (each prints PASS / FAIL; the script exits with an error if any fail):
  1. The model answers 10 POPE questions; we print them so you can read them.
  2. A question about an object that IS in the image gets "yes".
  3. Image tokens: count matches the image grid, they are contiguous and
     wrapped by <|vision_start|>/<|vision_end|>, and stay within MAX_PIXELS.
  4. Attention: one tensor per decoder layer, no NaN/inf, every row sums to 1.
  5. Determinism: asking the same question twice gives the same answer.
It also times generation, so you can estimate how long full runs will take.

Usage:  python scripts/sanity_check.py [--n 10] [--dtype float16]
"""
import _path  # noqa: F401
import argparse
import time

import torch

from halluc import config
from halluc.data import load_image, load_pope
from halluc.io_utils import run_metadata, save_json
from halluc.metrics import parse_yes_no
from halluc.model import VLM, check_image_tokens, image_attention_share, image_token_mask

results = {}


def report(name, ok, detail=""):
    results[name] = {"pass": bool(ok), "detail": str(detail)}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="number of POPE questions to print")
    ap.add_argument("--dtype", default=config.DTYPE)
    args = ap.parse_args()

    t0 = time.time()
    vlm = VLM(dtype=args.dtype)
    print(f"Model loaded in {time.time() - t0:.0f}s on {vlm.device}; "
          f"{vlm.num_layers} decoder layers; GPU memory "
          f"{torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0:.1f} GB")

    # ---- 1 + 2: read some answers -----------------------------------------
    rows = load_pope("random")[: args.n]
    print(f"\n{'image':<32} {'question':<42} {'answer':<10} {'parsed':<6} label")
    times, correct = [], 0
    for r in rows:
        img = load_image(r["image"])
        t = time.time()
        ans = vlm.generate(img, r["text"] + config.POPE_SUFFIX)
        times.append(time.time() - t)
        pred, _ = parse_yes_no(ans)
        correct += pred == r["label"]
        print(f"{r['image']:<32} {r['text'][:40]:<42} {ans[:10]!r:<10} {pred:<6} {r['label']}")
    sec = sum(times[1:]) / max(len(times) - 1, 1)  # skip the first (warm-up) call
    report("answers look like yes/no", correct >= len(rows) * 0.6, f"{correct}/{len(rows)} correct")

    present = next(r for r in rows if r["label"] == "yes")
    ans = vlm.generate(load_image(present["image"]), present["text"] + config.POPE_SUFFIX)
    report("present object gets 'yes'", parse_yes_no(ans)[0] == "yes", f"{present['text']} -> {ans!r}")

    # ---- 5: determinism -----------------------------------------------------
    ans2 = vlm.generate(load_image(present["image"]), present["text"] + config.POPE_SUFFIX)
    report("greedy decoding is deterministic", ans == ans2, f"{ans!r} vs {ans2!r}")

    # ---- 3: image tokens -----------------------------------------------------
    img = load_image(present["image"])
    inputs = vlm.build_inputs(img, present["text"] + config.POPE_SUFFIX)
    try:
        pos = check_image_tokens(inputs["input_ids"], inputs["image_grid_thw"], vlm.cfg, vlm.merge_size)
        n_img = len(pos)
        report("image-token positions", n_img <= vlm.max_image_tokens,
               f"{n_img} image tokens at positions {int(pos[0])}..{int(pos[-1])} "
               f"of {inputs['input_ids'].shape[1]} (max allowed {vlm.max_image_tokens}); "
               f"image {img.size} -> grid {inputs['image_grid_thw'].tolist()}")
    except AssertionError as e:
        report("image-token positions", False, e)

    # ---- 4: attention ---------------------------------------------------------
    out = vlm.forward_attentions(inputs)
    att = out.attentions
    report("one attention map per layer", att is not None and len(att) == vlm.num_layers,
           f"{len(att) if att else 0} maps, shape {tuple(att[0].shape) if att else None}")
    finite = all(torch.isfinite(a).all().item() for a in att)
    report("attention has no NaN/inf", finite)
    worst = max((a.float().sum(-1) - 1).abs().max().item() for a in att)
    report("attention rows sum to 1", worst < 1e-2, f"max |row sum - 1| = {worst:.2e}")
    logits_ok = torch.isfinite(out.logits).all().item()
    report("logits have no NaN/inf (float16 overflow check)", logits_ok)

    # Preview of E3: how much the last prompt token looks at the image, per layer.
    mask = image_token_mask(inputs["input_ids"], vlm.cfg.image_token_id)
    share = image_attention_share(att, mask)[:, -1].cpu()
    print("\nImage-attention share of the last prompt token, by layer:")
    print("  " + " ".join(f"L{i}:{s:.2f}" for i, s in enumerate(share.tolist())))

    print(f"\nTiming: {sec:.2f} s per POPE question "
          f"-> about {sec * 3000 / 60:.0f} min per POPE split (3,000 questions)")
    meta = run_metadata(args, seconds_per_pope_question=sec, checks=results,
                        image_attention_share_last_token=share.tolist())
    save_json(meta, config.RESULTS_DIR / "sanity" / "sanity_report.json")
    print("Saved report to", config.RESULTS_DIR / "sanity" / "sanity_report.json")

    failed = [k for k, v in results.items() if not v["pass"]]
    if failed:
        raise SystemExit(f"\n{len(failed)} check(s) FAILED: {failed}")
    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    main()
