"""E0: sanity checks. Run this before any real experiment, and again after
changing model-loading or attention code.

Checks (each prints PASS / FAIL; the script exits with an error if any fail):
  1. The model answers 10 POPE questions; we print them so you can read them.
  2. A question about an object that IS in the image gets "yes".
  3. Image tokens: count matches the image grid, they are contiguous and
     wrapped by <|vision_start|>/<|vision_end|>, and stay within MAX_PIXELS.
  4. Attention: one tensor per decoder layer, no NaN/inf, every row sums to 1.
  5. Determinism: asking the same question twice gives the same answer.

It then runs a speed benchmark on fixed inputs (the first --timing-questions
POPE "random" questions and the first --timing-captions POPE images), so the
numbers from different machines (Colab T4, TC1 V100) can be compared directly
with scripts/compare_timing.py. Image loading from disk/Drive is timed
separately from the GPU work, and a warm-up call is excluded.

The report is saved as results/sanity/sanity_report_<tag>.json, where <tag>
defaults to the GPU name (e.g. tesla-t4), so reports from different machines
do not overwrite each other when copied into the same Drive folder.

Usage:  python scripts/sanity_check.py [--n 10] [--dtype float16]
                                       [--timing-questions 30] [--timing-captions 3] [--tag NAME]
"""
import _path  # noqa: F401
import argparse
import os
import platform
import re
import statistics
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


def timed(fn):
    """Run fn() and return (result, seconds). Synchronises CUDA so GPU work is fully counted."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, time.perf_counter() - t


def host_label():
    """Where this ran: 'colab', 'slurm:<node>' (TC1) or the machine name."""
    if "COLAB_RELEASE_TAG" in os.environ or os.path.exists("/content"):
        return "colab"
    if "SLURM_JOB_ID" in os.environ:
        return f"slurm:{os.environ.get('SLURMD_NODENAME', platform.node())}"
    return platform.node()


def benchmark(vlm, n_questions, n_captions):
    """Speed on fixed inputs, so every machine times exactly the same work.

    POPE: short answers (<= POPE_MAX_NEW_TOKENS tokens), dominated by the prompt pass.
    Captions: long answers (<= CAPTION_MAX_NEW_TOKENS tokens), dominated by decoding.
    The two behave differently across GPUs, and CHAIR / the E4 sweep need captions,
    so both are measured.
    """
    pope_rows = load_pope("random")
    questions = pope_rows[:n_questions]
    caption_images = list(dict.fromkeys(r["image"] for r in pope_rows))[:n_captions]

    images, load_s = {}, []
    for name in dict.fromkeys([q["image"] for q in questions] + caption_images):
        images[name], s = timed(lambda: load_image(name))
        load_s.append(s)

    # Warm-up (CUDA kernels, caches); not counted.
    vlm.generate(images[questions[0]["image"]], questions[0]["text"] + config.POPE_SUFFIX)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    pope_s = [timed(lambda q=q: vlm.generate(images[q["image"]], q["text"] + config.POPE_SUFFIX))[1]
              for q in questions]

    cap_s, cap_tokens = [], []
    for name in caption_images:
        (_, inputs, out), s = timed(lambda: vlm.generate(
            images[name], config.CAPTION_PROMPT,
            max_new_tokens=config.CAPTION_MAX_NEW_TOKENS, return_inputs=True))
        cap_s.append(s)
        cap_tokens.append(out.shape[1] - inputs["input_ids"].shape[1])

    pope_mean = statistics.mean(pope_s)
    cap_mean = statistics.mean(cap_s) if cap_s else float("nan")
    return {
        "n_pope_questions": len(pope_s),
        "pope_s_per_question_mean": pope_mean,
        "pope_s_per_question_median": statistics.median(pope_s),
        "n_images_loaded": len(load_s),
        "image_load_s_mean": statistics.mean(load_s),
        "n_captions": len(cap_s),
        "caption_s_mean": cap_mean,
        "caption_new_tokens_mean": statistics.mean(cap_tokens) if cap_tokens else float("nan"),
        "caption_tokens_per_s": sum(cap_tokens) / sum(cap_s) if cap_s else float("nan"),
        "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
        # Projections for planning (GPU time only; add image loading on top).
        "projected_pope_9000_questions_min": pope_mean * 9000 / 60,
        "projected_chair_500_captions_min": cap_mean * 500 / 60,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="number of POPE questions to print")
    ap.add_argument("--dtype", default=config.DTYPE)
    ap.add_argument("--timing-questions", type=int, default=30, help="POPE questions in the speed benchmark")
    ap.add_argument("--timing-captions", type=int, default=3, help="captions in the speed benchmark")
    ap.add_argument("--tag", default=None, help="report name suffix (default: GPU name, e.g. tesla-t4)")
    args = ap.parse_args()

    vlm, load_model_s = timed(lambda: VLM(dtype=args.dtype))
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    tag = args.tag or re.sub(r"[^a-z0-9]+", "-", gpu.lower()).strip("-")
    print(f"Running on {gpu} ({host_label()}), report tag '{tag}'")
    print(f"Model loaded in {load_model_s:.0f}s on {vlm.device}; "
          f"{vlm.num_layers} decoder layers; GPU memory "
          f"{torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0:.1f} GB")

    # ---- 1 + 2: read some answers -----------------------------------------
    rows = load_pope("random")[: args.n]
    print(f"\n{'image':<32} {'question':<42} {'answer':<10} {'parsed':<6} label")
    correct = 0
    for r in rows:
        ans = vlm.generate(load_image(r["image"]), r["text"] + config.POPE_SUFFIX)
        pred, _ = parse_yes_no(ans)
        correct += pred == r["label"]
        print(f"{r['image']:<32} {r['text'][:40]:<42} {ans[:10]!r:<10} {pred:<6} {r['label']}")
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

    del out, att  # free the attention maps before timing

    # ---- speed benchmark (same inputs on every machine) -----------------------
    print(f"\nSpeed benchmark: {args.timing_questions} POPE questions + {args.timing_captions} captions ...")
    timing = benchmark(vlm, args.timing_questions, args.timing_captions)
    timing.update(model_load_s=load_model_s, gpu=gpu, host=host_label(), dtype=args.dtype,
                  cpus=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 0)))
    print(f"  image load      {timing['image_load_s_mean']:.3f} s per image (disk / Drive)")
    print(f"  POPE question   {timing['pope_s_per_question_mean']:.3f} s mean, "
          f"{timing['pope_s_per_question_median']:.3f} s median "
          f"-> ~{timing['projected_pope_9000_questions_min']:.0f} min for all 9,000 POPE questions")
    print(f"  caption         {timing['caption_s_mean']:.1f} s mean, {timing['caption_new_tokens_mean']:.0f} tokens, "
          f"{timing['caption_tokens_per_s']:.1f} tokens/s "
          f"-> ~{timing['projected_chair_500_captions_min']:.0f} min for 500 CHAIR captions")
    print(f"  peak GPU memory {timing['peak_gpu_mem_gb']:.1f} GB during generation")

    meta = run_metadata(args, seconds_per_pope_question=timing["pope_s_per_question_mean"],
                        timing=timing, checks=results,
                        image_attention_share_last_token=share.tolist())
    report_path = config.RESULTS_DIR / "sanity" / f"sanity_report_{tag}.json"
    save_json(meta, report_path)
    print("Saved report to", report_path)

    failed = [k for k, v in results.items() if not v["pass"]]
    if failed:
        raise SystemExit(f"\n{len(failed)} check(s) FAILED: {failed}")
    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    main()
