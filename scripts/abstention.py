"""E5 (part 2): let the model abstain when its image-attention signal is low.

POPE (yes/no). For each question, eval_pope.py --signals saved the share of
attention on image tokens at the decision token (per layer) and P(yes), P(no).
We compare three abstention scores (higher = answer, lower = abstain):
  image attention   mean image-attention share over the layer band chosen on dev
  answer confidence |P(yes) - P(no)| after renormalising the two (the standard baseline)
  random            a fixed random order (what abstention looks like with no signal)

The layer band is the one with the best dev AUROC for predicting a wrong answer; the
threshold is set on dev to keep --coverage of the questions, then applied unchanged
to each test split. We report accuracy vs coverage curves and the accuracy at the
dev-chosen threshold.

Captions (CHAIR). As a second use of the E3 signal, object mentions whose image
attention (same band as E3 chose on dev) falls below a dev-chosen threshold are
flagged as unsupported, and we report CHAIR-i and recall after removing them.

Outputs: results/analysis/abstention.json, figures/e5_accuracy_coverage_<split>.png

Usage:  python scripts/abstention.py [--coverage 0.8]
"""
import _path  # noqa: F401
import argparse

import numpy as np

from halluc import config
from halluc.io_utils import load_json, read_jsonl, save_json
from halluc.stats import (accuracy_at_threshold, area_under_accuracy_coverage, auroc, risk_coverage,
                          threshold_for_coverage)

BAND_WIDTH = 8


def pope_arrays(rows):
    correct = np.array([r["pred"] == r["label"] for r in rows], dtype=float)
    share = np.array([r["img_share_last"] for r in rows])          # [n, layers]
    py, pn = np.array([r["p_yes"] for r in rows]), np.array([r["p_no"] for r in rows])
    conf = np.abs(py - pn) / np.maximum(py + pn, 1e-9)
    return correct, share, conf


def choose_band(correct, share, width=BAND_WIDTH):
    """Contiguous layer band whose mean share best separates right from wrong answers (dev only)."""
    aucs = [auroc(1 - correct, -share[:, i]) for i in range(share.shape[1])]   # low share -> wrong
    scores = [np.nanmean(aucs[i:i + width]) for i in range(len(aucs) - width + 1)]
    lo = int(np.nanargmax(scores))
    return lo, lo + width - 1, aucs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coverage", type=float, default=0.8, help="target coverage, threshold set on dev")
    ap.add_argument("--tag", default="baseline")
    args = ap.parse_args()
    pope_dir = config.RESULTS_DIR / "pope" / args.tag

    dev = [r for r in read_jsonl(pope_dir / "dev_random.jsonl") if "img_share_last" in r]
    if not dev:
        raise SystemExit("No dev signals: run eval_pope.py --split dev_random --signals (or sweep.py) first.")
    c_dev, s_dev, conf_dev = pope_arrays(dev)
    lo, hi, dev_aucs = choose_band(c_dev, s_dev)
    att_dev = s_dev[:, lo:hi + 1].mean(1)
    thr = {"image attention": threshold_for_coverage(att_dev, args.coverage),
           "answer confidence": threshold_for_coverage(conf_dev, args.coverage)}
    # Exploratory (not part of the pre-registered rule): the single layer where *high* image attention
    # best flags wrong answers on dev. Its direction was found by looking at dev, so it is reported
    # separately and never used for the frozen abstention threshold.
    rev = int(np.nanargmin(dev_aucs))
    result = {"coverage_target": args.coverage, "band": f"{lo}-{hi}", "dev_auroc_per_layer": dev_aucs,
              "thresholds": thr, "exploratory_high_attention_layer": rev, "pope": {}}
    print(f"POPE abstention band (dev): layers {lo}-{hi}; dev AUROC(wrong) of band mean = "
          f"{auroc(1 - c_dev, -att_dev):.3f}; answer-confidence AUROC = {auroc(1 - c_dev, -conf_dev):.3f}")

    from halluc.plots import plot_accuracy_coverage
    for split in ("dev_random",) + config.POPE_SPLITS:
        rows = [r for r in read_jsonl(pope_dir / f"{split}.jsonl") if "img_share_last" in r]
        if not rows:
            print(f"[{split}] no signals saved, skipping")
            continue
        c, s, conf = pope_arrays(rows)
        att = s[:, lo:hi + 1].mean(1)
        rnd = np.random.default_rng(0).random(len(c))
        scores = {"image attention": att, "answer confidence": conf, "random": rnd}
        res = {"n": len(c), "accuracy_full": float(c.mean())}
        curves = []
        for name, sc in scores.items():
            cov, acc = risk_coverage(c, sc)
            curves.append((name, cov, acc))
            entry = {"auroc_wrong": auroc(1 - c, -sc), "area_under_acc_cov": area_under_accuracy_coverage(c, sc)}
            if name in thr:
                entry["coverage_at_dev_threshold"], entry["accuracy_at_dev_threshold"] = \
                    accuracy_at_threshold(c, sc, thr[name])
            res[name] = entry
        pred_no = np.array([r["pred"] == "no" for r in rows])
        res["exploratory_high_attention"] = {
            "layer": rev, "auroc_wrong": auroc(1 - c, s[:, rev]),
            "auroc_wrong_within_pred_no": auroc(1 - c[pred_no], s[pred_no, rev]),
            "area_under_acc_cov": area_under_accuracy_coverage(c, -s[:, rev]),
            "n_wrong": int((1 - c).sum()), "n_false_negative": int(((1 - c) * pred_no).sum())}
        curves.append((f"high attention, layer {rev} (expl.)",) + tuple(risk_coverage(c, -s[:, rev])))
        result["pope"][split] = res
        print(f"[{split}] n={len(c)} acc={c.mean():.3f} | " + " | ".join(
            f"{k}: AUROC {v['auroc_wrong']:.3f}, AUACC {v['area_under_acc_cov']:.3f}"
            + (f", cov {v['coverage_at_dev_threshold']:.2f} acc {v['accuracy_at_dev_threshold']:.3f}"
               if "coverage_at_dev_threshold" in v else "")
            for k, v in res.items() if isinstance(v, dict)))
        plot_accuracy_coverage(curves, config.FIGURE_DIR / f"e5_accuracy_coverage_{split}.png",
                               title=f"POPE {split.replace('_random', '')}: accuracy vs coverage")

    # ---- captions: drop low-attention object mentions -----------------------------------------
    att_dir = config.RESULTS_DIR / "attention" / "baseline"
    choice = config.RESULTS_DIR / "analysis" / "e3_layer_choice.json"
    if choice.exists() and (att_dir / "dev.jsonl").exists() and (att_dir / "chair_test.jsonl").exists():
        sel = load_json(choice)
        a, b = map(int, sel["layers"].split("-"))
        sig = sel["signal"]

        def mentions(split):
            caps = read_jsonl(att_dir / f"{split}.jsonl")
            return caps, [m for cp in caps for m in cp["mentions"]]

        _, dev_m = mentions("dev")
        dev_scores = np.array([np.mean(m[sig][a:b + 1]) for m in dev_m])
        cap_res = {"band": sel["layers"], "signal": sig}
        for keep in (1.0, 0.9, 0.8):
            t = threshold_for_coverage(dev_scores, keep) if keep < 1 else -np.inf
            caps, _ = mentions("chair_test")
            gt_path = config.RESULTS_DIR / "chair" / "baseline" / "chair_test_scored.jsonl"
            gts = {r["image"]: set(r["gt_nodes"]) for r in read_jsonl(gt_path)}
            n_mention = n_hall = covered = gt_total = n_with_hall = 0
            caps = [cp for cp in caps if cp["image"] in gts]
            for cp in caps:
                kept = [m for m in cp["mentions"] if np.mean(m[sig][a:b + 1]) >= t]
                gt = gts[cp["image"]]
                n_mention += len(kept)
                n_hall += sum(m["hallucinated"] for m in kept)
                n_with_hall += any(m["hallucinated"] for m in kept)
                covered += len({m["node"] for m in kept if not m["hallucinated"]})
                gt_total += len(gt)
            cap_res[f"keep_{keep:.1f}"] = {"threshold": float(t) if np.isfinite(t) else None, "chair_i": n_hall / max(1, n_mention),
                                           "chair_s": n_with_hall / len(caps), "recall": covered / gt_total,
                                           "mentions_kept": n_mention}
        result["captions_filter"] = cap_res
        print("Caption filter (chair_test, thresholds from dev):",
              {k: {kk: (None if vv is None else round(vv, 3)) for kk, vv in v.items()} for k, v in cap_res.items() if isinstance(v, dict)})

    save_json(result, config.RESULTS_DIR / "analysis" / "abstention.json")
    print("Saved", config.RESULTS_DIR / "analysis" / "abstention.json")


if __name__ == "__main__":
    main()
