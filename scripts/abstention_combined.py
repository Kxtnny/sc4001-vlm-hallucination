"""E5 follow-up: does image attention add anything to answer confidence for abstention?

E5 found that *high* image attention in one late layer flags wrong POPE answers (the model's
typical error is a miss made while it attends to the image). Here we ask whether that signal is
useful on top of the standard baseline, answer confidence.

Protocol (no model run, only the signals saved by eval_pope.py --signals):
  * the layer is the one abstention.py reports as "exploratory_high_attention_layer": the layer
    whose image-attention share is highest for wrong answers on DEV (argmin of the dev AUROC of
    "low share -> wrong");
  * three logistic regressions are fitted on the DEV questions only, predicting "answer is wrong":
        confidence                 |P(yes) - P(no)| (one feature; same ranking as E5's baseline)
        confidence + attention     adds the share in that layer
        confidence + all layers    adds the share in every layer (upper bound for a linear use)
  * each fitted model is applied unchanged to the three POPE test splits. We report the AUROC for
    wrong answers, the area under the accuracy-coverage curve, and a 95% CI for the AUROC gain
    over confidence alone from a bootstrap that resamples whole images (2,000 resamples, seed 0).

Output: results/analysis/abstention_combined.json

Usage:  python scripts/abstention_combined.py [--tag baseline]
"""
import _path  # noqa: F401
import argparse

import numpy as np

from halluc import config
from halluc.io_utils import read_jsonl, save_json
from halluc.stats import area_under_accuracy_coverage, auroc

N_BOOT = 2000


def arrays(rows):
    correct = np.array([r["pred"] == r["label"] for r in rows], dtype=float)
    share = np.array([r["img_share_last"] for r in rows])
    py, pn = np.array([r["p_yes"] for r in rows]), np.array([r["p_no"] for r in rows])
    conf = np.abs(py - pn) / np.maximum(py + pn, 1e-9)
    images = np.array([r["image"] for r in rows])
    return correct, share, conf, images


def fit(X, wrong):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X, wrong)


def auroc_gain_ci(wrong, score_new, score_base, images, n_boot=N_BOOT, seed=0):
    """Percentile 95% CI of AUROC(new) - AUROC(base), resampling whole images."""
    uniq = np.unique(images)
    idx_by_image = [np.flatnonzero(images == g) for g in uniq]
    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by_image[i] for i in rng.integers(0, len(uniq), len(uniq))])
        gains.append(auroc(wrong[idx], score_new[idx]) - auroc(wrong[idx], score_base[idx]))
    lo, hi = np.nanpercentile(gains, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="baseline")
    args = ap.parse_args()
    pope_dir = config.RESULTS_DIR / "pope" / args.tag

    dev = [r for r in read_jsonl(pope_dir / "dev_random.jsonl") if "img_share_last" in r]
    if not dev:
        raise SystemExit("No dev signals: run eval_pope.py --split dev_random --signals first.")
    c_dev, s_dev, conf_dev, _ = arrays(dev)
    dev_aucs = [auroc(1 - c_dev, -s_dev[:, i]) for i in range(s_dev.shape[1])]
    layer = int(np.nanargmin(dev_aucs))             # same rule as abstention.py (chosen on dev)
    features = {
        "confidence": lambda s, conf: conf[:, None],
        "confidence + attention": lambda s, conf: np.column_stack([conf, s[:, layer]]),
        "confidence + all layers": lambda s, conf: np.column_stack([conf, s]),
    }
    models = {name: fit(f(s_dev, conf_dev), 1 - c_dev) for name, f in features.items()}
    coef = models["confidence + attention"][-1].coef_[0]
    result = {"layer": layer, "n_dev": len(dev), "n_dev_wrong": int((1 - c_dev).sum()),
              "standardised_coefficients_confidence_attention": [float(coef[0]), float(coef[1])], "pope": {}}
    print(f"layer chosen on dev: {layer}; dev questions {len(dev)} ({int((1 - c_dev).sum())} wrong); "
          f"standardised coefficients (confidence, attention) = {coef[0]:+.2f}, {coef[1]:+.2f}")

    for split in ("dev_random",) + config.POPE_SPLITS:
        rows = [r for r in read_jsonl(pope_dir / f"{split}.jsonl") if "img_share_last" in r]
        if not rows:
            print(f"[{split}] no signals saved, skipping")
            continue
        c, s, conf, images = arrays(rows)
        wrong = 1 - c
        scores = {name: models[name].predict_proba(f(s, conf))[:, 1] for name, f in features.items()}
        res = {"n": len(c), "n_wrong": int(wrong.sum())}
        for name, p_wrong in scores.items():
            entry = {"auroc_wrong": auroc(wrong, p_wrong),
                     "area_under_acc_cov": area_under_accuracy_coverage(c, -p_wrong)}
            if name != "confidence":
                entry["auroc_gain_over_confidence"] = entry["auroc_wrong"] - auroc(wrong, scores["confidence"])
                entry["auroc_gain_ci"] = auroc_gain_ci(wrong, p_wrong, scores["confidence"], images)
            res[name] = entry
        result["pope"][split] = res
        print(f"[{split}] n={len(c)} wrong={int(wrong.sum())} | " + " | ".join(
            f"{k}: AUROC {v['auroc_wrong']:.3f}, AUACC {100 * v['area_under_acc_cov']:.1f}"
            + (f", gain {v['auroc_gain_over_confidence']:+.3f} [{v['auroc_gain_ci'][0]:+.3f}, {v['auroc_gain_ci'][1]:+.3f}]"
               if "auroc_gain_ci" in v else "")
            for k, v in res.items() if isinstance(v, dict)))
    save_json(result, config.RESULTS_DIR / "analysis" / "abstention_combined.json")
    print("saved", config.RESULTS_DIR / "analysis" / "abstention_combined.json")


if __name__ == "__main__":
    main()
