"""E3 (part 2): do hallucinated objects get less image attention? (H1)

Reads results/attention/<tag>/<split>.jsonl (from extract_attention.py; no GPU needed) and,
for each decoder layer, compares the image-attention share of hallucinated vs real
object mentions:
  * mean share for each group and the gap (real - hallucinated),
  * AUROC of "low image attention" as a hallucination detector (0.5 = chance),
    with a 95% bootstrap CI that resamples whole images (mentions in one caption
    are not independent),
  * the same AUROC within caption-position quintiles (position is a confounder:
    later tokens attend less to the image AND hallucinations happen later),
  * cross-validated logistic-regression AUROCs: position only, token log-prob only,
    image attention only, and combinations, to see whether attention adds
    information beyond position and the model's own confidence.

On the dev split it also picks the layer band used in the E4 sweep and in the
abstention signal (the contiguous band of --band-width layers with the highest mean
AUROC), and saves it to results/analysis/e3_layer_choice.json. That choice is made on
dev captions only.

Outputs: results/analysis/attention_<split>.json, figures/e3_attention_<split>.png

Usage:  python scripts/analyze_attention.py --split dev
        python scripts/analyze_attention.py --split chair_test
"""
import _path  # noqa: F401
import argparse

import numpy as np

from halluc import config
from halluc.io_utils import read_jsonl, save_json
from halluc.stats import auroc, bootstrap_auroc_ci, cv_logistic_auroc


def load_mentions(path):
    rows = []
    for cap in read_jsonl(path):
        for m in cap["mentions"]:
            rows.append({**m, "image": cap["image"]})
    if not rows:
        raise SystemExit(f"No mentions in {path}: run extract_attention.py first.")
    return rows


def best_band(aurocs, width):
    scores = [np.mean(aurocs[i:i + width]) for i in range(len(aurocs) - width + 1)]
    start = int(np.argmax(scores))
    return start, start + width - 1, float(scores[start])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="chair_test")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--signal", default="share_pred", choices=["share_pred", "share_tok"])
    ap.add_argument("--band-width", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from the saved analysis JSON")
    args = ap.parse_args()
    if args.plot_only:
        from halluc.io_utils import load_json
        from halluc.plots import plot_attention_by_layer
        saved = load_json(config.RESULTS_DIR / "analysis" / f"attention_{args.split}.json")
        plot_attention_by_layer(saved, args.signal, config.FIGURE_DIR / f"e3_attention_{args.split}.png")
        return

    rows = load_mentions(config.RESULTS_DIR / "attention" / args.tag / f"{args.split}.jsonl")
    y = np.array([r["hallucinated"] for r in rows], dtype=int)
    groups = np.array([r["image"] for r in rows])
    pos = np.array([r["rel_pos"] for r in rows])
    abs_pos = np.array([r["tok_index"] for r in rows], dtype=float)
    logprob = np.array([r["logprob"] for r in rows])
    n_layers = len(rows[0]["share_pred"])
    print(f"[{args.split}] {len(rows)} object mentions, {y.sum()} hallucinated ({y.mean():.1%}), "
          f"{len(set(groups))} captions, {n_layers} layers")
    if y.sum() < 5 or (1 - y).sum() < 5:
        raise SystemExit("Too few hallucinated or real mentions for an analysis.")

    quint = np.minimum((pos * 5).astype(int), 4)
    result = {"split": args.split, "n_mentions": int(len(y)), "n_hallucinated": int(y.sum()),
              "n_captions": int(len(set(groups))), "layers": {}}
    for sig in ("share_pred", "share_tok"):
        S = np.array([r[sig] for r in rows])                 # [mentions, layers]
        per_layer = []
        for li in range(n_layers):
            s = S[:, li]
            a = auroc(y, -s)                                  # low attention -> hallucinated
            lo, hi = bootstrap_auroc_ci(y, -s, groups, n_boot=args.n_boot, seed=li)
            within = [auroc(y[quint == q], -s[quint == q]) for q in range(5)
                      if 0 < y[quint == q].sum() < (quint == q).sum()]
            per_layer.append({
                "layer": li,
                "mean_real": float(s[y == 0].mean()), "mean_halluc": float(s[y == 1].mean()),
                "sem_real": float(s[y == 0].std(ddof=1) / np.sqrt((y == 0).sum())),
                "sem_halluc": float(s[y == 1].std(ddof=1) / np.sqrt((y == 1).sum())),
                "gap": float(s[y == 0].mean() - s[y == 1].mean()),
                "auroc": a, "auroc_ci": [lo, hi],
                "auroc_within_position_quintiles": float(np.mean(within)) if within else None,
            })
        result["layers"][sig] = per_layer

    # Which layers? (decided on dev; on test we only report)
    sig_aurocs = [p["auroc"] for p in result["layers"][args.signal]]
    lo_l, hi_l, band_auc = best_band(sig_aurocs, args.band_width)
    result["best_band"] = {"signal": args.signal, "layers": f"{lo_l}-{hi_l}", "mean_auroc": band_auc}

    # Does attention add information beyond position and confidence?
    S = np.array([r[args.signal] for r in rows])
    band_share = S[:, lo_l:hi_l + 1].mean(1)
    if args.split != "dev":   # use the band chosen on dev, if available
        choice = config.RESULTS_DIR / "analysis" / "e3_layer_choice.json"
        if choice.exists():
            from halluc.io_utils import load_json
            lo_d, hi_d = map(int, load_json(choice)["layers"].split("-"))
            band_share = S[:, lo_d:hi_d + 1].mean(1)
            result["band_used_for_detectors"] = f"{lo_d}-{hi_d} (chosen on dev)"
    feats = {
        "position": np.c_[pos, abs_pos],
        "logprob": np.c_[logprob],
        "attention": np.c_[band_share],
        "position+logprob": np.c_[pos, abs_pos, logprob],
        "position+logprob+attention": np.c_[pos, abs_pos, logprob, band_share],
    }
    result["detectors_cv_auroc"] = {k: cv_logistic_auroc(X, y, groups) for k, X in feats.items()}
    result["single_feature_auroc"] = {"attention_band (low=halluc)": auroc(y, -band_share),
                                      "logprob (low=halluc)": auroc(y, -logprob),
                                      "rel_position (late=halluc)": auroc(y, pos)}

    out = config.RESULTS_DIR / "analysis" / f"attention_{args.split}.json"
    save_json(result, out)
    if args.split == "dev":
        save_json({"layers": f"{lo_l}-{hi_l}", "signal": args.signal, "mean_auroc_dev": band_auc,
                   "note": "chosen on dev captions by scripts/analyze_attention.py"},
                  config.RESULTS_DIR / "analysis" / "e3_layer_choice.json")

    print(f"\nlayer  real   halluc  gap     AUROC [95% CI]       within-position   ({args.signal})")
    for p in result["layers"][args.signal]:
        w = p["auroc_within_position_quintiles"]
        print(f"{p['layer']:>5}  {p['mean_real']:.3f}  {p['mean_halluc']:.3f}  {p['gap']:+.3f}  "
              f"{p['auroc']:.3f} [{p['auroc_ci'][0]:.3f}, {p['auroc_ci'][1]:.3f}]  "
              f"{w if w is None else round(w, 3)}")
    print(f"\nBest {args.band_width}-layer band: {result['best_band']}")
    print("Single-feature AUROC:", {k: round(v, 3) for k, v in result["single_feature_auroc"].items()})
    print("Cross-validated detector AUROC:", {k: round(v, 3) for k, v in result["detectors_cv_auroc"].items()})
    print("Saved", out)

    from halluc.plots import plot_attention_by_layer
    fig_path = config.FIGURE_DIR / f"e3_attention_{args.split}.png"
    plot_attention_by_layer(result, args.signal, fig_path)
    print("Figure", fig_path)


if __name__ == "__main__":
    main()
