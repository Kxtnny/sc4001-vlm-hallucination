"""E4: sweep alpha and layer ranges of the image-attention boost on the DEV set only.

For every (alpha, layers) setting it runs
  * POPE-style yes/no questions on the 200 dev images (results/pope/<tag>/dev_random.jsonl), and
  * captions on N_DEV_CAPTION_IMAGES dev images, scored with CHAIR (results/chair/<tag>/dev.jsonl),
plus the unmodified model (alpha = 0) as the reference. Everything resumes.

Selection rule (fixed before running, so the choice is not tuned by eye):
  among settings whose dev object recall drops by at most RECALL_TOL and whose dev POPE F1
  drops by at most F1_TOL compared with alpha = 0, pick the lowest dev CHAIR-i
  (ties: smaller alpha). If no setting satisfies the constraints, we fall back to the
  setting with the lowest dev CHAIR-i overall and record constraints_met = false, so the
  test run still measures the intervention (and the report states it failed the dev criteria).

Outputs: results/sweep/sweep_table.json, results/sweep/selected.json, figures/e4_sweep_dev.png

Usage:  python scripts/sweep.py                       # full grid from config.py (+ E3 layer band)
        python scripts/sweep.py --alphas 1 2 --layers 8-15 16-27
        python scripts/sweep.py --table-only          # rebuild table/selection from saved results
"""
import _path  # noqa: F401
import argparse

from caption_chair import load_chair_scorer, run_captions
from eval_pope import evaluate_split
from halluc import config
from halluc.io_utils import load_json, save_json

RECALL_TOL = 0.02   # absolute drop in dev object recall we accept
F1_TOL = 0.01       # absolute drop in dev POPE F1 we accept


def tag_for(alpha, layers):
    return "baseline" if alpha == 0 else f"sweep_a{alpha:g}_L{layers}"


class _Args:
    """Minimal stand-in for eval_pope's argparse namespace."""

    def __init__(self, tag, alpha, layers, limit=None):
        self.tag, self.alpha, self.layers = tag, alpha, layers
        self.limit, self.suffix = limit, config.POPE_SUFFIX
        self.batch_size = config.POPE_BATCH_SIZE


def run_setting(vlm, alpha, layers, n_caption_images, scorer, pope_limit=None):
    tag = tag_for(alpha, layers)
    out_dir = config.RESULTS_DIR / "pope" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    # The reference run also records the decision signals (used by E5 abstention).
    pope = evaluate_split(vlm, "dev_random", _Args(tag, alpha, layers, pope_limit), out_dir, signals=(alpha == 0))
    chair, _ = run_captions(vlm, "dev", tag, n_caption_images, alpha, layers, scorer=scorer)
    return {"tag": tag, "alpha": alpha, "layers": layers if alpha else "none",
            "pope_acc": pope["accuracy"], "pope_f1": pope["f1"], "pope_yes_ratio": pope["yes_ratio"],
            "pope_precision": pope["precision"], "pope_recall": pope["recall"],
            "chair_s": chair["chair_s"], "chair_i": chair["chair_i"], "recall": chair["recall"],
            "avg_len_words": chair["avg_len_words"], "complete": pope["complete"] and chair["complete"]}


def select(rows):
    base = next(r for r in rows if r["alpha"] == 0)
    cands = [r for r in rows if r["alpha"] != 0]
    ok = [r for r in cands
          if r["recall"] >= base["recall"] - RECALL_TOL and r["pope_f1"] >= base["pope_f1"] - F1_TOL
          and r["chair_i"] < base["chair_i"]]
    met = bool(ok)
    best = min(ok or cands, key=lambda r: (r["chair_i"], r["alpha"]))
    note = (f"lowest dev CHAIR-i with recall drop <= {RECALL_TOL} and POPE-F1 drop <= {F1_TOL}" if met else
            "no setting met the dev criteria; fallback = lowest dev CHAIR-i overall")
    return {"alpha": best["alpha"], "layers": best["layers"], "tag": best["tag"], "constraints_met": met,
            "note": note, "dev": {k: best[k] for k in ("chair_i", "chair_s", "recall", "pope_f1", "pope_acc")},
            "dev_baseline": {k: base[k] for k in ("chair_i", "chair_s", "recall", "pope_f1", "pope_acc")}}

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alphas", type=float, nargs="+", default=list(config.SWEEP_ALPHAS))
    ap.add_argument("--layers", nargs="+", default=list(config.SWEEP_LAYER_RANGES))
    ap.add_argument("--n-caption-images", type=int, default=config.N_DEV_CAPTION_IMAGES)
    ap.add_argument("--pope-limit", type=int, default=None, help="dev questions subset (smoke tests only)")
    ap.add_argument("--no-e3-band", action="store_true", help="do not add the layer band chosen in E3")
    ap.add_argument("--table-only", action="store_true", help="no model: rebuild table from saved results")
    ap.add_argument("--dtype", default=config.DTYPE)
    args = ap.parse_args()

    layer_ranges = list(args.layers)
    e3 = config.RESULTS_DIR / "analysis" / "e3_layer_choice.json"
    if not args.no_e3_band and e3.exists():
        band = load_json(e3)["layers"]
        if band not in layer_ranges:
            layer_ranges.append(band)
            print(f"Adding the E3 layer band {band} (chosen on dev captions)")
    grid = [(0.0, "none")] + [(a, lr) for lr in layer_ranges for a in args.alphas]

    vlm = None
    if not args.table_only:
        from halluc.model import VLM
        vlm = VLM(dtype=args.dtype)
    scorer = load_chair_scorer()
    rows = []
    for alpha, layers in grid:
        print(f"\n=== alpha={alpha} layers={layers} ===")
        rows.append(run_setting(vlm, alpha, layers, args.n_caption_images, scorer, args.pope_limit))

    sel = select(rows)
    out = config.RESULTS_DIR / "sweep"
    save_json({"rows": rows, "recall_tol": RECALL_TOL, "f1_tol": F1_TOL,
               "n_caption_images": args.n_caption_images, "split": "dev"}, out / "sweep_table.json")
    if all(r["complete"] for r in rows):
        save_json(sel, out / "selected.json")
    else:
        print("[warn] some settings are incomplete; selected.json not written")

    print(f"\n{'alpha':>6} {'layers':>7} {'POPE acc':>9} {'F1':>6} {'yes%':>6} {'CHAIR-s':>8} {'CHAIR-i':>8} "
          f"{'recall':>7} {'words':>6}")
    for r in rows:
        print(f"{r['alpha']:>6g} {r['layers']:>7} {r['pope_acc']:>9.3f} {r['pope_f1']:>6.3f} "
              f"{r['pope_yes_ratio']:>6.3f} {r['chair_s']:>8.3f} {r['chair_i']:>8.3f} {r['recall']:>7.3f} "
              f"{r['avg_len_words']:>6.1f}")
    print("\nSelected:", sel)

    from halluc.plots import plot_sweep
    base = rows[0]
    plot_sweep([r for r in rows if r["alpha"] != 0], base, config.FIGURE_DIR / "e4_sweep_dev.png")


if __name__ == "__main__":
    main()
