"""Build every report table (and the summary figure) from saved JSON. No model, no GPU.

Reads results/{pope,chair,sweep,analysis}/... and writes
  results/report/tables.md     all tables in Markdown (paste into the report)
  results/report/numbers.json  every number the report quotes, in one place
  figures/e4_test_tradeoff.png CHAIR-i vs recall on the test set, baseline vs intervention

It also runs paired significance tests, which need the per-example outputs:
  POPE   exact McNemar test on the questions both methods answered (baseline vs method)
  CHAIR  image-level paired bootstrap (2,000 resamples) of the CHAIR-s / CHAIR-i / recall difference

Usage:  python scripts/make_report.py
"""
import _path  # noqa: F401
import math
import re
from pathlib import Path

import numpy as np

from halluc import config
from halluc.io_utils import load_json, read_jsonl, save_json

R = config.RESULTS_DIR
OUT = R / "report"
METHODS_POPE = [("baseline", "Baseline"), ("cautious", "Cautious prompt"), ("intervention", "Image-attention boost")]
METHODS_CHAIR = [("baseline", "Baseline"), ("intervention", "Image-attention boost")]


def maybe(path):
    path = Path(path)
    return load_json(path) if path.exists() else None


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value; b, c = discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired_bootstrap_chair(base, other, n_boot=2000, seed=0):
    """95% CIs for (other - base) of CHAIR-s, CHAIR-i and recall, resampling images."""
    imgs = sorted(set(base) & set(other))
    def arr(d, key):
        return np.array([key(d[i]) for i in imgs], dtype=float)
    hb, ho = arr(base, lambda s: s["n_hallucinated"]), arr(other, lambda s: s["n_hallucinated"])
    mb, mo = arr(base, lambda s: s["n_mentions"]), arr(other, lambda s: s["n_mentions"])
    cb, co = arr(base, lambda s: len(s["correct_nodes"])), arr(other, lambda s: len(s["correct_nodes"]))
    g = arr(base, lambda s: len(s["gt_nodes"]))

    def stats(idx):
        return np.array([
            (ho[idx] > 0).mean() - (hb[idx] > 0).mean(),
            ho[idx].sum() / max(1, mo[idx].sum()) - hb[idx].sum() / max(1, mb[idx].sum()),
            co[idx].sum() / g[idx].sum() - cb[idx].sum() / g[idx].sum(),
        ])
    point = stats(np.arange(len(imgs)))
    rng = np.random.default_rng(seed)
    boots = np.array([stats(rng.integers(0, len(imgs), len(imgs))) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    names = ["chair_s", "chair_i", "recall"]
    return {n: {"diff": float(point[k]), "ci": [float(lo[k]), float(hi[k])]} for k, n in enumerate(names)}, len(imgs)


def truncate_scored(rec, n_words):
    """The scored caption cut after its first n_words words (only mentions inside the cut are kept)."""
    words = list(re.finditer(r"\S+", rec["caption"]))
    cut = words[n_words - 1].end() if 0 < n_words <= len(words) else (0 if n_words <= 0 else len(rec["caption"]))
    kept = [m for m in rec["mentions"] if m["end"] <= cut]
    return {"n_hallucinated": sum(m["hallucinated"] for m in kept), "n_mentions": len(kept),
            "correct_nodes": sorted({m["node"] for m in kept if not m["hallucinated"]}), "gt_nodes": rec["gt_nodes"]}


def length_matched(base, other):
    """Control for caption length: for each image cut both captions to the shorter one's word count."""
    b, o = {}, {}
    for img in set(base) & set(other):
        n = min(len(base[img]["caption"].split()), len(other[img]["caption"].split()))
        b[img], o[img] = truncate_scored(base[img], n), truncate_scored(other[img], n)

    def metrics(d):
        h = sum(r["n_hallucinated"] for r in d.values())
        m = sum(r["n_mentions"] for r in d.values())
        return {"chair_s": float(np.mean([r["n_hallucinated"] > 0 for r in d.values()])), "chair_i": h / max(1, m),
                "recall": sum(len(r["correct_nodes"]) for r in d.values()) / sum(len(r["gt_nodes"]) for r in d.values()),
                "avg_mentions": m / len(d)}
    diff, n = paired_bootstrap_chair(b, o)
    return {"baseline": metrics(b), "intervention": metrics(o), "intervention_minus_baseline": diff, "n_images": n}


def fmt(x, d=3):
    return "-" if x is None else f"{x:.{d}f}"


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    md, numbers = [], {}

    # ---------------- E1 / E4 / E5: POPE ----------------
    md += ["## Table: POPE (test, 3 x 3,000 questions)", "",
           "| Method | Split | Acc | Prec | Rec | F1 | Yes % | McNemar p vs baseline |",
           "|---|---|---|---|---|---|---|---|"]
    numbers["pope"] = {}
    for split in config.POPE_SPLITS:
        base_recs = {r["question_id"]: r for r in read_jsonl(R / "pope" / "baseline" / f"{split}.jsonl")}
        for tag, name in METHODS_POPE:
            m = maybe(R / "pope" / tag / f"{split}_metrics.json")
            if not m:
                continue
            p = None
            if tag != "baseline":
                recs = {r["question_id"]: r for r in read_jsonl(R / "pope" / tag / f"{split}.jsonl")}
                both = set(recs) & set(base_recs)
                b = sum(base_recs[q]["pred"] == base_recs[q]["label"] and recs[q]["pred"] != recs[q]["label"]
                        for q in both)
                c = sum(base_recs[q]["pred"] != base_recs[q]["label"] and recs[q]["pred"] == recs[q]["label"]
                        for q in both)
                p = mcnemar_exact(b, c)
                m = {**m, "mcnemar_p": p, "baseline_right_method_wrong": b, "baseline_wrong_method_right": c}
            numbers["pope"].setdefault(split, {})[tag] = m
            flag = "" if m.get("complete", True) else " (incomplete)"
            md.append(f"| {name}{flag} | {split} | {pct(m['accuracy'])} | {pct(m['precision'])} | "
                      f"{pct(m['recall'])} | {pct(m['f1'])} | {pct(m['yes_ratio'])} | "
                      f"{'-' if p is None else f'{p:.3g}'} |")
    md.append("")

    # ---------------- E2 / E4: CHAIR ----------------
    md += ["## Table: CHAIR (test, 500 COCO val2014 images, \"Describe this image in detail.\")", "",
           "| Method | CHAIR-s | CHAIR-i | Recall | Words | Objects / caption | Truncated % |",
           "|---|---|---|---|---|---|---|"]
    numbers["chair"] = {}
    base_scored = {r["image"]: r for r in read_jsonl(R / "chair" / "baseline" / "chair_test_scored.jsonl")}
    for tag, name in METHODS_CHAIR:
        m = maybe(R / "chair" / tag / "chair_test_metrics.json")
        if not m:
            continue
        numbers["chair"][tag] = m
        md.append(f"| {name} | {pct(m['chair_s'])} | {pct(m['chair_i'])} | {pct(m['recall'])} | "
                  f"{m['avg_len_words']:.1f} | {m['avg_mentions']:.2f} | {pct(m['frac_hit_max_tokens'])} |")
    other = {r["image"]: r for r in read_jsonl(R / "chair" / "intervention" / "chair_test_scored.jsonl")}
    if base_scored and other:
        diff, n = paired_bootstrap_chair(base_scored, other)
        numbers["chair"]["intervention_minus_baseline"] = {**diff, "n_images": n}
        md += ["", "Paired image-level bootstrap, intervention - baseline (95% CI, percentage points): " +
               "; ".join(f"{k} {100 * v['diff']:+.1f} [{100 * v['ci'][0]:+.1f}, {100 * v['ci'][1]:+.1f}]"
                         for k, v in diff.items())]
        lm = length_matched(base_scored, other)
        numbers["chair"]["length_matched"] = lm
        md += ["", "Length-matched control (each pair of captions cut to the shorter one's word count): " +
               f"baseline CHAIR-s {pct(lm['baseline']['chair_s'])}, CHAIR-i {pct(lm['baseline']['chair_i'])}, "
               f"recall {pct(lm['baseline']['recall'])}; intervention CHAIR-s {pct(lm['intervention']['chair_s'])}, "
               f"CHAIR-i {pct(lm['intervention']['chair_i'])}, recall {pct(lm['intervention']['recall'])}; difference " +
               "; ".join(f"{k} {100 * v['diff']:+.1f} [{100 * v['ci'][0]:+.1f}, {100 * v['ci'][1]:+.1f}]"
                         for k, v in lm["intervention_minus_baseline"].items())]
    md.append("")

    # ---------------- E4 sweep (dev) ----------------
    sweep = maybe(R / "sweep" / "sweep_table.json")
    sel = maybe(R / "sweep" / "selected.json")
    if sweep:
        numbers["sweep"] = sweep
        numbers["selected"] = sel
        md += [f"## Table: E4 sweep on the dev set ({sweep['n_caption_images']} dev captions, "
               f"dev POPE-style questions)", "",
               "| alpha | layers | POPE acc | POPE F1 | Yes % | CHAIR-s | CHAIR-i | Recall | Words |",
               "|---|---|---|---|---|---|---|---|---|"]
        for r in sweep["rows"]:
            star = " **(selected)**" if sel and r["tag"] == sel["tag"] else ""
            md.append(f"| {r['alpha']:g} | {r['layers']}{star} | {pct(r['pope_acc'])} | {pct(r['pope_f1'])} | "
                      f"{pct(r['pope_yes_ratio'])} | {pct(r['chair_s'])} | {pct(r['chair_i'])} | "
                      f"{pct(r['recall'])} | {r['avg_len_words']:.1f} |")
        if sel:
            md += ["", f"Selected: alpha = {sel['alpha']:g}, layers {sel['layers']} - {sel['note']}."]
        md.append("")

    # ---------------- E3 attention analysis ----------------
    md += ["## Table: E3 image attention as a hallucination signal", ""]
    numbers["attention"] = {}
    for split in ("dev", "chair_test"):
        a = maybe(R / "analysis" / f"attention_{split}.json")
        if not a:
            continue
        numbers["attention"][split] = a
        best = max(a["layers"]["share_pred"], key=lambda p: p["auroc"])
        worst = min(a["layers"]["share_pred"], key=lambda p: p["auroc"])
        md += [f"**{split}**: {a['n_mentions']} object mentions in {a['n_captions']} captions, "
               f"{a['n_hallucinated']} hallucinated. Highest per-layer AUROC {best['auroc']:.3f} "
               f"(layer {best['layer']}, 95% CI {best['auroc_ci'][0]:.3f}-{best['auroc_ci'][1]:.3f}); "
               f"lowest {worst['auroc']:.3f} (layer {worst['layer']}). Band "
               f"{a.get('band_used_for_detectors', a['best_band']['layers'])}.", "",
               "| Detector (5-fold CV by image) | AUROC |", "|---|---|"]
        md += [f"| {k} | {v:.3f} |" for k, v in a["detectors_cv_auroc"].items()]
        md += [f"| single feature: {k} | {v:.3f} |" for k, v in a["single_feature_auroc"].items()]
        md.append("")

    # ---------------- E5 abstention ----------------
    ab = maybe(R / "analysis" / "abstention.json")
    if ab:
        numbers["abstention"] = ab
        md += [f"## Table: E5 abstention on POPE (band {ab['band']} chosen on dev; thresholds keep "
               f"{100 * ab['coverage_target']:.0f}% of dev questions)", "",
               "| Split | Acc (all) | Signal | AUROC (wrong) | Area under acc-coverage | Coverage @ dev thr | "
               "Acc @ dev thr |", "|---|---|---|---|---|---|---|"]
        for split, res in ab["pope"].items():
            for sig in ("image attention", "answer confidence", "random"):
                e = res[sig]
                md.append(f"| {split} | {pct(res['accuracy_full'])} | {sig} | {fmt(e['auroc_wrong'])} | "
                          f"{pct(e['area_under_acc_cov'])} | {pct(e.get('coverage_at_dev_threshold'))} | "
                          f"{pct(e.get('accuracy_at_dev_threshold'))} |")
        if "captions_filter" in ab:
            cf = ab["captions_filter"]
            md += ["", f"Caption filter (drop object mentions with low image attention, band {cf['band']}, "
                       "thresholds from dev):", "", "| Mentions kept (dev quantile) | CHAIR-s | CHAIR-i | Recall |",
                   "|---|---|---|---|"]
            for k, v in cf.items():
                if isinstance(v, dict):
                    md.append(f"| {k.replace('keep_', '')} | {pct(v['chair_s'])} | {pct(v['chair_i'])} | "
                              f"{pct(v['recall'])} |")
        md.append("")

    # ---------------- E6 failure cases ----------------
    fc = maybe(R / "analysis" / "failure_cases.json")
    if fc:
        numbers["failure_cases"] = {"summary_by_cause": fc["summary_by_cause"], "n_cases": fc["n_cases"]}
        md += ["## Table: E6 failure cases by heuristic cause", "", "| Benchmark / kind | Causes |", "|---|---|"]
        for k, v in fc["summary_by_cause"].items():
            md.append(f"| {k} | " + ", ".join(f"{c}: {n}" for c, n in sorted(v.items(), key=lambda x: -x[1])) + " |")
        md.append("")

    # ---------------- E6 hand check of CHAIR labels (report/chair_hand_check.json) ----------------
    hc = maybe(Path(__file__).resolve().parents[1] / "report" / "chair_hand_check.json")
    if hc:
        numbers["hand_check"] = hc
        md += ["## E6 hand check of CHAIR hallucination labels", "", hc["description"], "",
               "; ".join(f"{k}: {v}" for k, v in hc["counts"].items()), ""]

    # ---------------- E0 ----------------
    sanity = sorted((R / "sanity").glob("sanity_report_*.json"))
    numbers["sanity"] = {p.stem: {k: v["pass"] for k, v in load_json(p).get("checks", {}).items()} for p in sanity}

    (OUT / "tables.md").write_text("\n".join(md), encoding="utf-8")
    save_json(numbers, OUT / "numbers.json")

    # Test-set trade-off figure: CHAIR-i vs recall.
    if numbers["chair"].get("baseline") and numbers["chair"].get("intervention"):
        import matplotlib.pyplot as plt
        from halluc.plots import BLUE, ORANGE, _save
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        for tag, color, name in [("baseline", BLUE, "baseline"), ("intervention", ORANGE, "image-attention boost")]:
            m = numbers["chair"][tag]
            ax.scatter(100 * m["recall"], 100 * m["chair_i"], s=40, color=color, label=name, zorder=3,
                       edgecolor="white", linewidth=1.5)
        ax.set_xlabel("object recall (%)")
        ax.set_ylabel("CHAIR-i (%)")
        ax.set_title("CHAIR test set: hallucination vs recall")
        ax.legend(fontsize=7.5)
        _save(fig, config.FIGURE_DIR / "e4_test_tradeoff.png")
    print((OUT / "tables.md").read_text(encoding="utf-8"))
    print("\nSaved", OUT / "tables.md", "and", OUT / "numbers.json")


if __name__ == "__main__":
    main()
