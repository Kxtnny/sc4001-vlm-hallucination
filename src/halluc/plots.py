"""Report figures (matplotlib, static PNG for the PDF report).

Style: thin 2px lines, recessive grid, one y-axis per panel (two measures with
different scales go in two panels, never a twin axis), legends plus direct
labels, and a fixed colour per entity across all figures
(real = blue, hallucinated = orange; methods keep their colour everywhere).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots in fixed order (validated colour-blind-safe reference palette).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

plt.rcParams.update({
    "font.family": "Arial", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 2,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_attention_by_layer(result, signal, path):
    """E3: (a) image-attention share per layer, real vs hallucinated; (b) AUROC per layer."""
    L = result["layers"][signal]
    x = [p["layer"] for p in L]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 2.6), gridspec_kw={"wspace": 0.38})
    # (a) Gap in image-attention share, real minus hallucinated (percentage points), with a 95% CI
    # from the two groups' standard errors. Above zero = hallucinated objects get less image attention.
    gap = [100 * p["gap"] for p in L]
    err = [100 * 1.96 * (p["sem_real"] ** 2 + p["sem_halluc"] ** 2) ** 0.5 for p in L]
    a.bar(x, gap, color="#8a8985", width=0.7)
    a.errorbar(x, gap, yerr=err, fmt="none", ecolor=INK2, elinewidth=0.8, capsize=1.5)
    a.axhline(0, color=INK2, linewidth=0.8)
    a.set_xlabel("decoder layer")
    a.set_ylabel("share gap, real - halluc. (pp)")
    a.set_title("(a) Image attention: real minus hallucinated")
    au = [p["auroc"] for p in L]
    b.plot(x, au, color=VIOLET, label="AUROC")
    b.fill_between(x, [p["auroc_ci"][0] for p in L], [p["auroc_ci"][1] for p in L], color=VIOLET,
                   alpha=0.15, linewidth=0, label="95% CI (image bootstrap)")
    wq = [p["auroc_within_position_quintiles"] for p in L]
    if all(v is not None for v in wq):
        b.plot(x, wq, color=INK2, linewidth=1.2, linestyle="--", label="within position quintiles")
    b.axhline(0.5, color=INK2, linewidth=0.8, linestyle=":")
    b.set_xlabel("decoder layer")
    b.set_ylabel("AUROC (low attn. = halluc.)")
    b.set_title("(b) Image attention as a hallucination detector")
    b.legend(loc="best", fontsize=7.5)
    name = {"chair_test": "test captions", "dev": "dev captions"}.get(result["split"], result["split"])
    fig.suptitle(f"{name}: {result['n_mentions']} object mentions, "
                 f"{result['n_hallucinated']} hallucinated", fontsize=8.5, color=INK2, y=1.02)
    _save(fig, path)


def plot_sweep(rows, baseline, path):
    """E4 dev sweep: alpha vs CHAIR-i, recall and POPE-dev F1, one line per layer range.

    rows: dicts with alpha, layers, chair_i, recall, pope_f1; baseline: same keys for alpha=0.
    """
    ranges = list(dict.fromkeys(r["layers"] for r in rows))
    panels = [("chair_i", "CHAIR-i % (lower is better)"), ("recall", "object recall %"),
              ("pope_f1", "POPE-dev F1 %")]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))
    for ax, (key, title) in zip(axes, panels):
        for i, lr in enumerate(ranges):
            pts = sorted([(0.0, baseline[key])] + [(r["alpha"], r[key]) for r in rows if r["layers"] == lr])
            ax.plot([p[0] for p in pts], [100 * p[1] for p in pts], color=SERIES[i], marker="o", markersize=3.5,
                    label=f"layers {lr}")
        ax.axhline(100 * baseline[key], color=INK2, linewidth=0.8, linestyle=":")
        ax.set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=len(labels), fontsize=7.5)
    fig.supxlabel("alpha (attention on image tokens multiplied by 1 + alpha; dotted line = unmodified model)",
                  fontsize=8.5, color=INK, y=0.02)
    fig.tight_layout()
    _save(fig, path)


def plot_accuracy_coverage(curves, path, title="POPE: accuracy vs coverage"):
    """E5 abstention. curves: list of (label, coverage array, accuracy array)."""
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    colors = {"image attention": VIOLET, "answer confidence": BLUE, "random": INK2}
    for i, (label, cov, acc) in enumerate(curves):
        ax.plot(cov, acc, color=colors.get(label, SERIES[i % len(SERIES)]), label=label,
                linestyle="--" if label == "random" else "-", linewidth=1.2 if label == "random" else 2)
    ax.set_xlabel("coverage (fraction answered)")
    ax.set_ylabel("accuracy on answered")
    ax.set_title(title)
    ax.set_xlim(0.1, 1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7, handlelength=1.6,
              columnspacing=1.0)
    _save(fig, path)
