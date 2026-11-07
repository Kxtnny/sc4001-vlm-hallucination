"""Compare E0 speed benchmarks from different machines (no GPU or model needed).

Reads every results/sanity/sanity_report_*.json (one per machine, written by
scripts/sanity_check.py) and prints one column per machine. Copy the TC1
report into the same Drive folder as the Colab one first (see tc1/README.md).

Usage:  python scripts/compare_timing.py                 # all reports in results/sanity/
        python scripts/compare_timing.py a.json b.json   # specific files
"""
import _path  # noqa: F401
import argparse
from pathlib import Path

from halluc import config
from halluc.io_utils import load_json

# (label, key in report["timing"] or top-level report, format)
ROWS = [
    ("host", "host", "{}"),
    ("GPU", "gpu", "{}"),
    ("torch", "torch", "{}"),
    ("transformers", "transformers", "{}"),
    ("dtype", "dtype", "{}"),
    ("CPUs", "cpus", "{}"),
    ("git commit", "git_commit", "{}"),
    ("model load (s)", "model_load_s", "{:.0f}"),
    ("image load (s/img)", "image_load_s_mean", "{:.3f}"),
    ("POPE (s/question)", "pope_s_per_question_mean", "{:.3f}"),
    ("caption (s/caption)", "caption_s_mean", "{:.1f}"),
    ("caption tokens/s", "caption_tokens_per_s", "{:.1f}"),
    ("caption length (tok)", "caption_new_tokens_mean", "{:.0f}"),
    ("peak GPU mem (GB)", "peak_gpu_mem_gb", "{:.1f}"),
    ("-> 9,000 POPE q (min)", "projected_pope_9000_questions_min", "{:.0f}"),
    ("-> 500 captions (min)", "projected_chair_500_captions_min", "{:.0f}"),
]
SPEEDUP = [("POPE speed-up", "pope_s_per_question_mean"), ("caption speed-up", "caption_s_mean")]


def lookup(report, key):
    return report.get("timing", {}).get(key, report.get(key, "-"))


def fmt(value, pattern):
    try:
        return pattern.format(value)
    except (ValueError, TypeError):
        return str(value)


def table(reports, names):
    """Return the comparison as a list of text lines (one column per report)."""
    width = max(14, *(len(n) for n in names))
    lines = [f"{'':<24}" + "".join(f"{n:>{width + 2}}" for n in names)]
    for label, key, pattern in ROWS:
        lines.append(f"{label:<24}" + "".join(f"{fmt(lookup(r, key), pattern):>{width + 2}}" for r in reports))
    if len(reports) > 1:  # relative to the first column (higher = faster)
        for label, key in SPEEDUP:
            base = lookup(reports[0], key)
            cells = []
            for r in reports:
                v = lookup(r, key)
                ok = all(isinstance(x, (int, float)) for x in (base, v)) and v > 0
                cells.append(f"{base / v:.2f}x" if ok else "-")
            lines.append(f"{label + ' vs col 1':<24}" + "".join(f"{c:>{width + 2}}" for c in cells))
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="report files (default: results/sanity/sanity_report_*.json)")
    args = ap.parse_args()

    paths = [Path(f) for f in args.files] or sorted((config.RESULTS_DIR / "sanity").glob("sanity_report_*.json"))
    reports = [(p, load_json(p)) for p in paths]
    reports = [(p, r) for p, r in reports if "timing" in r]  # skip reports from before the benchmark existed
    if not reports:
        raise SystemExit(f"No benchmark reports found in {config.RESULTS_DIR / 'sanity'}. Run sanity_check.py first.")

    names = [p.stem.replace("sanity_report_", "") for p, _ in reports]
    print("\n".join(table([r for _, r in reports], names)))
    print("\nTimes are GPU work on identical inputs (warm-up excluded); image loading is listed separately.")


if __name__ == "__main__":
    main()
