"""Build the code ZIP for NTULearn: every file git tracks or would track (code, tests, README,
RUN_LOG, report sources and PDF) plus the saved results (per-example outputs, metrics, figures,
numbers.json), so every number in the report can be re-derived with scripts/make_report.py, no GPU.
Data (COCO images, POPE files) and model weights are not included; scripts/prepare_data.py fetches them.

Usage:  python scripts/make_submission_zip.py [--out SC4001_code.zip] [--no-results]
        python scripts/make_submission_zip.py --results-only   (writes results.zip, the snapshot kept in git)

results.zip unpacks to results/; put that folder in SC4001_ROOT to re-derive the report without a GPU.
"""
import _path  # noqa: F401
import argparse
import subprocess
import zipfile
from pathlib import Path

from halluc import config

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "SC4001_code.zip"))
    ap.add_argument("--no-results", action="store_true", help="code only")
    ap.add_argument("--results-only", action="store_true",
                    help="only the results folder, as results/... (default --out results.zip)")
    args = ap.parse_args()
    if args.results_only:
        out = (REPO / "results.zip") if args.out == str(REPO / "SC4001_code.zip") else Path(args.out).resolve()
        n = 0
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for p in sorted(config.RESULTS_DIR.rglob("*")):
                if p.is_file() and not p.name.endswith(".tmp"):
                    z.write(p, f"results/{p.relative_to(config.RESULTS_DIR).as_posix()}")
                    n += 1
        print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB): {n} result files")
        return
    out = Path(args.out).resolve()

    files = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=REPO, check=True,
                           capture_output=True, text=True).stdout.split("\n")
    # results.zip (the git snapshot of the results) is skipped: the results folder itself goes in below.
    files = [f for f in files if f and f != "results.zip" and (REPO / f).is_file() and (REPO / f).resolve() != out]
    # The report PDF is ignored by nothing but may be untracked; make sure it is in.
    for extra in ("report/SC4001_report.pdf",):
        if (REPO / extra).exists() and extra not in files:
            files.append(extra)

    n_res = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in sorted(files):
            z.write(REPO / f, f"sc4001-vlm-hallucination/{f}")
        if not args.no_results and config.RESULTS_DIR.exists():
            for p in sorted(config.RESULTS_DIR.rglob("*")):
                if p.is_file() and not p.name.endswith(".tmp"):
                    z.write(p, f"sc4001-vlm-hallucination/results/{p.relative_to(config.RESULTS_DIR).as_posix()}")
                    n_res += 1
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB): {len(files)} repository files, {n_res} result files")


if __name__ == "__main__":
    main()
