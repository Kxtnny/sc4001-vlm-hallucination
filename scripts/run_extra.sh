#!/bin/bash
# Follow-up experiments, run after scripts/run_all.sh. Each step resumes after an interruption.
#
#   export SC4001_ROOT=/path/to/sc4001
#   bash scripts/run_extra.sh
#
# 1. PAI-style comparison (E4b): PAI's pre-softmax logit amplification z + alpha|z| on image keys for
#    the token being generated (Liu et al., ECCV 2024; without PAI's image-free contrast term).
#    Grid: alpha 0.05-0.5 (alpha >= 0.5 on 26 layers already breaks the caption format on this model).
#    Its alpha and layers are chosen on dev with the SAME rule as our boost (scripts/sweep.py),
#    written to results/sweep_pai/selected.json, and then each test set is run once.
# 2. Robustness of the E4 choice (dev only): the same sweep as E4 on all 200 dev captions instead of
#    100. Written to results/sweep_n200/; it never replaces results/sweep/selected.json, and the test
#    sets are NOT re-run with it (that would choose a setting after seeing test results).
set -eo pipefail
PY=${PY:-python}
cd "$(dirname "$0")/.."

echo "== E4b: PAI-style sweep on dev -> results/sweep_pai/selected.json"
$PY scripts/sweep.py --mode pai --out-name sweep_pai --alphas 0.05 0.1 0.2 0.5 --layers 2-27 6-13 --no-e3-band

echo "== E4b: PAI-style method on the test sets (run once, frozen setting)"
$PY scripts/eval_pope.py --split all --tag pai --mode pai --selected --selected-from sweep_pai
$PY scripts/caption_chair.py --split chair_test --tag pai --mode pai --selected --selected-from sweep_pai

echo "== E4 robustness: our boost, same grid, 200 dev captions (dev only)"
$PY scripts/sweep.py --n-caption-images 200 --out-name sweep_n200

echo "== tables"
$PY scripts/make_report.py
echo "done"
