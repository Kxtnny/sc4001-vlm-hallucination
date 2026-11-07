#!/bin/bash
# Runs every experiment in the order the report depends on (E0 -> E6), then builds
# all tables and figures. Every step resumes, so after an interruption just run it again.
#
#   export SC4001_ROOT=/path/to/sc4001     # data/ and results/ live here
#   bash scripts/run_all.sh                # PY=/path/to/python to choose the interpreter
#
# Dev/test protocol: steps 1-3 use only the dev images (layer band, alpha, layers,
# abstention band/threshold are all chosen here). Steps 4-5 run each test set once with
# the settings frozen in results/sweep/selected.json.
set -eo pipefail
PY=${PY:-python}
cd "$(dirname "$0")/.."

echo "== E0 sanity checks"
$PY scripts/sanity_check.py

echo "== 1. dev baseline: POPE-style dev questions (+ decision signals) and dev captions"
$PY scripts/eval_pope.py --split dev_random --signals
# First the 100-image dev subset used by the sweep (so the baseline is generated in exactly the
# same batches as every sweep setting), then the other 100 dev images (E3 uses all 200).
$PY scripts/caption_chair.py --split dev --limit 100
$PY scripts/caption_chair.py --split dev

echo "== 2. E3 on dev captions -> layer band for E4 / abstention"
$PY scripts/extract_attention.py --split dev
$PY scripts/analyze_attention.py --split dev

echo "== 3. E4 sweep on dev -> results/sweep/selected.json"
$PY scripts/sweep.py

echo "== 4. test baselines: E1 POPE (+ signals), E2 CHAIR, E3 attention on test captions"
$PY scripts/eval_pope.py --split all --signals
$PY scripts/caption_chair.py --split chair_test --show 5
$PY scripts/extract_attention.py --split chair_test
$PY scripts/analyze_attention.py --split chair_test

echo "== 5. test runs of the frozen methods: E4 intervention, E5 cautious prompt"
$PY scripts/eval_pope.py --split all --tag intervention --selected
$PY scripts/caption_chair.py --split chair_test --tag intervention --selected
$PY scripts/eval_pope.py --split all --tag cautious --cautious

echo "== 6. analysis only (no model): E5 abstention, E6 failure cases, report tables/figures"
$PY scripts/abstention.py
$PY scripts/failure_cases.py
$PY scripts/make_report.py
echo "done"
