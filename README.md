# SC4001 Group Project – Object hallucination in Qwen2-VL-2B

Research question: when Qwen2-VL-2B-Instruct mentions an object that is not in the image, is it paying
less attention to the image, and can boosting attention to image tokens at inference time reduce
hallucination without hurting answer quality? Benchmarks: POPE and CHAIR on COCO val2014.

**Status:** all experiments E0–E6 are implemented, unit-tested and run end to end by `scripts/run_all.sh`;
every report table and figure is rebuilt from saved JSON by `scripts/make_report.py`, and the report
itself (`report/SC4001_report.pdf`) by `report/build_report.js`.

**Main results** (test sets, numbers from `results/report/numbers.json`):
- POPE accuracy 88.2 / 87.2 / 85.8 % (random / popular / adversarial). The model is conservative
  (yes-ratio about 40 %), and most errors are misses, not hallucinated "yes" answers.
- CHAIR: 36.2 % of captions and 8.0 % of object mentions are hallucinated (recall 72.3 %).
- E3: hallucinated mentions get slightly less image attention (best AUROC 0.65, layer 0), but most of that
  is caption position, and attention adds nothing to position + token log-probability.
- E4: the dev-selected boost (alpha 2, layers 6-13) leaves POPE unchanged and CHAIR-i within noise. Its
  CHAIR-s drop (-4.4 points) disappears when caption length is matched.
- E5: answer confidence is a much better abstention signal than image attention. The cautious prompt
  adds +0.3 to +0.6 accuracy points.

## How the pieces fit

```
laptop (Claude Code) --git push--> GitHub --git pull--> Colab (T4 GPU) --writes--> Google Drive
      edit code                     shared code          runs scripts          data + all results
```

- **Code** lives only in GitHub. Edit it on your laptop, never in Colab.
- **Data and results** live only in the shared Drive folder `MyDrive/sc4001` (never in git).
- **Colab** runs `notebooks/colab_runner.ipynb`, which pulls the latest code and calls the scripts.

## One-time setup

### 1. GitHub (one person creates, others join)
1. Create a repo named `sc4001-vlm-hallucination` on github.com (private is fine), with no README.
2. Push this folder:
   ```bash
   cd sc4001-vlm-hallucination
   git init -b main
   git add .
   git commit -m "Starter code: data prep, E0 sanity checks, E1 POPE"
   git remote add origin https://github.com/<you>/sc4001-vlm-hallucination.git
   git push -u origin main
   ```
3. Settings → Collaborators → add your teammates. They then `git clone` the repo.

### 2. Google Drive (one person creates, others add a shortcut)
1. Create the folder `MyDrive/sc4001` and share it with teammates as **Editor**.
2. Teammates: *Shared with me* → right-click `sc4001` → **Organize → Add shortcut → My Drive**.
   Now `/content/drive/MyDrive/sc4001` works in everyone's Colab, and results from any account land in the same place.

### 3. Colab
1. Open `notebooks/colab_runner.ipynb` in Colab (File → Open notebook → GitHub tab, or upload it).
2. Runtime → Change runtime type → **T4 GPU**.
3. Private repo only: create a GitHub *fine-grained token* (Settings → Developer settings) with read-only
   "Contents" access to this repo, then in Colab click the 🔑 icon and add a secret `GITHUB_TOKEN`
   with notebook access on. Each teammate uses their own token.
4. Edit `REPO = ...` in cell 3, then run cells 1–5 (every session) and cell 6 (once for the team).

## Running on the CCDS GPU cluster (TC1)
Long jobs can also run as SLURM batch jobs on TC1 (V100 32 GB, no browser tab to keep open).
Setup, job submission and copying results back to Drive are in [`tc1/README.md`](tc1/README.md).
Run every test-set result that goes into one table on the same hardware, so the comparison stays controlled.

## Running on a local GPU (how the report results were produced)
Any CUDA GPU with >= 8 GB works in bfloat16 (e.g. an RTX 4060 Laptop GPU):
```bash
python -m venv ~/venvs/sc4001 && source ~/venvs/sc4001/bin/activate   # Windows: venvs\sc4001\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
export SC4001_ROOT=~/sc4001          # data/ and results/ go here (keep it out of the repo)
python scripts/prepare_data.py && bash scripts/run_all.sh
```
On a laptop, plug in the charger and use a performance power mode: on battery the GPU stays in a
low-power state and decoding is several times slower.

## Daily workflow
1. `git pull` on your laptop; make changes with Claude Code; run `python -m pytest -q tests`.
2. `git commit` and `git push` (small commits, clear messages). Use a branch if you are touching
   someone else's file, and open a pull request.
3. In Colab, re-run cell 3 (pulls the new code), then run the experiment cell.
4. Add a line to `RUN_LOG.md` for every run.
5. If Colab shows an error, paste the full traceback back into Claude Code.

Do not commit the notebook with outputs (Edit → Clear all outputs before saving it to GitHub),
otherwise every run creates merge conflicts.

## Repository layout

| Path | What it is |
|------|-----------|
| `src/halluc/config.py` | All settings: model, dtype, image size, prompts, batch sizes, sweep grid, split sizes, paths |
| `src/halluc/model.py` | Model loading, (batched) greedy generation, image-token checks, attention-share helper, **E4 image-attention boost** |
| `src/halluc/chair.py` | CHAIR scoring (re-implementation of the original chair.py, original synonym list) |
| `src/halluc/data.py` | POPE loading, fixed subsets, image loading |
| `src/halluc/metrics.py` | Yes/no parsing and POPE metrics |
| `src/halluc/stats.py` | AUROC, image-level bootstrap CIs, cross-validated detectors, accuracy-coverage curves |
| `src/halluc/plots.py` | Report figures (one style, fixed colours per entity) |
| `src/halluc/io_utils.py` | Resumable JSONL writing, run metadata |
| `scripts/prepare_data.py` | Downloads data, builds frozen dev/test splits |
| `scripts/sanity_check.py` | E0 sanity checks + speed benchmark |
| `scripts/eval_pope.py` | E1 POPE baseline; E4/E5 methods via `--alpha/--layers`, `--selected`, `--cautious`; `--signals` for E5 |
| `scripts/caption_chair.py` | E2 captions + CHAIR (and E4 intervention captions) |
| `scripts/extract_attention.py` | E3 per-object image-attention statistics (teacher forcing) |
| `scripts/analyze_attention.py` | E3 analysis: per-layer gap, AUROC + CIs, position control, layer-band choice (dev) |
| `scripts/sweep.py` | E4 alpha x layer sweep on dev; writes the frozen setting `results/sweep/selected.json` |
| `scripts/abstention.py` | E5 abstention: accuracy vs coverage (attention vs answer confidence vs random); caption filter |
| `scripts/failure_cases.py` | E6 failure cases: where the intervention hurts or still hallucinates |
| `scripts/make_report.py` | All report tables (`results/report/tables.md`), significance tests, summary figure |
| `scripts/run_all.sh` | Runs E0-E6 + report in the right order (resumable) |
| `scripts/run_extra.sh` | Follow-ups: PAI-style comparison (dev sweep + one test run) and the 200-caption robustness sweep (dev only) |
| `scripts/make_submission_zip.py` | Builds `SC4001_code.zip` (code, report, saved results; no data or weights) |
| `scripts/compare_timing.py` | Compares E0 speed reports from different machines |
| `tc1/` | SLURM job scripts and setup notes for the CCDS TC1 cluster |
| `tests/` | CPU tests: parsing, metrics, CHAIR, AUROC, and the attention boost on a tiny random Qwen2-VL (incl. alpha = 0 gives identical output) |
| `notebooks/colab_runner.ipynb` | The Colab runner |
| `RUN_LOG.md` | Every run we performed |
| `report/` | Report source: `build_report.js` (docx-js; text in `sections_static.js` / `sections_results.js`, tables in `tables.js`), `to_pdf.ps1` (Word -> PDF), `chair_hand_check.json` (hand-checked CHAIR labels), output `SC4001_report.docx/.pdf` |

## Reproducing the results

**Dependencies.** Python 3.10+ (tested with 3.12), PyTorch 2.11 with CUDA, and
`pip install -r requirements.txt` (`transformers==5.22.2`, `inflect`, `scikit-learn`, `matplotlib`, ...).
Model: `Qwen/Qwen2-VL-2B-Instruct`, eager attention (needed for attention weights), greedy decoding,
images limited to 200,704 pixels (at most 256 image tokens), batched generation (12 POPE questions /
8 captions per batch, left padding; every method uses identically composed batches).

**Precision.** `DTYPE = "auto"`: bfloat16 on GPUs with native support (RTX 30/40, L4, A100), otherwise
float32 (T4, V100). **Do not use float16**: with transformers 5.22.2 the model overflows in float16
(NaN logits from decoder layer 1, every answer `"!!!!!"`); `scripts/sanity_check.py` catches this. All
results in the report were produced in bfloat16 on one RTX 4060 Laptop GPU (8 GB).

**Data.** `scripts/prepare_data.py` downloads the POPE question files, the COCO 2014 annotations, the
CHAIR synonym list and only the ~1,200 COCO val2014 images we use, and builds the frozen splits (seed 0).

**Commands.** Each step resumes after an interruption; `bash scripts/run_all.sh` runs them all
(about 8 h on an RTX 4060 Laptop GPU).

```bash
export SC4001_ROOT=/path/to/sc4001                 # data/ and results/ (Colab: /content/drive/MyDrive/sc4001)
python scripts/prepare_data.py                     # data + frozen splits
python scripts/sanity_check.py                     # E0
# --- dev set only: every choice is made here ---
python scripts/eval_pope.py --split dev_random --signals
python scripts/caption_chair.py --split dev --limit 100      # sweep subset first (identical batches)
python scripts/caption_chair.py --split dev
python scripts/extract_attention.py --split dev
python scripts/analyze_attention.py --split dev               # -> results/analysis/e3_layer_choice.json
python scripts/sweep.py                                       # -> results/sweep/selected.json  (Fig. E4 sweep)
# --- test sets, each run once with the frozen settings ---
python scripts/eval_pope.py --split all --signals             # E1 (Table POPE, baseline rows)
python scripts/caption_chair.py --split chair_test --show 5   # E2 (Table CHAIR, baseline row)
python scripts/extract_attention.py --split chair_test        # E3
python scripts/analyze_attention.py --split chair_test        # E3 (Fig. E3, Table E3)
python scripts/eval_pope.py --split all --tag intervention --selected               # E4 (POPE)
python scripts/caption_chair.py --split chair_test --tag intervention --selected    # E4 (CHAIR)
python scripts/eval_pope.py --split all --tag cautious --cautious                   # E5 comparison
# --- analysis only, no GPU ---
python scripts/abstention.py                                  # E5 (Fig. accuracy vs coverage, Table E5)
python scripts/failure_cases.py                               # E6 (Fig. failure cases, failure_cases.md)
python scripts/make_report.py                                 # all tables -> results/report/tables.md
bash scripts/run_extra.sh                                     # follow-ups (about 5 h): PAI-style comparison, 200-caption sweep
python scripts/analyze_attention.py --split dev --plot-only   # (redraw an E3 figure from saved JSON, seconds)
python scripts/make_submission_zip.py                         # SC4001_code.zip for NTULearn
python -m pytest -q tests                                     # unit tests (CPU, ~40 s)
```

**Report.** `report/SC4001_report.docx` is the working copy of the report: **edit it directly in Word**
(cover page names and group number, AI Use Statement, wording), then save it as PDF from Word
(File > Save As > PDF) or run `report/to_pdf.ps1`, which refreshes the contents page and exports
`report/SC4001_report.pdf` without touching the text. The first version was generated from the saved
results with Node.js and the `docx` npm package (`npm install docx`):

```bash
SC4001_RESULTS=$SC4001_ROOT/results node report/build_report.js --overwrite   # regenerate the .docx (DISCARDS Word edits)
powershell -ExecutionPolicy Bypass -File report/to_pdf.ps1                     # fill in contents, save .docx, -> PDF
```

`build_report.js` refuses to overwrite an existing `.docx` without `--overwrite`, so edits made in Word are
not lost by accident.

Outputs: per-example JSONL and metrics under `results/{pope,chair,attention}/<tag>/`, analyses under
`results/analysis/`, the sweep under `results/sweep/`, figures under `results/figures/`, tables under
`results/report/`. Every output folder has a `run_meta*.json` with the exact settings, versions and git commit.

## Data protocol (for the report)
- **Test sets:** POPE's 500 COCO images (3 x 3,000 questions) and 500 other COCO val2014 images for CHAIR.
- **Dev set:** 200 further COCO val2014 images, disjoint from both test sets, with 902 POPE-style
  questions (up to 3 present + 3 absent objects per image) built with POPE's random-negative protocol
  (`--split dev_random`), and dev captions for CHAIR. All choices (E3 layer band, E4 alpha and layers via
  a rule fixed in `scripts/sweep.py`, E5 abstention band and thresholds) are made on dev only. Each test
  set is run once with the final settings.
- Splits are created once with seed 0 and never rebuilt.

## Method summary (E4)
During generation, in the chosen decoder layers, every attention row whose query comes after the image
is rescaled: weights on image tokens are multiplied by (1 + alpha) and the row is renormalised to sum
to 1 (`ImageAttentionBoost` in `src/halluc/model.py`). This is a post-softmax variant of the idea in
PAI (Liu et al., ECCV 2024); alpha = 0 reproduces the original model exactly (unit-tested).

For comparison, `--mode pai` (in `sweep.py`, `eval_pope.py`, `caption_chair.py`) applies PAI's own
attention step instead: before the softmax, the image-key logits of the token being generated become
z + alpha |z| (PAI's image-free contrastive term is not used). Its alpha and layers are chosen on dev
with the same rule (`results/sweep_pai/selected.json`).

## Acknowledgements
Qwen2-VL-2B-Instruct (Alibaba Qwen team) via Hugging Face Transformers; POPE question files from
the POPE authors' repository (github.com/AoiDragon/POPE); COCO 2014 images and annotations; the CHAIR
synonym list and scoring rules from the CHAIR authors' repository (github.com/LisaAnne/Hallucination),
re-implemented in `src/halluc/chair.py` (with `inflect` instead of `pattern.en` for singularising).
Everything else (attention extraction and analysis, the intervention, sweep, abstention, failure
analysis and report scripts) was written for this project.
