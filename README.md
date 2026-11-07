# SC4001 Group Project – Object hallucination in Qwen2-VL-2B

Research question: when Qwen2-VL-2B-Instruct mentions an object that is not in the image, is it paying
less attention to the image, and can boosting attention to image tokens at inference time reduce
hallucination without hurting answer quality? Benchmarks: POPE and CHAIR on COCO val2014.

**Status:** E0 (sanity checks) and E1 (POPE baseline) are implemented. E2–E6 are next.

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
| `src/halluc/config.py` | All settings: model, dtype, image size, prompts, split sizes, paths |
| `src/halluc/model.py` | Model loading, greedy generation, image-token checks, attention-share helper |
| `src/halluc/data.py` | POPE loading, fixed subsets, image loading |
| `src/halluc/metrics.py` | Yes/no parsing and POPE metrics |
| `src/halluc/io_utils.py` | Resumable JSONL writing to Drive, run metadata |
| `scripts/prepare_data.py` | Downloads data, builds frozen dev/test splits |
| `scripts/sanity_check.py` | E0 |
| `scripts/eval_pope.py` | E1 (and later: other methods via `--tag`) |
| `scripts/compare_timing.py` | Compares E0 speed reports from different machines |
| `tc1/` | SLURM job scripts and setup notes for the CCDS TC1 cluster |
| `tests/` | CPU tests, including a tiny random Qwen2-VL for the attention helpers |
| `notebooks/colab_runner.ipynb` | The Colab runner |
| `RUN_LOG.md` | Every run we performed |

## Reproducing the results

Dependencies: Python 3.10+, PyTorch (preinstalled on Colab), `pip install -r requirements.txt`
(`transformers==5.22.2`). Model: `Qwen/Qwen2-VL-2B-Instruct` in float16, eager attention,
greedy decoding, images limited to 200,704 pixels (at most 256 image tokens).

```bash
export SC4001_ROOT=/content/drive/MyDrive/sc4001   # set automatically by the notebook
python scripts/prepare_data.py                     # data + frozen splits (seed 0)
python scripts/sanity_check.py                     # E0 (+ speed benchmark)
python scripts/eval_pope.py --split all            # E1: Table "POPE baseline"
```

## Data protocol (for the report)
- **Test sets:** POPE's 500 COCO images (3 × 3,000 questions) and 500 other COCO val2014 images for CHAIR.
- **Dev set:** 200 further COCO val2014 images, disjoint from both test sets, with up to 1,200 POPE-style
  questions (3 present + 3 absent objects per image) built with POPE's random-negative protocol (`--split dev_random`). All choices of alpha,
  layers and prompts are made on dev only. Each test set is run once with the final settings.
- Splits are created once with seed 0 and never rebuilt.

## Acknowledgements
Qwen2-VL-2B-Instruct (Alibaba Qwen team) via Hugging Face Transformers; POPE question files from
the authors' repository (github.com/AoiDragon/POPE); COCO 2014 images and annotations.
