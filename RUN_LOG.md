# Run log

Every experiment run goes here. The report only describes runs listed in this file.
`git commit` is in the `run_meta*.json` saved next to the results ("-dirty" = uncommitted edits at run time).

| Date | Who | Exp | Command | Tag / output folder | Git commit | Notes |
|------|-----|-----|---------|---------------------|-----------|-------|
| 2026-11-07 | Woollace (via Claude Code, RTX 4060 Laptop 8 GB) | data | `python scripts/prepare_data.py` | data/ (splits seed 0) | 16ab744-dirty | 9,000 POPE q; 902 dev q; 1,200/1,200 images; CHAIR synonyms (80 categories) |
| 2026-11-07 | Woollace (via Claude Code, RTX 4060) | E0 | `python scripts/sanity_check.py` (float16, first attempt) | (overwritten) | 16ab744-dirty | **4 checks FAIL**: NaN attention/logits from layer 1, every answer "!!!!!" -> float16 unusable; re-recorded on 2026-11-08 (below) |
| 2026-11-07 | Woollace (via Claude Code, RTX 4060) | E0 | `python scripts/sanity_check.py --dtype bfloat16 --tag rtx4060-bf16` | results/sanity/sanity_report_rtx4060-bf16.json | 16ab744-dirty | all checks pass; 10/10 answers correct; max row-sum error 3.6e-3 |
| 2026-11-07 | Woollace (via Claude Code, RTX 4060) | E0-E6 | `bash scripts/run_all.sh` (DTYPE auto = bfloat16, batched) | results/{pope,chair,attention,sweep,analysis,figures,report} | 16ab744-dirty | 18:12 to 02:16 (about 8 h). Stages: E0; dev POPE + signals; dev captions (100 + 100); E3 dev -> band 6-13; E4 sweep (21 settings) -> selected alpha 2, layers 6-13 (constraints met); test POPE (3 x 3,000) + signals, CHAIR test (500), E3 test; frozen intervention POPE + CHAIR; cautious POPE; E5, E6, report tables. Log: logs/run_all.log |
| 2026-11-08 | Woollace (via Claude Code, RTX 4060) | E0 | `python scripts/sanity_check.py --dtype float16 --tag rtx4060-fp16` | results/sanity/sanity_report_rtx4060-fp16.json | 16ab744-dirty | evidence for the report: 4 checks FAIL (image share defined in layer 0, NaN in layers 1-27) |
| 2026-11-08 | Claude Code | analysis | `python scripts/analyze_attention.py --split dev` / `--split chair_test`; `python scripts/sweep.py --table-only` | results/analysis, results/figures | 16ab744-dirty | figures redrawn in the final style only; same numbers, same layer choice (6-13) and selected setting |
| 2026-11-08 | Claude Code | E4/E5/E6 analysis | `python scripts/abstention.py`; `python scripts/make_report.py` | results/analysis/abstention.json, results/report/ | 16ab744-dirty | added the length-matched CHAIR control, the exploratory high-attention layer (reported separately, not used for the frozen threshold) and the hand check of 30 CHAIR labels (report/chair_hand_check.json; labels proposed by Claude from caption text, **team to verify**) |
