# Running on the CCDS GPU cluster (TC1)

TC1 runs our scripts as background SLURM jobs on an NVIDIA V100 (32 GB). Code still comes from
GitHub; data and results live in `~/sc4001` on TC1 (the same layout as `MyDrive/sc4001`).
Per-person limits (QoS `normal`): 1 GPU, 20 CPUs, 64 GB RAM, **6 hours per job**. Our scripts resume,
so a job that hits the limit can simply be resubmitted. Check your own limits with `MyTCinfo`.

Rules from the TC1 user guide: never run Python/model code on the head node (only install packages
and copy files there), and there is **no backup** of your home directory.

## One-time setup (about 30 min)

1. **Connect.** Off campus, connect to the NTU VPN first. Then
   `ssh <your-ntu-username>@10.96.189.11` (username in lower case, NTU password).
2. **Get the code.**
   ```bash
   git clone https://github.com/Kxtnny/sc4001-vlm-hallucination.git
   cd sc4001-vlm-hallucination
   ```
   For a private repo, enter your GitHub username and a fine-grained token (read access) as the password.
3. **Create the Python environment (a venv in `~/venvs/sc4001`) and download the model** (head node, installs only):
   ```bash
   bash tc1/setup_env.sh
   ```
4. **Copy the data from Drive.** Run notebook cell 6 (`prepare_data.py`) on Colab first, if nobody has yet.
   Then download `MyDrive/sc4001/data` from Google Drive to your laptop, unzip it, and upload the
   `data` folder with WinSCP (SFTP, host `10.96.189.11`, port 22) to `~/sc4001/data`. Copy the
   frozen splits rather than rebuilding them, so TC1 and Colab use exactly the same files. Check:
   ```bash
   ls ~/sc4001/data/pope ~/sc4001/data/splits; ls ~/sc4001/data/images/val2014 | wc -l   # ~1,200
   ```

## E0: sanity checks and speed benchmark

```bash
cd ~/sc4001-vlm-hallucination && git pull
sbatch tc1/e0_sanity.sbatch        # prints "Submitted batch job <jobid>"
squeue -u $USER                    # PD = waiting, R = running
cat e0_sanity_<jobid>.out          # results; errors are in e0_sanity_<jobid>.err
```

To compare with Colab, copy `~/sc4001/results/sanity/sanity_report_tesla-v100-*.json` with WinSCP into
`MyDrive/sc4001/results/sanity/` (next to the Colab `sanity_report_tesla-t4.json`), then run
`!python scripts/compare_timing.py` in the notebook (or on your laptop with `SC4001_ROOT` set).

## All experiments (E0-E6) in one job

```bash
cd ~/sc4001-vlm-hallucination && git pull
sbatch tc1/run_all.sbatch          # runs scripts/run_all.sh; resubmit if it hits the 6 h limit (it resumes)
```
On the V100 the model runs in float32 (`DTYPE = "auto"`): the V100 has no native bfloat16 and float16
overflows in this model. Copy `~/sc4001/results` to Drive afterwards. Keep all test-set numbers of one
table from the same machine and precision.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Job stays `PD` with `QOSMax...` | You asked for more than your QoS; check with `MyTCinfo` and lower `--mem` / `--time` |
| Job stays `PD` with `Resources` / `Priority` | The cluster is busy; it will start when a GPU is free |
| `conda create` fails with `Too many levels of symbolic links` | Expected on TC1 home directories; that is why `setup_env.sh` builds a venv instead of a conda env. Clean up with `rm -rf ~/.conda/pkgs/*` |
| `No such file .../venvs/sc4001/bin/activate` in the job log | Run `bash tc1/setup_env.sh` first |
| `no kernel image is available` | The PyTorch build lacks V100 kernels: re-run `setup_env.sh` with the cu126 index (the default) |
| `LocalEntryNotFoundError` / offline errors | The model is not in `~/hf_cache`: re-run the download line of `setup_env.sh` |
| `No POPE questions at ...` | The data folder is not at `~/sc4001/data` (step 4) |
| Strange characters / `$'\r': command not found` | The script has Windows line endings: `sed -i 's/\r$//' tc1/*.sbatch tc1/*.sh` |
