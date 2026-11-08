"""Small helpers for saving results safely on Google Drive.

Results are written as JSON Lines (one JSON object per line) and flushed after
every example, so a Colab disconnect loses at most the example in progress.
Scripts read the file back on start and skip what is already done (resume).
"""
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def read_jsonl(path):
    """Return a list of dicts. Missing file -> empty list. A half-written last
    line (from a disconnect mid-write) is skipped instead of crashing."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[warn] skipping a corrupt line in {path}")
    return rows


class JsonlWriter:
    """Append-only JSONL writer that flushes to disk after each record."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "a", encoding="utf-8")

    def write(self, record):
        self.f.write(json.dumps(record) + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)  # atomic: never leaves a half-written JSON


def write_jsonl(rows, path):
    """Write a whole JSONL file at once (atomic), for derived files that are recomputed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def git_commit():
    """Commit hash of the code that produced a result (for the run log)."""
    repo = Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def run_metadata(args=None, **extra):
    """Everything needed to say exactly how a result was produced."""
    meta = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "argv": sys.argv,
    }
    try:
        import torch
        import transformers
        meta["torch"] = torch.__version__
        meta["transformers"] = transformers.__version__
        meta["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except ImportError:
        pass
    if args is not None:
        meta["args"] = {k: str(v) for k, v in vars(args).items()}
    meta.update(extra)
    return meta


def selected_setting(results_dir, name="sweep"):
    """(alpha, layers) chosen on the dev set by scripts/sweep.py (results/<name>/selected.json)."""
    path = Path(results_dir) / name / "selected.json"
    if not path.exists():
        raise SystemExit(f"{path} not found: run scripts/sweep.py (dev set) first.")
    sel = load_json(path)
    return float(sel["alpha"]), sel["layers"]
