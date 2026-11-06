import sys
from pathlib import Path

# Lets scripts do `from halluc import ...` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
