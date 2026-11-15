#!/usr/bin/env python3
"""Explicit production launch. Run only after approving the prepared four-year computation."""

from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "mpl_cache"))
os.chdir(ROOT)
from hestonlab.weekly_pipeline import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["run", *sys.argv[1:]]))
