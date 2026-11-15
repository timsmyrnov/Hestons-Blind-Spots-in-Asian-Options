#!/usr/bin/env python3
"""Prepare all local inputs and renderer assets; NEVER launch the production fit."""

from pathlib import Path
import argparse
import hashlib
import os
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "mpl_cache"))
os.environ["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
ASSETS = {
    "three.module.js": (
        "build/three.module.js",
        "76dea8151bc9352aef3528b4262e249b2604f62543828328db978d060d61a495",
    ),
    "CSS2DRenderer.js": (
        "examples/jsm/renderers/CSS2DRenderer.js",
        "a4f0f79184c043f6b9d2654d8ba051e49a7d631d34e8f437c1804798a68c379f",
    ),
    "LICENSE": ("LICENSE", "852e0e8699169bf9f6fdc6bda3e682d078dcbc738b5d33e74df594721bff271d"),
}


def assets():
    folder = ROOT / "outputs" / "web_assets"
    folder.mkdir(parents=True, exist_ok=True)
    for name, (remote, digest) in ASSETS.items():
        target = folder / name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
            continue
        with urllib.request.urlopen(
            "https://cdn.jsdelivr.net/npm/three@0.160.0/" + remote, timeout=30
        ) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"Renderer asset checksum mismatch: {name}")
        target.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "HestonLab_Asian_Data")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "weekly_cpp")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "weekly_cpp.json")
    args = parser.parse_args()
    os.chdir(ROOT)
    import imageio_ffmpeg
    import matplotlib
    import numpy
    import pandas
    import pyarrow
    import scipy
    from hestonlab.movie_render import find_chrome
    from hestonlab.weekly_store import Config, write_json
    from hestonlab.weekly_pipeline import prepare

    assets()
    environment = {
        "python": sys.version,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "pyarrow": pyarrow.__version__,
        "matplotlib": matplotlib.__version__,
        "ffmpeg": imageio_ffmpeg.get_ffmpeg_exe(),
        "chrome": find_chrome(),
        "renderer_assets": "pinned SHA256; offline",
        "production_started": False,
    }
    prepare(args.root, args.out, Config.read(args.config))
    write_json(args.out / "environment.json", environment)
    print("Prepared. Production has NOT run. Await explicit green light.")


if __name__ == "__main__":
    main()
