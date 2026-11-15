"""Configuration, atomic artifacts and content-addressed identities for weekly runs."""

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import os
import numpy as np
import pandas as pd
import scipy


@dataclass(frozen=True)
class Config:
    start: str = "2022-01-01"
    end: str = "2025-12-31"
    history: int = 120
    market_rank: int = 5
    market_validation: int = 20
    fit_paths: int = 4096
    fit_calls: int = 1200
    fit_starts: int = 3
    surface_paths: int = 16384
    surface_grid: int = 11
    slice_radius: float = 0.06
    validation_paths: int = 65536
    validation_batches: int = 4
    refinement_paths: int = 16384
    refinement_calls: int = 180
    step_check_every: int = 13
    daily_checks: int = 12
    max_step_days: float = 1.0
    absolute_tolerance_usd: float = 0.02
    relative_tolerance: float = 0.15
    derivative_tolerance: float = 0.25
    smoothing_days: float = 4.0
    seconds: float = 26.0
    fps: int = 60
    budget_seconds: float = 10800.0
    render_reserve_seconds: float = 600.0
    seed: int = 2024

    def __post_init__(self):
        if pd.Timestamp(self.start) > pd.Timestamp(self.end):
            raise ValueError("Start must precede end")
        for name in ("fit_paths", "surface_paths", "validation_paths", "refinement_paths"):
            v = getattr(self, name)
            if not isinstance(v, int) or v < 4 or v % 2:
                raise ValueError(f"{name} must be even and >= 4")
        for name in (
            "history",
            "market_rank",
            "market_validation",
            "fit_calls",
            "fit_starts",
            "surface_grid",
            "validation_batches",
            "refinement_calls",
            "step_check_every",
            "fps",
        ):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.surface_grid < 3 or self.surface_grid % 2 != 1:
            raise ValueError("Surface grid must be odd and >= 3")
        if self.fit_calls < self.fit_starts or self.daily_checks < 0:
            raise ValueError("Invalid fitting or daily-check budget")
        if self.history < self.market_validation + 2 * self.market_rank + 5:
            raise ValueError("History is too short for chronological market validation")
        if not 0 < self.budget_seconds <= 10800:
            raise ValueError("Production budget must be positive and at most three hours")
        if not 0 < self.render_reserve_seconds < self.budget_seconds:
            raise ValueError("Render reserve must be smaller than the wall-clock budget")
        for name in (
            "max_step_days",
            "absolute_tolerance_usd",
            "relative_tolerance",
            "derivative_tolerance",
            "smoothing_days",
            "slice_radius",
            "seconds",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @classmethod
    def read(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("wb") as file:
        np.savez_compressed(file, **arrays)
    os.replace(temp, path)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity():
    folder = Path(__file__).parent
    names = [
        "weekly_store.py",
        "weekly_pipeline.py",
        "direct_fit.py",
        "cpp_pricer.py",
        "market_grid.py",
        "market_manifold.py",
        "weekly_geometry.py",
        "heston.py",
        "calibration.py",
        "native/heston.cpp",
    ]
    return {name: sha_file(folder / name) for name in names}


def identity(config, panel_hash, build_info):
    payload = {
        "config": asdict(config),
        "panel_sha256": panel_hash,
        "sources": source_identity(),
        "build": build_info,
        "numerical_versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


def output_root(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    if root == out or root in out.parents:
        raise ValueError("Write outputs outside the preserved market-data archive")
    if out == Path.cwd().resolve() or out == Path.cwd().resolve() / "src":
        raise ValueError("Use a dedicated output directory")
    out.mkdir(parents=True, exist_ok=True)
    return root, out


def weekly_indices(dates, config):
    days = pd.DatetimeIndex(dates)
    chosen = {}
    for i, day in enumerate(days):
        if config.start <= str(day.date()) <= config.end and i >= config.history:
            chosen[str(day.to_period("W-FRI"))] = i
    return np.array(list(chosen.values()), dtype=int)
