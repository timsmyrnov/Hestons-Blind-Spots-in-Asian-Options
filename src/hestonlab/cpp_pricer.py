"""Compile/load the C++17 pricer through its small C ABI; no extra binding package."""

from dataclasses import dataclass
from pathlib import Path
import ctypes
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
import numpy as np
from .heston import HestonParameters, MonteCarloConfig

SOURCE = Path(__file__).with_name("native") / "heston.cpp"


def build(cache=Path("outputs/native")):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if not compiler:
        raise FileNotFoundError(
            "A C++17 compiler is required; install Xcode command line tools or GCC"
        )
    version = subprocess.check_output([compiler, "--version"], text=True, timeout=15)
    flags = ["-O3", "-std=c++17", "-fPIC", "-shared"]
    identity = hashlib.sha256(
        SOURCE.read_bytes() + repr((version, flags, platform.machine())).encode()
    ).hexdigest()
    cache = Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (identity[:20] + (".dylib" if platform.system() == "Darwin" else ".so"))
    if not target.exists():
        temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
        try:
            subprocess.run(
                [compiler, *flags, str(SOURCE), "-o", str(temporary)],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return target, {
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "build_id": identity,
        "compiler": version.splitlines()[0],
        "flags": flags,
    }


@dataclass
class Estimate:
    prices: np.ndarray
    standard_errors: np.ndarray
    covariance: np.ndarray  # covariance of estimated means, not of individual payoffs


class CppPricer:
    def __init__(self, contracts, cache=Path("outputs/native")):
        self.contracts = tuple(contracts)
        if not self.contracts:
            raise ValueError("At least one contract required")
        library, self.build_info = build(cache)
        self.library = ctypes.CDLL(str(library))
        array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        ints = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        self.library.heston_price.argtypes = [
            ctypes.c_int,
            ints,
            array,
            array,
            ints,
            array,
            array,
            ints,
            array,
            array,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_double,
            ctypes.c_int,
            array,
            array,
            array,
        ]
        self.library.heston_price.restype = ctypes.c_int
        self.library.heston_error.restype = ctypes.c_char_p
        c = self.contracts
        self.args = (
            np.array([0, *np.cumsum([len(x.fixing_times) for x in c])], dtype=np.int32),
            np.array([t for x in c for t in x.fixing_times], dtype=np.float64),
            np.array([f for x in c for f in x.forwards], dtype=np.float64),
            np.array([x.total_fixings for x in c], dtype=np.int32),
            np.array([x.known_fixing_sum for x in c], dtype=np.float64),
            np.array([x.strike for x in c], dtype=np.float64),
            np.array([1 if x.side == "C" else -1 for x in c], dtype=np.int32),
            np.array([x.discount_factor for x in c], dtype=np.float64),
        )

    def price(self, parameters, paths=4096, seed=2024, max_step=1 / 365, covariance=False):
        if isinstance(parameters, HestonParameters):
            parameters = [
                parameters.v0,
                parameters.kappa,
                parameters.theta,
                parameters.sigma,
                parameters.rho,
            ]
        parameters = np.ascontiguousarray(parameters, dtype=np.float64)
        if parameters.shape != (5,):
            raise ValueError("Five Heston parameters required")
        HestonParameters(*parameters)
        MonteCarloConfig(paths, max_step, seed)
        if seed >= 2**64 or paths > 2**31 - 1:
            raise ValueError("Seed or path count exceeds native integer range")
        n = len(self.contracts)
        prices, errors, cov = np.empty(n), np.empty(n), np.empty((n, n))
        status = self.library.heston_price(
            n, *self.args, parameters, paths, seed, max_step, int(covariance), prices, errors, cov
        )
        if status:
            raise FloatingPointError(self.library.heston_error().decode())
        return Estimate(prices, errors, cov)

    def benchmark(self, counts=(4096, 16384, 65536), repeats=3):
        rows = []
        self.price(HestonParameters(), paths=128)
        for paths in counts:
            times = []
            for i in range(repeats):
                start = time.perf_counter()
                self.price(HestonParameters(), paths=paths, seed=2024 + i, covariance=True)
                times.append(time.perf_counter() - start)
            rows.append(
                {
                    "paths": paths,
                    "median_seconds": float(np.median(times)),
                    "max_seconds": max(times),
                }
            )
        return {"build": self.build_info, "contracts": len(self.contracts), "timings": rows}


if __name__ == "__main__":
    from .market_grid import GridSpec, heston_grid_contracts

    print(json.dumps(CppPricer(heston_grid_contracts(GridSpec())).benchmark(), indent=2))
