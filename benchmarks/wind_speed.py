"""Speed of the wind solver: million cell updates per second (MLUPS), and memory per cell.

A realistic step on a 64 x 64 x 64 box: a solid ground layer and a block on it, a free-slip
top, a body force, the Smagorinsky model and the regularised collision. For D3Q19 and D3Q27,
in float32 and float64, compiled with JAX for this machine's processor (or graphics card, if
JAX finds one). The first call is timed separately: it includes compiling.

Writes results/wind_speed_<device>.json.

    uv run python benchmarks/wind_speed.py [n]
"""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import sys
import time
from pathlib import Path

import jax
import numpy as np

from aircanopy import __version__
from aircanopy.wind.lattice import D3Q19, D3Q27
from aircanopy.wind.solver import Params, Solver

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]


def measure(lat, dtype, n: int, steps: int) -> dict:
    solid = np.zeros((n, n, n), bool)
    solid[0] = True
    solid[1 : n // 4, n // 3 : n // 2, n // 3 : n // 2] = True
    params = Params(tau0=0.5001, smagorinsky=0.15, g=(1e-6, 0.0, 0.0), free_slip_top=True)
    s = Solver(lat, solid, params, "jax", dtype)
    start = time.perf_counter()
    s.run(1)
    s.f.block_until_ready()
    first = time.perf_counter() - start
    s.run(steps // 10)  # warm up
    s.f.block_until_ready()
    start = time.perf_counter()
    s.run(steps)
    s.f.block_until_ready()
    seconds = time.perf_counter() - start
    rho, _ = s.fields()
    return {
        "lattice": lat.name,
        "dtype": np.dtype(dtype).name,
        "cells": n**3,
        "steps": steps,
        "seconds": round(seconds, 3),
        "mlups": round(n**3 * steps / seconds / 1e6, 2),
        "first_call_seconds": round(first, 2),
        "population_bytes_per_cell": lat.q * np.dtype(dtype).itemsize,
        "finite": bool(np.isfinite(rho).all()),
    }


def main(n: int = 64, steps: int = 200) -> None:
    device = jax.devices()[0]
    runs = [measure(lat, dt_, n, steps) for lat in (D3Q19, D3Q27)
            for dt_ in (np.float32, np.float64)]  # fmt: skip
    for r in runs:
        print(r)
    out = {
        "what": "cell updates per second of the regularised Smagorinsky step with walls, force "
        "and free-slip top",
        "run_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "aircanopy": __version__,
        "jax": jax.__version__,
        "device": {"platform": device.platform, "kind": device.device_kind},
        "machine": {"cpus": os.cpu_count(), "system": platform.system()},
        "runs": runs,
    }
    path = ROOT / "results" / f"wind_speed_{device.platform}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
