"""The 3D Taylor-Green vortex at Reynolds number 1600: the first test with real turbulence.

The case of the 1st International Workshop on High-Order CFD Methods (problem C3.5,
https://cfd.ku.edu/hiocfd/case_c3.5.pdf, read 2026-10-23), compared with its reference: a
dealiased pseudo-spectral simulation on 512^3 points (`spectral_Re1600_512.gdiag`, time,
kinetic energy, dissipation rate and enstrophy, all divided by the right powers of V0 and L).
The reference file states no licence, so it is downloaded when needed, checked against its
SHA-256 and kept in the cache folder, not in the repository.

Pass criteria, written down before the first run (JOURNAL.md, Step 11): on 256^3 cells with no
turbulence model, the peak of the dissipation rate within 5 % of the reference's peak, and at
a time within 0.5 time units of it. Coarser grids are recorded without pass or fail: they are
too coarse to resolve the flow, and show how the answer approaches the reference.

    uv run python benchmarks/wind_tgv1600.py 64 128          # sizes; DNS mode
    uv run python benchmarks/wind_tgv1600.py 64 --smagorinsky 0.1

Writes results/wind_tgv1600_<device>.json (runs are added to what is already there).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import jax
import numpy as np

from aircanopy import __version__
from aircanopy.sources.cache import USER_AGENT, cache_dir
from aircanopy.wind import cases
from aircanopy.wind.lattice import LATTICES

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_URL = "https://cfd.ku.edu/hiocfd/spectral_Re1600_512.gdiag"
REFERENCE_SHA256 = "60bfeaee2dd32de1b9e7171c70cf3774b6053d82a785d526c055547a0757017a"
PEAK_TOLERANCE = 0.05
PEAK_TIME_TOLERANCE = 0.5
RESOLVED_SIZE = 256


def reference() -> np.ndarray:
    """Columns: time, kinetic energy, dissipation rate, enstrophy."""
    path = cache_dir() / "reference" / "spectral_Re1600_512.gdiag"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(REFERENCE_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=120) as r:
            path.write_bytes(r.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != REFERENCE_SHA256:
        raise SystemExit(f"{path} has SHA-256 {digest}, expected {REFERENCE_SHA256}")
    return np.loadtxt(path, comments="#")


def compare(run: dict, ref: np.ndarray) -> dict:
    t = np.asarray(run["times"])
    eps = np.asarray(cases.dissipation(run["times"], run["energy"]))
    ref_t, ref_e, ref_eps = ref[:, 0], ref[:, 1], ref[:, 2]
    k, k_ref = int(np.argmax(eps)), int(np.argmax(ref_eps))
    inside = t <= ref_t[-1]
    energy_error = np.abs(np.asarray(run["energy"])[inside] - np.interp(t[inside], ref_t, ref_e))
    out = {
        "peak_dissipation": float(eps[k]),
        "peak_time": float(t[k]),
        "reference_peak_dissipation": float(ref_eps[k_ref]),
        "reference_peak_time": float(ref_t[k_ref]),
        "peak_relative_error": float(abs(eps[k] - ref_eps[k_ref]) / ref_eps[k_ref]),
        "peak_time_error": float(abs(t[k] - ref_t[k_ref])),
        "max_energy_error": float(energy_error.max()),
        "dissipation": [float(v) for v in eps],
    }
    if run["n"] >= RESOLVED_SIZE and run["smagorinsky"] == 0.0:
        out["criteria"] = {
            f"peak_within_{PEAK_TOLERANCE}": out["peak_relative_error"] <= PEAK_TOLERANCE,
            f"peak_time_within_{PEAK_TIME_TOLERANCE}": out["peak_time_error"]
            <= PEAK_TIME_TOLERANCE,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sizes", type=int, nargs="+")
    parser.add_argument("--lattice", default="D3Q19", choices=sorted(LATTICES))
    parser.add_argument("--smagorinsky", type=float, default=0.0)
    parser.add_argument("--t-end", type=float, default=20.0)
    args = parser.parse_args()
    ref = reference()
    device = jax.devices()[0]
    path = ROOT / "results" / f"wind_tgv1600_{device.platform}.json"
    out = json.loads(path.read_text()) if path.is_file() else {
        "what": "3D Taylor-Green vortex at Re 1600 against the 512^3 pseudo-spectral reference",
        "reference": {"url": REFERENCE_URL, "sha256": REFERENCE_SHA256},
        "runs": [],
    }  # fmt: skip
    for n in args.sizes:
        start = time.perf_counter()
        run = cases.taylor_green_3d(LATTICES[args.lattice], n, smagorinsky=args.smagorinsky,
                                    t_end=args.t_end)  # fmt: skip
        run["seconds"] = round(time.perf_counter() - start, 1)
        run["run_at"] = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        run["aircanopy"] = __version__
        run["jax"] = jax.__version__
        run["device"] = {"platform": device.platform, "kind": device.device_kind}
        run["dtype"] = "float32"
        run.update(compare(run, ref))
        keep = ["n", "lattice", "smagorinsky", "seconds", "peak_dissipation", "peak_time"]
        keep += ["peak_relative_error", "peak_time_error", "max_energy_error", "criteria"]
        print({k: run[k] for k in keep if k in run})
        out["runs"] = [
            r for r in out["runs"]
            if (r["n"], r["lattice"], r["smagorinsky"], r.get("dtype")) !=
            (n, args.lattice, args.smagorinsky, "float32")
        ] + [run]  # fmt: skip
        path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
