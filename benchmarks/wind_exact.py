"""The wind solver against flows with exact answers, on grids of three sizes.

For each flow, both velocity sets (D3Q19, D3Q27) and both collisions (regularised, BGK):
the error on each grid, the observed order of convergence between successive grids, and
whether the pass criteria hold. The criteria were written down before the first run (JOURNAL.md,
Step 10):

- shear wave: viscosity within 1 % at 32 cells per wavelength; order at least 1.8;
- Poiseuille flow: relative error below 1 % with 16 cells across; order at least 1.8;
- Taylor-Green vortex: relative velocity error below 1 % at 32 cells per period, at the time
  its speed has halved; order at least 1.8.

The grids are refined with the time step (diffusive scaling: tau fixed, the speed in lattice
units halved with each doubling of the grid). The Taylor-Green vortex is also run at speeds 4
and 16 times lower, because at the planned speed the gas's compressibility, which the exact
solution leaves out, is larger than the error being measured (Step 10).

Runs on the processor in float64, a few minutes. Writes results/wind_exact.json.

    uv run python benchmarks/wind_exact.py
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import jax

from aircanopy import __version__
from aircanopy.wind import cases
from aircanopy.wind.lattice import D3Q19, D3Q27

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
MAX_ERROR = 0.01
MIN_ORDER = 1.8
TAU = 0.8


def study(case, lat, regularised: bool, sizes: list[int], at: int, u16: float | None = None):
    """Run `case` on each grid size; with u16, the speed is u16 on 16 cells, scaled as 16/n."""
    results = []
    for n in sizes:
        speed = {} if u16 is None else {"u0": u16 * 16 / n}
        results.append(case(lat, n, TAU, regularised=regularised, **speed))
    errors = [r["relative_error"] for r in results]
    orders = cases.observed_order(sizes, errors)
    error_at = errors[sizes.index(at)]
    return {
        "sizes": sizes,
        "relative_errors": errors,
        "observed_orders": [round(o, 3) for o in orders],
        "criteria": {
            f"error_below_{MAX_ERROR}_at_{at}": error_at < MAX_ERROR,
            f"orders_at_least_{MIN_ORDER}": all(o >= MIN_ORDER for o in orders),
        },
        "runs": results,
    }


def main() -> None:
    out = {
        "what": "errors against exact solutions, and observed orders of convergence",
        "run_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "aircanopy": __version__,
        "jax": jax.__version__,
        "precision": "float64",
        "tau": TAU,
        "cases": {},
    }
    start = time.perf_counter()
    for lat in (D3Q19, D3Q27):
        for regularised in (True, False):
            key = f"{lat.name}_{'regularised' if regularised else 'bgk'}"
            r = {
                "shear_wave": study(cases.shear_wave, lat, regularised, [16, 32, 64], 32),
                "poiseuille": study(cases.poiseuille, lat, regularised, [8, 16, 32], 16),
            }
            for label, u16 in (("", 0.04), ("_speed_divided_by_4", 0.01),
                               ("_speed_divided_by_16", 0.0025)):  # fmt: skip
                r[f"taylor_green{label}"] = study(
                    cases.taylor_green, lat, regularised, [16, 32, 64], 32, u16
                )
            out["cases"][key] = r
            print(key, {k: v["criteria"] for k, v in r.items()})
    out["seconds"] = round(time.perf_counter() - start)
    path = ROOT / "results" / "wind_exact.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path} in {out['seconds']} s")


if __name__ == "__main__":
    main()
