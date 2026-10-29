"""The wall model in a neutral atmospheric boundary layer (wind/boundary_layer.py).

Pass criteria, written down before the first full run (JOURNAL.md, Step 11):

1. steady balance: the floor's mean friction equals the push on the column, rho g H, within 2 %;
2. law of the wall: from 2 cells above the floor to a fifth of the height, the mean wind is
   within 10 % of (u* / kappa) ln(z / z0), with u* = sqrt(g H) (the uncertainty that kappa
   alone brings to the friction is about that size, A-010);
3. recorded without pass or fail: phi(z) = (kappa z / u*) dU/dz, which is 1 on the law of the
   wall; large-eddy simulations with the Smagorinsky model are known to overshoot it near the
   ground (Mason and Thomson 1992; Brasseur and Wei 2010).

The wall model fed from higher up (--match, Step 12), decided before its first run: the
height it reads is h_wm = 0.2 delta (Larsson et al. 2016, Sec. 3.2), with delta = H, the depth
of the boundary layer in this box. For H = 32 that is 6.4 cells, read at the nearest layer
centre, 6.5 (--match 6); the cells are then 0.15 h_wm tall, finer than the 0.3 h_wm that Lee
et al. (2013) needed in their code (as reported by Larsson et al.). One more run reads at 3.5
(--match 3, cells 0.29 h_wm), to show how the answer depends on the height. They are judged
by the same criteria 1 and 2.

Finer grids, on a graphics card (scripts/gpu_jobs.txt), decided before their first run: the
same box shape, 2H x H x H cells, at H = 64 and 128, with the same u* in lattice units, so a
turnover takes 2 and 4 times as many steps. Cs = 0.17. The wall model reads at h_wm = 0.2
delta: 12.8 cells for H = 64 and 25.6 for H = 128, read at the nearest layer centres, 12.5
and 25.5 (--match 12, --match 25). At H = 64 also with Cs = 0.1, and once from the first cell
(--match 0), to see whether the log-layer mismatch stays as the cells shrink. The same
criteria 1 and 2.

    uv run python benchmarks/wind_abl.py --cs 0.1
    uv run python benchmarks/wind_abl.py --cs 0.17 --nx 64 --ny 32 --height 32
    uv run python benchmarks/wind_abl.py --cs 0.1 --lattice D3Q19     # the regularised collision
    uv run python benchmarks/wind_abl.py --cs 0.17 --match 6          # wall model reads z = 6.5

The default is D3Q27 with the recursive collision (Step 12). Adds the run to
results/wind_abl_<device>.json, replacing an earlier run with the same settings; runs recorded
before Step 12 have no "collision" and used the regularised one.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import jax
import numpy as np

from aircanopy import __version__
from aircanopy.wind import boundary_layer, lbm
from aircanopy.wind.lattice import LATTICES

ROOT = Path(__file__).resolve().parents[1]
BALANCE_TOLERANCE = 0.02
LAW_TOLERANCE = 0.10


def criteria(run: dict) -> dict:
    z = np.asarray(run["z"])
    ratio = np.asarray(run["mean_u"]) / np.asarray(run["log_law"])
    surface = (z >= 2) & (z <= 0.2 * run["h"])
    return {
        "worst_ratio_in_surface_layer": float(
            ratio[surface][np.argmax(np.abs(ratio[surface] - 1))]
        ),
        f"balance_within_{BALANCE_TOLERANCE}": abs(run["floor_stress_over_push"] - 1)
        <= BALANCE_TOLERANCE,
        f"law_of_the_wall_within_{LAW_TOLERANCE}": bool(
            np.all(np.abs(ratio[surface] - 1) <= LAW_TOLERANCE)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lattice", default="D3Q27", choices=sorted(LATTICES))
    parser.add_argument("--collision", choices=lbm.COLLISIONS,
                        help="default: recursive on D3Q27, regularised on D3Q19")  # fmt: skip
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--z0", type=float, default=0.05)
    parser.add_argument("--cs", type=float, default=0.1)
    # The wall model reads the wind this many cells above the first cell (0: the first cell).
    parser.add_argument("--match", type=int, default=0)
    parser.add_argument("--spin-up", type=float, default=10.0)
    parser.add_argument("--average", type=float, default=10.0)
    args = parser.parse_args()
    collision = args.collision or ("recursive" if args.lattice == "D3Q27" else "regularised")
    start = time.perf_counter()
    run = boundary_layer.run(LATTICES[args.lattice], args.nx, args.ny, args.height, z0=args.z0,
                             smagorinsky=args.cs, collision=collision, match_cells=args.match,
                             spin_up=args.spin_up,
                             average=args.average)  # fmt: skip
    device = jax.devices()[0]
    run.update(
        seconds=round(time.perf_counter() - start, 1),
        run_at=dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        aircanopy=__version__,
        jax=jax.__version__,
        device={"platform": device.platform, "kind": device.device_kind},
        spin_up_turnovers=args.spin_up,
        average_turnovers=args.average,
        criteria=criteria(run),
    )
    shown = ("lattice", "collision", "match_cells", "smagorinsky", "seconds",
             "floor_stress_over_push", "criteria")  # fmt: skip
    print({k: run[k] for k in shown})
    path = ROOT / "results" / f"wind_abl_{device.platform}.json"
    out = json.loads(path.read_text()) if path.is_file() else {
        "what": "neutral atmospheric boundary layer over a rough floor: the wall model's test",
        "runs": [],
    }  # fmt: skip

    def key(r):
        return (r["lattice"], r.get("collision", "regularised"), r.get("match_cells", 0), r["nx"],
                r["ny"], r["h"], r["z0"], r["smagorinsky"], r["spin_up_turnovers"],
                r["average_turnovers"])  # fmt: skip

    out["runs"] = [r for r in out["runs"] if key(r) != key(run)] + [run]
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
