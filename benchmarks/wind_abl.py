"""The wall model in a neutral atmospheric boundary layer (wind/boundary_layer.py).

Pass criteria, written down before the first full run (JOURNAL.md, Step 11):

1. steady balance: the floor's mean friction equals the push on the column, rho g H, within 2 %;
2. law of the wall: from 2 cells above the floor to a fifth of the height, the mean wind is
   within 10 % of (u* / kappa) ln(z / z0), with u* = sqrt(g H) (the uncertainty that kappa
   alone brings to the friction is about that size, A-010);
3. recorded without pass or fail: phi(z) = (kappa z / u*) dU/dz, which is 1 on the law of the
   wall; large-eddy simulations with the Smagorinsky model are known to overshoot it near the
   ground (Mason and Thomson 1992; Brasseur and Wei 2010).

    uv run python benchmarks/wind_abl.py --cs 0.1
    uv run python benchmarks/wind_abl.py --cs 0.17 --nx 64 --ny 32 --height 32
    uv run python benchmarks/wind_abl.py --cs 0.1 --lattice D3Q19     # the regularised collision

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
    parser.add_argument("--spin-up", type=float, default=10.0)
    parser.add_argument("--average", type=float, default=10.0)
    args = parser.parse_args()
    collision = args.collision or ("recursive" if args.lattice == "D3Q27" else "regularised")
    start = time.perf_counter()
    run = boundary_layer.run(LATTICES[args.lattice], args.nx, args.ny, args.height, z0=args.z0,
                             smagorinsky=args.cs, collision=collision, spin_up=args.spin_up,
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
    shown = ("lattice", "collision", "smagorinsky", "seconds", "floor_stress_over_push", "criteria")
    print({k: run[k] for k in shown})
    path = ROOT / "results" / f"wind_abl_{device.platform}.json"
    out = json.loads(path.read_text()) if path.is_file() else {
        "what": "neutral atmospheric boundary layer over a rough floor: the wall model's test",
        "runs": [],
    }  # fmt: skip

    def key(r):
        return (r["lattice"], r.get("collision", "regularised"), r["nx"], r["ny"], r["h"], r["z0"],
                r["smagorinsky"], r["spin_up_turnovers"], r["average_turnovers"])  # fmt: skip

    out["runs"] = [r for r in out["runs"] if key(r) != key(run)] + [run]
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
