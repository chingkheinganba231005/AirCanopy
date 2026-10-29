"""The boundary layer through an open box, fed from a periodic precursor (wind/inflow.py).

Step 14 of JOURNAL.md. The precursor is the boundary layer of Step 12 that passed both of its
criteria: D3Q27 with the recursive collision, H = 32 cells, a box of 2H x H x H, Cs = 0.17,
the wall model reading at 0.2 delta (6.5 cells, --match 6). The site is an open box with the
same cross-section, floor and push (body force), 8H long, fed at x = 0 from the precursor's
plane x = 0, with an absorbing layer of H columns at strength 0.2 before the outlet; its
target is the precursor's mean wind at the end of the precursor's spin-up.

The precursor first runs alone for 10 turnovers (H / u*). The site then starts from four copies
of the precursor laid end to end, and both run together for 2 turnovers, so that the site
forgets how it started (the wind crosses the site in less than a turnover), then for 10 more
turnovers, sampled 10 times per turnover. The site is cut into slabs H long. The last slab is
the absorbing layer, recorded but not judged.

Criteria, written down before the first run:

1. Mean wind: in every slab before the absorbing layer, at every height, within 3 % of the
   precursor's mean wind over the same time.
2. Turbulence: in every such slab, the variance of u and the variance of w, each averaged over
   the column, within 15 % of the precursor's.
3. Friction: in every such slab, the floor's mean shear stress within 10 % of the precursor's.

For scale, the same measures are recorded for the two halves of the precursor (two slabs H
long) against the whole precursor: the difference that sampling alone makes.

--coupling chooses how the precursor's plane reaches the inlet (wind/inflow.py). The first run,
with "velocity", failed criteria 1 and 2 near the inlet (JOURNAL.md, Step 14); "packets" is
the default since. On a graphics card the same test also runs at H = 64, the wall model
reading at 0.2 delta = 12.8 cells, at the layer centre 12.5 (--match 12), with the same
criteria (scripts/gpu_jobs.txt).

    uv run python benchmarks/wind_abl_open.py        # about two hours on the container's processor
    uv run python benchmarks/wind_abl_open.py --height 16 --match 3 --coupling velocity

Adds the run to results/wind_abl_open_<device>.json, replacing an earlier run with the same
settings.
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
from aircanopy.wind import boundary_layer, lbm, walls
from aircanopy.wind.inflow import COUPLINGS, Precursor
from aircanopy.wind.lattice import LATTICES

ROOT = Path(__file__).resolve().parents[1]
MEAN_TOLERANCE, VARIANCE_TOLERANCE, STRESS_TOLERANCE = 0.03, 0.15, 0.10
U_STAR, Z0 = 0.005, 0.05


class Slab:
    """Sums over the samples for one slab of a box: the mean wind by height, the variances of u
    (about each sample's layer mean) and w, and the floor's shear stress."""

    def __init__(self, h: int) -> None:
        self.u, self.uu, self.ww = np.zeros(h + 1), np.zeros(h + 1), np.zeros(h + 1)
        self.stress, self.n = 0.0, 0

    def add(self, rho, u, lam: float, read: int) -> None:
        ux, uy, uz = u
        layer = ux.mean(axis=(1, 2))
        du = ux - layer[:, None, None]
        self.u += layer
        self.uu += (du * du).mean(axis=(1, 2))
        self.ww += (uz * uz).mean(axis=(1, 2))
        speed = np.sqrt(ux[read] ** 2 + uy[read] ** 2 + uz[read] ** 2)
        self.stress += float((0.5 * lam * rho[1] * speed * ux[read]).mean())
        self.n += 1

    def result(self) -> dict:
        n = self.n
        return {"mean_u": [float(v) for v in self.u[1:] / n],
                "variance_u": [float(v) for v in self.uu[1:] / n],
                "variance_w": [float(v) for v in self.ww[1:] / n],
                "floor_stress": self.stress / n}  # fmt: skip


def compare(slab: dict, ref: dict) -> dict:
    """The criteria's measures for one slab against the reference (the whole precursor)."""
    ratio = np.asarray(slab["mean_u"]) / np.asarray(ref["mean_u"])
    var_u = np.mean(slab["variance_u"]) / np.mean(ref["variance_u"])
    var_w = np.mean(slab["variance_w"]) / np.mean(ref["variance_w"])
    stress = slab["floor_stress"] / ref["floor_stress"]
    worst = float(ratio[np.argmax(np.abs(ratio - 1))])
    return {"worst_mean_ratio": worst, "variance_u_ratio": float(var_u),
            "variance_w_ratio": float(var_w), "floor_stress_ratio": float(stress),
            "pass": bool(abs(worst - 1) <= MEAN_TOLERANCE
                         and abs(var_u - 1) <= VARIANCE_TOLERANCE
                         and abs(var_w - 1) <= VARIANCE_TOLERANCE
                         and abs(stress - 1) <= STRESS_TOLERANCE)}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lattice", default="D3Q27", choices=sorted(LATTICES))
    parser.add_argument("--collision", choices=lbm.COLLISIONS)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--length", type=int, default=8, help="the site's length, in H")
    parser.add_argument("--cs", type=float, default=0.17)
    parser.add_argument("--match", type=int, default=6)
    parser.add_argument("--absorb-strength", type=float, default=0.2)
    parser.add_argument("--spin-up", type=float, default=10.0)
    parser.add_argument("--settle", type=float, default=2.0)
    parser.add_argument("--average", type=float, default=10.0)
    parser.add_argument("--coupling", default="packets", choices=COUPLINGS)
    args = parser.parse_args()
    lat = LATTICES[args.lattice]
    collision = args.collision or ("recursive" if lat.name == "D3Q27" else "regularised")
    h, n_slabs = args.height, args.length
    if n_slabs % 2:
        raise SystemExit("--length must be even: the site starts from copies of the 2H precursor")
    t0 = time.perf_counter()
    common = dict(z0=Z0, u_star=U_STAR, smagorinsky=args.cs, collision=collision,
                  match_cells=args.match)  # fmt: skip
    pre = boundary_layer.box(lat, 2 * h, h, h, **common)
    # The same start as boundary_layer.run: the law of the wall with random gusts.
    z = np.arange(h + 1) - 0.5
    log_law = np.where(z > 0, U_STAR / walls.KAPPA * np.log(np.maximum(z, Z0) / Z0), 0.0)
    shape = pre.solid.shape
    profile = np.broadcast_to(log_law[:, None, None], shape)
    rng = np.random.default_rng(0)
    noise = [0.3 * profile * rng.standard_normal(shape) for _ in range(3)]
    pre.set_equilibrium(np.ones(shape), (profile + noise[0], noise[1], noise[2]))
    turnover = round(h / U_STAR)
    column_u = []
    while pre.time < round(args.spin_up * turnover):
        pre.run(turnover)
        column_u.append(float(pre.fields()[1][0][1:].mean() / U_STAR))
    mean_wind = pre.fields()[1][0].mean(axis=(1, 2))
    inlet = (np.broadcast_to(mean_wind[:, None], (h + 1, h)), 0.0, 0.0)
    site = boundary_layer.box(lat, n_slabs * h, h, h, **common, inlet_u=inlet, absorb_cells=h,
                              absorb_strength=args.absorb_strength)  # fmt: skip
    site.f = site.xp.tile(pre.f, (1, 1, 1, n_slabs // 2))
    both = Precursor(pre, site, coupling=args.coupling)
    both.run(round(args.settle * turnover))
    read = 1 + args.match
    lam = 2 * walls.drag_coefficient(Z0, height=read - 0.5)
    whole, halves = Slab(h), [Slab(h), Slab(h)]
    slabs = [Slab(h) for _ in range(n_slabs)]
    every = max(1, round(turnover / 10))
    for _ in range(round(args.average * 10)):
        both.run(every)
        rho, u = pre.fields()
        whole.add(rho, u, lam, read)
        column_u.append(float(u[0][1:].mean() / U_STAR))
        for k, half in enumerate(halves):
            cut = slice(k * h, (k + 1) * h)
            half.add(rho[..., cut], [c[..., cut] for c in u], lam, read)
        rho, u = site.fields()
        for k, slab in enumerate(slabs):
            cut = slice(k * h, (k + 1) * h)
            slab.add(rho[..., cut], [c[..., cut] for c in u], lam, read)
    ref = whole.result()
    site_slabs = [s.result() | compare(s.result(), ref) for s in slabs]
    judged = site_slabs[:-1]  # the last slab is the absorbing layer
    device = jax.devices()[0]
    run = {
        "run_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "aircanopy": __version__,
        "jax": jax.__version__,
        "device": {"platform": device.platform, "kind": device.device_kind},
        "lattice": lat.name,
        "collision": collision,
        "coupling": args.coupling,
        "h": h,
        "precursor_cells": [h + 1, h, 2 * h],
        "site_cells": [h + 1, h, n_slabs * h],
        "smagorinsky": args.cs,
        "match_cells": args.match,
        "z0": Z0,
        "u_star": U_STAR,
        "absorb_cells": h,
        "absorb_strength": args.absorb_strength,
        "turnover_steps": turnover,
        "spin_up_turnovers": args.spin_up,
        "settle_turnovers": args.settle,
        "average_turnovers": args.average,
        "z": [float(v) for v in z[1:]],
        "precursor_column_u_over_u_star": column_u,
        "precursor": ref,
        "precursor_halves": [s.result() | compare(s.result(), ref) for s in halves],
        "site_slabs": site_slabs,
        "criteria": {
            "1_mean_wind_within": MEAN_TOLERANCE,
            "2_variances_within": VARIANCE_TOLERANCE,
            "3_floor_stress_within": STRESS_TOLERANCE,
            "passed": all(s["pass"] for s in judged),
        },
        "seconds": round(time.perf_counter() - t0, 1),
    }
    shown = ("worst_mean_ratio", "variance_u_ratio", "variance_w_ratio", "floor_stress_ratio",
             "pass")  # fmt: skip
    for k, slab in enumerate(site_slabs):
        print(k, {name: slab[name] for name in shown})
    print("passed:", run["criteria"]["passed"], "seconds:", run["seconds"])
    path = ROOT / "results" / f"wind_abl_open_{device.platform}.json"
    old = json.loads(path.read_text()) if path.is_file() else {"runs": []}
    if "runs" not in old:  # the first run (Step 14) was the whole file, with velocity coupling
        old = {"runs": [{"coupling": "velocity", **old}]}

    def key(r):
        return (r["lattice"], r["collision"], r["coupling"], r["h"], r["site_cells"][2],
                r["smagorinsky"], r["match_cells"], r["absorb_strength"], r["spin_up_turnovers"],
                r["settle_turnovers"], r["average_turnovers"])  # fmt: skip

    out = {"what": "the boundary layer through an open box fed from a periodic precursor",
           "runs": [r for r in old["runs"] if key(r) != key(run)] + [run]}  # fmt: skip
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
