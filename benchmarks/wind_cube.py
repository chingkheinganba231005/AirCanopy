"""The flow around a cube on the floor of a channel, against the wind tunnel (wind/cube.py).

JOURNAL.md, Part 4 (the cube, planned on 2026-10-29). The experiment: Martinuzzi and Tropea
(1993), ERCOFTAC case 041. The setup follows Krajnović and Davidson (2002), with the changes
listed in wind/cube.py: D3Q27 with the recursive collision, Cs = 0.17, Re = 4e4 on the bulk
velocity U_b and the cube's height H, U_b = 0.05 cells per step, H = 32 cells (--height).

Times are in H / U_b, written T. Three stages:

1. The approach channel alone (the precursor, periodic), from the law of the wall with random
   gusts, for --spin-up T. Its bulk velocity over the last --calibrate T corrects the walls'
   roughness length once, z0 -> z0 exp(kappa (U_sim - U_b) / u*), the change that moves a
   log-law wind by the bulk velocity's error; the channel then runs --respin T more.
2. The site starts from copies of the channel laid end to end, with the cube put in, and runs
   fed by the channel for --settle T, so that it forgets how it started (the wind crosses it
   in 14 T).
3. Both run --average T more, sampled every T / 10: the channel's mean wind and turbulence by
   height; on the site's symmetry plane, the mean wind, and the sideways wind at three points
   behind the cube.

With --channel-only, the channel runs stage 3 alone, without the site.

Criteria, written down before the first run. For the approach flow, after the correction,
against the profile measured 18 H in front of the cube (x/H = -18.02 on the symmetry plane,
ERCOFTAC file x01z09_a):

A1. The channel's bulk velocity within 2 % of U_b: the correction worked.
A2. The mean wind, scaled by the channel's own bulk velocity, within 5 % of the measured at
    every measured height from 0.1 H to 1.9 H.

For the cube, from the mean wind in the first cell above the floor along the symmetry line
(cube.separation_lengths), against the lengths measured in the experiment (Iaccarino and
Durbin 2000, Table I; docs/sources.md):

C1. The separation in front of the cube, x_F / H, within 25 % of 1.040.
C2. The reattachment behind it, x_R / H, within 15 % of 1.612.

The tolerances are about the spread of the published large-eddy simulations (Krajnović and
Davidson 2002, Table 2: x_F / H from 0.81 to 1.29, x_R / H from 1.38 to 1.72). If the approach
flow fails A1 or A2, the cube's result is still recorded, but it does not count as a clean
test of the solver. Recorded without pass or fail: the simulated and measured mean wind at
every measured station on the symmetry plane from x/H = -3 to 8, among them 0.5, 1, 2 and 4;
the channel's turbulence against the measured (the simulation's resolved part only; the
measurements hold all of it); the Strouhal number f H / U_b of the wake's sideways sway, from
the sideways wind on the symmetry plane at x/H = 2.5 and 0.25, 0.5 and 0.75 H up (measured
0.145; Iaccarino and Durbin 2000, p. 268); and x_F and x_R from each half of the averaging,
for the share of the error that sampling makes. x is measured from the cube's front face.

    uv run python benchmarks/wind_cube.py --height 8 --channel-only   # checks the pipeline
    uv run python benchmarks/wind_cube.py --height 32                 # on a graphics card
    uv run python benchmarks/wind_cube.py --height 32 --u-bulk 0.025  # half the Mach number

Adds the run to results/wind_cube_<device>.json, replacing an earlier run with the same
settings: first when the roughness has been corrected, then when the averaging is done. The
file holds only simulated values and measures of the difference, not the measurements, whose
licence keeps them out of the repository (sources/ercoftac.py). If they cannot be downloaded
during a run, the run is recorded unjudged, and judged later from the file:

    uv run python benchmarks/wind_cube.py --judge results/wind_cube_gpu.json
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import time
from pathlib import Path

import jax
import numpy as np
from scipy import signal

from aircanopy import __version__
from aircanopy.sources import ercoftac
from aircanopy.wind import cube, lbm, walls
from aircanopy.wind.inflow import Precursor
from aircanopy.wind.lattice import LATTICES

ROOT = Path(__file__).resolve().parents[1]
BULK_TOLERANCE, PROFILE_TOLERANCE, PROFILE_HEIGHTS = 0.02, 0.05, (0.1, 1.9)
X_F, X_F_TOLERANCE, X_R, X_R_TOLERANCE = 1.040, 0.25, 1.612, 0.15
APPROACH_X, STATIONS_X = -18.02, (-3.0, 8.0)
PROBE_X, PROBE_HEIGHTS = 2.5, (0.25, 0.5, 0.75)


class Sums:
    """Running sums of dictionaries of arrays, for means over samples."""

    def __init__(self) -> None:
        self.sums: dict = {}
        self.n = 0

    def add(self, values: dict) -> None:
        for k, v in values.items():
            self.sums[k] = self.sums.get(k, 0.0) + np.asarray(v, dtype=np.float64)
        self.n += 1

    def mean(self) -> dict:
        return {k: v / self.n for k, v in self.sums.items()}


def heights(c: cube.Cube) -> np.ndarray:
    """The air layers' centres above the floor, in H."""
    return (np.arange(1, 2 * c.h + 1) - 0.5) / c.h


def start_channel(c: cube.Cube, pre) -> None:
    """The law of the wall from each wall to the middle, with random gusts (as
    boundary_layer.run)."""
    z = np.arange(2 * c.h + 1) - 0.5
    wall = np.minimum(z, 2 * c.h - z)
    law = np.where(z > 0, c.u_star / walls.KAPPA * np.log(np.maximum(wall, c.z0) / c.z0), 0.0)
    shape = pre.solid.shape
    profile = np.broadcast_to(law[:, None, None], shape)
    rng = np.random.default_rng(0)
    noise = [0.3 * profile * rng.standard_normal(shape) for _ in range(3)]
    pre.set_equilibrium(np.ones(shape), (profile + noise[0], noise[1], noise[2]))


def approach(c: cube.Cube, moments: dict) -> dict:
    """The channel's mean wind and turbulence by height (resolved part), scaled by its own bulk
    velocity, and criterion A1."""
    u = moments["u"][1:]
    bulk = float(u.mean())
    var_u = moments["uu"][1:] - u**2
    var_w = moments["ww"][1:] - moments["w"][1:] ** 2
    uw = moments["uw"][1:] - u * moments["w"][1:]
    bulk_ratio = bulk / c.u_bulk
    return {"bulk_velocity": bulk, "bulk_ratio": bulk_ratio, "u_star_over_bulk": c.u_star / bulk,
            "mean_u_over_bulk": [float(v) for v in u / bulk],
            "var_u_over_bulk2": [float(v) for v in var_u / bulk**2],
            "var_w_over_bulk2": [float(v) for v in var_w / bulk**2],
            "uw_over_bulk2": [float(v) for v in uw / bulk**2],
            "A1_bulk_within": BULK_TOLERANCE,
            "A1_passed": bool(abs(bulk_ratio - 1) <= BULK_TOLERANCE)}  # fmt: skip


def judge(c: cube.Cube, run: dict, measured: list) -> None:
    """Compare a run's simulated fields, as stored in the results, with the measurements:
    criterion A2, and the difference at every station on the symmetry plane. Also used on
    runs read back from the results file (--judge), when a run could not download them."""
    a = run["approach"]
    upstream = min((p for p in measured if p.z == 0.0), key=lambda p: abs(p.x - APPROACH_X))
    y = upstream["y"]
    ratio = np.interp(y, heights(c), a["mean_u_over_bulk"]) / upstream["U"]
    judged = (y >= PROFILE_HEIGHTS[0]) & (y <= PROFILE_HEIGHTS[1])
    worst = int(np.argmax(np.where(judged, np.abs(ratio - 1), -1.0)))
    a2 = bool(abs(ratio[worst] - 1) <= PROFILE_TOLERANCE)
    a |= {"station": {"x": upstream.x, "z": upstream.z}, "worst_ratio": float(ratio[worst]),
          "worst_at_y": float(y[worst]), "A2_profile_within": PROFILE_TOLERANCE,
          "A2_passed": a2, "passed": bool(a["A1_passed"] and a2)}  # fmt: skip
    if run.get("cube"):
        cu = run["cube"]
        solid = cube.site_solid(c)[:, c.symmetry_rows().start]
        u, w = np.asarray(cu["plane_u_over_bulk"]), np.asarray(cu["plane_w_over_bulk"])
        cu["stations"] = stations(c, solid, u, w, measured)
        cu["clean_test"] = a["passed"]
    run["judged"] = True


def column_profile(c: cube.Cube, solid: np.ndarray, field: np.ndarray, x: float, y) -> list:
    """The field on the symmetry plane at x (in H from the cube's front face) and heights y (in
    H), interpolated linearly across the two nearest columns, each over its own air cells."""
    position = c.cube_x()[0] + x * c.h - 0.5  # in column indices, whose centres are i + 0.5
    left = math.floor(position)
    share = position - left
    z = np.arange(solid.shape[0]) - 0.5
    out = 0.0
    for i, weight in ((left, 1 - share), (left + 1, share)):
        air = ~solid[:, i]
        out = out + weight * np.interp(np.asarray(y) * c.h, z[air], field[air, i])
    return [float(v) for v in out]


def stations(c: cube.Cube, solid, u, w, measured: list) -> list:
    """How far the simulated mean wind on the symmetry plane (u and w in U_b) is from the
    measured, at every measured station from x/H = -3 to 8: the root mean square of the
    difference, in U_b."""
    out = []
    for p in measured:
        if p.z != 0.0 or not STATIONS_X[0] <= p.x <= STATIONS_X[1]:
            continue
        sim_u = np.array(column_profile(c, solid, u, p.x, p["y"]))
        sim_w = np.array(column_profile(c, solid, w, p.x, p["y"]))
        out.append({"x": p.x, "rms_U": float(np.sqrt(np.mean((sim_u - p["U"]) ** 2))),
                    "rms_W": float(np.sqrt(np.mean((sim_w - p["V"]) ** 2)))})  # fmt: skip
    return out


def strouhal(c: cube.Cube, series: np.ndarray, every: int, bulk: float) -> dict:
    """The peak of the sideways wind's spectrum, averaged over the probes, as f H / U_b."""
    per_step = 1.0 / every
    segment = min(len(series), max(16, len(series) // 2))
    freq, power = signal.welch(series - series.mean(axis=0), fs=per_step, nperseg=segment,
                               axis=0)  # fmt: skip
    power = power.mean(axis=1)
    st = freq * c.h / bulk
    keep = (st > 0.02) & (st < 1.0)
    peak = float(st[keep][np.argmax(power[keep])]) if keep.any() else None
    return {"peak": peak, "st": [float(v) for v in st[keep]],
            "power": [float(v) for v in power[keep]], "segment_samples": segment}  # fmt: skip


def write(run: dict, device) -> Path:
    path = ROOT / "results" / f"wind_cube_{device.platform}.json"
    old = json.loads(path.read_text()) if path.is_file() else {"runs": []}

    def key(r):
        return tuple(r["settings"][k] for k in sorted(r["settings"]))

    out = {"what": "the flow around a cube on the floor of a channel, against the wind tunnel",
           "runs": [r for r in old["runs"] if key(r) != key(run)] + [run]}  # fmt: skip
    path.write_text(json.dumps(out, indent=1) + "\n")
    return path


def measurements() -> list | None:
    """The measured profiles, or None if they cannot be downloaded now (the comparison can
    then be made later from the results file, with --judge)."""
    try:
        return ercoftac.profiles("a")
    except (OSError, ValueError) as error:  # no network, the server, or a changed file
        print("the measurements could not be read:", error)
        return None


def judge_file(path: Path) -> None:
    """Judge every run in a results file that could not be judged when it ran."""
    measured = ercoftac.profiles("a")
    data = json.loads(path.read_text())
    for run in data["runs"]:
        if not run.get("judged") and run.get("approach"):
            judge(cube.Cube(h=run["settings"]["h"], u_bulk=run["settings"]["u_bulk"]), run,
                  measured)  # fmt: skip
            print("judged", run["settings"], run["approach"]["passed"])
    path.write_text(json.dumps(data, indent=1) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lattice", default="D3Q27", choices=sorted(LATTICES))
    parser.add_argument("--collision", choices=lbm.COLLISIONS)
    parser.add_argument("--height", type=int, default=32, help="cells across the cube")
    parser.add_argument("--u-bulk", type=float, default=0.05)
    parser.add_argument("--cs", type=float, default=0.17)
    parser.add_argument("--spin-up", type=float, default=200.0)
    parser.add_argument("--calibrate", type=float, default=40.0)
    parser.add_argument("--respin", type=float, default=60.0)
    parser.add_argument("--settle", type=float, default=20.0)
    parser.add_argument("--average", type=float, default=200.0)
    parser.add_argument("--channel-only", action="store_true")
    parser.add_argument("--judge", type=Path, help="judge the runs of this results file, only")
    args = parser.parse_args()
    if args.judge:
        judge_file(args.judge)
        return
    lat = LATTICES[args.lattice]
    collision = args.collision or ("recursive" if lat.name == "D3Q27" else "regularised")
    t0 = time.perf_counter()
    measured = measurements()
    c = cube.Cube(h=args.height, u_bulk=args.u_bulk, smagorinsky=args.cs)
    period = round(c.h / c.u_bulk)  # T in steps
    every = max(1, round(period / 10))
    device = jax.devices()[0]
    settings = {"lattice": lat.name, "collision": collision, "h": c.h, "u_bulk": c.u_bulk,
                "smagorinsky": c.smagorinsky, "spin_up": args.spin_up,
                "calibrate": args.calibrate, "respin": args.respin, "settle": args.settle,
                "average": args.average, "channel_only": args.channel_only}  # fmt: skip
    run = {"run_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "aircanopy": __version__, "jax": jax.__version__,
           "device": {"platform": device.platform, "kind": device.device_kind},
           "settings": settings, "period_steps": period, "sample_every_steps": every,
           "tau0": c.tau0, "g": c.g, "match_cells": c.match_cells}  # fmt: skip

    # 1. The approach channel alone, then one correction of the roughness length.
    pre = cube.precursor_box(c, lat, collision=collision)
    start_channel(c, pre)
    bulk_history = []
    calibration = Sums()
    while pre.time < round(args.spin_up * period):
        pre.run(period)
        bulk_history.append(float(pre.layer_moments()["u"][1:].mean()))
        if pre.time > round((args.spin_up - args.calibrate) * period):
            calibration.add({"u": bulk_history[-1]})
    before = float(calibration.mean()["u"])
    z0 = c.z0 * math.exp(walls.KAPPA * (before - c.u_bulk) / c.u_star)
    run["calibration"] = {"z0_estimate_cells": c.z0, "bulk_before": before,
                          "z0_cells": z0}  # fmt: skip
    run["channel_bulk_every_period"] = bulk_history
    print("calibration:", run["calibration"], "wrote", write(run, device))
    c = dataclasses.replace(c, z0_over_h=z0 / c.h)
    corrected = cube.precursor_box(c, lat, collision=collision)
    corrected.f, corrected.time = pre.f, pre.time
    pre = corrected
    profile = Sums()
    while pre.time < round((args.spin_up + args.respin) * period):
        pre.run(period)
        layers = pre.layer_moments()
        bulk_history.append(float(layers["u"][1:].mean()))
        profile.add({"u": layers["u"]})
    mean_wind = profile.mean()["u"]

    # 2. The site, started from copies of the channel with the cube put in.
    site = both = None
    if not args.channel_only:
        site = cube.site_box(c, lat, mean_wind, collision=collision)
        copies = -(-site.solid.shape[2] // pre.solid.shape[2])
        site.f = site.xp.tile(pre.f, (1, 1, 1, copies))[..., : site.solid.shape[2]]
        both = Precursor(pre, site)
        while site.time < round(args.settle * period):
            both.run(period)
            bulk_history.append(float(pre.layer_moments()["u"][1:].mean()))

    # 3. Sampling.
    channel = Sums()
    plane, halves = Sums(), [Sums(), Sums()]
    rows = c.symmetry_rows()
    probe_x = c.cube_x()[0] + round(PROBE_X * c.h)
    probe_z = [int(y * c.h) + 1 for y in PROBE_HEIGHTS]  # the layer whose centre is just above
    probes = []
    n_samples = round(args.average * 10)
    for k in range(n_samples):
        (both or pre).run(every)
        layers = pre.layer_moments()
        channel.add(layers)
        if k % 10 == 9:
            bulk_history.append(float(layers["u"][1:].mean()))
        if site is not None:
            _, (u, v, w) = site.fields(rows)
            u, v, w = u.mean(axis=1), v.mean(axis=1), w.mean(axis=1)  # the plane between
            plane.add({"u": u, "w": w})
            halves[2 * k // n_samples].add({"floor": u[1]})
            probes.append([v[z, probe_x] for z in probe_z])
    run["approach"] = approach(c, channel.mean())
    run["channel_bulk_every_period"] = bulk_history
    run["cube"] = None
    if site is not None:
        bulk = run["approach"]["bulk_velocity"]
        mean = plane.mean()
        lengths = cube.separation_lengths(c, mean["u"][1])
        x_f, x_r = lengths["x_F"], lengths["x_R"]
        c1 = x_f is not None and abs(x_f / X_F - 1) <= X_F_TOLERANCE
        c2 = x_r is not None and abs(x_r / X_R - 1) <= X_R_TOLERANCE
        run["cube"] = {
            "x_F": x_f, "x_R": x_r,
            "halves": [cube.separation_lengths(c, half.mean()["floor"]) for half in halves],
            "measured": {"x_F": X_F, "x_R": X_R},
            "C1_x_F_within": X_F_TOLERANCE, "C2_x_R_within": X_R_TOLERANCE,
            "C1_passed": bool(c1), "C2_passed": bool(c2), "passed": bool(c1 and c2),
            "strouhal": strouhal(c, np.asarray(probes), every, bulk)
                        | {"x": PROBE_X, "heights": list(PROBE_HEIGHTS), "measured": 0.145},
            "floor_u_over_bulk": [round(float(v) / bulk, 5) for v in mean["u"][1]],
            "plane_u_over_bulk": [[round(float(v) / bulk, 4) for v in row]
                                  for row in mean["u"]],
            "plane_w_over_bulk": [[round(float(v) / bulk, 4) for v in row]
                                  for row in mean["w"]],
        }  # fmt: skip
    measured = measured or measurements()
    if measured is None:
        run["judged"] = False
    else:
        judge(c, run, measured)
    run["seconds"] = round(time.perf_counter() - t0, 1)
    shown = ("bulk_ratio", "worst_ratio", "passed")
    print("approach:", {k: run["approach"].get(k) for k in shown})
    if run["cube"]:
        shown = ("x_F", "x_R", "passed", "clean_test")
        print("cube:", {k: run["cube"].get(k) for k in shown},
              "St:", run["cube"]["strouhal"]["peak"])  # fmt: skip
    print("seconds:", run["seconds"], "wrote", write(run, device))


if __name__ == "__main__":
    main()
