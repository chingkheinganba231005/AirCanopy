"""Open boundaries: a velocity inlet, a pressure outlet, and the absorbing layer before it.

Step 13 of JOURNAL.md. On the processor, in float64, for D3Q27 with the recursive collision
(the default, D-018) and D3Q19 with the regularised one, unless said otherwise:

1. A uniform wind at an angle, u = (0.05, 0.01, -0.02), through a box of 4 x 5 x 40 cells open
   along x, for 200 steps, with no absorbing layer and with one of 10 columns at strength 0.2;
   every collision of the solver. Nothing should change.
2. A channel driven by a body force: h = 8, 16 and 32 cells of air between two walls, 4h cells
   long, fed at the inlet with the developed profile of the same channel made periodic. tau =
   0.8; peak speed 0.4 / h, so that the Mach number falls with the cell size as the viscosity
   in lattice units does (diffusive scaling). 5 h^2 / nu steps. The open boundaries should keep
   the profile; the largest change is recorded in each column, as a share of the peak speed.
3. The same channel without a force, driven by the pressure drop between an inlet that holds
   Poiseuille's parabola and an outlet at density 1, for 5 h^2 / nu steps. Recorded: the mass
   flux through each section; the slope of the density along the middle half of the channel,
   against Poiseuille's law dp/dx = -12 nu m / h^2, with m the measured mean mass flux and p =
   c_s^2 rho; and the largest difference between the shape of the mass flux profile and the
   parabola.
4. A pulse of sound, density 1 + 1e-3 exp(-(x - 48)^2 / 18), in a wind of 0.05 along a box of
   96 cells, with tau0 = 0.5001: the share of its energy (c_s^2 (rho - 1)^2 + rho |u - u0|^2,
   summed) left in the box after 100 to 1600 steps, with no absorbing layer and with layers of
   8, 16 and 32 columns at strengths 0.05 and 0.2.

Criteria, written down before this benchmark's first run, after exploratory runs of tests 2
to 4 at h = 8 in a scratch folder:

1. Every collision, with and without the layer: density and velocity unchanged to 1e-12.
2. At every size: the profile changes by less than 1 % of the peak speed anywhere, and by less
   than 0.1 % in the middle third of the channel.
3. At every size: the mass flux is the same through every section to 0.1 %, and the slope is
   within 2 % of Poiseuille's law.
4. With a layer of 16 columns at strength 0.2, less than 1e-3 of the energy is left after 400
   steps, and at least 100 times less than without a layer (so that the test can tell).

    uv run python benchmarks/wind_open.py

Writes results/wind_open_cpu.json.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import jax
import numpy as np

from aircanopy import __version__
from aircanopy.wind.lattice import CS2, D3Q19, D3Q27
from aircanopy.wind.solver import Params, Solver

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = ((D3Q27, "recursive"), (D3Q19, "regularised"))
PAIRS = ((D3Q19, "regularised"), (D3Q19, "bgk"), (D3Q27, "recursive"), (D3Q27, "regularised"),
         (D3Q27, "bgk"))  # fmt: skip
SIZES = (8, 16, 32)
TAU = 0.8
NU = CS2 * (TAU - 0.5)
PULSE_CHECKS = (100, 200, 400, 800, 1600)


def uniform(lat, collision: str, layer: int) -> dict:
    shape, u0 = (4, 5, 40), (0.05, 0.01, -0.02)
    params = Params(tau0=0.6, collision=collision, open_x=True, absorb_cells=layer,
                    absorb_strength=0.2 if layer else 0.0)  # fmt: skip
    s = Solver(lat, np.zeros(shape, bool), params, "numpy", np.float64, inlet_u=u0)
    s.set_equilibrium(np.ones(shape), tuple(np.full(shape, c) for c in u0))
    s.run(200)
    rho, u = s.fields()
    r = {"lattice": lat.name, "collision": collision, "layer": layer, "steps": 200,
         "max_rho_error": float(np.abs(rho - 1).max()),
         "max_u_error": float(max(np.abs(u[a] - u0[a]).max() for a in range(3)))}  # fmt: skip
    r["pass"] = r["max_rho_error"] < 1e-12 and r["max_u_error"] < 1e-12
    return r


def channel(h: int, nx: int) -> np.ndarray:
    solid = np.zeros((h + 2, 1, nx), bool)
    solid[0] = solid[-1] = True
    return solid


def parabola(h: int, u_max: float) -> np.ndarray:
    z = np.arange(h + 2) - 0.5
    return np.where((z > 0) & (z < h), 4.0 * u_max * z * (h - z) / h**2, 0.0)


def start(s: Solver, profile: np.ndarray) -> None:
    shape = s.solid.shape
    s.set_equilibrium(np.ones(shape), (np.broadcast_to(profile[..., None], shape),
                                       np.zeros(shape), np.zeros(shape)))  # fmt: skip


def forced_channel(lat, collision: str, h: int) -> dict:
    nx, u_max = 4 * h, 0.4 / h
    g = (u_max * 8.0 * NU / h**2, 0.0, 0.0)
    steps = int(5 * h**2 / NU)
    periodic = Solver(lat, channel(h, 1), Params(tau0=TAU, g=g, collision=collision), "jax",
                      np.float64)  # fmt: skip
    start(periodic, parabola(h, u_max)[:, None])
    periodic.run(2 * steps)
    profile = periodic.fields()[1][0][:, :, 0]
    t0 = time.perf_counter()
    params = Params(tau0=TAU, g=g, collision=collision, open_x=True)
    s = Solver(lat, channel(h, nx), params, "jax", np.float64, inlet_u=(profile, 0.0, 0.0))
    start(s, profile)
    s.run(steps)
    rho, u = s.fields()
    change = np.abs(u[0] - profile[..., None])[1:-1].max(axis=(0, 1)) / u_max
    middle = change[nx // 3 : nx - nx // 3]
    r = {"lattice": lat.name, "collision": collision, "h": h, "nx": nx, "tau": TAU,
         "u_max": u_max, "g": g[0], "steps": steps,
         "max_change_share": float(change.max()), "middle_change_share": float(middle.max()),
         "change_share_by_column": [float(c) for c in change],
         "max_rho_deviation": float(np.abs(rho - 1)[1:-1].max()),
         "max_uz_share": float(np.abs(u[2]).max() / u_max),
         "seconds": round(time.perf_counter() - t0, 1)}  # fmt: skip
    r["pass"] = r["max_change_share"] < 0.01 and r["middle_change_share"] < 0.001
    return r


def pressure_channel(lat, collision: str, h: int) -> dict:
    nx, u_max = 4 * h, 0.4 / h
    steps = int(5 * h**2 / NU)
    inlet = parabola(h, u_max)[:, None]
    t0 = time.perf_counter()
    s = Solver(lat, channel(h, nx), Params(tau0=TAU, collision=collision, open_x=True), "jax",
               np.float64, inlet_u=(inlet, 0.0, 0.0))  # fmt: skip
    start(s, inlet)
    spread = {}
    for at in (steps // 2, steps):
        s.run(at - s.time)
        rho, u = s.fields()
        flux = (rho * u[0])[1:-1, 0].sum(axis=0)
        spread[at] = float(flux.max() / flux.min() - 1.0)
    middle = slice(nx // 4, nx - nx // 4)
    slope = float(np.polyfit(np.arange(nx)[middle], rho[1:-1, 0, middle].mean(axis=0), 1)[0])
    m = float(flux[middle].mean() / h)
    expected = -12.0 * NU * m / h**2 / CS2
    shape = (rho * u[0])[1:-1, 0] / flux[None, :]
    exact = parabola(h, 1.0)[1:-1]
    exact = exact / exact.sum()
    r = {"lattice": lat.name, "collision": collision, "h": h, "nx": nx, "tau": TAU,
         "u_max": u_max, "steps": steps,
         "flux_spread": spread[steps], "flux_spread_halfway": spread[steps // 2],
         "slope": slope, "poiseuille_slope": expected, "slope_ratio": slope / expected,
         "inlet_rho_mean": float(rho[1:-1, 0, 0].mean()),
         "outlet_rho_mean": float(rho[1:-1, 0, -1].mean()),
         "profile_shape_error": float(np.abs(shape - exact[:, None]).max() / exact.max()),
         "seconds": round(time.perf_counter() - t0, 1)}  # fmt: skip
    r["pass"] = r["flux_spread"] < 1e-3 and abs(r["slope_ratio"] - 1.0) < 0.02
    return r


def pulse(lat, collision: str, layer: int, strength: float) -> dict:
    nx, u0 = 96, 0.05
    shape = (1, 1, nx)
    x = np.arange(nx)
    rho0 = 1.0 + 1e-3 * np.exp(-((x - 48.0) ** 2) / 18.0) + np.zeros(shape)
    params = Params(tau0=0.5001, collision=collision, open_x=True, absorb_cells=layer,
                    absorb_strength=strength)  # fmt: skip
    s = Solver(lat, np.zeros(shape, bool), params, "numpy", np.float64, inlet_u=(u0, 0, 0))
    s.set_equilibrium(rho0, (np.full(shape, u0), np.zeros(shape), np.zeros(shape)))

    def energy() -> float:
        rho, u = s.fields()
        e = CS2 * (rho - 1) ** 2 + rho * ((u[0] - u0) ** 2 + u[1] ** 2 + u[2] ** 2)
        return float(e.sum())

    e0, left = energy(), {}
    for at in PULSE_CHECKS:
        s.run(at - s.time)
        left[str(at)] = energy() / e0
    return {"lattice": lat.name, "collision": collision, "layer": layer, "strength": strength,
            "energy_left": left}  # fmt: skip


def main() -> None:
    device = jax.devices()[0]
    out = {
        "what": "open boundaries: uniform wind, forced and pressure-driven channels, a sound "
        "pulse and the absorbing layer",
        "run_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "aircanopy": __version__,
        "jax": jax.__version__,
        "device": {"platform": device.platform, "kind": device.device_kind},
        "dtype": "float64",
    }
    out["uniform"] = [uniform(lat, c, layer) for lat, c in PAIRS for layer in (0, 10)]
    for r in out["uniform"]:
        print(r)
    out["forced_channel"], out["pressure_channel"] = [], []
    for lat, c in DEFAULTS:
        for h in SIZES:
            for name, case in (("forced_channel", forced_channel),
                               ("pressure_channel", pressure_channel)):  # fmt: skip
                r = case(lat, c, h)
                out[name].append(r)
                print(name, {k: v for k, v in r.items() if k != "change_share_by_column"})
    out["pulse"] = []
    for lat, c in DEFAULTS:
        for layer, strength in ((0, 0.0), (8, 0.05), (8, 0.2), (16, 0.05), (16, 0.2),
                                (32, 0.05), (32, 0.2)):  # fmt: skip
            r = pulse(lat, c, layer, strength)
            out["pulse"].append(r)
            print(r)
    checks = []
    for lat, c in DEFAULTS:
        runs = [r for r in out["pulse"] if r["lattice"] == lat.name]
        none = next(r for r in runs if r["layer"] == 0)["energy_left"]["400"]
        best = next(r for r in runs if r["layer"] == 16 and r["strength"] == 0.2)
        best = best["energy_left"]["400"]
        checks.append({"lattice": lat.name, "collision": c, "left_400_without": none,
                       "left_400_with_16_at_0.2": best,
                       "pass": best < 1e-3 and best * 100 < none})  # fmt: skip
    out["pulse_criterion"] = checks
    out["passed"] = {
        "1_uniform": all(r["pass"] for r in out["uniform"]),
        "2_forced_channel": all(r["pass"] for r in out["forced_channel"]),
        "3_pressure_channel": all(r["pass"] for r in out["pressure_channel"]),
        "4_pulse": all(r["pass"] for r in checks),
    }
    print(out["passed"])
    path = ROOT / "results" / f"wind_open_{device.platform}.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
