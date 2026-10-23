"""Test flows with exact answers, in lattice units.

- `shear_wave`: u_x = U sin(k z) in a periodic box. The Navier-Stokes equations say it decays
  as exp(-nu k^2 t) and nothing else happens, so the measured decay rate gives the viscosity
  the scheme really has.
- `poiseuille`: air between two plates pushed by a body force g along x. The exact profile is a
  parabola, u(z) = g / (2 nu) (z - z0) (z1 - z), with the plates at z0 and z1. With halfway
  bounce-back the plates lie halfway between the last solid and the first air cell.
- `taylor_green`: the array of vortices u = U sin(kx) cos(ky), v = -U cos(kx) sin(ky), an exact
  solution of the Navier-Stokes equations that decays as exp(-2 nu k^2 t); its pressure is
  p = +rho U^2 (cos 2kx + cos 2ky) / 4, decaying twice as fast. (Books often write the minus
  sign: that belongs to the same vortices shifted by a quarter period, u = -U cos(kx) sin(ky).
  Here grad p = -rho (u . grad) u gives the plus sign.) In a 3D periodic box with no variation
  along z.

Each returns a dict of what it measured, for tests and for benchmarks/wind_exact.py.
"""

from __future__ import annotations

import math

import numpy as np

from .lattice import CS2, VelocitySet
from .solver import Params, Solver
from .units import viscosity_for_tau


def _relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum() / (b**2).sum()))


def shear_wave(
    lat: VelocitySet,
    n: int,
    tau: float = 0.8,
    u0: float = 0.01,
    regularised: bool = True,
    backend: str = "jax",
) -> dict:
    """Decay of a shear wave one box long (n cells): measured against expected viscosity.

    The rate is measured between a tenth of the decay time and the decay time, after the
    start-up of the non-equilibrium part of the packets.
    """
    nu = viscosity_for_tau(tau)
    k = 2 * math.pi / n
    t_decay = 1.0 / (nu * k * k)
    t1, t2 = round(0.1 * t_decay), round(t_decay)
    z = np.arange(n, dtype=float).reshape(n, 1, 1)
    s = Solver(lat, np.zeros((n, 1, 1), bool), Params(tau0=tau, regularised=regularised),
               backend, np.float64)  # fmt: skip
    zero = np.zeros((n, 1, 1))
    s.set_equilibrium(np.ones((n, 1, 1)), (u0 * np.sin(k * z), zero, zero))

    def amplitude() -> float:
        _, (ux, _, _) = s.fields()
        return float(2.0 / n * (ux * np.sin(k * z)).sum())

    s.run(t1)
    a1 = amplitude()
    s.run(t2 - t1)
    a2 = amplitude()
    nu_measured = math.log(a1 / a2) / ((t2 - t1) * k * k)
    return {
        "n": n,
        "tau": tau,
        "nu": nu,
        "nu_measured": nu_measured,
        "relative_error": abs(nu_measured - nu) / nu,
    }


def poiseuille(
    lat: VelocitySet,
    h: int,
    tau: float = 0.8,
    u_max: float = 0.01,
    regularised: bool = True,
    backend: str = "jax",
    tolerance: float = 1e-12,
    max_steps: int = 2_000_000,
) -> dict:
    """Steady flow between plates h cells apart, against the exact parabola.

    Starts from the exact profile and runs until the velocity changes by less than
    `tolerance` u_max in 1000 steps.
    """
    nu = viscosity_for_tau(tau)
    g = 8.0 * nu * u_max / h**2
    solid = np.zeros((h + 2, 1, 1), bool)
    solid[0] = solid[-1] = True
    z = np.arange(h + 2, dtype=float)
    exact = g / (2 * nu) * (z - 0.5) * (h + 0.5 - z)
    exact[0] = exact[-1] = 0.0
    shape = solid.shape
    s = Solver(lat, solid, Params(tau0=tau, g=(g, 0.0, 0.0), regularised=regularised),
               backend, np.float64)  # fmt: skip
    zero = np.zeros(shape)
    s.set_equilibrium(np.ones(shape), (exact.reshape(shape), zero, zero))
    previous = exact
    steps = 0
    while steps < max_steps:
        s.run(1000)
        steps += 1000
        _, (ux, _, _) = s.fields()
        u = ux[:, 0, 0]
        if np.abs(u - previous).max() < tolerance * u_max:
            break
        previous = u
    fluid = slice(1, h + 1)
    return {
        "h": h,
        "tau": tau,
        "steps": steps,
        "relative_error": _relative_l2(u[fluid], exact[fluid]),
        "max_error_over_u_max": float(np.abs(u[fluid] - exact[fluid]).max() / u_max),
    }


def taylor_green(
    lat: VelocitySet,
    n: int,
    tau: float = 0.8,
    u0: float = 0.04,
    regularised: bool = True,
    backend: str = "jax",
    settle: bool = True,
) -> dict:
    """The 2D Taylor-Green vortex, n cells per period, at the time its speed has halved.

    With `settle`, the start is made consistent first (Solver.set_flow), so that no sound
    waves are left to add an error that comes and goes with their phase.
    """
    nu = viscosity_for_tau(tau)
    k = 2 * math.pi / n
    steps = round(math.log(2.0) / (2 * nu * k * k))
    y, x = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float), indexing="ij")
    shape = (1, n, n)

    def exact(t: float):
        decay = math.exp(-2 * nu * k * k * t)
        ux = u0 * np.sin(k * x) * np.cos(k * y) * decay
        uy = -u0 * np.cos(k * x) * np.sin(k * y) * decay
        p = 0.25 * u0**2 * (np.cos(2 * k * x) + np.cos(2 * k * y)) * decay**2
        return 1.0 + p / CS2, ux, uy

    s = Solver(lat, np.zeros(shape, bool), Params(tau0=tau, regularised=regularised),
               backend, np.float64)  # fmt: skip
    rho0, ux0, uy0 = exact(0.0)
    u_start = (ux0.reshape(shape), uy0.reshape(shape), np.zeros(shape))
    settled = s.set_flow(rho0.reshape(shape), u_start, settle=settle)
    s.run(steps)
    _, (ux, uy, uz) = s.fields()
    _, ux_e, uy_e = exact(steps)
    num = np.stack([ux[0], uy[0], uz[0]])
    ref = np.stack([ux_e, uy_e, np.zeros_like(ux_e)])
    return {
        "n": n,
        "tau": tau,
        "u0": u0,
        "steps": steps,
        "settling_steps": settled,
        "relative_error": _relative_l2(num, ref),
    }


def observed_order(sizes: list[int], errors: list[float]) -> list[float]:
    """Order of convergence between successive grids: log(e1 / e2) / log(n2 / n1)."""
    return [
        math.log(errors[i] / errors[i + 1]) / math.log(sizes[i + 1] / sizes[i])
        for i in range(len(sizes) - 1)
    ]
