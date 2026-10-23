"""Noise in a fast wind with almost no viscosity: which collision keeps it from growing.

Step 12 of JOURNAL.md. A periodic box of 32^3 cells holds a uniform wind at a given Mach
number, along x, along the diagonal of the xy plane or along the diagonal of the box, plus
random noise of 1e-4 (in lattice units) in each component of the velocity. The molecular
viscosity is 1e-6 (tau0 = 0.500003), as for air on metre-sized cells, and the Smagorinsky
model is off, so nothing but the collision itself can damp the noise. Every 2000 steps, for
20 000 steps, the kinetic energy of the noise (the variance of the velocity about its mean) is
recorded as a share of its starting value.

Criterion, written down before this benchmark's first run, after exploratory runs of the same
test: a collision is stable in a case if the run stays finite and the noise energy stays below
its starting value at every check. A finite run cannot tell slow growth from fluctuation, so
every value is recorded.

The collisions of the solver (lbm.COLLISIONS), and four candidates that were tried and not
adopted, written here from Coreixas, Chopard and Latt (2019, arXiv 1904.12948):

- "bgk_extended": BGK with the extended (product) equilibrium of D3Q27;
- "regularised_extended": the regularised collision with the extended equilibrium, its
  higher-order non-equilibrium parts set to zero ("REG" in their naming);
- "recursive_trace_free": the recursive collision built from the trace-free stress, so that
  the bulk stress relaxes fully, as in the regularised collision (Dellar 2001);
- "recursive" on D3Q19: populations from the 19 raw moments of their basis B_Q19 (their
  Sec. VI B and App. G, Eq. G7), with the Hermite coefficients of the recursive collision.

    uv run python benchmarks/wind_stability.py

Writes results/wind_stability_<device>.json.
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from aircanopy import __version__
from aircanopy.wind import lbm
from aircanopy.wind.lattice import CS2, D3Q19, D3Q27

ROOT = Path(__file__).resolve().parents[1]
N = 32
STEPS = 20_000
CHECKS = 10
NOISE = 1e-4
TAU0 = 0.500003
DIRECTIONS = {"x": (1, 0, 0), "xy": (1, 1, 0), "xyz": (1, 1, 1)}

# The raw moments D3Q19 holds (Coreixas et al. 2019, Sec. VI B): those of D3Q27 without
# xyz, x^2yz, xy^2z, xyz^2, x^2y^2z, x^2yz^2, xy^2z^2 and x^2y^2z^2.
Q19_ORDERS = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 0), (0, 2, 0), (0, 0, 2),
              (1, 1, 0), (1, 0, 1), (0, 1, 1), (2, 1, 0), (1, 2, 0), (2, 0, 1), (1, 0, 2),
              (0, 2, 1), (0, 1, 2), (2, 2, 0), (2, 0, 2), (0, 2, 2))  # fmt: skip


def hermite_to_populations(lat) -> tuple[tuple, np.ndarray]:
    """For a lattice's raw-moment orders, the matrix taking Hermite coefficients to packets:
    raw moments from Hermite ones (xi = H1, xi^2 = H2 + c_s^2 H0, multiplied out), then packets
    from raw moments by inverting the lattice's moment matrix."""
    orders = Q19_ORDERS if lat.q == 19 else tuple(itertools.product(range(3), repeat=3))
    c = lat.c.astype(float)
    raw = np.array([[np.prod(c[i] ** np.array(o)) for i in range(lat.q)] for o in orders])
    one_d = {0: ((0, 1.0),), 1: ((1, 1.0),), 2: ((2, 1.0), (0, CS2))}
    expand = np.zeros((len(orders), len(orders)))
    for row, o in enumerate(orders):
        for (a, ka), (b, kb), (d, kd) in itertools.product(*(one_d[k] for k in o)):
            expand[row, orders.index((a, b, d))] += ka * kb * kd
    return orders, np.linalg.solve(raw, expand)


def candidate(lat, kind: str):
    """A collision f -> f_post (no force, no drag, no turbulence model) for a candidate."""
    orders, build = hermite_to_populations(lat)
    m_mat, _ = lbm.moment_matrices(lat)

    def collide(f):
        cells = f.shape[1:]
        m = (jnp.asarray(m_mat, f.dtype) @ f.reshape(lat.q, -1)).reshape(10, *cells)
        rho = m[0]
        u = [m[1] / rho, m[2] / rho, m[3] / rho]
        p = [m[4 + k] - rho * u[a] * u[b] - (rho * CS2 if a == b else 0.0)
             for k, (a, b) in enumerate(lbm.PAIRS)]  # fmt: skip
        if kind == "recursive_trace_free":
            third = (p[0] + p[1] + p[2]) / 3.0
            p = [p[k] - third if k < 3 else p[k] for k in range(6)]
        omega = 1.0 - 1.0 / TAU0
        coeffs = []
        for o in orders:
            a = rho * u[0] ** o[0] * u[1] ** o[1] * u[2] ** o[2]
            n = sum(o)
            # Second order: every kind but BGK keeps the measured stress; higher orders: only
            # the recursive kinds rebuild a non-equilibrium part.
            rebuilt = n == 2 or (n > 2 and kind in ("recursive_trace_free", "recursive"))
            if kind != "bgk_extended" and rebuilt:
                a = a + omega * lbm._recursive_coefficient(o, u, p)
            coeffs.append(a)
        post = jnp.asarray(build, f.dtype) @ jnp.stack(coeffs).reshape(len(orders), -1)
        post = post.reshape(lat.q, *cells)
        return f + (post - f) / TAU0 if kind == "bgk_extended" else post

    return collide


def run_case(lat, collision: str, direction: str, mach: float) -> dict:
    shape = (N, N, N)
    solid = jnp.zeros(shape, bool)
    cfg = {"solid": solid, "fluid": ~solid, "tau0": TAU0, "cs": 0.0, "g": (0.0, 0.0, 0.0)}
    in_solver = collision in lbm.COLLISIONS and not (lat.q == 19 and collision == "recursive")
    if in_solver:
        collide = lambda f: lbm.collide_moments(jnp, lat, f, cfg, collision)  # noqa: E731
    else:
        collide = candidate(lat, collision)

    def body(_, f):
        return collide(lbm.stream(jnp, lat, f, solid))

    advance = jax.jit(lambda f, k: jax.lax.fori_loop(0, k, body, f), static_argnums=1)
    d = np.array(DIRECTIONS[direction], float)
    d /= np.linalg.norm(d)
    rng = np.random.default_rng(0)
    u = [mach * CS2**0.5 * d[a] + NOISE * rng.standard_normal(shape) for a in range(3)]
    rho = np.ones(shape)
    extended = collision != "regularised" and collision != "bgk" and lat.q == 27
    f = lbm.product_terms(np, lat, rho, u)[0] if extended else lbm.equilibrium(np, lat, rho, u)
    f = jnp.asarray(f, jnp.float32)
    m_mat, _ = lbm.moment_matrices(lat)

    def energy(f):
        m = m_mat @ np.asarray(f, np.float64).reshape(lat.q, -1)
        u = m[1:4] / m[0]
        return float(((u - u.mean(axis=1, keepdims=True)) ** 2).mean(axis=1).sum())

    start, e0 = time.perf_counter(), energy(f)
    ratios = []
    for _ in range(CHECKS):
        f = advance(f, STEPS // CHECKS)
        e = energy(f)
        ratios.append(e / e0 if np.isfinite(e) else None)
        if not np.isfinite(e):
            break
    finite = all(r is not None for r in ratios)
    return {
        "lattice": lat.name,
        "collision": collision,
        "adopted": in_solver,
        "direction": direction,
        "mach": mach,
        "steps_per_check": STEPS // CHECKS,
        "noise_energy_over_start": ratios,
        "finite": finite,
        "stable": finite and all(r < 1.0 for r in ratios),
        "seconds": round(time.perf_counter() - start, 1),
    }


def main() -> None:
    cases = [(lat, collision, d, 0.17) for lat, collision in (
        (D3Q19, "regularised"), (D3Q19, "bgk"), (D3Q19, "recursive"),
        (D3Q27, "regularised"), (D3Q27, "bgk"), (D3Q27, "recursive"),
        (D3Q27, "bgk_extended"), (D3Q27, "regularised_extended"),
        (D3Q27, "recursive_trace_free")) for d in DIRECTIONS]  # fmt: skip
    cases += [(D3Q27, "recursive", d, 0.25) for d in DIRECTIONS]
    # Would a slower wind rescue D3Q19? (A slower wind costs more steps for the same time.)
    cases += [(D3Q19, collision, d, 0.06) for collision in ("regularised", "bgk")
              for d in DIRECTIONS]  # fmt: skip
    device = jax.devices()[0]
    runs = []
    for lat, collision, direction, mach in cases:
        r = run_case(lat, collision, direction, mach)
        print({k: r[k] for k in ("lattice", "collision", "direction", "mach", "stable",
                                 "seconds")}, r["noise_energy_over_start"][-1])  # fmt: skip
        runs.append(r)
    out = {
        "what": "growth of tiny noise in a uniform wind with almost no viscosity, by collision",
        "run_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "aircanopy": __version__,
        "jax": jax.__version__,
        "device": {"platform": device.platform, "kind": device.device_kind},
        "cells": N**3,
        "tau0": TAU0,
        "noise": NOISE,
        "dtype": "float32",
        "runs": runs,
    }
    path = ROOT / "results" / f"wind_stability_{device.platform}.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
