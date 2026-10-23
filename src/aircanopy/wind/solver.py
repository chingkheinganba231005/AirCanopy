"""Run the lattice Boltzmann step (lbm.py) with NumPy, for checking, or with JAX, compiled.

    solver = Solver(D3Q19, solid, Params(tau0=0.8), backend="jax")
    solver.set_equilibrium(rho, (ux, uy, uz))
    solver.run(1000)
    rho, (ux, uy, uz) = solver.fields()

Everything is in lattice units (units.py). `solid` is a boolean array (nz, ny, nx), true in
cells the air cannot enter: buildings and ground from the site's voxel grid, or the walls of a
test case. The domain is periodic in all three directions unless walls close it; with a
free-slip top, the top layer must be all air and the bottom layer all solid, so that nothing
wraps around from the top to the bottom.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

from . import lbm
from .lattice import CS2, LATTICES, VelocitySet


@dataclass(frozen=True)
class Params:
    #: Relaxation time from the molecular viscosity: nu0 = c_s^2 (tau0 - 1/2).
    tau0: float
    #: Smagorinsky constant Cs; 0 switches the turbulence model off.
    smagorinsky: float = 0.0
    #: Body acceleration (gx, gy, gz) in lattice units, the same in every cell; None for none.
    g: tuple[float, float, float] | None = None
    #: Regularised collision (lbm.py); False gives plain BGK, kept for comparison.
    regularised: bool = True
    free_slip_top: bool = False


@functools.cache
def _jax_runner(lattice: str, regularised: bool, free_slip_top: bool, forced: bool):
    import jax
    import jax.numpy as jnp

    lat = LATTICES[lattice]

    def run(f, cfg, n):
        def body(_, f):
            return lbm.step(jnp, lat, f, cfg, regularised, free_slip_top, forced)

        return jax.lax.fori_loop(0, n, body, f)

    return jax.jit(run)


class Solver:
    def __init__(
        self,
        lat: VelocitySet,
        solid: np.ndarray,
        params: Params,
        backend: str = "jax",
        dtype=np.float32,
    ) -> None:
        solid = np.asarray(solid, dtype=bool)
        if solid.ndim != 3:
            raise ValueError("solid must have the shape (nz, ny, nx)")
        if params.free_slip_top and (solid[-1].any() or not solid[0].all()):
            raise ValueError("with a free-slip top, the top layer must be air, the bottom solid")
        if backend == "jax":
            import jax
            import jax.numpy as xp

            if np.dtype(dtype) == np.float64 and not jax.config.jax_enable_x64:
                raise ValueError("float64 needs jax.config.update('jax_enable_x64', True)")
        elif backend == "numpy":
            xp = np
        else:
            raise ValueError(f"unknown backend {backend!r}")
        self.lat = lat
        self.params = params
        self.backend = backend
        self.xp = xp
        self.dtype = np.dtype(dtype)
        scalar = lambda v: xp.asarray(v, dtype=self.dtype)  # noqa: E731
        g = params.g if params.g is not None else (0.0, 0.0, 0.0)
        self.cfg = {
            "solid": xp.asarray(solid),
            "fluid": xp.asarray(~solid),
            "tau0": scalar(params.tau0),
            "cs": scalar(params.smagorinsky),
            "g": tuple(scalar(v) for v in g),
        }
        self.solid = solid
        self.time = 0
        ones = np.ones(solid.shape)
        self.set_equilibrium(ones, (0 * ones, 0 * ones, 0 * ones))

    def set_equilibrium(self, rho, u) -> None:
        """Start from the equilibrium of the given density and velocity fields.

        The state is post-collision, whose momentum is rho u + F / 2, so with a body force the
        equilibrium is taken at u + g / 2; fields() then returns exactly (rho, u).
        """
        xp = self.xp
        as_x = lambda a: xp.asarray(np.asarray(a, dtype=np.float64), dtype=self.dtype)  # noqa: E731
        g = self.params.g
        if g is not None:
            u = tuple(np.asarray(u[a]) + 0.5 * g[a] for a in range(3))
        f = lbm.equilibrium(xp, self.lat, as_x(rho), tuple(as_x(c) for c in u))
        self.f = xp.asarray(f, dtype=self.dtype)

    def set_flow(
        self, rho, u, settle: bool = False, tolerance: float = 1e-12, max_steps: int = 50_000
    ) -> int:
        """Start from the given density and velocity, with the stress their gradients carry.

        A bare equilibrium has no viscous stress: in the first steps momentum then diffuses at
        the wrong rate, and sound waves start (Skordos 1993; Mei et al. 2006). So the packets
        also get the non-equilibrium part that the Chapman-Enskog expansion gives for the
        velocity gradients, P = -rho c_s^2 tau (grad u + grad u^T), relaxed as one collision
        would (times 1 - 1/tau, since the state is post-collision). The gradients are central
        differences, periodic at the edges of the box.

        A density that is only nearly right still starts sound waves. With `settle`, the step
        then runs with the velocity in the equilibrium held at u, so that only the density and
        the stress change, until the density changes by less than `tolerance` in a step (Mei et
        al. 2006). Returns the number of settling steps.
        """
        rho = np.asarray(rho, dtype=np.float64)
        u = tuple(np.asarray(c, dtype=np.float64) for c in u)
        # d u_a / d x_b; x, y, z are array axes 2, 1, 0.
        grad = [
            [(np.roll(u[a], -1, axis=2 - b) - np.roll(u[a], 1, axis=2 - b)) / 2 for b in range(3)]
            for a in range(3)
        ]
        strain = {(a, b): 0.5 * (grad[a][b] + grad[b][a]) for a, b in lbm.PAIRS}
        s_norm = np.sqrt(
            2.0 * sum((1.0 if a == b else 2.0) * strain[a, b] ** 2 for a, b in lbm.PAIRS)
        )
        p = self.params
        tau = p.tau0 + p.smagorinsky**2 * s_norm / CS2
        stress = [-2.0 * rho * CS2 * tau * strain[a, b] for a, b in lbm.PAIRS]
        if p.g is not None:
            u = tuple(u[a] + 0.5 * p.g[a] for a in range(3))
        f = lbm.equilibrium(np, self.lat, rho, u)
        f = f + (1.0 - 1.0 / tau) * lbm.regularised_neq(np, self.lat, stress)
        steps = 0
        if settle:
            if p.g is not None:
                raise ValueError("settling is written for flows without a body force")
            lat, fluid, previous = self.lat, ~self.solid, rho
            while steps < max_steps:
                steps += 1
                f = lbm.stream(np, lat, f, self.solid, p.free_slip_top)
                rho_now = lbm.moment(f, [1] * lat.q)
                feq = lbm.equilibrium(np, lat, rho_now, u)
                flux = lbm.flux(np, lat, f, feq, u)
                tau = lbm.smagorinsky_tau(np, flux, rho_now, p.tau0, p.smagorinsky)
                neq = lbm.regularised_neq(np, lat, flux) if p.regularised else f - feq
                f = np.where(fluid, feq + (1.0 - 1.0 / tau) * neq, f)
                change = np.abs(rho_now - previous)[fluid].max()
                previous = rho_now
                if change < tolerance:
                    break
        self.f = self.xp.asarray(f, dtype=self.dtype)
        return steps

    def run(self, n: int) -> None:
        p = self.params
        forced = p.g is not None
        if self.backend == "jax":
            runner = _jax_runner(self.lat.name, p.regularised, p.free_slip_top, forced)
            self.f = runner(self.f, self.cfg, n)
        else:
            for _ in range(n):
                self.f = lbm.step(
                    np, self.lat, self.f, self.cfg, p.regularised, p.free_slip_top, forced
                )
        self.time += n

    def fields(self) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Density and velocity after the last step, as NumPy arrays in float64.

        Read from the post-collision populations: their momentum is rho u + F / 2, so half the
        force is taken off again. Solid cells report air at rest at density 1.
        """
        f = np.asarray(self.f, dtype=np.float64)
        rho = lbm.moment(f, [1] * self.lat.q)
        u = [lbm.moment(f, self.lat.c[:, a]) / rho for a in range(3)]
        if self.params.g is not None:
            u = [u[a] - 0.5 * self.params.g[a] for a in range(3)]
        fluid = ~self.solid
        return np.where(fluid, rho, 1.0), tuple(np.where(fluid, c, 0.0) for c in u)
