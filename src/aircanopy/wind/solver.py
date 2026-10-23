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

`drag` is an optional field of drag coefficients lambda (per cell, lattice units): each cell
feels a force -(lambda / 2) rho |u| u. walls.py makes one for the friction of rough floors.
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
    #: Floors (solid below air) are free-slip, for a wall model (lbm.stream, walls.py).
    slip_floors: bool = False


@functools.cache
def _jax_runner(lattice: str, flags: tuple[bool, ...]):
    import jax
    import jax.numpy as jnp

    lat = LATTICES[lattice]
    regularised, free_slip_top, forced, dragged, slip_floors = flags

    def run(f, cfg, n):
        def body(_, f):
            return lbm.step(
                jnp, lat, f, cfg, regularised, free_slip_top, forced, dragged, slip_floors
            )

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
        drag: np.ndarray | None = None,
    ) -> None:
        solid = np.asarray(solid, dtype=bool)
        if solid.ndim != 3:
            raise ValueError("solid must have the shape (nz, ny, nx)")
        if params.free_slip_top and (solid[-1].any() or not solid[0].all()):
            raise ValueError("with a free-slip top, the top layer must be air, the bottom solid")
        if drag is not None and np.shape(drag) != solid.shape:
            raise ValueError("drag must have the same shape as solid")
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
            "drag": scalar(0.0) if drag is None else scalar(np.where(solid, 0.0, drag)),
        }
        self.solid = solid
        self.drag = None if drag is None else np.where(solid, 0.0, np.asarray(drag, float))
        self.time = 0
        ones = np.ones(solid.shape)
        self.set_equilibrium(ones, (0 * ones, 0 * ones, 0 * ones))

    @property
    def _flags(self) -> tuple[bool, ...]:
        p = self.params
        return (p.regularised, p.free_slip_top, p.g is not None, self.drag is not None,
                p.slip_floors)  # fmt: skip

    def _post_velocity(self, u):
        """The velocity of a post-collision state whose collision velocity is u.

        After a collision the packets carry momentum rho u + F / 2, F = rho g - (lambda / 2)
        rho |u| u, so their equilibrium is taken at u + g / 2 - (lambda / 4) |u| u.
        """
        u = [np.asarray(c, dtype=np.float64) for c in u]
        g = self.params.g
        out = [u[a] + (0.5 * g[a] if g is not None else 0.0) for a in range(3)]
        if self.drag is not None:
            speed = np.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
            out = [out[a] - 0.25 * self.drag * speed * u[a] for a in range(3)]
        return tuple(out)

    def _velocity(self, xp, f, rho, drag):
        """The collision velocity u from post-collision packets: the inverse of _post_velocity.

        With w = sum f c / rho - g / 2 = u - (lambda / 4) |u| u, u = 2 w / (1 + sqrt(1 -
        lambda |w|)). This needs lambda |w| <= 1, which holds while lambda |u| < 3, far beyond
        the drag of floors or leaves; it is clipped there rather than give NaN.
        """
        w = [lbm.moment(f, self.lat.c[:, a]) / rho for a in range(3)]
        g = self.params.g
        if g is not None:
            w = [w[a] - 0.5 * g[a] for a in range(3)]
        if drag is not None:
            speed = xp.sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2])
            k = 2.0 / (1.0 + xp.sqrt(xp.maximum(1.0 - drag * speed, 0.0)))
            w = [k * c for c in w]
        return w

    def set_equilibrium(self, rho, u) -> None:
        """Start from the equilibrium of the given density and velocity fields.

        The state is post-collision (see _post_velocity); fields() then returns (rho, u).
        """
        xp = self.xp
        as_x = lambda a: xp.asarray(np.asarray(a, dtype=np.float64), dtype=self.dtype)  # noqa: E731
        f = lbm.equilibrium(xp, self.lat, as_x(rho), tuple(as_x(c) for c in self._post_velocity(u)))
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
        f = lbm.equilibrium(np, self.lat, rho, self._post_velocity(u))
        f = f + (1.0 - 1.0 / tau) * lbm.regularised_neq(np, self.lat, stress)
        steps = 0
        if settle:
            if p.g is not None or self.drag is not None:
                raise ValueError("settling is written for flows without forces")
            lat, fluid, previous = self.lat, ~self.solid, rho
            while steps < max_steps:
                steps += 1
                f = lbm.stream(np, lat, f, self.solid, p.free_slip_top, p.slip_floors)
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
        if self.backend == "jax":
            self.f = _jax_runner(self.lat.name, self._flags)(self.f, self.cfg, n)
        else:
            for _ in range(n):
                self.f = lbm.step(np, self.lat, self.f, self.cfg, *self._flags)
        self.time += n

    def fields(self) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Density and velocity after the last step, as NumPy arrays in float64.

        Read from the post-collision packets (see _velocity). Solid cells report air at rest at
        density 1.
        """
        f = np.asarray(self.f, dtype=np.float64)
        rho = lbm.moment(f, [1] * self.lat.q)
        u = self._velocity(np, f, rho, self.drag)
        fluid = ~self.solid
        return np.where(fluid, rho, 1.0), tuple(np.where(fluid, c, 0.0) for c in u)

    def mean_kinetic_energy(self) -> float:
        """Mean of rho |u|^2 / 2 over the air cells, computed where the state lives (on the
        graphics card, with JAX), so that only one number is copied back."""
        xp, f, fluid = self.xp, self.f, self.cfg["fluid"]
        rho = lbm.moment(f, [1] * self.lat.q)
        drag = None if self.drag is None else self.cfg["drag"]
        u = self._velocity(xp, f, rho, drag)
        e = 0.5 * rho * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2])
        return float(xp.sum(xp.where(fluid, e, 0.0)) / xp.sum(fluid))
