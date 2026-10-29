"""Run the lattice Boltzmann step (lbm.py) with NumPy, for checking, or with JAX, compiled.

    solver = Solver(D3Q27, solid, Params(tau0=0.8), backend="jax")
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
`match_drag`, with Params(match_cells=k), is a second such field whose force is driven by the
wind k cells above instead of the cell's own (walls.matched_floor_drag).
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
    #: Collision model (lbm.COLLISIONS): "recursive" needs D3Q27; "regularised" and "bgk" are
    #: kept for comparison and for D3Q19 (JOURNAL.md, Step 12).
    collision: str = "recursive"
    free_slip_top: bool = False
    #: Floors (solid below air) are free-slip, for a wall model (lbm.stream, walls.py).
    slip_floors: bool = False
    #: The floors' wall model reads the wind this many cells above the first cell
    #: (walls.matched_floor_drag); 0 for the first-cell model alone.
    match_cells: int = 0
    #: Open along x: a velocity inlet at x = 0 (Solver(inlet_u=...)) and a pressure outlet at
    #: the last x (lbm.open_boundaries), instead of periodic sides.
    open_x: bool = False
    #: An absorbing layer before the outlet (open_x only): in its last `absorb_cells` columns,
    #: after each collision, the packets move part of the way towards the equilibrium of
    #: density 1 and the inlet's velocity (as given to the Solver), by a share that rises as
    #: the square of the depth into the layer, up to `absorb_strength`. Eddies and sound fade
    #: there instead of reflecting off the outlet (JOURNAL.md, Step 13).
    absorb_cells: int = 0
    absorb_strength: float = 0.0


@functools.cache
def _jax_runner(lattice: str, flags: tuple):
    import jax
    import jax.numpy as jnp

    lat = LATTICES[lattice]

    def run(f, cfg, n):
        def body(_, f):
            return lbm.step(jnp, lat, f, cfg, *flags)

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
        match_drag: np.ndarray | None = None,
        inlet_u: tuple | None = None,
    ) -> None:
        solid = np.asarray(solid, dtype=bool)
        if solid.ndim != 3:
            raise ValueError("solid must have the shape (nz, ny, nx)")
        if params.open_x != (inlet_u is not None):
            raise ValueError("Params(open_x=True) and inlet_u, the inlet's velocity, go together")
        if params.absorb_cells and not params.open_x:
            raise ValueError("an absorbing layer needs Params(open_x=True)")
        if params.free_slip_top and (solid[-1].any() or not solid[0].all()):
            raise ValueError("with a free-slip top, the top layer must be air, the bottom solid")
        for field in (drag, match_drag):
            if field is not None and np.shape(field) != solid.shape:
                raise ValueError("drag fields must have the same shape as solid")
        if (match_drag is not None) != (params.match_cells > 0):
            raise ValueError("match_drag and Params(match_cells > 0) go together")
        lbm.check_collision(lat, params.collision)
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
            "match_drag": scalar(0.0) if match_drag is None else scalar(
                np.where(solid, 0.0, match_drag)),
            "inlet_u": None if inlet_u is None else tuple(
                scalar(np.broadcast_to(c, solid.shape[:2])) for c in inlet_u),
            "outlet_rho": scalar(1.0),
        }  # fmt: skip
        self.solid = solid
        self.drag = None if drag is None else np.where(solid, 0.0, np.asarray(drag, float))
        self.match_drag = (None if match_drag is None
                           else np.where(solid, 0.0, np.asarray(match_drag, float)))  # fmt: skip
        if params.absorb_cells:
            cells = params.absorb_cells
            depth = (np.arange(cells, dtype=float) + 1.0) / cells  # 1/L ... 1 to the outlet
            self.cfg["absorb_sigma"] = scalar(params.absorb_strength * depth**2)
            shape = (*solid.shape[:2], cells)
            # The layer acts on post-collision packets, which carry rho (u + g / 2).
            u_target = tuple(np.broadcast_to(np.asarray(c, float)[..., None] if np.ndim(c) else c,
                                             shape) + 0.5 * g[a]
                             for a, c in enumerate(inlet_u))  # fmt: skip
            self.cfg["absorb_target"] = xp.asarray(
                self._equilibrium(np, np.ones(shape), u_target), dtype=self.dtype
            )
        self.time = 0
        ones = np.ones(solid.shape)
        self.set_equilibrium(ones, (0 * ones, 0 * ones, 0 * ones))

    @property
    def _flags(self) -> tuple:
        p = self.params
        return (p.collision, p.free_slip_top, p.g is not None, self.drag is not None,
                p.slip_floors, p.match_cells, p.open_x, p.absorb_cells)  # fmt: skip

    def set_inlet(self, u) -> None:
        """Change the inlet's velocity: three arrays (nz, ny), or numbers, for the next steps.
        The absorbing layer keeps the target it was given when the solver was made."""
        xp = self.xp
        shape = self.solid.shape[:2]
        self.cfg["inlet_u"] = tuple(xp.asarray(np.broadcast_to(c, shape), dtype=self.dtype)
                                    for c in u)  # fmt: skip

    def _equilibrium(self, xp, rho, u):
        """The equilibrium the collision relaxes to: the extended one for "recursive"."""
        if self.params.collision == "recursive":
            return lbm.product_terms(xp, self.lat, rho, u)[0]
        return lbm.equilibrium(xp, self.lat, rho, u)

    def _neq(self, xp, rho, u, p):
        """The non-equilibrium packets the collision builds from the flux p (PAIRS order)."""
        if self.params.collision == "recursive":
            return lbm.product_terms(xp, self.lat, rho, u, p)[1]
        return lbm.regularised_neq(xp, self.lat, p)

    def _post_velocity(self, u):
        """The velocity of a post-collision state whose collision velocity is u.

        After a collision the packets carry momentum rho u + F / 2, F = rho g - (lambda / 2)
        rho |u| u, so their equilibrium is taken at u + g / 2 - (lambda / 4) |u| u. A matched
        drag adds -(lambda_m / 4) |u_k| u_k, with u_k the velocity k cells above.
        """
        u = [np.asarray(c, dtype=np.float64) for c in u]
        g = self.params.g
        out = [u[a] + (0.5 * g[a] if g is not None else 0.0) for a in range(3)]
        if self.drag is not None:
            speed = np.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
            out = [out[a] - 0.25 * self.drag * speed * u[a] for a in range(3)]
        if self.match_drag is not None:
            up = [np.roll(c, -self.params.match_cells, axis=0) for c in u]
            speed = np.sqrt(up[0] ** 2 + up[1] ** 2 + up[2] ** 2)
            out = [out[a] - 0.25 * self.match_drag * speed * up[a] for a in range(3)]
        return tuple(out)

    def _velocity(self, xp, f, rho, drag, match_drag=None):
        """The collision velocity u from post-collision packets: the inverse of _post_velocity.

        With w = sum f c / rho - g / 2 = u - (lambda / 4) |u| u, u = 2 w / (1 + sqrt(1 -
        lambda |w|)). This needs lambda |w| <= 1, which holds while lambda |u| < 3, far beyond
        the drag of floors or leaves; it is clipped there rather than give NaN. A matched drag
        is undone with w of the cell k above, which is that cell's u where it has no drag of its
        own: u = w + (lambda_m / 4) |w_k| w_k.
        """
        w = [lbm.moment(f, self.lat.c[:, a]) / rho for a in range(3)]
        g = self.params.g
        if g is not None:
            w = [w[a] - 0.5 * g[a] for a in range(3)]
        if match_drag is not None:
            up = [xp.roll(c, -self.params.match_cells, axis=0) for c in w]
            speed = xp.sqrt(up[0] * up[0] + up[1] * up[1] + up[2] * up[2])
            w = [w[a] + 0.25 * match_drag * speed * up[a] for a in range(3)]
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
        f = self._equilibrium(xp, as_x(rho), tuple(as_x(c) for c in self._post_velocity(u)))
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
        post_u = self._post_velocity(u)
        f = self._equilibrium(np, rho, post_u)
        f = f + (1.0 - 1.0 / tau) * self._neq(np, rho, post_u, stress)
        steps = 0
        if settle:
            if p.g is not None or self.drag is not None or self.match_drag is not None:
                raise ValueError("settling is written for flows without forces")
            lat, fluid, previous = self.lat, ~self.solid, rho
            while steps < max_steps:
                steps += 1
                f = lbm.stream(np, lat, f, self.solid, p.free_slip_top, p.slip_floors)
                rho_now = lbm.moment(f, [1] * lat.q)
                # P is the same with either equilibrium (lbm.collide).
                flux = lbm.flux(np, lat, f, lbm.equilibrium(np, lat, rho_now, u), u)
                feq = self._equilibrium(np, rho_now, u)
                tau = lbm.smagorinsky_tau(np, flux, rho_now, p.tau0, p.smagorinsky)
                neq = f - feq if p.collision == "bgk" else self._neq(np, rho_now, u, flux)
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
        u = self._velocity(np, f, rho, self.drag, self.match_drag)
        fluid = ~self.solid
        return np.where(fluid, rho, 1.0), tuple(np.where(fluid, c, 0.0) for c in u)

    def mean_kinetic_energy(self) -> float:
        """Mean of rho |u|^2 / 2 over the air cells, computed where the state lives (on the
        graphics card, with JAX), so that only one number is copied back."""
        xp, f, fluid = self.xp, self.f, self.cfg["fluid"]
        rho = lbm.moment(f, [1] * self.lat.q)
        drag = None if self.drag is None else self.cfg["drag"]
        match_drag = None if self.match_drag is None else self.cfg["match_drag"]
        u = self._velocity(xp, f, rho, drag, match_drag)
        e = 0.5 * rho * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2])
        return float(xp.sum(xp.where(fluid, e, 0.0)) / xp.sum(fluid))
