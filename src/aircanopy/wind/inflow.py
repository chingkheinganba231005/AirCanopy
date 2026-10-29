"""Turbulent inflow from a precursor: a periodic box of the approach wind feeds the inlet.

Wind reaching a city is already turbulent, and its gusts matter: they mix fumes and shake the
wakes of buildings. A steady profile at the inlet would need a long distance to grow gusts of
its own. The usual remedy is a *precursor*: a separate box of the approach flow, periodic along
the wind, such as the boundary layer of JOURNAL.md Steps 11 and 12, run at the same time as the
site. At every step, one cross-section of the precursor's wind becomes the site's inlet.

    precursor = Solver(lat, solid_pre, params_pre, ...)            # periodic along x
    site = Solver(lat, solid_site, params_site, ..., inlet_u=...)  # Params(open_x=True)
    both = Precursor(precursor, site)
    both.run(1000)

Both boxes advance together, one step each, in one compiled loop (`coupled_step`). The inlet
takes, cell by cell, the velocity the precursor's own collision starts from on that plane: its
momentum after streaming divided by its density, plus half a step of the body force, before
any wall drag (lbm.velocity_and_force). The site's inlet cells then feel the same drag from the
floor's wall model as the precursor's cells did, so the two winds agree there.

The two boxes need the same lattice, cross-section (nz, ny) and walls on that plane, the same
collision and wall model, and the same backend and precision. The site's absorbing layer, if
any, keeps the target it was given when it was made: the mean wind the precursor brings.
"""

from __future__ import annotations

import functools

import numpy as np

from . import lbm
from .lattice import LATTICES
from .solver import Solver


def plane_velocity(xp, lat, f, cfg, forced: bool, plane: int):
    """u0 = sum_i f_i c_i / rho + g / 2 on the plane x = `plane` of pre-collision packets f, as
    three arrays (nz, ny); 0 in solid cells."""
    col = f[..., plane]
    fluid = cfg["fluid"][..., plane]
    rho = xp.where(fluid, lbm.moment(col, [1] * lat.q), 1.0)
    g = cfg["g"]
    return tuple(
        xp.where(fluid, lbm.moment(col, lat.c[:, a]) / rho + (0.5 * g[a] if forced else 0.0), 0.0)
        for a in range(3)
    )


def coupled_step(xp, lat, f_pre, f_site, cfg_pre, cfg_site, flags_pre, flags_site, plane=0):
    """One step of both boxes: the precursor streams, gives its plane's velocity to the site's
    inlet, and collides; then the site takes its step (lbm.step) with that inlet."""
    collision, free_slip_top, forced, dragged, slip_floors, matched = flags_pre[:6]
    f = lbm.stream(xp, lat, f_pre, cfg_pre["solid"], free_slip_top, slip_floors)
    inlet = plane_velocity(xp, lat, f, cfg_pre, forced, plane)
    f_pre = lbm.collide_moments(xp, lat, f, cfg_pre, collision, forced, dragged, matched)
    f_site = lbm.step(xp, lat, f_site, {**cfg_site, "inlet_u": inlet}, *flags_site)
    return f_pre, f_site


@functools.cache
def _jax_runner(lattice: str, flags_pre: tuple, flags_site: tuple, plane: int):
    import jax
    import jax.numpy as jnp

    lat = LATTICES[lattice]

    def run(f_pre, f_site, cfg_pre, cfg_site, n):
        def body(_, fs):
            return coupled_step(jnp, lat, *fs, cfg_pre, cfg_site, flags_pre, flags_site, plane)

        return jax.lax.fori_loop(0, n, body, (f_pre, f_site))

    return jax.jit(run)


class Precursor:
    """A precursor and the site it feeds, stepped together (module docstring)."""

    def __init__(self, precursor: Solver, site: Solver, plane: int = 0) -> None:
        pre, sp = precursor.params, site.params
        if pre.open_x or not sp.open_x:
            raise ValueError("the precursor must be periodic along x and the site open")
        if precursor.lat.name != site.lat.name:
            raise ValueError("the precursor and the site must use the same lattice")
        if (precursor.backend, precursor.dtype) != (site.backend, site.dtype):
            raise ValueError("the precursor and the site must use the same backend and dtype")
        same = ("collision", "free_slip_top", "slip_floors", "match_cells")
        if any(getattr(pre, k) != getattr(sp, k) for k in same):
            raise ValueError(f"the precursor and the site must have the same {', '.join(same)}")
        if precursor.solid.shape[:2] != site.solid.shape[:2]:
            raise ValueError("the precursor and the site must have the same cross-section")
        if not 0 <= plane < precursor.solid.shape[2]:
            raise ValueError("the plane must lie inside the precursor")
        if not np.array_equal(precursor.solid[..., plane], site.solid[..., 0]):
            raise ValueError("the precursor's plane and the site's inlet must have the same walls")
        for name in ("drag", "match_drag"):
            a, b = getattr(precursor, name), getattr(site, name)
            if (a is None) != (b is None) or (
                a is not None and not np.allclose(a[..., plane], b[..., 0])
            ):
                raise ValueError(f"the precursor's plane and the site's inlet need the same {name}")
        self.precursor, self.site, self.plane = precursor, site, plane

    def run(self, n: int) -> None:
        pre, site = self.precursor, self.site
        if pre.backend == "jax":
            runner = _jax_runner(pre.lat.name, pre.flags, site.flags, self.plane)
            pre.f, site.f = runner(pre.f, site.f, pre.cfg, site.cfg, n)
        else:
            for _ in range(n):
                pre.f, site.f = coupled_step(np, pre.lat, pre.f, site.f, pre.cfg, site.cfg,
                                             pre.flags, site.flags, self.plane)  # fmt: skip
        pre.time += n
        site.time += n
