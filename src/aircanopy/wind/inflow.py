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

Both boxes advance together, one step each, in one compiled loop (`coupled_step`), and the
plane is handed over in one of two ways (`coupling`):

- "packets" (the default): the inlet takes the packets the precursor streams into that plane
  from upstream (c_x = +1), as they are (lbm.open_boundaries). They bring the precursor's
  density and stress along with its velocity, and the packets leaving the site through the
  inlet are left alone, so waves going upstream leave instead of reflecting.
- "velocity": the inlet is a velocity inlet (lbm._open_column) given, cell by cell, the
  velocity the precursor's own collision starts from on that plane: its momentum after
  streaming over its density, plus half a step of the body force, before any wall drag
  (lbm.velocity_and_force). With the wall model, the inlet's cells then feel the same drag as
  the precursor's. The first version; in the turbulent test of Step 14 it made the vertical
  gusts after the inlet two to seven times too strong.

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

#: How the precursor's plane is handed to the site's inlet (module docstring).
COUPLINGS = ("packets", "velocity")


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


def coupled_step(xp, lat, f_pre, f_site, cfg_pre, cfg_site, flags_pre, flags_site, plane=0,
                 coupling="packets"):  # fmt: skip
    """One step of both boxes: the precursor streams, hands its plane to the site's inlet (as
    packets or as a velocity, module docstring), and collides; then the site takes its step
    (lbm.step) with that inlet."""
    collision, free_slip_top, forced, dragged, slip_floors, matched = flags_pre[:6]
    f = lbm.stream(xp, lat, f_pre, cfg_pre["solid"], free_slip_top, slip_floors)
    if coupling == "packets":
        given = {"inlet_packets": [f[i, ..., plane] for i in range(lat.q) if lat.c[i, 0] > 0]}
    else:
        given = {"inlet_u": plane_velocity(xp, lat, f, cfg_pre, forced, plane)}
    f_pre = lbm.collide_moments(xp, lat, f, cfg_pre, collision, forced, dragged, matched)
    f_site = lbm.step(xp, lat, f_site, {**cfg_site, **given}, *flags_site)
    return f_pre, f_site


@functools.cache
def _jax_runner(lattice: str, flags_pre: tuple, flags_site: tuple, plane: int, coupling: str):
    import jax
    import jax.numpy as jnp

    lat = LATTICES[lattice]

    def run(f_pre, f_site, cfg_pre, cfg_site, n):
        def body(_, fs):
            return coupled_step(jnp, lat, *fs, cfg_pre, cfg_site, flags_pre, flags_site, plane,
                                coupling)  # fmt: skip

        return jax.lax.fori_loop(0, n, body, (f_pre, f_site))

    return jax.jit(run)


class Precursor:
    """A precursor and the site it feeds, stepped together (module docstring)."""

    def __init__(self, precursor: Solver, site: Solver, plane: int = 0,
                 coupling: str = "packets") -> None:  # fmt: skip
        if coupling not in COUPLINGS:
            raise ValueError(f"unknown coupling {coupling!r}; choose one of {COUPLINGS}")
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
        self.coupling = coupling

    def run(self, n: int) -> None:
        pre, site = self.precursor, self.site
        if pre.backend == "jax":
            runner = _jax_runner(pre.lat.name, pre.flags, site.flags, self.plane, self.coupling)
            pre.f, site.f = runner(pre.f, site.f, pre.cfg, site.cfg, n)
        else:
            for _ in range(n):
                pre.f, site.f = coupled_step(np, pre.lat, pre.f, site.f, pre.cfg, site.cfg,
                                             pre.flags, site.flags, self.plane,
                                             self.coupling)  # fmt: skip
        pre.time += n
        site.time += n
