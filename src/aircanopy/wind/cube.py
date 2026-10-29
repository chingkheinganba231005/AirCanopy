"""The flow around a cube on the floor of a channel (Martinuzzi and Tropea 1993), the wind-tunnel
case the solver is checked against (JOURNAL.md, Part 4: the cube, planned on 2026-10-29).

Two boxes, measured in the cube's height H, which is `Cube.h` cells:

- the approach channel, a precursor (inflow.py): periodic along the flow and across it, 2H
  high between the floor and a free-slip lid, `precursor` H long and `width` H wide;
- the site: the same cross-section, open along x and fed by the precursor's packets, with the
  cube on the floor `upstream` H after the inlet, centred across, and `downstream` H of
  channel behind it, the last `absorb` H of which are the absorbing layer.

The sizes follow Krajnović and Davidson (2002): the channel 2H high and 7H wide, the cube 3H
from the inlet (docs/sources.md). The outlet is further away than their 6H behind the cube, so
that the absorbing layer stays out of the wake. Both boxes are pushed by the same body force,
g = u*^2 / H, which in a periodic channel 2H high is held back by the friction of its two
walls, rho u*^2 each. The Reynolds number U_b H / nu sets the molecular viscosity.

The walls have the wall model of walls.py, and where it reads the wind follows D-019. On the
channel's floor and lid it reads 0.2 delta from the wall, delta = H being the depth of each
wall's boundary layer (the channel's half-height): `match_cells`, the cell centre nearest that
height. Near the cube the law of the wall does not hold 0.2 H up: in front of the cube the
flow separates in a vortex only about that tall, and a wall model reading above it would push
the floor's air the wrong way. So from `first_cell_from` H in front of the cube to the outlet,
and on the cube's top, the floor reads the first cell. The cube's sides are no-slip (halfway
bounce-back, lbm.stream). The lid reads 0.2 H below it everywhere; it is H above the cube.

The walls of the wind tunnel were smooth. The wall model needs a roughness length, and the one
used here is an estimate made before any run: the z0 for which a mean wind on the law of the
wall from the wall to the channel's middle would have the measured friction and the bulk
velocity, U_b / u* = (ln(H / z0) - 1) / kappa. The approach channel's run corrects it once,
before the cube is put in (benchmarks/wind_cube.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import walls
from .lattice import CS2, VelocitySet
from .solver import Params, Solver


@dataclass(frozen=True)
class Cube:
    #: Cells across the cube; even, so that the symmetry plane lies between two rows of cells.
    h: int = 32
    #: The channel's bulk velocity, cells per step: Mach 0.087. The fastest wind, over the
    #: cube, stays below A-009's 0.1 cells per step.
    u_bulk: float = 0.05
    #: Friction velocity over bulk velocity: sqrt(0.00228), the turbulent stress -u'v' measured
    #: closest to the floor at x/H = -18 (ERCOFTAC file x01z09_a, y/H = 0.02).
    u_star_over_bulk: float = 0.0477
    #: U_b H / nu (Krajnović and Davidson 2002).
    reynolds: float = 4.0e4
    #: The walls' roughness length in H; None for the estimate (module docstring).
    z0_over_h: float | None = None
    smagorinsky: float = 0.17
    #: Where the channel walls' model reads the wind, in H from the wall (D-019).
    wall_model_height: float = 0.2
    #: The floor has the first-cell model from this many H in front of the cube to the outlet.
    first_cell_from: float = 2.0
    width: float = 7.0
    precursor: float = 6.0
    upstream: float = 3.0
    downstream: float = 10.0
    absorb: float = 1.0
    absorb_strength: float = 0.2

    def __post_init__(self) -> None:
        if self.h % 2:
            raise ValueError("h must be even, so that the symmetry plane lies between two rows")
        if self.match_cells < 1:
            raise ValueError("h is too small for the wall model to read above the first cell")

    def cells(self, lengths: float) -> int:
        return round(lengths * self.h)

    @property
    def tau0(self) -> float:
        return 0.5 + self.u_bulk * self.h / self.reynolds / CS2

    @property
    def u_star(self) -> float:
        return self.u_star_over_bulk * self.u_bulk

    @property
    def g(self) -> float:
        return self.u_star**2 / self.h

    @property
    def z0(self) -> float:
        """The walls' roughness length in cells."""
        if self.z0_over_h is not None:
            return self.z0_over_h * self.h
        return self.h * math.exp(-walls.KAPPA / self.u_star_over_bulk - 1.0)

    @property
    def match_cells(self) -> int:
        """Cells between the first cell and the one the channel walls' model reads: the cell
        centre k + 1/2 nearest wall_model_height H."""
        return int(self.wall_model_height * self.h)

    def cube_x(self) -> tuple[int, int]:
        """The cube's first column in the site, and the column after its last."""
        x0 = self.cells(self.upstream)
        return x0, x0 + self.h

    def cube_y(self) -> tuple[int, int]:
        y0 = (self.cells(self.width) - self.h) // 2
        return y0, y0 + self.h

    def symmetry_rows(self) -> slice:
        """The two rows of cells either side of the cube's symmetry plane."""
        mid = self.cube_y()[0] + self.h // 2
        return slice(mid - 1, mid + 1)


def channel(c: Cube, length: int) -> np.ndarray:
    """The solid cells of a channel 2H high and `width` H wide: the floor layer only (the lid is
    the free-slip top, above the top layer of air)."""
    solid = np.zeros((2 * c.h + 1, c.cells(c.width), length), bool)
    solid[0] = True
    return solid


def site_solid(c: Cube) -> np.ndarray:
    solid = channel(c, c.cells(c.upstream + 1 + c.downstream))
    (x0, x1), (y0, y1) = c.cube_x(), c.cube_y()
    solid[1 : c.h + 1, y0:y1, x0:x1] = True
    return solid


def wall_drag(c: Cube, solid: np.ndarray, first_cell_column: int | None = None) -> tuple:
    """The drag fields (drag, match_drag) of a box of the case: the wall model reading
    `match_cells` from the floor and from the lid, except that from `first_cell_column` on (the
    site's), the floors have the first-cell model (module docstring)."""
    local, matched = walls.matched_floor_drag(solid, c.z0, c.match_cells)
    if first_cell_column is not None:
        near = np.zeros(solid.shape, bool)
        near[..., first_cell_column:] = True
        local = np.where(near, walls.rough_floor_drag(solid, c.z0), local)
        matched = np.where(near, 0.0, matched)
    lid_local, lid_matched = walls.matched_lid_drag(solid, c.z0, c.match_cells)
    return local + lid_local, matched + lid_matched


def _solver(c: Cube, lat, solid, drag, backend, dtype, collision, inlet_u=None,
            **absorbing) -> Solver:  # fmt: skip
    params = Params(tau0=c.tau0, smagorinsky=c.smagorinsky, g=(c.g, 0.0, 0.0),
                    free_slip_top=True, slip_floors=True, collision=collision,
                    match_cells=c.match_cells, open_x=inlet_u is not None,
                    **absorbing)  # fmt: skip
    return Solver(lat, solid, params, backend, dtype, drag=drag[0], match_drag=drag[1],
                  inlet_u=inlet_u)  # fmt: skip


def precursor_box(c: Cube, lat: VelocitySet, backend: str = "jax", dtype=np.float32,
                  collision: str = "recursive") -> Solver:  # fmt: skip
    """The approach channel, not yet started."""
    solid = channel(c, c.cells(c.precursor))
    return _solver(c, lat, solid, wall_drag(c, solid), backend, dtype, collision)


def site_box(c: Cube, lat: VelocitySet, mean_wind: np.ndarray, backend: str = "jax",
             dtype=np.float32, collision: str = "recursive") -> Solver:  # fmt: skip
    """The site with the cube, not yet started. `mean_wind` (one value per layer, nz) is the
    approach channel's mean wind: the absorbing layer's target, and the inlet's wind until the
    precursor feeds it."""
    solid = site_solid(c)
    drag = wall_drag(c, solid, c.cube_x()[0] - c.cells(c.first_cell_from))
    inlet = (np.broadcast_to(np.asarray(mean_wind)[:, None], solid.shape[:2]), 0.0, 0.0)
    return _solver(c, lat, solid, drag, backend, dtype, collision, inlet_u=inlet,
                   absorb_cells=c.cells(c.absorb), absorb_strength=c.absorb_strength)  # fmt: skip


def zero_crossing(x: np.ndarray, u: np.ndarray, start: int, step: int) -> float | None:
    """From index `start`, walking by `step` (+1 or -1), the first place where u changes sign,
    by linear interpolation between the two cells; None if it never does."""
    k = start
    while 0 <= k + step < len(u):
        a, b = u[k], u[k + step]
        if a == 0.0:
            return float(x[k])
        if a * b < 0:
            return float(x[k] + (x[k + step] - x[k]) * a / (a - b))
        k += step
    return None


def separation_lengths(c: Cube, u_floor: np.ndarray) -> dict:
    """The lengths of the mean recirculations on the floor along the symmetry line, in H, from
    the mean wind u in the first cell of air above the floor, one value per column of the site.

    x_F, from the cube's front face upstream to where the approach flow separates: found by
    walking downstream from 2H in front of the cube to the first place the wind turns back.
    x_R, from the cube's back face downstream to where the flow reattaches: found by walking
    upstream from 4H behind the cube to the first place the wind turns back. Walking in from
    outside skips the small eddies in the corners at the foot of the cube's faces. A length is
    None if the wind already runs back where the walk starts, or never turns."""
    x = np.arange(len(u_floor)) + 0.5  # cell centres; the cube's faces lie at whole numbers
    x0, x1 = c.cube_x()
    start = max(0, x0 - 2 * c.h)
    front = zero_crossing(x[:x0], u_floor[:x0], start, +1) if u_floor[start] > 0 else None
    end = min(len(u_floor) - 1, x1 + 4 * c.h)
    back = (zero_crossing(x[x1 : end + 1], u_floor[x1 : end + 1], end - x1, -1)
            if u_floor[end] > 0 else None)  # fmt: skip
    return {"x_F": None if front is None else (x0 - front) / c.h,
            "x_R": None if back is None else (back - x1) / c.h}  # fmt: skip
