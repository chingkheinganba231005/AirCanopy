"""The friction of rough floors, for wind on a city-sized grid: a wall model.

With cells a metre or more across, the thin layer of air against the ground, in which the wind
slows to zero, cannot be resolved. A wall model stands in for it (assumption A-010):

1. Floors are free-slip (`Params(slip_floors=True)`, see lbm.stream): the air glides over them.
2. The air in the first cell above a floor feels the friction that the law of the wall gives
   for a rough surface. In neutral air the mean wind at height z above a surface of roughness
   length z0 is U(z) = (u* / kappa) ln(z / z0), where u* is the friction velocity (the
   surface's shear stress is rho u*^2) and kappa the von Kármán constant. With the wind U1 at
   the centre of the first cell, half a cell above the floor (z1 = 1/2), this gives
       u* = kappa U1 / ln(z1 / z0),   stress = rho C_d U1^2,   C_d = (kappa / ln(z1 / z0))^2.
   Spread over the cell, one cell thick, the stress is the drag -(lambda / 2) rho |u| u of
   lbm.velocity_and_force with lambda = 2 C_d.

The law of the wall describes the mean wind. Applying it to the wind of each moment is the
simplest wall model for large-eddy simulation, the one to test first.

The wind in the first cells above a wall is always the least accurate part of a large-eddy
simulation, because the eddies that carry momentum there are smaller than the cells: "the LES
is inevitably plagued by numerical and subgrid-modeling errors in the first few grid points
off the wall" (Larsson et al. 2016, Sec. 3.2, after Kawai and Larsson 2012). Fed from the
first cell, a wall model passes that error on, and the mean wind misses the law of the wall
(the "log-layer mismatch"). Their remedy is to read the wind at a height h_wm chosen for the
flow, not the grid, with cells small enough there. `matched_floor_drag` does this for floors:
the stress rho C_d(h) |U(h)| U(h), with C_d(h) = (kappa / ln(h / z0))^2 and U(h) the wind
`cells_up` cells above the first cell, at h = cells_up + 1/2, acts on the first cell.
`matched_lid_drag` does the same for a lid, reading the wind below the top layer.
"""

from __future__ import annotations

import math

import numpy as np

#: The von Kármán constant: "the canonical value 0.40" (Andreas et al. 2006, who measured
#: 0.387 +- 0.003 over rough sea ice; laboratory values cluster near 0.37-0.39 and 0.42-0.43).
#: Assumption A-010.
KAPPA = 0.40

#: Height of the first cell's centre above the floor, in cells.
Z1 = 0.5


def drag_coefficient(z0, kappa: float = KAPPA, height: float = Z1):
    """C_d = (kappa / ln(height / z0))^2 for a roughness length z0 in cells (0 < z0 < 1/2) and
    the height, in cells above the floor, of the wind it multiplies (the first cell's centre)."""
    z0 = np.asarray(z0, dtype=float)
    if np.any(z0 <= 0) or np.any(z0 >= Z1):
        raise ValueError(f"the roughness length must be between 0 and {Z1} cells")
    return (kappa / np.log(height / z0)) ** 2


def floor_cells(solid: np.ndarray) -> np.ndarray:
    """Air cells that stand on a solid cell (the bottom layer has nothing below it)."""
    solid = np.asarray(solid, dtype=bool)
    out = np.zeros_like(solid)
    out[1:] = solid[:-1] & ~solid[1:]
    return out


def rough_floor_drag(solid: np.ndarray, z0) -> np.ndarray:
    """The drag field lambda: 2 C_d in the air cells on a floor, 0 elsewhere.

    z0 is a roughness length in cells: one number, or one per cell (only the values in floor
    cells are used), for floors of different kinds such as roads, grass and roofs.
    """
    floors = floor_cells(solid)
    z0 = np.broadcast_to(np.asarray(z0, dtype=float), floors.shape)
    out = np.zeros(floors.shape)
    out[floors] = 2.0 * drag_coefficient(z0[floors])
    return out


def matched_floor_drag(solid: np.ndarray, z0, cells_up: int) -> tuple[np.ndarray, np.ndarray]:
    """The drag fields of a floor wall model fed from `cells_up` cells above the first cell.

    Returns (local, matched): `matched` holds 2 C_d(h), h = cells_up + 1/2, in the floor cells
    with at least cells_up more cells of air above them, for Solver(match_drag=...) with
    Params(match_cells=cells_up); `local` holds the first-cell drag (rough_floor_drag) of the
    other floor cells, where the air is too shallow, for Solver(drag=...).
    """
    if cells_up < 1:
        raise ValueError("cells_up must be at least 1; 0 is the first-cell model")
    solid = np.asarray(solid, dtype=bool)
    floors = floor_cells(solid)
    deep = floors.copy()
    for k in range(1, cells_up + 1):
        above = np.zeros_like(solid)
        above[:-k] = ~solid[k:]  # the cell k above is air (and inside the box)
        deep &= above
    z0 = np.broadcast_to(np.asarray(z0, dtype=float), floors.shape)
    local, matched = np.zeros(floors.shape), np.zeros(floors.shape)
    local[floors & ~deep] = 2.0 * drag_coefficient(z0[floors & ~deep])
    matched[deep] = 2.0 * drag_coefficient(z0[deep], height=cells_up + Z1)
    return local, matched


def rough_lid_drag(solid: np.ndarray, z0) -> np.ndarray:
    """The drag field lambda = 2 C_d in the top layer, under a free-slip lid: a ceiling with
    the same wall model as a floor (module docstring), for a channel closed at the top such as
    the cube's. The lid mirrors packets halfway above the top layer's centre, as a slip floor
    does halfway below the first cell's, so C_d is the same as a floor's of the same z0. Use
    with Params(free_slip_top=True), which needs the top layer to be all air."""
    solid = np.asarray(solid, dtype=bool)
    if solid[-1].any():
        raise ValueError("under a free-slip lid the top layer must be all air")
    out = np.zeros(solid.shape)
    out[-1] = 2.0 * drag_coefficient(np.broadcast_to(np.asarray(z0, dtype=float),
                                                     solid.shape[1:]))  # fmt: skip
    return out


def matched_lid_drag(solid: np.ndarray, z0, cells_down: int) -> tuple[np.ndarray, np.ndarray]:
    """The lid's wall model fed from `cells_down` cells below the top layer: matched_floor_drag
    turned upside down, for a channel closed at the top (rough_lid_drag).

    Returns (local, matched): `matched` holds 2 C_d(h), h = cells_down + 1/2, in the top layer
    where at least cells_down more cells of air lie below it, for Solver(match_drag=...) with
    Params(match_cells=cells_down) (lbm.matched_wind reads the wind below for the top layer);
    `local` holds the first-cell drag of the rest of the top layer, for Solver(drag=...).
    """
    if cells_down < 1:
        raise ValueError("cells_down must be at least 1; 0 is the first-cell model")
    solid = np.asarray(solid, dtype=bool)
    if solid[-1].any():
        raise ValueError("under a free-slip lid the top layer must be all air")
    deep = ~solid[-1 - cells_down : -1].any(axis=0)
    z0 = np.broadcast_to(np.asarray(z0, dtype=float), solid.shape[1:])
    local, matched = np.zeros(solid.shape), np.zeros(solid.shape)
    local[-1][~deep] = 2.0 * drag_coefficient(z0[~deep])
    matched[-1][deep] = 2.0 * drag_coefficient(z0[deep], height=cells_down + Z1)
    return local, matched


def friction_velocity(u1: float, z0: float, kappa: float = KAPPA) -> float:
    """u* from the wind u1 in the first cell above a floor of roughness length z0 (cells)."""
    return kappa * u1 / math.log(Z1 / z0)
