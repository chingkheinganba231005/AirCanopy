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


def drag_coefficient(z0, kappa: float = KAPPA):
    """C_d = (kappa / ln(z1 / z0))^2 for a roughness length z0 in cells (0 < z0 < 1/2)."""
    z0 = np.asarray(z0, dtype=float)
    if np.any(z0 <= 0) or np.any(z0 >= Z1):
        raise ValueError(f"the roughness length must be between 0 and {Z1} cells")
    return (kappa / np.log(Z1 / z0)) ** 2


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


def friction_velocity(u1: float, z0: float, kappa: float = KAPPA) -> float:
    """u* from the wind u1 in the first cell above a floor of roughness length z0 (cells)."""
    return kappa * u1 / math.log(Z1 / z0)
