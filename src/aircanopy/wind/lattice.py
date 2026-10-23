"""Velocity sets for the 3D lattice Boltzmann method: D3Q19 and D3Q27.

A velocity set lists the directions c_i in which packets of air move in one time step, each to a
neighbouring cell or staying put, with a weight w_i for each direction. Lattice units are used
throughout: the cell size and the time step are both 1.

The weights make the moments of the equilibrium those of a gas whose speed of sound squared is
c_s^2 = 1/3 (the isotropy conditions; Krüger et al. 2017):

    sum_i w_i = 1,    sum_i w_i c_ia c_ib = c_s^2 d_ab,
    sum_i w_i c_ia c_ib c_ic c_id = c_s^4 (d_ab d_cd + d_ac d_bd + d_ad d_bc),

with every odd moment zero (d is the Kronecker delta). For D3Q19 these conditions fix the
weights: 1/3 at rest, 1/18 towards the 6 faces, 1/36 towards the 12 edges. D3Q27 adds the 8
corners; its weights are the products of the one-dimensional weights 2/3 (rest) and 1/6
(either way) along the three axes, which also meet the conditions. tests/test_wind_lattice.py
checks all of this.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

CS2 = 1.0 / 3.0  # speed of sound squared, in lattice units


@dataclass(frozen=True, eq=False)
class VelocitySet:
    name: str
    c: np.ndarray  # (Q, 3) integers: (cx, cy, cz)
    w: np.ndarray  # (Q,)
    opp: np.ndarray  # (Q,) index of the direction -c_i
    mirror_z: np.ndarray  # (Q,) index of (cx, cy, -cz), for a free-slip top

    @property
    def q(self) -> int:
        return len(self.w)


def _velocity_set(name: str, max_norm2: int) -> VelocitySet:
    """Directions with |c|^2 <= max_norm2, ordered rest, faces, edges, corners."""
    norm2_of = lambda d: sum(x * x for x in d)  # noqa: E731
    dirs = [d for d in itertools.product((0, 1, -1), repeat=3) if norm2_of(d) <= max_norm2]
    dirs.sort(key=norm2_of)
    c = np.array(dirs, dtype=np.int64)
    norm2 = (c * c).sum(axis=1)
    if max_norm2 == 2:
        w = np.choose(norm2, [1 / 3, 1 / 18, 1 / 36])
    else:
        w = np.prod(np.where(c == 0, 2 / 3, 1 / 6), axis=1)
    index = {tuple(d): i for i, d in enumerate(dirs)}
    opp = np.array([index[(-x, -y, -z)] for x, y, z in dirs])
    mirror_z = np.array([index[(x, y, -z)] for x, y, z in dirs])
    return VelocitySet(name, c, w, opp, mirror_z)


D3Q19 = _velocity_set("D3Q19", 2)
D3Q27 = _velocity_set("D3Q27", 3)
LATTICES = {lat.name: lat for lat in (D3Q19, D3Q27)}
