"""Metres and seconds to lattice units, and back.

The solver works in lattice units: one cell is 1 long and one time step lasts 1. A run in
metres and seconds is put on the lattice by choosing the cell size dx (the voxel size of the
site) and the time step dt. Then

    speed:         u_lattice  = u dt / dx
    viscosity:     nu_lattice = nu dt / dx^2
    acceleration:  a_lattice  = a dt^2 / dx

The relaxation time of the collision follows from the lattice viscosity:
nu_lattice = c_s^2 (tau - 1/2).

The method behaves like an incompressible fluid only while the flow is slow compared with the
lattice's speed of sound, c_s = sqrt(1/3) = 0.577: its errors grow with the square of the Mach
number u_lattice / c_s (Krüger et al. 2017). So dt is chosen from the fastest wind expected in a
run, which may move at most MAX_LATTICE_SPEED cells per step (assumption A-009).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .lattice import CS2

MAX_LATTICE_SPEED = 0.1  # cells per step for the fastest expected wind (A-009): Mach 0.17


@dataclass(frozen=True)
class Units:
    dx: float  # metres per cell
    dt: float  # seconds per step

    @classmethod
    def for_wind(cls, dx: float, u_max: float, u_lattice: float = MAX_LATTICE_SPEED) -> Units:
        """Units in which a wind of `u_max` m/s moves `u_lattice` cells per step."""
        if not 0 < u_lattice < math.sqrt(CS2):
            raise ValueError(f"a lattice speed of {u_lattice} is not below the speed of sound")
        return cls(dx, u_lattice * dx / u_max)

    def speed(self, u: float) -> float:
        return u * self.dt / self.dx

    def speed_si(self, u_lattice: float) -> float:
        return u_lattice * self.dx / self.dt

    def viscosity(self, nu: float) -> float:
        return nu * self.dt / self.dx**2

    def viscosity_si(self, nu_lattice: float) -> float:
        return nu_lattice * self.dx**2 / self.dt

    def acceleration(self, a: float) -> float:
        return a * self.dt**2 / self.dx

    def steps(self, seconds: float) -> int:
        return round(seconds / self.dt)


def tau_for_viscosity(nu_lattice: float) -> float:
    return 0.5 + nu_lattice / CS2


def viscosity_for_tau(tau: float) -> float:
    return CS2 * (tau - 0.5)


def mach(u_lattice: float) -> float:
    return u_lattice / math.sqrt(CS2)
