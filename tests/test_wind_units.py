"""Lattice units."""

import pytest

from aircanopy.wind.units import (
    MAX_LATTICE_SPEED,
    Units,
    mach,
    tau_for_viscosity,
    viscosity_for_tau,
)


def test_wind_of_5_m_s_on_2_m_cells():
    u = Units.for_wind(dx=2.0, u_max=5.0)
    assert u.dt == pytest.approx(MAX_LATTICE_SPEED * 2.0 / 5.0)
    assert u.speed(5.0) == pytest.approx(MAX_LATTICE_SPEED)
    assert u.speed_si(u.speed(3.2)) == pytest.approx(3.2)
    assert u.viscosity_si(u.viscosity(1.5e-5)) == pytest.approx(1.5e-5)
    # 9.81 m/s^2 for one step of 0.04 s moves 9.81 * 0.04^2 m, out of 2 m cells.
    assert u.acceleration(9.81) == pytest.approx(9.81 * 0.04**2 / 2.0)
    assert u.steps(3600) == 90_000


def test_relaxation_time_and_mach_number():
    assert tau_for_viscosity(1 / 6) == pytest.approx(1.0)
    assert viscosity_for_tau(tau_for_viscosity(0.0123)) == pytest.approx(0.0123)
    assert mach(3**-0.5) == pytest.approx(1.0)


def test_a_lattice_speed_at_the_speed_of_sound_is_refused():
    with pytest.raises(ValueError, match="speed of sound"):
        Units.for_wind(dx=1.0, u_max=10.0, u_lattice=0.6)
