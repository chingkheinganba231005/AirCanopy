"""Flows with exact answers, on small grids (the full study is benchmarks/wind_exact.py)."""

import jax
import pytest

from aircanopy.wind import cases
from aircanopy.wind.lattice import D3Q19, D3Q27

jax.config.update("jax_enable_x64", True)


def converges(results: list[dict], sizes: list[int]) -> float:
    """The observed order of convergence between two grids."""
    (order,) = cases.observed_order(sizes, [r["relative_error"] for r in results])
    return order


@pytest.mark.parametrize("lat", [D3Q19, D3Q27], ids=lambda lat: lat.name)
def test_a_shear_wave_decays_at_the_rate_its_viscosity_gives(lat):
    runs = [cases.shear_wave(lat, n) for n in (16, 32)]
    assert runs[1]["relative_error"] < 0.01
    assert 1.8 <= converges(runs, [16, 32]) <= 2.2


def test_poiseuille_flow_is_a_parabola():
    runs = [cases.poiseuille(D3Q19, h) for h in (8, 16)]
    assert runs[1]["relative_error"] < 0.01
    assert 1.8 <= converges(runs, [8, 16]) <= 2.2


def test_taylor_green_vortex_decays_as_the_exact_solution():
    # Slow enough that the gas's compressibility stays below the error being measured.
    runs = [cases.taylor_green(D3Q19, n, u0=0.0025 * 16 / n) for n in (16, 32)]
    assert runs[1]["relative_error"] < 0.01
    assert converges(runs, [16, 32]) >= 1.8
