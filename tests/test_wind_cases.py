"""Flows with exact answers, on small grids (the full study is benchmarks/wind_exact.py)."""

import itertools

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


def test_3d_taylor_green_starts_right_and_loses_energy():
    run = cases.taylor_green_3d(D3Q19, 16, t_end=2.0, samples_per_tc=4)
    energy = run["energy"]
    # <u^2 + v^2> / 2 = (1/8 + 1/8) / 2 for the starting vortices, in units of V0^2.
    assert energy[0] == pytest.approx(0.125, rel=2e-3)
    assert all(b < a for a, b in itertools.pairwise(energy))
    assert all(d > 0 for d in cases.dissipation(run["times"], energy))


def test_mean_kinetic_energy_matches_the_fields():
    import numpy as np

    from aircanopy.wind.solver import Params, Solver

    solid = np.zeros((6, 5, 4), bool)
    solid[0] = True
    s = Solver(D3Q19, solid, Params(tau0=0.6, g=(1e-5, 0, 0)), "jax", np.float64)
    rng = np.random.default_rng(10)
    s.set_flow(1 + 0.01 * rng.standard_normal(solid.shape),
               tuple(0.02 * rng.standard_normal(solid.shape) for _ in range(3)))  # fmt: skip
    s.run(3)
    rho, u = s.fields()
    expected = (0.5 * rho * (u[0] ** 2 + u[1] ** 2 + u[2] ** 2))[~solid].mean()
    assert s.mean_kinetic_energy() == pytest.approx(expected, rel=1e-12)
