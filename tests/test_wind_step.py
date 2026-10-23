"""The lattice Boltzmann step: conservation, forcing, walls, free-slip top, NumPy against JAX."""

import jax
import numpy as np
import pytest

from aircanopy.wind import lbm
from aircanopy.wind.lattice import CS2, D3Q19, D3Q27
from aircanopy.wind.solver import Params, Solver

jax.config.update("jax_enable_x64", True)

LATS = [D3Q19, D3Q27]
ids = lambda lat: lat.name  # noqa: E731


def perturbed(shape, seed=0, du=0.02):
    """A random state near rest: density 1 +- 1 %, velocity up to `du` in lattice units."""
    rng = np.random.default_rng(seed)
    rho = 1.0 + 0.01 * rng.standard_normal(shape)
    u = tuple(du * rng.standard_normal(shape) for _ in range(3))
    return rho, u


@pytest.mark.parametrize("lat", LATS, ids=ids)
def test_equilibrium_has_the_moments_of_a_gas(lat):
    rho, u = perturbed((3, 4, 5), seed=1)
    feq = lbm.equilibrium(np, lat, rho, u)
    c = lat.c.astype(float)
    assert np.allclose(feq.sum(axis=0), rho, atol=1e-14)
    for a in range(3):
        assert np.allclose(np.einsum("i,i...->...", c[:, a], feq), rho * u[a], atol=1e-14)
        for b in range(3):
            flux = np.einsum("i,i...->...", c[:, a] * c[:, b], feq)
            expected = rho * u[a] * u[b] + (rho * CS2 if a == b else 0.0)
            assert np.allclose(flux, expected, atol=1e-14)


@pytest.mark.parametrize("lat", LATS, ids=ids)
@pytest.mark.parametrize("regularised", [True, False])
def test_uniform_flow_stays_uniform(lat, regularised):
    shape = (4, 5, 6)
    s = Solver(lat, np.zeros(shape, bool), Params(tau0=0.6, regularised=regularised), "numpy",
               np.float64)  # fmt: skip
    s.set_equilibrium(
        np.ones(shape), (np.full(shape, 0.05), np.full(shape, -0.02), 0 * np.ones(shape))
    )
    s.run(10)
    rho, (ux, uy, uz) = s.fields()
    assert np.allclose(rho, 1.0, atol=1e-13) and np.allclose(ux, 0.05, atol=1e-13)
    assert np.allclose(uy, -0.02, atol=1e-13) and np.allclose(uz, 0.0, atol=1e-13)


@pytest.mark.parametrize("lat", LATS, ids=ids)
@pytest.mark.parametrize("smagorinsky", [0.0, 0.17])
def test_mass_and_momentum_are_conserved_in_a_periodic_box(lat, smagorinsky):
    shape = (6, 7, 8)
    s = Solver(lat, np.zeros(shape, bool), Params(tau0=0.52, smagorinsky=smagorinsky),
               "numpy", np.float64)  # fmt: skip
    rho0, u0 = perturbed(shape, seed=2)
    s.set_equilibrium(rho0, u0)
    mass0 = rho0.sum()
    momentum0 = [(rho0 * c).sum() for c in u0]
    s.run(20)
    rho, u = s.fields()
    assert rho.sum() == pytest.approx(mass0, rel=1e-13)
    for a in range(3):
        assert (rho * u[a]).sum() == pytest.approx(momentum0[a], abs=1e-12)


@pytest.mark.parametrize("lat", LATS, ids=ids)
@pytest.mark.parametrize("regularised", [True, False])
def test_a_body_force_accelerates_uniform_air_at_exactly_g(lat, regularised):
    shape = (3, 3, 3)
    g = (1e-5, -2e-5, 3e-6)
    s = Solver(lat, np.zeros(shape, bool), Params(tau0=0.7, g=g, regularised=regularised),
               "numpy", np.float64)  # fmt: skip
    s.set_equilibrium(
        np.ones(shape), (np.full(shape, 0.01), 0 * np.ones(shape), 0 * np.ones(shape))
    )
    s.run(50)
    _, u = s.fields()
    assert np.allclose(u[0], 0.01 + 50 * g[0], atol=1e-15)
    assert np.allclose(u[1], 50 * g[1], atol=1e-15) and np.allclose(u[2], 50 * g[2], atol=1e-15)


@pytest.mark.parametrize("lat", LATS, ids=ids)
def test_walls_keep_the_air_in(lat):
    shape = (8, 7, 9)
    solid = np.ones(shape, bool)
    solid[1:-1, 1:-1, 1:-1] = False
    solid[3:5, 2:4, 4] = True  # an obstacle in the middle
    s = Solver(lat, solid, Params(tau0=0.55, smagorinsky=0.1), "numpy", np.float64)
    rho0, u0 = perturbed(shape, seed=3)
    s.set_equilibrium(rho0, u0)
    fluid = ~solid
    mass0 = rho0[fluid].sum()
    s.run(30)
    rho, _ = s.fields()
    assert rho[fluid].sum() == pytest.approx(mass0, rel=1e-13)


@pytest.mark.parametrize("lat", LATS, ids=ids)
@pytest.mark.parametrize("regularised", [True, False])
def test_free_slip_top_is_a_mirror_plane(lat, regularised):
    """A channel with walls at the bottom and the top is symmetric about its middle, so a
    half channel with a free-slip lid in the middle must reproduce its lower half exactly."""
    h, ny, nx = 5, 3, 4
    full = np.zeros((2 * h + 2, ny, nx), bool)
    full[0] = full[-1] = True
    half = np.zeros((h + 1, ny, nx), bool)
    half[0] = True
    g = (2e-5, 0.0, 0.0)
    params = dict(tau0=0.6, smagorinsky=0.1, g=g, regularised=regularised)
    a = Solver(lat, full, Params(**params), "numpy", np.float64)
    b = Solver(lat, half, Params(**params, free_slip_top=True), "numpy", np.float64)
    rng = np.random.default_rng(4)
    ux = 0.01 * rng.standard_normal((h, ny, nx))
    uz = 0.01 * rng.standard_normal((h, ny, nx))
    zero = np.zeros((1, ny, nx))
    # The full channel's upper half is the mirror image of its lower half: uz changes sign.
    ux_full = np.concatenate([zero, ux, ux[::-1], zero])
    uz_full = np.concatenate([zero, uz, -uz[::-1], zero])
    a.set_equilibrium(np.ones(full.shape), (ux_full, 0 * ux_full, uz_full))
    b.set_equilibrium(
        np.ones(half.shape), (np.concatenate([zero, ux]), 0 * half, np.concatenate([zero, uz]))
    )
    a.run(40)
    b.run(40)
    rho_a, u_a = a.fields()
    rho_b, u_b = b.fields()
    assert np.abs(rho_a[1 : h + 1] - rho_b[1:]).max() < 1e-13
    for k in range(3):
        assert np.abs(u_a[k][1 : h + 1] - u_b[k][1:]).max() < 1e-13
    assert np.abs(u_b[0][1:]).max() > 1e-3  # the flow is not trivially zero


def test_free_slip_needs_air_on_top_and_ground_below():
    solid = np.zeros((4, 3, 3), bool)
    with pytest.raises(ValueError, match="free-slip"):
        Solver(D3Q19, solid, Params(tau0=0.6, free_slip_top=True), "numpy")


def test_smagorinsky_relaxation_time_solves_its_defining_equation():
    rng = np.random.default_rng(5)
    p = [1e-3 * rng.standard_normal(50) for _ in range(6)]
    rho = 1.0 + 0.01 * rng.standard_normal(50)
    tau0, cs = 0.5001, 0.17
    tau = lbm.smagorinsky_tau(np, p, rho, tau0, cs)
    # nu = nu0 + Cs^2 |S|, with |S| = sqrt(2 S:S) and S = -P / (2 rho c_s^2 tau).
    pp = p[0] ** 2 + p[1] ** 2 + p[2] ** 2 + 2 * (p[3] ** 2 + p[4] ** 2 + p[5] ** 2)
    s_norm = np.sqrt(2.0 * pp) / (2.0 * rho * CS2 * tau)
    assert np.allclose(CS2 * (tau - 0.5), CS2 * (tau0 - 0.5) + cs**2 * s_norm, rtol=1e-12)
    assert np.allclose(lbm.smagorinsky_tau(np, p, rho, tau0, 0.0), tau0)


@pytest.mark.parametrize("lat", LATS, ids=ids)
def test_numpy_and_jax_agree(lat):
    shape = (6, 5, 7)
    solid = np.zeros(shape, bool)
    solid[0] = True
    solid[1:3, 2, 3] = True
    params = Params(tau0=0.51, smagorinsky=0.15, g=(1e-5, 0.0, -1e-6), free_slip_top=True)
    rho0, u0 = perturbed(shape, seed=6)
    out = []
    for backend in ("numpy", "jax"):
        s = Solver(lat, solid, params, backend, np.float64)
        s.set_equilibrium(rho0, u0)
        s.run(15)
        out.append(np.asarray(s.f))
    assert np.abs(out[0] - out[1]).max() < 1e-13
