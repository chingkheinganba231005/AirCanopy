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
#: Every collision on every lattice that allows it; the first of each lattice is its default.
PAIRS = [(D3Q19, "regularised"), (D3Q19, "bgk"), (D3Q27, "recursive"), (D3Q27, "regularised"),
         (D3Q27, "bgk")]  # fmt: skip
DEFAULTS = [(D3Q19, "regularised"), (D3Q27, "recursive")]
pair_ids = lambda pair: f"{pair[0].name}-{pair[1]}" if isinstance(pair, tuple) else str(pair)  # noqa: E731


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


@pytest.mark.parametrize(("lat", "collision"), PAIRS, ids=pair_ids)
def test_uniform_flow_stays_uniform(lat, collision):
    shape = (4, 5, 6)
    s = Solver(lat, np.zeros(shape, bool), Params(tau0=0.6, collision=collision), "numpy",
               np.float64)  # fmt: skip
    s.set_equilibrium(
        np.ones(shape), (np.full(shape, 0.05), np.full(shape, -0.02), 0 * np.ones(shape))
    )
    s.run(10)
    rho, (ux, uy, uz) = s.fields()
    assert np.allclose(rho, 1.0, atol=1e-13) and np.allclose(ux, 0.05, atol=1e-13)
    assert np.allclose(uy, -0.02, atol=1e-13) and np.allclose(uz, 0.0, atol=1e-13)


@pytest.mark.parametrize(("lat", "collision"), PAIRS, ids=pair_ids)
@pytest.mark.parametrize("smagorinsky", [0.0, 0.17])
def test_mass_and_momentum_are_conserved_in_a_periodic_box(lat, collision, smagorinsky):
    shape = (6, 7, 8)
    params = Params(tau0=0.52, smagorinsky=smagorinsky, collision=collision)
    s = Solver(lat, np.zeros(shape, bool), params, "numpy", np.float64)
    rho0, u0 = perturbed(shape, seed=2)
    s.set_equilibrium(rho0, u0)
    mass0 = rho0.sum()
    momentum0 = [(rho0 * c).sum() for c in u0]
    s.run(20)
    rho, u = s.fields()
    assert rho.sum() == pytest.approx(mass0, rel=1e-13)
    for a in range(3):
        assert (rho * u[a]).sum() == pytest.approx(momentum0[a], abs=1e-12)


@pytest.mark.parametrize(("lat", "collision"), PAIRS, ids=pair_ids)
def test_a_body_force_accelerates_uniform_air_at_exactly_g(lat, collision):
    shape = (3, 3, 3)
    g = (1e-5, -2e-5, 3e-6)
    s = Solver(lat, np.zeros(shape, bool), Params(tau0=0.7, g=g, collision=collision),
               "numpy", np.float64)  # fmt: skip
    s.set_equilibrium(
        np.ones(shape), (np.full(shape, 0.01), 0 * np.ones(shape), 0 * np.ones(shape))
    )
    s.run(50)
    _, u = s.fields()
    assert np.allclose(u[0], 0.01 + 50 * g[0], atol=1e-15)
    assert np.allclose(u[1], 50 * g[1], atol=1e-15) and np.allclose(u[2], 50 * g[2], atol=1e-15)


@pytest.mark.parametrize(("lat", "collision"), DEFAULTS, ids=pair_ids)
def test_walls_keep_the_air_in(lat, collision):
    shape = (8, 7, 9)
    solid = np.ones(shape, bool)
    solid[1:-1, 1:-1, 1:-1] = False
    solid[3:5, 2:4, 4] = True  # an obstacle in the middle
    params = Params(tau0=0.55, smagorinsky=0.1, collision=collision)
    s = Solver(lat, solid, params, "numpy", np.float64)
    rho0, u0 = perturbed(shape, seed=3)
    s.set_equilibrium(rho0, u0)
    fluid = ~solid
    mass0 = rho0[fluid].sum()
    s.run(30)
    rho, _ = s.fields()
    assert rho[fluid].sum() == pytest.approx(mass0, rel=1e-13)


@pytest.mark.parametrize(("lat", "collision"), PAIRS, ids=pair_ids)
def test_free_slip_top_is_a_mirror_plane(lat, collision):
    """A channel with walls at the bottom and the top is symmetric about its middle, so a
    half channel with a free-slip lid in the middle must reproduce its lower half exactly."""
    h, ny, nx = 5, 3, 4
    full = np.zeros((2 * h + 2, ny, nx), bool)
    full[0] = full[-1] = True
    half = np.zeros((h + 1, ny, nx), bool)
    half[0] = True
    g = (2e-5, 0.0, 0.0)
    params = dict(tau0=0.6, smagorinsky=0.1, g=g, collision=collision)
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


@pytest.mark.parametrize(("lat", "collision"), DEFAULTS, ids=pair_ids)
def test_numpy_and_jax_agree(lat, collision):
    shape = (6, 5, 7)
    solid = np.zeros(shape, bool)
    solid[0] = True
    solid[1:3, 2, 3] = True
    params = Params(tau0=0.51, smagorinsky=0.15, g=(1e-5, 0.0, -1e-6), free_slip_top=True,
                    collision=collision)  # fmt: skip
    rho0, u0 = perturbed(shape, seed=6)
    out = []
    for backend in ("numpy", "jax"):
        s = Solver(lat, solid, params, backend, np.float64)
        s.set_equilibrium(rho0, u0)
        s.run(15)
        out.append(np.asarray(s.f))
    assert np.abs(out[0] - out[1]).max() < 1e-13


@pytest.mark.parametrize(("lat", "collision"), PAIRS, ids=pair_ids)
@pytest.mark.parametrize("forced", [True, False])
def test_the_collision_through_moments_is_the_same_collision(lat, collision, forced):
    shape = (4, 5, 6)
    solid = np.zeros(shape, bool)
    solid[0, 1:3, 2] = True
    rng = np.random.default_rng(7)
    f = lbm.equilibrium(np, lat, *perturbed(shape, seed=8))
    f = f * (1.0 + 0.01 * rng.standard_normal(f.shape))  # some non-equilibrium content
    cfg = {"solid": solid, "fluid": ~solid, "tau0": 0.52, "cs": 0.17, "g": (1e-5, -2e-6, 3e-6)}
    direct = lbm.collide(np, lat, f, cfg, collision, forced)
    moments = lbm.collide_moments(np, lat, f, cfg, collision, forced)
    # The recursive collision is built two different ways (Hermite coefficients, product form).
    assert np.abs(direct - moments).max() < (1e-14 if collision == "recursive" else 1e-15)


@pytest.mark.parametrize(("lat", "collision"), DEFAULTS, ids=pair_ids)
def test_a_float32_run_stays_float32_with_64_bit_enabled(lat, collision):
    # NumPy's float64 scalars are not "weakly typed" in JAX: one of them in the step would
    # silently turn a float32 run into a float64 run (it did: the lattice weights).
    solid = np.zeros((4, 4, 4), bool)
    solid[0] = True
    params = Params(tau0=0.6, smagorinsky=0.1, g=(1e-6, 0, 0), collision=collision)
    s = Solver(lat, solid, params, "jax", np.float32)
    s.set_flow(*perturbed((4, 4, 4), seed=9))
    assert s.f.dtype == np.float32
    s.run(3)
    assert s.f.dtype == np.float32


def test_the_extended_equilibrium_has_the_right_third_moments():
    """Its point: sum f c_a c_a c_b = rho (c_s^2 u_b + u_a^2 u_b), as for a real gas, where the
    second-order equilibrium misses the u_a^2 u_b part (lbm module docstring)."""
    rho, u = perturbed((3, 4, 5), seed=10, du=0.05)
    c = D3Q27.c.astype(float)
    eq = lbm.product_terms(np, D3Q27, rho, u)[0]
    second = lbm.equilibrium(np, D3Q27, rho, u)
    for f, exact in ((eq, True), (second, False)):
        assert np.allclose(f.sum(axis=0), rho, atol=1e-14)
        for a in range(3):
            assert np.allclose(np.einsum("i,i...->...", c[:, a], f), rho * u[a], atol=1e-14)
        for a, b in [(0, 1), (1, 2), (2, 0), (0, 2)]:
            m = np.einsum("i,i...->...", c[:, a] ** 2 * c[:, b], f)
            gas = rho * (CS2 * u[b] + u[a] ** 2 * u[b])
            assert np.allclose(m, gas, atol=1e-14) == exact


def test_the_recursive_collision_needs_d3q27():
    with pytest.raises(ValueError, match="D3Q27"):
        Solver(D3Q19, np.zeros((3, 3, 3), bool), Params(tau0=0.6), "numpy")
    with pytest.raises(ValueError, match="unknown collision"):
        Solver(D3Q27, np.zeros((3, 3, 3), bool), Params(tau0=0.6, collision="cumulant"), "numpy")


@pytest.mark.parametrize(("collision", "stable"), [("recursive", True), ("regularised", False)])
def test_noise_in_a_fast_wind_with_almost_no_viscosity(collision, stable):
    """Step 12: in a uniform wind at Mach 0.17 with almost no viscosity, tiny noise dies away
    with the recursive collision and grows without bound with the regularised one."""
    shape = (8, 8, 16)
    s = Solver(D3Q27, np.zeros(shape, bool), Params(tau0=0.500003, collision=collision), "jax",
               np.float64)  # fmt: skip
    rng = np.random.default_rng(11)
    noise = [1e-4 * rng.standard_normal(shape) for _ in range(3)]
    s.set_equilibrium(np.ones(shape), (0.1 + noise[0], noise[1], noise[2]))

    def energy():
        _, u = s.fields()
        return sum(float(((c - c.mean()) ** 2).mean()) for c in u)

    start = energy()
    s.run(2000)
    end = energy()
    if stable:
        assert np.isfinite(end) and end < start
    else:
        assert not np.isfinite(end) or end > 100 * start
