"""Open boundaries: a velocity inlet, a pressure outlet, and the absorbing layer before it."""

import jax
import numpy as np
import pytest

from aircanopy.wind.lattice import CS2, D3Q19, D3Q27
from aircanopy.wind.solver import Params, Solver

from .test_wind_step import DEFAULTS, PAIRS, pair_ids

jax.config.update("jax_enable_x64", True)


def channel(h: int, nx: int) -> np.ndarray:
    """Walls below and above h cells of air; periodic across y."""
    solid = np.zeros((h + 2, 1, nx), bool)
    solid[0] = solid[-1] = True
    return solid


def parabola(h: int, u_max: float) -> np.ndarray:
    """Poiseuille's profile at the cell centres, with the walls halfway to the solid cells."""
    z = np.arange(h + 2) - 0.5
    return np.where((z > 0) & (z < h), 4.0 * u_max * z * (h - z) / h**2, 0.0)


@pytest.mark.parametrize(("lat", "collision"), PAIRS, ids=pair_ids)
@pytest.mark.parametrize("layer", [0, 8])
def test_a_uniform_wind_crosses_an_open_box_unchanged(lat, collision, layer):
    # At an angle to the boundaries: the extended equilibrium differs from the second-order
    # one only when the wind has more than one component, and the inlet once used the wrong
    # one (JOURNAL.md, Step 13).
    shape = (3, 4, 20)
    u0 = (0.05, 0.01, -0.02)
    params = Params(tau0=0.6, collision=collision, open_x=True, absorb_cells=layer,
                    absorb_strength=0.2 if layer else 0.0)  # fmt: skip
    s = Solver(lat, np.zeros(shape, bool), params, "numpy", np.float64, inlet_u=u0)
    s.set_equilibrium(np.ones(shape), tuple(np.full(shape, c) for c in u0))
    s.run(30)
    rho, u = s.fields()
    assert np.abs(rho - 1.0).max() < 1e-13
    for a in range(3):
        assert np.abs(u[a] - u0[a]).max() < 1e-13


@pytest.mark.parametrize(("lat", "collision"), DEFAULTS, ids=pair_ids)
def test_numpy_and_jax_agree_with_open_boundaries(lat, collision):
    shape = (6, 5, 16)
    solid = np.zeros(shape, bool)
    solid[0] = True
    solid[1:3, 2, 7] = True
    params = Params(tau0=0.51, smagorinsky=0.15, g=(1e-5, 0.0, 0.0), free_slip_top=True,
                    collision=collision, open_x=True, absorb_cells=5,
                    absorb_strength=0.1)  # fmt: skip
    rng = np.random.default_rng(12)
    inlet = (0.04 + 0.01 * rng.standard_normal(shape[:2]), 0.005 * rng.standard_normal(shape[:2]),
             0.0)  # fmt: skip
    out = []
    for backend in ("numpy", "jax"):
        s = Solver(lat, solid, params, backend, np.float64, inlet_u=inlet)
        s.set_equilibrium(np.ones(shape), (np.full(shape, 0.04), 0 * solid, 0 * solid))
        s.run(15)
        out.append(np.asarray(s.f))
    assert np.abs(out[0] - out[1]).max() < 1e-13


@pytest.mark.parametrize(("lat", "collision"), DEFAULTS, ids=pair_ids)
def test_an_open_channel_keeps_a_developed_flow(lat, collision):
    """A channel driven by a body force, fed at the inlet with the developed profile of the
    same channel made periodic, must keep that profile all the way to the outlet. The bounce-
    back of the non-equilibrium part is exact only for its second-order part, so the columns
    next to the boundaries differ a little."""
    h, nx, tau, u_max = 8, 32, 0.8, 0.05
    g = (u_max * 8.0 * CS2 * (tau - 0.5) / h**2, 0.0, 0.0)
    periodic = Solver(lat, channel(h, 1), Params(tau0=tau, g=g, collision=collision), "jax",
                      np.float64)  # fmt: skip
    start = parabola(h, u_max)[:, None, None]
    periodic.set_equilibrium(np.ones(start.shape), (start, 0 * start, 0 * start))
    periodic.run(20_000)
    profile = periodic.fields()[1][0][:, :, 0]
    params = Params(tau0=tau, g=g, collision=collision, open_x=True)
    s = Solver(lat, channel(h, nx), params, "jax", np.float64, inlet_u=(profile, 0.0, 0.0))
    shape = (h + 2, 1, nx)
    s.set_equilibrium(np.ones(shape), (np.broadcast_to(profile[..., None], shape),
                                       np.zeros(shape), np.zeros(shape)))  # fmt: skip
    s.run(2000)
    rho, u = s.fields()
    error = np.abs(u[0] - profile[..., None])[1:-1].max(axis=(0, 1)) / u_max
    assert error.max() < 3e-3  # next to the inlet and the outlet
    assert error[8:-8].max() < 1e-4  # in the middle
    assert np.abs(rho - 1.0)[1:-1].max() < 3e-4


@pytest.mark.parametrize(("lat", "collision"), DEFAULTS, ids=pair_ids)
def test_an_open_channel_driven_by_pressure_follows_poiseuille(lat, collision):
    """Without a force, the outlet's fixed density and the inlet's fixed velocity make the
    pressure drop that drives the flow. Poiseuille's law, dp/dx = -12 nu m / h^2 with m the
    mean mass flux rho u across the channel, and p = c_s^2 rho, gives the density's slope."""
    h, nx, tau, u_max = 8, 32, 0.8, 0.05
    nu = CS2 * (tau - 0.5)
    inlet = parabola(h, u_max)[:, None]
    params = Params(tau0=tau, collision=collision, open_x=True)
    s = Solver(lat, channel(h, nx), params, "jax", np.float64, inlet_u=(inlet, 0.0, 0.0))
    shape = (h + 2, 1, nx)
    s.set_equilibrium(np.ones(shape), (np.broadcast_to(inlet[..., None], shape),
                                       np.zeros(shape), np.zeros(shape)))  # fmt: skip
    s.run(5000)
    rho, u = s.fields()
    flux = (rho * u[0])[1:-1, 0].sum(axis=0)
    assert flux.max() / flux.min() - 1.0 < 1e-3  # the same through every section
    middle = slice(8, nx - 8)
    slope = np.polyfit(np.arange(nx)[middle], rho[1:-1, 0, middle].mean(axis=0), 1)[0]
    expected = -12.0 * nu * flux[middle].mean() / h / h**2 / CS2
    assert slope == pytest.approx(expected, rel=0.02)
    assert rho[1:-1, 0, -1] == pytest.approx(1.0, abs=1e-12)  # the outlet's density


@pytest.mark.parametrize(("lat", "collision"), DEFAULTS, ids=pair_ids)
def test_a_sound_pulse_leaves_through_the_absorbing_layer(lat, collision):
    """Sound reflects off both ends of an open box: the inlet holds the velocity, the outlet
    the density. The absorbing layer must take it out."""
    nx, u0 = 96, 0.05
    x = np.arange(nx)
    shape = (1, 1, nx)
    rho0 = 1.0 + 1e-3 * np.exp(-((x - 48.0) ** 2) / 18.0) + 0 * np.zeros(shape)

    def energy(s):
        rho, u = s.fields()
        return float((CS2 * (rho - 1) ** 2 + rho * ((u[0] - u0) ** 2 + u[1] ** 2)).sum())

    left = []
    for layer in (0, 16):
        params = Params(tau0=0.5001, collision=collision, open_x=True, absorb_cells=layer,
                        absorb_strength=0.2 if layer else 0.0)  # fmt: skip
        s = Solver(lat, np.zeros(shape, bool), params, "numpy", np.float64, inlet_u=(u0, 0, 0))
        s.set_equilibrium(rho0, (np.full(shape, u0), np.zeros(shape), np.zeros(shape)))
        start = energy(s)
        s.run(400)
        left.append(energy(s) / start)
    assert left[0] > 0.3  # reflected, and still in the box
    assert left[1] < 1e-3


def test_open_boundaries_need_their_settings():
    solid = np.zeros((3, 3, 6), bool)
    with pytest.raises(ValueError, match="inlet_u"):
        Solver(D3Q27, solid, Params(tau0=0.6, open_x=True), "numpy")
    with pytest.raises(ValueError, match="inlet_u"):
        Solver(D3Q27, solid, Params(tau0=0.6), "numpy", inlet_u=(0.05, 0, 0))
    with pytest.raises(ValueError, match="absorbing"):
        Solver(D3Q19, solid, Params(tau0=0.6, collision="bgk", absorb_cells=2), "numpy")
