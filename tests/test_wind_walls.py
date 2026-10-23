"""Free-slip floors, the drag field and the wall model."""

import math

import jax
import numpy as np
import pytest

from aircanopy.wind import lbm, walls
from aircanopy.wind.lattice import D3Q19, D3Q27
from aircanopy.wind.solver import Params, Solver

jax.config.update("jax_enable_x64", True)

LATS = [D3Q19, D3Q27]
ids = lambda lat: lat.name  # noqa: E731


def test_drag_coefficient_from_the_law_of_the_wall():
    assert walls.drag_coefficient(0.05) == pytest.approx((0.40 / math.log(0.5 / 0.05)) ** 2)
    with pytest.raises(ValueError, match="roughness length"):
        walls.drag_coefficient(0.5)
    # The friction velocity undoes it: rho u*^2 = rho C_d U1^2.
    u1, z0 = 0.05, 0.02
    assert walls.friction_velocity(u1, z0) ** 2 == pytest.approx(walls.drag_coefficient(z0) * u1**2)


def test_floor_cells_are_the_air_just_above_solid():
    solid = np.zeros((5, 4, 6), bool)
    solid[0] = True
    solid[1:3, 1, 2] = True  # a block two cells tall
    floors = walls.floor_cells(solid)
    assert floors[1].sum() == 4 * 6 - 1  # the first layer, except where the block stands
    assert floors[3, 1, 2] and floors.sum() == 4 * 6
    lam = walls.rough_floor_drag(solid, 0.05)
    assert set(np.unique(lam)) == {0.0, 2 * walls.drag_coefficient(0.05)}
    assert (lam[~floors] == 0).all()


@pytest.mark.parametrize("lat", LATS, ids=ids)
def test_air_glides_over_a_free_slip_floor(lat):
    # Without drag, a uniform wind over a slip floor under a free-slip top feels no friction.
    shape = (6, 4, 5)
    solid = np.zeros(shape, bool)
    solid[0] = True
    s = Solver(lat, solid, Params(tau0=0.6, free_slip_top=True, slip_floors=True), "numpy",
               np.float64)  # fmt: skip
    s.set_equilibrium(np.ones(shape), (np.full(shape, 0.05), np.full(shape, 0.02), 0 * solid))
    s.run(30)
    _, (ux, uy, uz) = s.fields()
    air = ~solid
    assert np.abs(ux[air] - 0.05).max() < 1e-14 and np.abs(uy[air] - 0.02).max() < 1e-14
    assert np.abs(uz[air]).max() < 1e-14


@pytest.mark.parametrize("lat", LATS, ids=ids)
def test_a_slip_floor_is_a_mirror_plane(lat):
    """A box periodic in height 2h, whose flow is mirror-symmetric about two planes, must be
    reproduced exactly by its half: a slip floor below and a free-slip top above."""
    h, ny, nx = 4, 3, 5
    rng = np.random.default_rng(11)
    ux = 0.01 * rng.standard_normal((h, ny, nx))
    uz = 0.01 * rng.standard_normal((h, ny, nx))
    full = Solver(lat, np.zeros((2 * h, ny, nx), bool), Params(tau0=0.6, smagorinsky=0.1),
                  "numpy", np.float64)  # fmt: skip
    ux_full = np.concatenate([ux, ux[::-1]])
    uz_full = np.concatenate([uz, -uz[::-1]])
    full.set_equilibrium(np.ones(ux_full.shape), (ux_full, 0 * ux_full, uz_full))
    solid = np.zeros((h + 1, ny, nx), bool)
    solid[0] = True
    half = Solver(lat, solid, Params(tau0=0.6, smagorinsky=0.1, free_slip_top=True,
                                     slip_floors=True), "numpy", np.float64)  # fmt: skip
    zero = np.zeros((1, ny, nx))
    half.set_equilibrium(np.ones(solid.shape), (np.concatenate([zero, ux]), 0 * solid,
                                                 np.concatenate([zero, uz])))  # fmt: skip
    full.run(25)
    half.run(25)
    _, u_full = full.fields()
    _, u_half = half.fields()
    for k in range(3):
        assert np.abs(u_full[k][:h] - u_half[k][1:]).max() < 1e-13
    assert np.abs(u_half[0][1:]).max() > 1e-4  # the flow is not trivially zero


@pytest.mark.parametrize("lat", LATS, ids=ids)
@pytest.mark.parametrize("regularised", [True, False])
def test_drag_is_the_same_through_moments(lat, regularised):
    shape = (4, 5, 6)
    rng = np.random.default_rng(12)
    rho = 1 + 0.01 * rng.standard_normal(shape)
    u = tuple(0.03 * rng.standard_normal(shape) for _ in range(3))
    f = lbm.equilibrium(np, lat, rho, u) * (1 + 0.01 * rng.standard_normal((lat.q, *shape)))
    solid = np.zeros(shape, bool)
    cfg = {"solid": solid, "fluid": ~solid, "tau0": 0.52, "cs": 0.17, "g": (1e-5, 0.0, 2e-6),
           "drag": 0.5 * rng.random(shape)}  # fmt: skip
    for forced in (True, False):
        direct = lbm.collide(np, lat, f, cfg, regularised, forced, dragged=True)
        moments = lbm.collide_moments(np, lat, f, cfg, regularised, forced, dragged=True)
        assert np.abs(direct - moments).max() < 1e-15


def test_the_velocity_read_back_with_drag_is_the_one_set():
    shape = (3, 4, 5)
    rng = np.random.default_rng(13)
    drag = 0.8 * rng.random(shape)
    s = Solver(D3Q19, np.zeros(shape, bool), Params(tau0=0.7, g=(1e-5, 0, 0)), "numpy",
               np.float64, drag=drag)  # fmt: skip
    rho = 1 + 0.01 * rng.standard_normal(shape)
    u = tuple(0.05 * rng.standard_normal(shape) for _ in range(3))
    s.set_equilibrium(rho, u)
    rho_back, u_back = s.fields()
    assert np.abs(rho_back - rho).max() < 1e-14
    assert max(np.abs(u_back[k] - u[k]).max() for k in range(3)) < 1e-14


def test_floor_friction_balances_the_push_of_the_wind():
    """In a steady channel, the floor's friction holds back the force on the whole column:
    lambda / 2 U1^2 = g h for h cells of air, so U1 = sqrt(2 g h / lambda)."""
    h = 8
    solid = np.zeros((h + 1, 2, 2), bool)
    solid[0] = True
    z0, g = 0.05, 2e-6
    drag = walls.rough_floor_drag(solid, z0)
    s = Solver(D3Q19, solid, Params(tau0=0.8, g=(g, 0, 0), free_slip_top=True, slip_floors=True),
               "jax", np.float64, drag=drag)  # fmt: skip
    lam = 2 * walls.drag_coefficient(z0)
    expected = math.sqrt(2 * g * h / lam)
    # Held back only by a quadratic drag, the column settles slowly (in about 1 / sqrt(g lambda /
    # 2h), some 10 000 steps here); starting near the answer, run until it stops changing.
    s.set_equilibrium(np.ones(solid.shape), (np.full(solid.shape, expected), 0 * solid, 0 * solid))
    previous = 0.0
    for _ in range(100):
        s.run(5000)
        _, (ux, _, _) = s.fields()
        if abs(ux[1].mean() - previous) < 1e-12:
            break
        previous = ux[1].mean()
    assert ux[1].mean() == pytest.approx(expected, rel=1e-6)
    assert np.all(np.diff(ux[1:, 0, 0]) > 0)  # faster with height, as in a boundary layer
