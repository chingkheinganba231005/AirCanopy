"""Free-slip floors, the drag field and the wall model."""

import math

import jax
import numpy as np
import pytest

from aircanopy.wind import lbm, walls
from aircanopy.wind.lattice import D3Q19, D3Q27
from aircanopy.wind.solver import Params, Solver

jax.config.update("jax_enable_x64", True)

#: Every collision on every lattice that allows it; the first of each lattice is its default.
PAIRS = [(D3Q19, "regularised"), (D3Q19, "bgk"), (D3Q27, "recursive"), (D3Q27, "regularised"),
         (D3Q27, "bgk")]  # fmt: skip
DEFAULTS = [(D3Q19, "regularised"), (D3Q27, "recursive")]
pair_ids = lambda pair: f"{pair[0].name}-{pair[1]}" if isinstance(pair, tuple) else str(pair)  # noqa: E731


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


@pytest.mark.parametrize(("lat", "collision"), PAIRS, ids=pair_ids)
def test_air_glides_over_a_free_slip_floor(lat, collision):
    # Without drag, a uniform wind over a slip floor under a free-slip top feels no friction.
    shape = (6, 4, 5)
    solid = np.zeros(shape, bool)
    solid[0] = True
    params = Params(tau0=0.6, free_slip_top=True, slip_floors=True, collision=collision)
    s = Solver(lat, solid, params, "numpy", np.float64)
    s.set_equilibrium(np.ones(shape), (np.full(shape, 0.05), np.full(shape, 0.02), 0 * solid))
    s.run(30)
    _, (ux, uy, uz) = s.fields()
    air = ~solid
    assert np.abs(ux[air] - 0.05).max() < 1e-14 and np.abs(uy[air] - 0.02).max() < 1e-14
    assert np.abs(uz[air]).max() < 1e-14


@pytest.mark.parametrize(("lat", "collision"), PAIRS, ids=pair_ids)
def test_a_slip_floor_is_a_mirror_plane(lat, collision):
    """A box periodic in height 2h, whose flow is mirror-symmetric about two planes, must be
    reproduced exactly by its half: a slip floor below and a free-slip top above."""
    h, ny, nx = 4, 3, 5
    rng = np.random.default_rng(11)
    ux = 0.01 * rng.standard_normal((h, ny, nx))
    uz = 0.01 * rng.standard_normal((h, ny, nx))
    params = {"tau0": 0.6, "smagorinsky": 0.1, "collision": collision}
    full = Solver(lat, np.zeros((2 * h, ny, nx), bool), Params(**params), "numpy", np.float64)
    ux_full = np.concatenate([ux, ux[::-1]])
    uz_full = np.concatenate([uz, -uz[::-1]])
    full.set_equilibrium(np.ones(ux_full.shape), (ux_full, 0 * ux_full, uz_full))
    solid = np.zeros((h + 1, ny, nx), bool)
    solid[0] = True
    half = Solver(lat, solid, Params(**params, free_slip_top=True, slip_floors=True), "numpy",
                  np.float64)  # fmt: skip
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


@pytest.mark.parametrize(("lat", "collision"), PAIRS, ids=pair_ids)
def test_drag_is_the_same_through_moments(lat, collision):
    shape = (4, 5, 6)
    rng = np.random.default_rng(12)
    rho = 1 + 0.01 * rng.standard_normal(shape)
    u = tuple(0.03 * rng.standard_normal(shape) for _ in range(3))
    f = lbm.equilibrium(np, lat, rho, u) * (1 + 0.01 * rng.standard_normal((lat.q, *shape)))
    solid = np.zeros(shape, bool)
    cfg = {"solid": solid, "fluid": ~solid, "tau0": 0.52, "cs": 0.17, "g": (1e-5, 0.0, 2e-6),
           "drag": 0.5 * rng.random(shape)}  # fmt: skip
    for forced in (True, False):
        direct = lbm.collide(np, lat, f, cfg, collision, forced, dragged=True)
        moments = lbm.collide_moments(np, lat, f, cfg, collision, forced, dragged=True)
        assert np.abs(direct - moments).max() < (1e-14 if collision == "recursive" else 1e-15)


@pytest.mark.parametrize(("lat", "collision"), DEFAULTS, ids=pair_ids)
def test_the_velocity_read_back_with_drag_is_the_one_set(lat, collision):
    shape = (3, 4, 5)
    rng = np.random.default_rng(13)
    drag = 0.8 * rng.random(shape)
    params = Params(tau0=0.7, g=(1e-5, 0, 0), collision=collision)
    s = Solver(lat, np.zeros(shape, bool), params, "numpy", np.float64, drag=drag)
    rho = 1 + 0.01 * rng.standard_normal(shape)
    u = tuple(0.05 * rng.standard_normal(shape) for _ in range(3))
    s.set_equilibrium(rho, u)
    rho_back, u_back = s.fields()
    assert np.abs(rho_back - rho).max() < 1e-14
    assert max(np.abs(u_back[k] - u[k]).max() for k in range(3)) < 1e-14


@pytest.mark.parametrize(("lat", "collision"), DEFAULTS, ids=pair_ids)
def test_floor_friction_balances_the_push_of_the_wind(lat, collision):
    """In a steady channel, the floor's friction holds back the force on the whole column:
    lambda / 2 U1^2 = g h for h cells of air, so U1 = sqrt(2 g h / lambda)."""
    h = 8
    solid = np.zeros((h + 1, 2, 2), bool)
    solid[0] = True
    z0, g = 0.05, 2e-6
    drag = walls.rough_floor_drag(solid, z0)
    params = Params(tau0=0.8, g=(g, 0, 0), free_slip_top=True, slip_floors=True,
                    collision=collision)  # fmt: skip
    s = Solver(lat, solid, params, "jax", np.float64, drag=drag)
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


def test_matched_floor_drag_splits_deep_and_shallow_floors():
    solid = np.zeros((7, 3, 4), bool)
    solid[0] = True
    solid[1:5, 1, 2] = True  # a block 4 cells tall: on its top, 2 cells of air remain
    local, matched = walls.matched_floor_drag(solid, 0.05, cells_up=3)
    floors = walls.floor_cells(solid)
    assert matched[1].sum() > 0 and (matched[1][floors[1]] > 0).all()
    assert local[5, 1, 2] == 2 * walls.drag_coefficient(0.05)  # too shallow: first-cell model
    assert matched[5, 1, 2] == 0
    assert set(np.unique(matched)) == {0.0, 2 * walls.drag_coefficient(0.05, height=3.5)}
    assert ((local > 0) & (matched > 0)).sum() == 0 and (
        (local > 0) | (matched > 0)
    ).sum() == floors.sum()
    with pytest.raises(ValueError, match="at least 1"):
        walls.matched_floor_drag(solid, 0.05, cells_up=0)


@pytest.mark.parametrize(("lat", "collision"), PAIRS, ids=pair_ids)
def test_matched_drag_is_the_same_through_moments(lat, collision):
    shape = (6, 4, 5)
    rng = np.random.default_rng(14)
    rho = 1 + 0.01 * rng.standard_normal(shape)
    u = tuple(0.03 * rng.standard_normal(shape) for _ in range(3))
    f = lbm.equilibrium(np, lat, rho, u) * (1 + 0.01 * rng.standard_normal((lat.q, *shape)))
    solid = np.zeros(shape, bool)
    match_drag = np.zeros(shape)
    match_drag[1] = 0.5 * rng.random(shape[1:])
    cfg = {"solid": solid, "fluid": ~solid, "tau0": 0.52, "cs": 0.17, "g": (1e-5, 0.0, 2e-6),
           "drag": 0.0, "match_drag": match_drag}  # fmt: skip
    for forced in (True, False):
        direct = lbm.collide(np, lat, f, cfg, collision, forced, matched=2)
        moments = lbm.collide_moments(np, lat, f, cfg, collision, forced, matched=2)
        assert np.abs(direct - moments).max() < (1e-14 if collision == "recursive" else 1e-15)
        # The force pulls the first layer against the wind two cells above, and nothing else.
        plain = lbm.collide_moments(np, lat, f, {**cfg, "match_drag": 0 * match_drag}, collision,
                                    forced)  # fmt: skip
        changed = np.abs(moments - plain).max(axis=(0, 2, 3))
        assert changed[1] > 1e-6 and np.all(changed[[0, 2, 3, 4, 5]] < 1e-15)


@pytest.mark.parametrize(("lat", "collision"), DEFAULTS, ids=pair_ids)
def test_the_velocity_read_back_with_matched_drag_is_the_one_set(lat, collision):
    shape = (6, 3, 4)
    solid = np.zeros(shape, bool)
    solid[0] = True
    _, match_drag = walls.matched_floor_drag(solid, 0.05, cells_up=2)
    params = Params(tau0=0.7, g=(1e-5, 0, 0), collision=collision, match_cells=2)
    s = Solver(lat, solid, params, "numpy", np.float64, match_drag=match_drag)
    rng = np.random.default_rng(15)
    rho = 1 + 0.01 * rng.standard_normal(shape)
    u = tuple(0.05 * rng.standard_normal(shape) for _ in range(3))
    s.set_equilibrium(rho, u)
    rho_back, u_back = s.fields()
    air = ~solid
    assert np.abs(rho_back - rho)[air].max() < 1e-14
    assert max(np.abs(u_back[k] - u[k])[air].max() for k in range(3)) < 1e-14
    with pytest.raises(ValueError, match="go together"):
        Solver(lat, solid, Params(tau0=0.7, collision=collision), "numpy", match_drag=match_drag)


@pytest.mark.parametrize(("lat", "collision"), DEFAULTS, ids=pair_ids)
def test_matched_floor_friction_balances_the_push_of_the_wind(lat, collision):
    """As the first-cell test, with the stress driven by the wind k cells up: in the steady
    channel lambda_m / 2 U_k^2 = g h, so the wind at the matching cell is sqrt(2 g h / lambda_m)."""
    h, k = 8, 3
    solid = np.zeros((h + 1, 2, 2), bool)
    solid[0] = True
    z0, g = 0.05, 2e-6
    local, match_drag = walls.matched_floor_drag(solid, z0, cells_up=k)
    assert not local.any()
    params = Params(tau0=0.8, g=(g, 0, 0), free_slip_top=True, slip_floors=True,
                    collision=collision, match_cells=k)  # fmt: skip
    s = Solver(lat, solid, params, "jax", np.float64, match_drag=match_drag)
    lam = 2 * walls.drag_coefficient(z0, height=k + 0.5)
    expected = math.sqrt(2 * g * h / lam)
    s.set_equilibrium(np.ones(solid.shape), (np.full(solid.shape, expected), 0 * solid, 0 * solid))
    previous = 0.0
    for _ in range(100):
        s.run(5000)
        _, (ux, _, _) = s.fields()
        if abs(ux[1 + k].mean() - previous) < 1e-12:
            break
        previous = ux[1 + k].mean()
    assert ux[1 + k].mean() == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(("lat", "collision"), DEFAULTS, ids=pair_ids)
def test_numpy_and_jax_agree_with_matched_drag(lat, collision):
    shape = (7, 4, 5)
    solid = np.zeros(shape, bool)
    solid[0] = True
    _, match_drag = walls.matched_floor_drag(solid, 0.05, cells_up=2)
    params = Params(tau0=0.51, smagorinsky=0.15, g=(1e-5, 0.0, 0.0), free_slip_top=True,
                    slip_floors=True, collision=collision, match_cells=2)  # fmt: skip
    rng = np.random.default_rng(16)
    rho0 = 1 + 0.01 * rng.standard_normal(shape)
    u0 = tuple(0.02 * rng.standard_normal(shape) for _ in range(3))
    out = []
    for backend in ("numpy", "jax"):
        s = Solver(lat, solid, params, backend, np.float64, match_drag=match_drag)
        s.set_equilibrium(rho0, u0)
        s.run(12)
        out.append(np.asarray(s.f))
    assert np.abs(out[0] - out[1]).max() < 1e-13
