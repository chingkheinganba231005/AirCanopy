"""Turbulent inflow: a periodic precursor feeds the site's inlet, step by step."""

import jax
import numpy as np
import pytest

from aircanopy.wind import walls
from aircanopy.wind.inflow import Precursor
from aircanopy.wind.lattice import CS2, D3Q19, D3Q27
from aircanopy.wind.solver import Params, Solver

from .test_wind_step import DEFAULTS, pair_ids, perturbed

jax.config.update("jax_enable_x64", True)


def boxes(lat, collision, backend="numpy", wall_model=False, nx_pre=8, nx_site=12):
    """A precursor and a site with the same cross-section: ground at the bottom, a free-slip lid,
    a body force; with `wall_model`, slip floors and the wall model reading 2 cells up."""
    nz, ny = 7, 5
    solid_pre = np.zeros((nz, ny, nx_pre), bool)
    solid_pre[0] = True
    solid_site = np.zeros((nz, ny, nx_site), bool)
    solid_site[0] = True
    common = dict(tau0=0.52, smagorinsky=0.15, g=(1e-5, 0.0, 0.0), free_slip_top=True,
                  collision=collision, slip_floors=wall_model,
                  match_cells=2 if wall_model else 0)  # fmt: skip
    kw_pre, kw_site = {}, {}
    if wall_model:
        kw_pre["match_drag"] = walls.matched_floor_drag(solid_pre, 0.05, 2)[1]
        kw_site["match_drag"] = walls.matched_floor_drag(solid_site, 0.05, 2)[1]
    pre = Solver(lat, solid_pre, Params(**common), backend, np.float64, **kw_pre)
    site = Solver(lat, solid_site, Params(**common, open_x=True), backend, np.float64,
                  inlet_u=(0.03, 0.0, 0.0), **kw_site)  # fmt: skip
    rho, u = perturbed(solid_pre.shape, seed=13)
    pre.set_equilibrium(rho, (0.03 + u[0], u[1], u[2]))
    rho, u = perturbed(solid_site.shape, seed=14)
    site.set_equilibrium(rho, (0.03 + u[0], u[1], u[2]))
    return pre, site


@pytest.mark.parametrize(("lat", "collision"), DEFAULTS, ids=pair_ids)
@pytest.mark.parametrize("wall_model", [False, True], ids=["walls", "wall-model"])
def test_the_inlet_gets_the_precursors_wind(lat, collision, wall_model):
    """The inlet takes the velocity the precursor's collision starts from, and the wall model's
    drag depends only on velocities, so the site's inlet wind equals the precursor's wind on its
    plane, to round-off, even with the wall model."""
    pre, site = boxes(lat, collision, wall_model=wall_model)
    both = Precursor(pre, site, plane=3)
    both.run(12)
    _, u_pre = pre.fields()
    _, u_site = site.fields()
    for a in range(3):
        assert np.abs(u_site[a][..., 0] - u_pre[a][..., 3]).max() < 1e-14
    assert np.abs(u_pre[0][1:, :, 3]).max() > 1e-3  # the wind is not trivially zero
    assert pre.time == site.time == 12


@pytest.mark.parametrize(("lat", "collision"), DEFAULTS, ids=pair_ids)
def test_numpy_and_jax_agree_for_the_coupled_step(lat, collision):
    out = []
    for backend in ("numpy", "jax"):
        pre, site = boxes(lat, collision, backend, wall_model=True)
        Precursor(pre, site).run(10)
        out.append((np.asarray(pre.f), np.asarray(site.f)))
    assert np.abs(out[0][0] - out[1][0]).max() < 1e-13
    assert np.abs(out[0][1] - out[1][1]).max() < 1e-13


@pytest.mark.parametrize(("lat", "collision"), DEFAULTS, ids=pair_ids)
def test_a_developed_channel_flows_from_the_precursor_through_the_site(lat, collision):
    """A periodic channel driven by a body force feeds an open copy of itself: the site keeps
    the precursor's profile, as in Step 13's test with a fixed inlet."""
    h, nx, tau, u_max = 8, 32, 0.8, 0.05
    g = (u_max * 8.0 * CS2 * (tau - 0.5) / h**2, 0.0, 0.0)
    solid_pre = np.zeros((h + 2, 1, 1), bool)
    solid_pre[0] = solid_pre[-1] = True
    solid_site = np.zeros((h + 2, 1, nx), bool)
    solid_site[0] = solid_site[-1] = True
    params = dict(tau0=tau, g=g, collision=collision)
    pre = Solver(lat, solid_pre, Params(**params), "jax", np.float64)
    z = np.arange(h + 2) - 0.5
    start = np.where((z > 0) & (z < h), 4.0 * u_max * z * (h - z) / h**2, 0.0)[:, None, None]
    pre.set_equilibrium(np.ones(start.shape), (start, 0 * start, 0 * start))
    pre.run(20_000)
    profile = pre.fields()[1][0][:, :, 0]
    site = Solver(lat, solid_site, Params(**params, open_x=True), "jax", np.float64,
                  inlet_u=(profile, 0.0, 0.0))  # fmt: skip
    shape = solid_site.shape
    site.set_equilibrium(np.ones(shape), (np.broadcast_to(profile[..., None], shape),
                                          np.zeros(shape), np.zeros(shape)))  # fmt: skip
    Precursor(pre, site).run(2000)
    change = np.abs(site.fields()[1][0] - profile[..., None])[1:-1].max(axis=(0, 1)) / u_max
    assert change.max() < 3e-3
    assert np.abs(pre.fields()[1][0][:, :, 0] - profile).max() < 1e-12  # the precursor stays put


def test_the_boxes_must_fit_together():
    pre, site = boxes(D3Q27, "recursive")
    with pytest.raises(ValueError, match="periodic"):
        Precursor(site, pre)
    with pytest.raises(ValueError, match="inside the precursor"):
        Precursor(pre, site, plane=8)
    solid = np.zeros((7, 4, 12), bool)
    solid[0] = True
    narrow = Solver(D3Q27, solid, Params(tau0=0.52, smagorinsky=0.15, g=(1e-5, 0.0, 0.0),
                                         free_slip_top=True, open_x=True),
                    "numpy", np.float64, inlet_u=(0.03, 0.0, 0.0))  # fmt: skip
    with pytest.raises(ValueError, match="cross-section"):
        Precursor(pre, narrow)
    other = Solver(D3Q19, site.solid, Params(tau0=0.52, collision="regularised", open_x=True,
                                             free_slip_top=True),
                   "numpy", np.float64, inlet_u=(0.03, 0.0, 0.0))  # fmt: skip
    with pytest.raises(ValueError, match="lattice"):
        Precursor(pre, other)
    _, matched = boxes(D3Q27, "recursive", wall_model=True)
    with pytest.raises(ValueError, match="same collision"):
        Precursor(pre, matched)
