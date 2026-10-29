"""The cube on the floor of a channel (wind/cube.py): its layout, units, walls and measures."""

import math

import numpy as np
import pytest

from aircanopy.wind import cube, walls
from aircanopy.wind.inflow import Precursor
from aircanopy.wind.lattice import CS2, D3Q27


def test_the_case_is_laid_out_as_planned():
    c = cube.Cube(h=8)
    solid = cube.site_solid(c)
    assert solid.shape == (17, 56, 112)  # 2H + the floor layer, 7H wide, 3H + H + 10H long
    assert cube.channel(c, c.cells(c.precursor)).shape == (17, 56, 48)
    assert solid[0].all() and not solid[-1].any()
    (x0, x1), (y0, y1) = c.cube_x(), c.cube_y()
    assert (x0, x1, y0, y1) == (24, 32, 24, 32)
    assert solid[1:9, y0:y1, x0:x1].all() and solid[1:].sum() == 8**3
    assert np.array_equal(solid, solid[:, ::-1])  # symmetric about the middle plane
    rows = c.symmetry_rows()
    assert (rows.start, rows.stop) == (27, 29) and rows.start + rows.stop == solid.shape[1]


def test_the_units_follow_from_the_reynolds_number_and_the_measured_friction():
    c = cube.Cube(h=16)
    assert CS2 * (c.tau0 - 0.5) == pytest.approx(c.u_bulk * c.h / c.reynolds)
    assert c.g * c.h == pytest.approx((c.u_star_over_bulk * c.u_bulk) ** 2)
    # The roughness estimate: a log-law wind from the wall to the middle has the bulk velocity.
    assert (math.log(c.h / c.z0) - 1) / walls.KAPPA == pytest.approx(1 / c.u_star_over_bulk)
    assert cube.Cube(h=32).z0 == pytest.approx(2 * c.z0)  # a length in H, the same at any h
    assert cube.Cube(h=16, z0_over_h=1e-4).z0 == pytest.approx(1.6e-3)
    assert [cube.Cube(h=h).match_cells for h in (8, 16, 32, 64)] == [1, 3, 6, 12]
    with pytest.raises(ValueError, match="even"):
        cube.Cube(h=15)
    with pytest.raises(ValueError, match="too small"):
        cube.Cube(h=4)


def test_the_floor_reads_the_first_cell_near_the_cube_and_higher_up_elsewhere():
    c = cube.Cube(h=8)
    solid = cube.site_solid(c)
    near = c.cube_x()[0] - c.cells(c.first_cell_from)
    drag, match_drag = cube.wall_drag(c, solid, near)
    floor, lid = walls.floor_cells(solid), np.zeros(solid.shape, bool)
    lid[-1] = True
    first_cell = 2 * walls.drag_coefficient(c.z0)
    read_up = 2 * walls.drag_coefficient(c.z0, height=c.match_cells + 0.5)
    assert np.allclose(match_drag[floor & (np.arange(112) < near)], read_up)
    assert np.allclose(drag[floor & (np.arange(112) >= near)], first_cell)
    assert drag[c.h + 1, c.cube_y()[0], c.cube_x()[0]] == first_cell  # the cube's top
    assert np.allclose(match_drag[lid], read_up) and not drag[lid].any()
    assert not (drag[~floor] != 0).any() and not (match_drag[~(floor | lid)] != 0).any()
    pre_drag, pre_match = cube.wall_drag(c, cube.channel(c, 48))
    assert not pre_drag.any() and np.allclose(pre_match[1], read_up)


def test_the_boxes_fit_together_and_the_air_stops_at_the_cube():
    """The precursor feeds the site (inflow.Precursor accepts them), and within a few steps the
    air just in front of the cube's face slows down, while the wind at the inlet does not."""
    c = cube.Cube(h=6)
    pre = cube.precursor_box(c, D3Q27, "numpy", np.float64)
    z = np.arange(2 * c.h + 1) - 0.5
    wind = np.where(z > 0, c.u_bulk, 0.0)
    site = cube.site_box(c, D3Q27, wind, "numpy", np.float64)
    for box in (pre, site):
        shape = box.solid.shape
        box.set_equilibrium(np.ones(shape), (np.broadcast_to(wind[:, None, None], shape),
                                             np.zeros(shape), np.zeros(shape)))  # fmt: skip
    both = Precursor(pre, site)
    both.run(6)
    _, (ux, _, _) = site.fields(c.symmetry_rows())
    x0 = c.cube_x()[0]
    middle = c.h // 2
    assert np.isfinite(ux).all()
    assert ux[middle, :, x0 - 1].mean() < 0.5 * c.u_bulk
    assert ux[middle, :, 1].mean() == pytest.approx(c.u_bulk, rel=0.02)
    assert not ux[1 : c.h + 1, :, x0 : x0 + c.h].any()  # no wind inside the cube


def test_a_zero_crossing_is_found_by_linear_interpolation():
    x = np.arange(5.0)
    u = np.array([1.0, 0.5, -0.5, -1.0, 2.0])
    assert cube.zero_crossing(x, u, 0, +1) == pytest.approx(1.5)
    assert cube.zero_crossing(x, u, 4, -1) == pytest.approx(3 + 1 / 3)
    assert cube.zero_crossing(x, np.ones(5), 0, +1) is None


def test_the_separation_lengths_skip_the_corner_eddies():
    """A made-up wind on the floor along the symmetry line: it turns back 1.04 H in front of the
    cube and comes forward again 1.612 H behind it, with a small eddy turning the other way in
    each corner at the foot of the cube, which the walk from outside must not take."""
    c = cube.Cube(h=8)
    n = c.cells(c.upstream + 1 + c.downstream)
    x = np.arange(n) + 0.5
    (x0, x1) = c.cube_x()
    separation, reattachment = x0 - 1.04 * c.h, x1 + 1.612 * c.h
    u = np.zeros(n)
    front, back = x < x0, x >= x1
    u[front] = np.clip((separation - x[front]) / 4, -0.3, 1.0)
    u[back] = np.clip((x[back] - reattachment) / 4, -0.3, 1.0)
    u[(x > x0 - 1.5) & front] = 0.1  # the corner eddies
    u[(x < x1 + 1.5) & back] = 0.1
    lengths = cube.separation_lengths(c, u)
    assert lengths["x_F"] == pytest.approx(1.04) and lengths["x_R"] == pytest.approx(1.612)
    u[front] = -0.1  # the flow runs back all the way from 2H in front of the cube
    assert cube.separation_lengths(c, u)["x_F"] is None
