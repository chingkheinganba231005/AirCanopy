"""Turning buildings into a voxel grid."""

import numpy as np
import pytest

from aircanopy.site.voxel import (
    BUILDING,
    Grid,
    block_volume,
    grid_for_site,
    load_voxels,
    save_voxels,
    voxelise_buildings,
)

from .site_helpers import building, empty_site, square

GRID = Grid(x0=-100.0, y0=-100.0, z0=0.0, dx=2.0, nx=100, ny=100, nz=20)


def test_a_block_fills_exactly_its_cells():
    site = empty_site()
    site.buildings = [building("b", square(-50, -50, 20), 10.0)]
    solid = voxelise_buildings(site, GRID)
    assert solid.dtype == np.uint8 and solid.shape == (20, 100, 100)
    assert int(solid.sum()) == 10 * 10 * 5
    # Cell (k=0, j=25, i=25) has its centre at (-49, -49, 1): inside.
    assert solid[0, 25, 25] == BUILDING
    # Centre at (-29, -49): just outside the east wall at x = -30.
    assert solid[0, 25, 35] == 0


def test_raised_part_and_half_open_heights():
    site = empty_site()
    site.buildings = [building("p", square(0, 0, 20), 25.0, base=5.0)]
    solid = voxelise_buildings(site, GRID)
    column = solid[:, 55, 55]
    assert column.tolist() == [0, 0] + [1] * 10 + [0] * 8  # 5 m to 25 m: 10 cells of 2 m


def test_courtyard_stays_open():
    site = empty_site()
    outer = square(0, 0, 30)
    hole = square(10, 10, 10)
    site.buildings = [building("c", outer.difference(hole), 6.0)]
    solid = voxelise_buildings(site, GRID)
    assert int(solid.sum()) == (15 * 15 - 5 * 5) * 3
    assert solid[0, 57, 57] == 0  # centre (15, 15), in the courtyard


def test_buildings_without_height_are_not_drawn_into_the_grid():
    site = empty_site()
    site.buildings = [building("u", square(0, 0, 10), None)]
    assert voxelise_buildings(site, GRID).sum() == 0


def test_grid_for_site_covers_the_box_and_the_tallest_building():
    site = empty_site(200)
    site.buildings = [building("b", square(0, 0, 10), 31.0)]
    g = grid_for_site(site, 2.0)
    xmin, ymin, xmax, ymax = site.extent()
    assert g.x0 == xmin and g.y0 == ymin
    assert g.nx * g.dx >= xmax - xmin and g.ny * g.dx >= ymax - ymin
    assert g.nz == 16  # 31 m on 2 m cells


def test_voxels_round_trip(tmp_path):
    site = empty_site()
    site.buildings = [building("b", square(-50, -50, 20), 10.0)]
    solid = voxelise_buildings(site, GRID)
    save_voxels(tmp_path / "v.npz", GRID, solid)
    g, s = load_voxels(tmp_path / "v.npz")
    assert g == GRID and np.array_equal(s, solid)


def test_grid_rejects_bad_cell_size():
    with pytest.raises(ValueError):
        grid_for_site(empty_site(), 0)


def test_block_volume_counts_overlaps_once():
    site = empty_site()
    a = building("a", square(0, 0, 10), 10.0)
    # Overlaps a by 4 m x 10 m. Its walls are on cell faces (even metres on this grid): a wall
    # exactly through a row of cell centres would leave those cells out, since a point on the
    # boundary is not inside.
    b = building("b", square(6, 0, 10), 20.0)
    raised = building("r", square(0, 0, 10), 30.0, base=20.0)  # sits on top of a
    assert block_volume([a]) == 1000
    assert block_volume([a, b]) == 1000 + 2000 - 400
    assert block_volume([a, raised]) == 1000 + 1000
    assert block_volume([building("u", square(0, 0, 10), None)]) == 0
    site.buildings = [a, b, raised]
    assert int(voxelise_buildings(site, GRID).sum()) * 8 == block_volume(site.buildings)
