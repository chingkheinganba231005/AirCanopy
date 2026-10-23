"""The ground: elevation models, the triangulated surface, and sites on slopes."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from aircanopy.geo import BBox, LocalFrame
from aircanopy.site import build
from aircanopy.site.mesh import ground_mesh, scene
from aircanopy.site.model import Site
from aircanopy.site.terrain import Terrain, flat, from_geotiff
from aircanopy.site.voxel import BUILDING, GROUND, grid_for_site, voxelise
from aircanopy.sources import osm
from aircanopy.sources.terrain import copernicus_tiles

from .site_helpers import building, empty_site, square

DATA = Path(__file__).parent / "data"
TERRAIN_SAMPLE = DATA / "terrain" / "hku_small_gedtm30.tif"
OSM_SAMPLE = DATA / "osm" / "hku_small.osm.gz"
SAMPLE_BOX = BBox(114.1360, 22.2832, 114.1380, 22.2848)


def plane(a=50.0, bx=0.3, by=-0.1, x0=-200.0, y0=-200.0, dx=20.0, n=21) -> Terrain:
    xs = x0 + np.arange(n) * dx
    ys = y0 + np.arange(n) * dx
    xx, yy = np.meshgrid(xs, ys)
    e = (a + bx * xx + by * yy).astype(np.float32)
    return Terrain(x0, y0, dx, e, np.full_like(e, np.nan), "test", "terrain")


def test_the_surface_reproduces_a_plane_exactly():
    t = plane()
    rng = np.random.default_rng(1)
    x = rng.uniform(-200, 200, 500)
    y = rng.uniform(-200, 200, 500)
    assert t.at(x, y) == pytest.approx(50 + 0.3 * x - 0.1 * y, abs=1e-4)


def test_the_surface_is_the_drawn_triangles():
    # A bumpy ground: on every drawn triangle, the surface at the triangle's centre must be
    # the mean of its corners (true only if `at` uses exactly the triangles of `faces`).
    rng = np.random.default_rng(2)
    t = plane()
    t.elevation = rng.uniform(0, 30, t.elevation.shape).astype(np.float32)
    mesh = ground_mesh(t)
    corners = mesh.vertices[mesh.faces]  # [triangle, corner, xyz]
    centres = corners.mean(axis=1)
    assert t.at(centres[:, 0], centres[:, 1]) == pytest.approx(centres[:, 2], abs=1e-4)
    # Counter-clockwise seen from above: every triangle's normal points up.
    assert (mesh.face_normals[:, 2] > 0).all()


def test_a_building_on_a_slope_stands_on_its_lowest_ground():
    t = plane(a=50, bx=0.3, by=0.0)
    b = building("b", square(10, -10, 20), 30.0)  # from x = 10 to 30
    assert t.lowest_under(b.footprint) == pytest.approx(50 + 0.3 * 10, abs=1e-4)


def test_resampling_a_geotiff_in_degrees(tmp_path):
    lon0, lat0, res = 114.0, 22.0, 1 / 3600
    w, h = 60, 60
    transform = from_origin(lon0 - 30 * res, lat0 + 30 * res, res, res)
    rows, cols = np.mgrid[0:h, 0:w]
    lon = lon0 - 30 * res + (cols + 0.5) * res  # pixel centres
    lat = lat0 + 30 * res - (rows + 0.5) * res
    elevation = 100 + 2000 * (lon - lon0) + 3000 * (lat - lat0)
    path = tmp_path / "dem.tif"
    profile = dict(
        driver="GTiff", width=w, height=h, count=2, dtype="float32", crs="EPSG:4326",
        transform=transform,
    )  # fmt: skip
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(np.stack([elevation, np.full_like(elevation, 1.5)]).astype(np.float32))
    frame = LocalFrame(lon0, lat0)
    t = from_geotiff(path, frame, (-300, -300, 300, 300), dx=50, model="m", kind="terrain")
    xs, ys = t.nodes()
    xx, yy = np.meshgrid(xs, ys)
    lon_n, lat_n = frame.to_lonlat(xx.ravel(), yy.ravel())
    expected = 100 + 2000 * (np.asarray(lon_n) - lon0) + 3000 * (np.asarray(lat_n) - lat0)
    # A field linear in degrees is linear between pixel centres, so resampling is exact
    # (to float32 rounding).
    assert t.elevation.ravel() == pytest.approx(expected, abs=1e-3)
    assert np.allclose(t.std, 1.5)
    with pytest.raises(ValueError, match="does not cover"):
        from_geotiff(path, frame, (-2000, -300, 300, 300), dx=50)


def test_flat_ground_and_saving(tmp_path):
    t = flat((-100, -50, 100, 50), dx=30)
    assert t.at([0, 99], [0, -49]).tolist() == [0.0, 0.0]
    assert t.nodes()[0][-1] >= 100 and t.nodes()[1][-1] >= 50
    t.save(tmp_path / "t.npz")
    again = Terrain.load(tmp_path / "t.npz")
    assert again.model == "flat" and np.array_equal(again.elevation, t.elevation)


def test_copernicus_tile_names():
    names = [u.rsplit("/", 1)[1] for u in copernicus_tiles(BBox(113.9, 22.2, 114.1, 22.3))]
    assert names == [
        "Copernicus_DSM_COG_10_N22_00_E113_00_DEM.tif",
        "Copernicus_DSM_COG_10_N22_00_E114_00_DEM.tif",
    ]
    (south_west,) = copernicus_tiles(BBox(-0.5, -33.9, -0.4, -33.8))
    assert south_west.endswith("Copernicus_DSM_COG_10_S34_00_W001_00_DEM.tif")


def test_voxels_on_sloping_ground():
    t = plane(a=50, bx=0.2, by=0.0)
    site = empty_site(200)
    b = building("b", square(-10, -10, 20), 30.0)
    b.ground = t.lowest_under(b.footprint)
    site.buildings = [b]
    grid = grid_for_site(site, 2.0, terrain=t)
    solid = voxelise(site, grid, t)
    xmin = site.extent()[0]
    assert grid.z0 <= 50 + 0.2 * xmin  # starts below the lowest ground
    assert grid.z0 + grid.nz * grid.dx >= b.ground + 30  # reaches the top of the building
    # In a column away from the building, the ground cells are those below the surface.
    j, i = 5, 5
    xc, yc = grid.centers("x")[i], grid.centers("y")[j]
    surface = float(t.at(xc, yc))
    column = solid[:, j, i]
    assert (column == GROUND).sum() == (grid.centers("z") < surface).sum()
    # The building's top cell is just below ground + height.
    jb = int((0 - grid.y0) // grid.dx)
    ib = int((0 - grid.x0) // grid.dx)
    top = np.nonzero(solid[:, jb, ib] == BUILDING)[0].max()
    assert grid.centers("z")[top] < b.ground + 30 <= grid.centers("z")[top] + grid.dx


def test_site_built_on_real_terrain(tmp_path):
    terrain, record, note = build.load_terrain(SAMPLE_BOX, terrain_file=TERRAIN_SAMPLE)
    assert note is None and record["sha256"]
    summary = build.build_from_osm_data(
        "hku_small", SAMPLE_BOX, osm.parse(OSM_SAMPLE), tmp_path, 2.0, [record],
        terrain=terrain,
    )  # fmt: skip
    t = summary["terrain"]
    assert t["model"] == "gedtm30" and t["kind"] == "terrain"
    # The small HKU box lies on the hillside, tens of metres above the sea.
    assert 0 < t["ground_min_m"] < t["ground_max_m"] < 300
    assert t["std_mean_m"] > 0
    assert summary["voxels"]["ground_cells"] > 0
    site = Site.load(tmp_path / "site.json")
    assert all(b.ground is not None for b in site.buildings)
    saved = Terrain.load(tmp_path / "terrain.npz")
    assert saved.model == "gedtm30"
    # The drawn ground reaches about a kilometre beyond the site on every side.
    xmin, _, xmax, _ = site.extent()
    ground = scene(site, saved).geometry["ground"]
    assert ground.bounds[0][0] <= xmin - 990 and ground.bounds[1][0] >= xmax + 990
