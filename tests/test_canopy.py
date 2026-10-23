"""Tree canopy: the map's tiles, resampling, and finding individual trees."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from aircanopy.geo import BBox, LocalFrame
from aircanopy.site import build
from aircanopy.site.canopy import (
    MIN_CROWN_DIAMETER_M,
    TREE_CANOPY_MIN_M,
    Canopy,
    find_crowns,
    from_geotiff,
    remove_buildings,
    trees_from_canopy,
)
from aircanopy.site.model import Site
from aircanopy.sources import osm
from aircanopy.sources.canopy import quadkey, tile_xy, tiles_for_bbox

from .site_helpers import building, square, tree

DATA = Path(__file__).parent / "data"
CANOPY_SAMPLE = DATA / "canopy" / "hku_small_meta_wri_v1.tif"
OSM_SAMPLE = DATA / "osm" / "hku_small.osm.gz"
SAMPLE_BOX = BBox(114.1360, 22.2832, 114.1380, 22.2848)


def test_quadkeys_match_the_maps_own_tiles():
    # Tile names taken from the map's index (tiles.geojson): the centre of each tile's square.
    assert quadkey(*tile_xy(-115.6640625, 33.43)) == "023013213"
    assert quadkey(*tile_xy(129.7265625, -0.35)) == "310111000"
    # The tile holding HKU, which the site build reads.
    assert quadkey(*tile_xy(114.137, 22.284)) == "132122232"


def test_a_box_across_a_tile_edge_needs_both_tiles():
    edge = -180 + 360 * tile_xy(114.137, 22.284)[0] / 2**9  # west edge of HKU's tile
    keys = tiles_for_bbox(BBox(edge - 0.01, 22.28, edge + 0.01, 22.29))
    assert len(keys) == 2 and "132122232" in keys


def disc(h, cx, cy, radius, top, dx=1.0):
    """A dome-shaped crown: `top` m high at (cx, cy), falling to 0 at `radius`."""
    ny, nx = h.shape
    yy, xx = np.mgrid[0:ny, 0:nx] * dx
    r = np.hypot(xx - cx, yy - cy)
    h[:] = np.maximum(h, top * np.sqrt(np.clip(1 - (r / radius) ** 2, 0, None)))


def test_two_separate_crowns_are_found_with_their_tops_and_sizes():
    h = np.zeros((60, 80), np.float32)
    disc(h, 20, 30, 8, 15)
    disc(h, 60, 30, 5, 10)
    c = Canopy(0.5, 0.5, 1.0, h)
    crowns, labels = find_crowns(c)
    crowns.sort(key=lambda k: k.x)
    assert len(crowns) == 2
    big, small = crowns
    assert (big.x, big.y) == pytest.approx((20.5, 30.5), abs=1.0)
    assert big.height == pytest.approx(15, abs=0.5)
    # Only canopy at least 5 m tall counts: radius 8 * sqrt(1 - (5/15)^2), about 7.5 m.
    assert big.diameter == pytest.approx(2 * 8 * np.sqrt(1 - (5 / 15) ** 2), rel=0.1)
    assert small.diameter == pytest.approx(2 * 5 * np.sqrt(1 - (5 / 10) ** 2), rel=0.15)
    assert labels.max() >= 2 and (labels[h < TREE_CANOPY_MIN_M] == 0).all()


def test_touching_crowns_are_split_and_tiny_or_low_ones_are_dropped():
    h = np.zeros((40, 60), np.float32)
    disc(h, 20, 20, 9, 14)
    disc(h, 34, 20, 9, 12)  # overlaps the first: one canopy, two tops
    disc(h, 50, 8, 1.6, 9)  # tall but under 3 m across
    disc(h, 50, 32, 6, 4)  # wide but under 5 m tall
    crowns, _ = find_crowns(Canopy(0.5, 0.5, 1.0, h))
    assert len(crowns) == 2
    assert all(k.diameter >= MIN_CROWN_DIAMETER_M for k in crowns)
    assert sorted(round(k.height) for k in crowns) == [12, 14]


def test_no_canopy_inside_buildings():
    h = np.full((20, 20), 10.0, np.float32)
    c = Canopy(0.5, 0.5, 1.0, h)
    removed = remove_buildings(c, [building("b", square(5, 5, 10), 20.0)])
    assert removed == 100
    assert c.height[5:15, 5:15].max() == 0 and c.height[0, 0] == 10


def test_mapped_trees_lend_their_tags_to_the_crown_they_stand_in():
    h = np.zeros((40, 40), np.float32)
    disc(h, 20, 20, 8, 12)
    c = Canopy(0.5, 0.5, 1.0, h)
    inside = tree(21, 19, None)
    inside.genus, inside.species = "Ficus", "Ficus microcarpa"
    measured = tree(21.5, 20.5, 13.0)  # a second mapped tree in the same crown
    outside = tree(2, 2, None)
    trees, counts = trees_from_canopy(c, [inside, measured, outside])
    assert counts == {"crowns": 1, "mapped_matched": 1, "mapped_alone": 2}
    crown = trees[0]
    assert crown.id == inside.id and crown.genus == "Ficus"
    assert crown.height_source == "canopy:meta-wri-chm-v1" and crown.height == pytest.approx(12)
    assert {t.id for t in trees[1:]} == {measured.id, outside.id}


def test_resampling_a_web_mercator_canopy_map(tmp_path):
    # A 10 m tall square of canopy 20 m on a side, centred 30 m east of the site's origin.
    frame = LocalFrame(114.0, 22.0)
    to_map = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    lon, lat = frame.to_lonlat([-100.0], [100.0])
    mx0, my0 = to_map.transform(lon[0], lat[0])
    res = 1.2
    data = np.zeros((200, 200), np.uint8)
    lon_c, lat_c = frame.to_lonlat([30.0], [0.0])
    mxc, myc = to_map.transform(lon_c[0], lat_c[0])
    col, row = (mxc - mx0) / res, (my0 - myc) / res
    # On the map, 20 m on the ground is 20 / cos(latitude) metres of Web Mercator.
    half = 10 / np.cos(np.radians(22.0)) / res
    data[int(row - half) : int(row + half), int(col - half) : int(col + half)] = 10
    path = tmp_path / "chm.tif"
    profile = dict(
        driver="GTiff", width=200, height=200, count=1, dtype="uint8", crs="EPSG:3857",
        transform=from_origin(mx0, my0, res, res),
    )  # fmt: skip
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(data, 1)
    c = from_geotiff(path, frame, (-50, -50, 80, 50))
    xs, ys = c.centres()
    tall = c.height >= 10
    assert xs[tall.any(axis=0)].min() == pytest.approx(20, abs=1.5)
    assert xs[tall.any(axis=0)].max() == pytest.approx(40, abs=1.5)
    assert ys[tall.any(axis=1)].min() == pytest.approx(-10, abs=1.5)
    assert tall.sum() == pytest.approx(400, rel=0.15)


def test_site_with_trees_from_the_real_sample(tmp_path):
    frame = LocalFrame.for_bbox(SAMPLE_BOX)
    canopy, record, note = build.load_canopy(
        SAMPLE_BOX, frame, frame.bbox_extent(SAMPLE_BOX), canopy_file=CANOPY_SAMPLE
    )
    assert note is None and record["licence"] == "CC-BY-4.0"
    assert canopy.imagery_dates == ["2018-04-03"]
    summary = build.build_from_osm_data(
        "hku_small", SAMPLE_BOX, osm.parse(OSM_SAMPLE), tmp_path, 2.0, [record],
        canopy=canopy,
    )  # fmt: skip
    c = summary["canopy"]
    assert c["crowns"] > 0 and 0 < c["cover_share"] < 1
    site = Site.load(tmp_path / "site.json")
    found = [t for t in site.trees if t.crown_source == "canopy:meta-wri-chm-v1"]
    assert len(found) == c["crowns"]
    assert len(site.trees) == c["crowns"] + c["mapped_alone"]
    assert all(t.height >= TREE_CANOPY_MIN_M and t.ground is not None for t in found)
    saved = Canopy.load(tmp_path / "canopy.npz")
    assert saved.imagery_dates == ["2018-04-03"] and saved.height.shape == canopy.height.shape
