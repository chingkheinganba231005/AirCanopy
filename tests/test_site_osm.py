"""Building a site from OpenStreetMap data."""

import math
from pathlib import Path

import pytest

from aircanopy.geo import BBox, LocalFrame
from aircanopy.site.from_osm import INDOOR_REASON, site_from_osm
from aircanopy.site.heights import STOREY_HEIGHT_M
from aircanopy.site.model import Site
from aircanopy.sources import osm

from .osm_builder import LAT0, LON0, OsmBuilder, square

FIXTURE = Path(__file__).parent / "data" / "osm" / "hku_small.osm.gz"


def synthetic_site() -> Site:
    b = OsmBuilder()
    # A plain building 20 m square, 10 m tall.
    b.ring(square(-50, -50, 20), {"building": "yes", "height": "10"})
    # A courtyard building: outer 30 m square drawn as two ways, inner 10 m square.
    p = [b.node(x, y) for x, y in square(10, 10, 30)]
    w1 = b.way([p[0], p[1], p[2]])
    w2 = b.way([p[2], p[3], p[0]])
    inner = b.ring(square(20, 20, 10))
    b.relation(
        [("way", w1, "outer"), ("way", w2, "outer"), ("way", inner, "inner")],
        {"type": "multipolygon", "building": "apartments", "building:levels": "6"},
    )
    # An outline with one detailed part inside: only the part is kept.
    b.ring(square(-50, 20, 20), {"building": "yes", "building:levels": "2"})
    b.ring(square(-45, 25, 10), {"building:part": "yes", "height": "25", "min_height": "5"})
    # An open roof (petrol-station canopy) and a roof part: skipped and reported.
    b.ring(square(60, -60, 10), {"building": "roof"})
    b.ring(square(60, 40, 10), {"building:part": "roof", "height": "6"})
    # An underground station: skipped and reported.
    b.ring(
        square(-80, -80, 40),
        {"building": "train_station", "location": "underground", "layer": "-2", "height": "3"},
    )
    # A tree, a road and a footway.
    b.node(0, -20, {"natural": "tree", "height": "9", "diameter_crown": "6", "genus": "Ficus"})
    b.way([b.node(-80, 0), b.node(80, 0)], {"highway": "primary", "lanes": "4", "width": "14"})
    b.way([b.node(-80, -5), b.node(80, -5)], {"highway": "footway"})
    data = osm.parse(b.xml())
    # Centred exactly on the builder's origin, so the site frame and the builder's frame agree.
    bbox = BBox(LON0 - 0.001, LAT0 - 0.001, LON0 + 0.001, LAT0 + 0.001)
    return site_from_osm("test", bbox, data)


def test_buildings_with_heights_holes_and_parts():
    site = synthetic_site()
    by_kind = {b.kind: b for b in site.buildings}
    assert set(by_kind) == {"yes", "apartments"}  # outline with a part and the roof are gone
    plain = next(b for b in site.buildings if b.kind == "yes" and not b.part)
    assert plain.height == 10 and plain.height_source == "osm:height"
    assert plain.footprint.area == pytest.approx(400, rel=1e-3)
    court = by_kind["apartments"]
    assert (
        court.height == pytest.approx(6 * STOREY_HEIGHT_M) and court.height_source == "osm:levels"
    )
    assert court.footprint.area == pytest.approx(900 - 100, rel=1e-3)
    assert len(court.footprint.interiors) == 1
    part = next(b for b in site.buildings if b.part)
    assert (part.height, part.base) == (25, 5)
    reasons = sorted(s["reason"] for s in site.skipped)
    assert reasons == [
        "open roof structure (not modelled yet)",
        "open roof structure (not modelled yet)",
        "underground (the model is above the ground)",
    ]


def test_trees_and_roads():
    site = synthetic_site()
    (tree,) = site.trees
    assert (tree.height, tree.crown_diameter, tree.genus) == (9, 6, "Ficus")
    # Coordinates are written with 9 decimals of a degree: about 0.1 mm.
    assert tree.x == pytest.approx(0, abs=1e-3) and tree.y == pytest.approx(-20, abs=1e-3)
    (road,) = site.roads
    assert (road.highway, road.lanes, road.width) == ("primary", 4, 14)
    assert road.line.length == pytest.approx(160, rel=1e-6)
    assert [f.highway for f in site.footways] == ["footway"]


def test_paths_inside_buildings_are_left_out():
    b = OsmBuilder()
    line = lambda y: [b.node(-40, y), b.node(40, y)]  # noqa: E731
    b.way(line(0), {"highway": "corridor"})
    b.way(line(5), {"highway": "footway", "indoor": "yes"})
    b.way(line(10), {"highway": "steps", "indoor": "corridor"})
    kept = b.way(line(15), {"highway": "footway", "indoor": "no"})
    b.way(line(20), {"barrier": "fence"})  # not a path at all: ignored, not listed
    bbox = BBox(LON0 - 0.001, LAT0 - 0.001, LON0 + 0.001, LAT0 + 0.001)
    site = site_from_osm("test", bbox, osm.parse(b.xml()))
    assert [f.id for f in site.footways] == [f"osm:way/{kept}"]
    assert [s["reason"] for s in site.skipped] == [INDOOR_REASON] * 3


def test_site_round_trips_through_json(tmp_path):
    site = synthetic_site()
    path = tmp_path / "site.json"
    site.save(path)
    again = Site.load(path)
    assert [b.id for b in again.buildings] == [b.id for b in site.buildings]
    assert [b.footprint.area for b in again.buildings] == pytest.approx(
        [b.footprint.area for b in site.buildings]
    )
    assert again.trees == site.trees
    assert again.frame.to_dict() == site.frame.to_dict()


def test_real_sample_near_hku():
    data = osm.parse(FIXTURE)
    bbox = data.bounds
    site = site_from_osm("hku_small", bbox, data)
    assert len(site.buildings) >= 20
    heights = [b.height for b in site.buildings if b.height is not None]
    assert heights and all(h > 0 for h in heights)
    # Most heights here come from storey counts; each says so.
    assert {b.height_source for b in site.buildings} <= {
        "osm:height",
        "osm:building:height",
        "osm:levels",
        "missing",
    }
    assert len(site.roads) + len(site.footways) > 0
    frame = LocalFrame.for_bbox(bbox)
    xmin, ymin, xmax, ymax = frame.bbox_extent(bbox)
    assert math.isclose(xmax - xmin, 206, rel_tol=0.02) and math.isclose(
        ymax - ymin, 177, rel_tol=0.02
    )
