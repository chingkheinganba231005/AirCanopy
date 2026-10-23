"""Reading OpenStreetMap XML and joining split outlines."""

from pathlib import Path

import pytest

from aircanopy.geo import BBox
from aircanopy.sources import osm

FIXTURE = Path(__file__).parent / "data" / "osm" / "hku_small.osm.gz"


def test_parse_small_document():
    xml = b"""<?xml version="1.0"?>
    <osm version="0.6">
      <bounds minlat="22.0" minlon="114.0" maxlat="22.1" maxlon="114.1"/>
      <node id="1" lat="22.01" lon="114.01"><tag k="natural" v="tree"/></node>
      <node id="2" lat="22.02" lon="114.02"/>
      <way id="10"><nd ref="1"/><nd ref="2"/><tag k="highway" v="residential"/></way>
      <relation id="20">
        <member type="way" ref="10" role="outer"/><tag k="type" v="multipolygon"/>
      </relation>
    </osm>"""
    d = osm.parse(xml)
    assert d.bounds == BBox(114.0, 22.0, 114.1, 22.1)
    assert d.nodes[1] == (114.01, 22.01)
    assert d.node_tags == {1: {"natural": "tree"}}
    assert d.ways[10] == ([1, 2], {"highway": "residential"})
    assert d.relations[20] == ([("way", 10, "outer")], {"type": "multipolygon"})


def test_rings_join_pieces_in_any_direction():
    rings, open_pieces = osm.assemble_rings([[1, 2, 3], [5, 4, 3], [5, 6, 1]])
    assert open_pieces == []
    assert len(rings) == 1
    ring = rings[0]
    assert ring[0] == ring[-1] and sorted(set(ring)) == [1, 2, 3, 4, 5, 6]


def test_rings_report_pieces_that_do_not_close():
    rings, open_pieces = osm.assemble_rings([[1, 2, 3, 1], [7, 8, 9]])
    assert rings == [[1, 2, 3, 1]]
    assert open_pieces == [[7, 8, 9]]


def test_map_url_and_area_limit():
    b = BBox(114.136, 22.2832, 114.138, 22.2848)
    assert osm.map_url(b).endswith("bbox=114.1360000,22.2832000,114.1380000,22.2848000")
    with pytest.raises(ValueError, match="square degrees"):
        osm.fetch_map(BBox(114.0, 22.0, 115.0, 23.0))


def test_parse_stored_sample():
    d = osm.parse(FIXTURE)
    assert d.bounds == BBox(114.136, 22.2832, 114.138, 22.2848)
    assert len(d.nodes) > 1000
    assert sum("building" in tags for _, tags in d.ways.values()) >= 30
