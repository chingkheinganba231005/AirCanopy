"""OpenStreetMap data for a small area, from the main OpenStreetMap API.

The main API's `map` call returns every node, way and relation in a bounding box, with all their
tags, as XML. It is meant for editors: the box may cover at most 0.25 square degrees (the API's
capabilities, https://api.openstreetmap.org/api/0.6/capabilities, `area maximum`, accessed
2026-10-23). Larger areas need Overpass or another source. We ask once per area and cache the
answer.

The data are © OpenStreetMap contributors, available under the Open Database Licence 1.0.
"""

from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ..geo import BBox
from .cache import Download, fetch

API_MAP = "https://api.openstreetmap.org/api/0.6/map"
LICENCE = "ODbL-1.0"
ATTRIBUTION = "© OpenStreetMap contributors"
#: The map call's limit on the box, in square degrees (the API's capabilities, see above).
MAX_AREA_DEG2 = 0.25

Tags = dict[str, str]


@dataclass
class OsmData:
    """Nodes, ways and relations as plain Python containers."""

    nodes: dict[int, tuple[float, float]] = field(default_factory=dict)  # id -> (lon, lat)
    node_tags: dict[int, Tags] = field(default_factory=dict)
    ways: dict[int, tuple[list[int], Tags]] = field(default_factory=dict)
    relations: dict[int, tuple[list[tuple[str, int, str]], Tags]] = field(default_factory=dict)
    bounds: BBox | None = None


def map_url(bbox: BBox) -> str:
    return f"{API_MAP}?bbox={bbox.west:.7f},{bbox.south:.7f},{bbox.east:.7f},{bbox.north:.7f}"


def fetch_map(bbox: BBox, *, refresh: bool = False) -> Download:
    """Download (or reuse) the OpenStreetMap data of a box."""
    if bbox.area_deg2() > MAX_AREA_DEG2:
        raise ValueError(
            f"the OpenStreetMap API serves at most {MAX_AREA_DEG2} square degrees; "
            f"this box is {bbox.area_deg2():.3f}"
        )
    name = "osm/map_{:.6f}_{:.6f}_{:.6f}_{:.6f}.osm".format(*bbox.as_tuple())
    return fetch(map_url(bbox), name, licence=LICENCE, attribution=ATTRIBUTION, refresh=refresh)


def _tags(element: ET.Element) -> Tags:
    return {t.get("k", ""): t.get("v", "") for t in element.findall("tag")}


def parse(source: bytes | str | Path) -> OsmData:
    """Parse OpenStreetMap XML from bytes, a string or a file (gzipped if it ends in .gz)."""
    if isinstance(source, Path):
        raw = source.read_bytes()
        root = ET.fromstring(gzip.decompress(raw) if source.suffix == ".gz" else raw)
    else:
        root = ET.fromstring(source)
    data = OsmData()
    b = root.find("bounds")
    if b is not None:
        data.bounds = BBox(
            float(b.get("minlon")),
            float(b.get("minlat")),
            float(b.get("maxlon")),
            float(b.get("maxlat")),
        )
    for n in root.iter("node"):
        nid = int(n.get("id"))
        data.nodes[nid] = (float(n.get("lon")), float(n.get("lat")))
        tags = _tags(n)
        if tags:
            data.node_tags[nid] = tags
    for w in root.iter("way"):
        refs = [int(nd.get("ref")) for nd in w.findall("nd")]
        data.ways[int(w.get("id"))] = (refs, _tags(w))
    for r in root.iter("relation"):
        members = [
            (m.get("type"), int(m.get("ref")), m.get("role", "")) for m in r.findall("member")
        ]
        data.relations[int(r.get("id"))] = (members, _tags(r))
    return data


def assemble_rings(parts: Iterable[list[int]]) -> tuple[list[list[int]], list[list[int]]]:
    """Join ways end to end into closed rings of node ids.

    Multipolygon outlines are often split into several ways. Returns the closed rings and the
    pieces that could not be closed (incomplete data at the edge of the downloaded box).
    """
    pending = [list(p) for p in parts if len(p) >= 2]
    rings: list[list[int]] = []
    open_pieces: list[list[int]] = []
    while pending:
        ring = pending.pop(0)
        grown = True
        while ring[0] != ring[-1] and grown:
            grown = False
            for i, p in enumerate(pending):
                if p[0] == ring[-1]:
                    ring = ring + p[1:]
                elif p[-1] == ring[-1]:
                    ring = ring + p[-2::-1]
                elif p[-1] == ring[0]:
                    ring = p + ring[1:]
                elif p[0] == ring[0]:
                    ring = p[::-1] + ring[1:]
                else:
                    continue
                pending.pop(i)
                grown = True
                break
        if ring[0] == ring[-1] and len(ring) >= 4:
            rings.append(ring)
        else:
            open_pieces.append(ring)
    return rings, open_pieces
