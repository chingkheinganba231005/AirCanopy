"""Small sites written directly in local metres, for tests."""

from __future__ import annotations

from shapely.geometry import LineString, Polygon

from aircanopy.geo import BBox, LocalFrame
from aircanopy.site.model import Building, Road, Site, Tree

LON0, LAT0 = 114.0, 22.0


def square(x0: float, y0: float, size: float) -> Polygon:
    return Polygon([(x0, y0), (x0 + size, y0), (x0 + size, y0 + size), (x0, y0 + size)])


def building(fid: str, poly: Polygon, height: float | None, base: float = 0.0) -> Building:
    return Building(
        id=fid,
        footprint=poly,
        height=height,
        base=base,
        height_source="osm:height" if height is not None else "missing",
        base_source="ground" if base == 0 else "osm:min_height",
        kind="yes",
    )


def empty_site(size_m: float = 200.0) -> Site:
    d = 0.5 * size_m / 111_000.0
    bbox = BBox(LON0 - d, LAT0 - d, LON0 + d, LAT0 + d)
    return Site(name="test", bbox=bbox, frame=LocalFrame(LON0, LAT0))


def tree(x: float, y: float, height: float | None, crown: float | None = None) -> Tree:
    return Tree(
        id=f"t{x},{y}",
        x=x,
        y=y,
        height=height,
        crown_diameter=crown,
        trunk_circumference=None,
        species=None,
        genus=None,
        leaf_type=None,
        leaf_cycle=None,
        height_source="osm:height" if height is not None else "missing",
        crown_source="osm:diameter_crown" if crown is not None else "missing",
    )


def road(coords, highway: str = "residential", width: float | None = None) -> Road:
    return Road(
        id=f"r{coords[0]}",
        line=LineString(coords),
        highway=highway,
        lanes=None,
        width=width,
        oneway=False,
        tunnel=False,
        bridge=False,
        lanes_source="missing",
        width_source="osm:width" if width is not None else "missing",
    )
