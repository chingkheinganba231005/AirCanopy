"""The site model: what a real place contains, in local metres.

Positions are in the site's LocalFrame (x east, y north, metres). Heights are above the local
ground. Every value that came from data says which tag or dataset it came from, and every value
that is missing stays missing (None) instead of being filled with a guess; later steps decide
how to fill gaps and label what they filled.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path

from shapely.geometry import LineString, Polygon, mapping, shape

from ..geo import BBox, LocalFrame

SITE_SCHEMA = 1


@dataclass
class Building:
    id: str  # for example "osm:way/123"
    footprint: Polygon  # local metres; may have holes (courtyards)
    height: float | None  # top above the ground, m; None if the data do not say
    base: float  # bottom above the ground, m (0 for most buildings)
    height_source: str  # "osm:height", "osm:building:height", "osm:levels" or "missing"
    base_source: str
    kind: str  # the building tag value, for example "yes", "residential", "school"
    part: bool = False  # True for a building:part (one piece of a more detailed building)


@dataclass
class Tree:
    id: str
    x: float
    y: float
    height: float | None  # m
    crown_diameter: float | None  # m
    trunk_circumference: float | None  # m, measured at breast height when mapped
    species: str | None
    genus: str | None
    leaf_type: str | None  # "broadleaved" or "needleleaved"
    leaf_cycle: str | None  # "evergreen" or "deciduous"
    height_source: str
    crown_source: str


@dataclass
class Road:
    id: str
    line: LineString  # local metres
    highway: str  # OpenStreetMap highway class, for example "primary", "residential"
    lanes: int | None
    width: float | None  # m
    oneway: bool
    tunnel: bool
    bridge: bool
    lanes_source: str
    width_source: str


@dataclass
class Site:
    name: str
    bbox: BBox
    frame: LocalFrame
    buildings: list[Building] = field(default_factory=list)
    trees: list[Tree] = field(default_factory=list)
    roads: list[Road] = field(default_factory=list)
    footways: list[Road] = field(default_factory=list)
    #: Features read from the data but not modelled yet, with the reason (for example open roofs).
    skipped: list[dict] = field(default_factory=list)
    #: Downloads the site was built from (see sources.cache.Download.record).
    sources: list[dict] = field(default_factory=list)
    created: str = field(
        default_factory=lambda: dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    def extent(self) -> tuple[float, float, float, float]:
        """xmin, ymin, xmax, ymax of the site's box in local metres."""
        return self.frame.bbox_extent(self.bbox)

    def max_building_height(self) -> float:
        heights = [b.height for b in self.buildings if b.height is not None]
        return max(heights, default=0.0)

    # --- saving and loading -----------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "schema": SITE_SCHEMA,
            "name": self.name,
            "bbox": list(self.bbox.as_tuple()),
            "frame": self.frame.to_dict(),
            "created": self.created,
            "sources": self.sources,
            "skipped": self.skipped,
            "buildings": [{**_plain(b), "footprint": mapping(b.footprint)} for b in self.buildings],
            "trees": [_plain(t) for t in self.trees],
            "roads": [{**_plain(r), "line": mapping(r.line)} for r in self.roads],
            "footways": [{**_plain(r), "line": mapping(r.line)} for r in self.footways],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Site:
        if d.get("schema") != SITE_SCHEMA:
            raise ValueError(f"unsupported site schema {d.get('schema')}")
        return cls(
            name=d["name"],
            bbox=BBox(*d["bbox"]),
            frame=LocalFrame.from_dict(d["frame"]),
            buildings=[
                Building(**{**b, "footprint": shape(b["footprint"])}) for b in d["buildings"]
            ],
            trees=[Tree(**t) for t in d["trees"]],
            roads=[Road(**{**r, "line": shape(r["line"])}) for r in d["roads"]],
            footways=[Road(**{**r, "line": shape(r["line"])}) for r in d["footways"]],
            skipped=d["skipped"],
            sources=d["sources"],
            created=d["created"],
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), separators=(",", ":")) + "\n")

    @classmethod
    def load(cls, path: Path) -> Site:
        return cls.from_dict(json.loads(path.read_text()))


def _plain(obj) -> dict:
    """The dataclass fields that are plain JSON values (geometry is added by the caller)."""
    return {k: v for k, v in obj.__dict__.items() if not isinstance(v, Polygon | LineString)}
