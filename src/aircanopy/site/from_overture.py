"""Join Overture Maps buildings to a site built from OpenStreetMap.

Overture's buildings include OpenStreetMap's own, so the two sources are joined, not stacked:

- A building whose first source is OpenStreetMap is already in the site (the same way or
  relation). Overture sometimes knows its height from another dataset, measured by lidar or
  estimated by machine learning. That height replaces the site's when it is of a more trusted
  kind (heights.HEIGHT_KINDS), and the dataset it came from is kept with it.
- A building from another dataset, such as a footprint traced from satellite images, is added,
  unless more than OVERLAP_SHARE of it overlaps a building already in the site: then it is the
  same building drawn twice, and the OpenStreetMap one is kept.
- Underground buildings are left out, as they are from OpenStreetMap.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa
import shapely
from shapely.geometry import MultiPolygon, Polygon

from ..geo import LocalFrame
from .heights import HEIGHT_KINDS, STOREY_HEIGHT_M, height_kind
from .model import Building, Site

#: A footprint overlapping a site building by more than this share of the smaller of the two
#: is taken to be the same building.
OVERLAP_SHARE = 0.5

_OSM_RECORD = re.compile(r"^([nwr])(\d+)(?:@\d+)?$")
_OSM_ID = re.compile(r"^osm:(node|way|relation)/(\d+)")


@dataclass
class JoinReport:
    release: str | None
    rows: int = 0
    added: int = 0
    heights_improved: int = 0
    duplicates: int = 0
    underground: int = 0
    added_by_dataset: Counter = field(default_factory=Counter)
    heights_by_dataset: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict:
        return {
            "release": self.release,
            "buildings_read": self.rows,
            "added": self.added,
            "heights_improved": self.heights_improved,
            "duplicates_left_out": self.duplicates,
            "underground_left_out": self.underground,
            "added_by_dataset": dict(self.added_by_dataset),
            "heights_by_dataset": dict(self.heights_by_dataset),
        }


def _dataset_for(sources: list[dict], prop: str) -> str | None:
    """The dataset that gave property `prop`, else the building's own (first) dataset."""
    for s in sources:
        if s.get("property") == prop:
            return s.get("dataset")
    return sources[0].get("dataset") if sources else None


def _rank(kind: str | None) -> int:
    return HEIGHT_KINDS.index(kind) if kind in HEIGHT_KINDS else len(HEIGHT_KINDS)


def _local_polygons(wkb: bytes, frame: LocalFrame) -> list[Polygon]:
    geom = shapely.from_wkb(wkb)

    def to_local(lonlat: np.ndarray) -> np.ndarray:
        x, y = frame.to_local(lonlat[:, 0], lonlat[:, 1])
        return np.column_stack([x, y])

    geom = shapely.make_valid(shapely.transform(geom, to_local))
    if isinstance(geom, Polygon):
        parts = [geom]
    elif isinstance(geom, MultiPolygon):
        parts = list(geom.geoms)
    else:
        parts = [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]
    return [p for p in parts if p.area > 0]


def _height(row: dict) -> tuple[float | None, str, str | None]:
    sources = row["sources"] or []
    if row["height"] is not None and row["height"] > 0:
        return float(row["height"]), "overture:height", _dataset_for(sources, "/properties/height")
    if row["num_floors"] is not None and row["num_floors"] > 0:
        dataset = _dataset_for(sources, "/properties/num_floors")
        return row["num_floors"] * STOREY_HEIGHT_M, "overture:num_floors", dataset
    return None, "missing", None


def _base(row: dict) -> tuple[float, str]:
    if row["min_height"] is not None and row["min_height"] > 0:
        return float(row["min_height"]), "overture:min_height"
    if row["min_floor"] is not None and row["min_floor"] > 0:
        return row["min_floor"] * STOREY_HEIGHT_M, "overture:min_floor"
    return 0.0, "ground"


def _is_duplicate(poly: Polygon, tree: shapely.STRtree, prints: list[Polygon]) -> bool:
    for i in tree.query(poly, predicate="intersects"):
        other = prints[i]
        shared = poly.intersection(other).area
        if shared > OVERLAP_SHARE * min(poly.area, other.area):
            return True
    return False


def join_overture(site: Site, table: pa.Table, release: str | None = None) -> JoinReport:
    """Add Overture's buildings and better heights to `site`, in place; returns what changed."""
    report = JoinReport(release=release, rows=table.num_rows)
    osm: dict[tuple[str, int], list[Building]] = {}
    for b in site.buildings:
        if m := _OSM_ID.match(b.id):
            osm.setdefault((m.group(1)[0], int(m.group(2))), []).append(b)
    xmin, ymin, xmax, ymax = site.extent()
    area = shapely.box(xmin, ymin, xmax, ymax)
    prints = [b.footprint for b in site.buildings]
    tree = shapely.STRtree(prints) if prints else None
    added: list[Building] = []

    for row in table.to_pylist():
        if row["is_underground"]:
            report.underground += 1
            continue
        sources = row["sources"] or []
        first = sources[0] if sources else {}
        top, top_source, top_dataset = _height(row)

        if first.get("dataset") == "OpenStreetMap":
            m = _OSM_RECORD.match(first.get("record_id") or "")
            # Overture's own copy of an OpenStreetMap height is the same tag we read already;
            # only a height from another dataset can add anything.
            if not m or top_source != "overture:height" or top_dataset == "OpenStreetMap":
                continue
            kind = height_kind(top_source, top_dataset)
            for b in osm.get((m.group(1), int(m.group(2))), []):
                if (
                    top is not None
                    and top > b.base
                    and _rank(kind) < _rank(height_kind(b.height_source, b.height_dataset))
                ):
                    b.height, b.height_source, b.height_dataset = top, top_source, top_dataset
                    report.heights_improved += 1
                    report.heights_by_dataset[top_dataset or "unknown"] += 1
            continue

        dataset = first.get("dataset") or "unknown"
        base, base_source = _base(row)
        if top is not None and top <= base:
            base, base_source = 0.0, "ground"
        for k, poly in enumerate(_local_polygons(row["geometry"], site.frame)):
            if not poly.intersects(area):
                continue
            if tree is not None and _is_duplicate(poly, tree, prints):
                report.duplicates += 1
                continue
            added.append(
                Building(
                    id=f"overture:{row['id']}" + (f"#{k}" if k else ""),
                    footprint=poly,
                    height=top,
                    base=base,
                    height_source=top_source,
                    base_source=base_source,
                    kind=row["class"] or row["subtype"] or "yes",
                    footprint_source=f"overture:{dataset}",
                    height_dataset=top_dataset if top is not None else None,
                )
            )
            report.added += 1
            report.added_by_dataset[dataset] += 1
            if top is not None:
                report.heights_by_dataset[top_dataset or "unknown"] += 1

    site.buildings.extend(added)
    return report
