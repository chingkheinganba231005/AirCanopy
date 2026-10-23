"""Build a Site from OpenStreetMap data.

- Buildings: closed ways and multipolygon relations tagged `building` or `building:part`. The
  "Simple 3D Buildings" scheme says: "When a building has any building:part areas, the building
  outline is not considered for 3D rendering" (https://wiki.openstreetmap.org/wiki/Simple_3D_Buildings,
  accessed 2026-10-23). So when an outline has parts inside it, the parts are used and the
  outline is left out, and the same volume is not counted twice.
- Underground buildings are skipped: `location=underground` is "Used where a feature is mainly
  under ground" (https://wiki.openstreetmap.org/wiki/Key:location, revision 3100870, accessed
  2026-10-23). The model is of the air above the ground; drawn on the surface, an underground
  station becomes a wall the wind would have to go round.
- Open roofs are skipped for now: `building=roof`, and `building:part=roof`, which is "A
  permanent roof structure without vertical walls that forms part of a building"
  (https://wiki.openstreetmap.org/wiki/Tag:building:part%3Droof, revision 3058073, accessed
  2026-10-23). As solid blocks they would put walls where air flows under the roof.
- Trees: nodes tagged `natural=tree`, with height, crown diameter, trunk circumference, species
  and leaf type when mapped.
- Roads: ways whose `highway` value is a road for vehicles; footways and other paths are kept
  separately (they are where people walk, which matters for comfort and exposure).

Features that are mapped but not modelled yet are listed in `Site.skipped` with the reason.
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Polygon

from ..geo import BBox, LocalFrame
from ..sources.osm import OsmData, assemble_rings
from .heights import building_height, parse_count, parse_length
from .model import Building, Road, Site, Tree

#: Highway values that carry motor traffic (https://wiki.openstreetmap.org/wiki/Key:highway).
ROAD_CLASSES = frozenset(
    {
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "unclassified",
        "residential",
        "service",
        "living_street",
        "road",
        "busway",
        "motorway_link",
        "trunk_link",
        "primary_link",
        "secondary_link",
        "tertiary_link",
    }
)
#: Highway values where people walk or cycle.
FOOT_CLASSES = frozenset({"footway", "pedestrian", "path", "steps", "cycleway", "corridor"})


def _is_yes(value: str | None) -> bool:
    return value is not None and value.lower() not in ("no", "false", "0")


class _Projector:
    def __init__(self, data: OsmData, frame: LocalFrame) -> None:
        ids = np.fromiter(data.nodes.keys(), dtype=np.int64)
        lonlat = np.array(list(data.nodes.values()), dtype=float).reshape(-1, 2)
        x, y = frame.to_local(lonlat[:, 0], lonlat[:, 1]) if len(ids) else (np.array([]),) * 2
        self.xy = {int(i): (float(a), float(b)) for i, a, b in zip(ids, x, y, strict=True)}

    def coords(self, refs: list[int]) -> list[tuple[float, float]] | None:
        """Local coordinates of the nodes, or None if any node is missing from the data."""
        out = []
        for r in refs:
            p = self.xy.get(r)
            if p is None:
                return None
            out.append(p)
        return out


def _polygons(exteriors: list[list], holes: list[list]) -> list[Polygon]:
    """Valid polygons from rings; each hole goes to the outer ring that contains it."""
    outs = [Polygon(e) for e in exteriors if len(e) >= 4]
    hole_polys = [Polygon(h) for h in holes if len(h) >= 4]
    result = []
    for o in outs:
        inner = [h.exterior.coords for h in hole_polys if o.contains(h.representative_point())]
        poly = shapely.make_valid(Polygon(o.exterior.coords, inner))
        result.extend(_only_polygons(poly))
    return result


def _only_polygons(geom) -> list[Polygon]:
    if isinstance(geom, Polygon):
        return [geom] if not geom.is_empty and geom.area > 0 else []
    if isinstance(geom, MultiPolygon):
        return [g for g in geom.geoms if g.area > 0]
    if hasattr(geom, "geoms"):
        return [g for part in geom.geoms for g in _only_polygons(part)]
    return []


def _building_features(data: OsmData, proj: _Projector, skipped: list[dict]):
    """(id, polygon, tags, is_part) for every building outline and part with complete geometry."""
    feats = []
    for wid, (refs, tags) in data.ways.items():
        is_building = _is_yes(tags.get("building"))
        is_part = _is_yes(tags.get("building:part"))
        if not (is_building or is_part):
            continue
        if len(refs) < 4 or refs[0] != refs[-1]:
            continue  # not closed: a multipolygon member, handled with its relation
        coords = proj.coords(refs)
        if coords is None:
            skipped.append({"id": f"osm:way/{wid}", "reason": "nodes missing from the download"})
            continue
        for poly in _polygons([coords], []):
            feats.append((f"osm:way/{wid}", poly, tags, is_part and not is_building))
    for rid, (members, tags) in data.relations.items():
        if tags.get("type") not in ("multipolygon", "building"):
            continue
        is_building = _is_yes(tags.get("building"))
        is_part = _is_yes(tags.get("building:part"))
        if not (is_building or is_part):
            continue
        outer = [
            data.ways[ref][0]
            for t, ref, role in members
            if t == "way" and role in ("outer", "") and ref in data.ways
        ]
        inner = [
            data.ways[ref][0]
            for t, ref, role in members
            if t == "way" and role == "inner" and ref in data.ways
        ]
        outer_rings, open_outer = assemble_rings(outer)
        inner_rings, _ = assemble_rings(inner)
        if open_outer:
            skipped.append(
                {"id": f"osm:relation/{rid}", "reason": "outline not closed in the download"}
            )
        ext = [c for r in outer_rings if (c := proj.coords(r)) is not None]
        hol = [c for r in inner_rings if (c := proj.coords(r)) is not None]
        for k, poly in enumerate(_polygons(ext, hol)):
            feats.append(
                (
                    f"osm:relation/{rid}" + (f"#{k}" if k else ""),
                    poly,
                    tags,
                    is_part and not is_building,
                )
            )
    return feats


def buildings_from_osm(data: OsmData, frame: LocalFrame, skipped: list[dict]) -> list[Building]:
    proj = _Projector(data, frame)
    feats = _building_features(data, proj, skipped)
    parts = [f for f in feats if f[3]]
    tree = shapely.STRtree([f[1].representative_point() for f in parts]) if parts else None
    out = []
    for fid, poly, tags, is_part in feats:
        if tags.get("location") == "underground":
            skipped.append({"id": fid, "reason": "underground (the model is above the ground)"})
            continue
        if tags.get("building:part" if is_part else "building") == "roof":
            skipped.append({"id": fid, "reason": "open roof structure (not modelled yet)"})
            continue
        if not is_part and tree is not None and len(tree.query(poly, predicate="contains")):
            # Outline with detailed parts inside: the parts describe the building.
            continue
        top, bottom, top_src, bottom_src = building_height(tags)
        out.append(
            Building(
                id=fid,
                footprint=poly,
                height=top,
                base=bottom,
                height_source=top_src,
                base_source=bottom_src,
                kind=tags.get("building") or tags.get("building:part") or "yes",
                part=is_part,
            )
        )
    return out


def trees_from_osm(data: OsmData, frame: LocalFrame) -> list[Tree]:
    ids = [nid for nid, tags in data.node_tags.items() if tags.get("natural") == "tree"]
    if not ids:
        return []
    lonlat = np.array([data.nodes[i] for i in ids])
    x, y = frame.to_local(lonlat[:, 0], lonlat[:, 1])
    trees = []
    for nid, xi, yi in zip(ids, x, y, strict=True):
        tags = data.node_tags[nid]
        height = parse_length(tags.get("height"))
        crown = parse_length(tags.get("diameter_crown"))
        trees.append(
            Tree(
                id=f"osm:node/{nid}",
                x=float(xi),
                y=float(yi),
                height=height,
                crown_diameter=crown,
                trunk_circumference=parse_length(tags.get("circumference")),
                species=tags.get("species") or tags.get("species:en"),
                genus=tags.get("genus"),
                leaf_type=tags.get("leaf_type"),
                leaf_cycle=tags.get("leaf_cycle"),
                height_source="osm:height" if height is not None else "missing",
                crown_source="osm:diameter_crown" if crown is not None else "missing",
            )
        )
    return trees


def roads_from_osm(
    data: OsmData, frame: LocalFrame, skipped: list[dict]
) -> tuple[list[Road], list[Road]]:
    proj = _Projector(data, frame)
    roads, foot = [], []
    for wid, (refs, tags) in data.ways.items():
        cls = tags.get("highway")
        if cls not in ROAD_CLASSES and cls not in FOOT_CLASSES:
            continue
        coords = proj.coords(refs)
        if coords is None or len(coords) < 2:
            skipped.append({"id": f"osm:way/{wid}", "reason": "nodes missing from the download"})
            continue
        lanes = parse_count(tags.get("lanes"))
        width = parse_length(tags.get("width"))
        road = Road(
            id=f"osm:way/{wid}",
            line=LineString(coords),
            highway=cls,
            lanes=int(lanes) if lanes else None,
            width=width,
            oneway=tags.get("oneway") in ("yes", "true", "1", "-1"),
            tunnel=_is_yes(tags.get("tunnel")),
            bridge=_is_yes(tags.get("bridge")),
            lanes_source="osm:lanes" if lanes else "missing",
            width_source="osm:width" if width is not None else "missing",
        )
        (roads if cls in ROAD_CLASSES else foot).append(road)
    return roads, foot


def site_from_osm(name: str, bbox: BBox, data: OsmData, sources: list[dict] | None = None) -> Site:
    """Everything in the box, in the box's own local frame.

    Features that cross the edge of the box are kept whole; the voxel grid crops them later.
    """
    frame = LocalFrame.for_bbox(bbox)
    xmin, ymin, xmax, ymax = frame.bbox_extent(bbox)
    area = shapely.box(xmin, ymin, xmax, ymax)
    skipped: list[dict] = []
    buildings = [
        b for b in buildings_from_osm(data, frame, skipped) if b.footprint.intersects(area)
    ]
    trees = [t for t in trees_from_osm(data, frame) if area.contains(shapely.Point(t.x, t.y))]
    roads, foot = roads_from_osm(data, frame, skipped)
    return Site(
        name=name,
        bbox=bbox,
        frame=frame,
        buildings=buildings,
        trees=trees,
        roads=[r for r in roads if r.line.intersects(area)],
        footways=[r for r in foot if r.line.intersects(area)],
        skipped=skipped,
        sources=sources or [],
    )
