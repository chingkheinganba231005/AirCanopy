"""From a box on the map to a site folder: data, model, meshes and voxels.

A site folder holds:

- `site.json`: the site model (buildings, trees, roads in local metres, with sources);
- `scene.glb`: meshes for drawing;
- `voxels.npz`: the voxel grid for the physics;
- `summary.json`: counts, how many heights of each kind (measured, mapped, from storeys,
  machine-learning estimate, from neighbours), and the data sources with their licences and the
  credit each dataset asks for.

- `terrain.npz`: the ground (site/terrain.py), over the site and its surroundings;
- `canopy.npz`: the height of the vegetation on a 1 m grid (site/canopy.py).

Buildings come from OpenStreetMap, joined with Overture Maps when the data are downloaded
(site/from_overture.py). The ground comes from an elevation model (sources/terrain.py), and the
trees from the canopy height map (sources/canopy.py) joined with the trees in OpenStreetMap. If
a second source cannot be reached, the site is built without it and the summary says so.
"""

from __future__ import annotations

import json
import urllib.error
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import rasterio.errors

from ..geo import BBox, LocalFrame
from ..sources import canopy as canopy_map
from ..sources import osm, overture
from ..sources import terrain as elevation
from ..sources.cache import sha256_file
from .canopy import SOURCE as CANOPY_SOURCE
from .canopy import TREE_CANOPY_MIN_M, Canopy, remove_buildings, trees_from_canopy
from .canopy import from_geotiff as canopy_from_geotiff
from .fill import MIN_NEIGHBOURS, NEIGHBOUR_MEDIAN, NEIGHBOUR_RADIUS_M, fill_building_heights
from .from_osm import site_from_osm
from .from_overture import join_overture
from .heights import HEIGHT_KINDS, STOREY_HEIGHT_M, height_kind
from .mesh import export_glb
from .model import Site
from .terrain import Terrain, flat, from_geotiff
from .voxel import BUILDING, GROUND, grid_for_site, save_voxels, voxelise

#: Spacing of the ground grid. The elevation models have about 30 m pixels; a finer grid only
#: interpolates between them.
TERRAIN_DX_M = 20.0
#: How far around the site the ground is drawn, so the view shows the real surroundings.
CONTEXT_M = 1000.0

_NETWORK_ERRORS = (
    OSError,
    ValueError,
    KeyError,
    urllib.error.URLError,
    pa.ArrowException,
    rasterio.errors.RasterioError,
)


def credits(site: Site) -> list[dict]:
    """Every dataset the site's buildings came from, with its credit, licence and count."""
    counts: Counter = Counter()
    for b in site.buildings:
        footprint = (
            "OpenStreetMap" if b.footprint_source == "osm" else b.footprint_source.split(":", 1)[1]
        )
        counts[footprint] += 1
        if b.height_dataset and b.height_dataset != footprint:
            counts[b.height_dataset] += 1
    out = []
    for dataset, n in counts.most_common():
        text, licence = overture.credit(dataset)
        out.append({"dataset": dataset, "credit": text, "licence": licence, "buildings": n})
    if (site.roads or site.footways or site.trees) and "OpenStreetMap" not in counts:
        # Roads and trees come from OpenStreetMap even when no building does.
        text, licence = overture.credit("OpenStreetMap")
        out.append({"dataset": "OpenStreetMap", "credit": text, "licence": licence, "buildings": 0})
    return out


def summarise(
    site: Site,
    dx: float,
    solid_cells: int,
    grid_shape: tuple[int, int, int],
    overture_status: dict | None = None,
    ground_cells: int = 0,
) -> dict:
    sources = Counter(b.height_source for b in site.buildings)
    kinds = Counter(height_kind(b.height_source, b.height_dataset) for b in site.buildings)
    return {
        "name": site.name,
        "bbox": list(site.bbox.as_tuple()),
        "size_m": [round(v, 1) for v in site.bbox.size_m()],
        "buildings": len(site.buildings),
        "building_heights": dict(sorted(sources.items())),
        "height_kinds": {k: kinds.get(k, 0) for k in HEIGHT_KINDS} | {"missing": kinds[None]},
        "footprint_sources": dict(Counter(b.footprint_source for b in site.buildings)),
        "buildings_without_height": sources.get("missing", 0),
        "estimated_from_neighbours": sources.get(NEIGHBOUR_MEDIAN, 0),
        "tallest_building_m": round(site.max_building_height(), 1),
        "trees": len(site.trees),
        "trees_with_height": sum(t.height is not None for t in site.trees),
        "roads": len(site.roads),
        "footways": len(site.footways),
        "skipped": len(site.skipped),
        "voxels": {
            "dx_m": dx,
            "shape_zyx": list(grid_shape),
            "building_cells": solid_cells,
            "ground_cells": ground_cells,
        },
        "terrain": site.terrain,
        # The values behind estimated heights (docs/assumptions.md), so a viewer can show them.
        "assumptions": {
            "A-001": {"storey_height_m": STOREY_HEIGHT_M},
            "A-002": {"radius_m": NEIGHBOUR_RADIUS_M, "min_neighbours": MIN_NEIGHBOURS},
        },
        "overture": overture_status,
        "credits": credits(site),
        "sources": site.sources,
    }


def _spread(values: np.ndarray) -> dict | None:
    """Smallest, median and largest of a set of values, rounded to 0.1."""
    if not len(values):
        return None
    lo, mid, hi = np.min(values), np.median(values), np.max(values)
    return {"min": round(float(lo), 1), "median": round(float(mid), 1), "max": round(float(hi), 1)}


def build_from_osm_data(
    name: str,
    bbox: BBox,
    data: osm.OsmData,
    out: Path,
    dx: float,
    sources: list[dict],
    overture_table: pa.Table | None = None,
    overture_status: dict | None = None,
    terrain: Terrain | None = None,
    terrain_note: str | None = None,
    canopy: Canopy | None = None,
    canopy_note: str | None = None,
) -> dict:
    site = site_from_osm(name, bbox, data, sources)
    if overture_table is not None:
        report = join_overture(site, overture_table, (overture_status or {}).get("release"))
        overture_status = {**(overture_status or {}), **report.to_dict()}
    fill_building_heights(site)
    canopy_status: dict = {"note": canopy_note} if canopy_note else {}
    if canopy is not None:
        removed = remove_buildings(canopy, site.buildings)
        site.trees, counts = trees_from_canopy(canopy, site.trees)
        crowns = [t for t in site.trees if t.crown_source == CANOPY_SOURCE]
        heights = np.array([t.height for t in crowns], dtype=float)
        widths = np.array([t.crown_diameter for t in crowns], dtype=float)
        canopy_status = {
            "file": "canopy.npz",
            "model": "meta-wri-chm-v1",
            "imagery_dates": canopy.imagery_dates,
            "grid_m": canopy.dx,
            "cover_share": round(canopy.cover(TREE_CANOPY_MIN_M), 4),
            "cover_min_height_m": TREE_CANOPY_MIN_M,
            "cells_removed_inside_buildings": removed,
            **counts,
            "crown_height_m": _spread(heights),
            "crown_diameter_m": _spread(widths),
            **canopy_status,
        }
    terrain = terrain or flat(site.extent(), TERRAIN_DX_M)
    for b in site.buildings:
        b.ground = terrain.lowest_under(b.footprint)
    for t in site.trees:
        t.ground = float(terrain.at(t.x, t.y))
    site.terrain = {"file": "terrain.npz", **terrain.summary(site.extent())}
    if terrain_note:
        site.terrain["note"] = terrain_note
    out.mkdir(parents=True, exist_ok=True)
    site.save(out / "site.json")
    terrain.save(out / "terrain.npz")
    if canopy is not None:
        canopy.save(out / "canopy.npz")
    export_glb(site, out / "scene.glb", terrain)
    grid = grid_for_site(site, dx, terrain=terrain)
    solid = voxelise(site, grid, terrain)
    save_voxels(out / "voxels.npz", grid, solid)
    summary = summarise(
        site,
        dx,
        int((solid == BUILDING).sum()),
        grid.shape,
        overture_status,
        int((solid == GROUND).sum()),
    )
    summary["canopy"] = canopy_status or None
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def surroundings(bbox: BBox, margin_m: float) -> tuple[BBox, tuple[float, ...]]:
    """The site's box `margin_m` wider on every side, as a local extent (xmin, ymin, xmax, ymax)
    of the site's frame, and the box in degrees to download for it. The download reaches one
    ground-grid step and one 30 m pixel further, because the grid's last points may overshoot
    the extent by up to a step and interpolation needs the pixel beyond."""
    frame = LocalFrame.for_bbox(bbox)
    xmin, ymin, xmax, ymax = frame.bbox_extent(bbox)
    extent = (xmin - margin_m, ymin - margin_m, xmax + margin_m, ymax + margin_m)
    spare = TERRAIN_DX_M + 30.0
    xs = [extent[0] - spare, extent[2] + spare]
    ys = [extent[1] - spare, extent[3] + spare]
    lon, lat = frame.to_lonlat([xs[0], xs[1], xs[0], xs[1]], [ys[0], ys[0], ys[1], ys[1]])
    return BBox(min(lon), min(lat), max(lon), max(lat)), extent


def load_terrain(
    bbox: BBox,
    model: str = "gedtm30",
    terrain_file: Path | None = None,
    refresh: bool = False,
    margin_m: float = CONTEXT_M,
) -> tuple[Terrain | None, dict | None, str | None]:
    """(terrain, source record, note) for a site; terrain None means level ground."""
    if model == "flat" and terrain_file is None:
        return None, None, "level ground: no elevation model was used"
    wide, extent = surroundings(bbox, margin_m)
    frame = LocalFrame.for_bbox(bbox)
    if terrain_file is not None:
        t = from_geotiff(terrain_file, frame, extent, TERRAIN_DX_M)
        known = elevation.MODELS.get(t.model)  # a copy of one of our own downloads
        record = {
            "path": str(terrain_file),
            "bytes": terrain_file.stat().st_size,
            "sha256": sha256_file(terrain_file),
            "licence": known.licence if known else "not stated",
            "attribution": known.attribution if known else "elevation file supplied by the user",
            "note": "local elevation file supplied by the user",
        }
        return t, record, None
    try:
        got = elevation.fetch_terrain(wide, model, refresh=refresh)
    except _NETWORK_ERRORS as e:
        return None, None, f"level ground: the elevation model could not be read ({e})"
    m = elevation.MODELS[model]
    t = from_geotiff(got.path, frame, extent, TERRAIN_DX_M, m.key, m.kind)
    note = None
    if m.kind == "surface":
        note = "surface model: the ground includes roofs and treetops in places"
    return t, got.record(), note


def load_canopy(
    bbox: BBox,
    frame: LocalFrame,
    extent: tuple[float, float, float, float],
    use_canopy: bool = True,
    canopy_file: Path | None = None,
    refresh: bool = False,
) -> tuple[Canopy | None, dict | None, str | None]:
    """(canopy, source record, note); canopy None means trees from OpenStreetMap only."""
    if canopy_file is not None:
        c = canopy_from_geotiff(canopy_file, frame, extent)
        record = {
            "path": str(canopy_file),
            "bytes": canopy_file.stat().st_size,
            "sha256": sha256_file(canopy_file),
            "licence": canopy_map.LICENCE,
            "attribution": canopy_map.ATTRIBUTION,
            "note": "local copy of the canopy height map supplied by the user",
        }
        return c, record, None
    if not use_canopy:
        return None, None, "trees from OpenStreetMap only: the canopy height map was not used"
    try:
        got = canopy_map.fetch_canopy(bbox, refresh=refresh)
    except _NETWORK_ERRORS as e:
        return None, None, f"trees from OpenStreetMap only: the canopy map could not be read ({e})"
    return canopy_from_geotiff(got.path, frame, extent), got.record(), None


def build_site(
    name: str,
    bbox: BBox | None,
    out: Path,
    dx: float = 2.0,
    refresh: bool = False,
    osm_file: Path | None = None,
    use_overture: bool = True,
    terrain_model: str | None = None,
    terrain_file: Path | None = None,
    use_canopy: bool | None = None,
    canopy_file: Path | None = None,
    extra_sources: list[dict] | None = None,
) -> dict:
    """Build the site folder.

    The OpenStreetMap data are downloaded for `bbox` (or reused from the cache) and joined with
    Overture Maps buildings, or read from `osm_file`, an OpenStreetMap XML file (`.osm` or
    `.osm.gz`) such as an export from openstreetmap.org, with no download at all. With a file
    and no box, the box is the file's own bounds.

    The ground comes from `terrain_model` ("gedtm30", "copernicus" or "flat"; by default
    GEDTM30 when downloading and level ground with an OpenStreetMap file), or from
    `terrain_file`, a GeoTIFF in longitude and latitude with elevation in band 1.

    Trees come from the canopy height map when downloading (unless `use_canopy` is False) or
    from `canopy_file`, joined with the trees in OpenStreetMap.

    `extra_sources` are added to the site's list of sources (for example, how a place name was
    turned into coordinates).
    """
    if osm_file is None:
        if bbox is None:
            raise ValueError("give a box, or an OpenStreetMap file")
        download = osm.fetch_map(bbox, refresh=refresh)
        data = osm.parse(download.path)
        records = [*(extra_sources or []), download.record()]
        table, status = None, None
        if use_overture:
            try:
                got, release = overture.fetch_buildings(bbox, refresh=refresh)
                table = overture.load(got.path)
                records.append({**got.record(), "release": release})
                status = {"release": release}
            except _NETWORK_ERRORS as e:
                # A second source being unreachable should not stop the build; say so instead.
                status = {"error": f"Overture Maps not used: {type(e).__name__}: {e}"}
        terrain, record, note = load_terrain(
            bbox, terrain_model or "gedtm30", terrain_file, refresh
        )
        if record:
            records.append(record)
        frame = LocalFrame.for_bbox(bbox)
        canopy, c_record, c_note = load_canopy(
            bbox, frame, frame.bbox_extent(bbox), use_canopy is not False, canopy_file, refresh
        )
        if c_record:
            records.append(c_record)
        return build_from_osm_data(
            name, bbox, data, out, dx, records, table, status, terrain, note, canopy, c_note
        )
    data = osm.parse(osm_file)
    bbox = bbox or data.bounds
    if bbox is None:
        raise ValueError(f"{osm_file} has no <bounds> element; give the box with --bbox")
    records = [
        *(extra_sources or []),
        {
            "path": str(osm_file),
            "bytes": osm_file.stat().st_size,
            "sha256": sha256_file(osm_file),
            "licence": osm.LICENCE,
            "attribution": osm.ATTRIBUTION,
            "note": "local OpenStreetMap file supplied by the user",
        },
    ]
    terrain, record, note = load_terrain(bbox, terrain_model or "flat", terrain_file, refresh)
    if record:
        records.append(record)
    frame = LocalFrame.for_bbox(bbox)
    canopy, c_record, c_note = load_canopy(
        bbox, frame, frame.bbox_extent(bbox), bool(use_canopy), canopy_file, refresh
    )
    if c_record:
        records.append(c_record)
    return build_from_osm_data(
        name, bbox, data, out, dx, records, None, None, terrain, note, canopy, c_note
    )
