"""From a box on the map to a site folder: data, model, meshes and voxels.

A site folder holds:

- `site.json`: the site model (buildings, trees, roads in local metres, with sources);
- `scene.glb`: meshes for drawing;
- `voxels.npz`: the voxel grid for the physics;
- `summary.json`: counts, how many heights of each kind (measured, mapped, from storeys,
  machine-learning estimate, from neighbours), and the data sources with their licences and the
  credit each dataset asks for.

Buildings come from OpenStreetMap, joined with Overture Maps when the data are downloaded
(site/from_overture.py). If Overture cannot be reached, the site is built without it and the
summary says so.
"""

from __future__ import annotations

import json
import urllib.error
from collections import Counter
from pathlib import Path

import pyarrow as pa

from ..geo import BBox
from ..sources import osm, overture
from ..sources.cache import sha256_file
from .fill import MIN_NEIGHBOURS, NEIGHBOUR_MEDIAN, NEIGHBOUR_RADIUS_M, fill_building_heights
from .from_osm import site_from_osm
from .from_overture import join_overture
from .heights import HEIGHT_KINDS, STOREY_HEIGHT_M, height_kind
from .mesh import export_glb
from .model import Site
from .voxel import grid_for_site, save_voxels, voxelise_buildings


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
        "voxels": {"dx_m": dx, "shape_zyx": list(grid_shape), "building_cells": solid_cells},
        # The values behind estimated heights (docs/assumptions.md), so a viewer can show them.
        "assumptions": {
            "A-001": {"storey_height_m": STOREY_HEIGHT_M},
            "A-002": {"radius_m": NEIGHBOUR_RADIUS_M, "min_neighbours": MIN_NEIGHBOURS},
        },
        "overture": overture_status,
        "credits": credits(site),
        "sources": site.sources,
    }


def build_from_osm_data(
    name: str,
    bbox: BBox,
    data: osm.OsmData,
    out: Path,
    dx: float,
    sources: list[dict],
    overture_table: pa.Table | None = None,
    overture_status: dict | None = None,
) -> dict:
    site = site_from_osm(name, bbox, data, sources)
    if overture_table is not None:
        report = join_overture(site, overture_table, (overture_status or {}).get("release"))
        overture_status = {**(overture_status or {}), **report.to_dict()}
    fill_building_heights(site)
    out.mkdir(parents=True, exist_ok=True)
    site.save(out / "site.json")
    export_glb(site, out / "scene.glb")
    grid = grid_for_site(site, dx)
    solid = voxelise_buildings(site, grid)
    save_voxels(out / "voxels.npz", grid, solid)
    summary = summarise(site, dx, int(solid.sum()), grid.shape, overture_status)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_site(
    name: str,
    bbox: BBox | None,
    out: Path,
    dx: float = 2.0,
    refresh: bool = False,
    osm_file: Path | None = None,
    use_overture: bool = True,
) -> dict:
    """Build the site folder.

    The OpenStreetMap data are downloaded for `bbox` (or reused from the cache) and joined with
    Overture Maps buildings, or read from `osm_file`, an OpenStreetMap XML file (`.osm` or
    `.osm.gz`) such as an export from openstreetmap.org, with no download at all. With a file
    and no box, the box is the file's own bounds.
    """
    if osm_file is None:
        if bbox is None:
            raise ValueError("give a box, or an OpenStreetMap file")
        download = osm.fetch_map(bbox, refresh=refresh)
        data = osm.parse(download.path)
        records = [download.record()]
        table, status = None, None
        if use_overture:
            try:
                got, release = overture.fetch_buildings(bbox, refresh=refresh)
                table = overture.load(got.path)
                records.append({**got.record(), "release": release})
                status = {"release": release}
            except (OSError, ValueError, KeyError, urllib.error.URLError, pa.ArrowException) as e:
                # A second source being unreachable should not stop the build; say so instead.
                status = {"error": f"Overture Maps not used: {type(e).__name__}: {e}"}
        return build_from_osm_data(name, bbox, data, out, dx, records, table, status)
    data = osm.parse(osm_file)
    bbox = bbox or data.bounds
    if bbox is None:
        raise ValueError(f"{osm_file} has no <bounds> element; give the box with --bbox")
    source = {
        "path": str(osm_file),
        "bytes": osm_file.stat().st_size,
        "sha256": sha256_file(osm_file),
        "licence": osm.LICENCE,
        "attribution": osm.ATTRIBUTION,
        "note": "local OpenStreetMap file supplied by the user",
    }
    return build_from_osm_data(name, bbox, data, out, dx, [source])
