"""From a box on the map to a site folder: data, model, meshes and voxels.

A site folder holds:

- `site.json`: the site model (buildings, trees, roads in local metres, with sources);
- `scene.glb`: meshes for drawing;
- `voxels.npz`: the voxel grid for the physics;
- `summary.json`: counts, how many heights were mapped, estimated from storeys or estimated
  from neighbours, and the data sources with their licences.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ..geo import BBox
from ..sources import osm
from ..sources.cache import sha256_file
from .fill import MIN_NEIGHBOURS, NEIGHBOUR_MEDIAN, NEIGHBOUR_RADIUS_M, fill_building_heights
from .from_osm import site_from_osm
from .heights import STOREY_HEIGHT_M
from .mesh import export_glb
from .model import Site
from .voxel import grid_for_site, save_voxels, voxelise_buildings


def summarise(site: Site, dx: float, solid_cells: int, grid_shape: tuple[int, int, int]) -> dict:
    sources = Counter(b.height_source for b in site.buildings)
    return {
        "name": site.name,
        "bbox": list(site.bbox.as_tuple()),
        "size_m": [round(v, 1) for v in site.bbox.size_m()],
        "buildings": len(site.buildings),
        "building_heights": dict(sorted(sources.items())),
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
        "sources": site.sources,
    }


def build_from_osm_data(
    name: str, bbox: BBox, data: osm.OsmData, out: Path, dx: float, sources: list[dict]
) -> dict:
    site = site_from_osm(name, bbox, data, sources)
    fill_building_heights(site)
    out.mkdir(parents=True, exist_ok=True)
    site.save(out / "site.json")
    export_glb(site, out / "scene.glb")
    grid = grid_for_site(site, dx)
    solid = voxelise_buildings(site, grid)
    save_voxels(out / "voxels.npz", grid, solid)
    summary = summarise(site, dx, int(solid.sum()), grid.shape)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_site(
    name: str,
    bbox: BBox | None,
    out: Path,
    dx: float = 2.0,
    refresh: bool = False,
    osm_file: Path | None = None,
) -> dict:
    """Build the site folder from OpenStreetMap data.

    The data are downloaded for `bbox` (or reused from the cache), or read from `osm_file`, an
    OpenStreetMap XML file (`.osm` or `.osm.gz`) such as an export from openstreetmap.org. With a
    file and no box, the box is the file's own bounds.
    """
    if osm_file is None:
        if bbox is None:
            raise ValueError("give a box, or an OpenStreetMap file")
        download = osm.fetch_map(bbox, refresh=refresh)
        data = osm.parse(download.path)
        return build_from_osm_data(name, bbox, data, out, dx, [download.record()])
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
