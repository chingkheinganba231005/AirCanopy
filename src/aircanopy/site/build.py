"""From a box on the map to a site folder: data, model, meshes and voxels.

A site folder holds:

- `site.json`: the site model (buildings, trees, roads in local metres, with sources);
- `scene.glb`: meshes for drawing;
- `voxels.npz`: the voxel grid for the physics;
- `summary.json`: counts, how many heights were measured, estimated from storeys or estimated
  from neighbours, and the data sources with their licences.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ..geo import BBox
from ..sources import osm
from .fill import NEIGHBOUR_MEDIAN, fill_building_heights
from .from_osm import site_from_osm
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


def build_site(name: str, bbox: BBox, out: Path, dx: float = 2.0, refresh: bool = False) -> dict:
    """Download (or reuse) OpenStreetMap data for the box and build the site folder."""
    download = osm.fetch_map(bbox, refresh=refresh)
    data = osm.parse(download.path)
    return build_from_osm_data(name, bbox, data, out, dx, [download.record()])
