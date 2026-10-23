"""How close the voxel grid's building volume is to the exact volume, for several cell sizes.

Uses the stored OpenStreetMap sample near HKU (tests/data/osm/hku_small.osm.gz) and only the
buildings wholly inside the grid, so no building is cut by the edge. Writes
results/voxel_volume.json.

    uv run python benchmarks/voxel_volume.py
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import shapely

from aircanopy import __version__
from aircanopy.geo import BBox
from aircanopy.site.fill import fill_building_heights
from aircanopy.site.from_osm import site_from_osm
from aircanopy.site.voxel import block_volume, grid_for_site, voxelise_buildings
from aircanopy.sources import osm
from aircanopy.sources.cache import sha256_file

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "data" / "osm" / "hku_small.osm.gz"
BOX = BBox(114.1360, 22.2832, 114.1380, 22.2848)
CELL_SIZES_M = (0.5, 1.0, 2.0, 4.0)


def main() -> None:
    site = site_from_osm("hku_small", BOX, osm.parse(FIXTURE))
    fill_building_heights(site)
    # Same buildings for every cell size: those inside the coarsest grid's box.
    grid = grid_for_site(site, max(CELL_SIZES_M))
    inner = shapely.box(*site.extent())
    site.buildings = [
        b for b in site.buildings if b.height is not None and inner.contains(b.footprint)
    ]
    exact = block_volume(site.buildings)
    rows = []
    for dx in CELL_SIZES_M:
        grid = grid_for_site(site, dx)
        cells = int(voxelise_buildings(site, grid).sum())
        volume = cells * dx**3
        rows.append(
            {
                "dx_m": dx,
                "building_cells": cells,
                "voxel_volume_m3": round(volume, 1),
                "relative_error": round(volume / exact - 1, 5),
            }
        )
    out = {
        "what": "building volume on the voxel grid against the exact block volume",
        "run_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "aircanopy": __version__,
        "fixture": str(FIXTURE.relative_to(ROOT)),
        "fixture_sha256": sha256_file(FIXTURE),
        "buildings": len(site.buildings),
        "exact_volume_m3": round(exact, 1),
        "cell_sizes": rows,
    }
    path = ROOT / "results" / "voxel_volume.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
