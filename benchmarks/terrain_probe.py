"""The two elevation models compared over the HKU site, and what reading them costs.

For the 500 m HKU square with 1 km of surroundings (as `aircanopy site build` reads it):
- the difference between the Copernicus surface model and the GEDTM30 terrain model inside the
  square (roofs and treetops show up as positive differences);
- GEDTM30's own uncertainty inside the square;
- the size of each remote file, and the bytes GDAL downloaded to read the window (the sum of
  the byte ranges in GDAL's "Downloading a-b" debug messages).

Needs the network. Writes results/terrain_probe.json.

    uv run python benchmarks/terrain_probe.py
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
import urllib.request
from pathlib import Path

import numpy as np
import rasterio

from aircanopy import __version__
from aircanopy.geo import BBox, LocalFrame
from aircanopy.site.build import CONTEXT_M, TERRAIN_DX_M, surroundings
from aircanopy.site.terrain import Terrain, from_geotiff
from aircanopy.sources import terrain as elevation

ROOT = Path(__file__).resolve().parents[1]
HKU = BBox.around(114.1370, 22.2840, 500)
RANGE = re.compile(r"Downloading (\d+)-(\d+) \(")


class _Ranges(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.bytes = 0
        self.requests = 0

    def emit(self, record: logging.LogRecord) -> None:
        if m := RANGE.search(record.getMessage()):
            self.bytes += int(m.group(2)) - int(m.group(1)) + 1
            self.requests += 1


def file_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as r:
        return int(r.headers["Content-Length"])


def main() -> None:
    wide, extent = surroundings(HKU, CONTEXT_M)
    frame = LocalFrame.for_bbox(HKU)
    square = frame.bbox_extent(HKU)
    tmp = ROOT / "results" / ".terrain_probe_tmp"
    tmp.mkdir(exist_ok=True)
    log = logging.getLogger("rasterio")
    log.setLevel(logging.DEBUG)
    grids, cost = {}, {}
    for model in ("gedtm30", "copernicus"):
        m = elevation.MODELS[model]
        counter = _Ranges()
        log.addHandler(counter)
        with rasterio.Env(CPL_DEBUG="ON"):
            bands, transform = elevation.read_model(m, wide)
        log.removeHandler(counter)
        url = m.url or elevation.copernicus_tiles(wide)[0]
        cost[model] = {
            "file_bytes": file_size(url),
            "downloaded_bytes": counter.bytes,
            "range_requests": counter.requests,
            "window_pixels": [int(bands.shape[2]), int(bands.shape[1])],
        }
        tif = tmp / f"{model}.tif"
        profile = dict(
            driver="GTiff", width=bands.shape[2], height=bands.shape[1], count=2,
            dtype="float32", crs="EPSG:4326", transform=transform,
        )  # fmt: skip
        with rasterio.open(tif, "w", **profile) as ds:
            ds.write(bands.astype(np.float32))
        grids[model] = from_geotiff(tif, frame, extent, TERRAIN_DX_M, model, m.kind)

    def inside(t: Terrain, a: np.ndarray) -> np.ndarray:
        xs, ys = t.nodes()
        keep = np.ix_((ys >= square[1]) & (ys <= square[3]), (xs >= square[0]) & (xs <= square[2]))
        return a[keep]

    g, c = grids["gedtm30"], grids["copernicus"]
    diff = inside(g, c.elevation.astype(float) - g.elevation.astype(float))
    std = inside(g, g.std.astype(float))
    out = {
        "what": "Copernicus GLO-30 (surface) minus GEDTM30 (terrain) over the HKU square, "
        "GEDTM30 uncertainty, and the cost of reading each model",
        "run_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "aircanopy": __version__,
        "site_bbox": list(HKU.as_tuple()),
        "read_bbox": list(wide.as_tuple()),
        "grid_m": TERRAIN_DX_M,
        "surface_minus_terrain_m": {
            "mean": round(float(diff.mean()), 2),
            "min": round(float(diff.min()), 2),
            "max": round(float(diff.max()), 2),
            "grid_points": int(diff.size),
        },
        "gedtm30_std_m": {
            "mean": round(float(np.nanmean(std)), 2),
            "max": round(float(np.nanmax(std)), 2),
        },
        "cost": cost,
    }
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    path = ROOT / "results" / "terrain_probe.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
