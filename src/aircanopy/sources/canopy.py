"""Tree canopy heights for any site: the global canopy height map of Meta and WRI.

Meta and the World Resources Institute mapped the height of tree canopy over the whole world
at about 1 m, from Maxar satellite images, with a model trained on aerial lidar (Tolan et al.
2024, Remote Sensing of Environment, doi:10.1016/j.rse.2023.113888). The paper reports a mean
absolute error of 2.8 m and a mean error of 0.6 m against lidar it was not trained on.

The map (version 1) is split into web-map tiles at zoom 9, each named by its quadkey: a string
of digits 0-3 that picks one quarter of the map, then one quarter of that, and so on, nine
times. Tiles are GeoTIFFs in Web Mercator (EPSG:3857) with pixels of about 1.2 m (about 1.1 m
on the ground at Hong Kong's latitude) holding whole metres. A metadata file per tile gives the
dates of the satellite images.

Licence CC BY 4.0. Version 2 of the map exists in the same bucket but its licence is not
documented yet, so it is not used.
"""

from __future__ import annotations

import datetime as dt
import json
import math

import numpy as np
import rasterio
import shapely
from rasterio.merge import merge
from rasterio.warp import transform_bounds
from shapely.geometry import shape

from ..geo import BBox
from .cache import Download, _meta_path, cache_dir, cached, sha256_file
from .overture import get_json
from .terrain import GDAL_ENV

BASE = "https://dataforgood-fb-data.s3.amazonaws.com/forests/v1/alsgedi_global_v6_float"
ZOOM = 9
LICENCE = "CC-BY-4.0"
#: The citation the AWS Open Data Registry gives for the dataset.
ATTRIBUTION = (
    "Meta and World Resources Institute (WRI) 2024, High Resolution Canopy Height Maps; "
    "source imagery © 2016 Maxar"
)


def tile_xy(lon: float, lat: float, zoom: int = ZOOM) -> tuple[int, int]:
    """Web-map tile column and row (Web Mercator, row 0 at the north)."""
    n = 2**zoom
    s = math.sin(math.radians(lat))
    x = int((lon + 180.0) / 360.0 * n)
    y = int((0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * n)
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def quadkey(x: int, y: int, zoom: int = ZOOM) -> str:
    """The tile's quadkey: one digit per zoom level, 0-3 for the quarter it falls in."""
    return "".join(str(((x >> i) & 1) + 2 * ((y >> i) & 1)) for i in range(zoom - 1, -1, -1))


def tiles_for_bbox(bbox: BBox) -> list[str]:
    x0, y0 = tile_xy(bbox.west, bbox.north)
    x1, y1 = tile_xy(bbox.east, bbox.south)
    return [quadkey(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]


def imagery_dates(bbox: BBox) -> list[str]:
    """Dates of the satellite images behind the canopy heights inside `bbox`."""
    area = shapely.box(*bbox.as_tuple())
    dates = set()
    for key in tiles_for_bbox(bbox):
        meta = get_json(f"{BASE}/metadata/{key}.geojson")
        for f in meta.get("features", []):
            if shape(f["geometry"]).intersects(area):
                dates.add(f["properties"]["acq_date"])
    return sorted(dates)


def fetch_canopy(bbox: BBox, refresh: bool = False) -> Download:
    """Canopy heights over `bbox`, saved in the cache as a small GeoTIFF (EPSG:3857, metres)."""
    w, s, e, n = bbox.as_tuple()
    name = f"canopy/meta_wri_v1_{w:.6f}_{s:.6f}_{e:.6f}_{n:.6f}.tif"
    if not refresh and (hit := cached(name)) is not None:
        return hit
    keys = tiles_for_bbox(bbox)
    with rasterio.Env(**GDAL_ENV):
        sources = [rasterio.open(f"{BASE}/chm/{k}.tif") for k in keys]
        try:
            bounds = transform_bounds("EPSG:4326", sources[0].crs, w, s, e, n)
            heights, transform = merge(sources, bounds=bounds, nodata=0)
            crs = sources[0].crs
        finally:
            for src in sources:
                src.close()
    dates = imagery_dates(bbox)
    path = cache_dir() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    profile = {
        "driver": "GTiff",
        "width": heights.shape[2],
        "height": heights.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
    }
    with rasterio.open(tmp, "w", **profile) as out:
        out.write(heights.astype(np.uint8))
        out.update_tags(
            model="meta-wri-chm-v1",
            units="metres",
            imagery_dates=json.dumps(dates),
            tiles=",".join(keys),
        )
    tmp.replace(path)
    download = Download(
        path=path,
        url=";".join(f"{BASE}/chm/{k}.tif" for k in keys),
        accessed=dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        bytes=path.stat().st_size,
        sha256=sha256_file(path),
        licence=LICENCE,
        attribution=ATTRIBUTION,
    )
    _meta_path(path).write_text(json.dumps(download.record(), indent=2) + "\n")
    return download
