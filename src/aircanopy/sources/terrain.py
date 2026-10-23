"""Ground elevation for any site, from global elevation models.

Two models, both at 1 arc-second (about 30 m):

- GEDTM30 (the default) is a terrain model: bare earth, with buildings and trees taken away. It
  was made by machine learning from about 30 billion lidar points of the ICESat-2 and GEDI
  satellites (Ho and Hengl; method in Ho et al. 2025, doi:10.7717/peerj.19673). Every pixel
  also has an uncertainty (a standard deviation, in metres), which we keep. Licence CC BY 4.0.
  Its authors call release 1.2 a first release, "for testing purposes".
- Copernicus GLO-30 is a surface model: it "represents the surface of the Earth including
  buildings, infrastructure and vegetation" (AWS Open Data Registry). In a city its "ground"
  lies on roofs and treetops, so it is only a fallback, and a site built on it says so.

Heights are metres above the geoid (EGM2008), close to mean sea level. Both models are
cloud-optimised GeoTIFFs: images stored in tiles, so that a window can be read over HTTP with
range requests, as for Overture's Parquet files. rasterio (GDAL) does the reading.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.windows import Window, from_bounds

from ..geo import BBox
from .cache import Download, _meta_path, cache_dir, cached, sha256_file

#: GDAL settings for reading remote files: do not list folders, retry dropped requests.
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "1",
}


@dataclass(frozen=True)
class ElevationModel:
    key: str
    name: str
    kind: str  # "terrain" (bare earth) or "surface" (includes buildings and trees)
    licence: str
    attribution: str
    url: str | None = None  # one global file
    std_url: str | None = None  # its uncertainty, if published


GEDTM30 = ElevationModel(
    key="gedtm30",
    name="GEDTM30 v1.2 (Global Ensemble Digital Terrain Model, 30 m)",
    kind="terrain",
    licence="CC-BY-4.0",
    attribution="GEDTM30 v1.2, Yufeng Ho and Tom Hengl, OpenGeoHub, doi:10.5281/zenodo.18887460",
    url="https://s3.opengeohub.org/global/dtm/v1.2/"
    "gedtm_rf_m_30m_s_20060101_20151231_go_epsg.4326.3855_v1.2.tif",
    std_url="https://s3.opengeohub.org/global/dtm/v1.2/"
    "gedtm_rf_std_30m_s_20060101_20151231_go_epsg.4326.3855_v1.2.tif",
)

COPERNICUS = ElevationModel(
    key="copernicus",
    name="Copernicus DEM GLO-30 (a surface model)",
    kind="surface",
    licence="Copernicus DEM licence (free, attribution required)",
    attribution="produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus "
    "Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and "
    "ESA; all rights reserved",
)

MODELS = {m.key: m for m in (GEDTM30, COPERNICUS)}


def copernicus_tiles(bbox: BBox) -> list[str]:
    """URLs of the 1-degree Copernicus GLO-30 tiles that cover `bbox`."""
    urls = []
    for lat in range(math.floor(bbox.south), math.floor(bbox.north) + 1):
        for lon in range(math.floor(bbox.west), math.floor(bbox.east) + 1):
            ns = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
            ew = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
            tile = f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"
            urls.append(f"https://copernicus-dem-30m.s3.amazonaws.com/{tile}/{tile}.tif")
    return urls


def _read_window(url: str, bbox: BBox) -> tuple[np.ndarray, rasterio.Affine, float | None]:
    with rasterio.open(url) as ds:
        w = from_bounds(*bbox.as_tuple(), ds.transform)
        # Round outward, so the window covers the whole box.
        col0, row0 = math.floor(w.col_off), math.floor(w.row_off)
        col1, row1 = math.ceil(w.col_off + w.width), math.ceil(w.row_off + w.height)
        window = Window(col0, row0, col1 - col0, row1 - row0)
        data = ds.read(1, window=window, boundless=True, fill_value=ds.nodata or np.nan)
        return data.astype(np.float32), ds.window_transform(window), ds.nodata


def read_model(model: ElevationModel, bbox: BBox) -> tuple[np.ndarray, rasterio.Affine]:
    """Bands [elevation, standard deviation (NaN if not published)] over `bbox`, and their
    geotransform. Missing values are NaN."""
    with rasterio.Env(**GDAL_ENV):
        if model.url:
            elevation, transform, nodata = _read_window(model.url, bbox)
            if nodata is not None:
                elevation[elevation == np.float32(nodata)] = np.nan
            std = np.full_like(elevation, np.nan)
            if model.std_url:
                std, _, std_nodata = _read_window(model.std_url, bbox)
                if std_nodata is not None:
                    std[std == np.float32(std_nodata)] = np.nan
            return np.stack([elevation, std]), transform
        sources = []
        for url in copernicus_tiles(bbox):
            try:
                sources.append(rasterio.open(url))
            except rasterio.errors.RasterioIOError:
                # The registry says ocean areas have no tiles and are at height zero.
                continue
        if not sources:
            raise ValueError(f"no Copernicus tile covers {bbox.as_tuple()} (open sea?)")
        try:
            elevation, transform = merge(sources, bounds=bbox.as_tuple(), nodata=np.nan)
        finally:
            for s in sources:
                s.close()
        elevation = elevation[0].astype(np.float32)
        return np.stack([elevation, np.full_like(elevation, np.nan)]), transform


def fetch_terrain(bbox: BBox, model: str = "gedtm30", refresh: bool = False) -> Download:
    """The model's window over `bbox`, saved in the cache as a small two-band GeoTIFF."""
    m = MODELS[model]
    w, s, e, n = bbox.as_tuple()
    name = f"terrain/{m.key}_{w:.6f}_{s:.6f}_{e:.6f}_{n:.6f}.tif"
    if not refresh and (hit := cached(name)) is not None:
        return hit
    bands, transform = read_model(m, bbox)
    path = cache_dir() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    profile = {
        "driver": "GTiff",
        "width": bands.shape[2],
        "height": bands.shape[1],
        "count": 2,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": np.nan,
        "compress": "deflate",
    }
    with rasterio.open(tmp, "w", **profile) as out:
        out.write(bands)
        out.set_band_description(1, "elevation_m")
        out.set_band_description(2, "elevation_std_m")
        out.update_tags(model=m.key, kind=m.kind, vertical_datum="EGM2008 geoid")
    tmp.replace(path)
    download = Download(
        path=path,
        url=m.url or ";".join(copernicus_tiles(bbox)),
        accessed=dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        bytes=path.stat().st_size,
        sha256=sha256_file(path),
        licence=m.licence,
        attribution=m.attribution,
    )
    _meta_path(path).write_text(json.dumps(download.record(), indent=2) + "\n")
    return download
