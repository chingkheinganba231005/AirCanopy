"""The ground of a site: elevations on a regular grid in the site's local metres.

The elevation models are grids in longitude and latitude (sources/terrain.py). The site works
in metres, so the model is resampled (bilinearly: a straight-line blend along x, then along y)
onto a grid of points `dx` metres apart in the site's local frame.

Between grid points the ground is made of flat triangles, two per grid square, split along the
diagonal from the south-west corner to the north-east corner. These are the triangles the 3D
view draws, so the voxel grid, buildings, trees and roads all stand on exactly the drawn ground.

Elevations are metres above the geoid (about mean sea level). The site's other heights
(buildings, trees) are above the ground, so a building's absolute top is its ground plus its
height. A building on a slope stands on the lowest ground under its footprint, because
OpenStreetMap measures `height` from the lowest point where the building meets the ground
(Simple 3D Buildings, wiki revision 3106809).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import shapely
from shapely.geometry import Polygon

from ..geo import LocalFrame


@dataclass
class Terrain:
    x0: float  # local x of grid point [0, 0], m
    y0: float  # local y of grid point [0, 0], m
    dx: float  # spacing of grid points, m
    elevation: np.ndarray  # [ny, nx], m above the geoid
    std: np.ndarray  # [ny, nx], uncertainty of the elevation (standard deviation), m; NaN if none
    model: str  # "gedtm30", "copernicus", or "flat"
    kind: str  # "terrain" (bare earth), "surface" (includes buildings and trees), or "flat"

    @property
    def shape(self) -> tuple[int, int]:
        return self.elevation.shape  # type: ignore[return-value]

    def nodes(self) -> tuple[np.ndarray, np.ndarray]:
        ny, nx = self.shape
        return self.x0 + np.arange(nx) * self.dx, self.y0 + np.arange(ny) * self.dx

    def at(self, x, y) -> np.ndarray:
        """Ground elevation at local points, on the triangulated surface; clamped at the edges."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        ny, nx = self.shape
        fx = np.clip((x - self.x0) / self.dx, 0, nx - 1)
        fy = np.clip((y - self.y0) / self.dx, 0, ny - 1)
        i = np.minimum(np.floor(fx).astype(int), nx - 2)
        j = np.minimum(np.floor(fy).astype(int), ny - 2)
        tx = fx - i
        ty = fy - j
        e = self.elevation.astype(float)
        z00, z10, z11, z01 = e[j, i], e[j, i + 1], e[j + 1, i + 1], e[j + 1, i]
        # Below the diagonal (tx >= ty) the triangle is (00, 10, 11); above it, (00, 11, 01).
        lower = z00 + tx * (z10 - z00) + ty * (z11 - z10)
        upper = z00 + ty * (z01 - z00) + tx * (z11 - z01)
        return np.where(tx >= ty, lower, upper)

    def faces(self) -> np.ndarray:
        """Triangles of the surface as indices into the grid points (row-major), two per square,
        counter-clockwise seen from above."""
        ny, nx = self.shape
        j, i = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), indexing="ij")
        p00 = (j * nx + i).ravel()
        p10, p01 = p00 + 1, p00 + nx
        p11 = p01 + 1
        return np.concatenate([np.column_stack([p00, p10, p11]), np.column_stack([p00, p11, p01])])

    def lowest_under(self, footprint: Polygon) -> float:
        """The lowest ground under a footprint: along its outline (every quarter grid step)
        and at the grid points inside it. The surface is flat between those, so its lowest
        point under the footprint is one of them, to within the outline's sampling."""
        outline = np.asarray(shapely.segmentize(footprint.exterior, self.dx / 4).coords)
        xs, ys = self.nodes()
        xmin, ymin, xmax, ymax = footprint.bounds
        gx = xs[(xs >= xmin) & (xs <= xmax)]
        gy = ys[(ys >= ymin) & (ys <= ymax)]
        values = [self.at(outline[:, 0], outline[:, 1]).min()]
        if len(gx) and len(gy):
            xx, yy = np.meshgrid(gx, gy)
            inside = shapely.contains_xy(footprint, xx, yy)
            if inside.any():
                values.append(self.at(xx[inside], yy[inside]).min())
        return float(min(values))

    def summary(self, area: tuple[float, float, float, float]) -> dict:
        """Lowest, highest and mean uncertainty of the ground inside `area` (xmin, ymin, xmax,
        ymax)."""
        xs, ys = self.nodes()
        inside = np.ix_((ys >= area[1]) & (ys <= area[3]), (xs >= area[0]) & (xs <= area[2]))
        e = self.elevation[inside]
        s = self.std[inside]
        return {
            "model": self.model,
            "kind": self.kind,
            "vertical_datum": "EGM2008 geoid (about mean sea level)",
            "grid_m": self.dx,
            "ground_min_m": round(float(np.nanmin(e)), 1) if e.size else None,
            "ground_max_m": round(float(np.nanmax(e)), 1) if e.size else None,
            "std_mean_m": round(float(np.nanmean(s)), 1) if np.isfinite(s).any() else None,
        }

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            x0=self.x0,
            y0=self.y0,
            dx=self.dx,
            elevation=self.elevation.astype(np.float32),
            std=self.std.astype(np.float32),
            model=self.model,
            kind=self.kind,
        )

    @classmethod
    def load(cls, path: Path) -> Terrain:
        with np.load(path) as z:
            return cls(
                x0=float(z["x0"]),
                y0=float(z["y0"]),
                dx=float(z["dx"]),
                elevation=z["elevation"],
                std=z["std"],
                model=str(z["model"]),
                kind=str(z["kind"]),
            )


def flat(extent: tuple[float, float, float, float], dx: float = 30.0) -> Terrain:
    """Level ground at elevation 0, for sites built without an elevation model."""
    xmin, ymin, xmax, ymax = extent
    nx = max(2, int(np.ceil((xmax - xmin) / dx)) + 1)
    ny = max(2, int(np.ceil((ymax - ymin) / dx)) + 1)
    zeros = np.zeros((ny, nx), dtype=np.float32)
    return Terrain(xmin, ymin, dx, zeros, np.full_like(zeros, np.nan), "flat", "flat")


def from_geotiff(
    path: Path,
    frame: LocalFrame,
    extent: tuple[float, float, float, float],
    dx: float = 30.0,
    model: str = "",
    kind: str = "",
) -> Terrain:
    """Resample a two-band GeoTIFF in longitude and latitude (elevation, standard deviation)
    onto local grid points `dx` apart covering `extent` (xmin, ymin, xmax, ymax)."""
    with rasterio.open(path) as ds:
        bands = ds.read().astype(np.float64)
        inverse = ~ds.transform
        tags = ds.tags()
    model = model or tags.get("model", "unknown")
    kind = kind or tags.get("kind", "unknown")
    xmin, ymin, xmax, ymax = extent
    nx = int(np.ceil((xmax - xmin) / dx)) + 1
    ny = int(np.ceil((ymax - ymin) / dx)) + 1
    xx, yy = np.meshgrid(xmin + np.arange(nx) * dx, ymin + np.arange(ny) * dx)
    lon, lat = frame.to_lonlat(xx.ravel(), yy.ravel())
    col, row = inverse @ (np.asarray(lon), np.asarray(lat))
    # Values sit at pixel centres, half a pixel in from the pixel's corner.
    fc = np.asarray(col) - 0.5
    fr = np.asarray(row) - 0.5
    h, w = bands.shape[1:]
    if fc.min() < -0.5 or fr.min() < -0.5 or fc.max() > w - 0.5 or fr.max() > h - 0.5:
        raise ValueError("the elevation file does not cover the site")
    fc = np.clip(fc, 0, w - 1)
    fr = np.clip(fr, 0, h - 1)
    c0 = np.minimum(np.floor(fc).astype(int), w - 2)
    r0 = np.minimum(np.floor(fr).astype(int), h - 2)
    tc = (fc - c0)[None]
    tr = (fr - r0)[None]
    top = bands[:, r0, c0] * (1 - tc) + bands[:, r0, c0 + 1] * tc
    bottom = bands[:, r0 + 1, c0] * (1 - tc) + bands[:, r0 + 1, c0 + 1] * tc
    values = (top * (1 - tr) + bottom * tr).reshape(2, ny, nx)
    elevation = values[0]
    if np.isnan(elevation).any():
        # Gaps (open water in some models) are at sea level, as the Copernicus registry says.
        elevation = np.where(np.isnan(elevation), 0.0, elevation)
    return Terrain(
        x0=xmin,
        y0=ymin,
        dx=dx,
        elevation=elevation.astype(np.float32),
        std=values[1].astype(np.float32),
        model=model,
        kind=kind,
    )
