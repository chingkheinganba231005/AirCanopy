"""Tree canopy of a site, and individual trees found in it.

The canopy height map (sources/canopy.py) is resampled onto a 1 m grid in the site's local
frame. It holds the height of the vegetation above the ground at every point (0 where there is
none). Inside building footprints it is set to 0: a tree cannot stand inside a building, and
there the map can only be wrong.

Individual trees are found with the usual method for canopy height maps (individual tree
detection):

1. The heights are smoothed a little (a Gaussian of 1 m), so that the map's whole-metre steps
   do not make false peaks.
2. A treetop is a point that is the highest within TOP_RADIUS_M around it (a local maximum).
3. Each crown is grown downhill from its treetop over the canopy until it meets the crown of a
   neighbour (watershed segmentation: water poured on an upside-down canopy runs down to one
   treetop or another, and the line where it divides is the edge between two crowns).
4. Only canopy at least TREE_CANOPY_MIN_M tall counts, and crowns narrower than
   MIN_CROWN_DIAMETER_M are dropped. Both values are the ones the map's own paper uses to tell
   trees from other things (Tolan et al. 2024: "thresholding height values of at least five
   meters (5m) as tree canopy extent"; human annotators labelled as trees vegetation "above
   one meter (1m) tall and with a canopy diameter of more than three meters (3m)").

A tree's crown diameter is that of a circle with the crown's area.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
import shapely
from pyproj import Transformer
from scipy import ndimage
from skimage.segmentation import watershed

from ..geo import LocalFrame
from .model import Building, Tree

TREE_CANOPY_MIN_M = 5.0  # Tolan et al. 2024, tree canopy extent
MIN_CROWN_DIAMETER_M = 3.0  # Tolan et al. 2024, annotation rule for a tree
TOP_RADIUS_M = 2.0  # a treetop is the highest point within this distance (assumption A-008)
SMOOTHING_M = 1.0  # standard deviation of the Gaussian smoothing (assumption A-008)

SOURCE = "canopy:meta-wri-chm-v1"


@dataclass
class Canopy:
    x0: float  # local x of the centre of cell [0, 0]
    y0: float
    dx: float
    height: np.ndarray  # [ny, nx], m above the ground; 0 where there is no vegetation
    imagery_dates: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return self.height.shape  # type: ignore[return-value]

    def centres(self) -> tuple[np.ndarray, np.ndarray]:
        ny, nx = self.shape
        return self.x0 + np.arange(nx) * self.dx, self.y0 + np.arange(ny) * self.dx

    def cover(self, min_height: float = TREE_CANOPY_MIN_M) -> float:
        """Share of the grid's area covered by canopy at least `min_height` tall."""
        return float((self.height >= min_height).mean()) if self.height.size else 0.0

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            x0=self.x0,
            y0=self.y0,
            dx=self.dx,
            height=self.height.astype(np.float32),
            imagery_dates=json.dumps(self.imagery_dates),
        )

    @classmethod
    def load(cls, path: Path) -> Canopy:
        with np.load(path) as z:
            return cls(
                x0=float(z["x0"]),
                y0=float(z["y0"]),
                dx=float(z["dx"]),
                height=z["height"],
                imagery_dates=json.loads(str(z["imagery_dates"])),
            )


def from_geotiff(
    path: Path, frame: LocalFrame, extent: tuple[float, float, float, float], dx: float = 1.0
) -> Canopy:
    """Resample a canopy height GeoTIFF onto cell centres `dx` apart covering `extent`.

    Nearest pixel, not a blend: heights are whole metres and crowns have sharp edges.
    """
    xmin, ymin, xmax, ymax = extent
    nx = max(1, math.ceil((xmax - xmin) / dx))
    ny = max(1, math.ceil((ymax - ymin) / dx))
    xc = xmin + (np.arange(nx) + 0.5) * dx
    yc = ymin + (np.arange(ny) + 0.5) * dx
    xx, yy = np.meshgrid(xc, yc)
    lon, lat = frame.to_lonlat(xx.ravel(), yy.ravel())
    with rasterio.open(path) as ds:
        data = ds.read(1).astype(np.float32)
        to_map = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        mx, my = to_map.transform(np.asarray(lon), np.asarray(lat))
        col, row = ~ds.transform @ (np.asarray(mx), np.asarray(my))
        tags = ds.tags()
    col = np.floor(col).astype(int)
    row = np.floor(row).astype(int)
    h, w = data.shape
    ok = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    values = np.zeros(col.shape, dtype=np.float32)
    values[ok] = data[row[ok], col[ok]]
    dates = json.loads(tags.get("imagery_dates", "[]"))
    return Canopy(float(xc[0]), float(yc[0]), dx, values.reshape(ny, nx), dates)


def remove_buildings(canopy: Canopy, buildings: list[Building]) -> int:
    """Set the canopy to 0 inside building footprints; returns how many cells changed."""
    xs, ys = canopy.centres()
    changed = 0
    for b in buildings:
        minx, miny, maxx, maxy = b.footprint.bounds
        i = np.nonzero((xs >= minx) & (xs <= maxx))[0]
        j = np.nonzero((ys >= miny) & (ys <= maxy))[0]
        if not len(i) or not len(j):
            continue
        xx, yy = np.meshgrid(xs[i], ys[j])
        inside = shapely.contains_xy(b.footprint, xx, yy)
        block = canopy.height[np.ix_(j, i)]
        changed += int((inside & (block > 0)).sum())
        block[inside] = 0.0
        canopy.height[np.ix_(j, i)] = block
    return changed


@dataclass
class Crown:
    x: float  # treetop, local metres
    y: float
    height: float  # m above the ground
    diameter: float  # m, of a circle with the crown's area
    label: int  # the crown's number in `labels`


def find_crowns(canopy: Canopy) -> tuple[list[Crown], np.ndarray]:
    """Crowns found in the canopy, and the grid of crown numbers (0 outside any crown)."""
    h = canopy.height.astype(float)
    smooth = ndimage.gaussian_filter(h, sigma=SMOOTHING_M / canopy.dx)
    tall = h >= TREE_CANOPY_MIN_M
    r = max(1, round(TOP_RADIUS_M / canopy.dx))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    disk = xx**2 + yy**2 <= r**2
    peaks = (smooth == ndimage.maximum_filter(smooth, footprint=disk)) & tall
    # A flat top gives several equal peaks side by side: one treetop per touching group.
    markers, _ = ndimage.label(peaks)
    labels = watershed(-smooth, markers, mask=tall)
    xs, ys = canopy.centres()
    crowns = []
    cell_area = canopy.dx**2
    for k, region in enumerate(ndimage.find_objects(labels), start=1):
        if region is None:
            continue
        inside = labels[region] == k
        diameter = 2.0 * math.sqrt(inside.sum() * cell_area / math.pi)
        if diameter < MIN_CROWN_DIAMETER_M:
            labels[region][inside] = 0
            continue
        block = np.where(inside, h[region], -np.inf)
        j, i = np.unravel_index(int(np.argmax(block)), block.shape)
        crowns.append(
            Crown(
                x=float(xs[region[1].start + i]),
                y=float(ys[region[0].start + j]),
                height=float(h[region][j, i]),
                diameter=diameter,
                label=k,
            )
        )
    return crowns, labels


def trees_from_canopy(canopy: Canopy, mapped: list[Tree]) -> tuple[list[Tree], dict[str, int]]:
    """Trees for the site: one per crown, joined with the trees mapped in OpenStreetMap.

    A mapped tree standing inside a crown lends it its id, species and other tags (and its
    height, if mapped: a height measured by a person is preferred to the map's estimate). A
    mapped tree inside no crown is kept as it is.
    """
    crowns, labels = find_crowns(canopy)
    xs, ys = canopy.centres()
    by_label: dict[int, Tree] = {}
    trees: list[Tree] = []
    for n, c in enumerate(crowns):
        t = Tree(
            id=f"canopy:{n}",
            x=c.x,
            y=c.y,
            height=c.height,
            crown_diameter=c.diameter,
            trunk_circumference=None,
            species=None,
            genus=None,
            leaf_type=None,
            leaf_cycle=None,
            height_source=SOURCE,
            crown_source=SOURCE,
        )
        by_label[c.label] = t
        trees.append(t)
    matched = 0
    kept = []
    for m in mapped:
        i = int(np.floor((m.x - xs[0]) / canopy.dx + 0.5))
        j = int(np.floor((m.y - ys[0]) / canopy.dx + 0.5))
        inside = 0 <= i < len(xs) and 0 <= j < len(ys)
        t = by_label.get(int(labels[j, i])) if inside else None
        if t is None or not t.id.startswith("canopy:"):
            kept.append(m)  # inside no crown, or the crown already has a mapped tree
            continue
        matched += 1
        t.id = m.id
        for attr in ("species", "genus", "leaf_type", "leaf_cycle", "trunk_circumference"):
            setattr(t, attr, getattr(m, attr))
        if m.height is not None:
            t.height, t.height_source = m.height, m.height_source
    counts = {"crowns": len(crowns), "mapped_matched": matched, "mapped_alone": len(kept)}
    return trees + kept, counts
