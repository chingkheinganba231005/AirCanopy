"""The site as a grid of small cubes (voxels), the form the physics works on.

A Grid covers a box in local metres with cubes of side `dx`. Arrays are indexed [k, j, i] =
[z, y, x], so x varies fastest in memory. Cell (k, j, i) spans
x in [x0 + i dx, x0 + (i + 1) dx), and the same for y and z; its centre is half a cell further.

Cells are AIR, BUILDING or GROUND. Heights are absolute (metres above the geoid, like the
terrain), so the grid's z0 is near the lowest ground of the site. A cell is GROUND if its centre
is below the ground surface (site/terrain.py). A building fills every cell whose centre lies
inside its footprint, at or above its base and below its top. Deciding by the cell centre is
the standard simple rule: a building's volume on the grid is then right on average, with an
error of up to half a cell at each wall.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import shapely

from .model import Building, Site
from .terrain import Terrain

AIR, BUILDING, GROUND = 0, 1, 2


@dataclass(frozen=True)
class Grid:
    x0: float
    y0: float
    z0: float
    dx: float
    nx: int
    ny: int
    nz: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nz, self.ny, self.nx)

    def centers(self, axis: str) -> np.ndarray:
        n, o = {"x": (self.nx, self.x0), "y": (self.ny, self.y0), "z": (self.nz, self.z0)}[axis]
        return o + (np.arange(n) + 0.5) * self.dx

    def index_range(self, lo: float, hi: float, axis: str) -> tuple[int, int]:
        """Indices [a, b) of the cells whose centres c satisfy lo <= c < hi.

        Half-open, so a building from 5 m to 25 m on 2 m cells fills exactly 10 cells, and two
        stacked parts never claim the same cell.
        """
        n, o = {"x": (self.nx, self.x0), "y": (self.ny, self.y0), "z": (self.nz, self.z0)}[axis]
        a = max(0, math.ceil((lo - o) / self.dx - 0.5))
        b = min(n, math.ceil((hi - o) / self.dx - 0.5))
        return a, max(a, b)


def _ground_range(site: Site, terrain: Terrain | None) -> tuple[float, float]:
    """Lowest and highest ground inside the site's square (0, 0 without terrain)."""
    if terrain is None:
        return 0.0, 0.0
    xmin, ymin, xmax, ymax = site.extent()
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 64), np.linspace(ymin, ymax, 64))
    z = terrain.at(xx, yy)
    xn, yn = terrain.nodes()
    nodes = terrain.elevation[np.ix_((yn >= ymin) & (yn <= ymax), (xn >= xmin) & (xn <= xmax))]
    lo = min(float(z.min()), float(nodes.min()) if nodes.size else np.inf)
    hi = max(float(z.max()), float(nodes.max()) if nodes.size else -np.inf)
    return lo, hi


def grid_for_site(
    site: Site, dx: float, height: float | None = None, terrain: Terrain | None = None
) -> Grid:
    """A grid over the site's box, from just below its lowest ground up to the highest building
    top or ground (or to `height` above the lowest ground)."""
    if dx <= 0:
        raise ValueError("the cell size must be positive")
    xmin, ymin, xmax, ymax = site.extent()
    low, high = _ground_range(site, terrain)
    z0 = math.floor(low / dx) * dx
    tops = [(b.ground or 0.0) + b.height for b in site.buildings if b.height is not None]
    top = max([high, *tops]) if height is None else low + height
    nx = max(1, math.ceil((xmax - xmin) / dx))
    ny = max(1, math.ceil((ymax - ymin) / dx))
    nz = max(1, math.ceil((top - z0) / dx))
    return Grid(x0=xmin, y0=ymin, z0=z0, dx=dx, nx=nx, ny=ny, nz=nz)


def voxelise_ground(grid: Grid, terrain: Terrain) -> np.ndarray:
    """uint8 array [z, y, x]: GROUND where the cell's centre is below the ground surface."""
    xx, yy = np.meshgrid(grid.centers("x"), grid.centers("y"))
    surface = terrain.at(xx, yy)  # [y, x]
    zc = grid.centers("z")
    return np.where(zc[:, None, None] < surface[None], GROUND, AIR).astype(np.uint8)


def voxelise(site: Site, grid: Grid, terrain: Terrain | None = None) -> np.ndarray:
    """Ground and buildings together; where both claim a cell, it is a building cell."""
    if terrain is None:
        solid = np.zeros(grid.shape, dtype=np.uint8)
    else:
        solid = voxelise_ground(grid, terrain)
    solid[voxelise_buildings(site, grid) == BUILDING] = BUILDING
    return solid


def voxelise_buildings(site: Site, grid: Grid) -> np.ndarray:
    """uint8 array [z, y, x]: BUILDING where a building fills the cell, AIR elsewhere."""
    solid = np.zeros(grid.shape, dtype=np.uint8)
    xc, yc = grid.centers("x"), grid.centers("y")
    for b in site.buildings:
        if b.height is None:
            continue
        minx, miny, maxx, maxy = b.footprint.bounds
        i0, i1 = grid.index_range(minx, maxx, "x")
        j0, j1 = grid.index_range(miny, maxy, "y")
        ground = b.ground or 0.0
        k0, k1 = grid.index_range(ground + b.base, ground + b.height, "z")
        if i0 >= i1 or j0 >= j1 or k0 >= k1:
            continue
        xx, yy = np.meshgrid(xc[i0:i1], yc[j0:j1])
        inside = shapely.contains_xy(b.footprint, xx, yy)
        solid[k0:k1, j0:j1, i0:i1][:, inside] = BUILDING
    return solid


def block_volume(buildings: Iterable[Building]) -> float:
    """Exact volume, in m^3, of buildings drawn as blocks, counting any overlap once.

    The blocks are cut into horizontal slabs at every base and top; in each slab the volume is
    the area of the union of the footprints present times the slab's thickness. Used to check
    the voxel grid.
    """
    blocks = [b for b in buildings if b.height is not None and b.height > b.base]
    spans = [((b.ground or 0.0) + b.base, (b.ground or 0.0) + b.height) for b in blocks]
    levels = sorted({z for span in spans for z in span})
    total = 0.0
    for lo, hi in itertools.pairwise(levels):
        prints = [
            b.footprint
            for b, (bottom, top) in zip(blocks, spans, strict=True)
            if bottom <= lo and top >= hi
        ]
        if prints:
            total += shapely.union_all(prints).area * (hi - lo)
    return total


def save_voxels(path: Path, grid: Grid, solid: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, solid=solid, **{k: np.asarray(v) for k, v in asdict(grid).items()})


def load_voxels(path: Path) -> tuple[Grid, np.ndarray]:
    with np.load(path) as z:
        grid = Grid(**{k: z[k].item() for k in ("x0", "y0", "z0", "dx", "nx", "ny", "nz")})
        return grid, z["solid"]
