"""Pictures of a site folder, seen from above, for the journal and the README.

    uv run python scripts/figures.py sites/hku docs/images/hku

writes, with north up:

- `<prefix>_voxels.png`: the voxel grid. Buildings are coloured by their height above the
  ground (light: low, dark: tall), on the shaded ground.
- `<prefix>_canopy.png`: the canopy height map (green, brighter is taller), buildings in grey,
  the crowns found in it outlined in yellow and their treetops in red.
- `<prefix>_terrain.png`: the ground over the site and its surroundings, lit from the
  north-west, with contour lines every 20 m, ground at 0 m (the sea) in blue and the
  simulated square in amber.

Every picture has a white scale bar in its lower-left corner: 100 m long, or 500 m in the
terrain picture.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from aircanopy.site.canopy import Canopy, find_crowns
from aircanopy.site.model import Site
from aircanopy.site.terrain import Terrain
from aircanopy.site.voxel import BUILDING, GROUND, load_voxels

AMBER = (255, 176, 0)


def hillshade(z: np.ndarray, dx: float, azimuth_deg: float = 315, altitude_deg: float = 45):
    """Brightness of a surface lit from `azimuth_deg` (clockwise from north), 0 to 1.

    Rows of `z` run south to north (row 0 is the south edge).
    """
    gy, gx = np.gradient(z.astype(float), dx)  # slopes towards north and east
    az, alt = np.radians(azimuth_deg), np.radians(altitude_deg)
    light = np.array([np.sin(az) * np.cos(alt), np.cos(az) * np.cos(alt), np.sin(alt)])
    normal = np.stack([-gx, -gy, np.ones_like(gx)])
    normal /= np.linalg.norm(normal, axis=0)
    return np.clip(np.tensordot(light, normal, axes=1), 0, 1)


def to_image(rgb: np.ndarray, scale: int) -> Image.Image:
    """An image from an RGB array whose row 0 is the south edge, enlarged `scale` times."""
    img = Image.fromarray(np.ascontiguousarray(rgb[::-1]).astype(np.uint8), "RGB")
    return img.resize((img.width * scale, img.height * scale), Image.NEAREST)


def scale_bar(img: Image.Image, metres_per_pixel: float, length_m: float = 100) -> None:
    draw = ImageDraw.Draw(img)
    n = round(length_m / metres_per_pixel)
    x0, y0 = 16, img.height - 20
    draw.rectangle([x0 - 2, y0 - 6, x0 + n + 2, y0 + 6], fill=(20, 24, 28))
    draw.rectangle([x0, y0 - 3, x0 + n, y0 + 3], fill=(255, 255, 255))


def voxels_figure(site_dir: Path, out: Path, scale: int = 3) -> None:
    grid, solid = load_voxels(site_dir / "voxels.npz")
    nz = solid.shape[0]
    zc = grid.z0 + (np.arange(nz) + 0.5) * grid.dx
    is_building = solid == BUILDING
    is_ground = solid == GROUND
    # Top of the ground and of buildings in every column.
    ground_top = np.where(
        is_ground.any(axis=0), zc[nz - 1 - np.argmax(is_ground[::-1], axis=0)], grid.z0
    )
    building_top = np.where(
        is_building.any(axis=0), zc[nz - 1 - np.argmax(is_building[::-1], axis=0)], np.nan
    )
    shade = hillshade(ground_top, grid.dx)
    rgb = np.empty((*solid.shape[1:], 3))
    for c, (lo, hi) in enumerate([(70, 150), (80, 160), (70, 140)]):
        rgb[..., c] = lo + (hi - lo) * shade
    above = building_top - ground_top
    has = np.isfinite(above)
    t = np.clip(above / 100.0, 0, 1)  # 0 m light, 100 m and more dark
    light, dark = np.array([226, 230, 238]), np.array([44, 70, 140])
    colour = light[None, None, :] * (1 - t[..., None]) + dark[None, None, :] * t[..., None]
    rgb[has] = colour[has]
    img = to_image(rgb, scale)
    scale_bar(img, grid.dx / scale)
    img.save(out)


def canopy_figure(site_dir: Path, out: Path, scale: int = 1) -> None:
    site = Site.load(site_dir / "site.json")
    canopy = Canopy.load(site_dir / "canopy.npz")
    h = canopy.height.astype(float)
    t = np.clip(h / 20.0, 0, 1)
    rgb = np.stack([30 + 20 * t, 30 + 200 * t, 35 + 30 * t], axis=-1)
    import shapely

    xs, ys = canopy.centres()
    xx, yy = np.meshgrid(xs, ys)
    for b in site.buildings:
        rgb[shapely.contains_xy(b.footprint, xx, yy)] = (92, 92, 104)
    crowns, labels = find_crowns(canopy)
    edge = np.zeros(h.shape, bool)
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[1:, :] |= labels[1:, :] != labels[:-1, :]
    rgb[edge & (labels > 0)] = (255, 220, 0)
    for c in crowns:
        i = round((c.x - xs[0]) / canopy.dx)
        j = round((c.y - ys[0]) / canopy.dx)
        rgb[max(0, j - 1) : j + 2, max(0, i - 1) : i + 2] = (255, 40, 40)
    img = to_image(rgb, scale)
    scale_bar(img, canopy.dx / scale)
    img.save(out)


def _smooth(a: np.ndarray, scale: int) -> np.ndarray:
    """`a` enlarged `scale` times by bilinear interpolation (row 0 still the south edge)."""
    img = Image.fromarray(a.astype(np.float32), mode="F")
    return np.array(img.resize((img.width * scale, img.height * scale), Image.BILINEAR))


def terrain_figure(site_dir: Path, out: Path, scale: int = 5) -> None:
    site = Site.load(site_dir / "site.json")
    terrain = Terrain.load(site_dir / "terrain.npz")
    z = _smooth(terrain.elevation, scale).astype(float)
    shade = hillshade(z, terrain.dx / scale)
    lo, hi = np.nanmin(z), np.nanmax(z)
    t = (z - lo) / max(hi - lo, 1.0)
    low, high = np.array([86, 128, 92]), np.array([214, 204, 170])
    base = low[None, None, :] * (1 - t[..., None]) + high[None, None, :] * t[..., None]
    rgb = base * (0.45 + 0.55 * shade[..., None])
    rgb[z <= 0.0] = (64, 104, 140)  # elevation 0: the sea, in the elevation model
    # Contours every 20 m: where the 20 m band changes from one pixel to the next.
    band = np.floor(z / 20.0)
    line = np.zeros(band.shape, bool)
    line[:, 1:] |= band[:, 1:] != band[:, :-1]
    line[1:, :] |= band[1:, :] != band[:-1, :]
    rgb[line & (z > 0)] *= 0.55
    img = to_image(rgb, 1)
    draw = ImageDraw.Draw(img)
    xmin, ymin, xmax, ymax = site.extent()
    px = lambda x: (x - terrain.x0) / terrain.dx * scale  # noqa: E731
    py = lambda y: img.height - (y - terrain.y0) / terrain.dx * scale  # noqa: E731
    draw.rectangle([px(xmin), py(ymax), px(xmax), py(ymin)], outline=AMBER, width=3)
    scale_bar(img, terrain.dx / scale, 500)
    img.save(out)


def main(site_dir: Path, prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    voxels_figure(site_dir, Path(f"{prefix}_voxels.png"))
    if (site_dir / "canopy.npz").is_file():
        canopy_figure(site_dir, Path(f"{prefix}_canopy.png"))
    if (site_dir / "terrain.npz").is_file():
        terrain_figure(site_dir, Path(f"{prefix}_terrain.png"))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
