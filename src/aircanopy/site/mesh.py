"""Triangle meshes of a site, for drawing, saved as glTF.

glTF (`.glb`) is the standard file format for 3D scenes on the web and in most 3D tools. glTF
uses "y up": its x points right, y up and z towards the viewer. Our site frame is "z up" (x east,
y north, z up), so every vertex (x, y, z) is written as (x, z, -y): east stays right, up becomes
y, and north points into the screen.

Some features have no size in the data (a road without a width, a tree without a mapped crown).
To draw them at all, this module uses DISPLAY_* values. They change only the picture: the physics
never reads them.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Polygon

from .fill import NEIGHBOUR_MEDIAN
from .model import Building, Road, Site, Tree

# Picture-only sizes for features the data do not size (see the module docstring).
DISPLAY_LANE_WIDTH_M = 3.0
DISPLAY_FOOTWAY_WIDTH_M = 2.0
DISPLAY_TRUNK_RADIUS_M = 0.15
DISPLAY_CROWN_SHARE = 0.6  # crown diameter as a share of the tree height, when not mapped
DISPLAY_CROWN_DEPTH_SHARE = 0.6  # crown depth as a share of the tree height (at most its width)
DISPLAY_TRUNK_INTO_CROWN = 0.25  # how far the drawn trunk reaches into the crown, as a share

#: Colours (RGBA, 0-255) by what a building's height is based on, so the picture shows it.
BUILDING_COLOURS = {
    "measured": (205, 202, 196, 255),
    "levels": (176, 186, 199, 255),
    "estimated": (214, 178, 150, 255),
}
ROAD_COLOUR = (88, 88, 92, 255)
FOOTWAY_COLOUR = (150, 150, 146, 255)
GROUND_COLOUR = (120, 132, 110, 255)
CROWN_COLOUR = (62, 122, 66, 255)
TRUNK_COLOUR = (96, 72, 52, 255)

Z_UP_TO_Y_UP = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
    dtype=float,
)


def height_basis(b: Building) -> str:
    if b.height_source == NEIGHBOUR_MEDIAN:
        return "estimated"
    if b.height_source == "osm:levels":
        return "levels"
    return "measured"


def building_mesh(b: Building) -> trimesh.Trimesh | None:
    if b.height is None or b.height <= b.base:
        return None
    mesh = trimesh.creation.extrude_polygon(b.footprint, b.height - b.base, engine="earcut")
    mesh.apply_translation((0.0, 0.0, b.base))
    return mesh


def _flat(poly: Polygon, z: float) -> trimesh.Trimesh | None:
    if poly.is_empty or poly.area <= 0:
        return None
    verts2, faces = trimesh.creation.triangulate_polygon(poly, engine="earcut")
    verts = np.column_stack([verts2, np.full(len(verts2), z)])
    return trimesh.Trimesh(verts, faces, process=False)


def road_mesh(r: Road, z: float = 0.02) -> trimesh.Trimesh | None:
    width = r.width or (r.lanes * DISPLAY_LANE_WIDTH_M if r.lanes else 2 * DISPLAY_LANE_WIDTH_M)
    if r.highway in ("footway", "pedestrian", "path", "steps", "cycleway", "corridor"):
        width = r.width or DISPLAY_FOOTWAY_WIDTH_M
    shape = r.line.buffer(width / 2, cap_style="flat", join_style="round")
    parts = [shape] if isinstance(shape, Polygon) else list(getattr(shape, "geoms", []))
    meshes = [m for p in parts if (m := _flat(p, z)) is not None]
    return trimesh.util.concatenate(meshes) if meshes else None


def tree_meshes(t: Tree) -> tuple[trimesh.Trimesh, trimesh.Trimesh] | None:
    """Trunk and crown of a tree with a known height; None if the height is unknown."""
    if t.height is None or t.height <= 0:
        return None
    diameter = t.crown_diameter or DISPLAY_CROWN_SHARE * t.height
    crown_height = min(t.height * DISPLAY_CROWN_DEPTH_SHARE, diameter)
    base = t.height - crown_height
    radius = (
        t.trunk_circumference / (2 * math.pi) if t.trunk_circumference else DISPLAY_TRUNK_RADIUS_M
    )
    trunk_top = base + DISPLAY_TRUNK_INTO_CROWN * crown_height
    trunk = trimesh.creation.cylinder(radius=radius, height=trunk_top, sections=10)
    trunk.apply_translation((t.x, t.y, trunk_top / 2))
    crown = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    crown.apply_scale((diameter / 2, diameter / 2, crown_height / 2))
    crown.apply_translation((t.x, t.y, base + crown_height / 2))
    return trunk, crown


def ground_mesh(site: Site) -> trimesh.Trimesh:
    xmin, ymin, xmax, ymax = site.extent()
    return _flat(Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]), 0.0)


def _with_material(mesh: trimesh.Trimesh, name: str, rgba: tuple[int, ...]) -> trimesh.Trimesh:
    # One named material per group (not per-face colours, which glTF does not have): smaller
    # files, and a viewer can restyle a whole group by its material name.
    material = trimesh.visual.material.PBRMaterial(
        name=name, baseColorFactor=rgba, metallicFactor=0.0, roughnessFactor=0.9
    )
    mesh.visual = trimesh.visual.TextureVisuals(material=material)
    return mesh


def scene(site: Site) -> trimesh.Scene:
    """One named mesh per kind of feature, in glTF's y-up axes."""
    groups: dict[str, list[trimesh.Trimesh]] = {}

    def add(name: str, mesh: trimesh.Trimesh | None) -> None:
        if mesh is not None and len(mesh.faces):
            groups.setdefault(name, []).append(mesh)

    add("ground", ground_mesh(site))
    for b in site.buildings:
        add(f"buildings_{height_basis(b)}", building_mesh(b))
    for r in site.roads:
        add("roads", road_mesh(r))
    for r in site.footways:
        add("footways", road_mesh(r, z=0.03))
    for t in site.trees:
        if (parts := tree_meshes(t)) is not None:
            add("tree_trunks", parts[0])
            add("tree_crowns", parts[1])

    colours = {
        "ground": GROUND_COLOUR,
        "roads": ROAD_COLOUR,
        "footways": FOOTWAY_COLOUR,
        "tree_trunks": TRUNK_COLOUR,
        "tree_crowns": CROWN_COLOUR,
        **{f"buildings_{k}": v for k, v in BUILDING_COLOURS.items()},
    }
    out = trimesh.Scene()
    for name, meshes in groups.items():
        mesh = trimesh.util.concatenate(meshes)
        mesh.apply_transform(Z_UP_TO_Y_UP)
        out.add_geometry(_with_material(mesh, name, colours[name]), node_name=name, geom_name=name)
    return out


def export_glb(site: Site, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(scene(site).export(file_type="glb"))
    return path
