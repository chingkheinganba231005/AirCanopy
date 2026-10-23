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
import shapely
import trimesh
from shapely.geometry import Polygon

from .heights import height_kind
from .model import Building, Road, Site, Tree
from .terrain import Terrain, flat

# Picture-only sizes for features the data do not size (see the module docstring).
DISPLAY_LANE_WIDTH_M = 3.0
DISPLAY_FOOTWAY_WIDTH_M = 2.0
DISPLAY_TRUNK_RADIUS_M = 0.15
DISPLAY_CROWN_SHARE = 0.6  # crown diameter as a share of the tree height, when not mapped
DISPLAY_CROWN_DEPTH_SHARE = 0.6  # crown depth as a share of the tree height (at most its width)
DISPLAY_TRUNK_INTO_CROWN = 0.25  # how far the drawn trunk reaches into the crown, as a share
DISPLAY_ROAD_LIFT_M = 0.15  # roads drawn this far above the ground, so it does not show through
DISPLAY_FOOTWAY_LIFT_M = 0.2
DISPLAY_OUTLINE_WIDTH_M = 3.0  # width of the band that marks the edge of the simulated area

#: Colours as seen on screen (sRGB, RGBA 0-255). glTF stores linear values; see srgb_to_linear.
#: Buildings are coloured by what their height is based on, so the picture shows it.
BUILDING_COLOURS = {
    "measured": (196, 214, 190, 255),
    "mapped": (205, 202, 196, 255),
    "levels": (176, 186, 199, 255),
    "ml": (198, 182, 216, 255),
    "estimated": (214, 178, 150, 255),
}
ROAD_COLOUR = (88, 88, 92, 255)
FOOTWAY_COLOUR = (150, 150, 146, 255)
GROUND_COLOUR = (120, 132, 110, 255)
CROWN_COLOUR = (62, 122, 66, 255)
TRUNK_COLOUR = (96, 72, 52, 255)
OUTLINE_COLOUR = (255, 176, 0, 255)

Z_UP_TO_Y_UP = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
    dtype=float,
)


def height_basis(b: Building) -> str:
    """The building's kind of height (heights.HEIGHT_KINDS), which picks its colour."""
    return height_kind(b.height_source, b.height_dataset) or "missing"


def building_mesh(b: Building) -> trimesh.Trimesh | None:
    """A block from the building's base to its top, standing on its ground."""
    if b.height is None or b.height <= b.base:
        return None
    mesh = trimesh.creation.extrude_polygon(b.footprint, b.height - b.base, engine="earcut")
    mesh.apply_translation((0.0, 0.0, (b.ground or 0.0) + b.base))
    return mesh


def _draped(poly: Polygon, terrain: Terrain, lift: float) -> trimesh.Trimesh | None:
    """A flat shape laid on the ground: its outline is sampled every quarter grid step, so the
    triangles follow the slopes."""
    if poly.is_empty or poly.area <= 0:
        return None
    dense = shapely.segmentize(poly, terrain.dx / 4)
    verts2, faces = trimesh.creation.triangulate_polygon(dense, engine="earcut")
    z = terrain.at(verts2[:, 0], verts2[:, 1]) + lift
    return trimesh.Trimesh(np.column_stack([verts2, z]), faces, process=False)


def _parts(shape) -> list[Polygon]:
    return [shape] if isinstance(shape, Polygon) else list(getattr(shape, "geoms", []))


def road_mesh(
    r: Road, terrain: Terrain, clip: Polygon | None = None, lift: float = DISPLAY_ROAD_LIFT_M
) -> trimesh.Trimesh | None:
    """The road's surface on the ground, cut to `clip` (the site's square) if given.

    Roads are kept whole in the site model, and some run far beyond the site; drawn whole they
    would stretch the picture far past the area that is simulated.
    """
    width = r.width or (r.lanes * DISPLAY_LANE_WIDTH_M if r.lanes else 2 * DISPLAY_LANE_WIDTH_M)
    if r.highway in ("footway", "pedestrian", "path", "steps", "cycleway", "corridor"):
        width = r.width or DISPLAY_FOOTWAY_WIDTH_M
    shape = r.line.buffer(width / 2, cap_style="flat", join_style="round")
    if clip is not None:
        shape = shape.intersection(clip)
    meshes = [m for p in _parts(shape) if (m := _draped(p, terrain, lift)) is not None]
    return trimesh.util.concatenate(meshes) if meshes else None


def tree_meshes(t: Tree) -> tuple[trimesh.Trimesh, trimesh.Trimesh] | None:
    """Trunk and crown of a tree with a known height; None if the height is unknown."""
    if t.height is None or t.height <= 0:
        return None
    ground = t.ground or 0.0
    diameter = t.crown_diameter or DISPLAY_CROWN_SHARE * t.height
    crown_height = min(t.height * DISPLAY_CROWN_DEPTH_SHARE, diameter)
    base = t.height - crown_height
    radius = (
        t.trunk_circumference / (2 * math.pi) if t.trunk_circumference else DISPLAY_TRUNK_RADIUS_M
    )
    trunk_top = base + DISPLAY_TRUNK_INTO_CROWN * crown_height
    trunk = trimesh.creation.cylinder(radius=radius, height=trunk_top, sections=10)
    trunk.apply_translation((t.x, t.y, ground + trunk_top / 2))
    crown = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    crown.apply_scale((diameter / 2, diameter / 2, crown_height / 2))
    crown.apply_translation((t.x, t.y, ground + base + crown_height / 2))
    return trunk, crown


def ground_mesh(terrain: Terrain) -> trimesh.Trimesh:
    """The ground surface: the terrain's grid points joined by its triangles."""
    xs, ys = terrain.nodes()
    xx, yy = np.meshgrid(xs, ys)
    verts = np.column_stack([xx.ravel(), yy.ravel(), terrain.elevation.ravel().astype(float)])
    return trimesh.Trimesh(verts, terrain.faces(), process=False)


def outline_mesh(site: Site, terrain: Terrain) -> trimesh.Trimesh | None:
    """A band on the ground just inside the edge of the site: the area that is simulated."""
    square = shapely.box(*site.extent())
    band = square.difference(square.buffer(-DISPLAY_OUTLINE_WIDTH_M, join_style="mitre"))
    return _draped(band, terrain, DISPLAY_FOOTWAY_LIFT_M + 0.05)


def srgb_to_linear(rgba: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    """A colour as seen on screen (sRGB, 0-255) as the linear values (0-1) that glTF stores.

    glTF's base colour factor is linear, and viewers convert it to sRGB for the screen, so an
    sRGB colour stored unconverted would look washed out. This is the inverse of the sRGB
    encoding C' = 12.92 C for C <= 0.0031308, else 1.055 C^(1/2.4) - 0.055 (ICC, "sRGB",
    https://www.color.org/chardata/rgb/srgb.xalter, accessed 2026-10-23). Alpha stays linear.
    """
    knee = 12.92 * 0.0031308

    def channel(c: int) -> float:
        v = c / 255
        return v / 12.92 if v <= knee else ((v + 0.055) / 1.055) ** 2.4

    r, g, b, a = rgba
    return channel(r), channel(g), channel(b), a / 255


def _with_material(mesh: trimesh.Trimesh, name: str, rgba: tuple[int, ...]) -> trimesh.Trimesh:
    # One named material per group (not per-face colours, which glTF does not have): smaller
    # files, and a viewer can restyle a whole group by its material name.
    material = trimesh.visual.material.PBRMaterial(
        name=name,
        baseColorFactor=srgb_to_linear(rgba),
        metallicFactor=0.0,
        roughnessFactor=0.9,
    )
    mesh.visual = trimesh.visual.TextureVisuals(material=material)
    return mesh


def scene(site: Site, terrain: Terrain | None = None) -> trimesh.Scene:
    """One named mesh per kind of feature, in glTF's y-up axes.

    Each group's features are merged into one mesh, so that a large city is a handful of draw
    calls instead of thousands. To still know which feature a triangle belongs to, every vertex
    carries a `_FEATURE_ID` attribute (glTF's form for an application's own per-vertex data),
    an index into the list of feature ids stored in the mesh's `extras` as `feature_ids`. The
    extras also hold the group's name as `group`.
    """
    groups: dict[str, list[tuple[str, trimesh.Trimesh]]] = {}

    def add(name: str, fid: str, mesh: trimesh.Trimesh | None) -> None:
        if mesh is not None and len(mesh.faces):
            groups.setdefault(name, []).append((fid, mesh))

    terrain = terrain or flat(site.extent())
    add("ground", "ground", ground_mesh(terrain))
    add("site_outline", "site_outline", outline_mesh(site, terrain))
    for b in site.buildings:
        add(f"buildings_{height_basis(b)}", b.id, building_mesh(b))
    square = shapely.box(*site.extent())
    for r in site.roads:
        add("roads", r.id, road_mesh(r, terrain, clip=square))
    for r in site.footways:
        add("footways", r.id, road_mesh(r, terrain, clip=square, lift=DISPLAY_FOOTWAY_LIFT_M))
    for t in site.trees:
        if (parts := tree_meshes(t)) is not None:
            add("tree_trunks", t.id, parts[0])
            add("tree_crowns", t.id, parts[1])

    colours = {
        "ground": GROUND_COLOUR,
        "site_outline": OUTLINE_COLOUR,
        "roads": ROAD_COLOUR,
        "footways": FOOTWAY_COLOUR,
        "tree_trunks": TRUNK_COLOUR,
        "tree_crowns": CROWN_COLOUR,
        **{f"buildings_{k}": v for k, v in BUILDING_COLOURS.items()},
    }
    out = trimesh.Scene()
    for name, items in groups.items():
        for k, (_, m) in enumerate(items):
            # float32 holds whole numbers exactly up to 2**24 (16.7 million features). The glTF
            # specification advises against 32-bit unsigned integers for vertex attributes, and
            # 16-bit ones would limit a group to 65 536 features.
            m.vertex_attributes["_FEATURE_ID"] = np.full(len(m.vertices), k, dtype=np.float32)
        mesh = trimesh.util.concatenate([m for _, m in items])
        mesh.metadata = {"group": name, "feature_ids": [fid for fid, _ in items]}
        mesh.apply_transform(Z_UP_TO_Y_UP)
        out.add_geometry(_with_material(mesh, name, colours[name]), node_name=name, geom_name=name)
    return out


def export_glb(site: Site, path: Path, terrain: Terrain | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(scene(site, terrain).export(file_type="glb"))
    return path
