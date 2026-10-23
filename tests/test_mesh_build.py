"""Meshes, the site folder and the command line."""

import json
from pathlib import Path

import numpy as np
import pytest
import shapely
import trimesh

from aircanopy import cli
from aircanopy.geo import BBox
from aircanopy.site import build
from aircanopy.site.fill import NEIGHBOUR_MEDIAN, fill_building_heights
from aircanopy.site.from_osm import site_from_osm
from aircanopy.site.mesh import ROAD_COLOUR, building_mesh, export_glb, scene, tree_meshes
from aircanopy.site.model import Site
from aircanopy.site.voxel import block_volume, grid_for_site, load_voxels, voxelise_buildings
from aircanopy.sources import osm

from .site_helpers import building, empty_site, road, square, tree

FIXTURE = Path(__file__).parent / "data" / "osm" / "hku_small.osm.gz"
FIXTURE_BBOX = BBox(114.1360, 22.2832, 114.1380, 22.2848)


def small_site() -> Site:
    site = empty_site()
    site.buildings = [
        building("m", square(-50, -50, 20), 10.0),
        building("l", square(10, 10, 20), 12.0),
        building("e", square(40, -40, 10), 7.0),
    ]
    site.buildings[1].height_source = "osm:levels"
    site.buildings[2].height_source = NEIGHBOUR_MEDIAN
    site.trees = [tree(0, -20, 9.0, 6.0), tree(5, -20, None)]
    site.roads = [road([(-80, 0), (80, 0)], "primary", width=14.0)]
    site.footways = [road([(-80, -5), (80, -5)], "footway")]
    return site


def test_building_mesh_is_a_closed_block_at_its_base():
    b = building("p", square(0, 0, 10), 25.0, base=5.0)
    mesh = building_mesh(b)
    assert mesh.is_watertight
    assert mesh.volume == pytest.approx(10 * 10 * 20)
    assert mesh.bounds[0][2] == pytest.approx(5.0) and mesh.bounds[1][2] == pytest.approx(25.0)


def test_trees_without_height_are_not_drawn():
    assert tree_meshes(tree(0, 0, None)) is None
    _, crown = tree_meshes(tree(0, 0, 9.0, 6.0))
    assert crown.bounds[1][2] == pytest.approx(9.0, abs=0.05)
    assert crown.extents[0] == pytest.approx(6.0, abs=0.1)


def test_scene_groups_and_y_up_axes():
    sc = scene(small_site())
    assert set(sc.geometry) == {
        "ground",
        "buildings_measured",
        "buildings_levels",
        "buildings_estimated",
        "roads",
        "footways",
        "tree_trunks",
        "tree_crowns",
    }
    # glTF is y-up: building heights end up on the y axis, and north (+y in the site) on -z.
    measured = sc.geometry["buildings_measured"]
    assert measured.bounds[1][1] == pytest.approx(10.0)
    assert measured.bounds[0][2] == pytest.approx(30.0)  # site y from -50 to -30 -> z 30..50
    assert measured.bounds[1][2] == pytest.approx(50.0)


def test_glb_loads_back(tmp_path):
    path = export_glb(small_site(), tmp_path / "scene.glb")
    loaded = trimesh.load(path)
    assert isinstance(loaded, trimesh.Scene)
    assert "buildings_estimated" in loaded.geometry
    material = loaded.geometry["roads"].visual.material
    assert material.name == "roads"
    assert tuple(material.baseColorFactor) == ROAD_COLOUR


def test_build_site_folder_from_real_fixture(tmp_path):
    data = osm.parse(FIXTURE)
    summary = build.build_from_osm_data("hku_small", FIXTURE_BBOX, data, tmp_path, 2.0, [])
    for name in ("site.json", "scene.glb", "voxels.npz", "summary.json"):
        assert (tmp_path / name).is_file()
    site = Site.load(tmp_path / "site.json")
    assert summary["buildings"] == len(site.buildings) > 0
    assert sum(summary["building_heights"].values()) == summary["buildings"]
    grid, solid = load_voxels(tmp_path / "voxels.npz")
    assert solid.shape == grid.shape
    assert summary["voxels"]["building_cells"] == int(solid.sum()) > 0
    assert json.loads((tmp_path / "summary.json").read_text()) == summary


@pytest.mark.parametrize(("dx", "tolerance"), [(0.5, 0.005), (2.0, 0.02)])
def test_voxel_volume_converges_to_the_exact_volume(dx, tolerance):
    site = site_from_osm("hku_small", FIXTURE_BBOX, osm.parse(FIXTURE))
    fill_building_heights(site)
    grid = grid_for_site(site, dx)
    # Only buildings wholly inside the grid, so none is cut by the edge.
    box = shapely.box(grid.x0, grid.y0, grid.x0 + grid.nx * dx, grid.y0 + grid.ny * dx)
    site.buildings = [
        b for b in site.buildings if b.height is not None and box.contains(b.footprint)
    ]
    assert len(site.buildings) > 10
    exact = block_volume(site.buildings)
    voxels = voxelise_buildings(site, grid).sum() * dx**3
    assert voxels == pytest.approx(exact, rel=tolerance)


def test_cli_parses_both_ways_of_giving_an_area():
    parser = cli.build_parser()
    a = parser.parse_args(["site", "build", "--around", "114.137,22.284,500", "--name", "hku"])
    box = cli._box(a)
    w, h = box.size_m()
    assert w == pytest.approx(500, rel=1e-3) and h == pytest.approx(500, rel=1e-3)
    b = parser.parse_args(["site", "build", "--bbox", "114.13,22.28,114.14,22.29", "--name", "x"])
    assert cli._box(b).as_tuple() == (114.13, 22.28, 114.14, 22.29)
    with pytest.raises(SystemExit):
        parser.parse_args(["site", "build", "--name", "x"])


def test_cli_site_build_writes_the_folder(tmp_path, monkeypatch, capsys):
    def fake_build_site(name, bbox, out, dx=2.0, refresh=False):
        return build.build_from_osm_data(name, bbox, osm.parse(FIXTURE), out, dx, [])

    monkeypatch.setattr(build, "build_site", fake_build_site)
    out = tmp_path / "site"
    code = cli.main(
        [
            "site",
            "build",
            "--bbox",
            "114.1360,22.2832,114.1380,22.2848",
            "--name",
            "t",
            "--out",
            str(out),
            "--dx",
            "4",
        ]
    )
    assert code == 0
    assert (out / "voxels.npz").is_file()
    printed = capsys.readouterr().out
    assert '"dx_m": 4.0' in printed
    assert np.load(out / "voxels.npz")["dx"].item() == 4.0
