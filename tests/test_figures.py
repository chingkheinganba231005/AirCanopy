"""The pictures for the journal are drawn from a site folder without errors."""

import importlib.util
from pathlib import Path

from PIL import Image

from aircanopy.geo import BBox, LocalFrame
from aircanopy.site import build
from aircanopy.sources import osm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"
BOX = BBox(114.1360, 22.2832, 114.1380, 22.2848)


def load_figures():
    spec = importlib.util.spec_from_file_location("figures", ROOT / "scripts" / "figures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_figures_from_a_sample_site(tmp_path):
    frame = LocalFrame.for_bbox(BOX)
    terrain, t_record, _ = build.load_terrain(
        BOX, terrain_file=DATA / "terrain" / "hku_small_gedtm30.tif"
    )
    canopy, c_record, _ = build.load_canopy(
        BOX,
        frame,
        frame.bbox_extent(BOX),
        canopy_file=DATA / "canopy" / "hku_small_meta_wri_v1.tif",
    )
    site_dir = tmp_path / "site"
    build.build_from_osm_data(
        "sample", BOX, osm.parse(DATA / "osm" / "hku_small.osm.gz"), site_dir, 2.0,
        [t_record, c_record], terrain=terrain, canopy=canopy,
    )  # fmt: skip
    load_figures().main(site_dir, tmp_path / "pictures" / "sample")
    for name in ("voxels", "canopy", "terrain"):
        with Image.open(tmp_path / "pictures" / f"sample_{name}.png") as img:
            assert img.mode == "RGB" and img.width > 100 and img.height > 100
