"""The `aircanopy` command.

aircanopy site build --around 114.1370,22.2840,500 --name hku
aircanopy site build --bbox 114.1335,22.2820,114.1395,22.2860 --name hku --dx 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .geo import BBox


def _box(args: argparse.Namespace) -> BBox:
    if args.bbox:
        return BBox.parse(args.bbox)
    lon, lat, size = (float(v) for v in args.around.split(","))
    return BBox.around(lon, lat, size)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aircanopy",
        description="Simulate how trees change street-level heat and air pollution, in 3D.",
    )
    parser.add_argument("--version", action="version", version=f"aircanopy {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="command")

    site = sub.add_parser("site", help="build a 3D site from open map data")
    site_sub = site.add_subparsers(dest="site_command", metavar="site_command", required=True)
    build = site_sub.add_parser("build", help="download the data of an area and build its site")
    where = build.add_mutually_exclusive_group(required=True)
    where.add_argument("--bbox", help="west,south,east,north in degrees")
    where.add_argument("--around", help="lon,lat,size: a square of `size` metres around a point")
    build.add_argument("--name", required=True, help="a short name for the site")
    build.add_argument("--out", type=Path, help="output folder (default: sites/NAME)")
    build.add_argument("--dx", type=float, default=2.0, help="voxel size in metres (default 2)")
    build.add_argument("--refresh", action="store_true", help="download again even if cached")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "site" and args.site_command == "build":
        from .site.build import build_site

        out = args.out or Path("sites") / args.name
        summary = build_site(args.name, _box(args), out, dx=args.dx, refresh=args.refresh)
        print(json.dumps({k: v for k, v in summary.items() if k != "sources"}, indent=2))
        print(f"wrote {out}/site.json, scene.glb, voxels.npz, summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
