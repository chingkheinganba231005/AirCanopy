"""The `aircanopy` command. Subcommands are added as each part of the project is built."""

from __future__ import annotations

import argparse
import sys

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aircanopy",
        description="Simulate how trees change street-level heat and air pollution, in 3D.",
    )
    parser.add_argument("--version", action="version", version=f"aircanopy {__version__}")
    parser.add_subparsers(dest="command", metavar="command")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
