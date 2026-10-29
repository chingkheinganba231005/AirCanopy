"""Wind-tunnel measurements around a cube on the floor of a channel, for checking the wind solver.

Martinuzzi and Tropea (1993), from the ERCOFTAC Classic Collection, case 041: a cube of height
H = 25 mm on the floor of a channel 2H high and 24H wide, in fully developed flow, at a Reynolds
number of about 80 000 on the channel height and the bulk velocity Ub. Laser-Doppler profiles
across the channel's height, at stations upstream of the cube, around it and behind it.

The files are licensed CC BY-NC-SA 4.0 (the footer of the wiki that hosts them;
docs/sources.md), so they are downloaded when needed into the cache, checked against their
SHA-256, and never copied into the repository.

In the zip, each profile is one file, `data_<k>/x<nn>z<mm>_<k>.dat`. The kind k is "a" (mean
velocities and Reynolds stresses from u-v measurements), "b" (third and fourth moments, u-v),
"c" (mean velocities and stresses, u-w) or "d" (moments, u-w). Lengths are in H: x along the
flow from the cube's front face, y up from the floor, z across from the symmetry plane;
velocities are in Ub and stresses in Ub^2, except the last two columns of kinds a and c, which
are in m^2/s^2 (readme.txt in the zip).
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import cache

URL = "http://cfd.mace.manchester.ac.uk/ercoftac/lib/exe/fetch.php?media=cdata:case041:smcb-allfiles.zip"
SHA256 = "db9527bfbb7869bb306e3ca6e122ff1555f9ff1f55555ee0ebd8d163cf0501f0"
LICENCE = "CC BY-NC-SA 4.0 (footer of the ERCOFTAC wiki, cfd.mace.manchester.ac.uk/ercoftac)"
ATTRIBUTION = ("R. J. Martinuzzi and C. Tropea, flow around a surface-mounted cube, ERCOFTAC "
               "Classic Collection, case 041")  # fmt: skip
KINDS = ("a", "b", "c", "d")


@dataclass(frozen=True)
class Profile:
    """One measured profile: its station (x/H, z/H) and a table, one column per quantity."""

    kind: str
    x: float
    z: float
    columns: tuple[str, ...]
    values: np.ndarray

    def __getitem__(self, name: str) -> np.ndarray:
        return self.values[:, self.columns.index(name)]


def parse(text: str, kind: str) -> Profile:
    """Read one data file of the given kind (module docstring)."""
    lines = text.splitlines()
    header = [line.lstrip("#").split() for line in lines if line.startswith("#")]
    station = next(h for h in header if len(h) == 4 and _numbers(h))
    x, z, rows = float(station[0]), float(station[1]), int(station[2])
    names = next(words for words in ([w for w in h if w != "|"] for h in header)
                 if words[:3] == ["x", "z", "y"])  # fmt: skip
    data = [[float(v) for v in line.split()] for line in lines
            if line.strip() and not line.startswith("#")]  # fmt: skip
    if len(data) != rows or any(len(row) != len(names) for row in data):
        raise ValueError(f"expected {rows} rows of {len(names)} numbers, found {len(data)}")
    return Profile(kind=kind, x=x, z=z, columns=tuple(names), values=np.array(data))


def _numbers(words: list[str]) -> bool:
    try:
        [float(w) for w in words]
    except ValueError:
        return False
    return True


def fetch() -> Path:
    """The zip of all profiles, downloaded into the cache if needed and checked."""
    download = cache.fetch(URL, "ercoftac/smcb-allfiles.zip", licence=LICENCE,
                           attribution=ATTRIBUTION)  # fmt: skip
    if download.sha256 != SHA256:
        raise ValueError(f"{download.path} does not match the SHA-256 recorded in this module")
    return download.path


def profiles(kind: str = "a", path: Path | None = None) -> list[Profile]:
    """Every profile of one kind, sorted by station (x, then z)."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; choose one of {KINDS}")
    with zipfile.ZipFile(path or fetch()) as z:
        names = [n for n in z.namelist() if n.startswith(f"data_{kind}/") and n.endswith(".dat")]
        out = [parse(z.read(n).decode("latin-1"), kind) for n in names]
    return sorted(out, key=lambda p: (p.x, p.z))


def station(x: float, z: float, kind: str = "a", path: Path | None = None) -> Profile:
    """The profile measured closest to (x, z), in H."""
    return min(profiles(kind, path), key=lambda p: (p.x - x) ** 2 + (p.z - z) ** 2)
