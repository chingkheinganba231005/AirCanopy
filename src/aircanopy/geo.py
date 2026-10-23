"""Places on the Earth, and the flat local frame in metres that the physics works in.

Maps give positions as longitude and latitude in degrees (WGS84). The physics needs flat
distances in metres. For a site a few kilometres across we lay a flat frame on the Earth at the
site's centre: a transverse Mercator projection whose central meridian passes through the origin.
Its scale error at a distance d from that meridian is about d^2 / (2 R^2) with R the Earth's
radius, so 1.2e-8 at 1 km and 4.9e-7 at 5 km: far below anything the physics can feel.

Axes: x points east, y north, z up, all in metres, origin at the frame's origin on the ground.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import Geod, Transformer

WGS84 = "EPSG:4326"
_GEOD = Geod(ellps="WGS84")

#: Latitudes beyond this are refused: the flat frame and most map data degrade near the poles.
MAX_ABS_LATITUDE = 85.0


@dataclass(frozen=True)
class BBox:
    """A box of longitude and latitude, in degrees: west, south, east, north."""

    west: float
    south: float
    east: float
    north: float

    def __post_init__(self) -> None:
        if not -180.0 <= self.west < self.east <= 180.0:
            raise ValueError(f"need -180 <= west < east <= 180, got {self.west}, {self.east}")
        if not -MAX_ABS_LATITUDE <= self.south < self.north <= MAX_ABS_LATITUDE:
            raise ValueError(
                f"need -{MAX_ABS_LATITUDE} <= south < north <= {MAX_ABS_LATITUDE}, "
                f"got {self.south}, {self.north}"
            )

    @property
    def center(self) -> tuple[float, float]:
        """Longitude and latitude of the middle of the box."""
        return (0.5 * (self.west + self.east), 0.5 * (self.south + self.north))

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)

    def size_m(self) -> tuple[float, float]:
        """Width (west to east, through the centre) and height (south to north) in metres."""
        lon, lat = self.center
        _, _, width = _GEOD.inv(self.west, lat, self.east, lat)
        _, _, height = _GEOD.inv(lon, self.south, lon, self.north)
        return float(width), float(height)

    def area_deg2(self) -> float:
        return (self.east - self.west) * (self.north - self.south)

    @classmethod
    def around(cls, lon: float, lat: float, width_m: float, height_m: float | None = None) -> BBox:
        """The box of the given size in metres centred on a point."""
        height_m = width_m if height_m is None else height_m
        if width_m <= 0 or height_m <= 0:
            raise ValueError("the box needs a positive size")
        east, _, _ = _GEOD.fwd(lon, lat, 90.0, width_m / 2)
        west, _, _ = _GEOD.fwd(lon, lat, 270.0, width_m / 2)
        _, north, _ = _GEOD.fwd(lon, lat, 0.0, height_m / 2)
        _, south, _ = _GEOD.fwd(lon, lat, 180.0, height_m / 2)
        return cls(float(west), float(south), float(east), float(north))

    @classmethod
    def parse(cls, text: str) -> BBox:
        """From "west,south,east,north" in degrees."""
        parts = [float(p) for p in text.replace(" ", "").split(",")]
        if len(parts) != 4:
            raise ValueError("a box needs four numbers: west,south,east,north")
        return cls(*parts)


class LocalFrame:
    """Flat x (east), y (north) in metres around an origin given in longitude and latitude."""

    def __init__(self, lon0: float, lat0: float) -> None:
        if not -MAX_ABS_LATITUDE <= lat0 <= MAX_ABS_LATITUDE:
            raise ValueError(f"origin latitude {lat0} is too close to a pole")
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        self.proj4 = (
            f"+proj=tmerc +lat_0={self.lat0!r} +lon_0={self.lon0!r} +k=1 +x_0=0 +y_0=0 "
            "+ellps=WGS84 +units=m +no_defs"
        )
        self._forward = Transformer.from_crs(WGS84, self.proj4, always_xy=True)
        self._inverse = Transformer.from_crs(self.proj4, WGS84, always_xy=True)

    @classmethod
    def for_bbox(cls, bbox: BBox) -> LocalFrame:
        return cls(*bbox.center)

    def to_local(self, lon, lat) -> tuple[np.ndarray, np.ndarray]:
        x, y = self._forward.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.asarray(x), np.asarray(y)

    def to_lonlat(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        lon, lat = self._inverse.transform(np.asarray(x, float), np.asarray(y, float))
        return np.asarray(lon), np.asarray(lat)

    def bbox_extent(self, bbox: BBox) -> tuple[float, float, float, float]:
        """The smallest box in local metres (xmin, ymin, xmax, ymax) holding the whole bbox."""
        lons = np.array(
            [bbox.west, bbox.east, bbox.east, bbox.west, bbox.center[0], bbox.center[0]]
        )
        lats = np.array([bbox.south, bbox.south, bbox.north, bbox.north, bbox.south, bbox.north])
        x, y = self.to_local(lons, lats)
        return float(x.min()), float(y.min()), float(x.max()), float(y.max())

    def to_dict(self) -> dict:
        return {"lon0": self.lon0, "lat0": self.lat0, "proj4": self.proj4}

    @classmethod
    def from_dict(cls, d: dict) -> LocalFrame:
        return cls(d["lon0"], d["lat0"])

    def __repr__(self) -> str:
        return f"LocalFrame(lon0={self.lon0}, lat0={self.lat0})"
