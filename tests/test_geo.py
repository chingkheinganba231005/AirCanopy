"""Coordinates: boxes in degrees and the flat local frame in metres."""

import numpy as np
import pytest
from pyproj import Geod

from aircanopy.geo import BBox, LocalFrame

GEOD = Geod(ellps="WGS84")


def test_bbox_rejects_impossible_boxes():
    with pytest.raises(ValueError):
        BBox(10, 0, 5, 1)  # west east of east
    with pytest.raises(ValueError):
        BBox(0, 1, 1, 0)  # south north of north
    with pytest.raises(ValueError):
        BBox(0, 86, 1, 87)  # too close to the pole


def test_bbox_parse_and_center():
    b = BBox.parse("114.13, 22.28, 114.14, 22.29")
    assert b.as_tuple() == (114.13, 22.28, 114.14, 22.29)
    assert b.center == pytest.approx((114.135, 22.285))


@pytest.mark.parametrize(("lat", "size"), [(22.28, 500.0), (51.5, 1200.0), (-33.9, 300.0)])
def test_bbox_around_has_the_requested_size(lat, size):
    w, h = BBox.around(114.0, lat, size, size / 2).size_m()
    assert w == pytest.approx(size, rel=1e-3)
    assert h == pytest.approx(size / 2, rel=1e-3)


def test_frame_origin_is_zero_and_round_trips():
    f = LocalFrame(114.1370, 22.2840)
    x, y = f.to_local(114.1370, 22.2840)
    assert abs(float(x)) < 1e-6 and abs(float(y)) < 1e-6
    lons = np.array([114.13, 114.14, 114.1375])
    lats = np.array([22.28, 22.29, 22.2843])
    lon2, lat2 = f.to_lonlat(*f.to_local(lons, lats))
    assert np.allclose(lon2, lons, atol=1e-10) and np.allclose(lat2, lats, atol=1e-10)


def test_frame_distances_match_the_ellipsoid():
    """Over a few kilometres the flat frame keeps distances to better than one part in a million."""
    f = LocalFrame(114.137, 22.284)
    lon_a, lat_a, lon_b, lat_b = 114.120, 22.270, 114.150, 22.300
    _, _, d_true = GEOD.inv(lon_a, lat_a, lon_b, lat_b)
    (xa, xb), (ya, yb) = f.to_local([lon_a, lon_b], [lat_a, lat_b])
    d_flat = np.hypot(xb - xa, yb - ya)
    assert d_flat == pytest.approx(d_true, rel=1e-6)


def test_frame_axes_point_east_and_north():
    f = LocalFrame(0.0, 45.0)
    x_e, y_e = f.to_local(0.001, 45.0)
    x_n, y_n = f.to_local(0.0, 45.001)
    assert float(x_e) > 0 and abs(float(y_e)) < 1e-3
    assert float(y_n) > 0 and abs(float(x_n)) < 1e-9


def test_frame_serialises():
    f = LocalFrame(-73.98, 40.75)
    g = LocalFrame.from_dict(f.to_dict())
    assert (g.lon0, g.lat0) == (f.lon0, f.lat0)
