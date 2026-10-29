"""Reading the cube's wind-tunnel profiles (sources/ercoftac.py), from made-up files in the same
layout: the real files may not be copied into the repository (their licence, docs/sources.md)."""

import zipfile

import numpy as np
import pytest

from aircanopy.sources import cache, ercoftac

# The layout of a file of kind "a"; the numbers are invented.
KIND_A = """#@# filename = Data_a/x05z09_a.dat
#@#--------------------------------------
#         -7.00      .00        3      120
#          x/H      z/H      lines    tabletype
#-------------------------------------------------------------------------------
#                |       normalized ( see readme-file )               |   in (m2/s2)
#                |                                                    |   ___      ____
#     x       z  |   y        U     u'^2        V     v'^2      u'v'  |   u^2      v^2
#===============================================================================
## 3
    -7.00    .00   .050     .700  .010000    -.010  .002000 -.002000    4.000     .600
    -7.00    .00   .500    1.000  .005000    -.005  .001500 -.001000    2.000     .500
    -7.00    .00  1.000    1.100  .003000     .000  .001000  .000000    1.000     .400
"""

# The layout of a file of kind "d", with fewer spaces; the numbers are invented.
KIND_D = """#@# filename = Data_d/x05z09_d.dat
#@#--------------------------------------
#         -7.00      .00        2      131
#         x/H       z/H      lines    tabletype
#------------------------------------------------------------------------------------------------
#                           normalized ( see readme-file )
#    x      z     y      u3     u2w     uw2      w3      u4     u3w    u2w2     uw3      w4
#================================================================================================
## 2
   -7.00   .00  .100  -.1000   .0500   .0100   .0100  2.6000  -.0500   .9000   .0100  3.2000
   -7.00   .00  .200  -.2000  -.0200  -.1000  -.1000  2.7000   .0800   .9500   .2000  3.4000
"""


def test_a_profile_is_read_with_its_station_and_columns():
    p = ercoftac.parse(KIND_A, "a")
    assert (p.kind, p.x, p.z) == ("a", -7.0, 0.0)
    assert p.columns == ("x", "z", "y", "U", "u'^2", "V", "v'^2", "u'v'", "u^2", "v^2")
    assert np.allclose(p["y"], [0.05, 0.5, 1.0]) and np.allclose(p["U"], [0.7, 1.0, 1.1])
    assert p["u'v'"][0] == pytest.approx(-0.002)
    d = ercoftac.parse(KIND_D, "d")
    assert d.columns[3:] == ("u3", "u2w", "uw2", "w3", "u4", "u3w", "u2w2", "uw3", "w4")
    assert d.values.shape == (2, 12)


def test_a_file_with_missing_rows_is_refused():
    short = "\n".join(KIND_A.splitlines()[:-1])
    with pytest.raises(ValueError, match="expected 3 rows"):
        ercoftac.parse(short, "a")


def test_profiles_are_read_from_the_zip_and_the_nearest_station_found(tmp_path):
    path = tmp_path / "smcb.zip"
    later = KIND_A.replace("-7.00", "-3.00")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("data_a/x05z09_a.dat", KIND_A)
        z.writestr("data_a/x09z09_a.dat", later)
        z.writestr("data_d/x05z09_d.dat", KIND_D)
        z.writestr("readme.txt", "not a profile")
    found = ercoftac.profiles("a", path)
    assert [p.x for p in found] == [-7.0, -3.0]
    assert ercoftac.station(-3.4, 0.1, "a", path).x == -3.0
    assert len(ercoftac.profiles("d", path)) == 1
    with pytest.raises(ValueError, match="unknown kind"):
        ercoftac.profiles("e", path)


def test_a_download_that_does_not_match_the_recorded_checksum_is_refused(tmp_path, monkeypatch):
    fake = cache.Download(path=tmp_path / "smcb.zip", url=ercoftac.URL, accessed="",
                          bytes=0, sha256="0" * 64, licence="", attribution="")  # fmt: skip
    monkeypatch.setattr(cache, "fetch", lambda *a, **k: fake)
    with pytest.raises(ValueError, match="SHA-256"):
        ercoftac.fetch()
