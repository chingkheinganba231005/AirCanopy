"""The checks on a zip of results that comes back from Colab."""

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("colab_unpack", ROOT / "scripts" / "colab_unpack.py")
unpack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unpack)


def make_zip(path: Path, files: dict[str, bytes], listed: dict[str, str] | None = None) -> Path:
    manifest = {
        "commit": "0" * 40,
        "gpu": "test",
        "packed_at": "2026-10-23T00:00:00Z",
        "files": listed
        if listed is not None
        else {n: hashlib.sha256(d).hexdigest() for n, d in files.items()},
    }
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
        z.writestr("manifest.json", json.dumps(manifest))
    return path


def test_a_good_zip_passes(tmp_path):
    files = {"results/wind_speed_gpu.json": b'{"runs": []}'}
    manifest, checked = unpack.check(make_zip(tmp_path / "a.zip", files))
    assert checked == files and manifest["gpu"] == "test"


@pytest.mark.parametrize(
    "files, listed, reason",
    [
        ({"results/../evil.json": b"{}"}, None, "unexpected entry"),
        ({"src/aircanopy/cli.py": b"{}"}, None, "unexpected entry"),
        ({"results/x.json": b"{}"}, {"results/x.json": "0" * 64}, "SHA-256"),
        ({"results/x.json": b"{}"}, {}, "manifest lists"),
        ({"results/x.json": b"not json"}, None, "not JSON"),
    ],
)
def test_a_bad_zip_is_rejected(tmp_path, files, listed, reason):
    with pytest.raises(unpack.Rejected, match=reason):
        unpack.check(make_zip(tmp_path / "b.zip", files, listed))


def test_an_unknown_commit_is_rejected(tmp_path, capsys):
    path = make_zip(tmp_path / "c.zip", {"results/x.json": b"{}"})
    assert unpack.main(path) == 1
    assert "not in this repository" in capsys.readouterr().out
