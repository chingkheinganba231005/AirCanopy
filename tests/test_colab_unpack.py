"""The checks on a zip of results that comes back from Colab or a rented graphics card."""

import hashlib
import importlib.util
import json
import os
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


def test_a_zip_from_gpu_pack_passes(tmp_path):
    """scripts/gpu_pack.py, used on rented machines, makes the zip the notebook makes."""
    spec = importlib.util.spec_from_file_location("gpu_pack", ROOT / "scripts" / "gpu_pack.py")
    gpu_pack = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gpu_pack)
    (tmp_path / "results").mkdir()
    old = tmp_path / "results" / "old.json"
    old.write_text("{}")
    os.utime(old, (1000.0, 1000.0))  # written before the run started
    (tmp_path / "results" / "wind_speed_gpu.json").write_text('{"runs": []}')
    names = gpu_pack.written_since(tmp_path, since=2000.0)
    assert names == ["results/wind_speed_gpu.json"]
    gpu_pack.pack(tmp_path, names, tmp_path / "out.zip", "0" * 40, "Fake Card, 24564 MiB")
    manifest, files = unpack.check(tmp_path / "out.zip")
    assert set(files) == {"results/wind_speed_gpu.json"} and manifest["gpu"].startswith("Fake")
