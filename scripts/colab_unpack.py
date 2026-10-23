"""Check a zip of results that came back from a Colab run, then copy its files into results/.

    uv run python scripts/colab_unpack.py aircanopy_gpu_results.zip

The zip is made by the last cell of colab/wind_gpu.ipynb: result files under results/, and a
manifest.json with the commit that ran, the graphics card, the time it was packed, and the
SHA-256 of each file. The zip came from outside, so nothing in it is trusted until checked:

- every entry is either manifest.json or a .json file directly under results/ (no other paths);
- the files in the zip and in the manifest are the same, and each matches its SHA-256;
- each file parses as JSON;
- the commit in the manifest exists in this repository.

Only then are the files written to results/, and a summary is printed.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


class Rejected(Exception):
    pass


def check(zip_path: Path) -> tuple[dict, dict[str, bytes]]:
    """The manifest and the checked files, by name; raises Rejected with the reason."""
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        if "manifest.json" not in names:
            raise Rejected("no manifest.json")
        manifest = json.loads(z.read("manifest.json"))
        files = {}
        for name in names:
            if name == "manifest.json":
                continue
            p = PurePosixPath(name)
            if p.parent != PurePosixPath("results") or p.suffix != ".json" or ".." in p.parts:
                raise Rejected(f"unexpected entry {name!r}")
            files[name] = z.read(name)
    listed = manifest.get("files", {})
    if set(listed) != set(files):
        raise Rejected(f"manifest lists {sorted(listed)}, zip holds {sorted(files)}")
    for name, data in files.items():
        if hashlib.sha256(data).hexdigest() != listed[name]:
            raise Rejected(f"{name} does not match its SHA-256 in the manifest")
        try:
            json.loads(data)
        except json.JSONDecodeError as e:
            raise Rejected(f"{name} is not JSON: {e}") from e
    return manifest, files


def known_commit(commit: str) -> bool:
    out = subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=ROOT,
                         capture_output=True)  # fmt: skip
    return out.returncode == 0


def main(zip_path: Path) -> int:
    try:
        manifest, files = check(zip_path)
        if not known_commit(manifest.get("commit", "")):
            raise Rejected(f"commit {manifest.get('commit')!r} is not in this repository")
    except (Rejected, zipfile.BadZipFile, json.JSONDecodeError) as e:
        print(f"rejected: {e}")
        return 1
    for name, data in files.items():
        (ROOT / name).write_bytes(data)
        print(f"wrote {name} ({len(data)} bytes)")
    print(f"commit {manifest['commit']}, {manifest.get('gpu')}, packed {manifest.get('packed_at')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
