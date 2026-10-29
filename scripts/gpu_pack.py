"""Pack the results of a graphics-card run into a zip that scripts/colab_unpack.py accepts.

    python scripts/gpu_pack.py --since EPOCH_SECONDS [--out aircanopy_gpu_results.zip]
    python scripts/gpu_pack.py results/wind_speed_gpu.json ...

Used by scripts/gpu_run.sh on rented machines; the last cell of colab/wind_gpu.ipynb makes
the same zip. With --since, it packs every results/*.json written after that time, so a run
packs what it wrote and nothing older. The manifest holds the commit that ran, the graphics
card, the time it was packed, and the SHA-256 of each file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def output(*command: str, cwd: Path = ROOT) -> str:
    try:
        return subprocess.run(command, capture_output=True, text=True, cwd=cwd).stdout.strip()
    except OSError:
        return ""


def written_since(root: Path, since: float) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in (root / "results").glob("*.json")
                  if p.stat().st_mtime >= since)  # fmt: skip


def pack(root: Path, names: list[str], out: Path, commit: str, gpu: str) -> dict:
    manifest = {
        "commit": commit,
        "gpu": gpu,
        "packed_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in names},
    }
    with zipfile.ZipFile(out, "w") as z:
        for n in names:
            z.write(root / n, n)
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
    return manifest


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="files under results/ to pack")
    parser.add_argument("--since", type=float, help="pack results/*.json written after this")
    parser.add_argument("--out", type=Path, default=ROOT / "aircanopy_gpu_results.zip")
    args = parser.parse_args(argv)
    names = args.names or (written_since(ROOT, args.since) if args.since is not None else [])
    if not names:
        print("nothing to pack")
        return 1
    gpu = output("nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader")  # fmt: skip
    manifest = pack(ROOT, names, args.out, output("git", "rev-parse", "HEAD"), gpu)
    print(json.dumps(manifest, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
