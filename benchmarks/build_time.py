"""How long a site takes to build, and how much it downloads.

The site is built twice, each time by the `aircanopy` command in a new process, as a user
would run it:

1. *cold*: with a new, empty cache folder, so every download is made again (the geocoder's
   answer, OpenStreetMap, Overture Maps, the elevation and canopy windows). Its time depends on
   the network and on the services' load at that moment.
2. *warm*: again with the same cache folder, so the time is the processing alone.

The site folders and the cache are made in a temporary folder and deleted afterwards.

Needs the network. Writes results/build_time_<name>.json.

    uv run python benchmarks/build_time.py mongkok --place "Mong Kok" --pick 2 --size 1000
    uv run python benchmarks/build_time.py hku --around 114.1370,22.2840,500
"""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from aircanopy import __version__

ROOT = Path(__file__).resolve().parents[1]


def folder_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def timed_build(command: list[str], cache: Path) -> float:
    env = {**os.environ, "AIRCANOPY_CACHE": str(cache)}
    start = time.perf_counter()
    subprocess.run(command, env=env, check=True, stdout=subprocess.DEVNULL)
    return time.perf_counter() - start


def main(name: str, build_args: list[str]) -> None:
    exe = shutil.which("aircanopy", path=str(Path(sys.executable).parent))
    if exe is None:
        raise SystemExit("the aircanopy command is not installed in this environment")
    work = Path(tempfile.mkdtemp(prefix="aircanopy-build-time-"))
    try:
        cache = work / "cache"
        runs = {}
        for run in ("cold", "warm"):
            out = work / run
            command = [exe, "site", "build", *build_args, "--name", name, "--out", str(out)]
            runs[run] = {"seconds": round(timed_build(command, cache), 1)}
            if run == "cold":
                runs[run]["downloaded_bytes"] = folder_bytes(cache)
        summary = json.loads((work / "warm" / "summary.json").read_text())
        files = {f.name: f.stat().st_size for f in sorted((work / "warm").iterdir()) if f.is_file()}
    finally:
        shutil.rmtree(work)
    result = {
        "what": "time to build a site with nothing cached (cold) and from the cache (warm)",
        "run_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "aircanopy": __version__,
        "machine": {
            "cpus": os.cpu_count(),
            "system": platform.system(),
            "python": platform.python_version(),
        },
        "build_args": build_args,
        "runs": runs,
        "site": {
            "size_m": summary["size_m"],
            "buildings": summary["buildings"],
            "trees": summary["trees"],
            "voxels_shape_zyx": summary["voxels"]["shape_zyx"],
        },
        "file_bytes": files,
    }
    path = ROOT / "results" / f"build_time_{name}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
