"""A local cache of downloads, each with a record of where it came from.

Files live in the cache folder (AIRCANOPY_CACHE, or ~/.cache/aircanopy). Next to each file a
`<name>.source.json` records the URL, the access time (UTC), the size, the SHA-256 checksum, the
licence and the attribution. A cached file is used again only if it still matches its checksum.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from .. import __version__

#: Sent with every request: services ask to be told who is calling and how to reach them.
USER_AGENT = f"AirCanopy/{__version__} (+https://github.com/chingkheinganba231005/aircanopy)"


@dataclass(frozen=True)
class Download:
    path: Path
    url: str
    accessed: str
    bytes: int
    sha256: str
    licence: str
    attribution: str

    def record(self) -> dict:
        """What a result should store about this download."""
        d = asdict(self)
        d["path"] = self.path.name
        return d


def cache_dir() -> Path:
    root = Path(os.environ.get("AIRCANOPY_CACHE", Path.home() / ".cache" / "aircanopy"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def _meta_path(path: Path) -> Path:
    return path.with_name(path.name + ".source.json")


def cached(name: str) -> Download | None:
    """The cached download called `name`, if it exists and is intact."""
    path = cache_dir() / name
    meta = _meta_path(path)
    if not (path.is_file() and meta.is_file()):
        return None
    d = json.loads(meta.read_text())
    if sha256_file(path) != d["sha256"]:
        return None
    return Download(path=path, **{k: v for k, v in d.items() if k != "path"})


def fetch(
    url: str,
    name: str,
    *,
    licence: str,
    attribution: str,
    refresh: bool = False,
    timeout: float = 120.0,
) -> Download:
    """Download `url` into the cache as `name` (or reuse an intact cached copy)."""
    if not refresh and (hit := cached(name)) is not None:
        return hit
    path = cache_dir() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = path.with_name(path.name + ".part")
    with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open("wb") as out:
        while block := response.read(1 << 20):
            out.write(block)
    tmp.replace(path)
    download = Download(
        path=path,
        url=url,
        accessed=dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        bytes=path.stat().st_size,
        sha256=sha256_file(path),
        licence=licence,
        attribution=attribution,
    )
    _meta_path(path).write_text(json.dumps(download.record(), indent=2) + "\n")
    return download
