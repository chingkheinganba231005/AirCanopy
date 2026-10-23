"""Buildings from Overture Maps, for anywhere in the world.

Overture Maps (https://overturemaps.org) joins OpenStreetMap with other open building data:
footprints traced from satellite images by machine learning (Microsoft, Google, and for East
Asia Shi et al. 2023), and heights measured by lidar or estimated by machine learning. Each
building lists the datasets it came from, and which dataset gave which property.

The data are GeoParquet files on Amazon S3. The buildings alone are 512 files, 277 GB in release
2026-10-21.0, and a site needs a few megabytes of that. So we read only those:

1. Overture's STAC catalogue (an index of the files, https://stac.overturemaps.org) gives every
   file's bounding box. Only files whose box overlaps the site are opened.
2. A Parquet file is split into row groups, blocks of about 40 000 buildings sorted by place. Its
   footer stores the smallest and largest value of every column in every row group, so reading
   the footer (under 1 MB) tells which row groups can overlap the site.
3. Only those row groups, and only the columns we use, are downloaded, with HTTP range requests.

Licence: the buildings theme is ODbL-1.0, and some datasets in it also ask for their own credit
(https://docs.overturemaps.org/attribution/, accessed 2026-10-23); see DATASET_CREDITS.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..geo import BBox
from .cache import USER_AGENT, Download, _meta_path, cache_dir, cached, sha256_file

STAC_CATALOG = "https://stac.overturemaps.org/catalog.json"
LICENCE = "ODbL-1.0"
#: The credit line Overture asks for when its data include OpenStreetMap (attribution page).
ATTRIBUTION = "© OpenStreetMap contributors, Overture Maps Foundation"

#: Columns read from the building files (https://docs.overturemaps.org/schema/).
COLUMNS = [
    "id",
    "sources",
    "height",
    "min_height",
    "num_floors",
    "min_floor",
    "is_underground",
    "has_parts",
    "subtype",
    "class",
    "geometry",
    "bbox",
]

#: Credit and licence of the datasets found in the buildings theme, keyed by the name Overture
#: uses in a building's `sources`. Credits are from https://docs.overturemaps.org/attribution/
#: (accessed 2026-10-23); the keys are the names seen in the data (release 2026-10-21.0).
DATASET_CREDITS: dict[str, tuple[str, str]] = {
    "OpenStreetMap": ("© OpenStreetMap contributors", "ODbL-1.0"),
    "Microsoft ML Buildings": (
        "Global ML Building Footprints, licensed by Microsoft",
        "ODbL-1.0",
    ),
    "Google Open Buildings": ("Google Open Buildings", "CC-BY-4.0"),
    "USGS Lidar": ("USGS 3D Elevation Program", "no licence stated by Overture"),
    "doi:10.5281/zenodo.8174931": (
        "Qian Shi et al., A First High-quality Vector Data of Buildings in East Asian Countries "
        "Based on a Comprehensive Large-scale Mapping Framework, Zenodo, 22 July 2023, "
        "doi:10.5281/zenodo.8174931",
        "CC-BY-4.0",
    ),
}


def credit(dataset: str) -> tuple[str, str]:
    """(credit, licence) for a dataset; unknown ones point to Overture's attribution page."""
    return DATASET_CREDITS.get(dataset, (dataset, "see https://docs.overturemaps.org/attribution/"))


# --- reading files over HTTP ---------------------------------------------------------------


def _open(request: urllib.request.Request, timeout: float, attempts: int = 4) -> bytes:
    """The response body, retrying network errors: S3 occasionally drops a connection."""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def get_json(url: str, timeout: float = 60.0) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return json.loads(_open(request, timeout))


class RangeFile(io.RawIOBase):
    """A remote file read in pieces with HTTP range requests, for pyarrow to open like a file."""

    def __init__(self, url: str, timeout: float = 120.0) -> None:
        self.url = url
        self.timeout = timeout
        head = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(head, timeout=timeout) as r:
            self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self.requests = 0
        self.received = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        request = urllib.request.Request(
            self.url,
            headers={"User-Agent": USER_AGENT, "Range": f"bytes={self.pos}-{self.pos + n - 1}"},
        )
        data = _open(request, self.timeout)
        self.requests += 1
        self.received += len(data)
        self.pos += len(data)
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


# --- finding what to read ------------------------------------------------------------------


def _child(catalog: dict, title: str) -> str:
    for link in catalog["links"]:
        if link.get("rel") == "child" and link.get("title") == title:
            return link["href"]
    raise KeyError(f"no child {title!r} in the catalogue")


def building_collection(release: str | None = None) -> tuple[str, dict]:
    """(release, STAC collection of building files), following the catalogue's links."""
    root = get_json(STAC_CATALOG)
    release = release or root["latest"]
    rel = get_json(_child(root, f"{release} Overture Release"))
    theme = get_json(_child(rel, "buildings"))
    return release, get_json(_child(theme, "building"))


def files_for_bbox(collection: dict, bbox: BBox) -> list[str]:
    """STAC item links of the files whose bounding box overlaps `bbox`.

    The collection's extent lists the overall box first, then one box per file, in the same
    order as its item links.
    """
    boxes = collection["extent"]["spatial"]["bbox"][1:]
    items = [link["href"] for link in collection["links"] if link.get("rel") == "item"]
    if len(boxes) != len(items):
        raise ValueError(f"{len(boxes)} file boxes but {len(items)} items in the collection")
    return [
        href
        for href, (w, s, e, n) in zip(items, boxes, strict=True)
        if w <= bbox.east and e >= bbox.west and s <= bbox.north and n >= bbox.south
    ]


def row_groups_for_bbox(meta: pq.FileMetaData, bbox: BBox) -> list[int]:
    """Row groups whose bounding-box statistics overlap `bbox` (or that have no statistics)."""
    index = {meta.schema.column(i).path: i for i in range(meta.num_columns)}
    columns = [index[f"bbox.{k}"] for k in ("xmin", "xmax", "ymin", "ymax")]
    out = []
    for g in range(meta.num_row_groups):
        stats = [meta.row_group(g).column(c).statistics for c in columns]
        if any(s is None or not s.has_min_max for s in stats):
            out.append(g)  # cannot rule it out
            continue
        xmin, xmax, ymin, ymax = stats
        if (
            xmin.min <= bbox.east
            and xmax.max >= bbox.west
            and ymin.min <= bbox.north
            and ymax.max >= bbox.south
        ):
            out.append(g)
    return out


def rows_in_bbox(table: pa.Table, bbox: BBox) -> pa.Table:
    """The buildings whose own bounding box overlaps `bbox`."""
    b = table.column("bbox")
    keep = pc.and_(
        pc.and_(
            pc.less_equal(pc.struct_field(b, "xmin"), bbox.east),
            pc.greater_equal(pc.struct_field(b, "xmax"), bbox.west),
        ),
        pc.and_(
            pc.less_equal(pc.struct_field(b, "ymin"), bbox.north),
            pc.greater_equal(pc.struct_field(b, "ymax"), bbox.south),
        ),
    )
    return table.filter(keep)


def read_from(source, bbox: BBox) -> pa.Table:
    """The buildings in `bbox` from one Parquet file: a path, or a file object (RangeFile)."""
    pf = pq.ParquetFile(source, pre_buffer=True)
    groups = row_groups_for_bbox(pf.metadata, bbox)
    if not groups:
        return pf.schema_arrow.empty_table().select(COLUMNS)
    return rows_in_bbox(pf.read_row_groups(groups, columns=COLUMNS), bbox)


# --- the download, cached --------------------------------------------------------------------


def fetch_buildings(
    bbox: BBox, *, release: str | None = None, refresh: bool = False
) -> tuple[Download, str]:
    """Overture buildings overlapping `bbox`, saved in the cache as one small Parquet file.

    Returns the download record and the release it came from.
    """
    release, collection = building_collection(release)
    w, s, e, n = bbox.as_tuple()
    name = f"overture/buildings_{release}_{w:.7f}_{s:.7f}_{e:.7f}_{n:.7f}.parquet"
    if not refresh and (hit := cached(name)) is not None:
        return hit, release
    tables = []
    for href in files_for_bbox(collection, bbox):
        item = get_json(href)
        tables.append(read_from(RangeFile(item["assets"]["aws"]["href"]), bbox))
    table = pa.concat_tables(tables) if tables else None
    path = cache_dir() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if table is None:
        raise ValueError(f"no Overture building file covers {bbox.as_tuple()}")
    tmp = path.with_name(path.name + ".part")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)
    download = Download(
        path=path,
        url=f"{STAC_CATALOG}#release={release}&theme=buildings&type=building"
        f"&bbox={w:.7f},{s:.7f},{e:.7f},{n:.7f}",
        accessed=dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        bytes=path.stat().st_size,
        sha256=sha256_file(path),
        licence=LICENCE,
        attribution=ATTRIBUTION,
    )
    _meta_path(path).write_text(json.dumps(download.record(), indent=2) + "\n")
    return download, release


def load(path: Path) -> pa.Table:
    return pq.read_table(path)
