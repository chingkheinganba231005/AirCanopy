"""Reading Overture Maps buildings, and joining them to a site from OpenStreetMap."""

import http.server
import re
import threading
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import shapely

from aircanopy.geo import BBox
from aircanopy.site import build
from aircanopy.site.from_osm import site_from_osm
from aircanopy.site.from_overture import join_overture
from aircanopy.site.heights import height_kind
from aircanopy.sources import osm, overture

from .site_helpers import building, empty_site, square

DATA = Path(__file__).parent / "data"
OSM_SAMPLE = DATA / "osm" / "hku_small.osm.gz"
OVERTURE_SAMPLE = DATA / "overture" / "hku_small.parquet"
SAMPLE_BOX = BBox(114.1360, 22.2832, 114.1380, 22.2848)

SOURCE = pa.struct(
    [
        ("property", pa.string()),
        ("dataset", pa.string()),
        ("license", pa.string()),
        ("record_id", pa.string()),
    ]
)
SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("sources", pa.list_(SOURCE)),
        ("height", pa.float64()),
        ("min_height", pa.float64()),
        ("num_floors", pa.int32()),
        ("min_floor", pa.int32()),
        ("is_underground", pa.bool_()),
        ("has_parts", pa.bool_()),
        ("subtype", pa.string()),
        ("class", pa.string()),
        ("geometry", pa.binary()),
        (
            "bbox",
            pa.struct([(k, pa.float64()) for k in ("xmin", "xmax", "ymin", "ymax")]),
        ),
    ]
)


def row(fid, lonlat_polygon, sources, **values):
    xmin, ymin, xmax, ymax = lonlat_polygon.bounds
    return {
        "id": fid,
        "sources": sources,
        "height": None,
        "min_height": None,
        "num_floors": None,
        "min_floor": None,
        "is_underground": False,
        "has_parts": False,
        "subtype": None,
        "class": None,
        "geometry": shapely.to_wkb(lonlat_polygon),
        "bbox": {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax},
        **values,
    }


def src(dataset, record_id="", prop="", licence=None):
    return {"property": prop, "dataset": dataset, "license": licence, "record_id": record_id}


def to_lonlat(site, poly):
    def convert(xy):
        lon, lat = site.frame.to_lonlat(xy[:, 0], xy[:, 1])
        return np.column_stack([lon, lat])

    return shapely.transform(poly, convert)


# --- finding and reading the right pieces ----------------------------------------------------


def test_files_for_bbox_uses_the_per_file_boxes():
    collection = {
        "extent": {"spatial": {"bbox": [[-180, -90, 180, 90], [0, 0, 10, 10], [10, 0, 20, 10]]}},
        "links": [
            {"rel": "root", "href": "r"},
            {"rel": "item", "href": "a.json"},
            {"rel": "item", "href": "b.json"},
        ],
    }
    assert overture.files_for_bbox(collection, BBox(1, 1, 2, 2)) == ["a.json"]
    assert overture.files_for_bbox(collection, BBox(9, 1, 11, 2)) == ["a.json", "b.json"]
    assert overture.files_for_bbox(collection, BBox(30, 1, 31, 2)) == []
    collection["links"].pop()
    with pytest.raises(ValueError, match="2 file boxes but 1 items"):
        overture.files_for_bbox(collection, BBox(1, 1, 2, 2))


def three_clusters(path: Path) -> None:
    """A Parquet file with three row groups, far apart: near lon 0, 10 and 20."""
    rows = []
    for lon in (0, 0.001, 10, 10.001, 20, 20.001):
        rows.append(row(f"b{lon}", shapely.box(lon, 0, lon + 0.0005, 0.0005), [src("X")]))
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path, row_group_size=2)


def test_only_row_groups_that_can_overlap_are_read(tmp_path):
    path = tmp_path / "three.parquet"
    three_clusters(path)
    meta = pq.ParquetFile(path).metadata
    assert meta.num_row_groups == 3
    assert overture.row_groups_for_bbox(meta, BBox(9.9, -1, 10.1, 1)) == [1]
    assert overture.row_groups_for_bbox(meta, BBox(-1, -1, 30, 1)) == [0, 1, 2]
    got = overture.read_from(path, BBox(9.9995, -0.1, 10.0002, 0.1))
    assert got.column("id").to_pylist() == ["b10"]  # its neighbour 0.001 degrees east is left out
    assert overture.read_from(path, BBox(50, 50, 51, 51)).num_rows == 0


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    data = b""

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.data)))
        self.end_headers()

    def do_GET(self):
        m = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
        if not m:
            self.send_error(416)
            return
        body = self.data[int(m.group(1)) : int(m.group(2)) + 1]
        self.send_response(206)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def serve_bytes(data: bytes):
    handler = type("H", (_RangeHandler,), {"data": data})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/file.parquet"


def test_range_file_reads_the_same_as_a_local_file():
    server, url = serve_bytes(OVERTURE_SAMPLE.read_bytes())
    try:
        got = overture.read_from(overture.RangeFile(url), SAMPLE_BOX)
        assert got.num_rows > 0
        assert got.equals(overture.read_from(OVERTURE_SAMPLE, SAMPLE_BOX))
    finally:
        server.shutdown()
        server.server_close()


def test_range_file_downloads_only_the_row_group_it_needs(tmp_path):
    # Three row groups of about 1 MB each; random padding keeps compression from shrinking them.
    rng = np.random.default_rng(0)
    rows = []
    for lon in (0.0, 10.0, 20.0):
        for k in range(1000):
            r = row(f"b{lon}-{k}", shapely.box(lon, 0, lon + 1e-4, 1e-4), [src("X")])
            r["pad"] = rng.bytes(1000)
            rows.append(r)
    path = tmp_path / "big.parquet"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=SCHEMA.append(pa.field("pad", pa.binary()))),
        path,
        row_group_size=1000,
    )
    server, url = serve_bytes(path.read_bytes())
    try:
        remote = overture.RangeFile(url)
        got = overture.read_from(remote, BBox(9.9, -1, 10.1, 1))
        assert got.num_rows == 1000
        # The padding column is never read either, so far less than a third comes over the wire.
        assert remote.received < remote.size / 10
        assert remote.requests <= 6
    finally:
        server.shutdown()
        server.server_close()


def test_credits_for_known_and_unknown_datasets():
    text, licence = overture.credit("doi:10.5281/zenodo.8174931")
    assert "Qian Shi" in text and licence == "CC-BY-4.0"
    assert overture.credit("Something New")[1].startswith("see https://docs.overturemaps.org")


# --- joining to a site ------------------------------------------------------------------------


def test_join_real_sample_adds_footprints_without_doubling_buildings():
    site = site_from_osm("hku_small", SAMPLE_BOX, osm.parse(OSM_SAMPLE))
    osm_count = len(site.buildings)
    table = overture.load(OVERTURE_SAMPLE)
    report = join_overture(site, table, "2026-10-21.0")
    assert report.rows == table.num_rows
    assert report.underground == 1  # HKU MTR station, left out from OpenStreetMap too
    added = site.buildings[osm_count:]
    assert len(added) == report.added > 0
    assert {b.footprint_source for b in added} == {"overture:doi:10.5281/zenodo.8174931"}
    assert len({b.id for b in site.buildings}) == len(site.buildings)
    # No added footprint is mostly inside a building that was already there.
    for b in added:
        for other in site.buildings[:osm_count]:
            shared = b.footprint.intersection(other.footprint).area
            assert shared <= 0.5 * min(b.footprint.area, other.footprint.area)


def test_better_heights_replace_worse_ones_and_never_the_reverse():
    site = empty_site()
    levels = building("osm:way/1", square(0, 0, 20), 30.0)
    levels.height_source = "osm:levels"
    unknown = building("osm:way/2", square(40, 0, 20), None)
    mapped = building("osm:way/3", square(-40, 0, 20), 25.0)  # osm:height
    site.buildings = [levels, unknown, mapped]
    ll = lambda poly: to_lonlat(site, poly)  # noqa: E731
    table = pa.Table.from_pylist(
        [
            # Lidar height for way/1: more trusted than storeys.
            row(
                "o1",
                ll(square(0, 0, 20)),
                [src("OpenStreetMap", "w1@3"), src("USGS Lidar", prop="/properties/height")],
                height=31.5,
            ),
            # A machine-learning height for way/2, which had none.
            row(
                "o2",
                ll(square(40, 0, 20)),
                [
                    src("OpenStreetMap", "w2@1"),
                    src("Microsoft ML Buildings", prop="/properties/height"),
                ],
                height=12.0,
            ),
            # A machine-learning height for way/3 is less trusted than its mapped height.
            row(
                "o3",
                ll(square(-40, 0, 20)),
                [
                    src("OpenStreetMap", "w3@1"),
                    src("Microsoft ML Buildings", prop="/properties/height"),
                ],
                height=99.0,
            ),
            # Overture's copy of an OpenStreetMap height adds nothing.
            row("o4", ll(square(-40, 0, 20)), [src("OpenStreetMap", "w3@1")], height=50.0),
            # A traced footprint over way/1: the same building twice.
            row("o5", ll(square(2, 2, 18)), [src("Microsoft ML Buildings")], height=28.0),
            # A traced footprint where the map has none: added, with its height.
            row("o6", ll(square(0, 50, 10)), [src("Microsoft ML Buildings")], height=7.5),
        ],
        schema=SCHEMA,
    )
    report = join_overture(site, table)
    assert (levels.height, levels.height_source, levels.height_dataset) == (
        31.5,
        "overture:height",
        "USGS Lidar",
    )
    assert height_kind(levels.height_source, levels.height_dataset) == "measured"
    assert (unknown.height, height_kind(unknown.height_source, unknown.height_dataset)) == (
        12.0,
        "ml",
    )
    assert (mapped.height, mapped.height_source) == (25.0, "osm:height")
    assert report.heights_improved == 2 and report.duplicates == 1 and report.added == 1
    new = site.buildings[-1]
    assert new.id == "overture:o6" and new.footprint_source == "overture:Microsoft ML Buildings"
    assert new.height == 7.5 and height_kind(new.height_source, new.height_dataset) == "ml"
    assert new.footprint.area == pytest.approx(100, rel=1e-3)


def test_site_summary_credits_every_dataset_used(tmp_path):
    summary = build.build_from_osm_data(
        "hku_small",
        SAMPLE_BOX,
        osm.parse(OSM_SAMPLE),
        tmp_path,
        2.0,
        [],
        overture.load(OVERTURE_SAMPLE),
        {"release": "2026-10-21.0"},
    )
    datasets = {c["dataset"]: c for c in summary["credits"]}
    assert datasets["OpenStreetMap"]["licence"] == "ODbL-1.0"
    assert datasets["doi:10.5281/zenodo.8174931"]["licence"] == "CC-BY-4.0"
    assert summary["overture"]["release"] == "2026-10-21.0"
    assert summary["overture"]["added"] == datasets["doi:10.5281/zenodo.8174931"]["buildings"]
    assert sum(summary["height_kinds"].values()) == summary["buildings"]
    assert sum(summary["footprint_sources"].values()) == summary["buildings"]
