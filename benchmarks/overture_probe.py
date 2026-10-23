"""What reading Overture Maps costs, and what its building data hold in a few regions.

1. Read cost: the requests and bytes needed for the buildings of the 500 m HKU site.
2. Contents: in the middle row group of seven files spread over the world, how many buildings
   have a height, and which datasets gave footprints and heights.

Needs the network. Writes results/overture_probe.json.

    uv run python benchmarks/overture_probe.py
"""

from __future__ import annotations

import datetime as dt
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from aircanopy import __version__
from aircanopy.geo import BBox
from aircanopy.sources import overture

ROOT = Path(__file__).resolve().parents[1]
HKU = BBox.around(114.1370, 22.2840, 500)
SAMPLE_FILES = (40, 120, 200, 260, 330, 420, 470)


def read_cost(collection: dict) -> dict:
    files = overture.files_for_bbox(collection, HKU)
    out = {"bbox": list(HKU.as_tuple()), "files_opened": len(files), "files": []}
    for href in files:
        item = overture.get_json(href)
        remote = overture.RangeFile(item["assets"]["aws"]["href"])
        pf = pq.ParquetFile(remote, pre_buffer=True)
        footer = {"requests": remote.requests, "bytes": remote.received}
        groups = overture.row_groups_for_bbox(pf.metadata, HKU)
        table = overture.rows_in_bbox(pf.read_row_groups(groups, columns=overture.COLUMNS), HKU)
        out["files"].append(
            {
                "file_bytes": remote.size,
                "row_groups": pf.metadata.num_row_groups,
                "row_groups_read": len(groups),
                "footer": footer,
                "total_requests": remote.requests,
                "total_bytes": remote.received,
                "buildings_in_box": table.num_rows,
            }
        )
    return out


def contents(collection: dict) -> list[dict]:
    items = [link["href"] for link in collection["links"] if link.get("rel") == "item"]
    boxes = collection["extent"]["spatial"]["bbox"][1:]
    rows = []
    for i in SAMPLE_FILES:
        item = overture.get_json(items[i])
        pf = pq.ParquetFile(overture.RangeFile(item["assets"]["aws"]["href"]))
        t = pf.read_row_group(pf.metadata.num_row_groups // 2, columns=["sources", "height"])
        footprints, heights = Counter(), Counter()
        for sources in t.column("sources").to_pylist():
            sources = sources or []
            if sources:
                footprints[sources[0]["dataset"]] += 1
            for s in sources:
                if s["property"] == "/properties/height":
                    heights[s["dataset"]] += 1
        rows.append(
            {
                "file": i,
                "file_bbox": boxes[i],
                "buildings": t.num_rows,
                "with_height": sum(h is not None for h in t.column("height").to_pylist()),
                "footprints_by_dataset": dict(footprints),
                "heights_given_by_another_dataset": dict(heights),
            }
        )
    return rows


def main() -> None:
    release, collection = overture.building_collection()
    out = {
        "what": "cost of reading Overture buildings for one site, and their contents by region",
        "run_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "aircanopy": __version__,
        "release": release,
        "read_cost_hku_500m": read_cost(collection),
        "sample_row_groups": contents(collection),
    }
    path = ROOT / "results" / "overture_probe.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
