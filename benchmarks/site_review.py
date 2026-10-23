"""How much of a site's buildings rests on estimated heights, and which tags leave heights in
doubt.

For a site folder made by `aircanopy site build`, and the OpenStreetMap file it was built from
(found in the cache and checked against the checksum the site recorded):

- the share of buildings, of footprint area and of volume for each kind of height;
- the largest buildings whose height was estimated from their neighbours, with their names;
- the tallest buildings, roads and paths in tunnels and on bridges, and the features left out;
- the buildings tagged `location=podium`. The OpenStreetMap wiki does not define this value
  (https://wiki.openstreetmap.org/wiki/Key:location, read 2026-10-23), and it defines
  `building:levels` as counted from the ground, skipped levels included
  (https://wiki.openstreetmap.org/wiki/Key:building:levels, read 2026-10-23). The site reads
  the storeys that way; if a mapper counted them from the top of the podium, the tower is too
  short by the podium's height. Those that also give `building:min_level` or `min_height` say
  where they start, and are not in doubt.

    uv run python benchmarks/site_review.py sites/mongkok

Writes results/site_review_<name>.json.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import shapely

from aircanopy import __version__
from aircanopy.site.heights import HEIGHT_KINDS, height_kind
from aircanopy.site.model import Site
from aircanopy.sources.cache import cache_dir, sha256_file

ROOT = Path(__file__).resolve().parents[1]
LARGEST = 5


def osm_file(summary: dict) -> tuple[Path, dict]:
    record = next(r for r in summary["sources"] if r.get("url", "").startswith("https://api."))
    path = cache_dir() / "osm" / record["path"]
    if sha256_file(path) != record["sha256"]:
        raise SystemExit(f"{path} is not the file the site was built from")
    return path, record


def osm_tags(path: Path) -> dict[str, dict[str, str]]:
    """Tags of every way and relation, by the id the site uses ("osm:way/123")."""
    tags = {}
    for _, el in ET.iterparse(path):
        if el.tag in ("way", "relation"):
            tags[f"osm:{el.tag}/{el.get('id')}"] = {t.get("k"): t.get("v") for t in el.iter("tag")}
            el.clear()
    return tags


def share(counter: Counter, total: float) -> dict[str, float]:
    return {k: round(counter[k] / total, 3) if total else 0.0 for k in [*HEIGHT_KINDS, "missing"]}


def main(site_dir: Path) -> None:
    site = Site.load(site_dir / "site.json")
    summary = json.loads((site_dir / "summary.json").read_text())
    path, record = osm_file(summary)
    tags = osm_tags(path)
    count, area, volume = Counter(), Counter(), Counter()
    estimated = []
    for b in site.buildings:
        kind = height_kind(b.height_source, b.height_dataset) or "missing"
        a = b.footprint.area
        count[kind] += 1
        area[kind] += a
        if b.height is not None:
            volume[kind] += a * (b.height - b.base)
        if kind == "estimated":
            t = tags.get(b.id, {})
            estimated.append(
                {
                    "id": b.id,
                    "name": t.get("name:en") or t.get("name"),
                    "building": t.get("building") or t.get("building:part"),
                    "footprint_m2": round(a),
                    "height_m": b.height,
                }
            )
    estimated.sort(key=lambda e: -e["footprint_m2"])
    in_site = {b.id for b in site.buildings}
    podium = {i: t for i, t in tags.items() if t.get("location") == "podium" and i in in_site}
    podium_without_start = [
        i for i, t in podium.items() if "building:min_level" not in t and "min_height" not in t
    ]
    tallest = sorted((b for b in site.buildings if b.height), key=lambda b: -b.height)[:3]
    square = shapely.box(*site.extent())
    ways = site.roads + site.footways

    def inside_m(lines) -> float:
        return round(sum(r.line.intersection(square).length for r in lines))

    tunnels = [r for r in ways if r.tunnel]
    bridges = [r for r in ways if r.bridge]
    result = {
        "what": "share of buildings, footprint area and volume by kind of height; the largest "
        "buildings with estimated heights; towers tagged location=podium",
        "run_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "aircanopy": __version__,
        "site": summary["name"],
        "site_built": site.created,
        "osm_file": {k: record[k] for k in ("url", "accessed", "sha256")},
        "buildings": len(site.buildings),
        "share_of_buildings": share(count, sum(count.values())),
        "share_of_footprint_area": share(area, sum(area.values())),
        "share_of_volume": share(volume, sum(volume.values())),
        "largest_estimated": estimated[:LARGEST],
        "location_podium": {
            "buildings": len(podium),
            "without_min_level_or_min_height": len(podium_without_start),
        },
        "tallest": [
            {
                "id": b.id,
                "name": tags.get(b.id, {}).get("name:en") or tags.get(b.id, {}).get("name"),
                "height_m": b.height,
                "height_kind": height_kind(b.height_source, b.height_dataset),
            }
            for b in tallest
        ],
        "roads_and_paths_inside_the_square": {
            "in_tunnels": len(tunnels),
            "in_tunnels_m": inside_m(tunnels),
            "on_bridges": len(bridges),
            "on_bridges_m": inside_m(bridges),
        },
        "skipped_in_the_download_by_reason": dict(Counter(s["reason"] for s in site.skipped)),
    }
    out = ROOT / "results" / f"site_review_{summary['name']}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
