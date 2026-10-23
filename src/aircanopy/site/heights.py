"""Building heights from OpenStreetMap tags, and where each height came from.

OpenStreetMap records heights in several ways (https://wiki.openstreetmap.org/wiki/Simple_3D_buildings,
accessed 2026-10-23): `height` is the top of the building above the ground, `min_height` the
bottom of a part that does not reach the ground, and `building:levels` / `building:min_level`
count storeys. When only storeys are given, the height is the number of storeys times a storey
height. That storey height is an assumption (docs/assumptions.md, A-001) and every height made
from it is labelled, so it can be replaced as soon as better data exist.
"""

from __future__ import annotations

import re

#: Height of one storey when a height must be estimated from a storey count (assumption A-001).
STOREY_HEIGHT_M = 3.0

#: Label of a height estimated as the median of neighbouring buildings (site/fill.py, A-002).
NEIGHBOUR_MEDIAN = "estimated:neighbour-median"

#: Kinds of building height, from most to least trusted. A site keeps the most trusted height
#: it has for each building, and the viewer colours buildings by kind:
#: measured (lidar), mapped (a height given in map data), levels (storeys x STOREY_HEIGHT_M),
#: ml (estimated by machine learning from images), estimated (median of neighbours).
HEIGHT_KINDS = ("measured", "mapped", "levels", "ml", "estimated")

#: How the datasets that Overture lists for heights obtained them (sources/overture.py).
#: A dataset not listed here counts as "mapped".
DATASET_HEIGHT_KIND = {
    "USGS Lidar": "measured",
    "Microsoft ML Buildings": "ml",
    "Google Open Buildings": "ml",
    "OpenStreetMap": "mapped",
}


def height_kind(source: str, dataset: str | None = None) -> str | None:
    """The kind of a height (one of HEIGHT_KINDS), or None when there is no height."""
    if source in ("osm:height", "osm:building:height"):
        return "mapped"
    if source in ("osm:levels", "overture:num_floors"):
        return "levels"
    if source == "overture:height":
        return DATASET_HEIGHT_KIND.get(dataset or "", "mapped")
    if source == NEIGHBOUR_MEDIAN:
        return "estimated"
    return None


FOOT_M = 0.3048  # exact, by definition of the international foot

_NUMBER = r"[-+]?\d+(?:\.\d+)?"
_METRES = re.compile(rf"^\s*({_NUMBER})\s*(m|metres|meters|metre|meter)?\s*$", re.I)
_FEET = re.compile(
    rf"^\s*({_NUMBER})\s*(?:'|ft|feet|foot)\s*(?:({_NUMBER})\s*(?:\"|in|inch|inches))?\s*$", re.I
)


def parse_length(text: str | None) -> float | None:
    """A length in metres from a tag value such as "12", "12.5 m", "40'", "12'6\""; else None."""
    if text is None:
        return None
    t = text.strip()
    if "," in t and "." not in t:  # a decimal comma, as in "12,5"
        t = t.replace(",", ".")
    if m := _METRES.match(t):
        return float(m.group(1))
    if m := _FEET.match(t):
        feet = float(m.group(1))
        inches = float(m.group(2)) if m.group(2) else 0.0
        return (feet + inches / 12.0) * FOOT_M
    return None


def parse_count(text: str | None) -> float | None:
    """A storey count such as "4" or "2.5"; None if it is not a plain non-negative number."""
    if text is None:
        return None
    try:
        v = float(text.strip().replace(",", "."))
    except ValueError:
        return None
    return v if v >= 0 else None


def building_height(tags: dict[str, str]) -> tuple[float | None, float, str, str]:
    """(top, bottom, source of top, source of bottom) above the ground, in metres.

    The top is None when the tags say nothing about it; the caller decides what to do then.
    Sources are tag names, or "osm:levels" when estimated from storeys, or "ground".
    """
    top: float | None = None
    top_source = "missing"
    for key in ("height", "building:height"):
        if (v := parse_length(tags.get(key))) is not None and v > 0:
            top, top_source = v, f"osm:{key}"
            break
    if top is None:
        levels = parse_count(tags.get("building:levels"))
        if levels is not None and levels > 0:
            roof = parse_length(tags.get("roof:height"))
            if roof is None:
                roof_levels = parse_count(tags.get("roof:levels")) or 0.0
                roof = roof_levels * STOREY_HEIGHT_M
            top, top_source = levels * STOREY_HEIGHT_M + roof, "osm:levels"
    bottom, bottom_source = 0.0, "ground"
    if (v := parse_length(tags.get("min_height"))) is not None and v >= 0:
        bottom, bottom_source = v, "osm:min_height"
    elif (v := parse_count(tags.get("building:min_level"))) is not None and v > 0:
        bottom, bottom_source = v * STOREY_HEIGHT_M, "osm:levels"
    if top is not None and top <= bottom:
        # Contradictory tags: keep the top, drop the bottom rather than invent a thickness.
        bottom, bottom_source = 0.0, "ground"
    return top, bottom, top_source, bottom_source
