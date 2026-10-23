"""Filling gaps in the data, with labelled estimates.

Map data are incomplete: many buildings have no height. Before a building can be drawn or
simulated it needs one. This module fills gaps with estimates made from the data themselves
(not from fixed guesses) and labels every filled value, so a result can always say how much of
it rests on estimates.

Building heights: a building without a height gets the median height of the buildings around it
that do have one (within `radius` metres of its centre, at least `min_neighbours` of them). The
median is used rather than the mean because one tower among low houses would pull a mean far up.
If too few neighbours have heights, the height stays missing and the building is reported.
"""

from __future__ import annotations

import numpy as np
import shapely

from .heights import NEIGHBOUR_MEDIAN
from .model import Site

__all__ = ["MIN_NEIGHBOURS", "NEIGHBOUR_MEDIAN", "NEIGHBOUR_RADIUS_M", "fill_building_heights"]
NEIGHBOUR_RADIUS_M = 100.0  # assumption A-002
MIN_NEIGHBOURS = 3  # assumption A-002


def fill_building_heights(
    site: Site, radius: float = NEIGHBOUR_RADIUS_M, min_neighbours: int = MIN_NEIGHBOURS
) -> int:
    """Fill missing building heights from neighbours; returns how many were filled."""
    known = [
        b for b in site.buildings if b.height is not None and b.height_source != NEIGHBOUR_MEDIAN
    ]
    if not known:
        return 0
    heights = np.array([b.height for b in known])
    tree = shapely.STRtree(shapely.centroid([b.footprint for b in known]))
    filled = 0
    for b in site.buildings:
        if b.height is not None:
            continue
        near = tree.query(b.footprint.centroid.buffer(radius), predicate="contains")
        if len(near) < min_neighbours:
            continue
        b.height = float(np.median(heights[near]))
        if b.height <= b.base:
            b.height = None
            continue
        b.height_source = NEIGHBOUR_MEDIAN
        filled += 1
    return filled
