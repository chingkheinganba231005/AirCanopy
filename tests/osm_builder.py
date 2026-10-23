"""Write small OpenStreetMap XML documents for tests.

Coordinates are given in metres east and north of (114.0 E, 22.0 N) and converted with the
site's own local frame, so test geometry can be stated in metres.
"""

from __future__ import annotations

from aircanopy.geo import LocalFrame

LON0, LAT0 = 114.0, 22.0


class OsmBuilder:
    def __init__(self) -> None:
        self.frame = LocalFrame(LON0, LAT0)
        self.nodes: list[str] = []
        self.ways: list[str] = []
        self.relations: list[str] = []
        self._next = 1

    def _id(self) -> int:
        self._next += 1
        return self._next

    @staticmethod
    def _tags(tags: dict | None) -> str:
        return "".join(f'<tag k="{k}" v="{v}"/>' for k, v in (tags or {}).items())

    def node(self, x: float, y: float, tags: dict | None = None) -> int:
        lon, lat = self.frame.to_lonlat(x, y)
        nid = self._id()
        attrs = f'id="{nid}" lat="{float(lat):.9f}" lon="{float(lon):.9f}"'
        self.nodes.append(f"<node {attrs}>{self._tags(tags)}</node>")
        return nid

    def way(self, refs: list[int], tags: dict | None = None) -> int:
        wid = self._id()
        nds = "".join(f'<nd ref="{r}"/>' for r in refs)
        self.ways.append(f'<way id="{wid}">{nds}{self._tags(tags)}</way>')
        return wid

    def ring(self, points: list[tuple[float, float]], tags: dict | None = None) -> int:
        refs = [self.node(x, y) for x, y in points]
        return self.way([*refs, refs[0]], tags)

    def relation(self, members: list[tuple[str, int, str]], tags: dict) -> int:
        rid = self._id()
        ms = "".join(f'<member type="{t}" ref="{r}" role="{role}"/>' for t, r, role in members)
        self.relations.append(f'<relation id="{rid}">{ms}{self._tags(tags)}</relation>')
        return rid

    def xml(self) -> bytes:
        body = "".join(self.nodes + self.ways + self.relations)
        return f'<?xml version="1.0"?><osm version="0.6">{body}</osm>'.encode()


def square(x0: float, y0: float, size: float) -> list[tuple[float, float]]:
    return [(x0, y0), (x0 + size, y0), (x0 + size, y0 + size), (x0, y0 + size)]
