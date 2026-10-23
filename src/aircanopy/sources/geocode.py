"""Find a place by name, with OpenStreetMap's geocoder (Nominatim).

Names are ambiguous. On 2026-10-23, "University of Hong Kong" returned one result: a fast-food
restaurant called "Hong-Kong" in Greifswald, Germany. "The University of Hong Kong" returned the
university. So a place is never used silently: the candidates are listed, and the one used is
said out loud.

Nominatim's usage policy (https://operations.osmfoundation.org/policies/nominatim/, read
2026-10-23) asks for at most 1 request per second, a User-Agent that identifies the
application, attribution, caching of results, and a service address that can be switched
without a software update. Set AIRCANOPY_GEOCODER to use another Nominatim server.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass

from .cache import USER_AGENT, cache_dir

DEFAULT_ENDPOINT = "https://nominatim.openstreetmap.org/search"
MIN_INTERVAL_S = 1.0  # the policy's maximum of 1 request per second
LICENCE = "ODbL-1.0"

_last_request = 0.0


@dataclass(frozen=True)
class Place:
    name: str  # the full address, as the geocoder writes it
    lon: float
    lat: float
    kind: str  # for example "amenity/university"
    osm: str  # the OpenStreetMap object, for example "relation/4246029"
    licence: str  # as the geocoder states it

    def to_dict(self) -> dict:
        return asdict(self)


def endpoint() -> str:
    return os.environ.get("AIRCANOPY_GEOCODER", DEFAULT_ENDPOINT)


def parse(results: list[dict]) -> list[Place]:
    return [
        Place(
            name=r.get("display_name", ""),
            lon=float(r["lon"]),
            lat=float(r["lat"]),
            kind=f"{r.get('category', '?')}/{r.get('type', '?')}",
            osm=f"{r.get('osm_type', '?')}/{r.get('osm_id', '?')}",
            licence=r.get("licence", ""),
        )
        for r in results
    ]


def _wait_turn() -> None:
    global _last_request
    wait = MIN_INTERVAL_S - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()


def search(query: str, limit: int = 5, refresh: bool = False) -> list[Place]:
    """Places matching `query`, best first (as ranked by the geocoder); cached on disk."""
    url = endpoint()
    key = hashlib.sha256(f"{url}|{query}|{limit}".encode()).hexdigest()[:24]
    path = cache_dir() / "geocode" / f"{key}.json"
    if not refresh and path.is_file():
        return parse(json.loads(path.read_text())["results"])
    params = {"q": query, "format": "jsonv2", "limit": str(limit), "accept-language": "en"}
    request = urllib.request.Request(
        f"{url}?{urllib.parse.urlencode(params)}", headers={"User-Agent": USER_AGENT}
    )
    _wait_turn()
    with urllib.request.urlopen(request, timeout=60) as response:
        results = json.load(response)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "query": query,
        "endpoint": url,
        "accessed": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results": results,
    }
    path.write_text(json.dumps(record, indent=2) + "\n")
    return parse(results)
