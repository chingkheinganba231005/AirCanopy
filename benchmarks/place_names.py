"""What OpenStreetMap's geocoder answers for a few names, to show why a place found by name is
never used silently (sources/geocode.py).

Asks afresh (not from the cache), at most once a second as the geocoder's usage policy asks.
Needs the network. Writes results/place_names.json.

    uv run python benchmarks/place_names.py
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from aircanopy import __version__
from aircanopy.sources import geocode

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["University of Hong Kong", "The University of Hong Kong", "Mong Kok"]


def main() -> None:
    answers = {}
    for name in NAMES:
        places = geocode.search(name, refresh=True)
        answers[name] = [
            {"name": p.name, "kind": p.kind, "osm": p.osm, "lat": p.lat, "lon": p.lon}
            for p in places
        ]
    result = {
        "what": "the places the geocoder lists for each name, best first",
        "run_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "aircanopy": __version__,
        "geocoder": geocode.endpoint(),
        "licence": "Data © OpenStreetMap contributors, ODbL 1.0",
        "answers": answers,
    }
    path = ROOT / "results" / "place_names.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
