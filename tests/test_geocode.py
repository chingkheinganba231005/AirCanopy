"""Finding places by name."""

import io
import json
from pathlib import Path

import pytest

from aircanopy import cli
from aircanopy.sources import geocode

SAMPLE = Path(__file__).parent / "data" / "geocode" / "the_university_of_hong_kong.json"


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_geocoder(monkeypatch, tmp_path):
    monkeypatch.setenv("AIRCANOPY_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(geocode.time, "sleep", lambda s: None)
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        return FakeResponse(SAMPLE.read_bytes())

    monkeypatch.setattr(geocode.urllib.request, "urlopen", urlopen)
    return requests


def test_parse_a_real_answer():
    (place,) = geocode.parse(json.loads(SAMPLE.read_text()))
    assert place.name.startswith("The University of Hong Kong")
    assert (place.lat, place.lon) == pytest.approx((22.2828768, 114.1378586))
    assert place.kind == "amenity/university" and place.osm == "relation/4246029"
    assert "OpenStreetMap contributors" in place.licence


def test_search_identifies_itself_and_caches(fake_geocoder):
    first = geocode.search("The University of Hong Kong")
    again = geocode.search("The University of Hong Kong")
    assert first == again
    assert len(fake_geocoder) == 1  # the second answer came from the cache
    request = fake_geocoder[0]
    assert request.get_header("User-agent").startswith("AirCanopy/")
    assert "q=The+University+of+Hong+Kong" in request.full_url


def test_at_most_one_request_per_second(monkeypatch):
    clock = iter([100.0, 100.0, 100.3, 100.3])
    slept = []
    monkeypatch.setattr(geocode.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(geocode.time, "sleep", slept.append)
    monkeypatch.setattr(geocode, "_last_request", 0.0)
    geocode._wait_turn()  # long after the last request: no wait
    geocode._wait_turn()  # 0.3 s after it: wait the rest of the second
    assert slept == [pytest.approx(0.7)]


def test_another_server_can_be_used(monkeypatch):
    monkeypatch.setenv("AIRCANOPY_GEOCODER", "https://nominatim.example.org/search")
    assert geocode.endpoint() == "https://nominatim.example.org/search"


def test_place_command_lists_what_it_found(fake_geocoder, capsys):
    assert cli.main(["place", "The University of Hong Kong"]) == 0
    out = capsys.readouterr().out
    assert 'Places found for "The University of Hong Kong"' in out
    assert "1. The University of Hong Kong" in out and "(22.28288, 114.13786)" in out


def test_build_uses_the_picked_place_and_says_so(fake_geocoder, monkeypatch, capsys):
    from aircanopy.site import build

    seen = {}

    def fake_build_site(name, bbox, out, **kwargs):
        seen.update(bbox=bbox, sources=kwargs["extra_sources"])
        return {"name": name}

    monkeypatch.setattr(build, "build_site", fake_build_site)
    args = ["site", "build", "--place", "The University of Hong Kong", "--size", "300"]
    assert cli.main([*args, "--name", "hku"]) == 0
    assert "<- used" in capsys.readouterr().out
    w, h = seen["bbox"].size_m()
    assert (w, h) == pytest.approx((300, 300), rel=1e-3)
    (record,) = seen["sources"]
    assert record["query"] == "The University of Hong Kong" and record["pick"] == 1
    with pytest.raises(SystemExit, match="--pick 2"):
        cli.main([*args, "--pick", "2", "--name", "hku"])
