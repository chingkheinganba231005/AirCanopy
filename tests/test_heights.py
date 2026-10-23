"""Building heights from OpenStreetMap tags."""

import pytest

from aircanopy.site.heights import STOREY_HEIGHT_M, building_height, parse_count, parse_length


@pytest.mark.parametrize(
    ("text", "metres"),
    [
        ("12", 12.0),
        ("12.5 m", 12.5),
        ("12.5m", 12.5),
        ("12,5", 12.5),
        ("40'", 40 * 0.3048),
        ("40 ft", 40 * 0.3048),
        ("12'6\"", 12.5 * 0.3048),
    ],
)
def test_parse_length(text, metres):
    assert parse_length(text) == pytest.approx(metres)


@pytest.mark.parametrize("text", [None, "", "tall", "12 storeys", "1,200.5"])
def test_parse_length_refuses_what_it_cannot_read(text):
    assert parse_length(text) is None


def test_parse_count():
    assert parse_count("4") == 4.0
    assert parse_count("2.5") == 2.5
    assert parse_count("-1") is None
    assert parse_count("many") is None


def test_height_tag_wins():
    assert building_height({"height": "30", "building:levels": "5"}) == (
        30.0,
        0.0,
        "osm:height",
        "ground",
    )


def test_building_height_tag():
    top, _, src, _ = building_height({"building:height": "18 m"})
    assert (top, src) == (18.0, "osm:building:height")


def test_levels_times_storey_height_with_roof():
    top, _, src, _ = building_height({"building:levels": "4", "roof:levels": "1"})
    assert top == pytest.approx(5 * STOREY_HEIGHT_M)
    assert src == "osm:levels"
    top, _, _, _ = building_height({"building:levels": "4", "roof:height": "2"})
    assert top == pytest.approx(4 * STOREY_HEIGHT_M + 2)


def test_bottom_of_raised_parts():
    assert building_height({"height": "20", "min_height": "8"})[1:] == (
        8.0,
        "osm:height",
        "osm:min_height",
    )
    _, bottom, _, src = building_height({"building:levels": "6", "building:min_level": "2"})
    assert (bottom, src) == (2 * STOREY_HEIGHT_M, "osm:levels")


def test_missing_height_stays_missing():
    assert building_height({"building": "yes"}) == (None, 0.0, "missing", "ground")


def test_contradictory_bottom_is_dropped():
    top, bottom, _, src = building_height({"height": "5", "min_height": "9"})
    assert (top, bottom, src) == (5.0, 0.0, "ground")
