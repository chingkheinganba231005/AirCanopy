"""Filling missing building heights from neighbours."""

from aircanopy.site.fill import NEIGHBOUR_MEDIAN, fill_building_heights

from .site_helpers import building, empty_site, square


def test_missing_height_becomes_the_neighbour_median():
    site = empty_site()
    site.buildings = [
        building("a", square(0, 0, 10), 9.0),
        building("b", square(20, 0, 10), 12.0),
        building("c", square(40, 0, 10), 60.0),  # one tower: the median ignores it
        building("u", square(20, 20, 10), None),
    ]
    assert fill_building_heights(site) == 1
    u = site.buildings[-1]
    assert u.height == 12.0
    assert u.height_source == NEIGHBOUR_MEDIAN


def test_too_few_neighbours_leaves_the_height_missing():
    site = empty_site()
    site.buildings = [
        building("a", square(0, 0, 10), 9.0),
        building("b", square(20, 0, 10), 12.0),
        building("u", square(20, 20, 10), None),
    ]
    assert fill_building_heights(site) == 0
    assert site.buildings[-1].height is None
    assert site.buildings[-1].height_source == "missing"


def test_neighbours_beyond_the_radius_do_not_count():
    site = empty_site(600)
    site.buildings = [
        building("a", square(200, 0, 10), 9.0),
        building("b", square(220, 0, 10), 12.0),
        building("c", square(240, 0, 10), 15.0),
        building("u", square(0, 0, 10), None),
    ]
    assert fill_building_heights(site, radius=100.0) == 0
    assert fill_building_heights(site, radius=300.0) == 1


def test_an_estimate_below_the_base_is_not_used():
    site = empty_site()
    site.buildings = [
        building("a", square(0, 0, 10), 6.0),
        building("b", square(20, 0, 10), 6.0),
        building("c", square(40, 0, 10), 6.0),
        building("u", square(20, 20, 10), None, base=8.0),  # a part raised 8 m: 6 m is no top
    ]
    assert fill_building_heights(site) == 0
    assert site.buildings[-1].height is None
    assert site.buildings[-1].height_source == "missing"


def test_estimates_are_never_used_as_neighbours():
    site = empty_site()
    site.buildings = [
        building("a", square(0, 0, 10), 9.0),
        building("b", square(20, 0, 10), 12.0),
        building("c", square(40, 0, 10), 15.0),
        building("u", square(20, 20, 10), None),
    ]
    fill_building_heights(site)
    site.buildings[0].height = None  # now only b, c and the estimate u have heights
    site.buildings[0].height_source = "missing"
    assert fill_building_heights(site) == 0
