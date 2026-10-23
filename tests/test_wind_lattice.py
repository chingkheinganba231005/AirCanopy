"""The velocity sets: weights, symmetry and the isotropy conditions."""

import itertools

import numpy as np
import pytest

from aircanopy.wind.lattice import CS2, D3Q19, D3Q27, LATTICES

LATS = [D3Q19, D3Q27]


@pytest.mark.parametrize("lat", LATS, ids=lambda lat: lat.name)
def test_isotropy_up_to_fourth_order(lat):
    c, w = lat.c.astype(float), lat.w
    d = np.eye(3)
    assert w.sum() == pytest.approx(1.0, abs=1e-15)
    assert np.allclose(np.einsum("i,ia->a", w, c), 0, atol=1e-15)
    assert np.allclose(np.einsum("i,ia,ib->ab", w, c, c), CS2 * d, atol=1e-15)
    assert np.allclose(np.einsum("i,ia,ib,ic->abc", w, c, c, c), 0, atol=1e-15)
    fourth = np.einsum("i,ia,ib,ic,id->abcd", w, c, c, c, c)
    iso = CS2**2 * (
        np.einsum("ab,cd->abcd", d, d)
        + np.einsum("ac,bd->abcd", d, d)
        + np.einsum("ad,bc->abcd", d, d)
    )
    assert np.allclose(fourth, iso, atol=1e-15)
    assert np.allclose(np.einsum("i,ia,ib,ic,id,ie->abcde", w, c, c, c, c, c), 0, atol=1e-15)


def test_sizes_and_weights():
    assert (D3Q19.q, D3Q27.q) == (19, 27)
    norm2 = (D3Q19.c**2).sum(axis=1)
    assert sorted(set(zip(norm2.tolist(), D3Q19.w.tolist(), strict=True))) == [
        (0, 1 / 3),
        (1, 1 / 18),
        (2, 1 / 36),
    ]
    norm2 = (D3Q27.c**2).sum(axis=1)
    weights = {n: set(D3Q27.w[norm2 == n].round(15)) for n in range(4)}
    assert weights == {
        0: {round(8 / 27, 15)},
        1: {round(2 / 27, 15)},
        2: {round(1 / 54, 15)},
        3: {round(1 / 216, 15)},
    }
    assert set(map(tuple, D3Q27.c.tolist())) == set(itertools.product((-1, 0, 1), repeat=3))


@pytest.mark.parametrize("lat", LATS, ids=lambda lat: lat.name)
def test_opposites_and_mirrors(lat):
    assert (lat.c[0] == 0).all()  # the rest direction comes first
    assert (lat.c[lat.opp] == -lat.c).all()
    assert (lat.opp[lat.opp] == np.arange(lat.q)).all()
    flipped = lat.c * np.array([1, 1, -1])
    assert (lat.c[lat.mirror_z] == flipped).all()
    assert LATTICES[lat.name] is lat
