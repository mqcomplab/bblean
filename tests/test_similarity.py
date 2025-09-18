import numpy as np
import pytest
from bblean.similarity import _popcount, jt_most_dissimilar_packed
from bblean.cpp_similarity import (
    jt_isim,
    jt_sim_packed,
    jt_most_dissimilar_packed as jt_most_dissimilar_packed_cpp,
    _popcount_1d,
    _popcount_2d,
    _unpack_fingerprints_2d,
    _calc_centroid_packed_u8_from_u64,
)
from bblean.fingerprints import (
    make_fake_fingerprints,
    calc_centroid,
    unpack_fingerprints,
)


def test_cpp_dissimilar() -> None:
    fps = make_fake_fingerprints(10, seed=17408390758220920002)
    expect_idx1 = 1
    expect_idx2 = 2
    expect_sims1 = np.array(
        [
            0.05083333,
            1.0,
            0.03805175,
            0.05077805,
            0.04651163,
            0.04683841,
            0.05954198,
            0.06254826,
            0.05578947,
            0.05006954,
        ]
    )
    expect_sims2 = np.array(
        [
            0.23452294,
            0.03805175,
            1.0,
            0.2352518,
            0.08961039,
            0.1166033,
            0.22281879,
            0.2363388,
            0.2045264,
            0.17490119,
        ]
    )
    (
        idx1,
        idx2,
        sims1,
        sims2,
    ) = jt_most_dissimilar_packed(fps)
    assert idx1 == expect_idx1
    assert idx2 == expect_idx2
    assert np.isclose(sims1, expect_sims1).all()
    assert np.isclose(sims2, expect_sims2).all()

    # Both 1 and 2 are equal, and both are incorrect
    (
        idx1,
        idx2,
        sims1,
        sims2,
    ) = jt_most_dissimilar_packed_cpp(fps)
    assert idx1 == expect_idx1
    assert idx2 == expect_idx2
    assert np.isclose(sims1, expect_sims1).all()
    assert np.isclose(sims2, expect_sims2).all()


def test_cpp_popcount() -> None:
    fps_1d = make_fake_fingerprints(1, seed=17408390758220920002).reshape(-1)
    fps_2d = make_fake_fingerprints(10, seed=17408390758220920002)
    expect_1d = 1137
    expect_2d = [1137, 124, 558, 1159, 281, 323, 1264, 1252, 879, 631]
    out = _popcount_1d(fps_1d)
    out = _popcount(fps_1d)
    assert (out == _popcount(fps_1d)).all()
    assert out.item() == expect_1d
    out = _popcount_2d(fps_2d)
    out = _popcount(fps_2d)
    assert (out == _popcount(fps_2d)).all()
    assert out.tolist() == expect_2d


def test_cpp_centroid() -> None:
    fps = make_fake_fingerprints(10, seed=17408390758220920002, pack=False)
    centroid = _calc_centroid_packed_u8_from_u64(fps.sum(0), len(fps))
    expect_centroid = calc_centroid(fps.sum(0), len(fps), pack=True)
    assert (centroid == expect_centroid).all()


def test_cpp_unpacking() -> None:
    fps = make_fake_fingerprints(
        10, seed=17408390758220920002, pack=True, n_features=32, dtype=np.uint8
    )
    expect_unpacked = unpack_fingerprints(fps)
    unpacked = _unpack_fingerprints_2d(fps)
    assert (expect_unpacked == unpacked).all()


def test_jt_sim_packed() -> None:
    fps = make_fake_fingerprints(10, seed=17408390758220920002)
    first = fps[0]
    out = jt_sim_packed(fps, first)
    expect = np.array(
        [
            1.0,
            0.050833333333333,
            0.234522942461763,
            0.400854179377669,
            0.128980891719745,
            0.130030959752322,
            0.411522633744856,
            0.411104548139398,
            0.309090909090909,
            0.246826516220028,
        ],
        dtype=np.float64,
    )
    assert np.isclose(out, expect).all()


def test_jt_isim() -> None:
    fps = make_fake_fingerprints(100, seed=17408390758220920002, pack=False)
    c_total = fps.sum(0)
    c_objects = len(fps)
    s = jt_isim(c_total, c_objects)
    assert s == 0.21824334501491158


def test_jt_isim_disjoint() -> None:
    fps = make_fake_fingerprints(1, seed=17408390758220920002, pack=False)
    disjoint = (~fps.astype(np.bool)).view(np.uint8)
    fps = np.concatenate((fps, disjoint))
    c_total = fps.sum(0)
    c_objects = len(fps)
    s = jt_isim(c_total, c_objects)
    assert s == 0.0

    fps = np.eye(2048, 2048, dtype=np.uint8)
    c_total = fps.sum(0)
    c_objects = len(fps)
    s = jt_isim(c_total, c_objects)
    assert s == 0.0


def test_jt_isim_homogeneous() -> None:
    fps = np.zeros((100, 2048), dtype=np.uint8)
    c_total = fps.sum(0)
    c_objects = len(fps)
    s = jt_isim(c_total, c_objects)
    assert s == 1.0

    fps = np.ones((100, 2048), dtype=np.uint8)
    c_total = fps.sum(0)
    c_objects = len(fps)
    s = jt_isim(c_total, c_objects)
    assert s == 1.0


def test_jt_isim_single() -> None:
    fps = make_fake_fingerprints(1, seed=17408390758220920002, pack=False)
    c_total = fps.sum(0)
    c_objects = len(fps)
    with pytest.warns(RuntimeWarning):
        _ = jt_isim(c_total, c_objects)
