import os
import numpy as np
import pytest

from inline_snapshot import snapshot

# TODO: Fix the tests with pytest-subtests so that both the _py_similarity and the
# _cpp_similarity are tested independently
import bblean._py_similarity as pysim

try:
    import bblean._cpp_similarity as csim

    CSIM_AVAIL = True
except ImportError:
    if os.getenv("BITBIRCH_CANT_SKIP_CPP_TESTS"):
        raise
    CSIM_AVAIL = False
from bblean.fingerprints import make_fake_fingerprints, unpack_fingerprints
from bblean.similarity import centroid_from_sum, centroid as centroid_from_fps
import bblean.similarity as gensim


def test_jt_most_dissimilar_packed() -> None:
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
    ) = pysim.jt_most_dissimilar_packed(fps)
    assert idx1 == expect_idx1
    assert idx2 == expect_idx2
    assert np.isclose(sims1, expect_sims1).all()
    assert np.isclose(sims2, expect_sims2).all()
    if CSIM_AVAIL:
        (
            idx1,
            idx2,
            sims1,
            sims2,
        ) = csim.jt_most_dissimilar_packed(fps)
        assert idx1 == expect_idx1
        assert idx2 == expect_idx2
        assert np.isclose(sims1, expect_sims1).all()
        assert np.isclose(sims2, expect_sims2).all()


@pytest.mark.skipif(not CSIM_AVAIL, reason="Requires C++ extensions")
def test_popcount_1d() -> None:
    fps_1d = make_fake_fingerprints(1, seed=17408390758220920002).reshape(-1)
    expect_1d = 1137
    out = csim._popcount_1d(fps_1d)
    assert (out == pysim._popcount(fps_1d)).all()
    assert out == expect_1d


@pytest.mark.skipif(not CSIM_AVAIL, reason="Requires C++ extensions")
def test_popcount_2d() -> None:
    fps_2d = make_fake_fingerprints(10, seed=17408390758220920002)
    expect_2d = [1137, 124, 558, 1159, 281, 323, 1264, 1252, 879, 631]
    out = csim._popcount_2d(fps_2d)
    assert (out == pysim._popcount(fps_2d)).all()
    assert out.tolist() == expect_2d


@pytest.mark.skipif(not CSIM_AVAIL, reason="Requires C++ extensions")
def test_cpp_centroid() -> None:
    fps = make_fake_fingerprints(10, seed=17408390758220920002, pack=False)
    _sum = fps.sum(0)
    num = len(fps)
    centroid = csim.centroid_from_sum(_sum, num, pack=False)
    expect_centroid = centroid_from_sum(_sum, num, pack=False)
    assert (centroid == expect_centroid).all()
    centroid = csim.centroid_from_sum(_sum, num, pack=True)
    expect_centroid = centroid_from_sum(_sum, num, pack=True)
    assert (centroid == expect_centroid).all()
    centroid = centroid_from_fps(fps, input_is_packed=False)
    assert (centroid == expect_centroid).all()


# TODO: Move this test
@pytest.mark.skipif(not CSIM_AVAIL, reason="Requires C++ extensions")
def test_cpp_unpacking() -> None:
    for seed in [
        17493821988544178123,
        4478748046060904849,
        4727712347772598054,
        15490537310413187550,
    ]:
        fps = make_fake_fingerprints(
            10, seed=17408390758220920002, pack=True, n_features=2024, dtype=np.uint8
        )
        expect_unpacked = unpack_fingerprints(fps)
        unpacked = csim._nochecks_unpack_fingerprints_2d(fps)
        assert (expect_unpacked == unpacked).all()
        unpacked = csim.unpack_fingerprints(fps)
        assert (expect_unpacked == unpacked).all()
        for fp in fps:
            expect_unpacked = unpack_fingerprints(fp)  # type: ignore
            unpacked = csim._nochecks_unpack_fingerprints_1d(fp)  # type: ignore
            assert (expect_unpacked == unpacked).all()
            unpacked = csim.unpack_fingerprints(fp)  # type: ignore
            assert (expect_unpacked == unpacked).all()


def test_jt_sim_packed() -> None:
    fps = make_fake_fingerprints(10, seed=17408390758220920002)
    first = fps[0]
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
    out = pysim._jt_sim_arr_vec_packed(fps, first)
    assert np.isclose(out, expect).all()

    if CSIM_AVAIL:
        out = csim._jt_sim_arr_vec_packed(fps, first)
        assert np.isclose(out, expect).all()

    # General
    out = gensim.jt_sim_packed(fps, first)
    assert np.isclose(out, expect).all()

    out = gensim.jt_sim_packed(first, fps)
    assert np.isclose(out, expect).all()

    out = gensim.jt_sim_packed(fps[0], first)
    assert out == expect[0]


def test_jt_isim_from_sum() -> None:
    fps = make_fake_fingerprints(100, seed=17408390758220920002, pack=False)
    c_total = fps.sum(0)
    c_objects = len(fps)
    s = pysim.jt_isim_from_sum(c_total, c_objects)
    assert s == 0.21824334501491158
    if CSIM_AVAIL:
        s = csim.jt_isim_from_sum(c_total, c_objects)
        assert s == 0.21824334501491158


def test_jt_isim() -> None:
    fps = make_fake_fingerprints(100, seed=17408390758220920002, pack=False)
    fps_packed = make_fake_fingerprints(100, seed=17408390758220920002, pack=True)
    s = pysim.jt_isim_unpacked(fps)
    assert s == 0.21824334501491158
    s2 = pysim.jt_isim_packed(fps_packed)
    assert s2 == 0.21824334501491158
    if CSIM_AVAIL:
        from bblean.similarity import (
            jt_isim_unpacked as jt_isim_unpacked_wrap,
            jt_isim_packed as jt_isim_packed_wrap,
        )

        s = csim.jt_isim_unpacked_u8(fps)
        assert s == 0.21824334501491158
        s2 = csim.jt_isim_packed_u8(fps_packed)
        assert s2 == 0.21824334501491158
        s = jt_isim_unpacked_wrap(fps)
        assert s == 0.21824334501491158
        s2 = jt_isim_packed_wrap(fps_packed)
        assert s2 == 0.21824334501491158


def test_jt_isim_from_sum_disjoint() -> None:
    fps = make_fake_fingerprints(1, seed=17408390758220920002, pack=False)
    disjoint = (~fps.astype(np.bool)).view(np.uint8)
    fps = np.concatenate((fps, disjoint))
    c_total = fps.sum(0)
    c_objects = len(fps)
    assert pysim.jt_isim_from_sum(c_total, c_objects) == 0
    if CSIM_AVAIL:
        assert csim.jt_isim_from_sum(c_total, c_objects) == 0

    fps = np.eye(2048, 2048, dtype=np.uint8)
    c_total = fps.sum(0)
    c_objects = len(fps)
    assert pysim.jt_isim_from_sum(c_total, c_objects) == 0
    if CSIM_AVAIL:
        assert csim.jt_isim_from_sum(c_total, c_objects) == 0


def test_jt_isim_from_sum_homogeneous() -> None:
    fps = np.zeros((100, 2048), dtype=np.uint8)
    c_total = fps.sum(0)
    c_objects = len(fps)
    assert pysim.jt_isim_from_sum(c_total, c_objects) == 1.0
    if CSIM_AVAIL:
        assert csim.jt_isim_from_sum(c_total, c_objects) == 1.0

    fps = np.ones((100, 2048), dtype=np.uint8)
    c_total = fps.sum(0)
    c_objects = len(fps)
    assert pysim.jt_isim_from_sum(c_total, c_objects) == 1.0
    if CSIM_AVAIL:
        assert csim.jt_isim_from_sum(c_total, c_objects) == 1.0


def test_jt_isim_from_sum_single() -> None:
    fps = make_fake_fingerprints(1, seed=17408390758220920002, pack=False)
    c_total = fps.sum(0)
    c_objects = len(fps)
    with pytest.warns(RuntimeWarning):
        _ = pysim.jt_isim_from_sum(c_total, c_objects)
    if CSIM_AVAIL:
        with pytest.warns(RuntimeWarning):
            _ = csim.jt_isim_from_sum(c_total, c_objects)


def test_jt_compl_isim() -> None:
    fps = make_fake_fingerprints(2, seed=17408390758220920002, pack=False)
    with pytest.warns(RuntimeWarning):
        _ = pysim.jt_compl_isim(fps)
    if CSIM_AVAIL:
        with pytest.warns(RuntimeWarning):
            _ = csim.jt_compl_isim(fps)

    fps = make_fake_fingerprints(10, seed=17408390758220920002, pack=False)
    output = pysim.jt_compl_isim(fps).tolist()
    assert output == snapshot(
        [
            0.20256457907452147,
            0.24748926949201983,
            0.22550084742079876,
            0.2002884861456855,
            0.23889840001690868,
            0.2364222674813306,
            0.1986207548061027,
            0.19904732709222533,
            0.21303348506016495,
            0.2225069540267648,
        ]
    )
    if CSIM_AVAIL:
        assert csim.jt_compl_isim(fps).tolist() == output
    assert (
        pysim.jt_compl_isim(np.zeros((10, 512), dtype=np.uint8))
        == np.ones(10, dtype=np.float64)
    ).all()
    if CSIM_AVAIL:
        assert (
            csim.jt_compl_isim(np.zeros((10, 512), dtype=np.uint8))
            == np.ones(10, dtype=np.float64)
        ).all()


def test_jt_isim_medoid() -> None:
    fps = make_fake_fingerprints(
        30, n_features=8, seed=17408390758220920002, pack=False
    )
    idx, m = pysim.jt_isim_medoid(fps)
    assert idx == snapshot(26)
    assert m.tolist() == snapshot([1, 1, 0, 1, 1, 1, 1, 1])
