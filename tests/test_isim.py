import numpy as np
import pytest
from bblean.similarity import jt_isim
from bblean.fingerprints import make_fake_fingerprints


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
