from numpy.typing import NDArray
import numpy as np
import pytest
from inline_snapshot import snapshot

from bblean.sklearn import BitBirch, UnpackedBitBirch
from bblean.fingerprints import make_fake_fingerprints


@pytest.fixture
def fps() -> NDArray[np.integer]:
    return make_fake_fingerprints(
        100, n_features=2048, seed=12620509540149709235, pack=True
    )


@pytest.fixture
def unpacked_fps() -> NDArray[np.integer]:
    return make_fake_fingerprints(
        100, n_features=2048, seed=12620509540149709235, pack=False
    )


def test_packed_bitbirch_fit(fps: NDArray[np.integer]) -> None:
    tree = BitBirch(branching_factor=50, threshold=0.1, merge_criterion="diameter")
    tree.fit(fps)
    assert tree.labels_.tolist()[:5] == snapshot([1, 1, 1, 1, 1])
    out = tree.transform(fps)
    assert out.tolist()[:5] == snapshot(
        [
            [0.9953650057937428],
            [0.9955357142857143],
            [0.9965928449744463],
            [0.9977477477477478],
            [0.9977553310886644],
        ]
    )
    tree.set_output(transform="pandas")
    out_df = tree.transform(fps)
    assert out_df.values.tolist()[:5] == snapshot(
        [
            [0.9953650057937428],
            [0.9955357142857143],
            [0.9965928449744463],
            [0.9977477477477478],
            [0.9977553310886644],
        ]
    )
    assert tree.get_params() == snapshot(
        {
            "branching_factor": 50,
            "compute_labels": True,
            "merge_criterion": "diameter",
            "threshold": 0.1,
            "tolerance": None,
        }
    )
    tree.set_params(threshold=0.65)
    assert tree.threshold == 0.65
    out_feats = tree.get_feature_names_out()
    for i, f in enumerate(out_feats):
        assert f == f"bitbirch{i}"


def test_packed_bitbirch_fit_transform(fps: NDArray[np.integer]) -> None:
    tree = BitBirch(branching_factor=50, threshold=0.1, merge_criterion="diameter")
    out = tree.fit_transform(fps)
    assert out.tolist()[:5] == snapshot(
        [
            [0.9953650057937428],
            [0.9955357142857143],
            [0.9965928449744463],
            [0.9977477477477478],
            [0.9977553310886644],
        ]
    )


def test_packed_bitbirch_predict(fps: NDArray[np.integer]) -> None:
    tree = BitBirch(branching_factor=50, threshold=0.1, merge_criterion="diameter")
    out = tree.fit_predict(fps)
    assert out.tolist()[:5] == snapshot([1, 1, 1, 1, 1])
    fps2 = make_fake_fingerprints(
        500, n_features=2048, seed=16297711681646304030, pack=True
    )
    out = tree.predict(fps2)
    assert out.tolist()[:5] == snapshot([1, 1, 1, 1, 1])
    tree.partial_fit(fps2)
    assert tree.labels_.tolist()[:5] == snapshot([1, 1, 1, 1, 1])


def test_unpacked_bitbirch_fit(unpacked_fps: NDArray[np.integer]) -> None:
    tree = UnpackedBitBirch(
        branching_factor=50, threshold=0.1, merge_criterion="diameter"
    )
    tree.fit(unpacked_fps)
    assert tree.labels_.tolist()[:5] == snapshot([1, 1, 1, 1, 1])
    out = tree.transform(unpacked_fps)
    assert out.tolist()[:5] == snapshot(
        [
            [0.9953650057937428],
            [0.9955357142857143],
            [0.9965928449744463],
            [0.9977477477477478],
            [0.9977553310886644],
        ]
    )
    tree.set_output(transform="pandas")
    out_df = tree.transform(unpacked_fps)
    assert out_df.values.tolist()[:5] == snapshot(
        [
            [0.9953650057937428],
            [0.9955357142857143],
            [0.9965928449744463],
            [0.9977477477477478],
            [0.9977553310886644],
        ]
    )
    assert tree.get_params() == snapshot(
        {
            "branching_factor": 50,
            "compute_labels": True,
            "merge_criterion": "diameter",
            "threshold": 0.1,
            "tolerance": None,
        }
    )
    tree.set_params(threshold=0.65)
    assert tree.threshold == 0.65
    out_feats = tree.get_feature_names_out()
    for i, f in enumerate(out_feats):
        assert f == f"unpackedbitbirch{i}"


def test_unpacked_bitbirch_fit_transform(unpacked_fps: NDArray[np.integer]) -> None:
    tree = UnpackedBitBirch(
        branching_factor=50, threshold=0.1, merge_criterion="diameter"
    )
    out = tree.fit_transform(unpacked_fps)
    assert out.tolist()[:5] == snapshot(
        [
            [0.9953650057937428],
            [0.9955357142857143],
            [0.9965928449744463],
            [0.9977477477477478],
            [0.9977553310886644],
        ]
    )


def test_unpacked_bitbirch_predict(unpacked_fps: NDArray[np.integer]) -> None:
    tree = UnpackedBitBirch(
        branching_factor=50, threshold=0.1, merge_criterion="diameter"
    )
    out = tree.fit_predict(unpacked_fps)
    assert out.tolist()[:5] == snapshot([1, 1, 1, 1, 1])
    fps2 = make_fake_fingerprints(
        500, n_features=2048, seed=16297711681646304030, pack=False
    )
    out = tree.predict(fps2)
    assert out.tolist()[:5] == snapshot([1, 1, 1, 1, 1])
    tree.partial_fit(fps2)
    assert tree.labels_.tolist()[:5] == snapshot([1, 1, 1, 1, 1])
