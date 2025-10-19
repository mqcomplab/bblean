from numpy.typing import NDArray
import numpy as np
import pytest

from bblean.fingerprints import make_fake_fingerprints
from bblean.metrics import jt_dbi, jt_isim_chi, jt_isim_dunn
from legacy_metrics import legacy_dunn, legacy_chi, legacy_dbi  # type: ignore


@pytest.fixture
def clusters() -> list[NDArray[np.uint8]]:
    fps = make_fake_fingerprints(
        50, n_features=32, seed=12620509540149709235, pack=True
    )
    clusters = []
    for i in range(5):
        clusters.append(fps[i : i + 10])
    return clusters


@pytest.fixture
def unpacked_clusters() -> list[NDArray[np.uint8]]:
    fps = make_fake_fingerprints(
        50, n_features=32, seed=12620509540149709235, pack=False
    )
    clusters = []
    for i in range(5):
        clusters.append(fps[i : i + 10])
    return clusters


def test_metrics(
    clusters: list[NDArray[np.uint8]], unpacked_clusters: list[NDArray[np.uint8]]
) -> None:
    dbi = jt_dbi(clusters)
    dunn = jt_isim_dunn(clusters)
    chi = jt_isim_chi(clusters)

    expect_dbi = legacy_dbi(unpacked_clusters)
    expect_dunn = legacy_dunn(unpacked_clusters)
    expect_chi = legacy_chi(unpacked_clusters)
    assert dbi == expect_dbi
    assert dunn == expect_dunn
    assert chi == expect_chi
    assert dbi == 0.5183197174528508
    assert dunn == 2.1591579714616436
    assert chi == 4.411591622357663
