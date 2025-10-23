import numpy as np
import pytest

from bblean.similarity import jt_stratified_sampling
from bblean.fingerprints import make_fake_fingerprints
from inline_snapshot import snapshot


def test_stratified() -> None:
    fps = make_fake_fingerprints(10, seed=1345)
    out = jt_stratified_sampling(fps, 3)
    assert out.tolist() == snapshot([0, 4, 2])

    assert jt_stratified_sampling(fps, 0).size == 0
    assert (
        jt_stratified_sampling(fps, 10) == np.arange(10, dtype=np.int64)
    ).all()
    with pytest.raises(ValueError):
        _ = jt_stratified_sampling(fps, 11)
