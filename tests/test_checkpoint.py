import io
import pickle
import sys

import numpy as np
import pytest

from bblean import BitBirch
from bblean.checkpoint import load_checkpoint, save_checkpoint


def members(tree):
    return [list(c) for c in tree.get_centroids_mol_ids()['mol_ids']]


@pytest.mark.parametrize('compact', [False, True])
def test_long_leaf_chain_roundtrip_and_resume(tmp_path, compact):
    rng = np.random.default_rng(79)
    x = rng.integers(0, 256, (4000, 32), dtype=np.uint8)
    tree = BitBirch(threshold=.85, branching_factor=4, compact_memberships=compact).fit(x)
    before = sys.getrecursionlimit()
    with pytest.raises(RecursionError):
        # The Python pickler consistently exposes graph recursion across Python
        # versions; C-pickler recursion limits changed in Python 3.12.
        pickle._Pickler(io.BytesIO(), protocol=5).dump(tree)
    path = tmp_path/'tree.bb'
    info = save_checkpoint(tree, path)
    restored = load_checkpoint(path)
    assert sys.getrecursionlimit() == before
    assert info['nodes'] > 1000
    assert members(restored) == members(tree)
    np.testing.assert_array_equal(restored.get_centroids(), tree.get_centroids())
    assert restored.num_fitted_fps == tree.num_fitted_fps
    for leaf in restored._get_leaves():
        assert leaf._prev_leaf._next_leaf is leaf
        if leaf._next_leaf is not None:
            assert leaf._next_leaf._prev_leaf is leaf
    tree.fit(x[:30]); restored.fit(x[:30])
    assert members(restored) == members(tree)
    assert restored.num_fitted_fps == 4030


def test_empty_and_reject_truncation(tmp_path):
    path = tmp_path/'empty.bb'
    save_checkpoint(BitBirch(), path)
    tree = load_checkpoint(path)
    assert not tree.is_init
    tree.fit(np.ones((2, 256), dtype=np.uint8))
    assert tree.num_fitted_fps == 2
    path.write_bytes(path.read_bytes()[:-4])
    with pytest.raises(ValueError, match='Truncated'):
        load_checkpoint(path)


def test_failed_write_preserves_previous_checkpoint(tmp_path, monkeypatch):
    import bblean.checkpoint as module
    path = tmp_path/'tree.bb'
    tree = BitBirch().fit(np.ones((3, 32), dtype=np.uint8))
    save_checkpoint(tree, path)
    original = path.read_bytes()
    def fail(*args, **kwargs):
        raise OSError('injected checkpoint write failure')
    monkeypatch.setattr(module._NodePickler, 'dump', fail)
    with pytest.raises(OSError, match='injected'):
        save_checkpoint(tree, path)
    assert path.read_bytes() == original
    assert load_checkpoint(path).num_fitted_fps == 3
    assert list(tmp_path.iterdir()) == [path]
