from array import array
import pickle

import numpy as np

from bblean import BitBirch


def test_compact_stream_matches_default_and_checkpoint():
    rng = np.random.default_rng(783)
    # Correlated fingerprints exercise both merges and multiple tree splits.
    prototypes = rng.integers(0, 256, (80, 32), dtype=np.uint8)
    x = prototypes[rng.integers(0, len(prototypes), 4000)].copy()
    x[:, 0] ^= rng.integers(0, 8, len(x), dtype=np.uint8)
    ordinary = BitBirch(threshold=.65, branching_factor=8).fit(x)
    compact = BitBirch(threshold=.65, branching_factor=8, compact_memberships=True)
    for chunk in np.array_split(x, 13):
        compact.fit(chunk)
    restored = pickle.loads(pickle.dumps(compact))
    expected = [list(c) for c in ordinary.get_centroids_mol_ids()['mol_ids']]
    actual = [list(c) for c in restored.get_centroids_mol_ids()['mol_ids']]
    assert actual == expected
    assert restored.num_fitted_fps == len(x)
    def check(node):
        for cluster in node._subclusters:
            assert isinstance(cluster.mol_indices, array)
            assert cluster.mol_indices.typecode == 'Q'
            if cluster.child:
                check(cluster.child)
    check(restored._root)
    restored.fit(x[:10])
    ids = [i for c in restored.get_centroids_mol_ids()['mol_ids'] for i in c]
    assert sorted(ids) == list(range(len(x) + 10))


def test_uint64_ids_above_uint32():
    x = np.full((20, 32), 17, dtype=np.uint8)
    tree = BitBirch(compact_memberships=True).fit(x, reinsert_indices=range(2**32, 2**32 + len(x)))
    assert sorted(i for c in tree.get_centroids_mol_ids()['mol_ids'] for i in c) == list(range(2**32, 2**32 + len(x)))
