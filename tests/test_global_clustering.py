import numpy as np
from bblean.bitbirch import BitBirch
from bblean.fingerprints import make_fake_fingerprints, unpack_fingerprints
from inline_snapshot import snapshot


def test_random_fps_consistency() -> None:
    fps = make_fake_fingerprints(
        3000, n_features=2048, seed=12620509540149709235, pack=True
    )
    tree = BitBirch(branching_factor=50, threshold=0.65, merge_criterion="diameter")
    tree.fit(fps, n_features=2048)
    output_cent = tree.get_centroids()
    output_med = tree.get_medoids(fps)
    assert [c.tolist()[:5] for c in output_cent[:5]] == snapshot(
        [
            [255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255],
            [255, 255, 253, 255, 255],
            [255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255],
        ]
    )
    assert output_med[:5, :5].tolist() == snapshot(
        [
            [239, 247, 247, 255, 214],
            [255, 255, 255, 255, 191],
            [255, 159, 223, 255, 219],
            [255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255],
        ]
    )
    tree.global_clustering(
        20,
        method="kmeans",
        n_init=1,
        init=unpack_fingerprints(np.vstack(output_cent))[::2][:20],
    )
    output_mol_ids = tree.get_cluster_mol_ids(global_clusters=True)
    output_med = tree.get_medoids(fps, global_clusters=True)
    assert [o[:5] for o in output_mol_ids[:5]] == snapshot(
        [
            [2195, 2196, 2378, 2440, 2443],
            [1958, 1800, 1928, 1948, 1959],
            [1943, 2823, 2907, 2974, 2988],
            [2216, 2272, 2632, 2651, 2843],
            [1844, 1921, 2866, 2854, 2731],
        ]
    )
    assert output_med[:5, :5].tolist() == snapshot(
        [
            [255, 255, 255, 255, 255],
            [190, 127, 215, 234, 114],
            [174, 83, 47, 191, 71],
            [22, 64, 107, 90, 159],
            [248, 199, 94, 119, 88],
        ]
    )
