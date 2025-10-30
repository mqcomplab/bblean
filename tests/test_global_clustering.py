import numpy as np
from bblean.bitbirch import BitBirch
from bblean.fingerprints import make_fake_fingerprints, unpack_fingerprints
from inline_snapshot import snapshot


def test_random_fps_consistency() -> None:
    fps = make_fake_fingerprints(3000, n_features=2048, seed=126205095409235, pack=True)
    tree = BitBirch(branching_factor=50, threshold=0.65, merge_criterion="diameter")
    tree.fit(fps, n_features=2048)
    output_cent = tree.get_centroids()
    output_med = tree.get_medoids(fps)
    assert [c.tolist()[:5] for c in output_cent[:5]] == snapshot(
        [
            [255, 255, 255, 255, 255],
            [255, 251, 255, 255, 255],
            [255, 239, 255, 247, 255],
            [255, 255, 191, 255, 255],
            [255, 127, 235, 127, 255],
        ]
    )
    assert output_med[:5, :5].tolist() == snapshot(
        [
            [255, 255, 126, 255, 111],
            [247, 255, 255, 255, 255],
            [191, 253, 191, 255, 255],
            [255, 255, 95, 255, 239],
            [235, 255, 123, 255, 255],
        ]
    )
    tree.global_clustering(
        20,
        method="kmeans",
        n_init=1,
        init=unpack_fingerprints(np.vstack(output_cent))[::2][:20],
        max_iter=10,
    )
    output_mol_ids = tree.get_cluster_mol_ids(global_clusters=True)
    output_med = tree.get_medoids(fps, global_clusters=True)
    assert [o[:5] for o in output_mol_ids[:5]] == snapshot(
        [
            [563, 568, 584, 585, 862],
            [1843, 1868, 1888, 1951, 1919],
            [57, 1855, 1844, 1848, 2023],
            [73, 1867, 2046, 2064, 2141],
            [53, 7, 18, 34, 46],
        ]
    )
    assert output_med[:5, :5].tolist() == snapshot(
        [
            [255, 255, 126, 255, 111],
            [135, 255, 78, 138, 230],
            [159, 189, 11, 228, 126],
            [74, 98, 71, 150, 111],
            [11, 158, 224, 132, 130],
        ]
    )
