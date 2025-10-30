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
    output_mol_ids = tree.get_cluster_mol_ids(global_clusters=True, sort=False)
    output_med = tree.get_medoids(fps, global_clusters=True, sort=False)
    assert [o[:5] for o in output_mol_ids[:5]] == snapshot(
        [
            [16, 1023, 1793, 2, 15],
            [1873, 1882, 1912, 1954, 1970],
            [12, 1877, 1861, 2068, 2012],
            [1560, 1901, 2065, 2037, 2396],
            [62, 73, 75, 87, 121],
        ]
    )
    assert output_med[:5, :5].tolist() == snapshot(
        [
            [255, 127, 252, 111, 223],
            [255, 255, 95, 255, 239],
            [123, 239, 238, 135, 126],
            [223, 14, 207, 187, 104],
            [255, 255, 255, 247, 255],
        ]
    )
