from bblean.bitbirch import BitBirch
from bblean.fingerprints import make_fake_fingerprints
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
    tree.global_clustering(20, method="kmeans", random_state=42)
    output_mol_ids = tree.get_cluster_mol_ids(global_clusters=True)
    output_med = tree.get_medoids(fps, global_clusters=True)
    assert [o[:5] for o in output_mol_ids[:5]] == snapshot(
        [
            [2320, 2272, 2862, 2501, 2076],
            [2957, 2738, 2811, 2617, 1994],
            [1786, 1803, 1818, 1839, 1920],
            [2435, 2406, 2534, 2266, 1152],
            [1943, 2632, 2859, 2788, 2684],
        ]
    )
    assert output_med[:5, :5].tolist() == snapshot(
        [
            [13, 88, 95, 198, 249],
            [43, 246, 40, 153, 77],
            [120, 30, 49, 212, 24],
            [134, 93, 118, 102, 210],
            [247, 128, 177, 211, 238],
        ]
    )
