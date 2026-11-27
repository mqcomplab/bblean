import pickle
from pathlib import Path
import numpy as np
from bblean.multiround import run_multiround_bitbirch
from bblean.fingerprints import make_fake_fingerprints


def test_multiround_bitbirch_parallel(tmp_path: Path) -> None:
    dir = tmp_path.resolve()
    for seed in range(1, 21, 2):
        fps = make_fake_fingerprints(100, seed=seed)
        np.save(dir / f"fps.{str(seed).zfill(4)}.npy", fps)
    out_dir = dir / "output"
    out_dir.mkdir()
    run_multiround_bitbirch(
        sorted(dir.glob("*.npy")),
        out_dir,
        num_initial_processes=10,
        bin_size=2,
        threshold=0.65,
        midsection_merge_criterion="tolerance-legacy",
    )
    with open(out_dir / "clusters.pkl", mode="rb") as f:
        clusters = pickle.load(f)
    assert EXPECT_CLUSTERS == clusters[: len(EXPECT_CLUSTERS)]


def test_multiround_bitbirch(tmp_path: Path) -> None:
    dir = tmp_path.resolve()
    for seed in range(1, 21, 2):
        fps = make_fake_fingerprints(100, seed=seed)
        np.save(dir / f"fps.{str(seed).zfill(4)}.npy", fps)
    out_dir = dir / "output"
    out_dir.mkdir()
    run_multiround_bitbirch(
        sorted(dir.glob("*.npy")),
        out_dir,
        num_initial_processes=1,
        bin_size=2,
        threshold=0.65,
        midsection_merge_criterion="tolerance-legacy",
    )
    with open(out_dir / "clusters.pkl", mode="rb") as f:
        clusters = pickle.load(f)
    assert EXPECT_CLUSTERS == clusters[: len(EXPECT_CLUSTERS)]


def test_multiround_bitbirch_unpacked_matches_packed(tmp_path: Path) -> None:
    packed_dir = tmp_path / "packed"
    unpacked_dir = tmp_path / "unpacked"
    packed_dir.mkdir()
    unpacked_dir.mkdir()
    for seed in (1, 3):
        unpacked = make_fake_fingerprints(100, n_features=64, seed=seed, pack=False)
        np.save(unpacked_dir / f"fps.{seed:04d}.npy", unpacked)
        np.save(packed_dir / f"fps.{seed:04d}.npy", np.packbits(unpacked, axis=-1))

    out_packed = tmp_path / "out_packed"
    out_unpacked = tmp_path / "out_unpacked"
    out_packed.mkdir()
    out_unpacked.mkdir()

    run_multiround_bitbirch(
        sorted(packed_dir.glob("*.npy")),
        out_packed,
        input_is_packed=True,
        num_initial_processes=1,
        bin_size=2,
        threshold=0.65,
        refinement_before_midsection="full",
        split_largest_after_each_midsection_round=True,
    )
    run_multiround_bitbirch(
        sorted(unpacked_dir.glob("*.npy")),
        out_unpacked,
        input_is_packed=False,
        num_initial_processes=1,
        bin_size=2,
        threshold=0.65,
        refinement_before_midsection="full",
        split_largest_after_each_midsection_round=True,
    )
    with open(out_packed / "clusters.pkl", mode="rb") as f:
        clusters_packed = pickle.load(f)
    with open(out_unpacked / "clusters.pkl", mode="rb") as f:
        clusters_unpacked = pickle.load(f)
    assert clusters_unpacked == clusters_packed


EXPECT_CLUSTERS = [
    [
        368,
        414,
        422,
        423,
        520,
        549,
        581,
        609,
        625,
        683,
        622,
        709,
        761,
        770,
        789,
        813,
        831,
        989,
    ],
    [23, 285, 209, 213, 276, 294, 316, 319, 358],
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9],
    [10],
    [11],
    [12],
    [13],
    [14],
    [15],
    [16],
    [17],
    [18],
    [19],
    [20],
    [21],
    [22],
    [24],
    [25],
    [26],
    [27],
    [28],
    [29],
    [30],
    [31],
    [32],
    [33],
    [34],
    [35],
    [36],
    [37],
    [38],
    [39],
    [40],
    [41],
    [42],
    [43],
    [44],
    [45],
    [46],
    [47],
    [48],
    [49],
    [50],
    [51],
    [52],
    [53],
    [54],
    [55],
    [56],
    [57],
    [58],
    [59],
    [60],
    [61],
    [62],
    [63],
    [64],
    [65],
    [66],
    [67],
    [68],
    [69],
    [70],
    [71],
    [72],
    [73],
    [74],
    [75],
    [76],
    [77],
    [78],
    [79],
    [80],
    [81],
    [82],
    [83],
    [84],
    [85],
    [86],
    [87],
    [88],
    [89],
    [90],
    [91],
    [92],
    [94],
    [95],
    [96],
    [97],
    [98],
    [99],
    [0],
    [101],
    [102],
    [103],
    [104],
    [105],
    [106],
    [107],
    [108],
    [109],
    [110],
    [111],
    [112],
    [113],
    [114],
    [115],
    [116],
    [117],
    [118],
    [119],
    [121],
    [122],
    [123],
    [124],
    [125],
    [126],
    [127],
    [128],
    [129],
    [130],
    [131],
]
