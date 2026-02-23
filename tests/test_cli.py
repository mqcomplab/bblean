import pickle
import numpy as np
from pathlib import Path
import tempfile
from typer.testing import CliRunner

from bblean.cli import app
from bblean.fingerprints import make_fake_fingerprints


def test_umap() -> None:
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as d:
        Path
        dir = Path(d).resolve()
        fps = make_fake_fingerprints(
            50, n_features=512, seed=12620509540149709235, pack=True
        )
        np.save(dir / "fingerprints.npy", fps)
        out_dir = dir / "output"
        result = runner.invoke(
            app,
            [
                "run",
                str(dir),
                "-o",
                str(out_dir),
                "-b",
                "50",
                "-t",
                "0.3",
                "--no-monitor-mem",
            ],
        )
        assert result.exit_code == 0
        result = runner.invoke(
            app,
            [
                "plot-umap",
                str(out_dir),
                "-f",
                str(dir / "fingerprints.npy"),
                "--no-show",
                "--top",
                "1",
                "--no-verbose",
            ],
        )
    assert result.exit_code == 0


def test_pops() -> None:
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as d:
        Path
        dir = Path(d).resolve()
        fps = make_fake_fingerprints(
            250, n_features=512, seed=12620509540149709235, pack=True
        )
        np.save(dir / "fingerprints.npy", fps)
        out_dir = dir / "output"
        result = runner.invoke(
            app,
            [
                "run",
                str(dir),
                "-o",
                str(out_dir),
                "-b",
                "50",
                "-t",
                "0.3",
                "--no-monitor-mem",
            ],
        )
        assert result.exit_code == 0
        result = runner.invoke(
            app,
            [
                "plot-pops",
                str(out_dir),
                "-f",
                str(dir / "fingerprints.npy"),
                "--min-size",
                "1",
                "--top",
                "2",
                "--no-show",
            ],
        )
        assert result.exit_code == 0


def test_pca() -> None:
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as d:
        Path
        dir = Path(d).resolve()
        fps = make_fake_fingerprints(
            250, n_features=512, seed=12620509540149709235, pack=True
        )
        np.save(dir / "fingerprints.npy", fps)
        out_dir = dir / "output"
        result = runner.invoke(
            app,
            [
                "run",
                str(dir),
                "-o",
                str(out_dir),
                "-b",
                "50",
                "-t",
                "0.3",
                "--no-monitor-mem",
            ],
        )
        assert result.exit_code == 0
        result = runner.invoke(
            app,
            [
                "plot-pca",
                str(out_dir),
                "-f",
                str(dir / "fingerprints.npy"),
                "--no-show",
                "--top",
                "5",
                "--no-verbose",
            ],
        )
    assert result.exit_code == 0


def test_tsne() -> None:
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as d:
        Path
        dir = Path(d).resolve()
        fps = make_fake_fingerprints(
            250, n_features=512, seed=12620509540149709235, pack=True
        )
        np.save(dir / "fingerprints.npy", fps)
        out_dir = dir / "output"
        result = runner.invoke(
            app,
            [
                "run",
                str(dir),
                "-o",
                str(out_dir),
                "-b",
                "50",
                "-t",
                "0.3",
                "--no-monitor-mem",
            ],
        )
        assert result.exit_code == 0
        result = runner.invoke(
            app,
            [
                "plot-tsne",
                str(out_dir),
                "-f",
                str(dir / "fingerprints.npy"),
                "--no-show",
                "--top",
                "2",
                "--no-verbose",
            ],
        )
    assert result.exit_code == 0


def test_summary() -> None:
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as d:
        Path
        dir = Path(d).resolve()
        fps = make_fake_fingerprints(
            500, n_features=2048, seed=12620509540149709235, pack=True
        )
        np.save(dir / "fingerprints.npy", fps)
        out_dir = dir / "output"
        result = runner.invoke(
            app,
            [
                "run",
                str(dir),
                "-o",
                str(out_dir),
                "-b",
                "50",
                "-t",
                "0.1",
                "--no-monitor-mem",
            ],
        )
        assert result.exit_code == 0
        result = runner.invoke(
            app,
            [
                "plot-summary",
                str(out_dir),
                "-f",
                str(dir / "fingerprints.npy"),
                "--no-show",
                "--no-verbose",
            ],
        )
    assert result.exit_code == 0


def test_fps_from_smiles() -> None:
    runner = CliRunner()
    smiles_path = Path(__file__).parent / "chembl-sample-3k.smi"
    with tempfile.TemporaryDirectory() as d:
        dir = Path(d).resolve()
        out_dir = dir / "output"
        result = runner.invoke(
            app, ["fps-from-smiles", str(smiles_path), "-o", str(out_dir)]
        )
        assert result.exit_code == 0
        file = list(out_dir.glob("*.npy"))[0]
        fps_raw = np.load(file).reshape(-1)
    nonzero = fps_raw.nonzero()[0].reshape(-1)
    actual = fps_raw[nonzero][:19].tolist()
    expect = [4, 128, 2, 16, 8, 16, 4, 16, 128, 16, 1, 128, 1, 64, 1, 1, 128, 32, 32]
    assert actual == expect


# NOTE: This is an acceptance test only
def test_fps_from_smiles_invalid() -> None:
    runner = CliRunner()
    smiles_path = Path(__file__).parent / "chembl-sample-bad.smi"
    with tempfile.TemporaryDirectory() as d:
        dir = Path(d).resolve()
        out_dir = dir / "output"
        result = runner.invoke(
            app,
            [
                "fps-from-smiles",
                str(smiles_path),
                "-o",
                str(out_dir),
                "--sanitize",
                "minimal",
                "--skip-invalid",
            ],
        )
        assert result.exit_code == 0
        file = list(out_dir.glob("*fps*.npy"))[0]
        fps_raw = np.load(file).reshape(-1)
    nonzero = fps_raw.nonzero()[0].reshape(-1)
    actual = fps_raw[nonzero][:19].tolist()
    expect = [2, 4, 32, 1, 2, 128, 4, 128, 32, 32, 80, 128, 64, 128, 1, 16, 64, 4, 16]
    assert actual == expect


def test_info() -> None:
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as d:
        dir = Path(d).resolve()
        fps = make_fake_fingerprints(
            3000, n_features=2048, seed=12620509540149709235, pack=True
        )
        np.save(dir / "fingerprints.npy", fps)
        result = runner.invoke(app, ["fps-info", str(dir)])
    if isinstance(result.output, bytes):
        out = result.output.decode("utf-8")
    else:
        out = result.output
    assert result.exit_code == 0
    assert "Valid fingerprint file" in out
    assert "256" in out
    assert "uint8" in out


def test_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "CLI tool for serial or parallel fast clustering" in result.output


def test_multiround() -> None:
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as d:
        dir = Path(d).resolve()
        for seed in range(1, 21, 2):
            fps = make_fake_fingerprints(100, seed=seed)
            np.save(dir / f"fps.{str(seed).zfill(4)}.npy", fps)
        out_dir = dir / "output"
        out_dir.mkdir()
        result = runner.invoke(
            app,
            [
                "multiround",
                str(dir),
                "-o",
                str(out_dir),
                "-t",
                "0.65",
                "--bin-size",
                "2",
                "--ps",
                "1",
                "--no-verbose",
                "--set-mid-merge",
                "tolerance-legacy",
                "--no-monitor-mem",
            ],
        )
        with open(out_dir / "clusters.pkl", mode="rb") as f:
            obj = pickle.load(f)
    assert result.exit_code == 0
    assert EXPECT_MULTIROUND_CLUSTERS == obj[: len(EXPECT_MULTIROUND_CLUSTERS)]


def test_run() -> None:
    runner = CliRunner()
    ids_expect = [
        [2195, 2196, 2378, 2440, 2443, 2454, 2463, 2464, 2465, 2467, 2527, 2544],
        [199, 228, 255, 270, 273, 438, 457, 458, 461, 470, 477, 496],
        [700, 728, 773, 798, 825, 891, 919, 962, 963, 968, 998],
        [1448, 1567, 1590, 1606, 1612, 1637, 1640, 1648, 1686, 1694],
        [1059, 1065, 1072, 1077, 1154, 1194, 1301],
        [1779, 1802, 1807, 1828, 1856, 1864],
        [2826, 2896, 2970, 2973, 2975],
        [1986, 2107, 2139, 2141],
        [1933, 1949],
        [2233, 2294],
        [1551, 1552],
        [1219, 1226],
        [614, 637],
    ]
    with tempfile.TemporaryDirectory() as d:
        Path
        dir = Path(d).resolve()
        fps = make_fake_fingerprints(
            3000, n_features=2048, seed=12620509540149709235, pack=True
        )
        np.save(dir / "fingerprints.npy", fps)
        out_dir = dir / "output"
        result = runner.invoke(
            app,
            [
                "run",
                str(dir),
                "-o",
                str(out_dir),
                "-b",
                "50",
                "-t",
                "0.65",
                "--no-monitor-mem",
            ],
        )
        with open(out_dir / "clusters.pkl", mode="rb") as f:
            obj = pickle.load(f)
        assert result.exit_code == 0
        assert obj[:13] == ids_expect


EXPECT_MULTIROUND_CLUSTERS = [
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
