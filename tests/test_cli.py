import pickle
import numpy as np
from pathlib import Path
import tempfile
from typer.testing import CliRunner

from bblean.cli import app
from bblean.fingerprints import make_fake_fingerprints


runner = CliRunner()


def test_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "CLI tool for serial or parallel fast clustering" in result.output


def test_run() -> None:
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
        result = runner.invoke(app, ["run", str(dir), "-o", str(out_dir), "-b", "50"])
        with open(out_dir / "clusters.pkl", mode="rb") as f:
            obj = pickle.load(f)
        assert result.exit_code == 0
        assert obj[:13] == ids_expect
        assert "Running single-round, serial" in result.output
