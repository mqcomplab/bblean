from pathlib import Path
import tempfile
import numpy as np
import itertools
from bblean.fingerprints import (
    _get_fingerprints_from_file_seq,
    make_fake_fingerprints,
    fps_from_smiles,
)
from bblean.smiles import load_smiles


def test_fps_from_smiles() -> None:
    smiles = load_smiles(Path(__file__).parent / "chembl-sample-3k.smi")
    fps_raw = fps_from_smiles(smiles).reshape(-1)
    nonzero = fps_raw.nonzero()[0].reshape(-1)
    actual = fps_raw[nonzero][:19].tolist()
    expect = [4, 128, 2, 16, 8, 16, 4, 16, 128, 16, 1, 128, 1, 64, 1, 1, 128, 32, 32]
    assert actual == expect


# NOTE: This is an acceptance test only
def test_fps_from_smiles_invalid() -> None:
    smiles = load_smiles(Path(__file__).parent / "chembl-sample-bad.smi")
    fps_raw, _ = fps_from_smiles(smiles, skip_invalid=True, sanitize="minimal")
    fps_raw = fps_raw.reshape(-1)
    nonzero = fps_raw.nonzero()[0].reshape(-1)
    actual = fps_raw[nonzero][:19].tolist()
    expect = [2, 4, 32, 1, 2, 128, 4, 128, 32, 32, 80, 128, 64, 128, 1, 16, 64, 4, 16]
    assert actual == expect


def test_fps_from_smiles_invalid_ids() -> None:
    smiles = load_smiles(Path(__file__).parent / "chembl-sample-bad.smi")
    fps_raw, invalid_idxs = fps_from_smiles(
        smiles, skip_invalid=True, sanitize="all"
    )
    actual = invalid_idxs.tolist()
    expect = [0, 1]
    assert actual == expect
    assert fps_raw.shape[0] == len(smiles) - len(expect)


def test_fingerprints_from_file_seq_empty() -> None:
    fps = make_fake_fingerprints(
        100, n_features=32, seed=12620509540149709235, pack=True
    )
    sections = [0, 10, 20, 30, 50, 80, 100]
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for i, (start, end) in enumerate(itertools.pairwise(sections)):
            path = Path(d).resolve() / f"fps.{str(i).zfill(3)}.npy"
            paths.append(path)
            np.save(path, fps[start:end])

        idxs: list[int] = []
        expect = fps[idxs]
        actual = _get_fingerprints_from_file_seq(paths, idxs)
        assert (expect == actual).all()


def test_fingerprints_from_file_seq() -> None:
    fps = make_fake_fingerprints(
        100, n_features=32, seed=12620509540149709235, pack=True
    )
    sections = [0, 10, 20, 30, 50, 80, 100]
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for i, (start, end) in enumerate(itertools.pairwise(sections)):
            path = Path(d).resolve() / f"fps.{str(i).zfill(3)}.npy"
            paths.append(path)
            np.save(path, fps[start:end])

        idxs = [0, 1, 2, 3, 4, 21, 22, 29, 50, 51, 55, 83]
        expect = fps[idxs]
        actual = _get_fingerprints_from_file_seq(paths, idxs)
        assert (expect == actual).all()


def test_fingerprints_from_file_seq_variation() -> None:
    fps = make_fake_fingerprints(
        100, n_features=32, seed=12620509540149709235, pack=True
    )
    sections = [0, 30, 80, 100]
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for i, (start, end) in enumerate(itertools.pairwise(sections)):
            path = Path(d).resolve() / f"fps.{str(i).zfill(3)}.npy"
            paths.append(path)
            np.save(path, fps[start:end])

        idxs = list(range(100))
        expect = fps[idxs]
        actual = _get_fingerprints_from_file_seq(paths, idxs)
        assert (expect == actual).all()
