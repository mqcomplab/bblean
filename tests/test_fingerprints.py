from pathlib import Path
import tempfile
import numpy as np
import itertools
from bblean.fingerprints import _get_fingerprints_from_file_seq, make_fake_fingerprints


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
