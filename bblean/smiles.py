r"""SMILES manipulation"""

import typing as tp
from numpy.typing import NDArray
import numpy as np
from pathlib import Path

from bblean.utils import batched

__all__ = [
    "load_smiles",
    "calc_num_smiles",
    "iter_smiles_from_paths",
]

SmilesPaths = tp.Iterable[Path | str] | Path | str


def load_smiles(smiles_paths: SmilesPaths, max_num: int = -1) -> NDArray[np.str_]:
    r"""Simple utility to load smiles from a ``*.smi`` file"""
    smiles = []
    for i, smi in enumerate(iter_smiles_from_paths(smiles_paths)):
        if i == max_num:
            break
        smiles.append(smi)
    return np.asarray(smiles)


def calc_num_smiles(smiles_paths: SmilesPaths) -> int:
    r"""Get the total number of smiles in a sequene of paths"""
    return sum(1 for _ in iter_smiles_from_paths(smiles_paths))


def iter_smiles_from_paths(
    smiles_paths: SmilesPaths,
) -> tp.Iterator[str]:
    r"""Iterate over smiles in a sequence of smiles paths"""
    if isinstance(smiles_paths, (Path, str)):
        smiles_paths = [smiles_paths]
    for smi_path in smiles_paths:
        with open(smi_path, mode="rt", encoding="utf-8") as f:
            for smi in f:
                yield smi


def _iter_ranges_and_smiles_batches(
    smiles_paths: SmilesPaths,
    num_per_batch: int,
) -> tp.Iterable[tuple[tuple[int, int], tuple[str, ...]]]:
    start_idx = 0
    for batch in batched(iter_smiles_from_paths(smiles_paths), num_per_batch):
        size = len(batch)
        end_idx = start_idx + size
        yield (start_idx, end_idx), batch
        start_idx = end_idx


def _iter_idxs_and_smiles_batches(
    smiles_paths: SmilesPaths, num_per_batch: int
) -> tp.Iterable[tuple[int, tuple[str, ...]]]:
    yield from enumerate(batched(iter_smiles_from_paths(smiles_paths), num_per_batch))
