r"""SMILES manipulation"""

import typing as tp
from numpy.typing import NDArray
import numpy as np
from pathlib import Path

__all__ = ["load_smiles"]


def load_smiles(path: Path | str, max_num: int = -1) -> NDArray[np.str_]:
    r"""Simple utility to load smiles from a *.smi file"""
    path = Path(path)
    smiles = []
    with open(path, mode="rt", encoding="utf-8") as f:
        for i, smi in enumerate(f):
            if i == max_num:
                break
            smiles.append(smi)
    return np.asarray(smiles)


def iter_smiles_from_paths(smiles_paths: tp.Iterable[Path]) -> tp.Iterator[str]:
    for smi_path in smiles_paths:
        with open(smi_path, mode="rt", encoding="utf-8") as f:
            for smi in f:
                yield smi
