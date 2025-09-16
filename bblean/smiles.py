from numpy.typing import NDArray
import numpy as np
from pathlib import Path

__all__ = ["load_smiles"]


def load_smiles(path: Path | str, max_num: int = -1) -> NDArray[np.str_]:
    path = Path(path)
    smiles = []
    with open(path, mode="rt", encoding="utf-8") as f:
        for i, smi in enumerate(f):
            if i == max_num:
                break
            smiles.append(smi)
    return np.asarray(smiles)
