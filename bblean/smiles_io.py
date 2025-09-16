from numpy.typing import NDArray
import numpy as np
from pathlib import Path


def load_smiles(path: Path | str) -> NDArray[np.str_]:
    path = Path(path)
    with open(path, mode="rt", encoding="utf-8") as f:
        smiles = f.readlines()
    return np.asarray(smiles)
