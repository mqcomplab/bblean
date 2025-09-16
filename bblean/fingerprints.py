from numpy.typing import NDArray, DTypeLike
import numpy as np
import typing as tp

from rdkit.Chem import rdFingerprintGenerator, DataStructs, MolFromSmiles

from bblean.packing import pack_fingerprints


def fps_from_smiles(
    smiles: tp.Iterable[str],
    kind: str = "rdkit",
    n_features: int = 2048,
    dtype: DTypeLike = np.uint8,
    pack: bool = True,
) -> NDArray[np.uint8]:
    if isinstance(smiles, str):
        smiles = [smiles]

    if pack and not (np.dtype(dtype) == np.dtype(np.uint8)):
        raise ValueError("Packing only supported for uint8 dtype")

    if kind == "rdkit":
        fpg = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=n_features)
    elif kind == "ecfp4":
        fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_features)
    elif kind == "ecfp6":
        fpg = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=n_features)
    mols = []
    for smi in smiles:
        mol = MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Could not parse smiles {smi}")
        mols.append(mol)

    fps = np.empty((len(mols), n_features), dtype=dtype)
    for i, fp in enumerate(fpg.GetFingerprints(mols)):
        DataStructs.ConvertToNumpyArray(fp, fps[i, :])
    if pack:
        return pack_fingerprints(fps)
    return fps
