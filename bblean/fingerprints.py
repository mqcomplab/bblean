from pathlib import Path
from numpy.typing import NDArray, DTypeLike
import numpy as np
import typing as tp

from rich.console import Console
from rdkit.Chem import rdFingerprintGenerator, DataStructs, MolFromSmiles
from scipy.stats import truncnorm

__all__ = [
    "make_fake_fingerprints",
    "fps_from_smiles",
    "pack_fingerprints",
    "unpack_fingerprints",
]


def pack_fingerprints(a: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Packs binary uint8 arrays (only 0s and 1s) to uint8 arrays"""
    # packbits may pad with zeros if n_features is not a multiple of 8
    return np.packbits(a, axis=-1)


def unpack_fingerprints(a: NDArray[np.uint8], n_features: int) -> NDArray[np.uint8]:
    """Unpacks packed uint8 arrays into binary uint8 arrays (with only 0s and 1s)"""
    # n_features is required to discard padded zeros if it is not a multiple of 8
    return np.unpackbits(a, axis=-1, count=n_features)


def make_fake_fingerprints(
    num: int,
    n_features: int = 2048,
    pack: bool = False,
    seed: int | None = None,
    dtype: DTypeLike = np.uint8,
) -> NDArray[np.uint8]:
    # Generate "synthetic" fingerprints with a popcount distribution
    # similar to one in a real smiles database
    # Fps are guaranteed to *not* be all zeros or all ones
    if pack:
        if np.dtype(dtype) != np.dtype(np.uint8):
            raise ValueError("Only np.uint8 dtype is supported for packed input")
    loc = 750
    scale = 400
    bounds = (0, n_features)
    rng = np.random.default_rng(seed)
    safe_bounds = (bounds[0] + 1, bounds[1] - 1)
    a = (safe_bounds[0] - loc) / scale
    b = (safe_bounds[1] - loc) / scale
    popcounts_fake_float = truncnorm.rvs(
        a, b, loc=loc, scale=scale, size=num, random_state=rng
    )
    popcounts_fake = np.rint(popcounts_fake_float).astype(np.int64)
    zerocounts_fake = n_features - popcounts_fake
    repeats_fake = np.empty((num * 2), dtype=np.int64)
    repeats_fake[0::2] = popcounts_fake
    repeats_fake[1::2] = zerocounts_fake
    initial = np.tile(np.array([1, 0], np.uint8), num)
    expanded = np.repeat(initial, repeats=repeats_fake)
    fps_fake = rng.permuted(expanded.reshape(num, n_features), axis=-1)
    if pack:
        return np.packbits(fps_fake, axis=1)
    return fps_fake.astype(dtype, copy=False)


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


def _get_fps_file_num(path: Path) -> int:
    with open(path, mode="rb") as f:
        major, minor = np.lib.format.read_magic(f)
        shape, _, _ = getattr(np.lib.format, f"read_array_header_{major}_{minor}")(f)
        return shape[0]


def _get_fps_file_shape_and_dtype(
    path: Path,
) -> tuple[tuple[int, int], np.dtype, bool, bool]:
    with open(path, mode="rb") as f:
        major, minor = np.lib.format.read_magic(f)
        shape, _, dtype = getattr(np.lib.format, f"read_array_header_{major}_{minor}")(
            f
        )
    shape_is_valid = len(shape) == 2
    dtype_is_valid = np.issubdtype(dtype, np.integer)
    return shape, dtype, shape_is_valid, dtype_is_valid


def _print_fps_file_info(path: Path, console: Console | None = None) -> None:
    if console is None:
        console = Console()
    shape, dtype, shape_is_valid, dtype_is_valid = _get_fps_file_shape_and_dtype(path)

    console.print(f"File: {path.resolve()}")
    if shape_is_valid and dtype_is_valid:
        console.print("    - [green]Valid fingerprint file[/green]")
    else:
        console.print("    - [red]Invalid fingerprint file[/red]")
    if shape_is_valid:
        console.print(f"    - Num. fingerprints: {shape[0]:,}")
        console.print(f"    - Num. features: {shape[1]:,}")
    else:
        console.print(f"    - Shape: {shape}")
    console.print(f"    - DType: [yellow]{dtype.name}[/yellow]")
    console.print()
