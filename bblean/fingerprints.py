r"""Utilites for manipulating fingerprints and fingerprint files"""

import sys
import math
import weakref
import warnings
import dataclasses
from pathlib import Path
from numpy.typing import NDArray, DTypeLike
import numpy as np
import typing as tp
import multiprocessing as mp
import multiprocessing.shared_memory as shmem

from rich.console import Console
from rdkit.Chem import rdFingerprintGenerator, MolFromSmiles, SanitizeFlags, SanitizeMol

from bblean._config import DEFAULTS
from bblean._console import get_console
from bblean.smiles import _iter_ranges_and_smiles_batches
from bblean.utils import _num_avail_cpus

__all__ = [
    "make_fake_fingerprints",
    "fps_from_smiles",
    "pack_fingerprints",
    "unpack_fingerprints",
]


# Deprecated
def calc_centroid(
    linear_sum: NDArray[np.integer], n_samples: int, *, pack: bool = True
) -> NDArray[np.uint8]:
    warnings.warn(
        "Please use `bblean.similarity.centroid_from_sum(...)` instead",
        DeprecationWarning,
        stacklevel=2,
    )
    if n_samples <= 1:
        centroid = linear_sum.astype(np.uint8, copy=False)
    else:
        centroid = (linear_sum >= n_samples * 0.5).view(np.uint8)
    if pack:
        return np.packbits(centroid, axis=-1)
    return centroid


centroid_from_sum = calc_centroid


def pack_fingerprints(a: NDArray[np.uint8]) -> NDArray[np.uint8]:
    r"""Pack binary (only 0s and 1s) uint8 fingerprint arrays"""
    # packbits may pad with zeros if n_features is not a multiple of 8
    return np.packbits(a, axis=-1)


def unpack_fingerprints(
    a: NDArray[np.uint8], n_features: int | None = None
) -> NDArray[np.uint8]:
    r"""Unpack packed uint8 arrays into binary uint8 arrays (with only 0s and 1s)

    .. note::

        If ``n_features`` is not passed, unpacking will only recover the correct number
        of features if it is a multiple of 8, otherwise fingerprints will be padded with
        zeros to the closest multiple of 8. This is generally not an issue since most
        common fingerprints feature sizes (2048, 1024, etc) are multiples of 8, but if
        you are using a non-standard number of features you should pass ``n_features``
        explicitly.
    """
    # n_features is required to discard padded zeros if it is not a multiple of 8
    return np.unpackbits(a, axis=-1, count=n_features)


def make_fake_fingerprints(
    num: int,
    n_features: int = DEFAULTS.n_features,
    pack: bool = True,
    seed: int | None = None,
    dtype: DTypeLike = np.uint8,
) -> NDArray[np.uint8]:
    r"""Make random fingerprints with statistics similar to (some) real databases"""
    import scipy.stats  # Hide this import since scipy is heavy

    if n_features < 1 or n_features % 8 != 0:
        raise ValueError("n_features must be a multiple of 8, and greater than 0")
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
    popcounts_fake_float = scipy.stats.truncnorm.rvs(
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


def _get_generator(kind: str, n_features: int) -> tp.Any:
    if kind == "rdkit":
        return rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=n_features)
    elif kind == "ecfp4":
        return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_features)
    elif kind == "ecfp6":
        return rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=n_features)
    elif kind == "topological":
        return rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=n_features)
    elif kind == "ap":
        return rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_features)
    raise ValueError(f"Unknown kind {kind}. Use 'rdkit|ecfp4|ecfp6|topological|ap'")


def _get_sanitize_flags(sanitize: str) -> tp.Any:
    if sanitize == "all":
        return SanitizeFlags.SANITIZE_ALL
    elif sanitize == "minimal":
        flags = SanitizeFlags.SANITIZE_CLEANUP | SanitizeFlags.SANITIZE_SYMMRINGS
        return flags
    else:
        raise ValueError("Unknown 'sanitize', must be one of 'all', 'minimal'")


@tp.overload
def fps_from_smiles(
    smiles: tp.Iterable[str],
    kind: str = DEFAULTS.fp_kind,
    n_features: int = DEFAULTS.n_features,
    dtype: DTypeLike = np.uint8,
    sanitize: str = "all",
    skip_invalid: tp.Literal[False] = False,
    pack: bool = True,
) -> NDArray[np.uint8]:
    pass


@tp.overload
def fps_from_smiles(
    smiles: tp.Iterable[str],
    kind: str = DEFAULTS.fp_kind,
    n_features: int = DEFAULTS.n_features,
    dtype: DTypeLike = np.uint8,
    sanitize: str = "all",
    skip_invalid: tp.Literal[True] = True,
    pack: bool = True,
) -> tuple[NDArray[np.uint8], NDArray[np.int64]]:
    pass


def fps_from_smiles(
    smiles: tp.Iterable[str],
    kind: str = DEFAULTS.fp_kind,
    n_features: int = DEFAULTS.n_features,
    dtype: DTypeLike = np.uint8,
    sanitize: str = "all",
    skip_invalid: bool = False,
    pack: bool = True,
) -> tp.Union[NDArray[np.uint8], tuple[NDArray[np.uint8], NDArray[np.int64]]]:
    r"""Convert a sequence of smiles into chemical fingerprints"""
    if n_features < 1 or n_features % 8 != 0:
        raise ValueError("n_features must be a multiple of 8, and greater than 0")
    if isinstance(smiles, str):
        smiles = [smiles]

    if pack and not (np.dtype(dtype) == np.dtype(np.uint8)):
        raise ValueError("Packing only supported for uint8 dtype")

    fpg = _get_generator(kind, n_features)

    sanitize_flags = _get_sanitize_flags(sanitize)

    smiles = list(smiles)
    fps = np.empty((len(smiles), n_features), dtype=dtype)

    invalid_idxs = []
    for i, smi in enumerate(smiles):
        mol = MolFromSmiles(smi, sanitize=False)
        if mol is None:
            if skip_invalid:
                invalid_idxs.append(i)
                continue
            else:
                raise ValueError(f"Unable to parse smiles {smi}")
        try:
            SanitizeMol(mol, sanitizeOps=sanitize_flags)
            fps[i, :] = fpg.GetFingerprintAsNumPy(mol)
        except Exception:
            if skip_invalid:
                invalid_idxs.append(i)
                continue
            raise

    if invalid_idxs:
        fps = np.delete(fps, invalid_idxs, axis=0)
    if pack:
        if skip_invalid:
            return pack_fingerprints(fps), np.array(invalid_idxs, dtype=np.int64)
        return pack_fingerprints(fps)
    if skip_invalid:
        return fps, np.array(invalid_idxs, dtype=np.int64)
    return fps


def _get_fps_file_num(path: Path) -> int:
    with open(path, mode="rb") as f:
        major, minor = np.lib.format.read_magic(f)
        shape, _, _ = getattr(np.lib.format, f"read_array_header_{major}_{minor}")(f)
        return shape[0]


def _get_fps_file_shape_and_dtype(
    path: Path, raise_if_invalid: bool = False
) -> tuple[tuple[int, int], np.dtype, bool, bool]:
    with open(path, mode="rb") as f:
        major, minor = np.lib.format.read_magic(f)
        shape, _, dtype = getattr(np.lib.format, f"read_array_header_{major}_{minor}")(
            f
        )
    shape_is_valid = len(shape) == 2
    dtype_is_valid = np.issubdtype(dtype, np.integer)
    if raise_if_invalid and (not shape_is_valid) or (not dtype_is_valid):
        raise ValueError(
            f"Fingerprints file {path} is invalid. Shape: {shape}, DType {dtype}"
        )
    return shape, dtype, shape_is_valid, dtype_is_valid


def _print_fps_file_info(path: Path, console: Console | None = None) -> None:
    if console is None:
        console = Console()
    shape, dtype, shape_is_valid, dtype_is_valid = _get_fps_file_shape_and_dtype(path)

    console.print(f"File: {path.resolve()}")
    has_nonzero = None
    if shape_is_valid and dtype_is_valid:
        console.print("    - [green]Valid fingerprint file[/green]")
        if shape[0] > 0:
            first_fp = np.load(path, mmap_mode="r")[0]
            has_nonzero = (first_fp > 1).any()
            if has_nonzero:
                console.print("    - Guessed format: [cyan]Packed[/cyan]")
            else:
                console.print("    - Guessed format: [magenta]Unpacked[/magenta]")
        else:
            console.print("    - Guessed format: [red]Unknown[/red]")
    else:
        console.print("    - [red]Invalid fingerprint file[/red]")
    if shape_is_valid:
        console.print(f"    - Num. fingerprints: {shape[0]:,}")
        if has_nonzero:
            console.print(
                f"    - Num. features: {shape[1]:,} (guessed unpacked: {shape[1] * 8:,})"  # noqa
            )
        else:
            console.print(f"    - Num. features: {shape[1]:,}")
    else:
        console.print(f"    - Shape: {shape}")
    console.print(f"    - DType: [yellow]{dtype.name}[/yellow]")
    console.print()


class _FingerprintFileSequence:
    def __init__(self, files: tp.Iterable[Path]) -> None:
        self._files = list(files)
        if len(self._files) == 0:
            raise ValueError("At least 1 fingerprint file must be provided")

    def __getitem__(self, idxs: tp.Sequence[int]) -> NDArray[np.uint8]:
        return _get_fingerprints_from_file_seq(self._files, idxs)

    @property
    def shape(self) -> tuple[int, int]:
        shape, dtype, _, _ = _get_fps_file_shape_and_dtype(
            self._files[0], raise_if_invalid=True
        )
        return shape


# TODO: The logic of this function is pretty complicated, maybe there is a way to
# simplify it?
def _get_fingerprints_from_file_seq(
    files: tp.Iterable[Path], idxs: tp.Sequence[int]
) -> NDArray[np.uint8]:
    if sorted(idxs) != list(idxs):
        raise ValueError("idxs must be sorted")
    # Sequence of files is assumed to have indexes in an increasing order,
    # for example, if the first two files have 10k fingerprints, then the
    # assoc. idxs are 0-9999 and 10000-19999. 'idxs' will index this sequence of files
    # iter_idxs = iter(idxs)
    n_features = None
    local_file_idxs = []
    consumed_idxs = 0
    running_count = 0
    for f in files:
        (num, _n_features), _, _, _ = _get_fps_file_shape_and_dtype(
            f, raise_if_invalid=True
        )
        # Fetch idxs Append array([]) if no idxs in the file
        file_idxs = list(
            filter(lambda x: x < running_count + num, idxs[consumed_idxs:])
        )
        consumed_idxs += len(file_idxs)
        local_file_idxs.append(np.array(file_idxs, dtype=np.uint64) - running_count)
        running_count += num

        if n_features is None:
            n_features = _n_features
        elif _n_features != n_features:
            raise ValueError(
                f"Incompatible in fingerprint file {f},"
                f" expected {n_features}, found {_n_features}"
            )
    if len(idxs) != sum(arr.size for arr in local_file_idxs):
        raise ValueError("idxs could not be extracted from files")

    arr = np.empty((len(idxs), tp.cast(int, n_features)), dtype=np.uint8)
    i = 0
    for file, local_idxs in zip(files, local_file_idxs):
        size = local_idxs.size
        if not size:
            continue
        arr[i : i + size] = np.load(file, mmap_mode="r")[local_idxs].astype(
            np.uint8, copy=False
        )
        i += size
    return arr


# TODO: Skipping invalid smiles is a bit inefficient
# NOTE: Mostly convenient for usage in multiprocessing workflows
@dataclasses.dataclass
class _FingerprintFileCreator:
    dtype: str
    out_dir: Path
    out_name: str
    digits: int | None
    pack: bool
    kind: str
    n_features: int
    sanitize: str
    skip_invalid: bool
    verbose: bool

    def __call__(self, input_: tuple[int, tp.Sequence[str]]) -> None:
        console = get_console(self.verbose)
        fpg = _get_generator(self.kind, self.n_features)
        file_idx, batch = input_
        fps = np.empty((len(batch), self.n_features), dtype=self.dtype)
        out_name = self.out_name
        sanitize_flags = _get_sanitize_flags(self.sanitize)
        invalid = []
        for i, smi in enumerate(batch):
            mol = MolFromSmiles(smi, sanitize=False)
            if mol is None:
                if self.skip_invalid:
                    invalid.append(i)
                    continue
                else:
                    raise ValueError(f"Unable to parse smiles {smi}")
            try:
                SanitizeMol(mol, sanitizeOps=sanitize_flags)  # Raises if invalid
            except Exception:
                if self.skip_invalid:
                    invalid.append(i)
                    continue
                raise
            fps[i, :] = fpg.GetFingerprintAsNumPy(mol)
        if self.pack:
            fps = pack_fingerprints(fps)
        if self.digits is not None:
            out_name = f"{out_name}.{str(file_idx).zfill(self.digits)}"
        if invalid:
            prev_num = len(fps)
            fps = np.delete(fps, invalid, axis=0)
            new_num = len(fps)
            console.print(
                f"File {file_idx}: Generated {new_num} fingerprints\n"
                f" File {file_idx}: Skipped {prev_num - new_num} invalid smiles"
            )
        np.save(self.out_dir / out_name, fps)


@dataclasses.dataclass
class _FingerprintArrayFiller:
    invalid_mask_shmem_name: str
    shmem_name: str
    kind: str
    fp_size: int
    pack: bool
    dtype: str
    num_smiles: int
    sanitize: str
    skip_invalid: bool

    def __call__(self, idx_range: tuple[int, int], batch: tp.Sequence[str]) -> None:
        fpg = _get_generator(self.kind, self.fp_size)
        (idx0, idx1) = idx_range
        fps_shmem = shmem.SharedMemory(name=self.shmem_name)
        invalid_mask_shmem = shmem.SharedMemory(name=self.invalid_mask_shmem_name)
        sanitize_flags = _get_sanitize_flags(self.sanitize)

        if self.pack:
            out_dim = (self.fp_size + 7) // 8
        else:
            out_dim = self.fp_size
        fps = np.ndarray(
            (self.num_smiles, out_dim), dtype=self.dtype, buffer=fps_shmem.buf
        )
        invalid_mask = np.ndarray(
            (self.num_smiles,), dtype=np.bool, buffer=invalid_mask_shmem.buf
        )
        for i, smi in zip(range(idx0, idx1), batch):
            mol = MolFromSmiles(smi, sanitize=False)
            if mol is None:
                if self.skip_invalid:
                    invalid_mask[i] = True
                    continue
                else:
                    raise ValueError(f"Unable to parse smiles {smi}")
            try:
                SanitizeMol(mol, sanitizeOps=sanitize_flags)  # Raises if invalid
            except Exception:
                if self.skip_invalid:
                    invalid_mask[i] = True
                    continue
                raise
            fp = fpg.GetFingerprintAsNumPy(mol)
            if self.pack:
                fp = pack_fingerprints(fp)
            fps[i, :] = fp
        fps_shmem.close()
        invalid_mask_shmem.close()


@tp.overload
def fps_from_smiles_parallel(
    smiles: tp.Iterable[str],
    kind: str = DEFAULTS.fp_kind,
    n_features: int = DEFAULTS.n_features,
    dtype: DTypeLike = np.uint8,
    sanitize: str = "all",
    skip_invalid: tp.Literal[False] = False,
    pack: bool = True,
    num_ps: int = 1,
    replace_dummy_atoms: bool = False,
    tab_separated: bool = False,
    mp_context: tp.Any = None,
) -> NDArray[np.uint8]:
    pass


@tp.overload
def fps_from_smiles_parallel(
    smiles: tp.Iterable[str],
    kind: str = DEFAULTS.fp_kind,
    n_features: int = DEFAULTS.n_features,
    dtype: DTypeLike = np.uint8,
    sanitize: str = "all",
    skip_invalid: tp.Literal[True] = True,
    pack: bool = True,
    num_ps: int = 1,
    replace_dummy_atoms: bool = False,
    tab_separated: bool = False,
    mp_context: tp.Any = None,
) -> tp.Union[NDArray[np.uint8], tuple[NDArray[np.uint8], NDArray[np.int64]]]:
    pass


# NOTE: This function is proof of concept and kinda dangerous since it registers
# a custom destructor for the numpy array
# It is also *only usable if called inside an if __name__ == "__main__" guard*
# For now lets hide it
def fps_from_smiles_parallel(
    smiles: tp.Iterable[str],
    kind: str = DEFAULTS.fp_kind,
    n_features: int = DEFAULTS.n_features,
    dtype: DTypeLike = np.uint8,
    sanitize: str = "all",
    skip_invalid: bool = False,
    pack: bool = True,
    num_ps: int | None = None,
    replace_dummy_atoms: bool = False,
    tab_separated: bool = False,
    mp_context: tp.Any = None,
) -> tp.Union[NDArray[np.uint8], tuple[NDArray[np.uint8], NDArray[np.int64]]]:
    r""":meta private:"""
    if mp_context is None:
        mp_context = mp.get_context("forkserver" if sys.platform == "linux" else None)
    if isinstance(smiles, str):
        smiles = [smiles]
    smiles = list(smiles)
    smiles_num = len(smiles)
    if num_ps is None:
        num_ps = _num_avail_cpus()

    if pack:
        out_dim = (n_features + 7) // 8
    else:
        out_dim = n_features
    shmem_size = smiles_num * out_dim * np.dtype(dtype).itemsize
    fps_shmem = shmem.SharedMemory(create=True, size=shmem_size)
    invalid_mask_shmem = shmem.SharedMemory(create=True, size=smiles_num)
    fps_array_filler = _FingerprintArrayFiller(
        shmem_name=fps_shmem.name,
        invalid_mask_shmem_name=invalid_mask_shmem.name,
        kind=kind,
        fp_size=n_features,
        num_smiles=smiles_num,
        dtype=np.dtype(dtype).name,
        pack=pack,
        sanitize=sanitize,
        skip_invalid=skip_invalid,
    )
    num_per_batch = math.ceil(smiles_num / num_ps)
    with mp_context.Pool(processes=num_ps) as pool:
        pool.starmap(
            fps_array_filler,
            _iter_ranges_and_smiles_batches(
                smiles,
                num_per_batch,
                tab_separated,
                replace_dummy_atoms,
                assume_paths=False,
            ),
        )
    fps = np.ndarray((smiles_num, out_dim), dtype=dtype, buffer=fps_shmem.buf)
    mask = np.ndarray((smiles_num,), dtype=np.bool, buffer=invalid_mask_shmem.buf)
    if skip_invalid:
        fps = np.delete(fps, mask, axis=0)
        weakref.finalize(mask, invalid_mask_shmem.close)
        weakref.finalize(mask, invalid_mask_shmem.unlink)
        weakref.finalize(fps, fps_shmem.close)
        weakref.finalize(fps, fps_shmem.unlink)
        return fps, mask

    del mask
    invalid_mask_shmem.close()
    invalid_mask_shmem.unlink()
    weakref.finalize(fps, fps_shmem.close)
    weakref.finalize(fps, fps_shmem.unlink)
    return fps
