r"""Misc. utility functions"""

import itertools
import numpy as np
import typing as tp
import sys
import subprocess
import platform
import importlib

import psutil
from numpy.typing import NDArray

__all__ = ["batched", "calc_centroid"]

_T = tp.TypeVar("_T")


def _import_bitbirch_variant(
    variant: str = "lean",
) -> tuple[tp.Any, tp.Callable[..., None]]:
    if variant not in ("lean", "int64_dense", "uint8"):
        raise ValueError(f"Unknown variant {variant}")
    if variant == "lean":
        # Most up-to-date bb variant
        module = importlib.import_module("bblean.bitbirch")
    elif variant == "uint8":
        # Legacy variant of bb that uses uint8 and supports packing, but no extra optim
        module = importlib.import_module("bblean._legacy.bb_uint8")
    elif variant == "int64_dense":
        # Legacy variant of bb that uses int64 fps (dense only)
        module = importlib.import_module("bblean._legacy.bb_int64_dense")

    Cls = getattr(module, "BitBirch")
    fn = getattr(module, "set_merge")
    return Cls, fn


# Itertools recipe
def batched(iterable: tp.Iterable[_T], n: int) -> tp.Iterator[tuple[_T, ...]]:
    r"""Batch data into tuples of length n. The last batch may be shorter.

    This is equivalent to the batched receip from `itertools`.
    """
    # batched('ABCDEFG', 3) --> ('A', 'B', 'C') ('D', 'E', 'F') ('G',)
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


def _num_avail_cpus() -> int:
    return len(psutil.Process().cpu_affinity())


def _cpu_name() -> str:
    if sys.platform == "darwin":
        try:
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except Exception:
            pass

    if sys.platform == "linux":
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()

    # Fallback for windows and all cases where it could not be found
    return platform.processor()


def calc_centroid(
    linear_sum: NDArray[np.integer], n_samples: int, *, pack: bool
) -> NDArray[np.uint8]:
    """Calculates centroid

    Parameters
    ----------

    linear_sum : np.ndarray
                 Sum of the elements column-wise
    n_samples : int
                Number of samples
    pack : bool
        Whether to pack the resulting fingerprints

    Returns
    -------
    centroid : np.ndarray[np.uint8]
               Centroid fingerprints of the given set
    """
    # NOTE: Numpy guarantees bools are stored as 0xFF -> True and 0x00 -> False,
    # so this view is fully safe
    if n_samples <= 1:
        centroid = linear_sum.astype(np.uint8, copy=False)
    else:
        centroid = (linear_sum >= n_samples * 0.5).view(np.uint8)
    if pack:
        return np.packbits(centroid, axis=-1)
    return centroid


# Requires numpy >= 2.0
def _popcount(a: NDArray[np.uint8]) -> NDArray[np.uint32]:
    # a is packed uint8 array with last axis = bytes
    # Sum bit-counts across bytes to get per-object totals

    # If the array has columns that are a multiple of 8, doing a bitwise count
    # over the buffer reinterpreted as uint64 is slightly faster.
    # This is zero cost if the exception is not triggered. Not having a be a multiple of
    # 8 is a very unlikely scenario, since fps are typically 1024 or 2048
    b: NDArray[np.integer]
    try:
        b = a.view(np.uint64)
    except ValueError:
        b = a
    return np.bitwise_count(b).sum(axis=-1, dtype=np.uint32)
