r"""Misc. utility functions"""

import os
from pathlib import Path
import itertools
import typing as tp
import sys
import subprocess
import platform
import importlib

import psutil
import numpy as np

__all__ = [
    "batched",
    "min_safe_uint",
    "cpp_extensions_are_enabled",
    "cpp_extensions_are_installed",
]

_T = tp.TypeVar("_T")


def min_safe_uint(nmax: int) -> np.dtype:
    r"""Returns the min uint dtype that holds a (positive) py int, excluding "object".

    Input must be a positive python integer.
    """
    out = np.min_scalar_type(nmax)
    # Check if the dtype is a pointer to a python bigint
    if out.hasobject:
        raise ValueError(f"n_samples: {nmax} is too large to hold in a uint64 array")
    return out


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


def _import_bitbirch_variant(
    variant: str = "lean",
) -> tuple[tp.Any, tp.Callable[..., None]]:
    if variant not in ("lean", "int64", "uint8"):
        raise ValueError(f"Unknown variant {variant}")
    if variant == "lean":
        # Most up-to-date bb variant
        module = importlib.import_module("bblean.bitbirch")
    elif variant == "uint8":
        # Legacy variant of bb that uses uint8 and supports packing, but no extra optim
        module = importlib.import_module("bblean._legacy.bb_uint8")
    elif variant == "int64":
        # Legacy variant of bb that uses int64 fps (dense only)
        module = importlib.import_module("bblean._legacy.bb_int64")

    Cls = getattr(module, "BitBirch")
    fn = getattr(module, "set_merge")
    return Cls, fn


def _num_avail_cpus() -> int:
    if sys.platform == "darwin":
        # macOS doesn't expose cpu affinity, so assume all cpu's are available
        return os.cpu_count()
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


def _has_files_or_valid_symlinks(path: Path) -> bool:
    has_files = False
    for p in path.iterdir():
        if p.is_symlink() and not p.exists():
            return False

        if p.is_file():
            has_files = True
    return has_files


def cpp_extensions_are_enabled() -> bool:
    r"""Query whether the C++ BitBRICH extensions are currently enabled"""
    if os.getenv("BITBIRCH_NO_EXTENSIONS"):
        return False
    try:
        from bblean._cpp_similarity import jt_isim_from_sum  # noqa

        return True
    except ImportError:
        return False


def cpp_extensions_are_installed() -> bool:
    r"""Query whether the C++ BitBRICH extensions are currently installed"""
    try:
        from bblean._cpp_similarity import jt_isim_from_sum  # noqa

        return True
    except ImportError:
        return False
