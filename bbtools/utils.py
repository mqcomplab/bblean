import itertools
import numpy as np
import typing as tp
from pathlib import Path
import sys
import subprocess
import platform
import importlib

from numpy.typing import NDArray

_T = tp.TypeVar("_T")


def _import_bitbirch_variant(
    variant: str = "bblean",
) -> tuple[tp.Any, tp.Callable[..., None]]:
    if variant not in ("lean", "leanv1", "int64"):
        raise ValueError(f"Unknown variant {variant}")
    if variant == "lean":
        module = importlib.import_module("bbtools.bblean")
    elif variant == "leanv1":
        module = importlib.import_module("bbtools.bblean_v1")
    elif variant == "int64":
        module = importlib.import_module("bbtools.bb_int64")
    return getattr(module, "BitBirch"), getattr(module, "set_merge")


# Itertools recipe
def batched(iterable: tp.Iterable[_T], n: int) -> tp.Iterator[tuple[_T, ...]]:
    r"""Batch data into tuples of length n. The last batch may be shorter."""
    # batched('ABCDEFG', 3) --> ('A', 'B', 'C') ('D', 'E', 'F') ('G',)
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


def cpu_name() -> str:
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


# Save a list of numpy arrays into a single array in a streaming fashion, avoiding
# stacking them in memory
def numpy_streaming_save(
    fps_bfs: list[list[NDArray[tp.Any]]], path: Path | str
) -> None:
    path = Path(path)
    for fp_list in fps_bfs:
        first_arr = np.ascontiguousarray(fp_list[0])
        header = np.lib.format.header_data_from_array_1_0(first_arr)
        header["shape"] = (len(fp_list), len(first_arr))
        with open(path.with_name(f"{path.name}_{first_arr.dtype.name}.npy"), "wb") as f:
            np.lib.format.write_array_header_1_0(f, header)
            for arr in fp_list:
                np.ascontiguousarray(arr).tofile(f)
