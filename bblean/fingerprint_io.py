from pathlib import Path

from numpy.typing import NDArray
import numpy as np
from rich.console import Console


def get_file_num_fps(path: Path) -> int:
    with open(path, mode="rb") as f:
        major, minor = np.lib.format.read_magic(f)
        shape, _, _ = getattr(np.lib.format, f"read_array_header_{major}_{minor}")(f)
        return shape[0]


def get_file_shape_and_dtype(
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


def print_file_info(path: Path, console: Console | None = None) -> None:
    if console is None:
        console = Console()
    shape, dtype, shape_is_valid, dtype_is_valid = get_file_shape_and_dtype(path)

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


# Save a list of numpy arrays into a single array in a streaming fashion, avoiding
# stacking them in memory
def numpy_streaming_save(fp_list: list[NDArray[np.integer]], path: Path | str) -> None:
    first_arr = np.ascontiguousarray(fp_list[0])
    header = np.lib.format.header_data_from_array_1_0(first_arr)
    header["shape"] = (len(fp_list), len(first_arr))
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".npy")
    with open(path, "wb") as f:
        np.lib.format.write_array_header_1_0(f, header)
        for arr in fp_list:
            np.ascontiguousarray(arr).tofile(f)
