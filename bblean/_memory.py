r"""Monitor and collect memory stats"""

import typing as tp
import mmap
import warnings
from enum import Enum
import ctypes
import dataclasses
from pathlib import Path
import sys
import time
import os
import multiprocessing as mp

import typing_extensions as tpx
import psutil
import numpy as np
from numpy.typing import NDArray
from rich.console import Console

try:
    import resource
except Exception:
    # resource is only available on Unix systems
    pass


_BYTES_TO_GIB = 1 / 1024**3


class Madv(Enum):
    WILLNEED = 3
    SEQUENTIAL = 2
    # PAGEOUT and DONTNEED reduce memory usage around 40%
    # TODO: Check exactly what DONTNEED does. I believe PAGEOUT *swaps out*
    # so DONTNEED may be preferred since it may have less perf. penalty
    DONTNEED = 4
    PAGEOUT = 21
    FREE = 8  # *ONLY* works on anonymous pages (not file-backed like numpy arrays)
    # Cold does *not* immediatly release memory, it is just a soft hint that
    # those pages won't be needed soon
    COLD = 20


# Get handle to the system's libc
def _get_libc() -> tp.Any:
    if sys.platform == "linux":
        return ctypes.CDLL("libc.so.6", use_errno=True)
    elif sys.platform == "darwin":
        return ctypes.CDLL("libc.dylib", use_errno=True)
    # For now, do nothing in Windows
    return


# This reduces memory usage around 40%, since the kernel can release
# pages once the array has been iterated over. The issue is, after this has been done,
# the array is out of the RAM, so refinement is not possible.
def _madvise_dontneed(page_start: int, size: int) -> None:
    _madvise(page_start, size, Madv.DONTNEED)


# let the kernel know that access to this range of addrs will be sequential
# (pages can be read-ahead and discarded fast after read if needed)
def _madvise_sequential(page_start: int, size: int) -> None:
    _madvise(page_start, size, Madv.SEQUENTIAL)


def _madvise(page_start: int, size: int, opt: Madv) -> None:
    libc = _get_libc()
    if libc is None:
        return
    if libc.madvise(ctypes.c_void_p(page_start), size, opt.value) != 0:
        errno = ctypes.get_errno()
        warnings.warn(f"{opt} failed with error code {errno}")


_Input = tp.Union[NDArray[np.integer], list[NDArray[np.integer]]]


@dataclasses.dataclass
class _ArrayMemPagesManager:
    can_release: bool
    _pagesizex: int
    _iters_per_pagex: int
    _curr_page_start_addr: int

    @classmethod
    def from_array(cls, X: _Input) -> tpx.Self:
        pagesizex = mmap.PAGESIZE * 512
        if (
            isinstance(X, np.memmap)
            and X.ndim == 2
            and (pagesizex % X.shape[1] == 0)
            and X.offset < X.shape[1]
        ):
            # In most cases pagesizex % n_features == 0 and offset < n_features
            # Every n_iters, release the prev page and add pagesizex to start_addr
            iters_per_pagex = int(pagesizex / X.shape[1])  # ~ 8192 iterations
            curr_page_start_addr = X.ctypes.data - X.offset
            can_release = True
        else:
            iters_per_pagex = 0
            curr_page_start_addr = 0
            can_release = False
        return cls(can_release, pagesizex, iters_per_pagex, curr_page_start_addr)

    def should_release_curr_page(self, row_idx: int) -> bool:
        return row_idx % self._iters_per_pagex == 0

    def release_curr_page_and_update_addr(self) -> None:
        _madvise_dontneed(self._curr_page_start_addr, self._pagesizex)
        self._curr_page_start_addr += self._pagesizex


def _mmap_file_and_madvise_sequential(
    path: Path, max_fps: int | None = None
) -> NDArray[np.integer]:
    arr = np.load(path, mmap_mode="r")[:max_fps]
    # Numpy actually puts the *whole file* in mmap mode (arr + header)
    # This means the array data starts from a nonzero offset starting from the backing
    # buffer if we want the address to the start of the file we need to displace the
    # addr of the arry by the bsize of the header, which can be accessed by arr.offset
    #
    # This is required since madvise needs a page-aligned address (address must
    # be a multiple of mmap.PAGESIZE (portable) == os.sysconf("SC_PAGE_SIZE")
    # (mac|linux), typically 4096 B).
    _madvise_sequential(arr.ctypes.data - arr.offset, arr.nbytes)
    return arr


def system_mem_gib() -> tuple[int, int] | tuple[None, None]:
    mem = psutil.virtual_memory()
    return mem.total * _BYTES_TO_GIB, mem.available * _BYTES_TO_GIB


@dataclasses.dataclass
class PeakMemoryStats:
    self_gib: float
    child_gib: float | None

    @property
    def children_were_tracked(self) -> bool:
        if mp.get_start_method() == "forkserver":
            return False
        return True


def get_peak_memory(num_processes: int) -> PeakMemoryStats | None:
    # Can't track peak memory in non-unix systems
    if "resource" not in sys.modules:
        return None
    max_mem_bytes_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    max_mem_bytes_child = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if sys.platform == "linux":
        # In linux these are returned kiB, not bytes
        max_mem_bytes_self *= 1024
        max_mem_bytes_child *= 1024
    max_mem_gib_self = max_mem_bytes_self * _BYTES_TO_GIB
    max_mem_gib_child = max_mem_bytes_child * _BYTES_TO_GIB

    if num_processes == 1:
        return PeakMemoryStats(max_mem_gib_self, None)
    return PeakMemoryStats(max_mem_gib_self, max_mem_gib_child)


def monitor_rss_process(file: Path | str, interval_s: float, start_time: float) -> None:
    def total_rss() -> float:
        total_rss = 0.0
        for proc in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
            info = proc.info
            cmdline = info["cmdline"]
            if cmdline is None:
                continue
            if Path(__file__).name in cmdline:
                total_rss += info["memory_info"].rss
        return total_rss

    t = start_time
    with open(file, mode="w", encoding="utf-8") as f:
        f.write("rss_gib,time_s\n")
        f.flush()
        os.fsync(f.fileno())

    while True:
        total_rss_gib = total_rss() * _BYTES_TO_GIB
        t = time.perf_counter() - start_time
        with open(file, mode="a", encoding="utf-8") as f:
            f.write(f"{total_rss_gib},{t}\n")
            f.flush()
            os.fsync(f.fileno())
        time.sleep(interval_s)


def launch_monitor_rss_daemon(
    out_file: Path, interval_s: float, console: Console | None = None
) -> None:
    if console is not None:
        console.print("** Monitoring total RAM usage **\n")
    mp.Process(
        target=monitor_rss_process,
        kwargs=dict(
            file=out_file,
            interval_s=interval_s,
            start_time=time.perf_counter(),
        ),
        daemon=True,
    ).start()
