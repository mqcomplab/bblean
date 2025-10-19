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
    def from_bb_input(cls, X: _Input, can_release: bool | None = None) -> tpx.Self:
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
            _can_release = True
        else:
            iters_per_pagex = 0
            curr_page_start_addr = 0
            _can_release = False
        if can_release is not None:
            _can_release = can_release
        return cls(_can_release, pagesizex, iters_per_pagex, curr_page_start_addr)

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
    #
    # TODO: In some cases, for some reason, this fails with errno 22
    # failure is harmless, but could incurr in a slight perf penalty
    _madvise_sequential(arr.ctypes.data - arr.offset, arr.nbytes)
    return arr


def system_mem_gib() -> tuple[int, int] | tuple[None, None]:
    mem = psutil.virtual_memory()
    return mem.total * _BYTES_TO_GIB, mem.available * _BYTES_TO_GIB


def get_peak_memory_gib(out_dir: Path) -> float | None:
    file = out_dir / "max-rss.txt"
    if not file.exists():
        return None
    with open(file, mode="r", encoding="utf-8") as f:
        peak_mem_gib = float(f.read().strip())
    return peak_mem_gib


def monitor_rss_process(
    file: Path | str, interval_s: float, start_time: float, parent_pid: int
) -> None:
    file = Path(file)
    this_pid = os.getpid()
    ps = psutil.Process(parent_pid)

    def total_rss() -> float:
        total_rss = ps.memory_info().rss
        for proc in ps.children(recursive=True):
            if proc.pid == this_pid:
                continue
            try:
                total_rss += proc.memory_info().rss
            except psutil.NoSuchProcess:
                # Prevent race condition since process may have finished before it can
                # be polled
                continue
        return total_rss

    with open(file, mode="w", encoding="utf-8") as f:
        f.write("rss_gib,time_s\n")
        f.flush()
        os.fsync(f.fileno())

    max_rss_gib = 0.0
    while True:
        total_rss_gib = total_rss() * _BYTES_TO_GIB
        with open(file, mode="a", encoding="utf-8") as f:
            f.write(f"{total_rss_gib},{time.perf_counter() - start_time}\n")
            f.flush()
            os.fsync(f.fileno())
        if total_rss_gib > max_rss_gib:
            max_rss_gib = total_rss_gib
            with open(file.parent / "max-rss.txt", mode="w", encoding="utf-8") as f:
                f.write(f"{max_rss_gib}\n")
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
            parent_pid=os.getpid(),
        ),
        daemon=True,
    ).start()
