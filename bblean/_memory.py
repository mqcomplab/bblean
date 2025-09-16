r"""Monitor and collect memory stats"""

import dataclasses
from pathlib import Path
import sys
import time
import os
import multiprocessing as mp

import psutil
from rich.console import Console

try:
    import resource
except Exception:
    # resource is only available on Unix systems
    pass


_BYTES_TO_GIB = 1 / 1024**3


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
