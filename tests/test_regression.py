import gc
import math
import time
import sys
from pathlib import Path
import tempfile
import pytest
import bblean

try:
    from memray import Tracker, FileReader
except Exception:
    # Not available in Windows
    pass

from bblean.fingerprints import make_fake_fingerprints


def test_memory_regression() -> None:
    if "memray" not in sys.modules:
        pytest.skip(
            "memory regression tests require memray, only avaliable in Linux and macOS"
        )
    max_allowed_bytes = 42_000_000  # 42 MB
    # Around 41.9 MB should be allocated for these 100k fps
    # Actual benchmarked allocation is 41_914_658 for
    # - py3.11
    # - numpy 2.3
    # - ubuntu-24.04
    # - GLIBC 2.39
    fps = make_fake_fingerprints(10_000, seed=4068791011890883085)
    with tempfile.TemporaryDirectory() as d:
        tmp_dir = Path(d)
        tree = bblean.BitBirch()
        with Tracker(tmp_dir / "memray.bin"):
            tree.fit(fps)
        reader = FileReader(tmp_dir / "memray.bin")
        total_alloc_bytes = sum(
            record.size
            for record in reader.get_high_watermark_allocation_records(
                merge_threads=True
            )
        )
    assert total_alloc_bytes < max_allowed_bytes


# NOTE: This test is pretty fragile and may fail if CI machines change
def test_speed_regression() -> None:
    max_allowed_time_ns = 1_150_000_000
    # Fitting 15_000 fps should take ~ 1.15-1.00s or less:
    # - py3.11
    # - numpy 2.3
    # - ubuntu-24.04
    # - GLIBC 2.39
    # - AMD Ryzen 5 7535HS
    fps = make_fake_fingerprints(10_000, seed=4068791011890883085)
    repeats = 3

    total_time_ns = 0
    tree = bblean.BitBirch()
    for _ in range(repeats):
        start_ns = time.process_time_ns()
        tree.fit(fps)
        total_time_ns += time.process_time_ns() - start_ns
        tree.reset()
        gc.collect()
    total_time_ns = math.ceil(total_time_ns / repeats)
    assert total_time_ns < max_allowed_time_ns
