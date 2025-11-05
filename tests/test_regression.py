# type: ignore
import gc
import math
import time
import sys
import os
from pathlib import Path
import tempfile
import pytest
import bblean

try:
    import bblean._cpp_similarity as csim  # noqa

    CSIM_AVAIL = True
except ImportError:
    if os.getenv("BITBIRCH_CANT_SKIP_CPP_TESTS"):
        raise
    CSIM_AVAIL = False

try:
    from memray import Tracker, FileReader
except Exception:
    # Not available in Windows
    pass

from bblean.fingerprints import make_fake_fingerprints


# NOTE: The reproducibility for allocations for 25k fps is not very good
def test_memory_regression(subtests) -> None:
    if "memray" not in sys.modules:
        pytest.skip(
            "memory regression tests require memray, only avaliable in Linux and macOS"
        )
    all_max_allowed_bytes = [43_000_000, 63_000_000, 86_000_000, 105_000_000]
    all_fps_nums = [10_000, 15_000, 20_000, 25_000]
    # Around 41.9 MB should be allocated for these 100k fps
    # If memory usage scaled linearly it would take ~ 3.7 GiB for 1M molecules
    # Actual benchmarked allocation is 41_914_658 for
    # - py3.11
    # - numpy 2.3
    # - ubuntu-24.04
    # - GLIBC 2.39
    for fps_num, max_allowed_bytes in zip(all_fps_nums, all_max_allowed_bytes):
        with subtests.test(fps_num=fps_num):
            fps = make_fake_fingerprints(fps_num, seed=4068791011890883085)
            with tempfile.TemporaryDirectory() as d:
                tmp_dir = Path(d)
                tree = bblean.BitBirch(branching_factor=50, threshold=0.65)
                with Tracker(tmp_dir / "memray.bin"):
                    tree.fit(fps)
                reader = FileReader(tmp_dir / "memray.bin")
                total_alloc_bytes = sum(
                    record.size
                    for record in reader.get_high_watermark_allocation_records(
                        merge_threads=True
                    )
                )
            print(
                f"Fps num: {fps_num}, Total alloc MB: {total_alloc_bytes / 1024**2},"
                f" Tot. alloc bytes: {total_alloc_bytes:_} (max: {max_allowed_bytes:_})"
            )
            assert total_alloc_bytes < max_allowed_bytes


# NOTE: This test is pretty fragile and may fail if CI machines change
def test_speed_regression(subtests) -> None:
    all_fps_nums = [10_000, 15_000, 20_000]
    # Fitting 15_000 fps should take ~ 1.15-1.00s or less:
    # - py3.11
    # - numpy 2.3
    # - ubuntu-24.04
    # - GLIBC 2.39
    # - AMD Ryzen 5 7535HS
    # For this system the following values are sufficient:
    # all_max_allowed_ns = [1_200_000_000, 1_900_000_000, 2_500_000_000]
    # For the ubuntu-24.04 in gh CI the following are required:
    if CSIM_AVAIL:
        all_max_allowed_ns = [900_000_000, 1_500_000_000, 2_000_000_000]
    else:
        all_max_allowed_ns = [1_700_000_000, 2_600_000_000, 3_600_000_000]
    for fps_num, max_allowed_ns in zip(all_fps_nums, all_max_allowed_ns):
        with subtests.test(fps_num=fps_num):
            fps = make_fake_fingerprints(fps_num, seed=4068791011890883085)
            repeats = 3

            total_time_ns = 0
            tree = bblean.BitBirch(branching_factor=50, threshold=0.65)
            for _ in range(repeats):
                start_ns = time.process_time_ns()
                tree.fit(fps)
                total_time_ns += time.process_time_ns() - start_ns
                tree.reset()
                gc.collect()
            total_time_ns = math.ceil(total_time_ns / repeats)
            print(
                f"Fps num: {fps_num}, Total time ns: {total_time_ns:_} "
                f" (max: {max_allowed_ns:_})"
            )
            assert total_time_ns < max_allowed_ns
