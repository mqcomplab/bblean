r"""Global defaults and related utilities"""

from copy import deepcopy
from pathlib import Path
import typing as tp
import json
import sys
import dataclasses
import multiprocessing as mp
import os

import numpy as np

from bblean._memory import system_mem_gib
from bblean.utils import _cpu_name


@dataclasses.dataclass(slots=True)
class BitBirchConfig:
    threshold: float = 0.65
    branching_factor: int = 254
    merge_criterion: str = "diameter"
    tolerance: float = 0.05
    n_features: int = 2048
    use_mmap: bool = True


DEFAULTS = BitBirchConfig()


def collect_system_specs_and_dump_config(
    config: dict[str, tp.Any],
) -> None:
    config = deepcopy(config)
    config_path = Path(config["out_dir"]) / "config.json"
    total_mem, avail_mem = system_mem_gib()
    # System info
    config["total_memory_gib"] = total_mem
    config["initial_available_memory_gib"] = avail_mem
    config["platform"] = sys.platform
    config["cpu"] = _cpu_name()
    config["numpy_version"] = np.__version__
    config["python_version"] = sys.version.split()[0]
    # Multiprocessing info
    if config.get("num_processes", 1) > 1:
        config["multiprocessing_start_method"] = mp.get_start_method()
        config["visible_cpu_cores"] = os.cpu_count()

    # Dump config after checking if the output dir has files
    with open(config_path, mode="wt", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
