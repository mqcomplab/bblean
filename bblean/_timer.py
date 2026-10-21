r"""General timing tools"""

import json
import time
import typing as tp
from pathlib import Path

from rich.console import Console


class Timer:
    def __init__(self, indent: bool = True, log_endpoints: bool = False) -> None:
        self._timings_s: dict[str, float] = {}
        self._log_endpoints = log_endpoints
        self._start_timings_s: dict[str, float] = {}
        self._end_timings_s: dict[str, float] = {}
        self._indent = indent
        self._global_init_time_s: float | None = None
        self._global_end_time_s: float | None = None

    @property
    def timings_s(self) -> dict[str, float]:
        return self._timings_s.copy()

    def init_timing(self, label: str = "total") -> None:
        if label != "total" and self._global_init_time_s is None:
            raise ValueError("Overal timing needs to be initialized")

        if label in self._timings_s:
            raise ValueError(f"{label} has already been tracked")
        self._timings_s[label] = time.perf_counter()

        if label == "total":
            self._global_init_time_s = self._timings_s[label]

        if self._log_endpoints and label != "total":
            self._start_timings_s[label] = self._timings_s[label] - tp.cast(
                float, self._global_init_time_s
            )

    def end_timing(
        self,
        label: str = "total",
        console: Console | None = None,
        indent: bool | None = None,
    ) -> None:
        if self._global_end_time_s is not None:
            raise ValueError("Timing has finished, reset timing before reusing timer")

        indent = indent if indent is not None else self._indent
        if label not in self._timings_s:
            raise ValueError(f"{label} has not been initialized")
        end_time_s = time.perf_counter()
        self._timings_s[label] = end_time_s - self._timings_s[label]
        t = self._timings_s[label]

        if self._log_endpoints and label != "total":
            self._end_timings_s[label] = end_time_s - tp.cast(
                float, self._global_init_time_s
            )
        if console is not None:
            if indent:
                indent_str = "    "
            else:
                indent_str = ""
            if label == "total":
                console.print(f"{indent_str}- Total time elapsed: {t:.4f} s")
            else:
                console.print(f"{indent_str}- Time for {label}: {t:.4f} s")
        if label == "total":
            self._global_end_time_s = end_time_s

    def dump(self, path: Path) -> None:
        if self._global_end_time_s is None:
            raise ValueError("Timing not finished yet")
        output: dict[str, float | dict[str, float]] = {}
        output.update(self._timings_s)
        if self._start_timings_s:
            output["start_timings"] = self._start_timings_s
        if self._end_timings_s:
            output["end_timings"] = self._end_timings_s

        with open(path, mode="wt", encoding="utf-8") as f:
            json.dump(output, f, indent=4)

    def reset(self) -> None:
        self._timings_s = {}
        self._start_timings_s = {}
        self._end_timings_s = {}
        self._global_init_time_s = None
        self._global_end_time_s = None
