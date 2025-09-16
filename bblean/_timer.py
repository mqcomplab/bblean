r"""General timing tools"""

import json
from pathlib import Path
import time

from rich.console import Console


class Timer:
    def __init__(self) -> None:
        self._timings_s: dict[str, float] = {}

    @property
    def timings_s(self) -> dict[str, float]:
        return self._timings_s.copy()

    def init_timing(self, label: str = "total") -> None:
        if label in self._timings_s:
            raise ValueError(f"{label} has already been tracked")
        self._timings_s[label] = time.perf_counter()

    def end_timing(
        self, label: str = "total", console: Console | None = None, indent: bool = True
    ) -> None:
        if label not in self._timings_s:
            raise ValueError(f"{label} has not been initialized")
        self._timings_s[label] = time.perf_counter() - self._timings_s[label]
        t = self._timings_s[label]
        if console is not None:
            if indent:
                indent_str = "    "
            else:
                indent_str = ""
            if label == "total":
                console.print(f"{indent_str}- Total time elapsed: {t:.4f} s")
            else:
                console.print(f"{indent_str}- Time for {label}: {t:.4f} s")

    def dump(self, path: Path) -> None:
        with open(path, mode="wt", encoding="utf-8") as f:
            json.dump(self._timings_s, f, indent=4)
