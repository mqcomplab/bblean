r"""Pretty printing"""

from pathlib import Path
import numpy as np
import typing as tp
import os
import multiprocessing as mp

from rich.console import Console

from bblean._memory import get_peak_memory_gib


class BBConsole(Console):
    def print_banner(self) -> None:
        if os.environ.get("BITBIRCHNOBANNER", ""):
            return
        banner = r"""[bold]
        ______ _ _  ______ _          _        
        | ___ (_) | | ___ (_)        | |     [/bold][cyan]  ______                      [/cyan][bold]
        | |_/ /_| |_| |_/ /_ _ __ ___| |__   [/bold][cyan]  ___  / ___________ _______  [/cyan][bold]
        | ___ \ | __| ___ \ | '__/ __| '_ \  [/bold][cyan]  __  /  _  _ \  __ `/_  __ \ [/cyan][bold]
        | |_/ / | |_| |_/ / | | | (__| | | | [/bold][cyan]  _  /___/  __/ /_/ /_  / / / [/cyan][bold]
        \____/|_|\__\____/|_|_|  \___|_| |_| [/bold][cyan]  /_____/\___/\__,_/ /_/ /_/  [/cyan][bold]"""  # noqa W291
        self.print(banner, highlight=False)
        self.print()
        self.print()
        self.print(
            r"""BitBirch-Lean is developed by the [bold]Miranda-Quintana Lab[/bold] https://github.com/mqcomplab
If you find this software useful please cite the following articles:
    [yellow]•[/yellow] [italic]BitBIRCH: efficient clustering of large molecular libraries[/italic]:
        https://doi.org/10.1039/D5DD00030K
    [yellow]•[/yellow] [italic]BitBIRCH Clustering Refinement Strategies[/italic]:
        https://doi.org/10.1021/acs.jcim.5c00627
    [yellow]•[/yellow] [italic]BitBIRCH-Lean[/italic]:
        (preprint) https://www.biorxiv.org/content/10.1101/2025.10.22.684015v1"""  # noqa
        )

    def print_peak_mem(self, out_dir: Path, indent: bool = True) -> None:
        peak_mem_gib = get_peak_memory_gib(out_dir)
        if peak_mem_gib is None:
            return
        indent_str = " " * 4 if indent else ""
        self.print(f"{indent_str}- Peak RAM use: {peak_mem_gib:.4f} GiB")

    def print_config(self, config: dict[str, tp.Any]) -> None:
        num_fps_loaded = np.array(config["num_fps_loaded"])
        total_fps_num = num_fps_loaded.sum()
        with np.printoptions(formatter={"int": "{:,}".format}, threshold=10):
            num_fps_str = str(num_fps_loaded)[1:-1]
            self.print(
                f"Running [bold]single-round, serial (1 process)[/bold] clustering\n\n"
                f"- Branching factor: {config['branching_factor']:,}\n"
                f"- Merge criterion: [yellow]{config['merge_criterion']}[/yellow]\n"
                f"- Threshold: {config['threshold']}\n"
                f"- Num. files loaded: {len(config['input_files']):,}\n"
                f"- Num. fingerprints loaded for each file: {num_fps_str}\n"
                f"- Total num. fingerprints: {total_fps_num:,}\n"
                f"- Output directory: {config['out_dir']}\n",
                end="",
            )
        bb_variant = config.get("bitbirch_variant", "lean")
        max_files = config.get("max_files", None)
        max_fps = config.get("max_fps", None)
        if "tolerance" in config["merge_criterion"]:
            self.print(f"- Tolerance: {config['tolerance']}\n", end="")
        if config["refine_num"] > 0:
            self.print(
                f"- Will refine largest {config['refine_num']} clusters\n", end=""
            )
            self.print(f"- Num. clusters to refine: {config['refine_num']}\n", end="")
            self.print(
                "- Refine criterion: "
                f"[yellow]{config['refine_merge_criterion']}[/yellow]\n",
                end="",
            )
            if "tolerance" in config["refine_merge_criterion"]:
                self.print(f"- Refine tolerance: {config['tolerance']}\n", end="")
            self.print(
                f"- Refine threshold change: {config['refine_threshold_change']}\n",
                end="",
            )
        if bb_variant != "lean":
            self.print(
                "- [bold]DEBUG:[/bold] Using bitbirch version: {variant}\n", end=""
            )
        if max_files is not None:
            self.print(
                f"- [bold]DEBUG:[/bold] Max files to load: {max_files:,}\n", end=""
            )
        if max_fps is not None:
            self.print(
                f"- [bold]DEBUG:[/bold] Max fps to load per file: {max_fps:,}\n", end=""
            )
        self.print()

    def print_multiround_config(
        self, config: dict[str, tp.Any], mp_context: tp.Any = None
    ) -> None:
        if mp_context is None:
            mp_context = mp.get_context()
        num_processes = config.get("num_initial_processes", 1)
        extra_desc = (
            f"parallel (max {num_processes:,} processes)"
            if num_processes > 1
            else "serial (1 process)"
        )
        desc = f"multi-round, {extra_desc}"
        num_fps_loaded = np.array(config["num_fps_loaded"])
        total_fps_num = num_fps_loaded.sum()
        with np.printoptions(formatter={"int": "{:,}".format}, threshold=10):
            num_fps_str = str(num_fps_loaded)[1:-1]
            self.print(
                f"Running [bold]{desc}[/bold] clustering\n\n"
                f"- Branching factor: {config['branching_factor']:,}\n"
                f"- Initial round merge criterion: [yellow]{config['initial_merge_criterion']}[/yellow]\n"  # noqa:E501
                f"- Threshold: {config['threshold']}\n"
                f"- Tolerance: {config['tolerance']}\n"
                f"- Num. files loaded: {len(config['input_files']):,}\n"
                f"- Num. fingerprints loaded for each file: {num_fps_str}\n"
                f"- Total num. fingerprints: {total_fps_num:,}\n"
                f"- Output directory: {config['out_dir']}\n",
                end="",
            )
        full_refinement_before_midsection = config.get(
            "full_refinement_before_midsection", False
        )
        bb_variant = config.get("bitbirch_variant", "lean")
        max_files = config.get("max_files", None)
        bin_size = config.get("bin_size", None)
        max_fps = config.get("max_fps", None)
        if bin_size is not None:
            self.print(f"- Bin size for second round: {bin_size:,}\n", end="")
        if num_processes > 1:
            self.print(
                f"- Multiprocessing method: [yellow]{mp_context._name}[/yellow]\n",
                end="",
            )
        if not full_refinement_before_midsection:
            self.print(
                f"- Full refinement before midsection: {full_refinement_before_midsection}\n",  # noqa:E501
                end="",
            )
        if bb_variant != "lean":
            self.print(
                "- [bold]DEBUG:[/bold] Using bitbirch version: {variant}\n", end=""
            )
        if max_files is not None:
            self.print(
                f"- [bold]DEBUG:[/bold] Max files to load: {max_files:,}\n", end=""
            )
        if max_fps is not None:
            self.print(
                f"- [bold]DEBUG:[/bold] Max fps to load per file: {max_fps:,}\n", end=""
            )
        self.print()


class SilentConsole(BBConsole):
    def print(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        pass

    def print_peak_mem(self, out_dir: Path, indent: bool = True) -> None:
        pass

    def print_banner(self) -> None:
        pass

    def status(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        class DummyStatus:
            def __enter__(self) -> tp.Any:
                return self

            def __exit__(self, *args: tp.Any, **kwargs: tp.Any) -> None:
                pass

        return DummyStatus()


_console = BBConsole()
_silent_console = SilentConsole()


def get_console(silent: bool = False) -> BBConsole:
    if silent:
        return _silent_console
    return _console
