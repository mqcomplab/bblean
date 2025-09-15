import typing as tp
import os
import multiprocessing as mp

from rich.console import Console

from bblean.memory import get_peak_memory, PeakMemoryStats


class BBConsole(Console):
    def print_banner(self) -> None:
        if os.environ.get("BITBIRCHNOBANNER", ""):
            return
        banner = r"""[bold]
            ______ _ _  ______ _          _        
            | ___ (_) | | ___ (_)        | |     [/bold][cyan]   ______                      [/cyan][bold] 
            | |_/ /_| |_| |_/ /_ _ __ ___| |__   [/bold][cyan]   ___  / ___________ _______  [/cyan][bold] 
            | ___ \ | __| ___ \ | '__/ __| '_ \  [/bold][cyan]   __  /  _  _ \  __ `/_  __ \ [/cyan][bold] 
            | |_/ / | |_| |_/ / | | | (__| | | | [/bold][cyan]   _  /___/  __/ /_/ /_  / / / [/cyan][bold] 
            \____/|_|\__\____/|_|_|  \___|_| |_| [/bold][cyan]   /_____/\___/\__,_/ /_/ /_/  [/cyan][bold] 
        [/bold]
        """  # noqa W291
        self.print(banner, highlight=False)
        self.print(
            r"""
If you find this work useful please cite the following articles:
    - [italic]BitBIRCH: efficient clustering of large molecular libraries[/italic]:
        https://doi.org/10.1039/D5DD00030K
    - [italic]BitBIRCH Clustering Refinement Strategies[/italic]:
        https://doi.org/10.1021/acs.jcim.5c00627
    - [italic]BitBIRCH-Lean[/italic]:
        (TODO)"""  # noqa
        )
        self.print()

    def print_peak_mem(self, num_processes: int) -> None:
        stats = get_peak_memory(num_processes)
        if stats is None:
            self.print("[Peak RAM not tracked for non-Unix systems]")
            return
        return self.print_peak_mem_raw(stats)

    def print_peak_mem_raw(self, stats: PeakMemoryStats) -> None:
        self.print("Peak RAM use:\n" f"    Main proc.: {stats.self_gib:.4f} GiB")
        if stats.child_gib is not None:
            self.print(f"    Max of child procs.: {stats.child_gib:.4f} GiB")

    def print_config(self, config: dict[str, tp.Any]) -> None:
        num_fps_loaded = ", ".join(f"{num:,}" for num in config["num_fps_loaded"])
        self.print(
            f"Running [bold]single-round, serial (1 process)[/bold] clustering\n\n"
            f"- Branching factor: {config['branching_factor']:,}\n"
            f"- Merge criterion: [yellow]{config['merge_criterion']}[/yellow]\n"
            f"- Threshold: {config['threshold']}\n"
            f"- Num. files loaded: {len(config['input_files']):,}\n"
            f"- Num. fingerprints loaded per file: [{num_fps_loaded}]\n"
            f"- Use mmap: {config['use_mmap']}\n"
            f"- Output directory: {config['out_dir']}\n",
            end="",
        )
        bb_variant = config.get("bitbirch_variant", "lean")
        max_files = config.get("max_files", None)
        max_fps = config.get("max_fps", None)
        if "tolerance" in config["merge_criterion"]:
            self.print(f"- Tolerance: {config['tolerance']}\n")
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

    def print_multiround_config(self, config: dict[str, tp.Any]) -> None:
        num_processes = config.get("num_processes", 1)
        extra_desc = (
            f"parallel (max {num_processes:,} processes)"
            if num_processes > 1
            else "serial (1 process)"
        )
        desc = f"multi-round, {extra_desc}"
        num_fps_loaded = ", ".join(f"{num:,}" for num in config["num_fps_loaded"])
        self.print(
            f"Running [bold]{desc}[/bold] clustering\n\n"
            f"- Branching factor: {config['branching_factor']:,}\n"
            f"- Round-1 merge criterion: [yellow]{config['merge_criterion']}[/yellow]\n"
            f"- Threshold: {config['threshold']}\n"
            f"- Tolerance: {config['tolerance']}\n"
            f"- Num. files loaded: {len(config['input_files']):,}\n"
            f"- Num. fingerprints loaded per file: [{num_fps_loaded}]\n"
            f"- Use mmap: {config['use_mmap']}\n"
            f"- Output directory: {config['out_dir']}\n",
            end="",
        )
        double_cluster_init = config.get("double_cluster_init", False)
        bb_variant = config.get("bitbirch_variant", "lean")
        max_files = config.get("max_files", None)
        bin_size = config.get("bin_size", None)
        max_fps = config.get("max_fps", None)
        if bin_size is not None:
            self.print(f"- Bin size for second round: {bin_size:,}\n", end="")
        if num_processes > 1:
            self.print(
                f"- Multiprocessing method: [yellow]{mp.get_start_method()}[/yellow]\n",
                end="",
            )
        if not double_cluster_init:
            self.print(
                f"- Use double-cluster-init: {double_cluster_init}\n",
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

    def print_peak_mem(self, num_processes: int) -> None:
        pass

    def print_peak_mem_raw(self, stats: PeakMemoryStats) -> None:
        pass

    def print_banner(self) -> None:
        pass


_console = BBConsole()
_silent_console = SilentConsole()


def get_console(silent: bool = False) -> BBConsole:
    if silent:
        return _silent_console
    return _console
