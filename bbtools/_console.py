import typing as tp
import multiprocessing as mp

from rich.console import Console

from bbtools.memory import get_peak_memory, PeakMemoryStats


class BBConsole(Console):
    def print_banner(self) -> None:
        banner = r"""
            ______ _ _  ______ _          _     
            | ___ (_) | | ___ (_)        | |    
            | |_/ /_| |_| |_/ /_ _ __ ___| |__  
            | ___ \ | __| ___ \ | '__/ __| '_ \ 
            | |_/ / | |_| |_/ / | | | (__| | | |
            \____/|_|\__\____/|_|_|  \___|_| |_|
        """  # noqa W291
        self.print(banner, highlight=False)

    def print_peak_mem(self, num_processes: int) -> None:
        stats = get_peak_memory(num_processes)
        if stats is None:
            self.print("[Peak memory stats not tracked for non-Unix systems]")
            return
        return self.print_peak_mem_raw(stats)

    def print_peak_mem_raw(self, stats: PeakMemoryStats) -> None:
        self.print("Peak RAM until now:\n" f"    Main proc.: {stats.self_gib:.4f} GiB")
        if not stats.children_were_tracked:
            self.print("    Max of child procs.: [Not tracked for 'forkserver']")
        elif stats.child_gib is not None:
            self.print(f"    Max of child procs.: {stats.child_gib:.4f} GiB")

    def print_config(
        self, config: dict[str, tp.Any], desc: str = "", add_processes: bool = False
    ) -> None:
        num_processes = config.get("num_processes", 1)
        if add_processes:
            extra_desc = (
                f"parallel (max {num_processes} processes)"
                if num_processes > 1
                else "serial (1 process)"
            )
            desc = f"{desc} {extra_desc}"
        self.print(
            f"Running [bold]{desc}[/bold] clustering\n\n"
            f"- Branching factor: {config['branching_factor']}\n"
            f"- Initial merge strategy: [yellow]{config['merge_criterion']}[/yellow]\n"
            f"- Threshold: {config['threshold']}\n"
            f"- Tolerance: {config['tolerance']}\n"
            f"- Num. files loaded: {len(config['input_files'])}\n"
            f"- Use mmap: {config['use_mmap']}\n"
            f"- Output directory: {config['out_dir']}\n",
            end="",
        )
        double_cluster_init = config.get("double_cluster_init", False)
        bb_variant = config.get("bitbirch_variant", "lean")
        max_files = config.get("max_files", None)
        max_fps = config.get("max_fps", None)
        bin_size = config.get("bin_size", None)
        if bin_size is not None:
            self.print(f"- Bin size for second round: {bin_size}\n", end="")
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
            self.print("- DEBUG: Using bitbirch version: {variant}\n", end="")
        if max_files is not None:
            self.print(
                f"- DEBUG: Max files to load: {max_files}\n", end=""
            )  # noqa:E501
        if max_fps is not None:
            self.print(
                f"- DEBUG: Max fingerprints to load per file: {max_fps}\n", end=""
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
