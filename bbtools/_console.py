import typing as tp

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
