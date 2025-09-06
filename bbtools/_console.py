import typing as tp
from rich.console import Console


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


class SilentConsole(BBConsole):
    def print(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        pass


_console = BBConsole()
_silent_console = SilentConsole()


def get_console(silent: bool = False) -> BBConsole:
    if silent:
        return _silent_console
    return _console
