r"""Command line interface entrypoints"""

import time
from typing import Annotated
from pathlib import Path

from typer import Typer, Argument, Option, Abort

from bbtools._console import get_console

app = Typer(
    rich_markup_mode="markdown",
    help=r"""## BitBirch

    CLI interface for serial and parallel fast clustering of molecular fingerprints
    using the O(N) BitBirch algorithm.

    If you find this work useful please cite the BitBirch article:
    https://doi.org/10.1039/D5DD00030K
    """,
)

console = get_console()


@app.command("make-fps")
def _make_fps(
    dtype: Annotated[
        str,
        Option("-d", "--dtype", help="NumPy dtype for the generated fingerprints"),
    ] = "uint8",
    packed: Annotated[
        bool,
        Option(
            "-p/-P",
            "--packed/--no-packed",
            help="Pack bits in last dimension of fingerprints",
        ),
    ] = True,
    smiles_paths: Annotated[
        list[Path] | None,
        Option("-s", "--smiles-path", show_default=False),
    ] = None,
    out_dir: Annotated[
        Path | None,
        Option("-o", "--out-dir", show_default=False),
    ] = None,
    kind: Annotated[
        str,
        Option("-k", "--kind"),
    ] = "rdkit",
    fp_size: Annotated[
        int,
        Option("-f", "--fp-size"),
    ] = 2048,
) -> None:
    r"""Generate a *.npy fingerprints file from one or more smiles files

    In order to use the memory efficient BitBirch u8 algorithm you *must* use the
    defaults: --dtype=uint8 and --packed
    """
    import numpy as np
    from rdkit.Chem import rdFingerprintGenerator, DataStructs, MolFromSmiles

    if kind not in ["rdkit", "ecfp4"]:
        console.print("Kind must be one of 'rdkit|ecfp4'", style="red")
        raise Abort()

    if kind == "rdkit":
        fpg = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=fp_size)
    elif kind == "ecfp4":
        fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_size)

    if smiles_paths is None:
        smiles_paths = list(Path.cwd().glob("*.smi"))
    if not smiles_paths:
        console.print("No *.smi files found", style="red")
        raise Abort()
    if out_dir is None:
        out_dir = Path.cwd() / ("packed-fps" if packed else "fps")
        out_dir.mkdir()

    # Pass 1: check the total number of smiles
    smiles_num = 0
    for smi_path in smiles_paths:
        with open(smi_path, mode="rt", encoding="utf-8") as f:
            for _ in f:
                smiles_num += 1

    # Pass 2: build the molecules
    mols = []
    for smi_path in smiles_paths:
        with open(smi_path, mode="rt", encoding="utf-8") as f:
            for i, smi in enumerate(f):
                mol = MolFromSmiles(smi)
                if mol is None:
                    console.print(
                        f"Invalid smiles {smi} from {str(smi_path)} (line {i + 1})"
                    )
                    raise Abort()
                mols.append(mol)

    # Pass 3: build the fingerprints
    fps = np.empty((len(mols), fp_size), dtype=dtype)
    for i, fp in enumerate(fpg.GetFingerprints(mols)):
        DataStructs.ConvertToNumpyArray(fp, fps[i, :])

    # Save the fingerprints as a NumPy array
    np.save(out_dir / f"{'packed-' if packed else ''}fps-{dtype}", fps)


@app.command("split-fps")
def _split_fps() -> None:
    r"""Split a *.npy fingerprint file into multiple files"""
    raise NotImplementedError("Not yet implemented")


@app.command("run")
def _run(
    fp_file: Annotated[
        Path,
        Argument(help="*.npy file with packed fingerprints"),
    ],
    max_fps: Annotated[
        int | None,
        Option("-m", "--max-fps"),
    ] = None,
    branching_factor: Annotated[
        int,
        Option("-b", "--branching-factor"),
    ] = 254,
    use_mmap: Annotated[
        bool,
        Option(help="Toggle mmap of the fingerprint files", rich_help_panel="Advanced"),
    ] = True,
    threshold: Annotated[
        float,
        Option("-t", "--threshold"),
    ] = 0.65,
    merge_strategy: Annotated[
        str,
        Option("-s", "--set-merge"),
    ] = "diameter",
    # Debug options
    use_old_bblean: Annotated[
        bool,
        Option("--use-bblean-v1/--no-use-bblean-v1", rich_help_panel="Debug"),
    ] = False,
) -> None:
    r"""Run standard BitBirch clustering"""
    import numpy as np

    if use_old_bblean:
        from bbtools.bblean_v1 import BitBirch, set_merge  # type: ignore
    else:
        from bbtools.bblean import BitBirch, set_merge  # type: ignore

    fps = np.load(fp_file, mmap_mode="r" if use_mmap else None)[:max_fps]

    set_merge(merge_strategy)
    brc = BitBirch(branching_factor=branching_factor, threshold=threshold)
    _start = time.perf_counter()
    brc.fit_reinsert(fps, list(range(len(fps))))
    console.print(f"Time elapsed: {time.perf_counter() - _start} s")
    console.print_peak_mem(num_processes=1)


@app.command("parallel")
def _parallel(
    out_dir: Annotated[
        Path | None,
        Option("-o", "--out-dir", help="Dir for output files"),
    ] = None,
    in_dir: Annotated[
        Path | None,
        Option(
            "-i",
            "--in-dir",
            help="Dir with input *.npy files with packed fingerprints",
        ),
    ] = None,
    overwrite_outputs: Annotated[
        bool, Option(help="Allow overwriting output files")
    ] = True,
    filename_idxs_are_slices: Annotated[
        bool,
        Option(
            help="Filename idxs are slices, e.g. *_1000_2000.npy, *_2000_3000.npy, ...",
        ),
    ] = False,
    round2_process_fraction: Annotated[
        float,
        Option(
            help="Fraction of processes to use for second round clustering. "
            "Second round clustering can be very memory intensive, "
            "so it may be desirable to use a value of 0.5-0.33 if multiprocessing.",
        ),
    ] = 1.0,
    num_processes: Annotated[
        int, Option("-p", "--processes", help="Num. processes")
    ] = 10,
    branching_factor: Annotated[int, Option(help="BitBirch branching factor")] = 254,
    threshold: Annotated[float, Option(help="BitBirch threshold")] = 0.65,
    tolerance: Annotated[
        float,
        Option(
            help="BitBirch tolerance"
            " (Used in Round 1 'double-cluster-init', Round 2, and Final clustering)"
        ),
    ] = 0.05,
    # Advanced options
    initial_merge_strategy: Annotated[
        str,
        Option(
            "--set-merge",
            help="Initial merge strategy for Round 1 ('diameter' is recommended)",
            rich_help_panel="Advanced",
        ),
    ] = "diameter",
    double_cluster_init: Annotated[
        bool,
        Option(
            help="Toggle 'double-cluster-init' ('True' is recommended)",
            rich_help_panel="Advanced",
        ),
    ] = True,
    max_tasks_per_process: Annotated[
        int, Option(help="Max tasks per process", rich_help_panel="Advanced")
    ] = 1,
    use_mmap: Annotated[
        bool,
        Option(help="Toggle mmap of the fingerprint files", rich_help_panel="Advanced"),
    ] = True,
    fork: Annotated[
        bool,
        Option(
            help="In linux, force the 'fork' multiposcessing start method",
            rich_help_panel="Advanced",
        ),
    ] = False,
    bin_size: Annotated[
        int,
        Option(help="Bin size for chunking during Round 2", rich_help_panel="Advanced"),
    ] = 10,
    # Debug options
    use_old_bblean: Annotated[
        bool,
        Option("--use-bblean-v1/--no-use-bblean-v1", rich_help_panel="Debug"),
    ] = False,
    only_first_round: Annotated[
        bool,
        Option(
            help="Only do first round clustering and exit early",
            rich_help_panel="Debug",
        ),
    ] = False,
    monitor_rss: Annotated[
        bool,
        Option(
            help="Monitor RAM used by all processes (requires psutil)",
            rich_help_panel="Debug",
        ),
    ] = True,
    monitor_rss_interval_s: Annotated[
        float,
        Option(
            "--monitor-rss-seconds",
            help="Interval in seconds for RSS monitoring",
            rich_help_panel="Debug",
        ),
    ] = 0.01,
    monitor_rss_file_name: Annotated[
        str,
        Option(
            "--monitor-rss-filename",
            help="File name for RSS monitoring",
            rich_help_panel="Debug",
        ),
    ] = "monitor-rss.csv",
    max_fps: Annotated[
        int | None,
        Option(
            help="Max num. of fps to load from each input file",
            rich_help_panel="Debug",
        ),
    ] = None,
    max_files: Annotated[
        int | None, Option(help="Max num. files to read", rich_help_panel="Debug")
    ] = None,
    verbose: Annotated[
        bool,
        Option("-v/-V", "--verbose/--no-verbose"),
    ] = True,
) -> None:
    r"""Run multi-round BitBirch clustering in parallel"""
    from bbtools.parallel import run_parallel_bitbirch

    if out_dir is None:
        out_dir = Path.cwd() / "bb_outputs"
    if in_dir is None:
        in_dir = Path.cwd() / "bb_inputs"
    run_parallel_bitbirch(
        in_dir=in_dir,
        out_dir=out_dir,
        overwrite_outputs=overwrite_outputs,
        filename_idxs_are_slices=filename_idxs_are_slices,
        round2_process_fraction=round2_process_fraction,
        initial_merge_strategy=initial_merge_strategy,
        num_processes=num_processes,
        branching_factor=branching_factor,
        threshold=threshold,
        tolerance=tolerance,
        # Advanced
        fork=fork,
        bin_size=bin_size,
        use_mmap=use_mmap,
        max_tasks_per_process=max_tasks_per_process,
        double_cluster_init=double_cluster_init,
        # Debug
        only_first_round=only_first_round,
        monitor_rss=monitor_rss,
        monitor_rss_file_name=monitor_rss_file_name,
        monitor_rss_interval_s=monitor_rss_interval_s,
        max_fps=max_fps,
        max_files=max_files,
        use_old_bblean=use_old_bblean,
        verbose=verbose,
    )
