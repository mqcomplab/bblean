r"""Command line interface entrypoints"""

import typing as tp
import math
import shutil
import json
import sys
import pickle
import uuid
import multiprocessing as mp
from typing import Annotated
from pathlib import Path

import numpy as np
from typer import Typer, Argument, Option, Abort, Context, Exit

from bblean._memory import launch_monitor_rss_daemon, get_peak_memory
from bblean._timer import Timer
from bblean._config import DEFAULTS, collect_system_specs_and_dump_config
from bblean.utils import _import_bitbirch_variant, batched

app = Typer(
    rich_markup_mode="markdown",
    add_completion=False,
    help=r"""CLI tool for serial or parallel fast clustering of molecular fingerprints
    using the memory-efficient and compute-efficient *O(N)* BitBIRCH algorithm ('Lean'
    version). For more info about the subcommands run `bb <subcommand> --help `.""",
)


def _print_help_banner(ctx: Context, value: bool) -> None:
    if value:
        from bblean._console import get_console

        console = get_console()
        console.print_banner()
        console.print(ctx.get_help())
        raise Exit()


def _validate_output_dir(out_dir: Path, overwrite: bool = False) -> None:
    if out_dir.exists():
        if not out_dir.is_dir():
            raise RuntimeError("Output dir should be a dir")
        if any(out_dir.iterdir()):
            if overwrite:
                shutil.rmtree(out_dir)
            else:
                raise RuntimeError(f"Output dir {out_dir} has files")


# Validate that the naming convention for the input files is correct
def _validate_input_dir(in_dir: Path | str) -> None:
    in_dir = Path(in_dir)
    if not in_dir.is_dir():
        raise RuntimeError(f"Input dir {in_dir} should be a dir")
    if not any(in_dir.glob("*.npy")):
        raise RuntimeError(f"Input dir {in_dir} should have *.npy fingerprint files")


@app.callback()
def _main(
    ctx: Context,
    help_: bool = Option(
        None,
        "--help",
        "-h",
        is_flag=True,
        is_eager=True,
        help="Show this message and exit.",
        callback=_print_help_banner,
    ),
) -> None:
    pass


@app.command("summary-plot")
def _summary_plot(
    clusters_path: Annotated[Path, Option("-c", "--clusters-path", show_default=False)],
    fps_path: Annotated[Path, Option("-f", "--fps-path", show_default=False)],
    smiles_path: Annotated[Path, Option("-s", "--smiles-path", show_default=False)],
    use_mmap: Annotated[
        bool,
        Option("--use-mmap/--no-use-mmap"),
    ] = True,
    title: Annotated[
        str | None,
        Option("--title"),
    ] = None,
    top: Annotated[
        int,
        Option("--top"),
    ] = 20,
    input_is_packed: Annotated[
        bool,
        Option("--packed-input/--unpacked-input"),
    ] = True,
    scaffold_fp_kind: Annotated[
        str,
        Option("--scaffold-fp-kind"),
    ] = "rdkit",
    n_features: Annotated[
        int | None,
        Option(
            "--n-features",
            help="Number of features in the fingerprints."
            " It must be provided for packed inputs *if it is not a multiple of 8*."
            " For typical fingerprint sizes (e.g. 2048, 1024), it is not required",
            rich_help_panel="Advanced",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        Option("-v/-V", "--verbose/--no-verbose"),
    ] = True,
) -> None:
    r"""Summary plot of the clustering results"""
    from bblean._console import get_console

    console = get_console(silent=not verbose)
    # Imports may take a bit of time since sklearn is slow, so start the spinner here
    with console.status("[italic]Analyzing clusters...[/italic]", spinner="dots"):
        import matplotlib.pyplot as plt

        from bblean.smiles import load_smiles
        from bblean.analysis import cluster_analysis
        from bblean.plotting import summary_plot

        if clusters_path.is_dir():
            clusters_path = clusters_path / "clusters.pkl"
        with open(clusters_path, mode="rb") as f:
            clusters = pickle.load(f)
        fps = np.load(fps_path, mmap_mode="r" if use_mmap else None)
        smiles = load_smiles(smiles_path)
        ca = cluster_analysis(
            clusters,
            smiles,
            fps,
            top=top,
            n_features=n_features,
            input_is_packed=input_is_packed,
            scaffold_fp_kind=scaffold_fp_kind,
        )
        summary_plot(ca, title)
    plt.show()


@app.command("run")
def _run(
    ctx: Context,
    input_: Annotated[
        Path | None,
        Argument(help="`*.npy` file with packed fingerprints, or dir `*.npy` files"),
    ] = None,
    out_dir: Annotated[
        Path | None,
        Option(
            "-o",
            "--out-dir",
            help="Dir to dump the output files",
        ),
    ] = None,
    overwrite: Annotated[bool, Option(help="Allow overwriting output files")] = False,
    branching_factor: Annotated[
        int,
        Option(
            help="BitBIRCH branching factor. Under most circumstances 254 is"
            " optimal for performance and memory efficiency. Set this above 254 for"
            " slightly less RAM usage at the cost of some performance."
        ),
    ] = DEFAULTS.branching_factor,
    threshold: Annotated[
        float,
        Option("--threshold"),
    ] = DEFAULTS.threshold,
    merge_criterion: Annotated[
        str,
        Option("--set-merge"),
    ] = "diameter",
    tolerance: Annotated[
        float,
        Option(
            help="BitBIRCH tolerance, only for --set-merge tolerance|tolerance_tough"
        ),
    ] = DEFAULTS.tolerance,
    use_mmap: Annotated[
        bool,
        Option(
            help="Toggle mmap of the fingerprint files (True recommended)",
            rich_help_panel="Advanced",
        ),
    ] = DEFAULTS.use_mmap,
    n_features: Annotated[
        int | None,
        Option(
            "--n-features",
            help="Number of features in the fingerprints."
            " It must be provided for packed inputs *if it is not a multiple of 8*."
            " For typical fingerprint sizes (e.g. 2048, 1024), it is not required",
            rich_help_panel="Advanced",
        ),
    ] = None,
    input_is_packed: Annotated[
        bool,
        Option(
            "--packed-input/--unpacked-input",
            help="Toggle whether the input consists on packed or unpacked fingerprints",
            rich_help_panel="Advanced",
        ),
    ] = True,
    # Debug options
    monitor_rss: Annotated[
        bool,
        Option(
            help="Monitor RAM used by all processes (requires psutil)",
            rich_help_panel="Debug",
        ),
    ] = False,
    monitor_rss_interval_s: Annotated[
        float,
        Option(
            "--monitor-rss-seconds",
            help="Interval in seconds for RSS monitoring",
            rich_help_panel="Debug",
        ),
    ] = 0.01,
    max_fps: Annotated[
        int | None,
        Option(
            "--max-fps",
            rich_help_panel="Debug",
            help="Max. num of fingerprints to read from each file",
        ),
    ] = None,
    variant: Annotated[
        str,
        Option(
            "--bb-variant",
            help="Use different bitbirch variants, *only for debugging*.",
            hidden=True,
        ),
    ] = "lean",
    verbose: Annotated[
        bool,
        Option("-v/-V", "--verbose/--no-verbose"),
    ] = True,
) -> None:
    r"""Run standard, serial BitBIRCH clustering over `*.npy` fingerprint files"""
    # TODO: Remove code duplication with multiround
    import numpy as np
    from bblean._console import get_console
    from bblean.fingerprints import _get_fps_file_num

    console = get_console(silent=not verbose)
    if variant == "int64_dense" and input_is_packed:
        raise ValueError("Packed inputs are not supported for the int64_dense variant")

    BitBirch, set_merge = _import_bitbirch_variant(variant)

    # NOTE: Files are sorted according to name
    if input_ is None:
        input_ = Path.cwd() / "bb_inputs"
        input_.mkdir(exist_ok=True)
        input_files = sorted(input_.glob("*.npy"))
        _validate_input_dir(input_)
    elif input_.is_dir():
        input_files = sorted(input_.glob("*.npy"))
        _validate_input_dir(input_)
    else:
        input_files = [input_]
    ctx.params.pop("input_")
    ctx.params["input_files"] = [str(p.resolve()) for p in input_files]
    ctx.params["num_fps_present"] = [_get_fps_file_num(p) for p in input_files]
    if max_fps is not None:
        ctx.params["num_fps_loaded"] = [
            min(n, max_fps) for n in ctx.params["num_fps_present"]
        ]
    else:
        ctx.params["num_fps_loaded"] = ctx.params["num_fps_present"]
    unique_id = str(uuid.uuid4()).split("-")[0]
    if out_dir is None:
        out_dir = Path.cwd() / "bb_run_outputs" / unique_id
    out_dir.mkdir(exist_ok=True, parents=True)
    _validate_output_dir(out_dir, overwrite)
    ctx.params["out_dir"] = str(out_dir.resolve())

    console.print_banner()
    console.print()
    console.print_config(ctx.params)

    # Optinally start a separate process that tracks RAM usage
    if monitor_rss:
        launch_monitor_rss_daemon(out_dir / "monitor-rss.csv", monitor_rss_interval_s)

    timer = Timer()
    timer.init_timing("total")
    if "lean" not in variant:
        set_merge(merge_criterion, tolerance)
        tree = BitBirch(branching_factor=branching_factor, threshold=threshold)
    else:
        tree = BitBirch(
            branching_factor=branching_factor,
            threshold=threshold,
            merge_criterion=merge_criterion,
            tolerance=tolerance,
        )
    with console.status("[italic]BitBirching...[/italic]", spinner="dots"):
        for file in input_files:
            fps = np.load(file, mmap_mode="r" if use_mmap else None)[:max_fps]
            tree.fit(fps, n_features=n_features, input_is_packed=input_is_packed)

        # TODO: Fix peak memory stats
        cluster_mol_ids = tree.get_cluster_mol_ids()
    timer.end_timing("total", console)
    stats = get_peak_memory(1)
    if stats is None:
        console.print("[Peak memory stats not tracked for non-Unix systems]")
    else:
        console.print_peak_mem_raw(stats)

    # Dump outputs (peak memory, timings, config, cluster ids)
    with open(out_dir / "clusters.pkl", mode="wb") as f:
        pickle.dump(cluster_mol_ids, f)

    collect_system_specs_and_dump_config(ctx.params)
    timer.dump(out_dir / "timings.json")

    peak_rss_fpath = out_dir / "peak-rss.json"
    with open(peak_rss_fpath, mode="wt", encoding="utf-8") as f:
        json.dump(
            {"self_max_rss_gib": None if stats is None else stats.self_gib}, f, indent=4
        )


# TODO: Currently sometimes after a round is triggered *more* files are output, since
# the files are divided *both* by uint8/uint16 and the batch idx. I believe this is not
# ideal
@app.command("multiround")
def _multiround(
    ctx: Context,
    in_dir: Annotated[
        Path | None,
        Argument(help="Directory with input `*.npy` files with packed fingerprints"),
    ] = None,
    out_dir: Annotated[
        Path | None,
        Option("-o", "--out-dir", help="Dir for output files"),
    ] = None,
    overwrite: Annotated[bool, Option(help="Allow overwriting output files")] = False,
    num_initial_processes: Annotated[
        int, Option("--ps", "--processes", help="Num. processes for first round")
    ] = 10,
    num_midsection_processes: Annotated[
        int | None,
        Option(
            "--mid-ps",
            "--mid-processes",
            help="Num. processes to use for the middle section (if multiprocessing)."
            "Middle section clustering can be very memory intensive, "
            "so it may be desirable to use 50%-30% of the first round processes",
        ),
    ] = None,
    branching_factor: Annotated[
        int,
        Option(
            help="BitBIRCH branching factor. Under most circumstances 254 is"
            " optimal for performance and memory efficiency. Set this above 254 for"
            " slightly less RAM usage at the cost of some performance."
        ),
    ] = DEFAULTS.branching_factor,
    threshold: Annotated[float, Option(help="BitBIRCH threshold")] = DEFAULTS.threshold,
    tolerance: Annotated[
        float,
        Option(
            help="BitBIRCH tolerance"
            " (Used in Round 1 'double-cluster-init', Round 2, and Final clustering)"
        ),
    ] = DEFAULTS.tolerance,
    initial_merge_criterion: Annotated[
        str,
        Option(
            "--set-merge",
            help="Initial merge criterion for Round 1 ('diameter' is recommended)",
        ),
    ] = DEFAULTS.merge_criterion,
    n_features: Annotated[
        int | None,
        Option(
            "--n-features",
            help="Number of features in the fingerprints."
            " It must be provided for packed inputs *if it is not a multiple of 8*."
            " For typical fingerprint sizes (e.g. 2048, 1024), it is not required",
            rich_help_panel="Advanced",
        ),
    ] = None,
    input_is_packed: Annotated[
        bool,
        Option(
            "--packed-input/--unpacked-input",
            help="Toggle whether the input consists on packed or unpacked fingerprints",
            rich_help_panel="Advanced",
        ),
    ] = True,
    # Advanced options
    num_midsection_rounds: Annotated[
        int,
        Option(
            "--num-midsection-rounds", help="Number of midsection rounds to perform"
        ),
    ] = 1,
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
        Option(
            help="Toggle mmap of the fingerprint files (True recommended)",
            rich_help_panel="Advanced",
        ),
    ] = DEFAULTS.use_mmap,
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
    variant: Annotated[
        str,
        Option(
            "--bb-variant",
            help="Use different bitbirch variants, *only for debugging*.",
            hidden=True,
        ),
    ] = "lean",
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
    ] = False,
    monitor_rss_interval_s: Annotated[
        float,
        Option(
            "--monitor-rss-seconds",
            help="Interval in seconds for RSS monitoring",
            rich_help_panel="Debug",
        ),
    ] = 0.01,
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
    r"""Run multi-round BitBIRCH clustering, optionally parallelize over `*.npy` files"""  # noqa:E501
    from bblean._console import get_console
    from bblean.multiround import run_multiround_bitbirch
    from bblean.fingerprints import _get_fps_file_num

    console = get_console(silent=not verbose)

    # Set multiprocessing start method
    if fork and not sys.platform == "linux":
        console.print("'fork' is only available on Linux", style="red")
        raise Abort()
    if sys.platform == "linux":
        mp.set_start_method("fork" if fork else "forkserver")

    # Collect inputs:
    # If not passed, input dir is bb_inputs/
    if in_dir is None:
        in_dir = Path.cwd() / "bb_inputs"
    _validate_input_dir(in_dir)
    # All files in the input dir with *.npy suffix are considered input files
    input_files = sorted(in_dir.glob("*.npy"))[:max_files]
    ctx.params["input_files"] = [str(p.resolve()) for p in input_files]
    ctx.params["num_fps"] = [_get_fps_file_num(p) for p in input_files]
    if max_fps is not None:
        ctx.params["num_fps_loaded"] = [min(n, max_fps) for n in ctx.params["num_fps"]]
    else:
        ctx.params["num_fps_loaded"] = ctx.params["num_fps"]

    # Set up outputs:
    # If not passed, output dir is constructed as bb_multiround_outputs/<unique-id>/
    unique_id = str(uuid.uuid4()).split("-")[0]
    if out_dir is None:
        out_dir = Path.cwd() / "bb_multiround_outputs" / unique_id
    out_dir.mkdir(exist_ok=True, parents=True)
    _validate_output_dir(out_dir, overwrite)
    ctx.params["out_dir"] = str(out_dir.resolve())

    console.print_banner()
    console.print()
    console.print_multiround_config(ctx.params)

    # Optinally start a separate process that tracks RAM usage
    if monitor_rss:
        launch_monitor_rss_daemon(out_dir / "monitor-rss.csv", monitor_rss_interval_s)

    timer = run_multiround_bitbirch(
        input_files=input_files,
        n_features=n_features,
        input_is_packed=input_is_packed,
        out_dir=out_dir,
        initial_merge_criterion=initial_merge_criterion,
        num_initial_processes=num_initial_processes,
        num_midsection_processes=num_midsection_processes,
        num_midsection_rounds=num_midsection_rounds,
        branching_factor=branching_factor,
        threshold=threshold,
        tolerance=tolerance,
        # Advanced
        bin_size=bin_size,
        use_mmap=use_mmap,
        max_tasks_per_process=max_tasks_per_process,
        double_cluster_init=double_cluster_init,
        # Debug
        only_first_round=only_first_round,
        max_fps=max_fps,
        verbose=verbose,
    )
    timer.dump(out_dir / "timings.json")
    # TODO: Also dump peak-rss.json
    collect_system_specs_and_dump_config(ctx.params)


@app.command("fps-info")
def _fps_info(
    fp_paths: Annotated[
        list[Path] | None,
        Argument(show_default=False, help="Paths to *.smi files with smiles"),
    ] = None,
) -> None:
    """Show info about a `*.npy` fingerprint file, or a dir with `*.npy` files"""
    from bblean._console import get_console
    from bblean.fingerprints import _print_fps_file_info

    console = get_console()
    if fp_paths is None:
        fp_paths = [Path.cwd()]

    for path in fp_paths:
        if path.is_dir():
            for file in path.glob("*.npy"):
                _print_fps_file_info(file, console)
        elif path.suffix == ".npy":
            _print_fps_file_info(file, console)


@app.command("fps-from-smiles")
def _fps_from_smiles(
    smiles_paths: Annotated[
        list[Path] | None,
        Argument(show_default=False, help="Paths to *.smi files with smiles"),
    ] = None,
    out_dir: Annotated[
        Path | None,
        Option("-o", "--out-dir", show_default=False),
    ] = None,
    out_name: Annotated[
        str | None,
        Option("--name", help="Base name of output file"),
    ] = None,
    kind: Annotated[
        str,
        Option("-k", "--kind"),
    ] = "rdkit",
    fp_size: Annotated[
        int,
        Option("--n-features", help="Num. features of the generated fingerprints"),
    ] = DEFAULTS.n_features,
    parts: Annotated[
        int | None,
        Option(
            "-n", "--num-parts", help="Split the created file into this number of parts"
        ),
    ] = None,
    max_fps_per_file: Annotated[
        int | None,
        Option(
            "-m",
            "--max-fps",
            help="Max. number of fps per file. Mutually exclusive with --num-parts",
            show_default=False,
        ),
    ] = None,
    pack: Annotated[
        bool,
        Option(
            "-p/-P",
            "--pack/--no-pack",
            help="Pack bits in last dimension of fingerprints",
            rich_help_panel="Advanced",
        ),
    ] = True,
    dtype: Annotated[
        str,
        Option(
            "-d",
            "--dtype",
            help="NumPy dtype for the generated fingerprints",
            rich_help_panel="Advanced",
        ),
    ] = "uint8",
    verbose: Annotated[
        bool,
        Option("-v/-V", "--verbose/--no-verbose"),
    ] = True,
) -> None:
    r"""Generate a `*.npy` fingerprints file from one or more `*.smi` smiles files

    In order to use the memory efficient BitBIRCH u8 algorithm you *must* use the
    defaults: --dtype=uint8 and --pack
    """
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator, DataStructs, MolFromSmiles

    from bblean._console import get_console
    from bblean.fingerprints import pack_fingerprints

    def iter_mols_from_paths(smiles_paths: tp.Iterable[Path]) -> tp.Iterator[Chem.Mol]:
        for smi_path in smiles_paths:
            with open(smi_path, mode="rt", encoding="utf-8") as f:
                for i, smi in enumerate(f):
                    mol = MolFromSmiles(smi)
                    if mol is None:
                        console.print(
                            f"Invalid smiles {smi} from {str(smi_path)} (line {i + 1})"
                        )
                        raise Abort()
                    yield mol

    console = get_console(silent=not verbose)

    if smiles_paths is None:
        smiles_paths = list(Path.cwd().glob("*.smi"))
    if not smiles_paths:
        console.print("No *.smi files found", style="red")
        raise Abort()

    if kind not in ["rdkit", "ecfp4", "ecfp6"]:
        console.print("Kind must be one of 'rdkit|ecfp4|ecfp6'", style="red")
        raise Abort()

    if kind == "rdkit":
        fpg = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=fp_size)
    elif kind == "ecfp4":
        fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_size)
    elif kind == "ecfp6":
        fpg = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=fp_size)

    # Pass 1: check the total number of smiles
    smiles_num = 0
    for smi_path in smiles_paths:
        with open(smi_path, mode="rt", encoding="utf-8") as f:
            for _ in f:
                smiles_num += 1

    digits: int | None
    if parts is not None and max_fps_per_file is None:
        num_per_batch = math.ceil(smiles_num / parts)
        digits = len(str(parts))
    elif parts is None and max_fps_per_file is not None:
        num_per_batch = max_fps_per_file
        digits = len(str(math.ceil(smiles_num / max_fps_per_file)))
    elif parts is None and max_fps_per_file is None:
        parts = 1
        num_per_batch = math.ceil(smiles_num / parts)
        digits = None
    else:
        console.print(
            "One and only one of '--max-fps' and '--num-parts' required", style="red"
        )
        raise Abort()

    if out_dir is None:
        out_dir = Path.cwd()
    out_dir.mkdir(exist_ok=True)
    out_dir = out_dir.resolve()

    # Pass 2: build the molecules
    if out_name is None:
        unique_id = str(uuid.uuid4()).split("-")[0]
        # Save the fingerprints as a NumPy array
        out_name = f"{'packed-' if pack else ''}fps-{dtype}-{unique_id}"
    else:
        # Strip suffix
        if out_name.endswith(".npy"):
            out_name = out_name[:-4]
    with console.status("[italic]Generating fingerprints...[/italic]", spinner="dots"):
        for file_idx, mol_batch in enumerate(
            batched(iter_mols_from_paths(smiles_paths), num_per_batch)
        ):
            fps = np.empty((len(mol_batch), fp_size), dtype=dtype)
            for i, fp in enumerate(fpg.GetFingerprints(mol_batch)):
                DataStructs.ConvertToNumpyArray(fp, fps[i, :])
            if pack:
                fps = pack_fingerprints(fps)
            if digits is not None:
                out_name = f"{out_name}.{str(file_idx).zfill(digits)}"
            np.save(out_dir / out_name, fps)
    if file_idx > 0:
        stem = out_name.split(".")[0]
        console.print(f"Finished. Outputs written to {str(out_dir / stem)}.<idx>.npy")
    else:
        console.print(f"Finished. Outputs written to {str(out_dir / out_name)}.npy")


@app.command("fps-split")
def _split_fps(
    input_: Annotated[
        Path,
        Argument(help="`*.npy` file with fingerprints"),
    ],
    out_dir: Annotated[
        Path | None,
        Option("-o", "--out-dir", show_default=False),
    ] = None,
    parts: Annotated[
        int | None,
        Option(
            "-n",
            "--num-parts",
            help="Num. of parts to split file into. Mutually exclusive with --max-fps",
            show_default=False,
        ),
    ] = None,
    max_fps_per_file: Annotated[
        int | None,
        Option(
            "-m",
            "--max-fps",
            help="Max. number of fps per file. Mutually exclusive with --num-parts",
            show_default=False,
        ),
    ] = None,
) -> None:
    r"""Split a `*.npy` fingerprint file into multiple `*.npy` files

    Usage to split into multiple files with a max number of fps each (e.g. 10k) is `bb
    split-fps --max-fps 10_000 ./fps.npy --out-dir ./split`. To split into a pre-defined
    number of parts (e.g. 10) `bb split-fps --num-parts 10 ./fps.npy --out-dir ./split`.
    """
    from bblean._console import get_console

    console = get_console()
    if parts is not None and parts < 2:
        console.print("Num must be >= 2", style="red")
        raise Abort()
    fps = np.load(input_, mmap_mode="r")
    if parts is not None and max_fps_per_file is None:
        num_per_batch = math.ceil(fps.shape[0] / parts)
        digits = len(str(parts))
    elif parts is None and max_fps_per_file is not None:
        num_per_batch = max_fps_per_file
        digits = len(str(math.ceil(fps.shape[0] / max_fps_per_file)))
    else:
        console.print(
            "One and only one of '--max-fps' and '--num-parts' required", style="red"
        )
        raise Abort()

    if out_dir is None:
        out_dir = Path.cwd()
    out_dir.mkdir(exist_ok=True)
    out_dir = out_dir.resolve()
    stem = input_.name.split(".")[0]
    with console.status("[italic]Splitting fingerprints...[/italic]", spinner="dots"):
        for i, batch in enumerate(batched(fps, num_per_batch)):
            suffixes = input_.suffixes
            name = f"{stem}{''.join(suffixes[:-1])}.{str(i).zfill(digits)}.npy"
            np.save(out_dir / name, batch)
    console.print(f"Finished. Outputs written to {str(out_dir / stem)}.<idx>.npy")
