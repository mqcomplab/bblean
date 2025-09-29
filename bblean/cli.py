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

from typer import Typer, Argument, Option, Abort, Context, Exit

from bblean._memory import launch_monitor_rss_daemon, get_peak_memory
from bblean._timer import Timer
from bblean._config import DEFAULTS, collect_system_specs_and_dump_config, TSNE_SEED
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


@app.command("tsne-plot")
def _tsne_plot(
    clusters_path: Annotated[Path, Option("-c", "--clusters-path", show_default=False)],
    fps_path: Annotated[Path, Option("-f", "--fps-path", show_default=False)],
    title: Annotated[
        str | None,
        Option("--title", help="Plot title"),
    ] = None,
    exaggeration: Annotated[
        float | None,
        Option("-e", "--exaggeration", rich_help_panel="Advanced"),
    ] = None,
    seed: Annotated[
        int | None,
        Option(
            "-s",
            "--seed",
            help=(
                "Seed for the rng, fixed value by default, for reproducibility."
                " Pass -1 to randomize"
            ),
            show_default=False,
        ),
    ] = TSNE_SEED,
    top: Annotated[
        int,
        Option("--top"),
    ] = 20,
    dof: Annotated[
        float,
        Option("-d", "--dof", rich_help_panel="Advanced"),
    ] = 1.0,
    perplexity: Annotated[
        int,
        Option(help="t-SNE perplexity", rich_help_panel="Advanced"),
    ] = 30,
    input_is_packed: Annotated[
        bool,
        Option("--packed-input/--unpacked-input", rich_help_panel="Advanced"),
    ] = True,
    scaling: Annotated[
        str,
        Option("--scaling", rich_help_panel="Advanced"),
    ] = "normalize",
    do_pca_init: Annotated[
        bool,
        Option(
            "--pca-init/--no-pca-init",
            rich_help_panel="Advanced",
            help="Use PCA for initialization",
        ),
    ] = True,
    pca_reduce: Annotated[
        int | None,
        Option(
            "-p",
            "--pca-reduce",
            rich_help_panel="Advanced",
            help=(
                "Reduce fingerprint dimensionality to N components using PCA."
                " A value of 50 or more maintains cluster structure in general"
            ),
        ),
    ] = None,
    workers: Annotated[
        int | None,
        Option(
            "-w",
            "--workers",
            help="Num. cores to use for parallel processing",
            rich_help_panel="Advanced",
        ),
    ] = None,
    multiscale: Annotated[
        bool,
        Option(
            "-m/-M",
            "--multiscale/--no-multiscale",
            rich_help_panel="Advanced",
            help="Use multiscale perplexities (WARNING: Can be very slow!)",
        ),
    ] = False,
    n_features: Annotated[
        int | None,
        Option(
            "--n-features",
            help="Number of features in the fingerprints."
            " Only for packed inputs *if it is not a multiple of 8*."
            " Not required for typical fingerprint sizes (e.g. 2048, 1024)",
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
        import numpy as np

        from bblean.analysis import cluster_analysis
        from bblean.plotting import tsne_plot

        if clusters_path.is_dir():
            clusters_path = clusters_path / "clusters.pkl"
        with open(clusters_path, mode="rb") as f:
            clusters = pickle.load(f)
        fps = np.load(fps_path, mmap_mode="r")
        ca = cluster_analysis(
            clusters,
            fps,
            smiles=(),
            top=top,
            n_features=n_features,
            input_is_packed=input_is_packed,
        )
        tsne_plot(
            ca,
            title,
            perplexity=perplexity,
            exaggeration=exaggeration,
            seed=seed,
            dof=dof,
            workers=workers,
            scaling=scaling,
            do_pca_init=do_pca_init,
            multiscale=multiscale,
            pca_reduce=pca_reduce,
        )
    plt.show()


@app.command("summary-plot")
def _summary_plot(
    clusters_path: Annotated[Path, Option("-c", "--clusters-path", show_default=False)],
    fps_path: Annotated[Path, Option("-f", "--fps-path", show_default=False)],
    smiles_path: Annotated[
        Path | None,
        Option(
            "-s",
            "--smiles-path",
            show_default=False,
            help="Optional smiles path, if passed a scaffold analysis is performed",
        ),
    ] = None,
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
        Option("--packed-input/--unpacked-input", rich_help_panel="Advanced"),
    ] = True,
    scaffold_fp_kind: Annotated[
        str,
        Option("--scaffold-fp-kind"),
    ] = DEFAULTS.fp_kind,
    annotate: Annotated[
        bool,
        Option(
            "--annotate/--no-annotate",
            help="Display scaffold and fingerprint number in each cluster",
        ),
    ] = True,
    n_features: Annotated[
        int | None,
        Option(
            "--n-features",
            help="Number of features in the fingerprints."
            " Only for packed inputs *if it is not a multiple of 8*."
            " Not required for typical fingerprint sizes (e.g. 2048, 1024)",
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
        import numpy as np

        from bblean.smiles import load_smiles
        from bblean.analysis import cluster_analysis
        from bblean.plotting import summary_plot

        if clusters_path.is_dir():
            clusters_path = clusters_path / "clusters.pkl"
        with open(clusters_path, mode="rb") as f:
            clusters = pickle.load(f)
        fps = np.load(fps_path, mmap_mode="r")
        smiles: tp.Iterable[str]
        if smiles_path is not None:
            smiles = load_smiles(smiles_path)
        else:
            smiles = ()
        ca = cluster_analysis(
            clusters,
            fps,
            smiles,
            top=top,
            n_features=n_features,
            input_is_packed=input_is_packed,
            scaffold_fp_kind=scaffold_fp_kind,
        )
        summary_plot(ca, title, annotate=annotate)
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
            "--branching",
            "-b",
            help="BitBIRCH branching factor (all rounds). Usually 254 is"
            " optimal. Set above 254 for slightly less RAM (at the cost of some perf.)",
        ),
    ] = DEFAULTS.branching_factor,
    threshold: Annotated[
        float,
        Option("--threshold", "-t", help="Threshold for merge criterion"),
    ] = DEFAULTS.threshold,
    merge_criterion: Annotated[
        str,
        Option("--set-merge", "-m", help="Merge criterion for initial clustsering"),
    ] = "diameter",
    tolerance: Annotated[
        float,
        Option(help="BitBIRCH tolerance. For refinement and --set-merge 'tolerance'"),
    ] = DEFAULTS.tolerance,
    refine_num: Annotated[
        int,
        Option(
            "-r",
            "--refine-num",
            help=(
                "Num. of largest clusters to refine."
                " '-r' 1 for standard refinement, '-r' 0 is the default (no refinement)"
            ),
        ),
    ] = 0,
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
            help="Monitor RAM used by all processes",
            rich_help_panel="Advanced",
        ),
    ] = False,
    monitor_rss_interval_s: Annotated[
        float,
        Option(
            "--monitor-rss-seconds",
            help="Interval in seconds for RSS monitoring",
            rich_help_panel="Debug",
            hidden=True,
        ),
    ] = 0.01,
    max_fps: Annotated[
        int | None,
        Option(
            help="Max. num of fingerprints to read from each file",
            rich_help_panel="Debug",
            hidden=True,
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
        set_merge(merge_criterion, tolerance=tolerance)
        tree = BitBirch(branching_factor=branching_factor, threshold=threshold)
    else:
        tree = BitBirch(
            branching_factor=branching_factor,
            threshold=threshold,
            merge_criterion=merge_criterion,
            tolerance=tolerance,
        )
    if len(input_files) > 1 and refine_num > 0:
        console.print("Refine currently only supported for single files", style="red")
        raise Abort()
    with console.status("[italic]BitBirching...[/italic]", spinner="dots"):
        for file in input_files:
            # Fitting a file uses mmap internally, and releases memory in a smart way
            tree.fit(
                file,
                n_features=n_features,
                input_is_packed=input_is_packed,
                max_fps=max_fps,
            )

        if refine_num > 0:
            tree.set_merge("tolerance", tolerance=tolerance)
            tree.refine_inplace(
                file, input_is_packed=input_is_packed, n_largest=refine_num
            )
        # TODO: Fix peak memory stats
        cluster_mol_ids = tree.get_cluster_mol_ids()
    timer.end_timing("total", console, indent=False)
    stats = get_peak_memory(1)
    if stats is None:
        console.print("[Peak memory stats not tracked for non-Unix systems]")
    else:
        console.print_peak_mem_raw(stats, indent=False)
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
            help="Num. processes for middle section rounds."
            " These are be memory intensive,"
            " you may want to use 50%-30% of --ps."
            " Default is same as --ps",
        ),
    ] = None,
    branching_factor: Annotated[
        int,
        Option(
            "--branching",
            "-b",
            help="BitBIRCH branching factor (all rounds). Usually 254 is"
            " optimal. Set above 254 for slightly less RAM (at the cost of some perf.)",
        ),
    ] = DEFAULTS.branching_factor,
    threshold: Annotated[
        float,
        Option("--threshold", "-t", help="Threshold for merge criterions (all stetps)"),
    ] = DEFAULTS.threshold,
    initial_merge_criterion: Annotated[
        str,
        Option(
            "--set-merge",
            "-m",
            help="Initial merge criterion for round 1. ('diameter' recommended)",
        ),
    ] = DEFAULTS.merge_criterion,
    tolerance: Annotated[
        float,
        Option(
            help="Tolerance value for all steps that use the 'tolerance' criterion"
            " (by default all except initial round)",
        ),
    ] = DEFAULTS.tolerance,
    n_features: Annotated[
        int | None,
        Option(
            "--n-features",
            help="Number of features in the fingerprints."
            " Only for packed inputs *if it is not a multiple of 8*."
            " Not required for typical fingerprint sizes (e.g. 2048, 1024)",
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
            "--num-mid-rounds",
            help="Number of midsection rounds to perform",
            rich_help_panel="Advanced",
        ),
    ] = 1,
    split_largest_after_midsection: Annotated[
        bool,
        Option(
            "--split-after-mid/--no-split-after-mid",
            help=(
                "Split largest cluster after each midsection round"
                " (to be refined by the next round)"
            ),
            rich_help_panel="Advanced",
        ),
    ] = False,
    full_refinement_before_midsection: Annotated[
        bool,
        Option(
            "--refine-before-mid/--no-refine-before-mid",
            help=(
                "Run a *full* refinement step after the initial clustering round"
                " (largest cluster is always split regardless of this flag)"
            ),
            rich_help_panel="Advanced",
        ),
    ] = True,
    max_tasks_per_process: Annotated[
        int, Option(help="Max tasks per process", rich_help_panel="Advanced")
    ] = 1,
    fork: Annotated[
        bool,
        Option(
            help="In linux, force the 'fork' multiprocessing start method",
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
            hidden=True,
        ),
    ] = False,
    monitor_rss: Annotated[
        bool,
        Option(
            help="Monitor RAM used by all processes",
            rich_help_panel="Advanced",
        ),
    ] = False,
    monitor_rss_interval_s: Annotated[
        float,
        Option(
            "--monitor-rss-seconds",
            help="Interval in seconds for RSS monitoring",
            rich_help_panel="Debug",
            hidden=True,
        ),
    ] = 0.01,
    max_fps: Annotated[
        int | None,
        Option(
            help="Max num. of fps to load from each input file",
            rich_help_panel="Debug",
            hidden=True,
        ),
    ] = None,
    max_files: Annotated[
        int | None,
        Option(help="Max num. files to read", rich_help_panel="Debug", hidden=True),
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
        branching_factor=branching_factor,
        threshold=threshold,
        tolerance=tolerance,
        # Advanced
        bin_size=bin_size,
        max_tasks_per_process=max_tasks_per_process,
        full_refinement_before_midsection=full_refinement_before_midsection,
        num_midsection_rounds=num_midsection_rounds,
        split_largest_after_each_midsection_round=split_largest_after_midsection,
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
    ] = DEFAULTS.fp_kind,
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
    num_ps: Annotated[
        int | None,
        Option(
            "--processes",
            help="Num. processes for multprocess generation"
            " (Currently only implemented when generating multiple files)",
        ),
    ] = None,
) -> None:
    r"""Generate a `*.npy` fingerprints file from one or more `*.smi` smiles files

    In order to use the memory efficient BitBIRCH u8 algorithm you should keep the
    defaults: --dtype=uint8 and --pack
    """
    from rdkit import Chem
    from rdkit.Chem import MolFromSmiles

    from bblean._console import get_console
    from bblean.utils import _num_avail_cpus
    from bblean.fingerprints import _FingerprintFileCreator
    from bblean.smiles import iter_smiles_from_paths

    if sys.platform == "linux":
        # Force forkserver since rdkit may use threads, and fork is unsafe with threads
        mp.set_start_method("forkserver")

    def iter_mols_from_paths(
        smiles_paths: tp.Iterable[Path], skip_bad_smiles: bool = False
    ) -> tp.Iterator[Chem.Mol]:
        for smi in iter_smiles_from_paths(smiles_paths):
            mol = MolFromSmiles(smi)
            if mol is None:
                if not skip_bad_smiles:
                    console.print(f"Could not parse smiles {smi}")
                    raise Abort()
            yield mol

    console = get_console(silent=not verbose)

    if smiles_paths is None:
        smiles_paths = list(Path.cwd().glob("*.smi"))
    if not smiles_paths:
        console.print("No *.smi files found", style="red")
        raise Abort()

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
        parts = math.ceil(smiles_num / max_fps_per_file)
        digits = len(str(parts))
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

    if parts > 1 and num_ps is None:
        # Get the number of cores *available for use for this process*
        # bound by the number of parts to avoid spawning useless processes
        num_ps = min(_num_avail_cpus(), parts)
    create_fp_file = _FingerprintFileCreator(
        dtype, out_dir, out_name, digits, pack, kind, fp_size
    )
    if parts > 1 and num_ps is not None and num_ps > 1:
        # Multiprocessing version
        # TODO: Currently only implemented for split files (parts > 1), it could be
        # implemented for single-files using shmem
        with console.status(
            f"[italic]Generating fingerprints (parallel, {num_ps} procs.) ...[/italic]",
            spinner="dots",
        ):
            idxs_batches = list(
                enumerate(batched(iter_smiles_from_paths(smiles_paths), num_per_batch))
            )
            with mp.Pool(processes=num_ps) as pool:
                pool.map(create_fp_file, idxs_batches)
    else:
        # Serial version
        with console.status(
            "[italic]Generating fingerprints (serial) ...[/italic]", spinner="dots"
        ):
            for idx_batch in enumerate(
                batched(iter_smiles_from_paths(smiles_paths), num_per_batch)
            ):
                create_fp_file(idx_batch)
    if parts > 1:
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
    import numpy as np

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


@app.command("fps-shuffle")
def _shuffle_fps(
    in_file: Annotated[
        Path,
        Argument(help="`*.npy` file with packed fingerprints"),
    ],
    out_dir: Annotated[
        Path | None,
        Option("-o", "--out-dir", show_default=False),
    ] = None,
    seed: Annotated[
        int | None,
        Option("--seed", hidden=True, rich_help_panel="Debug"),
    ] = None,
) -> None:
    """Shuffle a fingerprints file

    This function is not optimized and as such may have high RAM usage. It is
    meant for testing purposes only"""
    import numpy as np

    fps = np.load(in_file)
    stem = in_file.stem
    rng = np.random.default_rng(seed)
    rng.shuffle(fps, axis=0)
    if out_dir is None:
        out_dir = Path.cwd()
    out_dir.mkdir(exist_ok=True)
    out_dir = out_dir.resolve()
    np.save(out_dir / f"shuffled-{stem}.npy", fps)


@app.command("fps-merge")
def _merge_fps(
    in_dir: Annotated[
        Path,
        Argument(help="Directory with input `*.npy` files with packed fingerprints"),
    ],
    out_dir: Annotated[
        Path | None,
        Option("-o", "--out-dir", show_default=False),
    ] = None,
) -> None:
    r"""Merge a dir with multiple `*.npy` fingerprint file into a single `*.npy` file"""
    from bblean._console import get_console
    import numpy as np

    console = get_console()

    if out_dir is None:
        out_dir = Path.cwd()
    out_dir.mkdir(exist_ok=True)
    out_dir = out_dir.resolve()
    arrays = []
    with console.status("[italic]Merging fingerprints...[/italic]", spinner="dots"):
        stem = None
        for f in sorted(in_dir.glob("*.npy")):
            if stem is None:
                stem = f.name.split(".")[0]
            elif stem != f.name.split(".")[0]:
                raise ValueError(
                    "Name convention must be <name>.<idx>.npy"
                    " with all files having the same <name>"
                )
            arrays.append(np.load(f))
        if stem is None:
            console.print("No *.npy files found")
            return
        np.save(out_dir / stem, np.concatenate(arrays))
    console.print(f"Finished. Outputs written to {str(out_dir / stem)}.npy")
