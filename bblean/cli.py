r"""Command line interface entrypoints"""

import typing as tp
import math
import typing_extensions as tpx
import shutil
import json
import time
import sys
import pickle
import uuid
import multiprocessing as mp
from typing import Annotated
from pathlib import Path

import numpy as np
import typer
from typer import Typer, Argument, Option, Abort, Context

from bblean.memory import monitor_rss_daemon, get_peak_memory
from bblean.config import DEFAULTS, collect_system_specs_and_dump_config
from bblean.packing import pack_fingerprints
from bblean.utils import _import_bitbirch_variant, batched
from bblean.fingerprint_io import get_file_num_fps, print_file_info

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
        raise typer.Exit()


def _validate_output_dir(out_dir: Path, overwrite_outputs: bool = False) -> None:
    if out_dir.exists():
        if not out_dir.is_dir():
            raise RuntimeError("Output dir should be a dir")
        if any(out_dir.iterdir()):
            if overwrite_outputs:
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
def main(
    ctx: typer.Context,
    help_: bool = typer.Option(
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


@app.command("fps-info")
def _fps_info(
    fp_paths: Annotated[
        list[Path] | None,
        Argument(show_default=False, help="Paths to *.smi files with smiles"),
    ] = None,
) -> None:
    """Show info about a *.npy fingerprint file, or a dir with *.npy files"""
    from bblean._console import get_console

    console = get_console()
    if fp_paths is None:
        fp_paths = [Path.cwd()]

    for path in fp_paths:
        if path.is_dir():
            for file in path.glob("*.npy"):
                print_file_info(file, console)
        elif path.suffix == ".npy":
            print_file_info(file, console)


@app.command("fps-from-smiles")
def _fps_from_smiles(
    smiles_paths: Annotated[
        list[Path] | None,
        Argument(show_default=False, help="Paths to *.smi files with smiles"),
    ] = None,
    out_dir: Annotated[
        Path | None,
        Option("-o", "--output-dir", show_default=False),
    ] = None,
    dtype: Annotated[
        str,
        Option("-d", "--dtype", help="NumPy dtype for the generated fingerprints"),
    ] = "uint8",
    parts: tpx.Annotated[
        int | None,
        Option(
            "-n", "--num-parts", help="Split the created file into this number of parts"
        ),
    ] = None,
    max_fps_per_file: tpx.Annotated[
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
        ),
    ] = True,
    kind: Annotated[
        str,
        Option("-k", "--kind"),
    ] = "rdkit",
    fp_size: Annotated[
        int,
        Option("--fp-size"),
    ] = DEFAULTS.features_num,
    verbose: Annotated[
        bool,
        Option("-v/-V", "--verbose/--no-verbose"),
    ] = True,
) -> None:
    r"""Generate a *.npy fingerprints file from one or more *.smi smiles files

    In order to use the memory efficient BitBIRCH u8 algorithm you *must* use the
    defaults: --dtype=uint8 and --pack
    """
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator, DataStructs, MolFromSmiles
    from bblean._console import get_console

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

    # Pass 1: check the total number of smiles
    smiles_num = 0
    for smi_path in smiles_paths:
        with open(smi_path, mode="rt", encoding="utf-8") as f:
            for _ in f:
                smiles_num += 1

    digits: int | None
    if parts is not None and max_fps_per_file is None:
        num_per_batch = math.ceil(smiles_num / parts)
        digits = len(str(parts)) + 1
    elif parts is None and max_fps_per_file is not None:
        num_per_batch = max_fps_per_file
        digits = len(str(math.ceil(smiles_num / max_fps_per_file))) + 1
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
        unique_id = str(uuid.uuid4()).split("-")[0]
        out_dir = Path.cwd() / ("packed-fps" if pack else "fps") / unique_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pass 2: build the molecules
    for file_idx, mol_batch in enumerate(
        batched(iter_mols_from_paths(smiles_paths), num_per_batch)
    ):
        fps = np.empty((len(mol_batch), fp_size), dtype=dtype)
        for i, fp in enumerate(fpg.GetFingerprints(mol_batch)):
            DataStructs.ConvertToNumpyArray(fp, fps[i, :])
        if pack:
            fps = pack_fingerprints(fps)
        # Save the fingerprints as a NumPy array
        name = f"{'packed-' if pack else ''}fps-{dtype}"
        if digits is not None:
            name = f"{name}.{str(file_idx).zfill(digits)}"
        np.save(out_dir / name, fps)


@app.command("fps-split")
def _split_fps(
    input_: Annotated[
        Path,
        Argument(help="*.npy file with fingerprints"),
    ],
    parts: tpx.Annotated[
        int | None,
        Option(
            "-n",
            "--num-parts",
            help="Num. of parts to split file into. Mutually exclusive with --max-fps",
            show_default=False,
        ),
    ] = None,
    max_fps_per_file: tpx.Annotated[
        int | None,
        Option(
            "-m",
            "--max-fps",
            help="Max. number of fps per file. Mutually exclusive with --num-parts",
            show_default=False,
        ),
    ] = None,
    out_dir: tpx.Annotated[
        Path | None,
        Option("-o", "--out-dir", show_default=False),
    ] = None,
) -> None:
    r"""Split a *.npy fingerprint file into multiple *.npy files

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
        digits = len(str(parts)) + 1
    elif parts is None and max_fps_per_file is not None:
        num_per_batch = max_fps_per_file
        digits = len(str(math.ceil(fps.shape[0] / max_fps_per_file))) + 1
    else:
        console.print(
            "One and only one of '--max-fps' and '--num-parts' required", style="red"
        )
        raise Abort()

    if out_dir is None:
        out_dir = input_.parent
    out_dir.mkdir(exist_ok=True)
    for i, batch in enumerate(batched(fps, num_per_batch)):
        name = f"{input_.with_suffix('').name}.{str(i).zfill(digits)}"
        np.save(out_dir / name, batch)


@app.command("run")
def _run(
    ctx: Context,
    input_: Annotated[
        Path | None,
        Argument(help="*.npy file with packed fingerprints, or dir *.npy files"),
    ] = None,
    out_dir: Annotated[
        Path | None,
        Option(
            "-o",
            "--output-dir",
            help="Dir to dump the output files",
        ),
    ] = None,
    overwrite_outputs: Annotated[
        bool, Option(help="Allow overwriting output files")
    ] = False,
    branching_factor: Annotated[
        int,
        Option(
            help="BitBIRCH branching factor. Under most circumstances 254 is"
            " optimal for performance and memory efficiency. Set this above 254 for"
            " slightly less RAM usage at the cost of some performance."
        ),
    ] = DEFAULTS.branching_factor,
    use_mmap: Annotated[
        bool,
        Option(help="Toggle mmap of the fingerprint files", rich_help_panel="Advanced"),
    ] = DEFAULTS.use_mmap,
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
    n_features: tpx.Annotated[
        int,
        Option(
            "-n",
            "--n-features",
            help="Number of features in the fingerprints. Required for packed inputs",
        ),
    ] = 2048,
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
    variant: tpx.Annotated[
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
    r"""Run standard, serial BitBIRCH clustering over *.npy fingerprint files"""
    import numpy as np
    from bblean._console import get_console

    console = get_console(silent=not verbose)

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
    ctx.params["num_fps_present"] = [get_file_num_fps(p) for p in input_files]
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
    _validate_output_dir(out_dir, overwrite_outputs)
    ctx.params["out_dir"] = str(out_dir.resolve())

    console.print_banner()
    console.print_config(ctx.params)

    # Optinally start a separate process that tracks RAM usage
    if monitor_rss:
        console.print("** Monitoring total RAM usage **\n")
        mp.Process(
            target=monitor_rss_daemon,
            kwargs=dict(
                file=out_dir / "monitor-rss.csv",
                interval_s=monitor_rss_interval_s,
                start_time=time.perf_counter(),
            ),
            daemon=True,
        ).start()

    start_time = time.perf_counter()
    set_merge(merge_criterion, tolerance)
    tree = BitBirch(branching_factor=branching_factor, threshold=threshold)
    for file in input_files:
        fps = np.load(file, mmap_mode="r" if use_mmap else None)[:max_fps]
        tree.fit(fps, n_features=n_features)
    cluster_mol_ids = tree.get_cluster_mol_ids()
    total_time_s = time.perf_counter() - start_time
    console.print(f"Time elapsed: {total_time_s:.5f} s")
    stats = get_peak_memory(1)
    if stats is None:
        console.print("[Peak memory stats not tracked for non-Unix systems]")
    else:
        console.print_peak_mem_raw(stats)

    # Dump outputs (peak memory, timings, config, cluster ids)
    with open(out_dir / "clusters.pkl", mode="wb") as f:
        pickle.dump(cluster_mol_ids, f)

    collect_system_specs_and_dump_config(ctx.params)
    timings_fpath = out_dir / "timings.json"
    with open(timings_fpath, mode="wt", encoding="utf-8") as f:
        json.dump({"total": total_time_s}, f, indent=4)
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
        Argument(help="Directory with input *.npy files with packed fingerprints"),
    ] = None,
    out_dir: Annotated[
        Path | None,
        Option("-o", "--output-dir", help="Dir for output files"),
    ] = None,
    overwrite_outputs: Annotated[
        bool, Option(help="Allow overwriting output files")
    ] = False,
    num_initial_processes: Annotated[
        int, Option("--ps", "--processes", help="Num. processes for first round")
    ] = 10,
    num_midsection_processes: tpx.Annotated[
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
    n_features: tpx.Annotated[
        int,
        Option(
            "-n",
            "--n-features",
            help="Number of features in the fingerprints. Required for packed inputs",
        ),
    ] = 2048,
    # Advanced options
    num_midsection_rounds: tpx.Annotated[
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
        Option(help="Toggle mmap of the fingerprint files", rich_help_panel="Advanced"),
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
    variant: tpx.Annotated[
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
    r"""Run multi-round BitBIRCH clustering, optionally parallelize over *.npy files"""
    from bblean._console import get_console
    from bblean.multiround import run_multiround_bitbirch

    console = get_console(silent=not verbose)

    # Set the multiprocessing start method
    if fork and not sys.platform == "linux":
        console.print("'fork' is only available on Linux", style="red")
        raise Abort()
    if sys.platform == "linux":
        mp.set_start_method("fork" if fork else "forkserver")

    # If not passed, input dir is bb_inputs/
    if in_dir is None:
        in_dir = Path.cwd() / "bb_inputs"
    _validate_input_dir(in_dir)

    # All files in the input dir with *.npy suffix are considered input files
    input_files = sorted(
        in_dir.glob("*.npy"), key=lambda x: int(x.name.split(".")[0].split("_")[-2])
    )[:max_files]
    ctx.params["input_files"] = [str(p.resolve()) for p in input_files]
    ctx.params["num_fps"] = [get_file_num_fps(p) for p in input_files]
    if max_fps is not None:
        ctx.params["num_fps_loaded"] = [min(n, max_fps) for n in ctx.params["num_fps"]]
    else:
        ctx.params["num_fps_loaded"] = ctx.params["num_fps"]

    # If not passed, output dir is constructed as bb_multiround_outputs/<unique-id>/
    unique_id = str(uuid.uuid4()).split("-")[0]
    if out_dir is None:
        out_dir = Path.cwd() / "bb_multiround_outputs" / unique_id
    out_dir.mkdir(exist_ok=True, parents=True)
    _validate_output_dir(out_dir, overwrite_outputs)
    ctx.params["out_dir"] = str(out_dir.resolve())

    console.print_banner()
    console.print_multiround_config(ctx.params)

    # Optinally start a separate process that tracks RAM usage
    if monitor_rss:
        console.print("** Monitoring total RAM usage **\n")
        mp.Process(
            target=monitor_rss_daemon,
            kwargs=dict(
                file=out_dir / "monitor-rss.csv",
                interval_s=monitor_rss_interval_s,
                start_time=time.perf_counter(),
            ),
            daemon=True,
        ).start()

    if num_midsection_processes is None:
        num_midsection_processes = num_initial_processes
    else:
        # Sanity check
        if num_midsection_processes > num_initial_processes:
            console.print(
                "Num. midsection processes must be <= num. initial processes",
                style="red",
            )
            raise Abort()

    timings_stats = run_multiround_bitbirch(
        input_files=input_files,
        n_features=n_features,
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
    with open(out_dir / "timings.json", mode="wt", encoding="utf-8") as f:
        json.dump(timings_stats, f, indent=4)
    # TODO: Also dump peak-rss.json
    collect_system_specs_and_dump_config(ctx.params)
