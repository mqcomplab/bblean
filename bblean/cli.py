r"""Command line interface entrypoints"""

import math
import typing_extensions as tpx
import shutil
import json
import time
import sys
import pickle
import uuid
import re
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

app = Typer(
    rich_markup_mode="markdown",
    add_completion=False,
    help=r"""CLI interface for serial and parallel fast clustering of molecular
    fingerprints using the *O(N)* BitBirch algorithm. If you find this work useful
    please cite the following articles:
    - Original BitBirch article:
        [https://doi.org/10.1039/D5DD00030K](https://doi.org/10.1039/D5DD00030K)
    - BitBirch refinement strategies:
        (TODO)
    - BitBirch-Lean:
        (TODO)
    """,  # noqa
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
def _validate_input_dir(
    in_dir: Path | str, filename_idxs_are_slices: bool = True
) -> None:
    in_dir = Path(in_dir)
    if not in_dir.is_dir():
        raise RuntimeError(f"Input dir {in_dir} should be a dir")
    if not any(in_dir.glob("*.npy")):
        raise RuntimeError(f"Input dir {in_dir} should have *.npy fingerprint files")

    # TODO: There is currently no validation regarding fp sizes and stride sizes
    if filename_idxs_are_slices:
        return

    _file_idxs = []
    for f in in_dir.glob("*.npy"):
        matches = re.match(r".*_(\d+)_(\d+).npy", f.name)
        if matches is None:
            raise RuntimeError(f"Input file {str(f)} doesn't match name convention")
        _file_idxs.append(int(matches[1]))

    # Sort arrays
    sort_idxs = np.argsort(_file_idxs)
    file_idxs = np.array(_file_idxs)[sort_idxs]

    if not (file_idxs == np.arange(len(file_idxs))).all():
        raise RuntimeError(f"Input file indices {file_idxs} must be a seq 0, 1, 2, ...")


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


@app.command("fps-from-smiles")
def _fps_from_smiles(
    smiles_paths: Annotated[
        list[Path] | None,
        Option("-s", "--smiles-path", show_default=False),
    ] = None,
    out_dir: Annotated[
        Path | None,
        Option("-o", "--output-dir", show_default=False),
    ] = None,
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
    r"""Generate a *.npy fingerprints file from one or more smiles files

    In order to use the memory efficient BitBirch u8 algorithm you *must* use the
    defaults: --dtype=uint8 and --packed
    """
    import numpy as np
    from rdkit.Chem import rdFingerprintGenerator, DataStructs, MolFromSmiles
    from bblean._console import get_console

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
    if out_dir is None:
        unique_id = str(uuid.uuid4()).split("-")[0]
        out_dir = Path.cwd() / ("packed_fps" if packed else "fps") / unique_id
    out_dir.mkdir(parents=True, exist_ok=True)

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

    if packed:
        fps = pack_fingerprints(fps)

    # Save the fingerprints as a NumPy array
    np.save(out_dir / f"{'packed-' if packed else ''}fps-{dtype}", fps)


@app.command("split-fps")
def _split_fps(
    input_: Annotated[
        Path,
        Argument(help="*.npy file with fingerprints"),
    ],
    num: tpx.Annotated[
        int,
        Option("-n", "--num", help="Number fo files to split into"),
    ],
    out_dir: tpx.Annotated[
        Path | None,
        Option("-o", "--out-dir", show_default=False),
    ] = None,
) -> None:
    r"""Split a *.npy fingerprint file into multiple *.npy files"""
    if num < 2:
        raise ValueError("Num must be >= 2")
    if out_dir is None:
        out_dir = input_.parent
    fps = np.load(input_, mmap_mode="r")
    digits = len(str(num)) + 1
    num_per_batch = math.ceil(fps.shape[0] / num)
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
            help="BitBirch branching factor. Under most circumstances 254 is"
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
            help="BitBirch tolerance, only for --set-merge tolerance|tolerance_tough"
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
    r"""Run standard BitBirch clustering"""
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
    ctx.params["input_files"] = [str(p) for p in input_files]
    unique_id = str(uuid.uuid4()).split("-")[0]
    if out_dir is None:
        out_dir = Path.cwd() / "bb_run_outputs" / unique_id
    out_dir.mkdir(exist_ok=True, parents=True)
    _validate_output_dir(out_dir, overwrite_outputs)
    ctx.params["out_dir"] = str(out_dir)

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
    branching_factor: Annotated[
        int,
        Option(
            help="BitBirch branching factor. Under most circumstances 254 is"
            " optimal for performance and memory efficiency. Set this above 254 for"
            " slightly less RAM usage at the cost of some performance."
        ),
    ] = DEFAULTS.branching_factor,
    threshold: Annotated[float, Option(help="BitBirch threshold")] = DEFAULTS.threshold,
    tolerance: Annotated[
        float,
        Option(
            help="BitBirch tolerance"
            " (Used in Round 1 'double-cluster-init', Round 2, and Final clustering)"
        ),
    ] = DEFAULTS.tolerance,
    merge_criterion: Annotated[
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
    r"""Run multi-round BitBirch clustering, with the option to parallelize"""
    from bblean._console import get_console
    from bblean.multiround import run_multiround_bitbirch

    console = get_console(silent=not verbose)

    # Set the multiprocessing start method
    if fork and not sys.platform == "linux":
        raise ValueError("'fork' is only available on Linux")
    if sys.platform == "linux":
        mp.set_start_method("fork" if fork else "forkserver")

    # If not passed, input dir is bb_inputs/
    if in_dir is None:
        in_dir = Path.cwd() / "bb_inputs"
    _validate_input_dir(in_dir, filename_idxs_are_slices)

    # All files in the input dir with *.npy suffix are considered input files
    input_files = sorted(
        in_dir.glob("*.npy"), key=lambda x: int(x.name.split(".")[0].split("_")[-2])
    )[:max_files]
    ctx.params["input_files"] = [str(p) for p in input_files]

    # If not passed, output dir is constructed as bb_multiround_outputs/<unique-id>/
    unique_id = str(uuid.uuid4()).split("-")[0]
    if out_dir is None:
        out_dir = Path.cwd() / "bb_multiround_outputs" / unique_id
    out_dir.mkdir(exist_ok=True, parents=True)
    _validate_output_dir(out_dir, overwrite_outputs)
    ctx.params["out_dir"] = str(out_dir)

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

    timings_stats = run_multiround_bitbirch(
        input_files=input_files,
        n_features=n_features,
        out_dir=out_dir,
        filename_idxs_are_slices=filename_idxs_are_slices,
        round2_process_fraction=round2_process_fraction,
        merge_criterion=merge_criterion,
        num_processes=num_processes,
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
        max_files=max_files,
        verbose=verbose,
    )
    with open(out_dir / "timings.json", mode="wt", encoding="utf-8") as f:
        json.dump(timings_stats, f, indent=4)
    # TODO: Also dump peak-rss.json
    collect_system_specs_and_dump_config(ctx.params)
