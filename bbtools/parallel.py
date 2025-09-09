import functools
import os
import json
import sys
import pickle
import uuid
import gc
import re
import time
import typing as tp
import multiprocessing as mp
from copy import deepcopy
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from bbtools._console import get_console
from bbtools.memory import monitor_rss_daemon, system_mem_gib
from bbtools.utils import cpu_name, numpy_streaming_save, batched


def _glob_and_sort_by_uint_bits(path: Path | str, glob: str) -> list[str]:
    path = Path(path)
    return sorted(
        map(str, path.glob(glob)),
        key=lambda name: int((re.search(r"uint(\d+)", name) or [0, 0])[1]),
        reverse=True,
    )


def _load_fp_data_and_mol_idxs(
    out_dir: Path,
    fp_filename: str,
    round: int,
    mmap: bool = True,
) -> tuple[NDArray[tp.Any], tp.Any]:
    fp_data = np.load(fp_filename, mmap_mode="r" if mmap else None)
    # Infer the mol_idxs filename from the fp_np filename, and fetch the mol_idxs
    count, dtype = fp_filename.split(".")[0].split("_")[-2:]
    idxs_file = out_dir / f"mol_idxs_round{round}_{count}_{dtype}.pkl"
    with open(idxs_file, "rb") as f:
        mol_idxs = pickle.load(f)
    return fp_data, mol_idxs


def _validate_output_dir(out_dir: Path, overwrite_outputs: bool = False) -> None:
    if out_dir.exists():
        if not out_dir.is_dir():
            raise RuntimeError("Output dir should be a dir")
        if any(out_dir.iterdir()):
            if overwrite_outputs:
                for f in out_dir.iterdir():
                    if f.suffix in [".pkl", ".npy"]:
                        f.unlink()
            else:
                raise RuntimeError(f"Output dir {out_dir} has files")
    out_dir.mkdir(exist_ok=True)


# Validate also that the naming convention for the input files is correct
def _validate_input_dir(
    in_dir: Path | str, filename_idxs_are_slices: bool = False
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


def first_round(
    fp_file: str,
    double_cluster_init: bool,
    branching_factor: int,
    threshold: float,
    tolerance: float,
    out_dir: Path | str,
    filename_idxs_are_slices: bool = False,
    max_fps: int | None = None,
    mmap: bool = True,
    use_old_bblean: bool = False,
    initial_merge_criterion: str = "diameter",
) -> None:
    if use_old_bblean:
        from bbtools.bblean_v1 import BitBirch, set_merge  # type: ignore
    else:
        from bbtools.bblean import BitBirch, set_merge  # type: ignore
    out_dir = Path(out_dir)
    fps = np.load(fp_file, mmap_mode="r" if mmap else None)[:max_fps]

    # Fit the fps. fit_reinsert is necessary to keep track of proper molecule indices
    # Use indices of molecules in the current batch, according to the total set
    idx0, idx1 = map(int, fp_file.split(".")[0].split("_")[-2:])
    set_merge(initial_merge_criterion)  # Initial batch uses diameter BitBIRCH
    brc_diameter = BitBirch(branching_factor=branching_factor, threshold=threshold)
    if filename_idxs_are_slices:
        # idxs are <start_mol_idx>_<end_mol_idx>
        range_ = range(idx0, idx1)
        start_mol_idx = idx0
    else:
        # idxs are <file_idx>_<start_mol_idx>
        range_ = range(idx1, idx1 + len(fps))
        start_mol_idx = idx1
    if use_old_bblean:
        brc_diameter.fit_reinsert(fps, list(range_))
    else:
        brc_diameter.fit_reinsert(fps, list(range_), store_centroids=False)

    # Extract the BitFeatures info of the leaves to refine the top cluster
    # Use an if-statement since bblean v1.0 doesn't have this argument
    if use_old_bblean:
        fps_bfs, mols_bfs = brc_diameter.bf_to_np_refine(fps, initial_mol=start_mol_idx)
    else:
        fps_bfs, mols_bfs = brc_diameter.bf_to_np_refine(
            fps, initial_mol=start_mol_idx, return_fp_lists=True
        )
    del fps
    del brc_diameter
    gc.collect()

    if double_cluster_init:
        # Passing the previous BitFeatures through the new tree, singleton clusters are
        # passed at the end
        set_merge("tolerance", tolerance)  # 'tolerance' used to refine the tree
        brc_tolerance = BitBirch(branching_factor=branching_factor, threshold=threshold)
        for fp_type, mol_idxs in zip(fps_bfs, mols_bfs):
            brc_tolerance.fit_np_reinsert(fp_type, mol_idxs)

        # Get the info from the fitted BFs in compact list format
        if use_old_bblean:
            fps_bfs, mols_bfs = brc_tolerance.bf_to_np()
        else:
            fps_bfs, mols_bfs = brc_tolerance.bf_to_np(return_fp_lists=True)
        del brc_tolerance
        gc.collect()

    if use_old_bblean:
        for fp_type, mol_idxs in zip(fps_bfs, mols_bfs):
            suffix = f"round1_{idx0}_{str(fp_type.dtype)}"
            np.save(out_dir / f"fp_{suffix}", fp_type)
            with open(out_dir / f"mol_idxs_{suffix}.pkl", mode="wb") as f:
                pickle.dump(mol_idxs, f)
    else:
        numpy_streaming_save(fps_bfs, out_dir / f"round1_{idx0}")
        for fp_type, mol_idxs in zip(fps_bfs, mols_bfs):
            suffix = f"round1_{idx0}_{str(fp_type[0].dtype)}"
            with open(out_dir / f"mol_idxs_{suffix}.pkl", mode="wb") as f:
                pickle.dump(mol_idxs, f)


def second_round(
    chunk_info: tuple[int, tp.Sequence[str]],
    branching_factor: int,
    threshold: float,
    tolerance: float,
    out_dir: Path | str,
    mmap: bool = True,
    use_old_bblean: bool = False,
) -> None:
    if use_old_bblean:
        from bbtools.bblean_v1 import BitBirch, set_merge  # type: ignore
    else:
        from bbtools.bblean import BitBirch, set_merge  # type: ignore
    out_dir = Path(out_dir)
    chunk_idx, chunk_filenames = chunk_info

    set_merge("tolerance", tolerance)
    brc_chunk = BitBirch(branching_factor=branching_factor, threshold=threshold)
    for fp_filename in chunk_filenames:
        fp_data, mol_idxs = _load_fp_data_and_mol_idxs(out_dir, fp_filename, 1, mmap)
        brc_chunk.fit_np_reinsert(fp_data, mol_idxs)
        del mol_idxs
        del fp_data
        gc.collect()

    if use_old_bblean:
        fps_bfs, mols_bfs = brc_chunk.bf_to_np()
    else:
        fps_bfs, mols_bfs = brc_chunk.bf_to_np(return_fp_lists=True)
    del brc_chunk
    gc.collect()

    if use_old_bblean:
        for fp_type, mol_idxs in zip(fps_bfs, mols_bfs):
            suffix = f"round2_{chunk_idx}_{str(fp_type.dtype)}"
            np.save(out_dir / f"fp_{suffix}", fp_type)
            with open(out_dir / f"mol_idxs_{suffix}.pkl", mode="wb") as f:
                pickle.dump(mol_idxs, f)
    else:
        numpy_streaming_save(fps_bfs, out_dir / f"round2_{chunk_idx}")
        for fp_type, mol_idxs in zip(fps_bfs, mols_bfs):
            suffix = f"round2_{chunk_idx}_{str(fp_type[0].dtype)}"
            with open(out_dir / f"mol_idxs_{suffix}.pkl", mode="wb") as f:
                pickle.dump(mol_idxs, f)


def third_round(
    branching_factor: int,
    threshold: float,
    tolerance: float,
    out_dir: Path | str,
    mmap: bool = True,
    use_old_bblean: bool = False,
) -> None:
    if use_old_bblean:
        from bbtools.bblean_v1 import BitBirch, set_merge  # type: ignore
    else:
        from bbtools.bblean import BitBirch, set_merge  # type: ignore
    out_dir = Path(out_dir)

    set_merge("tolerance", tolerance)
    brc_final = BitBirch(branching_factor=branching_factor, threshold=threshold)

    sorted_files2 = _glob_and_sort_by_uint_bits(out_dir, "*round2*.npy")
    for fp_filename in sorted_files2:
        fp_data, mol_idxs = _load_fp_data_and_mol_idxs(out_dir, fp_filename, 2, mmap)
        brc_final.fit_np_reinsert(fp_data, mol_idxs)
        del fp_data
        del mol_idxs
        gc.collect()

    mol_ids = brc_final.get_cluster_mol_ids().copy()
    del brc_final
    gc.collect()

    with open(out_dir / "clusters.pkl", mode="wb") as f:
        pickle.dump(mol_ids, f)


# NOTE:
# In most tests a branching_factor of 254 was optimal for speed. 50 or 500 decreases
# performance
# There seems to be slightly lower RAM  usage with branching_factor > 254 (~1000)

# NOTE:
# The fps should be stored as .npy files
# In this new version, they should be packed np.uint8 files

# NOTE: 'double_cluster_init' indicates if the refinement of the batches is done
# before or after combining all the data in the final tree
#
# False: potentially slightly faster, but splits the biggest cluster of each batch
#     and doesn't try to re-form it until all the data goes through the final tree.
# True:  re-fits the splitted cluster in a new tree using tolerance merge this adds
#     a bit of time and memory overhead, so depending on the volume of data in each
#     batch it might need to be skipped, but this is a more solid/robust choice


def run_parallel_bitbirch(
    in_dir: Path,
    out_dir: Path,
    overwrite_outputs: bool = False,
    filename_idxs_are_slices: bool = False,
    round2_process_fraction: float = 1.0,
    initial_merge_criterion: str = "diameter",
    num_processes: int = 10,
    branching_factor: int = 254,
    threshold: float = 0.65,
    tolerance: float = 0.05,
    # Advanced
    fork: bool = False,
    bin_size: int = 10,
    use_mmap: bool = True,
    max_tasks_per_process: int = 1,
    double_cluster_init: bool = True,
    # Debug
    only_first_round: bool = False,
    monitor_rss: bool = True,
    monitor_rss_file_name: str = "monitor-rss.csv",
    monitor_rss_interval_s: float = 0.01,
    max_fps: int | None = None,
    max_files: int | None = None,
    verbose: bool = True,
    use_old_bblean: bool = False,
) -> None:
    # Capture the fn arguments TODO: this is dirty
    config = deepcopy(locals())

    console = get_console(silent=not verbose)

    # Set the multiprocessing start method
    if fork and not sys.platform == "linux":
        raise ValueError("'fork' is only available on Linux")
    if sys.platform == "linux":
        mp.set_start_method("fork" if fork else "forkserver")

    unique_id = str(uuid.uuid4()).split("-")[0]

    rss_path = out_dir / monitor_rss_file_name
    rss_path = rss_path.with_suffix(f".{unique_id}{rss_path.suffix}")
    timings_path = out_dir / f"timings.{unique_id}.json"
    config_path = out_dir / f"config.{unique_id}.json"
    monitor_rss_file_name = rss_path.name

    _validate_input_dir(in_dir, filename_idxs_are_slices)
    input_files = sorted(
        map(str, in_dir.glob("*.npy")),
        key=lambda x: int(x.split(".")[0].split("_")[-2]),
    )[:max_files]

    total_mem, avail_mem = system_mem_gib()
    config["timings_file_name"] = timings_path.name
    config["total_memory_gib"] = total_mem
    config["initial_available_memory_gib"] = avail_mem
    config["multiprocessing_start_method"] = mp.get_start_method()
    config["platform"] = sys.platform
    config["cpu"] = cpu_name()
    config["numpy_version"] = np.__version__
    config["python_version"] = sys.version.split()[0]
    config["visible_cpu_cores"] = os.cpu_count()
    config["in_dir"] = str(config["in_dir"])
    config["out_dir"] = str(config["out_dir"])
    config["input_filenames"] = input_files

    # BitBIRCH parameters
    common_kwargs: dict[str, tp.Any] = dict(
        branching_factor=branching_factor,
        threshold=threshold,
        tolerance=tolerance,
        mmap=use_mmap,
        use_old_bblean=use_old_bblean,
    )
    # Input and output dirs
    common_kwargs["out_dir"] = out_dir
    _validate_output_dir(common_kwargs["out_dir"], overwrite_outputs)

    # Dump config after checking if the output dir has files
    with open(config_path, mode="wt", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    start_all = time.perf_counter()
    if monitor_rss:
        mp.Process(
            target=monitor_rss_daemon,
            kwargs=dict(
                file=rss_path,
                interval_s=monitor_rss_interval_s,
                start_time=start_all,
            ),
            daemon=True,
        ).start()

    start_round1 = time.perf_counter()
    _start_msg = (
        f"parallel (max {num_processes} processes)"
        if num_processes > 1
        else "serial (1 process)"
    )
    console.print_banner()
    console.print(
        f"Starting {_start_msg} clustering\n\n"
        f"- Configuration path: {str(config_path)}\n"
        f"- Branching factor: {branching_factor}\n"
        f"- Initial merge strategy: [yellow]{initial_merge_criterion}[/yellow]\n"
        f"- Threshold: {threshold}\n"
        f"- Tolerance: {tolerance}\n"  # noqa:E501
        f"- Num. files loaded: {len(input_files)}\n"
        f"- Bin size for second round: {bin_size}\n"
        f"- Use mmap: {use_mmap}\n"
        f"- Monitor total RAM usage: {monitor_rss}\n",
        end="",
    )
    if num_processes > 1:
        console.print(
            f"- Multiprocessing method: [yellow]{mp.get_start_method()}[/yellow]\n",
            end="",
        )
    if not double_cluster_init:
        console.print(
            f"- Use 'double cluster init' on Round 1: {double_cluster_init}\n", end=""
        )
    if use_old_bblean:
        console.print("- DEBUG: Using old bb-lean version\n", end="")
    if max_files is not None:
        console.print(f"- DEBUG: Max files to load: {max_files}\n", end="")  # noqa:E501
    if max_fps is not None:
        console.print(
            f"- DEBUG: Max fingerprints to load per file: {max_fps}\n", end=""
        )
    console.print()

    console.print("Round 1: Processing initial batch of packed fingerprints...")
    round_1_fn: tp.Callable[[str], None] = functools.partial(
        first_round,
        double_cluster_init=double_cluster_init,
        max_fps=max_fps,
        filename_idxs_are_slices=filename_idxs_are_slices,
        initial_merge_criterion=initial_merge_criterion,
        **common_kwargs,
    )
    timings_s: dict[str, float | None] = {
        "round-1": None,
        "round-2": None,
        "round-3": None,
        "total": None,
    }
    num_processes = min(num_processes, len(input_files))
    console.print(f"Running on {len(input_files)} files with {num_processes} processes")
    if num_processes == 1:
        for file in input_files:
            round_1_fn(file)
    else:
        with mp.Pool(
            processes=num_processes,
            maxtasksperchild=max_tasks_per_process,
        ) as pool:
            pool.map(round_1_fn, input_files)
    sorted_files1 = _glob_and_sort_by_uint_bits(
        common_kwargs["out_dir"], "*round1*.npy"
    )
    chunks = list(enumerate(batched(sorted_files1, n=bin_size)))
    console.print(
        f"Finished. Collected {len(sorted_files1)} files,"
        f" chunked into {len(chunks)} chunks"
    )
    timings_s["round-1"] = time.perf_counter() - start_round1
    console.print(f"Time for round 1: {timings_s['round-1']:.4f} s")
    console.print_peak_mem(num_processes)

    if only_first_round:
        # Early exit for debugging
        timings_s["total"] = time.perf_counter() - start_all
        with open(timings_path, mode="wt", encoding="utf-8") as f:
            json.dump(timings_s, f, indent=4)
        return

    start_round2 = time.perf_counter()
    console.print("Round 2: Re-clustering in chunks...")
    round_2_fn: tp.Callable[[tuple[int, tp.Sequence[str]]], None] = functools.partial(
        second_round, **common_kwargs
    )

    num_processes = min(
        num_processes, round(num_processes * round2_process_fraction), len(chunks)
    )
    console.print(f"Running on {len(chunks)} chunks with {num_processes} processes")
    if num_processes == 1:
        for chunk_info_part in chunks:
            round_2_fn(chunk_info_part)
    else:
        with mp.Pool(
            processes=num_processes,
            maxtasksperchild=max_tasks_per_process,
        ) as pool:
            pool.map(round_2_fn, chunks)
    timings_s["round-2"] = time.perf_counter() - start_round2
    console.print(f"Time for round 2: {timings_s['round-2']:.4f} s")
    console.print_peak_mem(num_processes)

    start_final = time.perf_counter()
    console.print("Round 3: Final clustering...")
    third_round(**common_kwargs)

    timings_s["round-3"] = time.perf_counter() - start_final
    console.print(f"Time for round 3 (final clustering): {timings_s['round-3']:.4f} s")
    console.print_peak_mem(num_processes)

    timings_s["total"] = time.perf_counter() - start_all
    console.print(f"Total time: {timings_s['total']:.4f} s")
    with open(timings_path, mode="wt", encoding="utf-8") as f:
        json.dump(timings_s, f, indent=4)
