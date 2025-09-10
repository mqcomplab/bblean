import functools
import pickle
import gc
import re
import time
import typing as tp
import multiprocessing as mp
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from bbtools._console import get_console
from bbtools.utils import numpy_streaming_save, batched, _import_bitbirch_variant
from bbtools.config import DEFAULTS


def _glob_and_sort_by_uint_bits(path: Path | str, glob_expr: str) -> list[Path]:
    path = Path(path)
    return sorted(
        path.glob(glob_expr),
        key=lambda x: int((re.search(r"uint(\d+)", x.name) or [0, 0])[1]),
        reverse=True,
    )


def _load_fp_data_and_mol_idxs(
    out_dir: Path,
    fp_filepath: Path,
    round: int,
    mmap: bool = True,
) -> tuple[NDArray[tp.Any], tp.Any]:
    fp_data = np.load(fp_filepath, mmap_mode="r" if mmap else None)
    # Infer the mol_idxs filename from the fp_np filename, and fetch the mol_idxs
    count, dtype = fp_filepath.name.split(".")[0].split("_")[-2:]
    idxs_file = out_dir / f"mol_idxs_round{round}_{count}_{dtype}.pkl"
    with open(idxs_file, "rb") as f:
        mol_idxs = pickle.load(f)
    return fp_data, mol_idxs


def first_round(
    fp_file: Path,
    double_cluster_init: bool,
    branching_factor: int,
    threshold: float,
    tolerance: float,
    out_dir: Path | str,
    filename_idxs_are_slices: bool = False,
    max_fps: int | None = None,
    mmap: bool = True,
    bitbirch_variant: str = "lean",
    merge_criterion: str = "diameter",
) -> None:

    BitBirch, set_merge = _import_bitbirch_variant(bitbirch_variant)
    out_dir = Path(out_dir)
    fps = np.load(fp_file, mmap_mode="r" if mmap else None)[:max_fps]

    # Fit the fps. fit_reinsert is necessary to keep track of proper molecule indices
    # Use indices of molecules in the current batch, according to the total set
    idx0, idx1 = map(int, fp_file.name.split(".")[0].split("_")[-2:])
    set_merge(merge_criterion)  # Initial batch uses diameter BitBIRCH
    brc_diameter = BitBirch(branching_factor=branching_factor, threshold=threshold)
    if filename_idxs_are_slices:
        # idxs are <start_mol_idx>_<end_mol_idx>
        range_ = range(idx0, idx1)
        start_mol_idx = idx0
    else:
        # idxs are <file_idx>_<start_mol_idx>
        range_ = range(idx1, idx1 + len(fps))
        start_mol_idx = idx1
    if bitbirch_variant != "lean":
        brc_diameter.fit_reinsert(fps, list(range_))
    else:
        brc_diameter.fit_reinsert(fps, list(range_), store_centroids=False)

    # Extract the BitFeatures info of the leaves to refine the top cluster
    # Use an if-statement since bblean v1.0 doesn't have this argument
    if bitbirch_variant != "lean":
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
        if bitbirch_variant != "lean":
            fps_bfs, mols_bfs = brc_tolerance.bf_to_np()
        else:
            fps_bfs, mols_bfs = brc_tolerance.bf_to_np(return_fp_lists=True)
        del brc_tolerance
        gc.collect()

    if bitbirch_variant != "lean":
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
    chunk_info: tuple[int, tp.Sequence[Path]],
    branching_factor: int,
    threshold: float,
    tolerance: float,
    out_dir: Path | str,
    mmap: bool = True,
    bitbirch_variant: str = "lean",
) -> None:
    BitBirch, set_merge = _import_bitbirch_variant(bitbirch_variant)
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

    if bitbirch_variant != "lean":
        fps_bfs, mols_bfs = brc_chunk.bf_to_np()
    else:
        fps_bfs, mols_bfs = brc_chunk.bf_to_np(return_fp_lists=True)
    del brc_chunk
    gc.collect()

    if bitbirch_variant != "lean":
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
    bitbirch_variant: str = "lean",
) -> None:

    BitBirch, set_merge = _import_bitbirch_variant(bitbirch_variant)
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
def run_multiround_bitbirch(
    input_files: tp.Sequence[Path],
    out_dir: Path,
    filename_idxs_are_slices: bool = False,
    round2_process_fraction: float = 1.0,
    merge_criterion: str = DEFAULTS.merge_criterion,
    num_processes: int = 10,
    branching_factor: int = 254,
    threshold: float = DEFAULTS.threshold,
    tolerance: float = DEFAULTS.tolerance,
    # Advanced
    bin_size: int = 10,
    use_mmap: bool = DEFAULTS.use_mmap,
    max_tasks_per_process: int = 1,
    double_cluster_init: bool = True,
    # Debug
    only_first_round: bool = False,
    max_fps: int | None = None,
    max_files: int | None = None,
    verbose: bool = False,
    bitbirch_variant: str = "lean",
) -> dict[str, float | None]:
    # Returns timing and for the different rounds
    # TODO: Also return peak-rss
    console = get_console(silent=not verbose)

    # Common BitBIRCH params
    common_kwargs: dict[str, tp.Any] = dict(
        branching_factor=branching_factor,
        threshold=threshold,
        tolerance=tolerance,
        mmap=use_mmap,
        bitbirch_variant=bitbirch_variant,
        out_dir=out_dir,
    )
    start_time = time.perf_counter()
    start_round1 = time.perf_counter()
    timings_s: dict[str, float | None] = {
        "round-1": None,
        "round-2": None,
        "round-3": None,
        "total": None,
    }

    console.print("Round 1: Processing initial batch of packed fingerprints...")
    round_1_fn: tp.Callable[[Path], None] = functools.partial(
        first_round,
        double_cluster_init=double_cluster_init,
        max_fps=max_fps,
        filename_idxs_are_slices=filename_idxs_are_slices,
        merge_criterion=merge_criterion,
        **common_kwargs,
    )
    num_ps = min(num_processes, len(input_files))
    console.print(f"Running on {len(input_files)} files with {num_ps} processes")
    if num_ps == 1:
        for file in input_files:
            round_1_fn(file)
    else:
        with mp.Pool(processes=num_ps, maxtasksperchild=max_tasks_per_process) as pool:
            pool.map(round_1_fn, input_files)
    sorted_files1 = _glob_and_sort_by_uint_bits(out_dir, "*round1*.npy")
    chunks = list(enumerate(batched(sorted_files1, n=bin_size)))
    console.print(f"Finished. Collected {len(sorted_files1)} output files")
    console.print(f"Chunked files into {len(chunks)} chunks")
    timings_s["round-1"] = time.perf_counter() - start_round1
    console.print(f"Time for round 1: {timings_s['round-1']:.4f} s")
    console.print_peak_mem(num_ps)

    if only_first_round:  # Early exit for debugging
        timings_s["total"] = time.perf_counter() - start_time
        return timings_s

    start_round2 = time.perf_counter()
    console.print("Round 2: Re-clustering in chunks...")
    round_2_fn: tp.Callable[[tuple[int, tp.Sequence[Path]]], None] = functools.partial(
        second_round, **common_kwargs
    )

    num_ps = min(num_ps, round(num_ps * round2_process_fraction), len(chunks))
    console.print(f"Running on {len(chunks)} chunks with {num_ps} processes")
    if num_ps == 1:
        for chunk_info_part in chunks:
            round_2_fn(chunk_info_part)
    else:
        with mp.Pool(processes=num_ps, maxtasksperchild=max_tasks_per_process) as pool:
            pool.map(round_2_fn, chunks)
    timings_s["round-2"] = time.perf_counter() - start_round2
    console.print(f"Time for round 2: {timings_s['round-2']:.4f} s")
    console.print_peak_mem(num_ps)

    start_final = time.perf_counter()
    console.print("Round 3: Final clustering...")
    third_round(**common_kwargs)

    timings_s["round-3"] = time.perf_counter() - start_final
    console.print(f"Time for round 3 (final clustering): {timings_s['round-3']:.4f} s")
    console.print_peak_mem(num_ps)

    timings_s["total"] = time.perf_counter() - start_time
    console.print(f"Total time: {timings_s['total']:.4f} s")
    return timings_s
