# BitBIRCH-Lean Python Package: An open-source clustering module based on iSIM.
#
# If you find this software useful please cite the following articles:
# - BitBIRCH: efficient clustering of large molecular libraries:
#   https://doi.org/10.1039/D5DD00030K
# - BitBIRCH Clustering Refinement Strategies:
#   https://doi.org/10.1021/acs.jcim.5c00627
# - BitBIRCH-Lean: TO-BE-ADDED
#
# Copyright (C) 2025  The Miranda-Quintana Lab and other BitBirch developers, including:
# - Ramon Alain Miranda Quintana <ramirandaq@gmail.com>, <quintana@chem.ufl.edu>
# - Krisztina Zsigmond <kzsigmond@ufl.edu>
# - Ignacio Pickering <ipickering@chem.ufl.edu>
# - Kenneth Lopez Perez <klopezperez@chem.ufl.edu>
# - Miroslav Lzicar <miroslav.lzicar@deepmedchem.com>
#
# Authors of this file are:
# - Ramon Alain Miranda Quintana <ramirandaq@gmail.com>, <quintana@chem.ufl.edu>
# - Ignacio Pickering <ipickering@chem.ufl.edu>
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# version 3 (SPDX-License-Identifier: GPL-3.0-only).
#
# Portions of ./bblean/bitbirch.py are licensed under the BSD 3-Clause License
# Copyright (c) 2007-2024 The scikit-learn developers. All rights reserved.
# (SPDX-License-Identifier: BSD-3-Clause). Copies or reproductions of code in the
# ./bblean/bitbirch.py file must in addition adhere to the BSD-3-Clause license terms. A
# copy of the BSD-3-Clause license can be located at the root of this repository, under
# ./LICENSES/BSD-3-Clause.txt.
#
# Portions of ./bblean/bitbirch.py were previously licensed under the LGPL 3.0
# license (SPDX-License-Identifier: LGPL-3.0-only), they are relicensed in this program
# as GPL-3.0, with permission of all original copyright holders:
# - Ramon Alain Miranda Quintana <ramirandaq@gmail.com>, <quintana@chem.ufl.edu>
# - Vicky (Vic) Jung <jungvicky@ufl.edu>
# - Kenneth Lopez Perez <klopezperez@chem.ufl.edu>
# - Kate Huddleston <kdavis2@chem.ufl.edu>
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this
# program. This copy can be located at the root of this repository, under
# ./LICENSES/GPL-3.0-only.txt.  If not, see <http://www.gnu.org/licenses/gpl-3.0.html>.
r"""Multi-round BitBirch workflow for clustering huge datasets in parallel"""
import math
import pickle
import gc
import typing as tp
import multiprocessing as mp
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from rich.console import Console

from bblean._console import get_console
from bblean._timer import Timer
from bblean._config import DEFAULTS
from bblean.utils import batched
from bblean.bitbirch import BitBirch
from bblean.fingerprints import _get_fps_file_num

__all__ = ["run_multiround_bitbirch"]


# Save a list of numpy arrays into a single array in a streaming fashion, avoiding
# stacking them in memory
def _numpy_streaming_save(fp_list: list[NDArray[np.integer]], path: Path | str) -> None:
    first_arr = np.ascontiguousarray(fp_list[0])
    header = np.lib.format.header_data_from_array_1_0(first_arr)
    header["shape"] = (len(fp_list), len(first_arr))
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".npy")
    with open(path, "wb") as f:
        np.lib.format.write_array_header_1_0(f, header)
        for arr in fp_list:
            np.ascontiguousarray(arr).tofile(f)


# Glob and sort by uint bits and label, if a console is passed then the number of output
# files is printed
def _get_prev_round_buf_and_mol_idxs_files(
    path: Path, round_idx: int, console: Console | None = None
) -> list[tuple[Path, Path]]:
    path = Path(path)
    # TODO: Important: What should be the logic for batching? currently there doesn't
    # seem to be much logic for grouping the files
    buf_files = sorted(path.glob(f"round-{round_idx - 1}-bufs*.npy"))
    idx_files = sorted(path.glob(f"round-{round_idx - 1}-idxs*.pkl"))
    if console is not None:
        console.print(f"    - Collected {len(buf_files)} buffer-index file pairs")
    return list(zip(buf_files, idx_files))


def _chunk_file_pairs_in_batches(
    file_pairs: tp.Sequence[tuple[Path, Path]],
    bin_size: int,
    console: Console | None = None,
) -> list[tuple[str, tuple[tuple[Path, Path], ...]]]:
    z = len(str(math.ceil(len(file_pairs) / bin_size)))
    batches = [
        (str(i).zfill(z), b) for i, b in enumerate(batched(file_pairs, bin_size))
    ]
    if console is not None:
        console.print(f"    - Chunked files into {len(batches)} batches")
    return batches


def _load_bufs_and_mol_idxs(
    buf_path: Path, idx_path: Path, use_mmap: bool = True
) -> tuple[NDArray[tp.Any], tp.Any]:
    bufs = np.load(buf_path, mmap_mode="r" if use_mmap else None)
    with open(idx_path, "rb") as f:
        mol_idxs = pickle.load(f)
    return bufs, mol_idxs


def _save_bufs_and_mol_idxs(
    out_dir: Path,
    fps_bfs: dict[str, tp.Any],
    mols_bfs: dict[str, tp.Any],
    label: str,
    round_idx: int,
) -> None:
    for dtype, buf_list in fps_bfs.items():
        suffix = f".label-{label}-{dtype.replace('8', '08')}"
        _numpy_streaming_save(buf_list, out_dir / f"round-{round_idx}-bufs{suffix}.npy")
        with open(out_dir / f"round-{round_idx}-idxs{suffix}.pkl", mode="wb") as f:
            pickle.dump(mols_bfs[dtype], f)


class _InitialRound:
    def __init__(
        self,
        double_cluster_init: bool,
        branching_factor: int,
        threshold: float,
        tolerance: float,
        out_dir: Path | str,
        n_features: int | None = None,
        max_fps: int | None = None,
        use_mmap: bool = True,
        merge_criterion: str = "diameter",
        input_is_packed: bool = True,
    ) -> None:
        self.n_features = n_features
        self.double_cluster_init = double_cluster_init
        self.branching_factor = branching_factor
        self.threshold = threshold
        self.tolerance = tolerance
        self.out_dir = Path(out_dir)
        self.max_fps = max_fps
        self.use_mmap = use_mmap
        self.merge_criterion = merge_criterion
        self.input_is_packed = input_is_packed

    def __call__(self, file_info: tuple[str, Path, int, int]) -> None:
        file_label, fp_file, start_idx, end_idx = file_info
        fps = np.load(fp_file, mmap_mode="r" if self.use_mmap else None)[: self.max_fps]

        # First fit the fps in each process, in parallel.
        # `reinsert_indices` required to keep track of mol idxs in different processes.
        brc_init = BitBirch(
            branching_factor=self.branching_factor,
            threshold=self.threshold,
            merge_criterion=self.merge_criterion,
        )
        range_ = range(start_idx, end_idx)
        brc_init.fit(
            fps,
            reinsert_indices=range_,
            n_features=self.n_features,
            input_is_packed=self.input_is_packed,
        )
        # Extract the BitFeatures of the leaves, breaking the largest cluster apart
        fps_bfs, mols_bfs = brc_init._bf_to_np_refine(fps, initial_mol=start_idx)
        del fps
        del brc_init
        gc.collect()

        if self.double_cluster_init:
            # Rebuild the tree again, reinserting the fps from the largest cluster
            brc_tolerance = BitBirch(
                branching_factor=self.branching_factor,
                threshold=self.threshold,
                merge_criterion="tolerance",
                tolerance=self.tolerance,
            )
            for bufs, mol_idxs in zip(fps_bfs.values(), mols_bfs.values()):
                brc_tolerance._fit_np(
                    bufs,
                    reinsert_index_sequences=mol_idxs,
                )
            fps_bfs, mols_bfs = brc_tolerance._bf_to_np()
            del brc_tolerance
            gc.collect()

        _save_bufs_and_mol_idxs(self.out_dir, fps_bfs, mols_bfs, file_label, 1)


class _TreeMergingRound:
    def __init__(
        self,
        branching_factor: int,
        threshold: float,
        tolerance: float,
        round_idx: int,
        out_dir: Path | str,
        use_mmap: bool = True,
        is_final: bool = False,
    ) -> None:
        self.branching_factor = branching_factor
        self.threshold = threshold
        self.tolerance = tolerance
        self.round_idx = round_idx
        self.out_dir = Path(out_dir)
        self.use_mmap = use_mmap
        self.is_final = is_final

    def __call__(self, batch_info: tuple[str, tp.Sequence[tuple[Path, Path]]]) -> None:
        batch_label, batch_path_pairs = batch_info
        brc_merger = BitBirch(
            branching_factor=self.branching_factor,
            threshold=self.threshold,
            merge_criterion="tolerance",
            tolerance=self.tolerance,
        )
        # Rebuild a tree, inserting all BitFeatures from the corresponding batch
        for buf_path, idx_path in batch_path_pairs:
            bufs, mol_idxs = _load_bufs_and_mol_idxs(buf_path, idx_path, self.use_mmap)
            brc_merger._fit_np(bufs, reinsert_index_sequences=mol_idxs)
            del mol_idxs
            del bufs
            gc.collect()

        # If this this is the final round save the clusters and exit
        # In this case self.round_idx and batch_label are unused
        if self.is_final:
            cluster_mol_ids = brc_merger.get_cluster_mol_ids()
            del brc_merger
            gc.collect()
            with open(self.out_dir / "clusters.pkl", mode="wb") as f:
                pickle.dump(cluster_mol_ids, f)
            return

        # Otherwise fetch and save the bufs and idxs for the next round
        fps_bfs, mols_bfs = brc_merger._bf_to_np()
        del brc_merger
        gc.collect()

        _save_bufs_and_mol_idxs(
            self.out_dir, fps_bfs, mols_bfs, batch_label, self.round_idx
        )


# Create a list of tuples of labels, file paths and start-end idxs
def _get_files_range_tuples(
    files: tp.Sequence[Path],
) -> list[tuple[str, Path, int, int]]:
    running_idx = 0
    files_info = []
    z = len(str(len(files)))
    for i, file in enumerate(files):
        start_idx = running_idx
        end_idx = running_idx + _get_fps_file_num(file)
        files_info.append((str(i).zfill(z), file, start_idx, end_idx))
        running_idx = end_idx
    return files_info


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
    n_features: int | None = None,
    input_is_packed: bool = True,
    num_initial_processes: int = 10,
    num_midsection_processes: int | None = None,
    initial_merge_criterion: str = DEFAULTS.merge_criterion,
    branching_factor: int = DEFAULTS.branching_factor,
    threshold: float = DEFAULTS.threshold,
    tolerance: float = DEFAULTS.tolerance,
    # Advanced
    num_midsection_rounds: int = 1,
    bin_size: int = 10,
    use_mmap: bool = DEFAULTS.use_mmap,
    max_tasks_per_process: int = 1,
    double_cluster_init: bool = True,
    # Debug
    only_first_round: bool = False,
    max_fps: int | None = None,
    verbose: bool = False,
) -> Timer:
    r"""Perform (possibly parallel) multi-round BitBirch clustering

    ..  warning::

        The functionality provided by this function is stable, but its API
        (the arguments it takes and its return values) may change in the future.
    """
    # Returns timing and for the different rounds
    # TODO: Also return peak-rss
    console = get_console(silent=not verbose)

    if num_midsection_processes is None:
        num_midsection_processes = num_initial_processes
    else:
        # Sanity check
        if num_midsection_processes > num_initial_processes:
            raise ValueError("Num. midsection procs. must be <= num. initial processes")

    # Common params to all rounds BitBIRCH
    common_kwargs: dict[str, tp.Any] = dict(
        branching_factor=branching_factor,
        threshold=threshold,
        tolerance=tolerance,
        use_mmap=use_mmap,
        out_dir=out_dir,
    )
    timer = Timer()
    timer.init_timing("total")

    # Get starting and ending idxs for each file, and collect them into tuples
    files_range_tuples = _get_files_range_tuples(input_files)
    num_files = len(input_files)

    # Initial round of clustering
    round_idx = 1
    timer.init_timing(f"round-{round_idx}")
    console.print(f"(Initial) Round {round_idx}: Cluster initial batch of fingerprints")

    initial_fn = _InitialRound(
        n_features=n_features,
        double_cluster_init=double_cluster_init,
        max_fps=max_fps,
        merge_criterion=initial_merge_criterion,
        input_is_packed=input_is_packed,
        **common_kwargs,
    )
    num_ps = min(num_initial_processes, num_files)
    console.print(f"    - Processing {num_files} inputs with {num_ps} processes")
    with console.status("[italic]BitBirching...[/italic]", spinner="dots"):
        if num_ps == 1:
            for tup in files_range_tuples:
                initial_fn(tup)
        else:
            with mp.Pool(
                processes=num_ps, maxtasksperchild=max_tasks_per_process
            ) as pool:
                pool.map(initial_fn, files_range_tuples)

    timer.end_timing(f"round-{round_idx}", console)
    console.print_peak_mem(num_ps)

    if only_first_round:  # Early exit for debugging
        timer.end_timing("total")
        return timer

    # Mid-section "Tree-Merging" rounds of clustering
    for _ in range(num_midsection_rounds):
        round_idx += 1
        timer.init_timing(f"round-{round_idx}")
        console.print(f"(Midsection) Round {round_idx}: Re-clustering in chunks")

        file_pairs = _get_prev_round_buf_and_mol_idxs_files(out_dir, round_idx, console)
        batches = _chunk_file_pairs_in_batches(file_pairs, bin_size, console)
        merging_fn = _TreeMergingRound(round_idx=round_idx, **common_kwargs)
        num_ps = min(num_midsection_processes, len(batches))
        console.print(f"    - Processing {len(batches)} inputs with {num_ps} processes")
        with console.status("[italic]BitBirching...[/italic]", spinner="dots"):
            if num_ps == 1:
                for batch_info in batches:
                    merging_fn(batch_info)
            else:
                with mp.Pool(
                    processes=num_ps, maxtasksperchild=max_tasks_per_process
                ) as pool:
                    pool.map(merging_fn, batches)

        timer.end_timing(f"round-{round_idx}", console)
        console.print_peak_mem(num_ps)

    # Final "Tree-Merging" round of clustering
    round_idx += 1
    timer.init_timing(f"round-{round_idx}")
    console.print(f"(Final) Round {round_idx}: Final round of clustering")
    file_pairs = _get_prev_round_buf_and_mol_idxs_files(out_dir, round_idx, console)

    final_fn = _TreeMergingRound(round_idx=round_idx, is_final=True, **common_kwargs)
    with console.status("[italic]BitBirching...[/italic]", spinner="dots"):
        final_fn(("", file_pairs))

    timer.end_timing(f"round-{round_idx}", console)
    console.print_peak_mem(num_ps)

    console.print()
    timer.end_timing("total", console, indent=False)
    return timer
