# BitBIRCH-Lean Python Package: An open-source clustering module based on iSIM.
#
# If you find this software useful please cite the following articles:
# - BitBIRCH: efficient clustering of large molecular libraries:
#   https://doi.org/10.1039/D5DD00030K
# - BitBIRCH Clustering Refinement Strategies:
#   https://doi.org/10.1021/acs.jcim.5c00627
# - BitBIRCh-Lean:
#   (preprint) https://www.biorxiv.org/content/10.1101/2025.10.22.684015v1
#
# Copyright (C) 2025  The Miranda-Quintana Lab and other BitBirch developers, comprised
# exclusively by:
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
import sys
import math
import pickle
import typing as tp
import multiprocessing as mp
from pathlib import Path

from rich.console import Console
import numpy as np
from numpy.typing import NDArray


from bblean._console import get_console
from bblean._timer import Timer
from bblean._config import DEFAULTS
from bblean.utils import batched
from bblean.bitbirch import BitBirch
from bblean.fingerprints import _get_fps_file_num

__all__ = ["run_multiround_bitbirch"]


# Save a list of numpy arrays into a single array in a streaming fashion, avoiding
# stacking them in memory
def _numpy_streaming_save(
    fp_list: list[NDArray[np.integer]] | NDArray[np.integer], path: Path | str
) -> None:
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


def _sort_batch(b: tp.Sequence[tuple[Path, Path]]) -> tuple[tuple[Path, Path], ...]:
    return tuple(
        sorted(
            b,
            key=lambda b: int(b[0].name.split("uint")[-1].split(".")[0]),
            reverse=True,
        )
    )


def _chunk_file_pairs_in_batches(
    file_pairs: tp.Sequence[tuple[Path, Path]],
    bin_size: int,
    console: Console | None = None,
) -> list[tuple[str, tuple[tuple[Path, Path], ...]]]:
    z = len(str(math.ceil(len(file_pairs) / bin_size)))
    # Within each batch, sort the files by starting with the uint16 files, followed by
    # uint8 files, this helps that (approximately) the largest clusters are fitted first
    # which may improve final cluster quality
    batches = [
        (str(i).zfill(z), _sort_batch(b))
        for i, b in enumerate(batched(file_pairs, bin_size))
    ]
    if console is not None:
        console.print(f"    - Chunked files into {len(batches)} batches")
    return batches


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
        branching_factor: int,
        threshold: float,
        tolerance: float,
        out_dir: Path | str,
        refinement_before_midsection: str,
        refine_threshold_change: float,
        refine_merge_criterion: str,
        n_features: int | None = None,
        max_fps: int | None = None,
        merge_criterion: str = DEFAULTS.merge_criterion,
        input_is_packed: bool = True,
    ) -> None:
        self.n_features = n_features
        self.refinement_before_midsection = refinement_before_midsection
        if refinement_before_midsection not in ["full", "split", "none"]:
            raise ValueError(f"Unknown refinement kind {refinement_before_midsection}")
        self.branching_factor = branching_factor
        self.threshold = threshold
        self.tolerance = tolerance
        self.out_dir = Path(out_dir)
        self.max_fps = max_fps
        self.merge_criterion = merge_criterion
        self.refine_merge_criterion = refine_merge_criterion
        self.input_is_packed = input_is_packed
        self.refine_threshold_change = refine_threshold_change

    def __call__(self, file_info: tuple[str, Path, int, int]) -> None:
        file_label, fp_file, start_idx, end_idx = file_info

        # First fit the fps in each process, in parallel.
        # `reinsert_indices` required to keep track of mol idxs in different processes.
        tree = BitBirch(
            branching_factor=self.branching_factor,
            threshold=self.threshold,
            merge_criterion=self.merge_criterion,
        )

        range_ = range(start_idx, end_idx)
        tree.fit(
            fp_file,
            reinsert_indices=range_,
            n_features=self.n_features,
            input_is_packed=self.input_is_packed,
            max_fps=self.max_fps,
        )
        # Extract the BitFeatures of the leaves, breaking the largest cluster(s) apart,
        # to prepare for refinement
        tree.delete_internal_nodes()
        if self.refinement_before_midsection == "none":
            fps_bfs, mols_bfs = tree._bf_to_np()
        elif self.refinement_before_midsection in ["split", "full"]:
            fps_bfs, mols_bfs = tree._bf_to_np_refine(fp_file, initial_mol=start_idx)
            if self.refinement_before_midsection == "full":
                # Finish the first refinement step internally in this round
                tree.reset()
                tree.set_merge(
                    merge_criterion=self.refine_merge_criterion,
                    tolerance=self.tolerance,
                    threshold=self.threshold + self.refine_threshold_change,
                )
                for bufs, mol_idxs in zip(fps_bfs.values(), mols_bfs.values()):
                    tree._fit_buffers(bufs, reinsert_index_seqs=mol_idxs)
                    del mol_idxs
                    del bufs

                tree.delete_internal_nodes()
                fps_bfs, mols_bfs = tree._bf_to_np()

        _save_bufs_and_mol_idxs(self.out_dir, fps_bfs, mols_bfs, file_label, 1)


class _TreeMergingRound:
    def __init__(
        self,
        branching_factor: int,
        threshold: float,
        tolerance: float,
        round_idx: int,
        out_dir: Path | str,
        split_largest_cluster: bool,
        merge_criterion: str,
        all_fp_paths: tp.Sequence[Path] = (),
    ) -> None:
        self.all_fp_paths = list(all_fp_paths)
        self.branching_factor = branching_factor
        self.threshold = threshold
        self.tolerance = tolerance
        self.round_idx = round_idx
        self.out_dir = Path(out_dir)
        self.split_largest_cluster = split_largest_cluster
        self.merge_criterion = merge_criterion

    def __call__(self, batch_info: tuple[str, tp.Sequence[tuple[Path, Path]]]) -> None:
        batch_label, batch_path_pairs = batch_info
        tree = BitBirch(
            branching_factor=self.branching_factor,
            threshold=self.threshold,
            merge_criterion=self.merge_criterion,
            tolerance=self.tolerance,
        )
        # Rebuild a tree, inserting all BitFeatures from the corresponding batch
        for buf_path, idx_path in batch_path_pairs:
            with open(idx_path, "rb") as f:
                mol_idxs = pickle.load(f)
            tree._fit_buffers(buf_path, reinsert_index_seqs=mol_idxs)
            del mol_idxs

        # Either do a refinement step, or fetch and save the bufs and idxs for the next
        # round
        tree.delete_internal_nodes()
        if self.split_largest_cluster:
            fps_bfs, mols_bfs = tree._bf_to_np_refine(self.all_fp_paths)
        else:
            fps_bfs, mols_bfs = tree._bf_to_np()
        _save_bufs_and_mol_idxs(
            self.out_dir, fps_bfs, mols_bfs, batch_label, self.round_idx
        )


class _FinalTreeMergingRound(_TreeMergingRound):
    def __init__(
        self,
        branching_factor: int,
        threshold: float,
        tolerance: float,
        merge_criterion: str,
        out_dir: Path | str,
        save_tree: bool,
        save_centroids: bool,
    ) -> None:
        super().__init__(
            branching_factor,
            threshold,
            tolerance,
            -1,
            out_dir,
            False,
            merge_criterion,
            (),
        )
        self.save_tree = save_tree
        self.save_centroids = save_centroids

    def __call__(self, batch_info: tuple[str, tp.Sequence[tuple[Path, Path]]]) -> None:
        batch_path_pairs = batch_info[1]
        tree = BitBirch(
            branching_factor=self.branching_factor,
            threshold=self.threshold,
            merge_criterion=self.merge_criterion,
            tolerance=self.tolerance,
        )
        # Rebuild a tree, inserting all BitFeatures from the corresponding batch
        for buf_path, idx_path in batch_path_pairs:
            with open(idx_path, "rb") as f:
                mol_idxs = pickle.load(f)
            tree._fit_buffers(buf_path, reinsert_index_seqs=mol_idxs)
            del mol_idxs

        # Save clusters and exit
        if self.save_tree:
            # TODO: Find alternative solution
            tree.save(self.out_dir / "bitbirch.pkl")
        tree.delete_internal_nodes()
        if self.save_centroids:
            output = tree.get_centroids_mol_ids()
            with open(self.out_dir / "clusters.pkl", mode="wb") as f:
                pickle.dump(output["mol_ids"], f)
            with open(self.out_dir / "cluster-centroids-packed.pkl", mode="wb") as f:
                pickle.dump(output["centroids"], f)
        else:
            with open(self.out_dir / "clusters.pkl", mode="wb") as f:
                pickle.dump(tree.get_cluster_mol_ids(), f)


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


# NOTE: 'full_refinement_before_midsection' indicates if the refinement of the batches
# is fully done after the tree-merging rounds, or if the data is only split before the
# tree-merging rounds
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
    midsection_threshold_change: float = DEFAULTS.refine_threshold_change,
    tolerance: float = DEFAULTS.tolerance,
    # Advanced
    num_midsection_rounds: int = 1,
    bin_size: int = 10,
    max_tasks_per_process: int = 1,
    refinement_before_midsection: str = "full",
    split_largest_after_each_midsection_round: bool = False,
    midsection_merge_criterion: str = DEFAULTS.refine_merge_criterion,
    final_merge_criterion: str | None = None,
    mp_context: tp.Any = None,
    save_tree: bool = False,
    save_centroids: bool = True,
    # Debug
    max_fps: int | None = None,
    verbose: bool = False,
    cleanup: bool = True,
) -> Timer:
    r"""Perform (possibly parallel) multi-round BitBirch clustering

    .. warning::

        The functionality provided by this function is stable, but its API
        (the arguments it takes and its return values) may change in the future.
    """
    if final_merge_criterion is None:
        final_merge_criterion = midsection_merge_criterion
    if mp_context is None:
        mp_context = mp.get_context("forkserver" if sys.platform == "linux" else None)
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
        tolerance=tolerance,
        out_dir=out_dir,
    )
    timer = Timer()
    timer.init_timing("total")

    # Get starting and ending idxs for each file, and collect them into tuples
    files_range_tuples = _get_files_range_tuples(input_files)  # correct
    num_files = len(input_files)

    # Initial round of clustering
    round_idx = 1
    timer.init_timing(f"round-{round_idx}")
    console.print(f"(Initial) Round {round_idx}: Cluster initial batch of fingerprints")

    initial_fn = _InitialRound(
        n_features=n_features,
        refinement_before_midsection=refinement_before_midsection,
        max_fps=max_fps,
        merge_criterion=initial_merge_criterion,
        input_is_packed=input_is_packed,
        threshold=threshold,
        refine_merge_criterion=midsection_merge_criterion,
        refine_threshold_change=midsection_threshold_change,
        **common_kwargs,
    )
    num_ps = min(num_initial_processes, num_files)
    console.print(f"    - Processing {num_files} inputs with {num_ps} processes")
    with console.status("[italic]BitBirching...[/italic]", spinner="dots"):
        if num_ps == 1:
            for tup in files_range_tuples:
                initial_fn(tup)
        else:
            with mp_context.Pool(
                processes=num_ps, maxtasksperchild=max_tasks_per_process
            ) as pool:
                pool.map(initial_fn, files_range_tuples)

    timer.end_timing(f"round-{round_idx}", console)
    console.print_peak_mem(out_dir)

    # Mid-section "Tree-Merging" rounds of clustering
    for _ in range(num_midsection_rounds):
        round_idx += 1
        timer.init_timing(f"round-{round_idx}")
        console.print(f"(Midsection) Round {round_idx}: Re-clustering in chunks")

        file_pairs = _get_prev_round_buf_and_mol_idxs_files(out_dir, round_idx, console)
        batches = _chunk_file_pairs_in_batches(file_pairs, bin_size, console)
        merging_fn = _TreeMergingRound(
            round_idx=round_idx,
            all_fp_paths=input_files,
            split_largest_cluster=split_largest_after_each_midsection_round,
            merge_criterion=midsection_merge_criterion,
            threshold=threshold + midsection_threshold_change,
            **common_kwargs,
        )
        num_ps = min(num_midsection_processes, len(batches))
        console.print(f"    - Processing {len(batches)} inputs with {num_ps} processes")
        with console.status("[italic]BitBirching...[/italic]", spinner="dots"):
            if num_ps == 1:
                for batch_info in batches:
                    merging_fn(batch_info)
            else:
                with mp_context.Pool(
                    processes=num_ps, maxtasksperchild=max_tasks_per_process
                ) as pool:
                    pool.map(merging_fn, batches)

        timer.end_timing(f"round-{round_idx}", console)
        console.print_peak_mem(out_dir)

    # Final "Tree-Merging" round of clustering
    round_idx += 1
    timer.init_timing(f"round-{round_idx}")
    console.print(f"(Final) Round {round_idx}: Final round of clustering")
    file_pairs = _get_prev_round_buf_and_mol_idxs_files(out_dir, round_idx, console)

    final_fn = _FinalTreeMergingRound(
        save_tree=save_tree,
        save_centroids=save_centroids,
        merge_criterion=final_merge_criterion,
        threshold=threshold + midsection_threshold_change,
        **common_kwargs,
    )
    with console.status("[italic]BitBirching...[/italic]", spinner="dots"):
        final_fn(("", file_pairs))

    timer.end_timing(f"round-{round_idx}", console)
    console.print_peak_mem(out_dir)
    # Remove intermediate files
    if cleanup:
        for f in out_dir.glob("round-*.npy"):
            f.unlink()
        for f in out_dir.glob("round-*.pkl"):
            f.unlink()
    console.print()
    timer.end_timing("total", console, indent=False)
    return timer
