# BitBIRCH-Lean Python Package: An open-source clustering module based on iSIM.
#
# If you find this work useful please cite the following articles:
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
import pickle
import gc
import re
import time
import typing as tp
import multiprocessing as mp
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from bblean._console import get_console
from bblean.utils import numpy_streaming_save, batched
from bblean.config import DEFAULTS
from bblean.bitbirch import BitBirch  # type: ignore


def _glob_and_sort_by_uint_bits(path: Path | str, glob_expr: str) -> list[Path]:
    path = Path(path)
    return sorted(
        path.glob(glob_expr),
        key=lambda x: int((re.search(r"uint(\d+)", x.name) or [0, 0])[1]),
        reverse=True,
    )


def _load_buffers_and_mol_idxs(
    out_dir: Path,
    fp_filepath: Path,
    round: int,
    use_mmap: bool = True,
) -> tuple[NDArray[tp.Any], tp.Any]:
    fp_data = np.load(fp_filepath, mmap_mode="r" if use_mmap else None)
    # Infer the mol_idxs filename from the fp_np filename, and fetch the mol_idxs
    count, dtype = fp_filepath.name.split(".")[0].split("_")[-2:]
    idxs_file = out_dir / f"mol_idxs_round_{round}_{count}_{dtype}.pkl"
    with open(idxs_file, "rb") as f:
        mol_idxs = pickle.load(f)
    return fp_data, mol_idxs


class InitialRound:
    def __init__(
        self,
        n_features: int,
        double_cluster_init: bool,
        branching_factor: int,
        threshold: float,
        tolerance: float,
        out_dir: Path | str,
        filename_idxs_are_slices: bool = False,
        max_fps: int | None = None,
        use_mmap: bool = True,
        merge_criterion: str = "diameter",
    ) -> None:
        self.n_features = n_features
        self.double_cluster_init = double_cluster_init
        self.branching_factor = branching_factor
        self.threshold = threshold
        self.tolerance = tolerance
        self.out_dir = Path(out_dir)
        self.filename_idxs_are_slices = filename_idxs_are_slices
        self.max_fps = max_fps
        self.use_mmap = use_mmap
        self.merge_criterion = merge_criterion

    def __call__(self, fp_file: Path) -> None:
        fps = np.load(fp_file, mmap_mode="r" if self.use_mmap else None)[: self.max_fps]

        # Fit the fps. `reinsert_indices` is necessary to keep track of proper molecule
        # indices. Use indices of molecules in the current batch, according to the total
        # set
        idx0, idx1 = map(int, fp_file.name.split(".")[0].split("_")[-2:])
        brc_diameter = BitBirch(
            branching_factor=self.branching_factor,
            threshold=self.threshold,
            merge_criterion=self.merge_criterion,
        )
        # TODO: Check this
        if self.filename_idxs_are_slices:
            # idxs are <start_mol_idx>_<end_mol_idx>
            range_ = range(idx0, idx1)
            start_mol_idx = idx0
        else:
            # idxs are <file_idx>_<start_mol_idx>
            range_ = range(idx1, idx1 + len(fps))
            start_mol_idx = idx1

        brc_diameter.fit(fps, reinsert_indices=range_, n_features=self.n_features)
        # Extract the BitFeatures of the leaves to refine the largest cluster
        fps_bfs, mols_bfs = brc_diameter.bf_to_np_refine(fps, initial_mol=start_mol_idx)
        del fps
        del brc_diameter
        gc.collect()

        if self.double_cluster_init:
            # Passing the previous BitFeatures through the new tree, singleton clusters
            # are passed at the end
            brc_tolerance = BitBirch(
                branching_factor=self.branching_factor,
                threshold=self.threshold,
                merge_criterion="tolerance",
                tolerance=self.tolerance,
            )
            # For the "lean" variant, these are dicts
            iterable = zip(fps_bfs.values(), mols_bfs.values())
            for buffers, mol_idxs in iterable:
                brc_tolerance.fit_np(buffers, reinsert_index_sequences=mol_idxs)

            fps_bfs, mols_bfs = brc_tolerance.bf_to_np()
            del brc_tolerance
            gc.collect()

        for dtype, buf_list in fps_bfs.items():
            suffix = f"round_1_{idx0}_{dtype}"
            numpy_streaming_save(buf_list, self.out_dir / f"bufs_{suffix}")
            with open(self.out_dir / f"mol_idxs_{suffix}.pkl", mode="wb") as f:
                pickle.dump(mols_bfs[dtype], f)


class MidsectionRound:
    def __init__(
        self,
        branching_factor: int,
        threshold: float,
        tolerance: float,
        round_idx: int,
        out_dir: Path | str,
        use_mmap: bool = True,
    ) -> None:
        self.branching_factor = branching_factor
        self.threshold = threshold
        self.tolerance = tolerance
        self.round_idx = round_idx
        self.out_dir = Path(out_dir)
        self.use_mmap = use_mmap

    def __call__(self, chunk_info: tuple[int, tp.Sequence[Path]]) -> None:
        chunk_idx, chunk_filenames = chunk_info

        brc_chunk = BitBirch(
            branching_factor=self.branching_factor,
            threshold=self.threshold,
            merge_criterion="tolerance",
            tolerance=self.tolerance,
        )
        for fp_filename in chunk_filenames:
            buffers, mol_idxs = _load_buffers_and_mol_idxs(
                self.out_dir, fp_filename, self.round_idx - 1, self.use_mmap
            )
            brc_chunk.fit_np(buffers, reinsert_index_sequences=mol_idxs)
            del mol_idxs
            del buffers
            gc.collect()

        fps_bfs, mols_bfs = brc_chunk.bf_to_np()
        del brc_chunk
        gc.collect()

        for dtype, buf_list in fps_bfs.items():
            suffix = f"round_{self.round_idx}_{chunk_idx}_{dtype}"
            numpy_streaming_save(buf_list, self.out_dir / f"bufs_{suffix}")
            with open(self.out_dir / f"mol_idxs_{suffix}.pkl", mode="wb") as f:
                pickle.dump(mols_bfs[dtype], f)


def final_round(
    final_files: tp.Iterable[Path],
    branching_factor: int,
    threshold: float,
    tolerance: float,
    out_dir: Path | str,
    use_mmap: bool = True,
) -> None:
    out_dir = Path(out_dir)
    brc_final = BitBirch(
        branching_factor=branching_factor,
        threshold=threshold,
        merge_criterion="tolerance",
        tolerance=tolerance,
    )
    for fp_filename in final_files:
        buffers, mol_idxs = _load_buffers_and_mol_idxs(
            out_dir, fp_filename, 2, use_mmap
        )
        brc_final.fit_np(buffers, reinsert_index_sequences=mol_idxs)
        del buffers
        del mol_idxs
        gc.collect()

    cluster_mol_ids = brc_final.get_cluster_mol_ids()
    del brc_final
    gc.collect()

    with open(out_dir / "clusters.pkl", mode="wb") as f:
        pickle.dump(cluster_mol_ids, f)


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
    n_features: int,
    out_dir: Path,
    filename_idxs_are_slices: bool = False,
    merge_criterion: str = DEFAULTS.merge_criterion,
    num_processes: int = 10,
    num_round2_processes: int = 3,
    branching_factor: int = 254,
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
    max_files: int | None = None,
    verbose: bool = False,
) -> dict[str, float]:
    # Returns timing and for the different rounds
    # TODO: Also return peak-rss
    console = get_console(silent=not verbose)

    # Common BitBIRCH params
    common_kwargs: dict[str, tp.Any] = dict(
        branching_factor=branching_factor,
        threshold=threshold,
        tolerance=tolerance,
        use_mmap=use_mmap,
        out_dir=out_dir,
    )

    timings_s: dict[str, float] = {}
    start_time = time.perf_counter()
    start_round1 = time.perf_counter()

    # Initial round
    round_idx = 1
    round_label = f"Round {round_idx}"
    console.print(f"(Initial) {round_label}: Cluster initial batch of fingerprints...")

    initial_round_fn = InitialRound(
        n_features=n_features,
        double_cluster_init=double_cluster_init,
        max_fps=max_fps,
        filename_idxs_are_slices=filename_idxs_are_slices,
        merge_criterion=merge_criterion,
        **common_kwargs,
    )
    num_ps = min(num_processes, len(input_files))
    console.print(f"    - Running on {len(input_files)} files with {num_ps} processes")
    if num_ps == 1:
        for file in input_files:
            initial_round_fn(file)
    else:
        with mp.Pool(processes=num_ps, maxtasksperchild=max_tasks_per_process) as pool:
            pool.map(initial_round_fn, input_files)
    timings_s[round_label] = time.perf_counter() - start_round1
    console.print(f"    - Time for {round_label}: {timings_s[round_label]:.4f} s")
    console.print_peak_mem(num_ps)

    if only_first_round:  # Early exit for debugging
        timings_s["total"] = time.perf_counter() - start_time
        return timings_s

    # Mid-section rounds (round-2 round-3 ...)
    for _ in range(num_midsection_rounds):
        round_idx += 1
        round_label = f"Round {round_idx}"
        start_round2 = time.perf_counter()
        console.print(f"(Midsection) {round_label}: Re-clustering in chunks...")
        sorted_files = _glob_and_sort_by_uint_bits(
            out_dir, f"*round_{round_idx - 1}*.npy"
        )
        console.print(f"    - Collected {len(sorted_files)} output files")
        chunks = list(enumerate(batched(sorted_files, n=bin_size)))
        console.print(f"    - Chunked files into {len(chunks)} chunks")

        round_fn = MidsectionRound(round_idx=round_idx, **common_kwargs)
        num_ps = min(num_round2_processes, len(chunks))
        console.print(f"    - Running on {len(chunks)} chunks with {num_ps} processes")
        if num_ps == 1:
            for chunk_info_part in chunks:
                round_fn(chunk_info_part)
        else:
            with mp.Pool(
                processes=num_ps, maxtasksperchild=max_tasks_per_process
            ) as pool:
                pool.map(round_fn, chunks)
        timings_s[round_label] = time.perf_counter() - start_round2
        console.print(f"    - Time for {round_label}: {timings_s[round_label]:.4f} s")
        console.print_peak_mem(num_ps)

    # Final round
    round_idx += 1
    start_final = time.perf_counter()
    console.print(f"(Final) Round {round_idx}: Final round of clustering...")
    sorted_files = _glob_and_sort_by_uint_bits(out_dir, f"*round_{round_idx - 1}*.npy")
    console.print(f"    - Collected {len(sorted_files)} output files")
    final_round(sorted_files, **common_kwargs)

    round_label = f"Round {round_idx}"
    timings_s[round_label] = time.perf_counter() - start_final
    console.print(f"    - Time for {round_label}: {timings_s[round_label]:.4f} s")
    console.print_peak_mem(num_ps)

    timings_s["total"] = time.perf_counter() - start_time
    console.print()
    console.print(f"Total time: {timings_s['total']:.4f} s")
    return timings_s
