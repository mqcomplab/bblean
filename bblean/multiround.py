# BitBirch-Lean Python Package: An open-source clustering module based on iSIM.
#
# If you find this code useful, please cite the BitBirch paper:
# https://doi.org/10.1039/D5DD00030K
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

from bblean._console import get_console
from bblean.utils import numpy_streaming_save, batched, _import_bitbirch_variant
from bblean.config import DEFAULTS


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
    idxs_file = out_dir / f"mol_idxs_round{round}_{count}_{dtype}.pkl"
    with open(idxs_file, "rb") as f:
        mol_idxs = pickle.load(f)
    return fp_data, mol_idxs


def first_round(
    fp_file: Path,
    n_features: int,
    double_cluster_init: bool,
    branching_factor: int,
    threshold: float,
    tolerance: float,
    out_dir: Path | str,
    filename_idxs_are_slices: bool = False,
    max_fps: int | None = None,
    use_mmap: bool = True,
    bitbirch_variant: str = "lean",
    merge_criterion: str = "diameter",
) -> None:

    BitBirch, set_merge = _import_bitbirch_variant(bitbirch_variant)
    out_dir = Path(out_dir)
    fps = np.load(fp_file, mmap_mode="r" if use_mmap else None)[:max_fps]

    # Fit the fps. `reinsert_indices` is necessary to keep track of proper molecule
    # indices. Use indices of molecules in the current batch, according to the total set
    idx0, idx1 = map(int, fp_file.name.split(".")[0].split("_")[-2:])
    set_merge(merge_criterion)  # Initial batch uses diameter BitBIRCH
    brc_diameter = BitBirch(branching_factor=branching_factor, threshold=threshold)
    # TODO: Check this
    if filename_idxs_are_slices:
        # idxs are <start_mol_idx>_<end_mol_idx>
        range_ = range(idx0, idx1)
        start_mol_idx = idx0
    else:
        # idxs are <file_idx>_<start_mol_idx>
        range_ = range(idx1, idx1 + len(fps))
        start_mol_idx = idx1

    brc_diameter.fit(fps, reinsert_indices=range_, n_features=n_features)

    # Extract the BitFeatures info of the leaves to refine the top cluster
    fps_bfs, mols_bfs = brc_diameter.bf_to_np_refine(fps, initial_mol=start_mol_idx)
    del fps
    del brc_diameter
    gc.collect()

    if double_cluster_init:
        # Passing the previous BitFeatures through the new tree, singleton clusters are
        # passed at the end
        set_merge("tolerance", tolerance)  # 'tolerance' used to refine the tree
        brc_tolerance = BitBirch(branching_factor=branching_factor, threshold=threshold)
        if bitbirch_variant == "lean":
            # For the "lean" variant, these are dicts
            iterable = zip(fps_bfs.values(), mols_bfs.values())
        else:
            iterable = zip(fps_bfs, mols_bfs)

        for buffers, mol_idxs in iterable:
            brc_tolerance.fit_np(buffers, reinsert_index_sequences=mol_idxs)

        fps_bfs, mols_bfs = brc_tolerance.bf_to_np()
        del brc_tolerance
        gc.collect()

    if bitbirch_variant in ["lean", "lean_dense"]:
        # In this case the fps_bfs and mols_idxs are *dicts*
        for dtype, buf_list in fps_bfs.items():
            suffix = f"round1_{idx0}_{dtype}"
            numpy_streaming_save(buf_list, out_dir / f"bufs_{suffix}")
            with open(out_dir / f"mol_idxs_{suffix}.pkl", mode="wb") as f:
                pickle.dump(mols_bfs[dtype], f)
    else:
        # In this case fps_bfs is *list of arrays*, and mols_bfs is *list of lists*
        for buf, mol_idxs in zip(fps_bfs, mols_bfs):
            suffix = f"round1_{idx0}_{str(buf.dtype)}"
            np.save(out_dir / f"bufs_{suffix}", buf)
            with open(out_dir / f"mol_idxs_{suffix}.pkl", mode="wb") as f:
                pickle.dump(mol_idxs, f)


def second_round(
    chunk_info: tuple[int, tp.Sequence[Path]],
    branching_factor: int,
    threshold: float,
    tolerance: float,
    out_dir: Path | str,
    use_mmap: bool = True,
    bitbirch_variant: str = "lean",
) -> None:
    BitBirch, set_merge = _import_bitbirch_variant(bitbirch_variant)
    out_dir = Path(out_dir)
    chunk_idx, chunk_filenames = chunk_info

    set_merge("tolerance", tolerance)
    brc_chunk = BitBirch(branching_factor=branching_factor, threshold=threshold)
    for fp_filename in chunk_filenames:
        buffers, mol_idxs = _load_buffers_and_mol_idxs(
            out_dir, fp_filename, 1, use_mmap
        )
        brc_chunk.fit_np(buffers, reinsert_index_sequences=mol_idxs)
        del mol_idxs
        del buffers
        gc.collect()

    fps_bfs, mols_bfs = brc_chunk.bf_to_np()
    del brc_chunk
    gc.collect()

    if bitbirch_variant in ["lean", "lean_dense"]:
        # In this case the fps_bfs and mols_idxs are *dicts*
        for dtype, buf_list in fps_bfs.items():
            suffix = f"round2_{chunk_idx}_{dtype}"
            numpy_streaming_save(buf_list, out_dir / f"bufs_{suffix}")
            with open(out_dir / f"mol_idxs_{suffix}.pkl", mode="wb") as f:
                pickle.dump(mols_bfs[dtype], f)
    else:
        # In this case fps_bfs is *list of arrays*, and mols_bfs is *list of lists*
        for buf, mol_idxs in zip(fps_bfs, mols_bfs):
            suffix = f"round2_{chunk_idx}_{str(buf.dtype)}"
            np.save(out_dir / f"bufs_{suffix}", buf)
            with open(out_dir / f"mol_idxs_{suffix}.pkl", mode="wb") as f:
                pickle.dump(mol_idxs, f)


def third_round(
    branching_factor: int,
    threshold: float,
    tolerance: float,
    out_dir: Path | str,
    use_mmap: bool = True,
    bitbirch_variant: str = "lean",
) -> None:

    BitBirch, set_merge = _import_bitbirch_variant(bitbirch_variant)
    out_dir = Path(out_dir)

    set_merge("tolerance", tolerance)
    brc_final = BitBirch(branching_factor=branching_factor, threshold=threshold)

    sorted_files2 = _glob_and_sort_by_uint_bits(out_dir, "*round2*.npy")
    for fp_filename in sorted_files2:
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
        use_mmap=use_mmap,
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
        n_features=n_features,
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
