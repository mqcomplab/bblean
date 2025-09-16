r"""Analysis of clustering results"""

from pathlib import Path
from collections import defaultdict
import dataclasses
import typing as tp

import pandas as pd
import numpy as np
from numpy.typing import NDArray
from rdkit.Chem.Scaffolds import MurckoScaffold

from bblean.similarity import jt_isim
from bblean.fingerprints import fps_from_smiles, unpack_fingerprints

__all__ = [
    "scaffold_analysis",
    "cluster_analysis",
    "ScaffoldAnalysis",
    "ClusterAnalysis",
]


@dataclasses.dataclass
class ScaffoldAnalysis:
    r""":meta private:"""

    unique_num: int
    isim: float


@dataclasses.dataclass
class ClusterAnalysis:
    r""":meta private:"""

    df: pd.DataFrame
    fps: list[NDArray[np.uint8]]
    fps_are_packed: bool = True
    n_features: int | None = None

    @property
    def num_clusters(self) -> int:
        return len(self.df)

    def dump_metrics(self, path: Path) -> None:
        self.df.to_csv(path, index=False)


# Get the number of unique scaffolds and the scaffold isim
def scaffold_analysis(
    smiles: tp.Iterable[str], fp_kind: str = "rdkit"
) -> ScaffoldAnalysis:
    r"""Perform a scaffold analysis of a sequence of smiles"""
    if isinstance(smiles, str):
        smiles = [smiles]
    smiles = np.asarray(smiles)
    scaffolds = [MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smi) for smi in smiles]
    unique_scaffolds = set(scaffolds)
    unique_num = len(unique_scaffolds)
    scaffolds_fps = fps_from_smiles(unique_scaffolds, kind=fp_kind, pack=False)
    scaffolds_isim = jt_isim(np.sum(scaffolds_fps, axis=0), unique_num)
    return ScaffoldAnalysis(unique_num, scaffolds_isim)


def cluster_analysis(
    clusters: list[list[int]],
    smiles: tp.Iterable[str],
    fps: NDArray[np.integer],
    n_features: int | None = None,
    top: int = 20,
    assume_sorted: bool = True,
    scaffold_fp_kind: str = "rdkit",
    input_is_packed: bool = True,
) -> ClusterAnalysis:
    r"""Perform a cluster analysis starting from clusters, smiles, and fingerprints"""
    if isinstance(smiles, str):
        smiles = [smiles]
    smiles = np.asarray(smiles)
    fps_u8 = fps.astype(np.uint8, copy=False)

    if not assume_sorted:
        # Largest first
        clusters = sorted(clusters, key=lambda x: len(x), reverse=True)
    clusters = clusters[:top]

    info: dict[str, list[tp.Any]] = defaultdict(list)
    cluster_fps: list[NDArray[np.uint8]] = []
    for i, c in enumerate(clusters):
        analysis = scaffold_analysis(smiles[c], fp_kind=scaffold_fp_kind)
        size = len(c)
        _fps = fps_u8[c]
        if input_is_packed:
            _fps_unpacked = unpack_fingerprints(_fps, n_features=n_features)
        else:
            _fps_unpacked = _fps.copy()
        info["label"].append(i)
        info["mol_num"].append(size)
        info["isim"].append(jt_isim(np.sum(_fps_unpacked, axis=0), size))
        info["unique_scaffolds_num"].append(analysis.unique_num)
        info["unique_scaffolds_isim"].append(analysis.isim)
        cluster_fps.append(_fps)  # Lets see if something uses this
    return ClusterAnalysis(
        pd.DataFrame(info),
        cluster_fps,
        fps_are_packed=input_is_packed,
        n_features=n_features,
    )
