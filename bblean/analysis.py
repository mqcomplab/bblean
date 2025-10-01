r"""Analysis of clustering results"""

from pathlib import Path
from collections import defaultdict
import dataclasses
import typing as tp

import pandas as pd
import numpy as np
from numpy.typing import NDArray
from rdkit.Chem.Scaffolds import MurckoScaffold

from bblean._config import DEFAULTS
from bblean.similarity import jt_isim_packed, jt_isim_unpacked
from bblean.fingerprints import (
    fps_from_smiles,
    unpack_fingerprints,
    _FingerprintFileSequence,
)

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

    clusters: list[list[int]]
    df: pd.DataFrame
    fps: NDArray[np.uint8]
    fps_are_packed: bool = True
    n_features: int | None = None

    @property
    def unpacked_fps(self) -> NDArray[np.uint8]:
        return unpack_fingerprints(self.fps, self.n_features)

    @property
    def has_scaffolds(self) -> bool:
        return "unique_scaffolds_num" in self.df.columns

    @property
    def num_clusters(self) -> int:
        return len(self.df)

    def dump_metrics(self, path: Path) -> None:
        self.df.to_csv(path, index=False)


# Get the number of unique scaffolds and the scaffold isim
def scaffold_analysis(
    smiles: tp.Iterable[str], fp_kind: str = DEFAULTS.fp_kind
) -> ScaffoldAnalysis:
    r"""Perform a scaffold analysis of a sequence of smiles

    Note that the order of the input smiles is not relevant
    """
    if isinstance(smiles, str):
        smiles = [smiles]
    scaffolds = [MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smi) for smi in smiles]
    unique_scaffolds = set(scaffolds)
    scaffolds_fps = fps_from_smiles(unique_scaffolds, kind=fp_kind, pack=False)
    scaffolds_isim = jt_isim_unpacked(scaffolds_fps)
    return ScaffoldAnalysis(len(unique_scaffolds), scaffolds_isim)


def cluster_analysis(
    clusters: list[list[int]],
    fps: NDArray[np.integer] | Path | tp.Sequence[Path],
    smiles: tp.Iterable[str] = (),
    n_features: int | None = None,
    top: int = 20,
    assume_sorted: bool = True,
    scaffold_fp_kind: str = DEFAULTS.fp_kind,
    input_is_packed: bool = True,
) -> ClusterAnalysis:
    r"""Perform a cluster analysis starting from clusters, smiles, and fingerprints"""
    if isinstance(smiles, str):
        smiles = [smiles]
    smiles = np.asarray(smiles)

    if not assume_sorted:
        # Largest first
        clusters = sorted(clusters, key=lambda x: len(x), reverse=True)
    clusters = clusters[:top]

    info: dict[str, list[tp.Any]] = defaultdict(list)
    fps_provider: tp.Union[_FingerprintFileSequence, NDArray[np.uint8]]
    if isinstance(fps, Path):
        fps_provider = np.load(fps, mmap_mode="r")
    elif not isinstance(fps, np.ndarray):
        fps_provider = _FingerprintFileSequence(fps)
    else:
        fps_provider = tp.cast(NDArray[np.uint8], fps.astype(np.uint8, copy=False))
    selected = np.empty(
        (sum(len(c) for c in clusters), fps_provider.shape[1]), dtype=np.uint8
    )
    start = 0
    for i, c in enumerate(clusters, 1):
        size = len(c)
        # If a file sequence is passed, the cluster indices must be sorted.
        # the cluster analysis is idx-order-independent, so this is fine
        _fps = fps_provider[sorted(c)]
        info["label"].append(i)
        info["mol_num"].append(size)
        if input_is_packed:
            info["isim"].append(jt_isim_packed(_fps, n_features))  # type: ignore
        else:
            info["isim"].append(jt_isim_unpacked(_fps))  # type: ignore
        if smiles.size:
            analysis = scaffold_analysis(smiles[c], fp_kind=scaffold_fp_kind)
            info["unique_scaffolds_num"].append(analysis.unique_num)
            info["unique_scaffolds_isim"].append(analysis.isim)
        selected[start : start + size] = _fps
        start += size
    return ClusterAnalysis(
        clusters,
        pd.DataFrame(info),
        selected,
        fps_are_packed=input_is_packed,
        n_features=n_features,
    )
