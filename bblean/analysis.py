r"""Analysis of clustering results"""

from pathlib import Path
from collections import defaultdict
import dataclasses
import typing as tp
from functools import cached_property

import pandas as pd
import numpy as np
from numpy.typing import NDArray
from rdkit.Chem.Scaffolds import MurckoScaffold

from bblean._config import DEFAULTS
from bblean.similarity import jt_isim
from bblean.fingerprints import (
    fps_from_smiles,
    unpack_fingerprints,
    pack_fingerprints,
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


class ClusterAnalysis:
    r""":meta private:"""

    def __init__(
        self,
        selected_cluster_sizes: list[int],
        all_cluster_sizes: list[int],
        df: pd.DataFrame,
        total_fps_num: int,
        selected_fps: NDArray[np.uint8] | None = None,
        fps_are_packed: bool = True,
        n_features: int | None = None,
        min_size: int | None = None,
    ) -> None:
        self.total_fps = total_fps_num
        self.stats = pd.Series(all_cluster_sizes).describe()
        self._all_cluster_sizes = all_cluster_sizes
        self._selected_cluster_sizes = selected_cluster_sizes
        self._fps = selected_fps
        self._df = df
        self.fps_are_packed = fps_are_packed
        self.n_features = n_features
        self.min_size = min_size

    def all_clusters_num_with_size_above(self, size: int) -> int:
        return sum(1 for c in self._all_cluster_sizes if c > size)

    @cached_property
    def all_singletons_num(self) -> int:
        return sum(1 for c in self._all_cluster_sizes if c == 1)

    def get_top_cluster_fps(self, packed: bool = True) -> list[NDArray[np.uint8]]:
        if self._fps is None:
            raise RuntimeError("Fingerprints not present")
        fps = self.top_packed_fps if packed else self.top_unpacked_fps
        out = []
        offset = 0
        for s in self._selected_cluster_sizes:
            out.append(fps[offset : offset + s])
            offset += s
        return out

    @property
    def all_clusters_mean_size(self) -> float:
        return float(self.stats["mean"])

    @property
    def all_clusters_median_size(self) -> int:
        return int(self.stats["50%"])

    @property
    def all_clusters_q1(self) -> int:
        return int(self.stats["25%"])

    @property
    def all_clusters_q3(self) -> int:
        return int(self.stats["75%"])

    @property
    def all_clusters_min_size(self) -> int:
        return int(self.stats["min"])

    @property
    def all_clusters_max_size(self) -> int:
        return int(self.stats["max"])

    @property
    def all_clusters_num(self) -> int:
        return int(self.stats["count"])

    @property
    def top_unpacked_fps(self) -> NDArray[np.uint8]:
        if self._fps is None:
            raise RuntimeError("Fingerprints not present")
        if self.fps_are_packed:
            return unpack_fingerprints(self._fps, self.n_features)
        return self._fps

    @property
    def top_packed_fps(self) -> NDArray[np.uint8]:
        if self._fps is None:
            raise RuntimeError("Fingerprints not present")
        if self.fps_are_packed:
            return self._fps
        return pack_fingerprints(self._fps)

    @property
    def has_scaffolds(self) -> bool:
        return "unique_scaffolds_num" in self._df.columns

    @property
    def has_fps(self) -> bool:
        return self._fps is not None

    @property
    def has_all_clusters(self) -> bool:
        return self.clusters_num == self.all_clusters_num

    @property
    def clusters_num(self) -> int:
        return len(self._df)

    @property
    def isims(self) -> pd.Series:
        return self._df["isim"]

    @property
    def labels(self) -> pd.Series:
        return self._df["labels"]

    @property
    def sizes(self) -> pd.Series:
        return self._df["sizes"]

    @property
    def unique_scaffolds_num(self) -> pd.Series:
        return self._df["unique_scaffolds_num"]

    @property
    def unique_scaffolds_isim(self) -> pd.Series:
        return self._df["unique_scaffolds_isim"]

    def dump_metrics(self, path: Path) -> None:
        self._df.to_csv(path, index=False)


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
    scaffolds_isim = jt_isim(scaffolds_fps, input_is_packed=False)
    return ScaffoldAnalysis(len(unique_scaffolds), scaffolds_isim)


def cluster_analysis(
    clusters: list[list[int]],
    fps: NDArray[np.integer] | Path | tp.Sequence[Path] | None = None,
    smiles: tp.Iterable[str] = (),
    n_features: int | None = None,
    top: int | None = 20,
    assume_sorted: bool = True,
    scaffold_fp_kind: str = DEFAULTS.fp_kind,
    input_is_packed: bool = True,
    min_size: int = 0,
) -> ClusterAnalysis:
    r"""Perform a cluster analysis starting from clusters, smiles, and fingerprints"""
    if isinstance(smiles, str):
        smiles = [smiles]
    smiles = np.asarray(smiles)

    if not assume_sorted:
        # Largest first
        clusters = sorted(clusters, key=lambda x: len(x), reverse=True)
    all_cluster_sizes = [len(c) for c in clusters]
    total_fps = sum(all_cluster_sizes)
    # Filter by min size
    _clusters = []
    for i, c in enumerate(clusters):
        if all_cluster_sizes[i] < min_size:
            break
        if top is not None and i >= top:
            break
        _clusters.append(c)
    clusters = _clusters

    info: dict[str, list[tp.Any]] = defaultdict(list)
    fps_provider: tp.Union[_FingerprintFileSequence, NDArray[np.uint8], None]
    if fps is None:
        fps_provider = None
    elif isinstance(fps, Path):
        fps_provider = np.load(fps, mmap_mode="r")
    elif not isinstance(fps, np.ndarray):
        fps_provider = _FingerprintFileSequence(fps)
    else:
        fps_provider = tp.cast(NDArray[np.uint8], fps.astype(np.uint8, copy=False))

    if fps_provider is None:
        selected = None
    else:
        selected = np.empty(
            (sum(len(c) for c in clusters), fps_provider.shape[1]), dtype=np.uint8
        )
    start = 0
    for i, c in enumerate(clusters, 1):
        size = len(c)
        # If a file sequence is passed, the cluster indices must be sorted.
        # the cluster analysis is idx-order-independent, so this is fine
        info["labels"].append(i)
        info["sizes"].append(size)
        if smiles.size:
            analysis = scaffold_analysis(smiles[c], fp_kind=scaffold_fp_kind)
            info["unique_scaffolds_num"].append(analysis.unique_num)
            info["unique_scaffolds_isim"].append(analysis.isim)
        if fps_provider is not None:
            assert selected is not None
            _fps = fps_provider[sorted(c)]
            info["isim"].append(
                jt_isim(_fps, input_is_packed=input_is_packed, n_features=n_features)
            )
            selected[start : start + size] = _fps
        start += size
    return ClusterAnalysis(
        [len(c) for c in clusters],
        all_cluster_sizes,
        pd.DataFrame(info),
        selected_fps=selected,
        total_fps_num=total_fps,
        fps_are_packed=input_is_packed,
        n_features=n_features,
        min_size=min_size,
    )
