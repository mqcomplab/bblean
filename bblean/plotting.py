r"""Plotting and visualization convenience functions"""

from numpy.typing import NDArray
import typing as tp

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from bblean.utils import batched
from bblean.analysis import ClusterAnalysis
from bblean.fingerprints import unpack_fingerprints

__all__ = ["summary_plot", "tsne_plot", "dump_mol_images"]

# TODO: Mol relocation plots?


# Similar to "init_plot" in the original bitbirch
def summary_plot(
    c: ClusterAnalysis,
    /,
    title: str | None = None,
    annotate: bool = True,
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    r"""Obtain a summary plot from a cluster analysis"""
    orange = "tab:orange"
    blue = "tab:blue"
    df = c.df
    num_clusters = c.num_clusters
    if mpl.rcParamsDefault["font.size"] == plt.rcParams["font.size"]:
        plt.rcParams["font.size"] = 8
    if annotate:
        fig, ax = plt.subplots(figsize=(5, 2.5), dpi=250, constrained_layout=True)
    else:
        fig, ax = plt.subplots()

    # Plot and annotate the number of molecules
    label_strs = df["label"].astype(str)  # TODO: Is this necessary?
    ax.bar(
        label_strs,
        df["mol_num"],
        color=blue,
        label="Num. molecules",
        zorder=0,
    )
    if annotate:
        for i, mol in enumerate(df["mol_num"]):
            plt.text(
                i,
                mol,
                f"{mol}",
                ha="center",
                va="bottom",
                color="black",
                fontsize=5,
            )

    if c.has_scaffolds:
        # Plot and annotate the number of unique scaffolds
        plt.bar(
            label_strs,
            df["unique_scaffolds_num"],
            color=orange,
            label="Num. unique scaffolds",
            zorder=1,
        )
        if annotate:
            for i, s in enumerate(df["unique_scaffolds_num"]):
                plt.text(
                    i,
                    s,
                    f"{s}",
                    ha="center",
                    va="bottom",
                    color="white",
                    fontsize=5,
                )

    # Labels
    ax.set_xlabel("Cluster label")
    ax.set_ylabel("Num. molecules")
    ax.set_xticks(range(num_clusters))

    # Plot iSIM
    ax_isim = ax.twinx()
    ax_isim.plot(
        df["label"],
        df["isim"],
        color="tab:green",
        linestyle="dashed",
        linewidth=1.5,
        zorder=5,
        alpha=0.6,
    )
    ax_isim.scatter(
        df["label"],
        df["isim"],
        color="tab:green",
        marker="o",
        s=15,
        label="Tanimoto iSIM",
        edgecolor="darkgreen",
        zorder=100,
        alpha=0.6,
    )
    ax_isim.set_ylabel("Tanimoto iSIM (average similarity)")
    ax_isim.set_yticks(np.arange(0, 1.1, 0.1))
    ax_isim.set_ylim(0, 1)
    ax_isim.spines["right"].set_color("tab:green")
    ax_isim.tick_params(colors="tab:green")
    ax_isim.yaxis.label.set_color("tab:green")
    bbox = ax.get_position()
    fig.legend(
        loc="center right",
        bbox_to_anchor=(bbox.x0 + 0.95 * bbox.width, bbox.y0 + 0.5 * bbox.height),
    )
    msg = f"Metrics for top {num_clusters} largest clusters"
    if title is not None:
        msg = f"{msg} for {title}"
    fig.suptitle(msg)
    return fig, (ax, ax_isim)


def tsne_plot(
    c: ClusterAnalysis, /, title: str | None = None
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    r"""Obtain a t-SNE plot from a cluster analysis"""
    df = c.df
    fps_list: list[NDArray[np.uint8]] = []
    label_list: list[int] = []
    cluster_fps: list[NDArray[np.uint8]]
    for fps, label in zip(c.fps, df["label"]):
        if c.fps_are_packed:
            # fps is 2D so this is guaranteed to be a list of u8 arrays
            # Too complicated for mypy to parse, but correct
            cluster_fps = [
                fp for fp in unpack_fingerprints(fps, n_features=c.n_features)  # type: ignore # noqa:E501
            ]
        else:
            cluster_fps = list(fps)
        fps_list.extend(cluster_fps)
        label_list.extend([label] * len(cluster_fps))
    num_clusters = c.num_clusters
    scaler = StandardScaler()
    fps_scaled = scaler.fit_transform(fps_list)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    fps_tsne = tsne.fit_transform(fps_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        fps_tsne[:, 0], fps_tsne[:, 1], c=label_list, cmap="tab20", alpha=0.9
    )
    cbar = plt.colorbar(scatter, label="Cluster ID")
    cbar.set_ticks(list(range(num_clusters)))
    cbar.set_ticklabels(list(map(str, range(1, num_clusters + 1))))
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    if title is not None:
        fig.suptitle(f"t-SNE of Top {num_clusters} Largest Clusters for {title}")
    return fig, (ax,)


def dump_mol_images(
    smiles: tp.Iterable[str],
    clusters: list[list[int]],
    cluster_idx: int = 0,
    batch_size: int = 30,
) -> None:
    r"""Dump smiles associated with a specific cluster as *.png image files"""
    if isinstance(smiles, str):
        smiles = [smiles]
    smiles = np.asarray(smiles)
    idxs = clusters[cluster_idx]
    for i, idx_seq in enumerate(batched(idxs, batch_size)):
        mols = []
        for smi in smiles[list(idx_seq)]:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"Could not parse smiles {smi}")
            mols.append(mol)
        img = Draw.MolsToGridImage(mols, molsPerRow=5)
        with open(f"cluster_{cluster_idx}_{i}.png", "wb") as f:
            f.write(img.data)
