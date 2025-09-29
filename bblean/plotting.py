r"""Plotting and visualization convenience functions"""

import typing as tp

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from rdkit import Chem
from rdkit.Chem import Draw
import colorcet
from sklearn.preprocessing import StandardScaler, normalize as normalize_features
from sklearn.decomposition import PCA
from openTSNE.sklearn import TSNE
from openTSNE.affinity import Multiscale

from bblean.utils import batched, _num_avail_cpus
from bblean.analysis import ClusterAnalysis
from bblean._config import TSNE_SEED

__all__ = ["summary_plot", "tsne_plot", "dump_mol_images"]

# TODO: Mol relocation plots?


# Similar to "init_plot" in the original bitbirch
def summary_plot(
    c: ClusterAnalysis,
    /,
    title: str | None = None,
    annotate: bool = True,
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    r"""Create a summary plot from a cluster analysis.

    If the analysis contains scaffolds, a scaffold analysis is added to the plot"""
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
        df["label"] - 1,
        df["isim"],
        color="tab:green",
        linestyle="dashed",
        linewidth=1.5,
        zorder=5,
        alpha=0.6,
    )
    ax_isim.scatter(
        df["label"] - 1,
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
    c: ClusterAnalysis,
    /,
    title: str | None = None,
    seed: int | None = TSNE_SEED,
    perplexity: int = 30,
    workers: int | None = None,
    scaling: str = "normalize",
    exaggeration: float | None = None,
    do_pca_init: bool = True,
    multiscale: bool = False,
    pca_reduce: int | None = None,
    dof: float = 1.0,
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    r"""Create a t-SNE plot from a cluster analysis"""
    if workers is None:
        workers = _num_avail_cpus()
    df = c.df
    color_labels: list[int] = []
    for num, label in zip(df["mol_num"], df["label"]):
        color_labels.extend([label - 1] * num)  # color labels start with 0
    num_clusters = c.num_clusters

    # I don't think these should be transformed, like this, only normalized
    if scaling == "normalize":
        fps_scaled = normalize_features(c.unpacked_fps)
    elif scaling == "std":
        scaler = StandardScaler()
        fps_scaled = scaler.fit_transform(c.unpacked_fps)
    elif scaling == "none":
        fps_scaled = c.unpacked_fps
    else:
        raise ValueError(f"Unknown scaling {scaling}")
    if pca_reduce is not None:
        fps_scaled = PCA(n_components=pca_reduce).fit_transform(fps_scaled)

    # Learning rate is set to N / exaggeration (good default)
    # Early exaggeration defaults to max(12, exaggeration) (good default)
    # exaggeration_iter = 250, normal_iter = 500 (good defaults)
    # "pca" is the method used by Dimitry Kovak et. al. (good default), with some jitter
    # added for extra numerical stability
    # Multiscale may help with medium-sized datasets together with downsampling, but
    # it doesn't do much in my tests.
    # NOTE: Dimensionality reduction with PCA to ~50 features seems to mostly preserve
    # cluster structure
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        n_jobs=workers,
        dof=dof,
        exaggeration=exaggeration,  # second-phase exaggeration
        negative_gradient_method="fft",  # faster for large datasets
        initialization="pca" if do_pca_init else "random",
    )
    if multiscale:
        fps_tsne = (
            super(TSNE, tsne)
            .fit(
                fps_scaled,
                affinities=Multiscale(
                    n_jobs=workers,
                    random_state=seed,
                    data=fps_scaled,
                    perplexities=[perplexity, len(fps_scaled) / 100],
                ),
                initialization="pca" if do_pca_init else "random",
            )
            .view(np.ndarray)
        )
    else:
        fps_tsne = tsne.fit_transform(fps_scaled)

    fig, ax = plt.subplots(dpi=250, figsize=(4, 3.5))
    scatter = ax.scatter(
        fps_tsne[:, 0],
        fps_tsne[:, 1],
        c=color_labels,
        cmap=mpl.colors.ListedColormap(colorcet.glasbey_bw_minc_20[:num_clusters]),
        edgecolors="none",
        alpha=0.5,
        s=2,
    )
    # t-SNE plots *must be square*
    ax.set_aspect("equal", adjustable="box")
    cbar = plt.colorbar(scatter, label="Cluster label")
    cbar.set_ticks(list(range(num_clusters)))
    cbar.set_ticklabels(list(map(str, range(1, num_clusters + 1))))
    ax.set_xlabel("t-SNE component 1")
    ax.set_ylabel("t-SNE component 2")
    msg = f"t-SNE of top {num_clusters} largest clusters"
    if title is not None:
        msg = f"{msg} for {title}"
    fig.suptitle(msg)
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
