r"""Plotting and visualization convenience functions"""

import warnings
from pathlib import Path
import pickle
import random
import typing as tp

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw
import colorcet
from sklearn.preprocessing import StandardScaler, normalize as normalize_features
from sklearn.decomposition import PCA
from openTSNE.sklearn import TSNE
from openTSNE.affinity import Multiscale
import umap

from bblean.utils import batched, _num_avail_cpus, _has_files_or_valid_symlinks
from bblean.analysis import ClusterAnalysis, cluster_analysis
from bblean._config import TSNE_SEED

__all__ = [
    "summary_plot",
    "tsne_plot",
    "umap_plot",
    "pops_plot",
    "pca_plot",
    "dump_mol_images",
]


def pops_plot(
    c: ClusterAnalysis,
    /,
    title: str | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    r"""Distrubution of cluster populations using KDE"""
    fig, ax = plt.subplots()
    cluster_sizes = c.sizes
    sns.kdeplot(
        ax=ax,
        data=cluster_sizes,
        color="tab:purple",
        bw_adjust=0.25,
        gridsize=len(cluster_sizes) // 5,
        fill=True,
        warn_singular=False,
    )
    ax.set_xlabel("Density")
    ax.set_xlabel("Cluster size")
    msg = f"Populations for top {c.clusters_num} largest clusters"
    if c.min_size is not None:
        msg = f"{msg} (min. size = {c.min_size})"
    if title is not None:
        msg = f"{msg} for {title}"
    fig.suptitle(msg)
    return fig, (ax,)


# Similar to "init_plot" in the original bitbirch
def summary_plot(
    c: ClusterAnalysis,
    /,
    title: str | None = None,
    counts_ylim: int | None = None,
    annotate: bool = True,
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    r"""Create a summary plot from a cluster analysis

    If the analysis contains scaffolds, a scaffold analysis is added to the plot"""
    orange = "tab:orange"
    blue = "tab:blue"
    if mpl.rcParamsDefault["font.size"] == plt.rcParams["font.size"]:
        plt.rcParams["font.size"] = 8
    if annotate:
        fig, ax = plt.subplots(figsize=(5, 2.5), dpi=250, constrained_layout=True)
    else:
        fig, ax = plt.subplots()

    # Plot and annotate the number of molecules
    label_strs = c.labels.astype(str)  # TODO: Is this necessary?
    ax.bar(
        label_strs,
        c.sizes,
        color=blue,
        label="Num. molecules",
        zorder=0,
    )
    ax.set_ylim(0, counts_ylim)
    if annotate:
        for i, mol in enumerate(c.sizes):
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
            c.unique_scaffolds_num,
            color=orange,
            label="Num. unique scaffolds",
            zorder=1,
        )
        if annotate:
            for i, s in enumerate(c.unique_scaffolds_num):
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
    ax.set_xticks(range(c.clusters_num))

    # Plot iSIM
    if c.has_fps:
        ax_isim = ax.twinx()
        ax_isim.plot(
            c.labels - 1,
            c.isims,
            color="tab:green",
            linestyle="dashed",
            linewidth=1.5,
            zorder=5,
            alpha=0.6,
        )
        ax_isim.scatter(
            c.labels - 1,
            c.isims,
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
        loc="upper right",
        bbox_to_anchor=(bbox.x0 + 0.95 * bbox.width, bbox.y0 + 0.95 * bbox.height),
    )
    if c.has_all_clusters:
        msg = "Metrics of all clusters"
    else:
        msg = f"Metrics of top {c.clusters_num} largest clusters"
    if title is not None:
        msg = f"{msg} for {title}"
    fig.suptitle(msg)
    if not c.has_fps:
        return fig, (ax,)
    return fig, (ax, ax_isim)


def umap_plot(
    c: ClusterAnalysis,
    /,
    title: str | None = None,
    scaling: str = "normalize",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    densmap: bool = False,
    workers: int | None = None,
    deterministic: bool = False,
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    r"""Create a UMAP plot from a cluster analysis"""
    color_labels: list[int] = []
    for num, label in zip(c.sizes, c.labels):
        color_labels.extend([label - 1] * num)  # color labels start with 0
    num_top = c.clusters_num
    if workers is None:
        workers = _num_avail_cpus()

    # I don't think these should be transformed, like this, only normalized
    if scaling == "normalize":
        fps_scaled = normalize_features(c.top_unpacked_fps)
    elif scaling == "std":
        scaler = StandardScaler()
        fps_scaled = scaler.fit_transform(c.top_unpacked_fps)
    elif scaling == "none":
        fps_scaled = c.top_unpacked_fps
    else:
        raise ValueError(f"Unknown scaling {scaling}")
    fps_umap = umap.UMAP(
        densmap=densmap,
        random_state=42 if deterministic else None,
        n_components=2,
        n_jobs=workers,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
    ).fit_transform(fps_scaled)
    fig, ax = plt.subplots(dpi=250, figsize=(4, 3.5))
    scatter = ax.scatter(
        fps_umap[:, 0],
        fps_umap[:, 1],
        c=color_labels,
        cmap=mpl.colors.ListedColormap(colorcet.glasbey_bw_minc_20[:num_top]),
        edgecolors="none",
        alpha=0.5,
        s=2,
    )
    # t-SNE plots *must be square*
    ax.set_aspect("equal", adjustable="box")
    cbar = plt.colorbar(scatter, label="Cluster label")
    cbar.set_ticks(list(range(num_top)))
    cbar.set_ticklabels(list(map(str, range(1, num_top + 1))))
    ax.set_xlabel("UMAP component 1")
    ax.set_ylabel("UMAP component 2")
    if c.has_all_clusters:
        msg = "UMAP of all clusters"
    else:
        msg = f"UMAP of top {num_top} largest clusters"
    if title is not None:
        msg = f"{msg} for {title}"
    fig.suptitle(msg)
    return fig, (ax,)


def pca_plot(
    c: ClusterAnalysis,
    /,
    title: str | None = None,
    scaling: str = "normalize",
    whiten: bool = False,
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    r"""Create a t-SNE plot from a cluster analysis"""
    color_labels: list[int] = []
    for num, label in zip(c.sizes, c.labels):
        color_labels.extend([label - 1] * num)  # color labels start with 0
    num_top = c.clusters_num

    # I don't think these should be transformed, like this, only normalized
    if scaling == "normalize":
        fps_scaled = normalize_features(c.top_unpacked_fps)
    elif scaling == "std":
        scaler = StandardScaler()
        fps_scaled = scaler.fit_transform(c.top_unpacked_fps)
    elif scaling == "none":
        fps_scaled = c.top_unpacked_fps
    else:
        raise ValueError(f"Unknown scaling {scaling}")
    fps_pca = PCA(n_components=2, whiten=whiten, random_state=1234).fit_transform(
        fps_scaled
    )
    fig, ax = plt.subplots(dpi=250, figsize=(4, 3.5))
    scatter = ax.scatter(
        fps_pca[:, 0],
        fps_pca[:, 1],
        c=color_labels,
        cmap=mpl.colors.ListedColormap(colorcet.glasbey_bw_minc_20[:num_top]),
        edgecolors="none",
        alpha=0.5,
        s=2,
    )
    # t-SNE plots *must be square*
    ax.set_aspect("equal", adjustable="box")
    cbar = plt.colorbar(scatter, label="Cluster label")
    cbar.set_ticks(list(range(num_top)))
    cbar.set_ticklabels(list(map(str, range(1, num_top + 1))))
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    if c.has_all_clusters:
        msg = "PCA of all clusters"
    else:
        msg = f"PCA of top {num_top} largest clusters"
    if title is not None:
        msg = f"{msg} for {title}"
    fig.suptitle(msg)
    return fig, (ax,)


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
    metric: str = "euclidean",
    dof: float = 1.0,
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    r"""Create a t-SNE plot from a cluster analysis"""
    if workers is None:
        workers = _num_avail_cpus()
    color_labels: list[int] = []
    for num, label in zip(c.sizes, c.labels):
        color_labels.extend([label - 1] * num)  # color labels start with 0
    num_top = c.clusters_num

    # I don't think these should be transformed, like this, only normalized
    if scaling == "normalize":
        fps_scaled = normalize_features(c.top_unpacked_fps)
    elif scaling == "std":
        scaler = StandardScaler()
        fps_scaled = scaler.fit_transform(c.top_unpacked_fps)
    elif scaling == "none":
        fps_scaled = c.top_unpacked_fps
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
        cmap=mpl.colors.ListedColormap(colorcet.glasbey_bw_minc_20[:num_top]),
        edgecolors="none",
        alpha=0.5,
        s=2,
    )
    # t-SNE plots *must be square*
    ax.set_aspect("equal", adjustable="box")
    cbar = plt.colorbar(scatter, label="Cluster label")
    cbar.set_ticks(list(range(num_top)))
    cbar.set_ticklabels(list(map(str, range(1, num_top + 1))))
    ax.set_xlabel("t-SNE component 1")
    ax.set_ylabel("t-SNE component 2")
    if c.has_all_clusters:
        msg = "t-SNE of all clusters"
    else:
        msg = f"t-SNE of top {num_top} largest clusters"
    if title is not None:
        msg = f"{msg} for {title}"
    fig.suptitle(msg)
    return fig, (ax,)


def dump_mol_images(
    smiles: tp.Iterable[str],
    clusters: list[list[int]],
    cluster_idx: int = 0,
    batch_size: int = 30,
    limit: int = -1,
) -> None:
    r"""Dump smiles associated with a specific cluster as ``*.png`` image files"""
    if isinstance(smiles, str):
        smiles = [smiles]
    smiles = np.asarray(smiles)
    idxs = clusters[cluster_idx]
    num = 0
    for i, idx_seq in enumerate(batched(idxs, batch_size)):
        if num + len(idx_seq) > limit:
            idx_seq = idx_seq[: num + len(idx_seq) - limit]
        mols = []
        for smi in smiles[list(idx_seq)]:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"Could not parse smiles {smi}")
            mols.append(mol)
        img = Draw.MolsToGridImage(mols, molsPerRow=5)
        with open(f"cluster_{cluster_idx}_{i}.png", "wb") as f:
            f.write(img.data)
        num += len(idx_seq)
        if num >= limit:
            break


# For internal use, dispatches a visualization workflow and optionally saves
# plot to disk and/or displays it using mpl
def _dispatch_visualization(
    clusters_path: Path,
    fn_name: str,
    fn: tp.Callable[..., tp.Any],
    fn_kwargs: tp.Any,
    min_size: int = 0,
    smiles: tp.Iterable[str] = (),
    top: int | None = None,
    n_features: int | None = None,
    input_is_packed: bool = True,
    fps_path: Path | None = None,
    title: str | None = None,
    filename: str | None = None,
    verbose: bool = True,
    save: bool = True,
    show: bool = True,
) -> None:
    if clusters_path.is_dir():
        clusters_path = clusters_path / "clusters.pkl"
    with open(clusters_path, mode="rb") as f:
        clusters = pickle.load(f)
    if fps_path is None:
        input_fps_path = clusters_path.parent / "input-fps"
        if input_fps_path.is_dir() and _has_files_or_valid_symlinks(input_fps_path):
            fps_path = input_fps_path
        else:
            if fn_name != "summary":
                msg = "Could not find input fingerprints. Please use --fps-path"
                raise RuntimeError(msg)
            else:
                msg = (
                    "Could not find input fingerprints. Please use --fps-path."
                    " Summary plot without fingerprints doesn't include isim values"
                )
                warnings.warn(msg)
    if fps_path is None:
        fps_paths = None
    elif fps_path.is_dir():
        fps_paths = sorted(
            f for f in fps_path.glob("*.npy") if not f.stem.endswith(".indices")
        )
    else:
        fps_paths = [fps_path]
    ca = cluster_analysis(
        clusters,
        fps_paths,
        smiles=smiles,
        top=top,
        n_features=n_features,
        input_is_packed=input_is_packed,
        min_size=min_size,
    )
    fn(ca, title=title, **fn_kwargs)
    if save:
        if filename is None:
            unique_id = format(random.getrandbits(32), "08x")
            filename = f"{fn_name}-{unique_id}.pdf"
        plt.savefig(Path.cwd() / filename)
    if show:
        plt.show()
