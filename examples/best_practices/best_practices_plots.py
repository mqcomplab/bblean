# type: ignore
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
from collections import defaultdict
from rdkit.Chem import rdFMCS, SaltRemover
from rdkit import Chem
from rdkit.Chem import Draw
import bblean.similarity as iSIM

# This file includes the functions for the visualizations of the best practices paper.


def clusters_pop_plot(
    clusters: list,
    save_path: str = None,
):
    """Plot the population of each cluster as a bar chart.

    Args:
        clusters (list): List of cluster sizes.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """

    # Calculate the counts of the populations
    lenghts = [len(c) for c in clusters]
    n_1000 = sum(1 for lenght in lenghts if lenght > 1000)
    n_100 = sum(1 for lenght in lenghts if lenght > 100)
    n_10 = sum(1 for lenght in lenghts if lenght > 10)
    n_1 = sum(1 for lenght in lenghts if lenght > 1)
    n_0 = sum(1 for lenght in lenghts if lenght > 0)

    plt.figure(figsize=(3, 4))
    plt.bar("Num_cluster", n_0, label=">0", color="blue")
    plt.bar("Num_cluster", n_1, label=">1", color="orange")
    plt.bar("Num_cluster", n_10, label=">10", color="gray")
    plt.bar("Num_cluster", n_100, label=">100", color="green")
    plt.bar("Num_cluster", n_1000, label=">1000", color="red")
    plt.legend()
    plt.ylabel("Number of Clusters")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=400)
    else:
        plt.show()


def sankey_cluster_flow(
    clusters_before,
    clusters_after,
    min_flow=1,
    top_n=20,
    max_flow=None,
    title=None,
    show_counts=True,
):
    """Plot a Sankey diagram showing flow from clusters_before to clusters_after.
    Parameters:
    ---------------------
    clusters_before / clusters_after:
    list-of-lists of integer molecule indices.
    min_flow:
    minimum number of molecules for a link to be considered (per before->after link).
    top_n:
    show flows into the top_n largest clusters in clusters_after (by size).
    max_flow:
    if set, only show links with count <= max_flow (useful to inspect small flows).
    title:
    optional title for the figure.
    show_counts:
    include cluster sizes in node labels.

    Returns:
    ---------------------
    Plotly Figure object.
    """
    # choose top_n after-clusters by size
    after_sizes = [(i, len(cl)) for i, cl in enumerate(clusters_after)]
    after_sizes.sort(key=lambda x: x[1], reverse=True)
    top_after_ids = [i for i, _ in after_sizes[:top_n]]

    # build mapping from molecule index -> after cluster id (if any)
    after_map = {}
    for after_id, cl in enumerate(clusters_after):
        for idx in cl:
            after_map[idx] = after_id

    # count flows from before_id -> after_id (only to top_after_ids)
    flow = defaultdict(int)
    before_has_flow = set()
    for before_id, cl in enumerate(clusters_before):
        for idx in cl:
            after_id = after_map.get(idx, None)
            if after_id is None:
                continue
            if after_id not in top_after_ids:
                continue
            flow[(before_id, after_id)] += 1
            before_has_flow.add(before_id)

    # prepare node lists: only before nodes that contribute to selected after nodes
    before_ids = sorted(before_has_flow)
    n_before = len(before_ids)
    # n_after = len(top_after_ids)

    # prepare labels (use 'before'/'after' prefixes)
    if show_counts:
        bef_lab = [f"before:{i} ({len(clusters_before[i])})" for i in before_ids]
        aft_lab = [f"after:{j} ({len(clusters_after[j])})" for j in top_after_ids]
    else:
        bef_lab = [f"before:{i}" for i in before_ids]
        aft_lab = [f"after:{j}" for j in top_after_ids]
    labels = bef_lab + aft_lab

    # mapping from original ids to sankey node indices
    before_index = {bid: idx for idx, bid in enumerate(before_ids)}
    after_index = {aid: n_before + i for i, aid in enumerate(top_after_ids)}

    # Build link tuples and apply min_flow and max_flow (if provided)
    links = [
        (b, a, cnt)
        for (b, a), cnt in flow.items()
        if cnt >= min_flow and (max_flow is None or cnt <= max_flow)
    ]

    # sort links by size (descending) for stable plotting
    links.sort(key=lambda x: x[2], reverse=True)

    sources = [before_index[b] for b, _, _ in links]
    targets = [after_index[a] for _, a, _ in links]
    values = [cnt for _, _, cnt in links]

    if len(values) == 0:
        print("No flows match filters; adjust min_flow/max_flow or check clusters")

    node = dict(label=labels, pad=15, thickness=20)
    link = dict(source=sources, target=targets, value=values)

    fig = go.Figure(go.Sankey(node=node, link=link))
    title_text = title or f"Cluster flow - top {len(top_after_ids)} clusters"
    fig.update_layout(
        title_text=title_text,
        font_size=10,
    )
    return fig


def plot_cluster_refinement(
    df: pd.DataFrame | None = None,
    fps: np.ndarray | None = None,
    show: bool = True,
    count_scale: float = 1e3,
    save_path: str = None,
):
    """Plot cluster refinement summary.

    Parameters
    ----------
    df : pd.DataFrame | None
        DataFrame with columns
        ['iteration',
        'n_clusters',
        'n_singletons',
        'dispersions',
        'dispersions_no_singletons'].
    fps : np.ndarray | None
        Fingerprints used to compute iSIM line. Required if df is None.
    Returns
    -------
    fig, axes
        Matplotlib Figure and Axes objects.
    """

    isim_line = iSIM.jt_isim_packed(fps)

    # Extract data
    iterations = df["iteration"].to_numpy()
    num_clusters = df["n_clusters"].to_numpy()
    num_singletons = df["n_singletons"].to_numpy()
    singleton_ratio = num_singletons / num_clusters
    dispersion_medoids = df["dispersions"].to_numpy()
    dispersion_medoids_non_singleton = df["dispersions_no_singletons"].to_numpy()

    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Clusters, Singletons, and Singleton/Cluster Ratio
    ax1 = axes[0]
    ax1.plot(iterations, num_clusters / count_scale, label="Clusters", marker="o")
    ax1.plot(iterations, num_singletons / count_scale, label="Singletons", marker="o")
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel(f"Count (x{int(count_scale)})", fontsize=12)
    ax1.legend(loc="upper left")

    # Twin y-axis for Singleton/Cluster Ratio
    ax2 = ax1.twinx()
    ax2.plot(
        iterations,
        singleton_ratio,
        label="Singleton/Cluster Ratio",
        color="green",
        marker="o",
    )
    ax2.set_ylabel("Singleton/Cluster Ratio", color="green", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="green")

    # Panel 2: Dispersion of Medoids
    ax3 = axes[1]
    ax3.plot(iterations, dispersion_medoids, marker="o", label="All Clusters")
    ax3.plot(
        iterations,
        dispersion_medoids_non_singleton,
        marker="o",
        label="Non-Singleton Clusters",
    )
    ax3.legend(loc="upper right")
    ax3.set_xlabel("Iteration", fontsize=12)
    ax3.set_ylabel("Dispersion of Medoids (iSIM)", fontsize=12)
    ax3.hlines(
        isim_line,
        xmin=iterations[0],
        xmax=iterations[-1],
        colors="orange",
        linestyles="dashed",
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=400)
    if show:
        plt.show()
    return fig, axes


def plot_threshold_scan(
    df: pd.DataFrame,
    fps: np.ndarray | None = None,
    show: bool = True,
    save_path: str = None,
    count_scale: float = 1e6,
):

    isim_line = iSIM.jt_isim_packed(fps)

    # Extract data
    thresholds = df["thresholds"].to_numpy()
    num_clusters = df["n_clusters"].to_numpy()
    num_singletons = df["n_singletons"].to_numpy()
    singleton_ratio = num_singletons / num_clusters
    dispersion_medoids = df["dispersions"].to_numpy()
    dispersion_medoids_non_singleton = df["dispersions_non_singleton"].to_numpy()

    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Clusters, Singletons, and Singleton/Cluster Ratio
    ax1 = axes[0]
    ax1.plot(thresholds, num_clusters / count_scale, label="Clusters", marker="o")
    ax1.plot(thresholds, num_singletons / count_scale, label="Singletons", marker="o")
    ax1.set_xlabel("Threshold", fontsize=12)
    ax1.set_ylabel(f"Count (x{int(count_scale)})", fontsize=12)
    ax1.set_xticks(np.arange(0, 1.1, 0.1))
    ax1.legend(loc="upper left")

    # Twin y-axis for Singleton/Cluster Ratio
    ax2 = ax1.twinx()
    ax2.plot(
        thresholds,
        singleton_ratio,
        label="Singleton/Cluster Ratio",
        color="green",
        marker="o",
    )
    ax2.set_ylabel("Singleton/Cluster Ratio", color="green", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="green")

    # Add top axis for standard deviations
    ax1_top = ax1.twiny()
    n_steps = len(thresholds) - 1
    step_min, step_max = 0, n_steps
    val_min, val_max = thresholds[0], thresholds[-1]

    def threshold_to_step(x):
        return step_min + (x - val_min) * (step_max - step_min) / (val_max - val_min)

    ax1_top.set_xlim(
        threshold_to_step(ax1.get_xlim()[0]),
        threshold_to_step(ax1.get_xlim()[1]),
    )
    ax1_top.set_xticks(np.arange(0, n_steps + 1, 1))
    ax1_top.set_xlabel("Standard Deviations above iSIM", fontsize=12)

    # Panel 2: Dispersion of Medoids
    ax3 = axes[1]
    ax3.plot(thresholds, dispersion_medoids, marker="o", label="All Clusters")
    ax3.plot(
        thresholds,
        dispersion_medoids_non_singleton,
        marker="o",
        label="Non-Singleton Clusters",
    )
    ax3.legend(loc="upper right")
    ax3.set_xlabel("Threshold", fontsize=12)
    ax3.set_ylabel("Dispersion of Medoids (iSIM)", fontsize=12)
    ax3.set_xticks(np.arange(0, 1.1, 0.1))
    ax3.hlines(
        isim_line,
        xmin=thresholds[0],
        xmax=thresholds[-1],
        colors="orange",
        linestyles="dashed",
    )

    # Add top axis for standard deviations
    ax3_top = ax3.twiny()
    ax3_top.set_xlim(
        threshold_to_step(ax3.get_xlim()[0]), threshold_to_step(ax3.get_xlim()[1])
    )
    ax3_top.set_xticks(np.arange(0, n_steps + 1, 1))
    ax3_top.set_xlabel("Standard Deviations above iSIM", fontsize=12)

    # Adjust layout and show the figure
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=400)
    if show:
        plt.show()


def plot_distributions_by_fp(pairwise_similarities: dict, save_path: str = None):
    plt.figure(figsize=(8, 6))
    colors = ["blue", "orange", "green", "gray", "red"]
    for fp_type, color in zip(pairwise_similarities.keys(), colors):
        plt.hist(
            pairwise_similarities[fp_type].flatten(),
            bins=50,
            alpha=0.5,
            label=fp_type,
            color=color,
        )
        plt.axvline(
            x=np.mean(pairwise_similarities[fp_type]),
            ymin=0,
            ymax=plt.ylim()[1],
            color=color,
            linestyle="dashed",
        )
    plt.xlim(0, 1)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.xlabel("Tanimoto Similarity")
    plt.ylabel("Frequency")
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=400)
    else:
        plt.show()


def plot_branching_factor_scan(
    df: pd.DataFrame,
    fps: np.ndarray | None = None,
    sigmas_list: list[float] | None = None,
    save_path: str = None,
):
    df["ratio"] = df["n_singletons"] / df["n_clusters"]

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    what_list = ["n_clusters", "n_singletons", "ratio", "dispersions_non_singleton"]

    for i, what in enumerate(what_list):
        pivot_df = df.pivot(index="branching_factor", columns="thresholds", values=what)
        ax = axs.flatten()[i]

        # Show colorbar for both, with specific range for the bottom row
        show_cbar = True
        cbar_kws = {"ticks": [0, 0.25, 0.5, 0.75, 1]} if i // 2 == 1 else {}
        im = sns.heatmap(
            pivot_df[::-1],  # reverse row order for correct y-axis orientation
            ax=ax,
            cmap="RdYlBu_r",
            vmin=0,
            vmax=1 if i // 2 == 1 else None,  # set range 0..1 for bottom row
            cbar=show_cbar,
            cbar_kws=cbar_kws,
            annot=False,
        )

        ax.set_title(what.replace("_", " ").title())

        # Only show x-axis labels on bottom row
        if i // 2 == 0:
            ax.set_xticklabels([])  # hide top-row x tick labels
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Thresholds (std above iSIM)")
            ax.set_xticklabels([str(t) for t in sigmas_list], rotation=0)

        # Only show y-axis labels on left column
        if i % 2 == 1:
            ax.set_yticklabels([])  # hide right-column y tick labels
            ax.set_ylabel("")
        else:
            ax.set_ylabel("Branching Factor")

        # If this panel has a colorbar, remove its label
        if show_cbar:
            mappable = im.collections[0]
            cbar = getattr(mappable, "colorbar", None)
            if cbar is not None:
                cbar.set_label("")  # remove colorbar label
                cbar.ax.tick_params(labelsize=8)

                if what == "dispersions_non_singleton":
                    # draw marker line at min_t and annotate (clamped if outside)
                    cmin, cmax = cbar.vmin, cbar.vmax
                    y_plot = min(max(iSIM.jt_isim_packed(fps), cmin), cmax)
                    # draw horizontal line (data coords)
                    cbar.ax.axhline(y_plot, color="k", linewidth=2, zorder=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=400)
    else:
        plt.show()


def sampled_MSC(cluster, smiles, n_samples=100):
    """Calculate the MSC for a sampled subset of molecules in a cluster."""
    n_samples = min(n_samples, len(cluster))

    # Get the SMILES for the sampled molecules
    remover = SaltRemover.SaltRemover()
    sampled_smiles = [smiles[i] for i in cluster[:n_samples]]
    mols = [Chem.MolFromSmiles(smi) for smi in sampled_smiles]
    mols = [remover.StripMol(mol) for mol in mols]

    # Find the MSC of those molecules
    MSC = rdFMCS.FindMCS(mols, threshold=0.75)
    MSC_mol = Chem.MolFromSmarts(MSC.smartsString)
    img2 = Draw.MolToImage(MSC_mol)

    # Find substructure matches in the sampled molecules with the MSC
    for mol in mols:
        if mol.HasSubstructMatch(MSC_mol):
            match = mol.GetSubstructMatch(MSC_mol)
            atom_indices = list(match)
            highlight_atoms = atom_indices
            # Highlight the matching substructure
            mol.SetProp("_highlightAtoms", ",".join(map(str, highlight_atoms)))

    highlight_lists = []
    for mol in mols:
        if mol.HasProp("_highlightAtoms"):
            vals = mol.GetProp("_highlightAtoms").split(",")
            highlight_lists.append(list(map(int, vals)))
        else:
            highlight_lists.append([])
    img1 = Draw.MolsToGridImage(
        mols, highlightAtomLists=highlight_lists, molsPerRow=5, subImgSize=(200, 200)
    )
    return img1, img2
