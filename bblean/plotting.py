r"""Plotting and visualization convenience functions"""

from numpy.typing import NDArray
import typing as tp

import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from bblean.utils import batched
from bblean.analysis import ClusterAnalysis
from bblean.packing import unpack_fingerprints

# TODO: Mol relocation plots?


# Similar to "init_plot" in the original bitbirch
def summary_plot(
    c: ClusterAnalysis, /, title: str | None = None
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    df = c.df
    num_clusters = c.num_clusters
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot and annotate the number of molecules
    label_strs = df["label"].astype(str)  # TODO: Is this necessary?
    ax.bar(label_strs, df["mol_num"], color="blue", label="Molecules")
    for i, mol in enumerate(df["mol_num"]):
        plt.text(i, mol, f"{mol}", ha="center", va="bottom", fontsize=10, color="black")

    # Plot and annotate the number of unique scaffolds
    plt.bar(label_strs, df["unique_scaffolds_num"], color="orange", label="Scaffolds")
    for i, s in enumerate(df["unique_scaffolds_num"]):
        plt.text(i, s, f"{s}", ha="center", va="bottom", fontsize=8, color="white")

    # Labels
    ax.set_xlabel("Cluster ID", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_xticks(range(num_clusters))
    ax.legend(loc="upper right")

    # Plot iSIM
    ax_isim = ax.twinx()
    ax_isim.plot(
        df["label"], df["isim"], color="green", marker="o", label="iSIM", linewidth=2
    )
    ax_isim.set_ylabel("iSIM", fontsize=12)
    ax_isim.set_yticks(np.arange(0, 1.1, 0.1))
    ax_isim.set_ylim(0, 1)
    ax_isim.legend(loc="upper right", bbox_to_anchor=(0.80, 1))

    # fig.tight_layout()
    if title is not None:
        fig.suptitle(f"Top {num_clusters} Cluster Metrics for {title}", fontsize=14)
    return fig, (ax, ax_isim)


def tsne_plot(
    c: ClusterAnalysis, /, title: str | None = None
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    df = c.df
    fps_list: list[NDArray[np.uint8]] = []
    label_list: list[int] = []
    for fps, label in zip(c.fps, df["label"]):
        cluster_fps = [fp for fp in unpack_fingerprints(fps, n_features=c.n_features)]
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
