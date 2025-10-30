r"""Command line interface entrypoints"""

import warnings
import random
import typing as tp
import math
import shutil
import sys
import pickle
import multiprocessing as mp
import multiprocessing.shared_memory as shmem
from typing import Annotated
from pathlib import Path

from typer import Typer, Argument, Option, Abort, Context, Exit

from bblean._memory import launch_monitor_rss_daemon
from bblean._timer import Timer
from bblean._config import DEFAULTS, collect_system_specs_and_dump_config, TSNE_SEED
from bblean.utils import _import_bitbirch_variant, batched

app = Typer(
    rich_markup_mode="markdown",
    add_completion=False,
    help=r"""CLI tool for serial or parallel fast clustering of molecular fingerprints
    using the memory-efficient and compute-efficient *O(N)* BitBIRCH algorithm ('Lean'
    version). For more info about the subcommands run `bb <subcommand> --help `.""",
)


def _print_help_banner(ctx: Context, value: bool) -> None:
    if value:
        from bblean._console import get_console

        console = get_console()
        console.print_banner()
        console.print(ctx.get_help())
        raise Exit()


def _validate_output_dir(out_dir: Path, overwrite: bool = False) -> None:
    if out_dir.exists():
        if not out_dir.is_dir():
            raise RuntimeError("Output dir should be a dir")
        if any(out_dir.iterdir()):
            if overwrite:
                shutil.rmtree(out_dir)
            else:
                raise RuntimeError(f"Output dir {out_dir} has files")


# Validate that the naming convention for the input files is correct
def _validate_input_dir(in_dir: Path | str) -> None:
    in_dir = Path(in_dir)
    if not in_dir.is_dir():
        raise RuntimeError(f"Input dir {in_dir} should be a dir")
    if not any(in_dir.glob("*.npy")):
        raise RuntimeError(f"Input dir {in_dir} should have *.npy fingerprint files")


@app.callback()
def _main(
    ctx: Context,
    help_: Annotated[
        bool,
        Option(
            "--help/ ",
            "-h",
            is_eager=True,
            help="Show this message and exit.",
            callback=_print_help_banner,
        ),
    ] = False,
) -> None:
    pass


@app.command("plot-pops", rich_help_panel="Analysis")
def _plot_pops(
    clusters_path: Annotated[
        Path,
        Argument(help="Path to the clusters file, or a dir with a clusters.pkl file"),
    ],
    fps_path: Annotated[
        Path | None,
        Option(
            "-f",
            "--fps-path",
            help="Path to fingerprint file, or directory with fingerprint files",
            show_default=False,
        ),
    ] = None,
    title: Annotated[
        str | None,
        Option("--title", help="Plot title"),
    ] = None,
    top: Annotated[
        int | None,
        Option("--top"),
    ] = None,
    input_is_packed: Annotated[
        bool,
        Option("--packed-input/--unpacked-input", rich_help_panel="Advanced"),
    ] = True,
    min_size: Annotated[
        int,
        Option("--min-size"),
    ] = 0,
    n_features: Annotated[
        int | None,
        Option(
            "--n-features",
            help="Number of features in the fingerprints."
            " Only for packed inputs *if it is not a multiple of 8*."
            " Not required for typical fingerprint sizes (e.g. 2048, 1024)",
            rich_help_panel="Advanced",
        ),
    ] = None,
    save: Annotated[
        bool,
        Option("--save/--no-save"),
    ] = True,
    filename: Annotated[
        str | None,
        Option("--filename"),
    ] = None,
    verbose: Annotated[
        bool,
        Option("-v/-V", "--verbose/--no-verbose"),
    ] = True,
    show: Annotated[
        bool,
        Option("--show/--no-show", hidden=True),
    ] = True,
    use_global: Annotated[bool, Option("--global/--no-global"),] = False,
) -> None:
    r"""Population plot of the clustering results"""
    from bblean._console import get_console

    console = get_console(silent=not verbose)
    # Imports may take a bit of time since sklearn is slow, so start the spinner here
    with console.status("[italic]Analyzing clusters...[/italic]", spinner="dots"):
        from bblean.plotting import _dispatch_visualization, pops_plot

        _dispatch_visualization(
            clusters_path,
            "pops",
            pops_plot,
            {},
            min_size=min_size,
            top=top,
            n_features=n_features,
            input_is_packed=input_is_packed,
            fps_path=fps_path,
            title=title,
            filename=filename,
            verbose=verbose,
            save=save,
            show=show,
            use_global=use_global,
        )


@app.command("plot-umap", rich_help_panel="Analysis")
def _plot_umap(
    clusters_path: Annotated[
        Path,
        Argument(help="Path to the clusters file, or a dir with a clusters.pkl file"),
    ],
    fps_path: Annotated[
        Path | None,
        Option(
            "-f",
            "--fps-path",
            help="Path to fingerprint file, or directory with fingerprint files",
            show_default=False,
        ),
    ] = None,
    title: Annotated[
        str | None,
        Option("--title", help="Plot title"),
    ] = None,
    save: Annotated[
        bool,
        Option("--save/--no-save"),
    ] = True,
    top: Annotated[
        int,
        Option("--top"),
    ] = 20,
    input_is_packed: Annotated[
        bool,
        Option("--packed-input/--unpacked-input", rich_help_panel="Advanced"),
    ] = True,
    scaling: Annotated[
        str,
        Option("--scaling", rich_help_panel="Advanced"),
    ] = "normalize",
    min_size: Annotated[
        int,
        Option("--min-size"),
    ] = 0,
    n_features: Annotated[
        int | None,
        Option(
            "--n-features",
            help="Number of features in the fingerprints."
            " Only for packed inputs *if it is not a multiple of 8*."
            " Not required for typical fingerprint sizes (e.g. 2048, 1024)",
            rich_help_panel="Advanced",
        ),
    ] = None,
    filename: Annotated[
        str | None,
        Option("--filename"),
    ] = None,
    verbose: Annotated[
        bool,
        Option("-v/-V", "--verbose/--no-verbose"),
    ] = True,
    show: Annotated[
        bool,
        Option("--show/--no-show", hidden=True),
    ] = True,
    deterministic: Annotated[
        bool,
        Option("--deterministic/--no-deterministic"),
    ] = False,
    n_neighbors: Annotated[
        int,
        Option("-n", "--neighbors"),
    ] = 15,
    min_dist: Annotated[
        float,
        Option("-d", "--min-dist"),
    ] = 0.5,
    metric: Annotated[
        str,
        Option("--metric"),
    ] = "euclidean",
    densmap: Annotated[
        bool,
        Option("--densmap/--no-densmap"),
    ] = False,
    workers: Annotated[
        int | None,
        Option(
            "-w",
            "--workers",
            help="Num. cores to use for parallel processing",
            rich_help_panel="Advanced",
        ),
    ] = None,
    use_global: Annotated[bool, Option("--global/--no-global"),] = False,
) -> None:
    r"""UMAP visualization of the clustering results"""
    from bblean._console import get_console

    console = get_console(silent=not verbose)
    # Imports may take a bit of time since sklearn is slow, so start the spinner here
    with console.status("[italic]Analyzing clusters...[/italic]", spinner="dots"):
        from bblean.plotting import _dispatch_visualization, umap_plot

        kwargs = dict(
            metric=metric,
            densmap=densmap,
            deterministic=deterministic,
            n_neighbors=n_neighbors,
            workers=workers,
            min_dist=min_dist,
        )
        _dispatch_visualization(
            clusters_path,
            "umap",
            umap_plot,
            kwargs,
            min_size=min_size,
            top=top,
            n_features=n_features,
            input_is_packed=input_is_packed,
            fps_path=fps_path,
            title=title,
            filename=filename,
            verbose=verbose,
            save=save,
            show=show,
            use_global=use_global,
        )


@app.command("plot-pca", rich_help_panel="Analysis")
def _plot_pca(
    clusters_path: Annotated[
        Path,
        Argument(help="Path to the clusters file, or a dir with a clusters.pkl file"),
    ],
    fps_path: Annotated[
        Path | None,
        Option(
            "-f",
            "--fps-path",
            help="Path to fingerprint file, or directory with fingerprint files",
            show_default=False,
        ),
    ] = None,
    title: Annotated[
        str | None,
        Option("--title", help="Plot title"),
    ] = None,
    top: Annotated[
        int,
        Option("--top"),
    ] = 20,
    min_size: Annotated[
        int,
        Option("--min-size"),
    ] = 0,
    input_is_packed: Annotated[
        bool,
        Option("--packed-input/--unpacked-input", rich_help_panel="Advanced"),
    ] = True,
    scaling: Annotated[
        str,
        Option("--scaling", rich_help_panel="Advanced"),
    ] = "normalize",
    n_features: Annotated[
        int | None,
        Option(
            "--n-features",
            help="Number of features in the fingerprints."
            " Only for packed inputs *if it is not a multiple of 8*."
            " Not required for typical fingerprint sizes (e.g. 2048, 1024)",
            rich_help_panel="Advanced",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        Option("-v/-V", "--verbose/--no-verbose"),
    ] = True,
    show: Annotated[
        bool,
        Option("--show/--no-show", hidden=True),
    ] = True,
    whiten: Annotated[
        bool,
        Option("--whiten/--no-whiten"),
    ] = False,
    save: Annotated[
        bool,
        Option("--save/--no-save"),
    ] = True,
    filename: Annotated[
        str | None,
        Option("--filename"),
    ] = None,
    use_global: Annotated[bool, Option("--global/--no-global"),] = False,
) -> None:
    r"""PCA visualization of the clustering results"""
    from bblean._console import get_console

    console = get_console(silent=not verbose)
    # Imports may take a bit of time since sklearn is slow, so start the spinner here
    with console.status("[italic]Analyzing clusters...[/italic]", spinner="dots"):
        from bblean.plotting import _dispatch_visualization, pca_plot

        _dispatch_visualization(
            clusters_path,
            "pca",
            pca_plot,
            {"whiten": whiten},
            min_size=min_size,
            top=top,
            n_features=n_features,
            input_is_packed=input_is_packed,
            fps_path=fps_path,
            title=title,
            filename=filename,
            verbose=verbose,
            save=save,
            show=show,
            use_global=use_global,
        )


@app.command("plot-tsne", rich_help_panel="Analysis")
def _plot_tsne(
    clusters_path: Annotated[
        Path,
        Argument(help="Path to the clusters file, or a dir with a clusters.pkl file"),
    ],
    fps_path: Annotated[
        Path | None,
        Option(
            "-f",
            "--fps-path",
            help="Path to fingerprint file, or directory with fingerprint files",
            show_default=False,
        ),
    ] = None,
    title: Annotated[
        str | None,
        Option("--title", help="Plot title"),
    ] = None,
    save: Annotated[
        bool,
        Option("--save/--no-save"),
    ] = True,
    min_size: Annotated[
        int,
        Option("--min-size"),
    ] = 0,
    filename: Annotated[
        str | None,
        Option("--filename"),
    ] = None,
    exaggeration: Annotated[
        float | None,
        Option("-e", "--exaggeration", rich_help_panel="Advanced"),
    ] = None,
    seed: Annotated[
        int | None,
        Option(
            "-s",
            "--seed",
            help=(
                "Seed for the rng, fixed value by default, for reproducibility."
                " Pass -1 to randomize"
            ),
            show_default=False,
            rich_help_panel="Advanced",
        ),
    ] = TSNE_SEED,
    top: Annotated[
        int,
        Option("--top"),
    ] = 20,
    metric: Annotated[
        str,
        Option("--metric", help="Metric to use in the t-SNE source space"),
    ] = "euclidean",
    dof: Annotated[
        float,
        Option("-d", "--dof", rich_help_panel="Advanced"),
    ] = 1.0,
    perplexity: Annotated[
        int,
        Option(help="t-SNE perplexity", rich_help_panel="Advanced"),
    ] = 30,
    input_is_packed: Annotated[
        bool,
        Option("--packed-input/--unpacked-input", rich_help_panel="Advanced"),
    ] = True,
    n_features: Annotated[
        int | None,
        Option(
            "--n-features",
            help="Number of features in the fingerprints."
            " Only for packed inputs *if it is not a multiple of 8*."
            " Not required for typical fingerprint sizes (e.g. 2048, 1024)",
            rich_help_panel="Advanced",
        ),
    ] = None,
    scaling: Annotated[
        str,
        Option("--scaling", rich_help_panel="Advanced"),
    ] = "normalize",
    do_pca_init: Annotated[
        bool,
        Option(
            "--pca-init/--no-pca-init",
            rich_help_panel="Advanced",
            help="Use PCA for initialization",
        ),
    ] = True,
    pca_reduce: Annotated[
        int | None,
        Option(
            "-p",
            "--pca-reduce",
            rich_help_panel="Advanced",
            help=(
                "Reduce fingerprint dimensionality to N components using PCA."
                " A value of 50 or more maintains cluster structure in general"
            ),
        ),
    ] = None,
    workers: Annotated[
        int | None,
        Option(
            "-w",
            "--workers",
            help="Num. cores to use for parallel processing",
            rich_help_panel="Advanced",
        ),
    ] = None,
    multiscale: Annotated[
        bool,
        Option(
            "-m/-M",
            "--multiscale/--no-multiscale",
            rich_help_panel="Advanced",
            help="Use multiscale perplexities (WARNING: Can be very slow!)",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        Option("-v/-V", "--verbose/--no-verbose"),
    ] = True,
    show: Annotated[
        bool,
        Option("--show/--no-show", hidden=True),
    ] = True,
    use_global: Annotated[bool, Option("--global/--no-global"),] = False,
) -> None:
    r"""t-SNE visualization of the clustering results"""
    from bblean._console import get_console

    console = get_console(silent=not verbose)
    # Imports may take a bit of time since sklearn is slow, so start the spinner here
    with console.status("[italic]Analyzing clusters...[/italic]", spinner="dots"):
        from bblean.plotting import _dispatch_visualization, tsne_plot

        kwargs = dict(
            metric=metric,
            seed=seed,
            perplexity=perplexity,
            exaggeration=exaggeration,
            dof=dof,
            workers=workers,
            scaling=scaling,
            do_pca_init=do_pca_init,
            multiscale=multiscale,
            pca_reduce=pca_reduce,
        )
        _dispatch_visualization(
            clusters_path,
            "tsne",
            tsne_plot,
            kwargs,
            min_size=min_size,
            top=top,
            n_features=n_features,
            input_is_packed=input_is_packed,
            fps_path=fps_path,
            title=title,
            filename=filename,
            verbose=verbose,
            save=save,
            show=show,
            use_global=use_global,
        )


@app.command("summary", rich_help_panel="Analysis")
def _table_summary(
    clusters_path: Annotated[
        Path,
        Argument(help="Path to the clusters file, or a dir with a clusters.pkl file"),
    ],
    fps_path: Annotated[
        Path | None,
        Option(
            "-f",
            "--fps-path",
            help="Path to fingerprint file, or directory with fingerprint files",
            show_default=False,
        ),
    ] = None,
    min_size: Annotated[
        int,
        Option("--min-size"),
    ] = 0,
    smiles_path: Annotated[
        Path | None,
        Option(
            "-s",
            "--smiles-path",
            show_default=False,
            help="Optional smiles path, if passed a scaffold analysis is performed",
        ),
    ] = None,
    top: Annotated[
        int,
        Option("--top"),
    ] = 20,
    input_is_packed: Annotated[
        bool,
        Option("--packed-input/--unpacked-input", rich_help_panel="Advanced"),
    ] = True,
    scaffold_fp_kind: Annotated[
        str,
        Option("--scaffold-fp-kind"),
    ] = DEFAULTS.fp_kind,
    n_features: Annotated[
        int | None,
        Option(
            "--n-features",
            help="Number of features in the fingerprints."
            " Only for packed inputs *if it is not a multiple of 8*."
            " Not required for typical fingerprint sizes (e.g. 2048, 1024)",
            rich_help_panel="Advanced",
        ),
    ] = None,
    metrics: Annotated[
        bool,
        Option(
            "--metrics/--no-metrics",
            help="Calculate clustering indices (Dunn, DBI, CHI)",
        ),
    ] = False,
    chosen_metrics: Annotated[
        str,
        Option(
            "-m",
            "--metrics-choice",
            help=(
                "Chosen metrics. "
                " Comma-separated list including dunn (slow), dbi or chi"
            ),
        ),
    ] = "dunn,dbi,chi",
    metrics_top: Annotated[
        int | None,
        Option("--metrics-top", rich_help_panel="Advanced"),
    ] = 100,
    metrics_min_size: Annotated[
        int,
        Option("--metrics-min-size", hidden=True),
    ] = 1,
    use_global: Annotated[bool, Option("--global/--no-global"),] = False,
    verbose: Annotated[
        bool,
        Option("--verbose/--no-verbose", hidden=True),
    ] = True,
) -> None:
    r"""Summary table of clustering results, together with cluster metrics"""
    from bblean._console import get_console
    from bblean.smiles import load_smiles
    from bblean.analysis import cluster_analysis
    from bblean.utils import _has_files_or_valid_symlinks
    from bblean.metrics import jt_dbi, jt_isim_chi, jt_isim_dunn, _calc_centrals
    from rich.table import Table

    console = get_console(silent=not verbose)
    # Imports may take a bit of time since sklearn is slow, so start the spinner here
    with console.status("[italic]Analyzing clusters...[/italic]", spinner="dots"):
        if clusters_path.is_dir():
            clusters_path = clusters_path / (
                "global-clusters.pkl" if use_global else "clusters.pkl"
            )
        with open(clusters_path, mode="rb") as f:
            clusters = pickle.load(f)
        if fps_path is None:
            input_fps_path = clusters_path.parent / "input-fps"
            if input_fps_path.is_dir() and _has_files_or_valid_symlinks(input_fps_path):
                fps_path = input_fps_path
            else:
                msg = (
                    "Could not find input fingerprints. Please use --fps-path."
                    " Summary plot without fingerprints doesn't include isim values"
                )
                warnings.warn(msg)
        if fps_path is None:
            fps_paths = None
        elif fps_path.is_dir():
            fps_paths = sorted(fps_path.glob("*.npy"))
        else:
            fps_paths = [fps_path]
        ca = cluster_analysis(
            clusters,
            fps_paths,
            smiles=load_smiles(smiles_path) if smiles_path is not None else (),
            top=top,
            n_features=n_features,
            input_is_packed=input_is_packed,
            min_size=min_size,
        )
        table = Table(title=(f"Top {top} clusters" if top is not None else "Clusters"))
        table.add_column("Size", justify="center")
        table.add_column("% fps", justify="center")
        table.add_column("iSIM", justify="center")
        if smiles_path is not None:
            table.add_column("Size/Scaff.", justify="center")
            table.add_column("Num. Scaff.", justify="center")
            table.add_column("Scaff. iSIM", justify="center")
        sizes = ca.sizes
        isims = ca.isims
        total_fps = ca.total_fps
        for i in range(ca.clusters_num):
            size = sizes[i]
            percent = size / total_fps * 100
            table.add_row(f"{size:,}", f"{percent:.2f}", f"{isims[i]:.3f}")
        console.print(table)
        console.print()
        console.print(f"Total num. fps: {total_fps:,}")
        console.print(f"Total num. clusters: {ca.all_clusters_num:,}")
        singles = ca.all_singletons_num
        singles_percent = singles * 100 / ca.all_clusters_num
        console.print(f"Total num. singletons: {singles:,} ({singles_percent:.2f} %)")
        gt10 = ca.all_clusters_num_with_size_above(10)
        gt10_percent = gt10 * 100 / ca.all_clusters_num
        console.print(
            f"Total num. clusters, size > 10: {gt10:,} ({gt10_percent:.2f} %)"
        )
        gt100 = ca.all_clusters_num_with_size_above(100)
        gt100_percent = gt100 * 100 / ca.all_clusters_num
        console.print(
            f"Total num. clusters, size > 100: {gt100:,} ({gt100_percent:.2f} %)"
        )
        console.print(
            f"num-clusters/num-fps ratio: {ca.all_clusters_num / total_fps:.2f}"
        )
        console.print(f"Mean size: {ca.all_clusters_mean_size:.2f}")
        console.print(f"Max. size: {ca.all_clusters_max_size:,}")
        console.print(f"Q3 (75%) size: {ca.all_clusters_q3:,}")
        console.print(f"Median size: {ca.all_clusters_median_size:,}")
        console.print(f"Q1 (25%) size: {ca.all_clusters_q1:,}")
        console.print(f"Min. size: {ca.all_clusters_min_size:,}")
    if metrics:
        chosen = set(s.lower() for s in chosen_metrics.split(","))
        assert all(s in ["dunn", "chi", "dbi"] for s in chosen)
        # Redo cluster analysis with more *top* clusters
        console.print()
        if metrics_top is None:
            console.print("Clustering metrics:")
        else:
            console.print(f"Clustering metrics considering top {metrics_top} clusters:")
        with console.status("[italic]Reanalyzing clusters...[/italic]", spinner="dots"):
            ca = cluster_analysis(
                clusters,
                fps_paths,
                smiles=(),
                top=metrics_top,
                n_features=n_features,
                input_is_packed=input_is_packed,
                min_size=metrics_min_size,
            )
            clusters = ca.get_top_cluster_fps()
        with console.status("[italic]Calculating centrals...[/italic]", spinner="dots"):
            centrals = _calc_centrals(clusters, kind="centroid")
        if "chi" in chosen:
            chi = jt_isim_chi(clusters, centrals=centrals, verbose=verbose)
            console.print(f"    - CHI index: {chi:.4f} (Higher is better)")
        if "dbi" in chosen:
            dbi = jt_dbi(clusters, centrals=centrals, verbose=verbose)
            console.print(f"    - DBI index: {dbi:.4e} (Lower is better)")
        if "dunn" in chosen:
            dunn = jt_isim_dunn(clusters, verbose=verbose)
            console.print(f"    - Dunn index: {dunn:.4f} (Higher is better)")


@app.command("plot-summary", rich_help_panel="Analysis")
def _plot_summary(
    clusters_path: Annotated[
        Path,
        Argument(help="Path to the clusters file, or a dir with a clusters.pkl file"),
    ],
    fps_path: Annotated[
        Path | None,
        Option(
            "-f",
            "--fps-path",
            help="Path to fingerprint file, or directory with fingerprint files",
            show_default=False,
        ),
    ] = None,
    save: Annotated[
        bool,
        Option("--save/--no-save"),
    ] = True,
    ylim: Annotated[
        int | None,
        Option("--ylim"),
    ] = None,
    min_size: Annotated[
        int,
        Option("--min-size"),
    ] = 0,
    smiles_path: Annotated[
        Path | None,
        Option(
            "-s",
            "--smiles-path",
            show_default=False,
            help="Optional smiles path, if passed a scaffold analysis is performed",
        ),
    ] = None,
    title: Annotated[
        str | None,
        Option("--title"),
    ] = None,
    filename: Annotated[
        str | None,
        Option("--filename"),
    ] = None,
    top: Annotated[
        int,
        Option("--top"),
    ] = 20,
    input_is_packed: Annotated[
        bool,
        Option("--packed-input/--unpacked-input", rich_help_panel="Advanced"),
    ] = True,
    scaffold_fp_kind: Annotated[
        str,
        Option("--scaffold-fp-kind"),
    ] = DEFAULTS.fp_kind,
    n_features: Annotated[
        int | None,
        Option(
            "--n-features",
            help="Number of features in the fingerprints."
            " Only for packed inputs *if it is not a multiple of 8*."
            " Not required for typical fingerprint sizes (e.g. 2048, 1024)",
            rich_help_panel="Advanced",
        ),
    ] = None,
    annotate: Annotated[
        bool,
        Option(
            "--annotate/--no-annotate",
            help="Display scaffold and fingerprint number in each cluster",
        ),
    ] = True,
    verbose: Annotated[
        bool,
        Option("-v/-V", "--verbose/--no-verbose"),
    ] = True,
    show: Annotated[
        bool,
        Option("--show/--no-show", hidden=True),
    ] = True,
    use_global: Annotated[bool, Option("--global/--no-global"),] = False,
) -> None:
    r"""Summary plot of the clustering results"""
    from bblean._console import get_console

    console = get_console(silent=not verbose)
    # Imports may take a bit of time since sklearn is slow, so start the spinner here
    with console.status("[italic]Analyzing clusters...[/italic]", spinner="dots"):
        from bblean.plotting import _dispatch_visualization, summary_plot
        from bblean.smiles import load_smiles

        _dispatch_visualization(
            clusters_path,
            "summary",
            summary_plot,
            {"annotate": annotate, "counts_ylim": ylim},
            smiles=load_smiles(smiles_path) if smiles_path is not None else (),
            min_size=min_size,
            top=top,
            n_features=n_features,
            input_is_packed=input_is_packed,
            fps_path=fps_path,
            title=title,
            filename=filename,
            verbose=verbose,
            save=save,
            show=show,
            use_global=use_global,
        )


@app.command("run")
def _run(
    ctx: Context,
    input_: Annotated[
        Path | None,
        Argument(help="`*.npy` file with packed fingerprints, or dir `*.npy` files"),
    ] = None,
    out_dir: Annotated[
        Path | None,
        Option(
            "-o",
            "--out-dir",
            help="Dir to dump the output files",
        ),
    ] = None,
    smiles_path: Annotated[
        Path | None,
        Option(
            "-s",
            "--smiles",
            show_default=False,
            help="Optional smiles path, to store smiles with results",
        ),
    ] = None,
    overwrite: Annotated[bool, Option(help="Allow overwriting output files")] = False,
    branching_factor: Annotated[
        int,
        Option(
            "--branching",
            "-b",
            help="BitBIRCH branching factor (all rounds). Usually 254 is"
            " optimal. Set above 254 for slightly less RAM (at the cost of some perf.)",
        ),
    ] = DEFAULTS.branching_factor,
    threshold: Annotated[
        float,
        Option("--threshold", "-t", help="Threshold for merge criterion"),
    ] = DEFAULTS.threshold,
    refine_threshold_change: Annotated[
        float,
        Option(
            "--refine-threshold-change",
            help="Modify threshold for refinement criterion, can be negative",
        ),
    ] = DEFAULTS.refine_threshold_change,
    save_tree: Annotated[
        bool,
        Option("--save-tree/--no-save-tree", rich_help_panel="Advanced"),
    ] = False,
    save_centroids: Annotated[
        bool,
        Option("--save-centroids/--no-save-centroids", rich_help_panel="Advanced"),
    ] = True,
    merge_criterion: Annotated[
        str,
        Option("--set-merge", "-m", help="Merge criterion for initial clustsering"),
    ] = DEFAULTS.merge_criterion,
    refine_merge_criterion: Annotated[
        str,
        Option("--set-refine-merge", help="Merge criterion for refinement clustsering"),
    ] = DEFAULTS.refine_merge_criterion,
    tolerance: Annotated[
        float,
        Option(help="BitBIRCH tolerance. For refinement and --set-merge 'tolerance'"),
    ] = DEFAULTS.tolerance,
    refine_num: Annotated[
        int,
        Option(
            "--refine-num",
            help=(
                "Num. of largest clusters to refine."
                " 1 for standard refinement, 0 is the default (no refinement)"
            ),
        ),
    ] = 0,
    refine_rounds: Annotated[
        int | None,
        Option(
            "--refine-rounds",
            help=("Num. of refinement rounds. "),
            hidden=True,
        ),
    ] = None,
    recluster_rounds: Annotated[
        int,
        Option(
            "--recluster-rounds",
            help=("Num. of reclustering rounds. "),
            hidden=True,
        ),
    ] = 0,
    recluster_shuffle: Annotated[
        bool,
        Option("--recluster-shuffle/--no-recluster-shuffle", hidden=True),
    ] = True,
    n_features: Annotated[
        int | None,
        Option(
            "--n-features",
            help="Number of features in the fingerprints."
            " It must be provided for packed inputs *if it is not a multiple of 8*."
            " For typical fingerprint sizes (e.g. 2048, 1024), it is not required",
            rich_help_panel="Advanced",
        ),
    ] = None,
    input_is_packed: Annotated[
        bool,
        Option(
            "--packed-input/--unpacked-input",
            help="Toggle whether the input consists on packed or unpacked fingerprints",
            rich_help_panel="Advanced",
        ),
    ] = True,
    # Debug options
    monitor_rss: Annotated[
        bool,
        Option(
            "--monitor-mem/--no-monitor-mem",
            "--monitor-rss/--no-monitor-rss",
            help="Monitor RAM used by all processes",
            rich_help_panel="Advanced",
        ),
    ] = True,
    monitor_rss_interval_s: Annotated[
        float,
        Option(
            "--monitor-mem-seconds",
            "--monitor-rss-seconds",
            help="Interval in seconds for RAM monitoring",
            rich_help_panel="Debug",
            hidden=True,
        ),
    ] = 1.0,
    max_fps: Annotated[
        int | None,
        Option(
            help="Max. num of fingerprints to read from each file",
            rich_help_panel="Debug",
            hidden=True,
        ),
    ] = None,
    variant: Annotated[
        str,
        Option(
            "--bb-variant",
            help="Use different bitbirch variants, *only for debugging*.",
            hidden=True,
        ),
    ] = "lean",
    copy_inputs: Annotated[
        bool,
        Option(
            "--copy/--no-copy",
            rich_help_panel="Advanced",
            help="Copy the input files instead of symlink",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        Option("-v/-V", "--verbose/--no-verbose"),
    ] = True,
) -> None:
    r"""Run standard, serial BitBIRCH clustering over `*.npy` fingerprint files"""
    # TODO: Remove code duplication with multiround
    from bblean._console import get_console
    from bblean.fingerprints import _get_fps_file_num

    console = get_console(silent=not verbose)
    if variant == "int64" and input_is_packed:
        raise ValueError("Packed inputs are not supported for the int64 variant")
    if refine_rounds is None:
        refine_rounds = 1 if refine_num > 0 else 0
    if refine_rounds > 0 and refine_num == 0:
        refine_num = 1
    ctx.params["refine_rounds"] = refine_rounds
    ctx.params["refine_num"] = refine_num

    BitBirch, set_merge = _import_bitbirch_variant(variant)

    # NOTE: Files are sorted according to name
    if input_ is None:
        input_ = Path.cwd() / "bb_inputs"
        input_.mkdir(exist_ok=True)
        input_files = sorted(input_.glob("*.npy"))
        _validate_input_dir(input_)
    elif input_.is_dir():
        input_files = sorted(input_.glob("*.npy"))
        _validate_input_dir(input_)
    else:
        input_files = [input_]
    ctx.params.pop("input_")
    ctx.params["input_files"] = [str(p.resolve()) for p in input_files]
    ctx.params["num_fps_present"] = [_get_fps_file_num(p) for p in input_files]
    if max_fps is not None:
        ctx.params["num_fps_loaded"] = [
            min(n, max_fps) for n in ctx.params["num_fps_present"]
        ]
    else:
        ctx.params["num_fps_loaded"] = ctx.params["num_fps_present"]
    unique_id = format(random.getrandbits(32), "08x")
    if out_dir is None:
        out_dir = Path.cwd() / "bb_run_outputs" / unique_id
    out_dir.mkdir(exist_ok=True, parents=True)
    _validate_output_dir(out_dir, overwrite)
    ctx.params["out_dir"] = str(out_dir.resolve())

    console.print_banner()
    console.print()
    console.print_config(ctx.params)

    # Optinally start a separate process that tracks RAM usage
    if monitor_rss:
        launch_monitor_rss_daemon(out_dir / "monitor-rss.csv", monitor_rss_interval_s)

    timer = Timer()
    timer.init_timing("total")
    if "lean" not in variant:
        set_merge(merge_criterion, tolerance=tolerance)
        tree = BitBirch(branching_factor=branching_factor, threshold=threshold)
    else:
        tree = BitBirch(
            branching_factor=branching_factor,
            threshold=threshold,
            merge_criterion=merge_criterion,
            tolerance=tolerance,
        )
    with console.status("[italic]BitBirching...[/italic]", spinner="dots"):
        for file in input_files:
            # Fitting a file uses mmap internally, and releases memory in a smart way
            tree.fit(
                file,
                n_features=n_features,
                input_is_packed=input_is_packed,
                max_fps=max_fps,
            )
    if recluster_rounds != 0 or refine_rounds != 0:
        tree.set_merge(
            refine_merge_criterion,
            tolerance=tolerance,
            threshold=threshold + refine_threshold_change,
        )
        for r in range(refine_rounds):
            msg = (
                f"[italic]Refinement, round {r + 1}"
                f" (will split {refine_num} largest clusters)...[/italic]"
            )
            with console.status(msg, spinner="dots"):
                tree.refine_inplace(
                    input_files,
                    input_is_packed=input_is_packed,
                    n_largest=refine_num,
                )
        for r in range(recluster_rounds):
            msg = f"[italic]Reclustering, round {r + 1}...[/italic]"
            with console.status(msg, spinner="dots"):
                tree.recluster_inplace(shuffle=recluster_shuffle)

    timer.end_timing("total", console, indent=False)
    console.print_peak_mem(out_dir, indent=False)
    if variant == "lean":
        if save_tree:
            # TODO: BitBIRCH is highly recursive. pickling may crash python,
            # an alternative solution would be better
            _old_limit = sys.getrecursionlimit()
            sys.setrecursionlimit(100_000)
            with open(out_dir / "bitbirch.pkl", mode="wb") as f:
                pickle.dump(tree, f)
            sys.setrecursionlimit(_old_limit)
        tree.delete_internal_nodes()
        # Dump outputs (peak memory, timings, config, cluster ids)
        if save_centroids:
            output = tree.get_centroids_mol_ids()
            with open(out_dir / "clusters.pkl", mode="wb") as f:
                pickle.dump(output["mol_ids"], f)
            with open(out_dir / "cluster-centroids-packed.pkl", mode="wb") as f:
                pickle.dump(output["centroids"], f)
        else:
            with open(out_dir / "clusters.pkl", mode="wb") as f:
                pickle.dump(tree.get_cluster_mol_ids(), f)

    collect_system_specs_and_dump_config(ctx.params)
    timer.dump(out_dir / "timings.json")

    # Symlink or copy fingerprint files
    input_fps_dir = (out_dir / "input-fps").resolve()
    input_fps_dir.mkdir()

    input_smiles_dir = (out_dir / "input-smiles").resolve()
    input_smiles_dir.mkdir()
    if copy_inputs:
        for file in input_files:
            shutil.copy(file, input_fps_dir / file.name)
        if smiles_path:
            shutil.copy(smiles_path, input_smiles_dir / smiles_path.name)
    else:
        for file in input_files:
            (input_fps_dir / file.name).symlink_to(file.resolve())
        if smiles_path:
            (input_smiles_dir / smiles_path.name).symlink_to(smiles_path.resolve())


# TODO: Currently sometimes after a round is triggered *more* files are output, since
# the files are divided *both* by uint8/uint16 and the batch idx. I believe this is not
# ideal
@app.command("multiround")
def _multiround(
    ctx: Context,
    in_dir: Annotated[
        Path | None,
        Argument(help="Directory with input `*.npy` files with packed fingerprints"),
    ] = None,
    out_dir: Annotated[
        Path | None,
        Option("-o", "--out-dir", help="Dir for output files"),
    ] = None,
    overwrite: Annotated[bool, Option(help="Allow overwriting output files")] = False,
    num_initial_processes: Annotated[
        int, Option("--ps", "--processes", help="Num. processes for first round")
    ] = 10,
    num_midsection_processes: Annotated[
        int | None,
        Option(
            "--mid-ps",
            "--mid-processes",
            help="Num. processes for middle section rounds."
            " These are be memory intensive,"
            " you may want to use 50%-30% of --ps."
            " Default is same as --ps",
        ),
    ] = None,
    branching_factor: Annotated[
        int,
        Option(
            "--branching",
            "-b",
            help="BitBIRCH branching factor (all rounds). Usually 254 is"
            " optimal. Set above 254 for slightly less RAM (at the cost of some perf.)",
        ),
    ] = DEFAULTS.branching_factor,
    threshold: Annotated[
        float,
        Option("--threshold", "-t", help="Thresh for merge criterion (initial step)"),
    ] = DEFAULTS.threshold,
    mid_threshold_change: Annotated[
        float,
        Option("--mid-threshold-change", help="Modify threshold for refinement"),
    ] = DEFAULTS.refine_threshold_change,
    initial_merge_criterion: Annotated[
        str,
        Option(
            "--set-merge",
            "-m",
            help="Initial merge criterion for round 1. ('diameter' recommended)",
        ),
    ] = DEFAULTS.merge_criterion,
    save_tree: Annotated[
        bool,
        Option("--save-tree/--no-save-tree", rich_help_panel="Advanced"),
    ] = False,
    save_centroids: Annotated[
        bool,
        Option("--save-centroids/--no-save-centroids", rich_help_panel="Advanced"),
    ] = True,
    mid_merge_criterion: Annotated[
        str,
        Option(
            "--set-mid-merge",
            help="Merge criterion for midsection rounds ('diameter' recommended)",
        ),
    ] = DEFAULTS.refine_merge_criterion,
    tolerance: Annotated[
        float,
        Option(
            help="Tolerance value for all steps that use the 'tolerance' criterion"
            " (by default all except initial round)",
        ),
    ] = DEFAULTS.tolerance,
    n_features: Annotated[
        int | None,
        Option(
            "--n-features",
            help="Number of features in the fingerprints."
            " Only for packed inputs *if it is not a multiple of 8*."
            " Not required for typical fingerprint sizes (e.g. 2048, 1024)",
            rich_help_panel="Advanced",
        ),
    ] = None,
    input_is_packed: Annotated[
        bool,
        Option(
            "--packed-input/--unpacked-input",
            help="Toggle whether the input consists on packed or unpacked fingerprints",
            rich_help_panel="Advanced",
        ),
    ] = True,
    # Advanced options
    num_midsection_rounds: Annotated[
        int,
        Option(
            "--num-mid-rounds",
            help="Number of midsection rounds to perform",
            rich_help_panel="Advanced",
        ),
    ] = 1,
    split_largest_after_midsection: Annotated[
        bool,
        Option(
            "--split-after-mid/--no-split-after-mid",
            help=(
                "Split largest cluster after each midsection round"
                " (to be refined by the next round)"
            ),
            rich_help_panel="Advanced",
        ),
    ] = False,
    refinement_before_midsection: Annotated[
        str,
        Option(
            "--initial-refine",
            help=(
                "Run a *full* refinement step after the initial clustering round,"
                " only *split* largest cluster, or do *none*."
            ),
            rich_help_panel="Advanced",
        ),
    ] = "full",
    max_tasks_per_process: Annotated[
        int, Option(help="Max tasks per process", rich_help_panel="Advanced")
    ] = 1,
    fork: Annotated[
        bool,
        Option(
            help="In linux, force the 'fork' multiprocessing start method",
            rich_help_panel="Advanced",
        ),
    ] = False,
    bin_size: Annotated[
        int,
        Option(help="Bin size for chunking during Round 2", rich_help_panel="Advanced"),
    ] = 10,
    # Debug options
    variant: Annotated[
        str,
        Option(
            "--bb-variant",
            help="Use different bitbirch variants, *only for debugging*.",
            hidden=True,
        ),
    ] = "lean",
    monitor_rss: Annotated[
        bool,
        Option(
            "--monitor-mem",
            "--monitor-rss",
            help="Monitor RAM used by all processes",
            rich_help_panel="Advanced",
        ),
    ] = True,
    monitor_rss_interval_s: Annotated[
        float,
        Option(
            "--monitor-mem-seconds",
            "--monitor-rss-seconds",
            help="Interval in seconds for RAM monitoring",
            rich_help_panel="Debug",
            hidden=True,
        ),
    ] = 1.0,
    max_fps: Annotated[
        int | None,
        Option(
            help="Max num. of fps to load from each input file",
            rich_help_panel="Debug",
            hidden=True,
        ),
    ] = None,
    max_files: Annotated[
        int | None,
        Option(help="Max num. files to read", rich_help_panel="Debug", hidden=True),
    ] = None,
    copy_inputs: Annotated[
        bool,
        Option(
            "--copy/--no-copy",
            rich_help_panel="Advanced",
            help="Copy the input files instead of symlink",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        Option("-v/-V", "--verbose/--no-verbose"),
    ] = True,
    cleanup: Annotated[
        bool,
        Option("--cleanup/--no-cleanup", hidden=True),
    ] = True,
) -> None:
    r"""Run multi-round BitBIRCH clustering, optionally parallelize over `*.npy` files"""  # noqa:E501
    from bblean._console import get_console
    from bblean.multiround import run_multiround_bitbirch
    from bblean.fingerprints import _get_fps_file_num

    console = get_console(silent=not verbose)

    # Set multiprocessing start method
    if fork and not sys.platform == "linux":
        console.print("'fork' is only available on Linux", style="red")
        raise Abort()
    if sys.platform == "linux":
        mp_context = mp.get_context("fork" if fork else "forkserver")
    else:
        mp_context = mp.get_context()

    # Collect inputs:
    # If not passed, input dir is bb_inputs/
    if in_dir is None:
        in_dir = Path.cwd() / "bb_inputs"
    _validate_input_dir(in_dir)
    # All files in the input dir with *.npy suffix are considered input files
    input_files = sorted(in_dir.glob("*.npy"))[:max_files]
    ctx.params["input_files"] = [str(p.resolve()) for p in input_files]
    ctx.params["num_fps"] = [_get_fps_file_num(p) for p in input_files]
    if max_fps is not None:
        ctx.params["num_fps_loaded"] = [min(n, max_fps) for n in ctx.params["num_fps"]]
    else:
        ctx.params["num_fps_loaded"] = ctx.params["num_fps"]

    # Set up outputs:
    # If not passed, output dir is constructed as bb_multiround_outputs/<unique-id>/
    unique_id = format(random.getrandbits(32), "08x")
    if out_dir is None:
        out_dir = Path.cwd() / "bb_multiround_outputs" / unique_id
    out_dir.mkdir(exist_ok=True, parents=True)
    _validate_output_dir(out_dir, overwrite)
    ctx.params["out_dir"] = str(out_dir.resolve())

    console.print_banner()
    console.print()
    console.print_multiround_config(ctx.params, mp_context)

    # Optinally start a separate process that tracks RAM usage
    if monitor_rss:
        launch_monitor_rss_daemon(out_dir / "monitor-rss.csv", monitor_rss_interval_s)

    timer = run_multiround_bitbirch(
        input_files=input_files,
        n_features=n_features,
        input_is_packed=input_is_packed,
        out_dir=out_dir,
        initial_merge_criterion=initial_merge_criterion,
        midsection_merge_criterion=mid_merge_criterion,
        num_initial_processes=num_initial_processes,
        num_midsection_processes=num_midsection_processes,
        branching_factor=branching_factor,
        threshold=threshold,
        midsection_threshold_change=mid_threshold_change,
        tolerance=tolerance,
        # Advanced
        save_tree=save_tree,
        save_centroids=save_centroids,
        bin_size=bin_size,
        max_tasks_per_process=max_tasks_per_process,
        refinement_before_midsection=refinement_before_midsection,
        num_midsection_rounds=num_midsection_rounds,
        split_largest_after_each_midsection_round=split_largest_after_midsection,
        # Debug
        max_fps=max_fps,
        verbose=verbose,
        mp_context=mp_context,
        cleanup=cleanup,
    )
    timer.dump(out_dir / "timings.json")
    # TODO: Also dump peak-rss.json
    collect_system_specs_and_dump_config(ctx.params)

    # Symlink or copy fingerprint files
    input_fps_dir = (out_dir / "input-fps").resolve()
    input_fps_dir.mkdir()
    if copy_inputs:
        for file in input_files:
            shutil.copy(file, input_fps_dir / file.name)
    else:
        for file in input_files:
            (input_fps_dir / file.name).symlink_to(file.resolve())


@app.command("fps-info", rich_help_panel="Fingerprints")
def _fps_info(
    fp_paths: Annotated[
        list[Path] | None,
        Argument(show_default=False, help="Paths to *.smi files with smiles"),
    ] = None,
) -> None:
    """Show info about a `*.npy` fingerprint file, or a dir with `*.npy` files"""
    from bblean._console import get_console
    from bblean.fingerprints import _print_fps_file_info

    console = get_console()
    if fp_paths is None:
        fp_paths = [Path.cwd()]

    for path in fp_paths:
        if path.is_dir():
            for file in path.glob("*.npy"):
                _print_fps_file_info(file, console)
        elif path.suffix == ".npy":
            _print_fps_file_info(file, console)


@app.command("fps-from-smiles", rich_help_panel="Fingerprints")
def _fps_from_smiles(
    smiles_paths: Annotated[
        list[Path] | None,
        Argument(show_default=False, help="Paths to *.smi files with smiles"),
    ] = None,
    out_dir: Annotated[
        Path | None,
        Option("-o", "--out-dir", show_default=False),
    ] = None,
    out_name: Annotated[
        str | None,
        Option("--name", help="Base name of output file"),
    ] = None,
    kind: Annotated[
        str,
        Option("-k", "--kind"),
    ] = DEFAULTS.fp_kind,
    fp_size: Annotated[
        int,
        Option("--n-features", help="Num. features of the generated fingerprints"),
    ] = DEFAULTS.n_features,
    parts: Annotated[
        int | None,
        Option(
            "-n", "--num-parts", help="Split the created file into this number of parts"
        ),
    ] = None,
    max_fps_per_file: Annotated[
        int | None,
        Option(
            "-m",
            "--max-fps-per-file",
            help="Max. number of fps per file. Mutually exclusive with --num-parts",
            show_default=False,
        ),
    ] = None,
    pack: Annotated[
        bool,
        Option(
            "-p/-P",
            "--pack/--no-pack",
            help="Pack bits in last dimension of fingerprints",
            rich_help_panel="Advanced",
        ),
    ] = True,
    dtype: Annotated[
        str,
        Option(
            "-d",
            "--dtype",
            help="NumPy dtype for the generated fingerprints",
            rich_help_panel="Advanced",
        ),
    ] = "uint8",
    verbose: Annotated[
        bool,
        Option("-v/-V", "--verbose/--no-verbose"),
    ] = True,
    num_ps: Annotated[
        int | None,
        Option(
            "--ps",
            "--processes",
            help=(
                "Num. processes for multprocess generation."
                " One process per file is used for multi-file generation"
            ),
        ),
    ] = None,
    sanitize: Annotated[
        str,
        Option(
            "--sanitize",
            help="RDKit sanitization operations to perform ('all' or 'minimal')",
        ),
    ] = "all",
    skip_invalid: Annotated[
        bool,
        Option(
            "--skip-invalid/--no-skip-invalid",
            help=(
                "Skip invalid smiles."
                " If False, an error is raised on invalid smiles. If True they are"
                " silently skipped (this is be more memory intensive, especially for"
                " parallel processing)"
            ),
        ),
    ] = False,
) -> None:
    r"""Generate a `*.npy` fingerprints file from one or more `*.smi` smiles files

    By default this function runs in parallel and uses all available CPUs. In order to
    use the memory efficient BitBIRCH u8 algorithm you should keep the defaults:
    --dtype=uint8 and --pack
    """
    import numpy as np

    from bblean._console import get_console
    from bblean.utils import _num_avail_cpus
    from bblean.fingerprints import _FingerprintFileCreator, _FingerprintArrayFiller
    from bblean.smiles import (
        calc_num_smiles,
        _iter_ranges_and_smiles_batches,
        _iter_idxs_and_smiles_batches,
    )

    # Force forkserver since rdkit may use threads, and fork is unsafe with threads
    mp_context = mp.get_context("forkserver" if sys.platform == "linux" else None)

    console = get_console(silent=not verbose)

    if smiles_paths is None:
        smiles_paths = list(Path.cwd().glob("*.smi"))
    if not smiles_paths:
        console.print("No *.smi files found", style="red")
        raise Abort()

    smiles_num = calc_num_smiles(smiles_paths)

    def parse_num_per_batch(
        smiles_num: int, parts: int | None, max_fps_per_file: int | None
    ) -> tuple[int, int, int | None]:
        digits: int | None
        if parts is not None and max_fps_per_file is None:
            num_per_batch = math.ceil(smiles_num / parts)
            digits = len(str(parts))
        elif parts is None and max_fps_per_file is not None:
            num_per_batch = max_fps_per_file
            parts = math.ceil(smiles_num / max_fps_per_file)
            digits = len(str(parts))
        elif parts is None and max_fps_per_file is None:
            parts = 1
            num_per_batch = math.ceil(smiles_num / parts)
            digits = None
        else:
            raise ValueError("parts and max_fps_per_file are mutually exclusive")
        return parts, num_per_batch, digits

    try:
        parts, num_per_batch, digits = parse_num_per_batch(
            smiles_num, parts, max_fps_per_file
        )
    except ValueError:
        console.print(
            "'--max-fps-per-file' and '--num-parts' are mutually exclusive",
            style="red",
        )
        raise Abort() from None
    if out_dir is None:
        out_dir = Path.cwd()
    out_dir.mkdir(exist_ok=True)
    out_dir = out_dir.resolve()

    # Pass 2: build the molecules
    unique_id = format(random.getrandbits(32), "08x")
    if out_name is None:
        # Save the fingerprints as a NumPy array
        out_name = f"{'packed-' if pack else ''}fps-{dtype}-{kind}-{unique_id}"
    else:
        # Strip suffix
        if out_name.endswith(".npy"):
            out_name = out_name[:-4]

    if num_ps is None:
        # Get the number of cores *available for use for this process*
        # bound by the number of parts to avoid spawning useless processes
        if parts == 1:
            num_ps = _num_avail_cpus()
        else:
            num_ps = min(_num_avail_cpus(), parts)
    create_fp_file = _FingerprintFileCreator(
        dtype,
        out_dir,
        out_name,
        digits,
        pack,
        kind,
        fp_size,
        sanitize=sanitize,
        skip_invalid=skip_invalid,
        verbose=verbose,
    )
    timer = Timer()
    timer.init_timing("total")
    if parts > 1 and num_ps is not None and num_ps > 1:
        # Multiprocessing version, 1 process per file
        with console.status(
            f"[italic]Generating fingerprints ({parts} files, parallel, {num_ps} procs.) ...[/italic]",  # noqa:E501
            spinner="dots",
        ):
            with mp_context.Pool(processes=num_ps) as pool:
                pool.map(
                    create_fp_file,
                    _iter_idxs_and_smiles_batches(smiles_paths, num_per_batch),
                )
        timer.end_timing("total", console, indent=False)
        stem = out_name.split(".")[0]
        console.print(f"Finished. Outputs written to {str(out_dir / stem)}.<idx>.npy")
        return

    # Parallel or serial, single file version
    msg = "parallel" if num_ps > 1 else "serial"
    with console.status(
        f"[italic]Generating fingerprints ({parts} files, {msg}, {num_ps} procs.) ...[/italic]",  # noqa:E501
        spinner="dots",
    ):
        if pack:
            out_dim = (fp_size + 7) // 8
        else:
            out_dim = fp_size
        shmem_size = smiles_num * out_dim * np.dtype(dtype).itemsize
        fps_shmem = shmem.SharedMemory(create=True, size=shmem_size)
        invalid_mask_shmem = shmem.SharedMemory(create=True, size=smiles_num)
        fps_array_filler = _FingerprintArrayFiller(
            shmem_name=fps_shmem.name,
            invalid_mask_shmem_name=invalid_mask_shmem.name,
            kind=kind,
            fp_size=fp_size,
            num_smiles=smiles_num,
            dtype=dtype,
            pack=pack,
            sanitize=sanitize,
            skip_invalid=skip_invalid,
        )
        if num_ps > 1 and parts == 1:
            # Split into batches anyways if we have a single batch but multiple
            # processes
            _, num_per_batch, _ = parse_num_per_batch(
                smiles_num, num_ps, max_fps_per_file
            )
        with mp_context.Pool(processes=num_ps) as pool:
            pool.starmap(
                fps_array_filler,
                _iter_ranges_and_smiles_batches(smiles_paths, num_per_batch),
            )
        fps = np.ndarray((smiles_num, out_dim), dtype=dtype, buffer=fps_shmem.buf)
        mask = np.ndarray((smiles_num,), dtype=np.bool, buffer=invalid_mask_shmem.buf)
        if skip_invalid:
            prev_num = len(fps)
            fps = np.delete(fps, mask, axis=0)
            new_num = len(fps)
            console.print(f"Generated {new_num} fingerprints")
            console.print(f"Skipped {prev_num - new_num} invalid smiles")
            invalid_name = f"invalid-{unique_id}.npy"
            console.print(
                f"Invalid smiles idxs written to {str(out_dir / invalid_name)}"
            )
            np.save(out_dir / f"invalid-{unique_id}.npy", mask.nonzero()[0].reshape(-1))

        np.save(
            out_dir / out_name,
            fps,
        )
        del mask
        del fps
        # Cleanup
        fps_shmem.unlink()
        invalid_mask_shmem.unlink()
    timer.end_timing("total", console, indent=False)
    console.print(f"Finished. Outputs written to {str(out_dir / out_name)}.npy")


@app.command("fps-split", rich_help_panel="Fingerprints")
def _split_fps(
    input_: Annotated[
        Path,
        Argument(help="`*.npy` file with fingerprints"),
    ],
    out_dir: Annotated[
        Path | None,
        Option("-o", "--out-dir", show_default=False),
    ] = None,
    parts: Annotated[
        int | None,
        Option(
            "-n",
            "--num-parts",
            help="Num. of parts to split file into. Mutually exclusive with --max-fps",
            show_default=False,
        ),
    ] = None,
    max_fps_per_file: Annotated[
        int | None,
        Option(
            "-m",
            "--max-fps",
            help="Max. number of fps per file. Mutually exclusive with --num-parts",
            show_default=False,
        ),
    ] = None,
) -> None:
    r"""Split a `*.npy` fingerprint file into multiple `*.npy` files

    Usage to split into multiple files with a max number of fps each (e.g. 10k) is `bb
    split-fps --max-fps 10_000 ./fps.npy --out-dir ./split`. To split into a pre-defined
    number of parts (e.g. 10) `bb split-fps --num-parts 10 ./fps.npy --out-dir ./split`.
    """
    from bblean._console import get_console
    import numpy as np

    console = get_console()
    if parts is not None and parts < 2:
        console.print("Num must be >= 2", style="red")
        raise Abort()
    fps = np.load(input_, mmap_mode="r")
    if parts is not None and max_fps_per_file is None:
        num_per_batch = math.ceil(fps.shape[0] / parts)
        digits = len(str(parts))
    elif parts is None and max_fps_per_file is not None:
        num_per_batch = max_fps_per_file
        digits = len(str(math.ceil(fps.shape[0] / max_fps_per_file)))
    else:
        console.print(
            "One and only one of '--max-fps' and '--num-parts' required", style="red"
        )
        raise Abort()

    stem = input_.name.split(".")[0]
    with console.status("[italic]Splitting fingerprints...[/italic]", spinner="dots"):
        i = -1
        for i, batch in enumerate(batched(fps, num_per_batch)):
            suffixes = input_.suffixes
            name = f"{stem}{''.join(suffixes[:-1])}.{str(i).zfill(digits)}.npy"

            # Generate out dir when first fp file is being saved
            if out_dir is None:
                out_dir = Path.cwd() / stem
            out_dir.mkdir(exist_ok=True)
            out_dir = out_dir.resolve()

            np.save(out_dir / name, batch)

        if i == -1:
            console.print("Warning: No fingerprints written", style="yellow")
            return
    console.print(
        f"Finished. Outputs written to {str(tp.cast(Path, out_dir) / stem)}.<idx>.npy"
    )


@app.command("query-idx", hidden=True)
def _query_idx(
    idx_path: Annotated[
        Path,
        Argument(help="Path to the index dir, or its parent dir with index files"),
    ],
    query: Annotated[
        str,
        Option("-s", "--smiles"),
    ] = "",
    k: Annotated[
        int,
        Option("-k", "--num"),
    ] = 10,
    threshold: Annotated[
        float,
        Option("-t", "--threshold"),
    ] = 0.0,
    num_probe: Annotated[
        int,
        Option("-p", "--probe"),
    ] = 1,
) -> None:
    from bblean._console import get_console
    from bblean.fingerprints import fps_from_smiles
    from bblean.ivf import IVFIndex

    console = get_console()
    if idx_path.is_dir() and (idx_path / "index").is_dir():
        idx_path = idx_path / "index"
    else:
        idx_path = idx_path

    index = IVFIndex.from_dir(idx_path)
    # TODO: This is a placeholder, must be modified!!
    query_fp = fps_from_smiles([query], kind="rdkit", n_features=2048, pack=True)[0]
    results_list = index.search(query_fp, n_probe=num_probe, threshold=threshold, k=k)

    for r in results_list:
        console.print(r)


def _get_centroids_and_clusters_paths(clusters_path: Path) -> tuple[Path, Path]:
    if clusters_path.is_dir():
        centroids_path = clusters_path / "cluster-centroids-packed.pkl"
        clusters_path = clusters_path / "clusters.pkl"
        # TODO: If this file doesn't exist, it will have to be created on-the-fly
        if not centroids_path.exists():
            raise ValueError(
                "Centroids must be saved for building IVF or global clustering."
                " This limitation may be lifted in the future"
            )
    else:
        raise ValueError(
            "clusters path must be a dir for building IVF or global clustering."
            " This limitation may be lifted in the future"
        )
    return clusters_path, centroids_path


@app.command("build-idx", hidden=True)
def _build_idx(
    clusters_path: Annotated[
        Path,
        Argument(help="Path to the clusters file, or a dir with a clusters.pkl file"),
    ],
    fps_path: Annotated[
        Path | None,
        Option(
            "-f",
            "--fps-path",
            help="Path to fingerprint file, or directory with fingerprint files",
            show_default=False,
        ),
    ] = None,
    smiles_path: Annotated[
        Path | None,
        Option(
            "-s",
            "--smiles-path",
            show_default=False,
            help="Optional smiles path, if used smiles are returned in search results",
        ),
    ] = None,
    method: Annotated[
        str,
        Option("--method"),
    ] = "kmeans",
    n_clusters: Annotated[
        int | None,
        Option("--n-clusters"),
    ] = None,
    verbose: Annotated[
        bool,
        Option("--verbose/--no-verbose"),
    ] = True,
) -> None:
    import numpy as np
    from bblean.utils import _has_files_or_valid_symlinks
    from bblean.ivf import IVFIndex
    from bblean.smiles import load_smiles
    from bblean._console import get_console

    console = get_console(silent=not verbose)
    clusters_path, centroids_path = _get_centroids_and_clusters_paths(clusters_path)
    with open(clusters_path, mode="rb") as f:
        cluster_members = pickle.load(f)

    with open(centroids_path, mode="rb") as f:
        centroids_packed = np.vstack(pickle.load(f))

    if fps_path is None:
        input_fps_path = clusters_path.parent / "input-fps"
        if input_fps_path.is_dir() and _has_files_or_valid_symlinks(input_fps_path):
            fps_path = input_fps_path
        else:
            msg = (
                "Could not find input fingerprints. Please use --fps-path."
                " Summary plot without fingerprints doesn't include isim values"
            )
            warnings.warn(msg)
    if smiles_path is None:
        input_smiles_path = clusters_path.parent / "input-smiles"
        if input_smiles_path.is_dir() and _has_files_or_valid_symlinks(
            input_smiles_path
        ):
            smiles_paths = sorted(input_smiles_path.glob("*.smi"))
            if len(smiles_paths) > 1:
                raise ValueError("Currently only a single smiles file is supported")
            smiles_path = smiles_paths[0]
        else:
            msg = (
                "Could not find input smiles. Please use --smiles-path."
                " Search results won't include smiles"
            )
            warnings.warn(msg)

    if fps_path is None:
        raise ValueError("Fingerprints are required to build the index")
    elif fps_path.is_dir():
        fps_paths = sorted(fps_path.glob("*.npy"))
    else:
        fps_paths = [fps_path]
    if len(fps_paths) > 1:
        raise ValueError(
            "Currently only a single fp file is supported,"
            " this restriction will be lifted in the future"
        )
    fps_path = fps_paths[0]

    fps = np.load(fps_path)  # packed
    kwargs = {"random_state": 42} if method.startswith("kmeans") else {}

    with console.status("[italic]Building IVF index...[/italic]", spinner="dots"):
        index = IVFIndex.from_bitbirch_clusters(
            cluster_members,
            centroids_packed,
            fps,
            load_smiles(smiles_path) if smiles_path is not None else (),
            method,
            n_clusters,
            input_is_packed=True,
            sort=True,
            **kwargs,
        )

    with console.status("[italic]Saving global clusters...[/italic]", spinner="dots"):
        global_cluster_medoids_path = (
            clusters_path.parent / "global-cluster-medoids-packed.npy"
        )
        np.save(
            global_cluster_medoids_path,
            index._medoids_packed,
        )
        global_clusters_path = clusters_path.parent / "global-clusters.pkl"
        with open(global_clusters_path, mode="wb") as f:
            pickle.dump(index._members, f)
        idx_path = clusters_path.parent / "index"
        if idx_path.exists():
            shutil.rmtree(idx_path)
        idx_path.mkdir(exist_ok=True)
        if smiles_path is not None:
            (idx_path / "smiles.smi").symlink_to(smiles_path.resolve())
        (idx_path / "fps.npy").symlink_to(fps_path.resolve())
        (idx_path / "global-clusters.pkl").symlink_to(global_clusters_path.resolve())
        (idx_path / "global-cluster-medoids-packed.npy").symlink_to(
            global_cluster_medoids_path.resolve()
        )
    console.print("Successfully built IVF index")


@app.command("fps-shuffle", rich_help_panel="Fingerprints")
def _shuffle_fps(
    in_file: Annotated[
        Path,
        Argument(help="`*.npy` file with packed fingerprints"),
    ],
    out_dir: Annotated[
        Path | None,
        Option("-o", "--out-dir", show_default=False),
    ] = None,
    seed: Annotated[
        int | None,
        Option("--seed", hidden=True, rich_help_panel="Debug"),
    ] = None,
) -> None:
    """Shuffle a fingerprints file

    This function is not optimized and as such may have high RAM usage. It is
    meant for testing purposes only"""
    import numpy as np

    fps = np.load(in_file)
    stem = in_file.stem
    rng = np.random.default_rng(seed)
    rng.shuffle(fps, axis=0)
    if out_dir is None:
        out_dir = Path.cwd()
    out_dir.mkdir(exist_ok=True)
    out_dir = out_dir.resolve()
    np.save(out_dir / f"shuffled-{stem}.npy", fps)


@app.command("fps-merge", rich_help_panel="Fingerprints")
def _merge_fps(
    in_dir: Annotated[
        Path,
        Argument(help="Directory with input `*.npy` files with packed fingerprints"),
    ],
    out_dir: Annotated[
        Path | None,
        Option("-o", "--out-dir", show_default=False),
    ] = None,
) -> None:
    r"""Merge a dir with multiple `*.npy` fingerprint file into a single `*.npy` file"""
    from bblean._console import get_console
    import numpy as np

    console = get_console()

    if out_dir is None:
        out_dir = Path.cwd()
    out_dir.mkdir(exist_ok=True)
    out_dir = out_dir.resolve()
    arrays = []
    with console.status("[italic]Merging fingerprints...[/italic]", spinner="dots"):
        stem = None
        for f in sorted(in_dir.glob("*.npy")):
            if stem is None:
                stem = f.name.split(".")[0]
            elif stem != f.name.split(".")[0]:
                raise ValueError(
                    "Name convention must be <name>.<idx>.npy"
                    " with all files having the same <name>"
                )
            arrays.append(np.load(f))
        if stem is None:
            console.print("No *.npy files found")
            return
        np.save(out_dir / stem, np.concatenate(arrays))
    console.print(f"Finished. Outputs written to {str(out_dir / stem)}.npy")
