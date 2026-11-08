#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "h5py",
#   "matplotlib",
#   "numpy",
# ]
# ///
"""Load and plot a COAST .observe.h5 capture file.

Usage:
    scripts/load_observe.py <file.observe.h5>                         # list mode
    scripts/load_observe.py <file.observe.h5> --plot                  # sample plot per category
    scripts/load_observe.py <file.observe.h5> --plot <site>           # plot one exact site
    scripts/load_observe.py <file.observe.h5> --plot-category activity \
        --scale category                                              # stable batch scale

Handles both writer-side HDF5 layouts:
  (A) Flat datasets:   '/neuron[42]__dendrite[3]__activity'
                       '/neuron[42]__dendrite[3]__activity__step_index'
  (B) Nested groups:   '/neuron[42]/dendrite[3]/activity'
                       '/neuron[42]/dendrite[3]/activity__step_index'

A 'site' is identified by the full HDF5 path of its value dataset.
Step-index companions end in '__step_index'.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    import h5py
    import numpy as np
except ImportError as e:
    missing = e.name or str(e)
    sys.stderr.write(
        f"error: missing required Python module '{missing}'.\n"
        f"hint:  pip install h5py numpy\n"
    )
    sys.exit(2)


def _import_matplotlib():
    """Lazy import — only needed for --plot. Listing works without it."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless-safe; must precede pyplot import
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:
        missing = e.name or str(e)
        sys.stderr.write(
            f"error: --plot requires matplotlib (missing '{missing}').\n"
            f"hint:  pip install matplotlib\n"
        )
        sys.exit(2)


_STEP_SUFFIX = "__step_index"
_AUTO_SITE = "__auto__"
_CATEGORY_VALUE_LIMITS = {
    "activity": (0.0, 1.0),
    "attenuation": (0.0, 1.0),
    "out": (0.0, 1.0),
    "segment_activity": (0.0, 1.0),
}


def _is_step_dataset(name: str) -> bool:
    """True if dataset name is a step-index companion."""
    return name.endswith(_STEP_SUFFIX)


def _category_name(site_path: str) -> str:
    """Return the metric/category name for flat or nested observe paths."""
    leaf = site_path.rsplit("/", 1)[-1]
    if "__" in leaf:
        return leaf.rsplit("__", 1)[-1]
    return leaf


def discover_sites(h5_file: h5py.File) -> list[str]:
    """Return paths of all value datasets (excluding step-index siblings)."""
    sites: list[str] = []

    def visitor(name: str, obj: object) -> None:
        if isinstance(obj, h5py.Dataset) and not _is_step_dataset(name):
            sites.append(name)

    h5_file.visititems(visitor)
    return sorted(sites)


def find_step_dataset(h5_file: h5py.File, site_path: str) -> h5py.Dataset | None:
    """Return the step-index dataset paired with site_path, or None."""
    candidate = site_path + _STEP_SUFFIX
    if candidate in h5_file:
        return h5_file[candidate]
    return None


def list_sites(h5_path: Path) -> int:
    """List mode: print every site with shape and row count."""
    with h5py.File(h5_path, "r") as f:
        sites = discover_sites(f)
        for site in sites:
            ds = f[site]
            row_shape = ds.shape[1:] if ds.ndim >= 1 else ()
            n_rows = ds.shape[0] if ds.ndim >= 1 else 0
            print(f"{site}  shape={row_shape}  rows={n_rows}")
            if find_step_dataset(f, site) is None:
                sys.stderr.write(
                    f"warning: site '{site}' has no step-index sibling\n"
                )
        print(f"-- {len(sites)} site(s) --")
    return 0


def _sanitise(site_path: str) -> str:
    return site_path.replace("/", "__")


def _plot_output_path(
    h5_path: Path,
    site: str,
    output_dir: Path | None,
) -> Path:
    parent = h5_path.parent if output_dir is None else output_dir
    parent.mkdir(parents=True, exist_ok=True)
    return parent.joinpath(f"{h5_path.name}.{_sanitise(site)}.png").resolve()


def _category_value_limits(site: str) -> tuple[float, float] | None:
    return _CATEGORY_VALUE_LIMITS.get(_category_name(site))


def _is_count_category(site: str) -> bool:
    return _category_name(site).endswith("count")


def _scalar_ylim(
    site: str,
    scale_mode: str,
    explicit_ylim: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if explicit_ylim is not None:
        return explicit_ylim
    if scale_mode == "category":
        return _category_value_limits(site)
    return None


def _heatmap_clim(
    site: str,
    scale_mode: str,
    explicit_clim: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if explicit_clim is not None:
        return explicit_clim
    if scale_mode == "category":
        return _category_value_limits(site)
    return None


def plot_site(
    h5_path: Path,
    site: str,
    scale_mode: str,
    explicit_ylim: tuple[float, float] | None,
    explicit_clim: tuple[float, float] | None,
    output_dir: Path | None,
) -> int:
    """Plot mode: render <site> as PNG next to input file.

    Dispatches on the site's per-row shape:
      - scalar       (shape=())     -> line plot of value vs. step
      - 1-D vector   (shape=(N,))   -> heatmap of value over (step, index)
      - higher rank                 -> error
    """
    plt = _import_matplotlib()
    with h5py.File(h5_path, "r") as f:
        if site not in f:
            sys.stderr.write(f"error: site '{site}' not found in {h5_path}\n")
            return 1

        return _plot_site_in_file(
            plt, h5_path, f, site, scale_mode, explicit_ylim, explicit_clim,
            output_dir,
        )


def plot_category_samples(
    h5_path: Path,
    scale_mode: str,
    explicit_ylim: tuple[float, float] | None,
    explicit_clim: tuple[float, float] | None,
    output_dir: Path | None,
) -> int:
    """Plot one representative plottable site for each metric/category."""
    plt = _import_matplotlib()
    with h5py.File(h5_path, "r") as f:
        samples: dict[str, str] = {}
        for site in discover_sites(f):
            ds = f[site]
            row_shape = ds.shape[1:] if ds.ndim >= 1 else ()
            if row_shape == () or len(row_shape) == 1:
                samples.setdefault(_category_name(site), site)

        if not samples:
            sys.stderr.write("error: no plottable sites found in file\n")
            return 1

        failures = 0
        for category, site in sorted(samples.items()):
            print(f"{category}: {site}")
            failures += _plot_site_in_file(
                plt, h5_path, f, site, scale_mode, explicit_ylim, explicit_clim,
                output_dir,
            )
        plotted = len(samples) - failures
        print(f"-- plotted {plotted}/{len(samples)} category sample(s) --")
        return 1 if failures else 0


def plot_category(
    h5_path: Path,
    category: str,
    scale_mode: str,
    explicit_ylim: tuple[float, float] | None,
    explicit_clim: tuple[float, float] | None,
    output_dir: Path | None,
) -> int:
    """Plot every site whose metric/category name matches category."""
    plt = _import_matplotlib()
    with h5py.File(h5_path, "r") as f:
        sites = [
            site for site in discover_sites(f)
            if _category_name(site) == category
        ]
        if not sites:
            sys.stderr.write(f"error: no sites found for category '{category}'\n")
            return 1

        failures = 0
        for site in sites:
            failures += _plot_site_in_file(
                plt, h5_path, f, site, scale_mode, explicit_ylim, explicit_clim,
                output_dir,
            )
        plotted = len(sites) - failures
        print(f"-- plotted {plotted}/{len(sites)} site(s) for category '{category}' --")
        return 1 if failures else 0


def _plot_site_in_file(
    plt,
    h5_path: Path,
    f,
    site: str,
    scale_mode: str,
    explicit_ylim: tuple[float, float] | None,
    explicit_clim: tuple[float, float] | None,
    output_dir: Path | None,
) -> int:
    ds = f[site]
    row_shape = ds.shape[1:] if ds.ndim >= 1 else ()

    if row_shape == ():
        return _plot_scalar(
            plt,
            h5_path,
            f,
            site,
            _scalar_ylim(site, scale_mode, explicit_ylim),
            output_dir,
        )
    if len(row_shape) == 1:
        return _plot_vector_heatmap(
            plt,
            h5_path,
            f,
            site,
            _heatmap_clim(site, scale_mode, explicit_clim),
            output_dir,
        )
    sys.stderr.write(
        f"error: site '{site}' has per-row shape {row_shape} of rank > 1; "
        f"plot mode supports scalar or 1-D vector sites only\n"
    )
    return 1


def _resolve_steps(f, site: str, n: int):
    """Return the step-index dataset's values, or arange(n) with a warning."""
    step_ds = find_step_dataset(f, site)
    if step_ds is not None:
        return step_ds[...]
    sys.stderr.write(
        f"warning: no step-index sibling for '{site}'; using arange index\n"
    )
    return np.arange(n)


def _plot_scalar(
    plt,
    h5_path: Path,
    f,
    site: str,
    ylim: tuple[float, float] | None,
    output_dir: Path | None,
) -> int:
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    ds = f[site]
    values = ds[...]
    steps  = _resolve_steps(f, site, len(values))

    out_path = _plot_output_path(h5_path, site, output_dir)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, values, linewidth=1.0)
    ax.set_xlabel("step")
    ax.set_ylabel(site)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if _is_count_category(site):
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    ax.set_title(f"{h5_path.name} :: {site}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(str(out_path))
    return 0


def _plot_vector_heatmap(
    plt,
    h5_path: Path,
    f,
    site: str,
    clim: tuple[float, float] | None,
    output_dir: Path | None,
) -> int:
    ds = f[site]
    data   = ds[...]                                    # (n_rows, n_cols)
    n_rows, n_cols = data.shape
    if n_rows == 0:
        # A capture that was opened but never received a sample before the
        # session ended leaves a 0-row dataset. steps[-1]/steps[0] below would
        # raise IndexError, so bail out cleanly instead.
        sys.stderr.write(
            f"warning: site '{site}' has 0 rows (no samples captured); skipping plot\n"
        )
        return 1
    steps  = _resolve_steps(f, site, n_rows)

    out_path = _plot_output_path(h5_path, site, output_dir)

    # Width scales with number of columns (the vector dimension), height with
    # rows but clamped — for long captures, aspect='auto' keeps the figure
    # readable rather than producing a 1-pixel-wide strip per neuron.
    fig_w = max(6.0, min(16.0, 0.18 * n_cols + 4.0))
    fig_h = max(4.0, min(12.0, 0.05 * n_rows + 3.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # extent maps array indices to data coordinates so y-axis shows real step
    # numbers, not row indices. origin='upper' so step 0 (or earliest captured
    # step) sits at the top.
    im = ax.imshow(
        data,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        vmin=clim[0] if clim is not None else None,
        vmax=clim[1] if clim is not None else None,
        extent=[-0.5, n_cols - 0.5, int(steps[-1]) + 0.5, int(steps[0]) - 0.5],
    )
    ax.set_xlabel("index along vector")
    ax.set_ylabel("step")
    ax.set_title(f"{h5_path.name} :: {site}   shape={data.shape}  dtype={data.dtype}")
    fig.colorbar(im, ax=ax, label="value")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(str(out_path))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load a COAST .observe.h5 file.")
    parser.add_argument("file", type=Path, help="path to .observe.h5 file")
    parser.add_argument(
        "--plot", nargs="?", const=_AUTO_SITE, default=None,
        metavar="SITE",
        help="plot named site (scalar -> line, 1-D vector -> heatmap); "
             "if SITE omitted, plot one representative sample per category",
    )
    parser.add_argument(
        "--plot-category",
        metavar="CATEGORY",
        help="plot every site whose metric/category name matches CATEGORY "
             "(for example: activity, segment_activity, out)",
    )
    parser.add_argument(
        "--scale",
        choices=("auto", "category"),
        default="auto",
        help="with 'category', use stable value ranges for known bounded "
             "categories (activity/attenuation/segment_activity/out -> 0..1)",
    )
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="fixed y-axis range for scalar line plots",
    )
    parser.add_argument(
        "--clim",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="fixed color range for vector heatmaps",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory for generated PNGs; defaults to the input file directory",
    )
    args = parser.parse_args(argv)

    if not args.file.is_file():
        sys.stderr.write(f"error: file not found: {args.file}\n")
        return 1

    ylim = _validate_range(parser, "--ylim", args.ylim)
    clim = _validate_range(parser, "--clim", args.clim)

    if args.plot is None and args.plot_category is None:
        return list_sites(args.file)
    if args.plot is not None and args.plot_category is not None:
        parser.error("--plot and --plot-category are mutually exclusive")
    if args.plot_category is not None:
        return plot_category(
            args.file, args.plot_category, args.scale, ylim, clim, args.output_dir
        )

    if args.plot == _AUTO_SITE:
        return plot_category_samples(args.file, args.scale, ylim, clim, args.output_dir)

    site = args.plot
    return plot_site(args.file, site, args.scale, ylim, clim, args.output_dir)


def _validate_range(
    parser: argparse.ArgumentParser,
    option_name: str,
    values: list[float] | None,
) -> tuple[float, float] | None:
    if values is None:
        return None
    low, high = values
    if low >= high:
        parser.error(f"{option_name} requires MIN < MAX")
    return low, high


if __name__ == "__main__":
    sys.exit(main())
