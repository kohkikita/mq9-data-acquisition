#!/usr/bin/env python3
"""
compare_props.py  —  Binned Mean ± Std Comparison

Reads all files matching:
    *_combined_event_aligned.csv
from a directory (default: current directory), then compares propeller runs by
binning Force (N) and computing, per bin:

    mean(audio_rms_dbfs) ± std(audio_rms_dbfs)

This produces a much cleaner comparison than plotting every raw sample.

Outputs:
  - PNG plot (default: propeller_comparison_binned_mean_std.png)

Usage:
  python compare_props.py
  python compare_props.py --dir runs
  python compare_props.py --bin_width 1.0
  python compare_props.py --x_min 0 --x_max 30
  python compare_props.py --out_png myplot.png --no-show

Notes:
  - dBFS is relative (not absolute SPL). Comparisons are best when mic/gain/setup
    are consistent across runs.
  - Force binning treats Force as the independent variable and summarizes
    noise distribution within each force range.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------- Discovery / Parsing -------------------------

def find_combined_csvs(search_dir: Path) -> list[Path]:
    """Find all *_combined_event_aligned.csv files in the given directory."""
    return sorted(search_dir.glob("*_combined_event_aligned.csv"))


def infer_label_from_path(path: Path) -> str:
    """
    Infer a readable label from filename.

    Example:
      baseline5foot_2026-01-14_22-02-10_combined_event_aligned.csv
      -> baseline5foot
    """
    stem = path.stem
    suffix = "_combined_event_aligned"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]

    # Optional: strip trailing timestamp chunk: YYYY-MM-DD_HH-MM-SS
    parts = stem.split("_")
    if len(parts) >= 3 and parts[-2].count("-") == 2 and parts[-1].count("-") == 2:
        # If your timestamps are formatted differently, remove or adjust this.
        stem = "_".join(parts[:-2])

    return stem


def load_combined(csv_path: Path) -> pd.DataFrame:
    """
    Load a combined_event_aligned CSV and validate required columns.
    Returns only columns needed for comparison.
    """
    df = pd.read_csv(csv_path)

    required = {"force_N", "audio_rms_dbfs"}
    if not required.issubset(df.columns):
        raise RuntimeError(
            f"{csv_path.name} missing required columns {sorted(required)}. "
            f"Columns present: {list(df.columns)}"
        )

    df = df.copy()
    df["force_N"] = pd.to_numeric(df["force_N"], errors="coerce")
    df["audio_rms_dbfs"] = pd.to_numeric(df["audio_rms_dbfs"], errors="coerce")
    df = df.dropna(subset=["force_N", "audio_rms_dbfs"])

    return df[["force_N", "audio_rms_dbfs"]]


# ------------------------- Binning / Aggregation -------------------------

def binned_stats(
    df: pd.DataFrame,
    *,
    bin_width: float,
    x_min: float | None = None,
    x_max: float | None = None,
    min_count: int = 3,
) -> pd.DataFrame:
    """
    Compute binned mean/std of audio_rms_dbfs as a function of force_N.

    Returns a DataFrame with:
      bin_center, count, mean_db, std_db
    Only bins with count >= min_count are kept.

    Bins are [edge_i, edge_{i+1}) except the last edge.
    """
    if df.empty:
        return pd.DataFrame(columns=["bin_center", "count", "mean_db", "std_db"])

    force = df["force_N"].to_numpy()
    audio = df["audio_rms_dbfs"].to_numpy()

    if x_min is None:
        x_min = float(np.nanmin(force))
    if x_max is None:
        x_max = float(np.nanmax(force))

    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        raise RuntimeError(f"Invalid force range for binning: x_min={x_min}, x_max={x_max}")

    if bin_width <= 0:
        raise RuntimeError("bin_width must be > 0")

    # Build bin edges. Add a tiny epsilon to include max in the last bin edge range.
    edges = np.arange(x_min, x_max + bin_width * 1.000001, bin_width)
    if edges.size < 2:
        raise RuntimeError("Not enough bins; increase x-range or decrease bin_width.")

    # Digitize -> bin index in [0, n_bins-1]
    idx = np.digitize(force, edges) - 1
    n_bins = edges.size - 1

    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue

        vals = audio[mask]
        count = int(vals.size)
        if count < min_count:
            continue

        mean_db = float(np.mean(vals))
        std_db = float(np.std(vals, ddof=1)) if count > 1 else 0.0

        left = edges[b]
        right = edges[b + 1]
        center = 0.5 * (left + right)

        rows.append(
            {
                "bin_center": center,
                "count": count,
                "mean_db": mean_db,
                "std_db": std_db,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("bin_center").reset_index(drop=True)

    return out


# ------------------------- Plotting -------------------------

def plot_binned_mean_std(
    runs: list[tuple[str, pd.DataFrame]],
    *,
    out_png: Path,
    title: str,
    show: bool,
    alpha_band: float = 0.20,
) -> None:
    """
    Plot mean line and ±1 std shaded band for each run.
    """
    if not runs:
        raise RuntimeError("No runs to plot.")

    fig, ax = plt.subplots(figsize=(11, 6.5))

    for label, stats_df in runs:
        if stats_df.empty:
            print(f"[WARN] No valid bins for: {label} (skipping)")
            continue

        x = stats_df["bin_center"].to_numpy()
        y = stats_df["mean_db"].to_numpy()
        s = stats_df["std_db"].to_numpy()

        # Line
        line = ax.plot(x, y, linewidth=2.2, marker="o", markersize=4, label=label)[0]

        # Shaded ±1 std band (use same line color)
        c = line.get_color()
        ax.fill_between(x, y - s, y + s, color=c, alpha=alpha_band, linewidth=0)

    ax.set_title(title)
    ax.set_xlabel("Force (N)")
    ax.set_ylabel("Audio Level (dBFS)")

    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Saved binned comparison plot → {out_png}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ------------------------- Main -------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare propellers by plotting binned mean ± std of audio_rms_dbfs vs force_N "
                    "from *_combined_event_aligned.csv files."
    )
    parser.add_argument("--dir", default=".", help="Directory to search (default: current directory).")
    parser.add_argument("--bin_width", type=float, default=1.0, help="Force bin width in N (default: 1.0).")
    parser.add_argument("--min_count", type=int, default=3, help="Min samples per bin (default: 3).")
    parser.add_argument("--x_min", type=float, default=None, help="Optional min force for binning.")
    parser.add_argument("--x_max", type=float, default=None, help="Optional max force for binning.")
    parser.add_argument("--out_png", default="propeller_comparison_binned_mean_std.png", help="Output PNG filename.")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot window.")
    parser.add_argument("--title", default="Propeller Comparison: Binned Mean ± Std (Audio vs Force)",
                        help="Plot title.")
    args = parser.parse_args()

    search_dir = Path(args.dir)
    if not search_dir.exists():
        raise RuntimeError(f"Directory does not exist: {search_dir}")

    csv_paths = find_combined_csvs(search_dir)
    if not csv_paths:
        raise RuntimeError(f"No *_combined_event_aligned.csv files found in: {search_dir}")

    print(f"Found {len(csv_paths)} combined CSV files in {search_dir}:")
    for p in csv_paths:
        print(f"  - {p.name}")

    runs: list[tuple[str, pd.DataFrame]] = []

    for path in csv_paths:
        label = infer_label_from_path(path)
        df = load_combined(path)

        stats_df = binned_stats(
            df,
            bin_width=args.bin_width,
            x_min=args.x_min,
            x_max=args.x_max,
            min_count=args.min_count,
        )
        runs.append((label, stats_df))

    plot_binned_mean_std(
        runs,
        out_png=Path(args.out_png),
        title=args.title,
        show=not args.no_show,
        alpha_band=0.20,
    )


if __name__ == "__main__":
    main()
