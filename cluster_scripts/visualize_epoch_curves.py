import os
import csv
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# Use Times New Roman for consistency with other figures
matplotlib.rcParams["font.family"] = "Times New Roman"


# Input CSV paths (produced during training/validation logging)
DATA_DIR = "cluster_scripts/cluster_data"
INSTANCE_SYNTHETIC_CSV = os.path.join(DATA_DIR, "instance_synthetic.csv")
INSTANCE_FOR_INSTANCE_CSV = os.path.join(DATA_DIR, "instance_for-instance.csv")
SEMANTIC_SYNTHETIC_CSV = os.path.join(DATA_DIR, "semantic_synthetic.csv")
SEMANTIC_FOR_INSTANCE_CSV = os.path.join(DATA_DIR, "semantic_for-instance.csv")


# Visual styling (reuse colors from other visualization)
PROCEDURALLY_GENERATED_COLOR = "#2E86AB"  # Synthetic
FOR_INSTANCE_COLOR = "#A23B72"  # For-Instance


def _load_metric_series(csv_path: str, metric_substring: str) -> Optional[List[float]]:
    """Load a metric time series from a CSV by matching the column name substring.

    The CSV headers differ for synthetic vs for-instance (prefix in the column name),
    so we detect the metric column by substring matching, e.g.:
      - "val/total/instance_detection_f1_score"
      - "val/total/m_iou"

    Returns a list of floats in CSV row order (treat each row as an epoch).
    """
    if not os.path.exists(csv_path):
        print(f"Warning: CSV not found: {csv_path}")
        return None

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        # Find the first column containing the metric substring
        col_name = next((h for h in reader.fieldnames or [] if metric_substring in h), None)
        if col_name is None:
            print(f"Warning: No column containing '{metric_substring}' found in {csv_path}")
            return None

        values: List[float] = []
        for row in reader:
            val_str = row.get(col_name, "")
            try:
                values.append(float(val_str))
            except (TypeError, ValueError):
                # Skip rows with non-parsable values
                continue
        return values


def _plot_overlay(series_a: List[float], series_b: List[float],
                  label_a: str, label_b: str,
                  title: str, ylabel: str,
                  color_a: str, color_b: str,
                  save_basename: str) -> plt.Figure:
    """Create a single overlay line plot for two time series and save to PDF/PNG."""
    fig, ax = plt.subplots(figsize=(8, 4))

    # X-axis: epoch index (1..N); each row corresponds to a validation at an epoch
    x_a = np.arange(1, len(series_a) + 1)
    x_b = np.arange(1, len(series_b) + 1)

    ax.plot(x_a, series_a, label=label_a, color=color_a, linewidth=2.0)
    ax.plot(x_b, series_b, label=label_b, color=color_b, linewidth=2.0)

    ax.set_xlabel("Epoch", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.grid(True, alpha=0.3)

    # Set y-limits with gentle padding within [0, 1]
    all_vals = [*series_a, *series_b] if (series_a and series_b) else (series_a or series_b or [0, 1])
    y_min = max(0.0, min(all_vals) - 0.05)
    y_max = min(1.0, max(all_vals) + 0.05)
    if y_max <= y_min:
        y_max = y_min + 0.1
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()

    out_dir = DATA_DIR
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, f"{save_basename}.pdf")
    png_path = os.path.join(out_dir, f"{save_basename}.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {os.path.relpath(pdf_path)}")
    print(f"Saved: {os.path.relpath(png_path)}")

    return fig


def main():
    # Load per-epoch Instance Detection F1
    f1_syn = _load_metric_series(INSTANCE_SYNTHETIC_CSV, "val/total/instance_detection_f1_score") or []
    f1_fi = _load_metric_series(INSTANCE_FOR_INSTANCE_CSV, "val/total/instance_detection_f1_score") or []

    # Load per-epoch mIoU
    miou_syn = _load_metric_series(SEMANTIC_SYNTHETIC_CSV, "val/total/m_iou") or []
    miou_fi = _load_metric_series(SEMANTIC_FOR_INSTANCE_CSV, "val/total/m_iou") or []

    # Report basic stats
    if f1_syn or f1_fi:
        print(f"Instance F1 points — Synthetic: {len(f1_syn)}, For-Instance: {len(f1_fi)}")
    if miou_syn or miou_fi:
        print(f"mIoU points — Synthetic: {len(miou_syn)}, For-Instance: {len(miou_fi)}")

    # Plot overlay: Instance Detection F1
    if f1_syn and f1_fi:
        _plot_overlay(
            f1_syn, f1_fi,
            label_a="Synthetic Prediction",
            label_b="For-Instance Prediction",
            title="Instance Detection F1 over Epochs",
            ylabel="F1 Score",
            color_a=PROCEDURALLY_GENERATED_COLOR,
            color_b=FOR_INSTANCE_COLOR,
            save_basename="instance_f1_over_epochs",
        )
    else:
        print("Skipping Instance F1 plot (missing data)")

    # Plot overlay: mIoU
    if miou_syn and miou_fi:
        _plot_overlay(
            miou_syn, miou_fi,
            label_a="Synthetic Prediction",
            label_b="For-Instance Prediction",
            title="mIoU over Epochs",
            ylabel="mIoU",
            color_a=PROCEDURALLY_GENERATED_COLOR,
            color_b=FOR_INSTANCE_COLOR,
            save_basename="miou_over_epochs",
        )
    else:
        print("Skipping mIoU plot (missing data)")

    # Show figures if interactive
    plt.show()


if __name__ == "__main__":
    main()
