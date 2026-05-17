"""Combine three SHAP figures into a single composite figure.

Layout (A/B on top row, C spanning the bottom row):
  +-------------------+-------------------+
  | A: SHAP summary   | B: mean |SHAP| bar|
  +-------------------+-------------------+
  | C: Top-10 cumulative importance       |
  +-------------------+-------------------+
"""
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
SHAP_DIR = ROOT / "output" / "shap_analysis" / "randomforest"

PANEL_A = SHAP_DIR / "shap_summary_all_features.png"
PANEL_B = SHAP_DIR / "shap_bar_all_features.png"
PANEL_C = SHAP_DIR / "Top10_features_cumulative.png"
OUT_PATH = SHAP_DIR / "shap_composite_three_panel.png"


def _add_panel(ax: plt.Axes, img_path: Path, label: str) -> None:
    ax.imshow(mpimg.imread(img_path))
    ax.axis("off")
    ax.text(
        -0.02, 1.02, f"({label})",
        transform=ax.transAxes,
        fontsize=14, fontweight="normal",
        va="bottom", ha="left",
    )


def main() -> None:
    fig = plt.figure(figsize=(16, 14), dpi=300)
    gs = GridSpec(
        2, 2, figure=fig,
        height_ratios=[1.25, 1.0],
        wspace=0.04, hspace=0.06,
    )

    _add_panel(fig.add_subplot(gs[0, 0]), PANEL_A, "A")
    _add_panel(fig.add_subplot(gs[0, 1]), PANEL_B, "B")
    _add_panel(fig.add_subplot(gs[1, :]), PANEL_C, "C")

    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved composite figure to: {OUT_PATH}")


if __name__ == "__main__":
    main()
