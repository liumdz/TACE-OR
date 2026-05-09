import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 0. PATH SETTINGS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]

BASELINE_PATH = ROOT / "output" / "baseline" / "cv_metrics_baseline.csv"
TUNED_PATH = ROOT / "output" / "tuned" / "cv_metrics_tuned.csv"

FIG_DIR = ROOT / "output" / "figures" / "baseline_vs_tuned"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = FIG_DIR / "baseline_vs_tuned_metric_comparison.csv"

# ============================================================
# 1. MODEL ORDER AND DISPLAY NAMES
# ============================================================
MODEL_ORDER = [
    "LogisticRegression",
    "ElasticNet",
    "RandomForest",
    "SVM",
    "LightGBM",
    "GradientBoosting",
    "XGBoost",
]

DISPLAY_NAMES = {
    "LogisticRegression": "Logistic Regression",
    "ElasticNet": "Elastic Net",
    "RandomForest": "Random Forest",
    "SVM": "SVM",
    "LightGBM": "LightGBM",
    "GradientBoosting": "Gradient Boosting",
    "XGBoost": "XGBoost",
}

METRICS = [
    ("AUC", "AUC"),
    ("PR_AUC", "AUPRC"),
    ("Accuracy", "Accuracy"),
    ("Precision", "Precision"),
    ("Sensitivity", "Sensitivity"),
    ("Specificity", "Specificity"),
    ("F1", "F1 score"),
]

KEY_TUNED_METRICS = [
    ("PR_AUC", "AUPRC"),
    ("Sensitivity", "Sensitivity"),
    ("Specificity", "Specificity"),
    ("F1", "F1 score"),
]

# 灰色 + 蓝色
BASELINE_COLOR = "#BDBDBD"
TUNED_COLOR = "#2E77BB"

# ============================================================
# 2. LOAD DATA
# ============================================================
if not BASELINE_PATH.exists():
    raise FileNotFoundError(f"Cannot find baseline file: {BASELINE_PATH}")

if not TUNED_PATH.exists():
    raise FileNotFoundError(f"Cannot find tuned file: {TUNED_PATH}")

baseline = pd.read_csv(BASELINE_PATH)
tuned = pd.read_csv(TUNED_PATH)

baseline = baseline.set_index("Model").loc[MODEL_ORDER].reset_index()
tuned = tuned.set_index("Model").loc[MODEL_ORDER].reset_index()

baseline["Display"] = baseline["Model"].map(DISPLAY_NAMES)
tuned["Display"] = tuned["Model"].map(DISPLAY_NAMES)

# ============================================================
# 3. SAVE COMPARISON TABLE
# ============================================================
comparison_rows = []

for model in MODEL_ORDER:
    b = baseline[baseline["Model"] == model].iloc[0]
    t = tuned[tuned["Model"] == model].iloc[0]

    row = {
        "Model": model,
        "Display": DISPLAY_NAMES[model],
    }

    for metric, _ in METRICS:
        row[f"Baseline_{metric}"] = float(b[metric])
        row[f"Tuned_{metric}"] = float(t[metric])
        row[f"Delta_{metric}"] = float(t[metric]) - float(b[metric])

        b_std_col = f"{metric}_std"
        t_std_col = f"{metric}_std"

        if b_std_col in baseline.columns:
            row[f"Baseline_{metric}_std"] = float(b[b_std_col])
        if t_std_col in tuned.columns:
            row[f"Tuned_{metric}_std"] = float(t[t_std_col])

    comparison_rows.append(row)

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")


# ============================================================
# 4. PLOT UTILS
# ============================================================
def save_current_figure(filename):
    png_path = FIG_DIR / f"{filename}.png"
    pdf_path = FIG_DIR / f"{filename}.pdf"

    plt.tight_layout()
    plt.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def get_error(df, metric):
    std_col = f"{metric}_std"
    if std_col in df.columns:
        return df[std_col].astype(float).values
    return np.zeros(len(df))


def add_value_labels(ax, bars, values, errors=None, y_offset=0.018, fontsize=7):
    """
    把数值标注放在 error bar 上方，避免被短竖线挡住。
    """
    if errors is None:
        errors = np.zeros(len(values))

    for bar, value, err in zip(bars, values, errors):
        label_y = value + err + y_offset
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=90,
        )


def set_dynamic_ylim(ax, values_a, values_b, err_a=None, err_b=None):
    """
    根据柱高和误差线自动设置 y 轴上限，给数值标注留空间。
    """
    if err_a is None:
        err_a = np.zeros_like(values_a)
    if err_b is None:
        err_b = np.zeros_like(values_b)

    top = max(
        np.max(values_a + err_a),
        np.max(values_b + err_b),
    )

    upper = min(1.25, max(1.02, top + 0.12))
    ax.set_ylim(0, upper)


# ============================================================
# 5. FIGURE 1: EACH METRIC BEFORE-AFTER COMPARISON
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(26, 12))
axes = axes.flatten()

x = np.arange(len(MODEL_ORDER))
width = 0.34

for idx, (metric, metric_title) in enumerate(METRICS):
    ax = axes[idx]

    baseline_vals = baseline[metric].astype(float).values
    tuned_vals = tuned[metric].astype(float).values

    baseline_err = get_error(baseline, metric)
    tuned_err = get_error(tuned, metric)

    bars_baseline = ax.bar(
        x - width / 2,
        baseline_vals,
        width,
        yerr=baseline_err,
        capsize=3,
        color=BASELINE_COLOR,
        edgecolor="black",
        linewidth=0.4,
        ecolor="black",
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
        label="Baseline",
    )

    bars_tuned = ax.bar(
        x + width / 2,
        tuned_vals,
        width,
        yerr=tuned_err,
        capsize=3,
        color=TUNED_COLOR,
        edgecolor="black",
        linewidth=0.4,
        ecolor="black",
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
        label="Tuned",
    )

    add_value_labels(ax, bars_baseline, baseline_vals, baseline_err)
    add_value_labels(ax, bars_tuned, tuned_vals, tuned_err)

    set_dynamic_ylim(ax, baseline_vals, tuned_vals, baseline_err, tuned_err)

    ax.set_title(metric_title, fontsize=15, fontweight="bold")
    ax.set_ylabel("Score", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [DISPLAY_NAMES[m] for m in MODEL_ORDER],
        rotation=35,
        ha="right",
        fontsize=8,
    )

    ax.grid(axis="y", linestyle="--", alpha=0.3)

    if idx == 0:
        ax.legend(frameon=False, loc="upper left", fontsize=10)

axes[-1].axis("off")

fig.suptitle(
    "Baseline vs Tuned Model Performance Across Evaluation Metrics",
    fontsize=20,
    fontweight="bold",
    y=1.02,
)

save_current_figure("fig1_baseline_vs_tuned_each_metric")


# ============================================================
# 6. FIGURE 2: FOUR-PANEL COMPARISON
#    AUC / AUPRC / Sensitivity / F1
# ============================================================

four_panel_metrics = [
    ("AUC", "AUC"),
    ("PR_AUC", "AUPRC"),
    ("Sensitivity", "Sensitivity"),
    ("F1", "F1 score"),
]

panel_labels = ["A", "B", "C", "D"]

fig, axes = plt.subplots(2, 2, figsize=(22, 12))
axes = axes.flatten()

for idx, (ax, (metric, title)) in enumerate(zip(axes, four_panel_metrics)):
    baseline_vals = baseline[metric].astype(float).values
    tuned_vals = tuned[metric].astype(float).values

    baseline_err = get_error(baseline, metric)
    tuned_err = get_error(tuned, metric)

    bars_baseline = ax.bar(
        x - width / 2,
        baseline_vals,
        width,
        yerr=baseline_err,
        capsize=4,
        color=BASELINE_COLOR,
        edgecolor="black",
        linewidth=0.4,
        ecolor="black",
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
        label="Baseline",
    )

    bars_tuned = ax.bar(
        x + width / 2,
        tuned_vals,
        width,
        yerr=tuned_err,
        capsize=4,
        color=TUNED_COLOR,
        edgecolor="black",
        linewidth=0.4,
        ecolor="black",
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
        label="Tuned",
    )

    add_value_labels(
        ax,
        bars_baseline,
        baseline_vals,
        baseline_err,
        y_offset=0.018,
        fontsize=8,
    )

    add_value_labels(
        ax,
        bars_tuned,
        tuned_vals,
        tuned_err,
        y_offset=0.018,
        fontsize=8,
    )

    set_dynamic_ylim(ax, baseline_vals, tuned_vals, baseline_err, tuned_err)

    ax.set_title(
        f"{panel_labels[idx]}. Baseline vs Tuned {title}",
        fontsize=14,
        fontweight="bold",
        loc="left",
    )

    ax.set_ylabel("Score", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [DISPLAY_NAMES[m] for m in MODEL_ORDER],
        rotation=30,
        ha="right",
        fontsize=9,
    )

    ax.grid(axis="y", linestyle="--", alpha=0.3)

# 只放一个总图例，避免每个子图重复
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    frameon=False,
    fontsize=11,
    bbox_to_anchor=(0.5, 0.98),
)

fig.suptitle(
    "Comparison of Candidate Model Performance Before and After Hyperparameter Tuning",
    fontsize=18,
    fontweight="bold",
    y=1.03,
)

plt.tight_layout(rect=[0, 0, 1, 0.95])

four_panel_png = FIG_DIR / "fig2_baseline_vs_tuned_four_panel.png"
four_panel_pdf = FIG_DIR / "fig2_baseline_vs_tuned_four_panel.pdf"

plt.savefig(four_panel_png, dpi=600, bbox_inches="tight")
plt.savefig(four_panel_pdf, bbox_inches="tight")
plt.close()

print(f"Saved: {four_panel_png}")
print(f"Saved: {four_panel_pdf}")

# ============================================================
# 7. FIGURE 3: TUNED KEY METRICS
# ============================================================
fig, ax = plt.subplots(figsize=(16, 7))

key_width = 0.18

for i, (metric, label) in enumerate(KEY_TUNED_METRICS):
    vals = tuned[metric].astype(float).values
    err = get_error(tuned, metric)

    offset = (i - (len(KEY_TUNED_METRICS) - 1) / 2) * key_width

    bars = ax.bar(
        x + offset,
        vals,
        key_width,
        yerr=err,
        capsize=3,
        edgecolor="black",
        linewidth=0.4,
        ecolor="black",
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
        label=label,
    )

    add_value_labels(ax, bars, vals, err, y_offset=0.015, fontsize=7)

all_vals = []
all_errs = []
for metric, _ in KEY_TUNED_METRICS:
    all_vals.extend(tuned[metric].astype(float).values)
    all_errs.extend(get_error(tuned, metric))

top = np.max(np.array(all_vals) + np.array(all_errs))
ax.set_ylim(0, min(1.25, max(1.02, top + 0.12)))

ax.set_title(
    "Key Performance Metrics of Tuned Models",
    fontsize=17,
    fontweight="bold",
)
ax.set_ylabel("Score")
ax.set_xticks(x)
ax.set_xticklabels(
    [DISPLAY_NAMES[m] for m in MODEL_ORDER],
    rotation=30,
    ha="right",
    fontsize=9,
)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.legend(frameon=False, ncol=4)

save_current_figure("fig4_tuned_key_metrics")


# ============================================================
# 8. FIGURE 4: DELTA HEATMAP — DIRECT TUNING ADVANTAGE
# ============================================================
delta_metrics = [
    ("AUC", "AUC"),
    ("PR_AUC", "AUPRC"),
    ("Accuracy", "Accuracy"),
    ("Precision", "Precision"),
    ("Sensitivity", "Sensitivity"),
    ("Specificity", "Specificity"),
    ("F1", "F1 score"),
]

delta_data = []

for model in MODEL_ORDER:
    row = []
    b = baseline[baseline["Model"] == model].iloc[0]
    t = tuned[tuned["Model"] == model].iloc[0]

    for metric, _ in delta_metrics:
        row.append(float(t[metric]) - float(b[metric]))

    delta_data.append(row)

delta_data = np.array(delta_data)

fig, ax = plt.subplots(figsize=(12, 6.8))

max_abs = np.max(np.abs(delta_data))
im = ax.imshow(
    delta_data,
    aspect="auto",
    cmap="RdBu",
    vmin=-max_abs,
    vmax=max_abs,
)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Tuned - Baseline")

ax.set_xticks(np.arange(len(delta_metrics)))
ax.set_xticklabels([name for _, name in delta_metrics], rotation=30, ha="right")

ax.set_yticks(np.arange(len(MODEL_ORDER)))
ax.set_yticklabels([DISPLAY_NAMES[m] for m in MODEL_ORDER])

ax.set_title(
    "Performance Change After Hyperparameter Tuning",
    fontsize=17,
    fontweight="bold",
)

for i in range(delta_data.shape[0]):
    for j in range(delta_data.shape[1]):
        value = delta_data[i, j]
        ax.text(
            j,
            i,
            f"{value:+.3f}",
            ha="center",
            va="center",
            fontsize=8,
        )

save_current_figure("fig5_tuning_delta_heatmap")


# ============================================================
# 9. DONE
# ============================================================
print("=" * 80)
print("DONE")
print("=" * 80)
print("Saved figures to:")
print(FIG_DIR)
print("\nSaved comparison table:")
print(OUT_CSV)
print("\nGenerated figures:")
print("1) fig1_baseline_vs_tuned_each_metric.png / .pdf")
print("2) fig2_auc_auprc_before_after.png / .pdf")
print("3) fig3_tuned_key_metrics.png / .pdf")
print("4) fig4_tuning_delta_heatmap.png / .pdf")
print("\nNote:")
print("This script only reads saved CV metric CSV files and does NOT refit any model.")
