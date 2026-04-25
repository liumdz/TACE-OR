import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ============================================================
# 0. SETTINGS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]

OOF_PATH = ROOT / "output" / "top10_model" / "randomforest" / "oof_predictions_full_vs_top10.csv"
THRESH_PATH = ROOT / "output" / "threshold" / "randomforest" / "best_threshold_top10.csv"

SAVE_DIR = ROOT / "output" / "figures" / "cv_full_vs_top10_after_threshold"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

METRICS = [
    "AUC",
    "AUPRC",
    "Accuracy",
    "Precision",
    "Sensitivity",
    "Specificity",
    "F1",
    "Youden",
]

MODEL_ORDER = [
    "Full_RandomForest",
    "Top10_RandomForest",
]

DISPLAY_NAMES = {
    "Full_RandomForest": "Full Random Forest",
    "Top10_RandomForest": "Top10 Random Forest",
}

COLORS = {
    "Full_RandomForest": "#BDBDBD",
    "Top10_RandomForest": "#2E77BB",
}

print("=" * 80)
print("CV METRICS AFTER TUNING AND THRESHOLD SELECTION")
print("Use saved OOF predictions only. No refit. No test/external.")
print("=" * 80)


# ============================================================
# 1. LOAD OOF PREDICTIONS
# ============================================================
if not OOF_PATH.exists():
    raise FileNotFoundError(
        f"Cannot find OOF prediction file:\n{OOF_PATH}\n"
        "Please run scripts/train_top10_randomforest.py first."
    )

oof = pd.read_csv(OOF_PATH)

required = {"Model", "Fold", "Sample_Index", "y_true", "y_prob"}
missing = required - set(oof.columns)
if missing:
    raise ValueError(f"OOF file missing columns: {missing}")

print(f"\nLoaded OOF predictions: {OOF_PATH}")
print(oof["Model"].value_counts())


# ============================================================
# 2. THRESHOLD FUNCTIONS
# ============================================================
def compute_metrics(y_true, y_prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = precision_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    youden = sensitivity + specificity - 1

    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "Accuracy": accuracy,
        "Precision": precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1": f1,
        "Youden": youden,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def search_youden_threshold(y_true, y_prob):
    thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)

    rows = []
    for t in thresholds:
        m = compute_metrics(y_true, y_prob, t)
        rows.append({
            "Threshold": t,
            "Youden": m["Youden"],
            "Sensitivity": m["Sensitivity"],
            "Specificity": m["Specificity"],
            "F1": m["F1"],
        })

    df = pd.DataFrame(rows)
    max_youden = df["Youden"].max()
    cand = df[df["Youden"] == max_youden].copy()

    # 如果多个阈值相同，选择更接近 0.5 的，避免极端阈值
    cand["distance_to_0_5"] = (cand["Threshold"] - 0.5).abs()
    best = cand.sort_values(["distance_to_0_5", "Threshold"]).iloc[0]

    return float(best["Threshold"]), df


# ============================================================
# 3. GET THRESHOLDS
# ============================================================
# Top10 使用已经正式确定的训练集 OOF Youden 阈值
if not THRESH_PATH.exists():
    raise FileNotFoundError(
        f"Cannot find threshold file:\n{THRESH_PATH}\n"
        "Please run scripts/threshold_selection_randomforest.py first."
    )

thresh_df = pd.read_csv(THRESH_PATH)
top10_threshold = float(thresh_df["Best_Threshold"].iloc[0])

# Full RF 这里仅为了公平画图，也用 OOF 预测寻找 Youden 阈值
# 注意：这个 Full 阈值不是最终模型阈值，只用于 CV 可视化比较
full_oof = oof[oof["Model"] == "Full_RandomForest"].copy()
full_threshold, full_curve = search_youden_threshold(
    full_oof["y_true"].values,
    full_oof["y_prob"].values,
)

threshold_map = {
    "Full_RandomForest": full_threshold,
    "Top10_RandomForest": top10_threshold,
}

print("\nThresholds used for CV metric visualization:")
print(f"Full Random Forest  : {full_threshold:.2f}  (OOF Youden, for comparison only)")
print(f"Top10 Random Forest : {top10_threshold:.2f}  (final selected threshold)")


# ============================================================
# 4. FOLD-LEVEL CV METRICS
# ============================================================
fold_rows = []

for model_name in MODEL_ORDER:
    sub = oof[oof["Model"] == model_name].copy()
    threshold = threshold_map[model_name]

    for fold in sorted(sub["Fold"].unique()):
        sf = sub[sub["Fold"] == fold].copy()

        y_true = sf["y_true"].values
        y_prob = sf["y_prob"].values

        m = compute_metrics(y_true, y_prob, threshold)

        row = {
            "Model": model_name,
            "Display": DISPLAY_NAMES[model_name],
            "Fold": int(fold),
            "Threshold": threshold,
        }
        row.update(m)
        fold_rows.append(row)

fold_df = pd.DataFrame(fold_rows)

summary_rows = []
for model_name in MODEL_ORDER:
    sub = fold_df[fold_df["Model"] == model_name]

    row = {
        "Model": model_name,
        "Display": DISPLAY_NAMES[model_name],
        "Threshold": threshold_map[model_name],
    }

    for metric in METRICS:
        row[metric] = sub[metric].mean()
        row[f"{metric}_std"] = sub[metric].std(ddof=1)

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)

fold_path = SAVE_DIR / "cv_fold_metrics_after_threshold.csv"
summary_path = SAVE_DIR / "cv_summary_metrics_after_threshold.csv"

fold_df.to_csv(fold_path, index=False, encoding="utf-8-sig")
summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

print("\nSaved:")
print(fold_path)
print(summary_path)

print("\nCV summary:")
print(summary_df[["Display", "Threshold"] + METRICS])


# ============================================================
# 5. FIGURE 1: GROUPED BAR CHART
# ============================================================
x = np.arange(len(METRICS))
width = 0.34

fig, ax = plt.subplots(figsize=(14, 6.5))

for i, model_name in enumerate(MODEL_ORDER):
    row = summary_df[summary_df["Model"] == model_name].iloc[0]

    vals = np.array([row[m] for m in METRICS], dtype=float)
    errs = np.array([row[f"{m}_std"] for m in METRICS], dtype=float)

    offset = -width / 2 if i == 0 else width / 2

    bars = ax.bar(
        x + offset,
        vals,
        width,
        yerr=errs,
        capsize=4,
        color=COLORS[model_name],
        edgecolor="black",
        linewidth=0.5,
        ecolor="black",
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
        label=f"{DISPLAY_NAMES[model_name]} (threshold={row['Threshold']:.2f})",
    )

    for bar, val, err in zip(bars, vals, errs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + err + 0.018,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

ax.set_xticks(x)
ax.set_xticklabels(METRICS, fontsize=11)
ax.set_ylabel("Score", fontsize=12, fontweight="bold")
ax.set_ylim(0, 1.18)
ax.set_title(
    "Full vs Top10 Random Forest: 5-Fold CV Metrics After Threshold Selection",
    fontsize=14,
    fontweight="bold",
)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.legend(frameon=False, fontsize=10, loc="upper right")

plt.tight_layout()

fig1_png = SAVE_DIR / "fig1_cv_full_vs_top10_metrics_bar.png"
fig1_pdf = SAVE_DIR / "fig1_cv_full_vs_top10_metrics_bar.pdf"

plt.savefig(fig1_png, dpi=600, bbox_inches="tight")
plt.savefig(fig1_pdf, bbox_inches="tight")
plt.close()

print("\nSaved figure:")
print(fig1_png)
print(fig1_pdf)


# ============================================================
# 6. FIGURE 2: EACH METRIC PANEL
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
axes = axes.flatten()

for idx, metric in enumerate(METRICS):
    ax = axes[idx]

    vals = []
    errs = []
    labels = []

    for model_name in MODEL_ORDER:
        row = summary_df[summary_df["Model"] == model_name].iloc[0]
        vals.append(float(row[metric]))
        errs.append(float(row[f"{metric}_std"]))
        labels.append(DISPLAY_NAMES[model_name])

    bars = ax.bar(
        np.arange(2),
        vals,
        yerr=errs,
        capsize=4,
        color=[COLORS[m] for m in MODEL_ORDER],
        edgecolor="black",
        linewidth=0.6,
        ecolor="black",
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
    )

    for bar, val, err in zip(bars, vals, errs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + err + 0.025,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title(metric, fontsize=13, fontweight="bold")
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(["Full RF", "Top10 RF"], fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

fig.suptitle(
    "Metric-by-Metric CV Comparison After Threshold Selection",
    fontsize=16,
    fontweight="bold",
    y=1.02,
)

plt.tight_layout()

fig2_png = SAVE_DIR / "fig2_cv_full_vs_top10_each_metric.png"
fig2_pdf = SAVE_DIR / "fig2_cv_full_vs_top10_each_metric.pdf"

plt.savefig(fig2_png, dpi=600, bbox_inches="tight")
plt.savefig(fig2_pdf, bbox_inches="tight")
plt.close()

print("Saved figure:")
print(fig2_png)
print(fig2_pdf)


# ============================================================
# 7. FIGURE 3: TOP10 DEFAULT VS YOUDEN THRESHOLD
# ============================================================
top10_oof = oof[oof["Model"] == "Top10_RandomForest"].copy()

threshold_scenarios = {
    "Default 0.50": 0.50,
    f"Youden {top10_threshold:.2f}": top10_threshold,
}

scenario_rows = []

for scenario, th in threshold_scenarios.items():
    for fold in sorted(top10_oof["Fold"].unique()):
        sf = top10_oof[top10_oof["Fold"] == fold]
        m = compute_metrics(sf["y_true"].values, sf["y_prob"].values, th)

        row = {
            "Scenario": scenario,
            "Fold": int(fold),
            "Threshold": th,
        }
        row.update(m)
        scenario_rows.append(row)

scenario_df = pd.DataFrame(scenario_rows)

scenario_summary_rows = []
for scenario, th in threshold_scenarios.items():
    sub = scenario_df[scenario_df["Scenario"] == scenario]

    row = {
        "Scenario": scenario,
        "Threshold": th,
    }

    for metric in METRICS:
        row[metric] = sub[metric].mean()
        row[f"{metric}_std"] = sub[metric].std(ddof=1)

    scenario_summary_rows.append(row)

scenario_summary = pd.DataFrame(scenario_summary_rows)
scenario_summary.to_csv(
    SAVE_DIR / "top10_default_vs_youden_cv_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

x = np.arange(len(METRICS))
width = 0.34

fig, ax = plt.subplots(figsize=(14, 6.5))

scenario_colors = ["#BDBDBD", "#2E77BB"]

for i, (_, row) in enumerate(scenario_summary.iterrows()):
    vals = np.array([row[m] for m in METRICS], dtype=float)
    errs = np.array([row[f"{m}_std"] for m in METRICS], dtype=float)

    offset = -width / 2 if i == 0 else width / 2

    bars = ax.bar(
        x + offset,
        vals,
        width,
        yerr=errs,
        capsize=4,
        color=scenario_colors[i],
        edgecolor="black",
        linewidth=0.5,
        ecolor="black",
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
        label=f"{row['Scenario']}",
    )

    for bar, val, err in zip(bars, vals, errs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + err + 0.018,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

ax.set_xticks(x)
ax.set_xticklabels(METRICS, fontsize=11)
ax.set_ylabel("Score", fontsize=12, fontweight="bold")
ax.set_ylim(0, 1.18)
ax.set_title(
    "Top10 Random Forest: Default vs Youden-Optimized Threshold in 5-Fold CV",
    fontsize=14,
    fontweight="bold",
)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.legend(frameon=False, fontsize=10, loc="upper right")

plt.tight_layout()

fig3_png = SAVE_DIR / "fig3_top10_default_vs_youden_metrics.png"
fig3_pdf = SAVE_DIR / "fig3_top10_default_vs_youden_metrics.pdf"

plt.savefig(fig3_png, dpi=600, bbox_inches="tight")
plt.savefig(fig3_pdf, bbox_inches="tight")
plt.close()

print("Saved figure:")
print(fig3_png)
print(fig3_pdf)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
print("This script:")
print("1. Uses saved train OOF predictions only.")
print("2. Does NOT refit any model.")
print("3. Does NOT read internal_test_processed.csv.")
print("4. Does NOT read external_processed.csv.")
