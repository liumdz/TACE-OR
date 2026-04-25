import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# 0. SETTINGS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]

# 这个文件由 scripts/train_top10_randomforest.py 生成
OOF_PATH = (
    ROOT
    / "output"
    / "top10_model"
    / "randomforest"
    / "oof_predictions_full_vs_top10.csv"
)

SAVE_DIR = ROOT / "output" / "threshold" / "randomforest"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)

MODEL_CONFIGS = {
    "Full_RandomForest": {
        "display_name": "Full Random Forest",
        "short_name": "full",
    },
    "Top10_RandomForest": {
        "display_name": "Top10 Random Forest",
        "short_name": "top10",
    },
}

BEST_THRESHOLD_CSV = SAVE_DIR / "best_threshold_full_vs_top10.csv"
BEST_THRESHOLD_JSON = SAVE_DIR / "best_threshold_full_vs_top10.json"
CURVE_CSV = SAVE_DIR / "threshold_youden_curve_full_vs_top10.csv"
METRICS_CSV = SAVE_DIR / "threshold_metrics_full_vs_top10.csv"

FIG_PATH_PNG = SAVE_DIR / "threshold_selection_full_vs_top10_youden.png"
FIG_PATH_PDF = SAVE_DIR / "threshold_selection_full_vs_top10_youden.pdf"

# 兼容旧脚本路径：其他 final evaluation 脚本可能还会读取这个文件
TOP10_COMPAT_CSV = SAVE_DIR / "best_threshold_top10.csv"
TOP10_COMPAT_JSON = SAVE_DIR / "best_threshold_top10.json"

print("=" * 80)
print("THRESHOLD SELECTION — FULL RANDOM FOREST vs TOP10 RANDOM FOREST")
print("Use train out-of-fold predictions only. No refit. No test/external.")
print("=" * 80)


# ============================================================
# 1. LOAD OOF PREDICTIONS
# ============================================================
if not OOF_PATH.exists():
    raise FileNotFoundError(
        f"Cannot find OOF prediction file:\n{OOF_PATH}\n\n"
        "Please run scripts/train_top10_randomforest.py first."
    )

oof = pd.read_csv(OOF_PATH)

required_cols = {"Model", "Fold", "Sample_Index", "y_true", "y_prob"}
missing_cols = required_cols - set(oof.columns)

if missing_cols:
    raise ValueError(f"OOF file missing columns: {missing_cols}")

print(f"\nLoaded OOF predictions: {OOF_PATH}")
print("Available model rows:")
print(oof["Model"].value_counts())

missing_models = [m for m in MODEL_CONFIGS if m not in oof["Model"].unique()]
if missing_models:
    raise ValueError(
        f"Missing expected models in OOF file: {missing_models}\n"
        f"Available models: {oof['Model'].unique().tolist()}"
    )


# ============================================================
# 2. METRIC FUNCTIONS
# ============================================================
def compute_threshold_metrics(y_true, y_prob, threshold, auc_value, auprc_value):
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = precision_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    youden = sensitivity + specificity - 1

    return {
        "Threshold": float(threshold),
        "AUC": float(auc_value),
        "AUPRC": float(auprc_value),
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Sensitivity": float(sensitivity),
        "Recall": float(sensitivity),
        "Specificity": float(specificity),
        "F1": float(f1),
        "Youden": float(youden),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def select_best_threshold(metrics_df):
    # 主阈值：最大 Youden index
    # 如果多个阈值 Youden 相同，选择更接近 0.5 的阈值，避免极端阈值
    max_youden = metrics_df["Youden"].max()
    candidate_df = metrics_df[metrics_df["Youden"] == max_youden].copy()
    candidate_df["distance_to_0_5"] = (candidate_df["Threshold"] - 0.5).abs()

    best_row = candidate_df.sort_values(
        ["distance_to_0_5", "Threshold"]
    ).iloc[0]

    return best_row


# ============================================================
# 3. SEARCH BEST THRESHOLD FOR FULL RF AND TOP10 RF
# ============================================================
all_metrics_rows = []
all_curve_rows = []
best_summaries = {}

for model_name, cfg in MODEL_CONFIGS.items():
    display_name = cfg["display_name"]

    print("\n" + "=" * 80)
    print(f"Processing: {display_name}")
    print("=" * 80)

    sub = oof[oof["Model"] == model_name].copy()

    if sub.empty:
        raise ValueError(f"No rows found for Model == {model_name}")

    sub = sub.sort_values("Sample_Index").reset_index(drop=True)

    y_true = sub["y_true"].astype(int).values
    y_prob = sub["y_prob"].astype(float).values

    print(f"\nLoaded OOF predictions: {sub.shape}")
    print("Label distribution in OOF:")
    print(pd.Series(y_true).value_counts().sort_index())
    print(pd.Series(y_true).value_counts(normalize=True).sort_index().round(4))

    auc_value = roc_auc_score(y_true, y_prob)
    auprc_value = average_precision_score(y_true, y_prob)

    print("\nThreshold-independent performance from OOF predictions:")
    print(f"AUC   = {auc_value:.4f}")
    print(f"AUPRC = {auprc_value:.4f}")

    rows = []

    for t in THRESHOLDS:
        m = compute_threshold_metrics(
            y_true=y_true,
            y_prob=y_prob,
            threshold=t,
            auc_value=auc_value,
            auprc_value=auprc_value,
        )
        m["Model"] = model_name
        m["Display"] = display_name
        rows.append(m)

    metrics_df_model = pd.DataFrame(rows)
    best_row = select_best_threshold(metrics_df_model)
    best_threshold = float(best_row["Threshold"])

    default_row = compute_threshold_metrics(
        y_true=y_true,
        y_prob=y_prob,
        threshold=0.5,
        auc_value=auc_value,
        auprc_value=auprc_value,
    )

    print("\nBest threshold selected by maximum Youden index:")
    print(f"Threshold   = {best_threshold:.2f}")
    print(f"Youden      = {best_row['Youden']:.4f}")
    print(f"Sensitivity = {best_row['Sensitivity']:.4f}")
    print(f"Specificity = {best_row['Specificity']:.4f}")
    print(f"Accuracy    = {best_row['Accuracy']:.4f}")
    print(f"Precision   = {best_row['Precision']:.4f}")
    print(f"F1          = {best_row['F1']:.4f}")

    print("\nDefault threshold = 0.50:")
    print(f"Youden      = {default_row['Youden']:.4f}")
    print(f"Sensitivity = {default_row['Sensitivity']:.4f}")
    print(f"Specificity = {default_row['Specificity']:.4f}")
    print(f"Accuracy    = {default_row['Accuracy']:.4f}")
    print(f"Precision   = {default_row['Precision']:.4f}")
    print(f"F1          = {default_row['F1']:.4f}")

    best_summary = {
        "Model": model_name,
        "Display": display_name,
        "Selection_Data": "training_out_of_fold_predictions",
        "Selection_Criterion": "maximum_Youden_index",
        "Best_Threshold": best_threshold,
        "AUC_OOF": round(float(auc_value), 6),
        "AUPRC_OOF": round(float(auprc_value), 6),
        "Best_Accuracy": round(float(best_row["Accuracy"]), 6),
        "Best_Precision": round(float(best_row["Precision"]), 6),
        "Best_Sensitivity": round(float(best_row["Sensitivity"]), 6),
        "Best_Recall": round(float(best_row["Recall"]), 6),
        "Best_Specificity": round(float(best_row["Specificity"]), 6),
        "Best_F1": round(float(best_row["F1"]), 6),
        "Best_Youden": round(float(best_row["Youden"]), 6),
        "Best_TN": int(best_row["TN"]),
        "Best_FP": int(best_row["FP"]),
        "Best_FN": int(best_row["FN"]),
        "Best_TP": int(best_row["TP"]),
        "Default_Threshold": 0.5,
        "Default_Accuracy": round(float(default_row["Accuracy"]), 6),
        "Default_Precision": round(float(default_row["Precision"]), 6),
        "Default_Sensitivity": round(float(default_row["Sensitivity"]), 6),
        "Default_Recall": round(float(default_row["Recall"]), 6),
        "Default_Specificity": round(float(default_row["Specificity"]), 6),
        "Default_F1": round(float(default_row["F1"]), 6),
        "Default_Youden": round(float(default_row["Youden"]), 6),
        "Note": (
            "Threshold was selected using training-set out-of-fold predictions only. "
            "No internal test or external validation data were used."
        ),
    }

    best_summaries[model_name] = best_summary

    all_metrics_rows.extend(metrics_df_model.to_dict("records"))

    curve_df_model = metrics_df_model[
        [
            "Model",
            "Display",
            "Threshold",
            "Youden",
            "Sensitivity",
            "Specificity",
            "Precision",
            "F1",
            "Accuracy",
        ]
    ].copy()

    all_curve_rows.extend(curve_df_model.to_dict("records"))


# ============================================================
# 4. SAVE RESULTS
# ============================================================
all_metrics_df = pd.DataFrame(all_metrics_rows)
all_curve_df = pd.DataFrame(all_curve_rows)
best_df = pd.DataFrame(list(best_summaries.values()))

all_metrics_df.to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")
all_curve_df.to_csv(CURVE_CSV, index=False, encoding="utf-8-sig")
best_df.to_csv(BEST_THRESHOLD_CSV, index=False, encoding="utf-8-sig")

with open(BEST_THRESHOLD_JSON, "w", encoding="utf-8") as f:
    json.dump(list(best_summaries.values()), f, indent=2, ensure_ascii=False)

# 兼容旧代码：单独保存 Top10 RandomForest 的阈值文件
top10_summary = best_summaries["Top10_RandomForest"]
pd.DataFrame([top10_summary]).to_csv(
    TOP10_COMPAT_CSV,
    index=False,
    encoding="utf-8-sig",
)

with open(TOP10_COMPAT_JSON, "w", encoding="utf-8") as f:
    json.dump(top10_summary, f, indent=2, ensure_ascii=False)

print("\nSaved:")
print(BEST_THRESHOLD_CSV)
print(BEST_THRESHOLD_JSON)
print(CURVE_CSV)
print(METRICS_CSV)
print("\nCompatibility files for Top10 final evaluation:")
print(TOP10_COMPAT_CSV)
print(TOP10_COMPAT_JSON)


# ============================================================
# 5. PLOT YOUDEN / SENSITIVITY / SPECIFICITY CURVES
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)

for ax, model_name in zip(axes, MODEL_CONFIGS.keys()):
    cfg = MODEL_CONFIGS[model_name]
    display_name = cfg["display_name"]

    df = all_curve_df[all_curve_df["Model"] == model_name].copy()
    best = best_summaries[model_name]
    best_threshold = float(best["Best_Threshold"])
    best_youden = float(best["Best_Youden"])

    ax.plot(
        df["Threshold"],
        df["Youden"],
        linewidth=2.8,
        label="Youden index",
    )

    ax.plot(
        df["Threshold"],
        df["Sensitivity"],
        linewidth=2.0,
        linestyle="--",
        label="Sensitivity",
    )

    ax.plot(
        df["Threshold"],
        df["Specificity"],
        linewidth=2.0,
        linestyle="--",
        label="Specificity",
    )

    ax.axhline(
        y=0,
        color="gray",
        linestyle=":",
        linewidth=1.2,
        alpha=0.8,
    )

    ax.axvline(
        best_threshold,
        linestyle=":",
        linewidth=2.0,
    )

    ax.scatter(
        [best_threshold],
        [best_youden],
        s=140,
        zorder=5,
    )

    ax.annotate(
        f"Best threshold = {best_threshold:.2f}\n"
        f"Youden = {best['Best_Youden']:.3f}\n"
        f"Sens = {best['Best_Sensitivity']:.3f}\n"
        f"Spec = {best['Best_Specificity']:.3f}",
        xy=(best_threshold, best_youden),
        xytext=(min(best_threshold + 0.07, 0.72), best_youden - 0.10),
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="black",
            alpha=0.88,
        ),
        arrowprops=dict(
            arrowstyle="->",
            color="black",
            linewidth=1.2,
        ),
    )

    ax.set_title(
        display_name,
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Threshold", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(fontsize=9, frameon=False, loc="best")

axes[0].set_ylabel("Metric value", fontsize=12, fontweight="bold")

y_min = min(
    all_curve_df["Youden"].min(),
    all_curve_df["Sensitivity"].min(),
    all_curve_df["Specificity"].min(),
) - 0.05

y_max = max(
    all_curve_df["Youden"].max(),
    all_curve_df["Sensitivity"].max(),
    all_curve_df["Specificity"].max(),
) + 0.12

for ax in axes:
    ax.set_xlim(0, 1)
    ax.set_ylim(max(-0.1, y_min), min(1.15, y_max))

fig.suptitle(
    "Threshold Selection for Full Random Forest and Top10 Random Forest\n"
    "Training Out-of-Fold Predictions, Youden Index",
    fontsize=15,
    fontweight="bold",
    y=1.03,
)

plt.tight_layout()
plt.savefig(FIG_PATH_PNG, dpi=300, bbox_inches="tight")
plt.savefig(FIG_PATH_PDF, bbox_inches="tight")
plt.close()

print("\nSaved figure:")
print(FIG_PATH_PNG)
print(FIG_PATH_PDF)


# ============================================================
# 6. DONE
# ============================================================
print("\n" + "=" * 80)
print("DONE")
print("=" * 80)

for model_name, summary in best_summaries.items():
    print(
        f"{summary['Display']}: "
        f"Best threshold = {summary['Best_Threshold']:.2f}, "
        f"Youden = {summary['Best_Youden']:.4f}, "
        f"Sensitivity = {summary['Best_Sensitivity']:.4f}, "
        f"Specificity = {summary['Best_Specificity']:.4f}"
    )

print("\nImportant:")
print("1. This script does NOT fit or refit any model.")
print("2. This script uses OOF predictions from train only.")
print("3. This script does NOT read internal_test_processed.csv.")
print("4. This script does NOT read external_processed.csv.")